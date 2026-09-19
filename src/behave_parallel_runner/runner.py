# -*- coding: UTF-8 -*-
"""
Parallel test runner for behave: runs feature files concurrently in worker
processes (used for: ``--jobs N`` with N > 1).

Select this runner in the behave config-file (or with ``-r/--runner``)::

    # -- FILE: behave.ini
    [behave]
    runner = behave_parallel_runner:ParallelRunner

With ``--jobs=1`` (the default) or in dry-run mode, this runner behaves
like the normal (sequential) behave runner.

DESIGN:

* One parent process (this runner) and up to ``config.jobs`` worker processes
  (:mod:`multiprocessing` with "spawn" start-method, one pipe per worker).
* Work unit: one feature file per task. The parent sends the feature file
  locations (``filename`` or ``filename:line``, so that scenario selection
  is preserved). A worker parses them, runs the feature with a normal
  (sequential) runner runtime and sends back a picklable result
  (status counts, captured output chunk, undefined steps, ...).
* A worker gets its next task when it is done with the current one. The
  parent always knows which feature a worker runs: if a worker process
  dies, this feature is reported as errored, the worker is replaced and
  the test-run goes on.
* A feature with the tag ``@serial`` (on the feature or on one of the
  scenarios that run) is never run together with another feature: a worker
  hands it back and the parent runs these features one by one, after the
  other features.
* The parent prints each feature's output chunk when its task completes
  (whole chunks, completion order), merges the counts into the summary
  reporter and composes the final exit status like the sequential runner.
  A formatter with an outfile writes one chunk per feature, too; the parent
  appends them to the outfile (JSON chunks are merged into one JSON array).

HOOKS (parallel mode never calls "before_all"/"after_all"):

* "before_parallel"/"after_parallel": run once in the PARENT process.
* "before_worker"/"after_worker": run once per worker process.
* All other hooks (feature/rule/scenario/step/tag) run in workers, unchanged.

An environment file that defines "before_all" (or "after_all") without a
matching parallel-mode hook is rejected: what happens with the ``*_all``
hook under parallel execution must be an explicit choice (call it from one
of the parallel hooks, split it up, or replace it).

.. note:: Programmatic use requires an importable main module
    (``if __name__ == "__main__":`` guard), because the "spawn"
    start-method re-imports ``__main__`` in each worker process.
"""

import importlib
import io
import json
import multiprocessing
import multiprocessing.connection
import os
import pickle
import sys
import time
import traceback
from collections import OrderedDict, deque, namedtuple

from behave.configuration import Configuration, DEFAULT_RUNNER_CLASS_NAME
from behave.exception import ConfigError
from behave.formatter._registry import make_formatters, select_formatter_class
from behave.formatter.base import StreamOpener
from behave.model_type import FileLocation, Status
from behave.parser import ParserError
from behave.reporter.summary import AbstractSummaryReporter, SummaryReporterV1
from behave.runner import Context, Runner
from behave.runner_util import FileLocationParser, parse_features, reset_runtime


# -----------------------------------------------------------------------------
# CONSTANTS:
# -----------------------------------------------------------------------------
#: Each "*_all" hook requires one of these hooks in parallel mode.
PARALLEL_HOOK_REQUIREMENTS = OrderedDict([
    ("before_all", ("before_parallel", "before_worker")),
    ("after_all", ("after_parallel", "after_worker")),
])

#: Built-in formatters that aggregate over the complete test-run
#: (or need one shared output stream): They cannot run once per worker.
#: HINT: Base classes are used, derived formatter classes are covered, too.
COMPLETE_TESTRUN_FORMATTER_CLASS_NAMES = (
    "behave.formatter.bad_steps:BadStepsFormatter",
    "behave.formatter.rerun:RerunFormatter",
    "behave.formatter.steps:AbstractStepsFormatter",
    "behave.formatter.tags:AbstractTagsFormatter",
)

#: Built-in formatters whose per-feature output chunks are JSON arrays
#: (the parent merges them into one JSON array).
JSON_FORMATTER_CLASS_NAMES = (
    "behave.formatter.json:JSONFormatter",
)

#: Modules of behave with the formatter base classes (no real formatters).
ABSTRACT_FORMATTER_MODULE_NAMES = (
    "behave.formatter.api",
    "behave.formatter.base",
)

#: A feature with this tag is never run together with another feature.
SERIAL_TAG = "serial"

#: Interval in seconds to check if the worker processes are still alive
#: (while the parent waits for their messages).
WORKER_POLL_INTERVAL = 1.0

#: Time in seconds that a worker process gets to end after it was
#: terminated (before it is killed).
WORKER_TERMINATE_TIMEOUT = 5.0

#: Configuration attributes that the parent may have changed after the
#: configuration was built (they select WHICH tests run or WHAT they see).
PROPAGATED_CONFIG_PARAMS = ("stage", "lang", "tags", "userdata")


class UndefinedStepInfo:
    """Undefined-step info: duck-types a Step for undefined-step snippets.

    Hashable (used for cross-worker deduplication), but "name" stays
    mutable because snippet generation may rewrite it for quote-escaping
    (see: :func:`behave.runner_util.make_undefined_step_snippet`).
    """
    __slots__ = ("step_type", "name")

    def __init__(self, step_type, name):
        self.step_type = step_type
        self.name = name

    def _key(self):
        return (self.step_type, self.name)

    def __eq__(self, other):
        other_key = getattr(other, "_key", None)
        return other_key is not None and self._key() == other_key()

    def __hash__(self):
        return hash(self._key())

    def __lt__(self, other):
        return self._key() < other._key()

    def __repr__(self):
        return "UndefinedStepInfo(%r, %r)" % (self.step_type, self.name)


class ScenarioInfo:
    """Duck-types a Scenario for the summary reporter's problem list."""
    __slots__ = ("location", "name")

    def __init__(self, location, name):
        self.location = location
        self.name = name


# -----------------------------------------------------------------------------
# PURE HELPER FUNCTIONS:
# -----------------------------------------------------------------------------
def group_locations_by_filename(locations):
    """Group feature file locations by their feature filename.

    Scenario selection by line number, like "alice.feature:12", must be
    preserved: all locations of one feature file become one work item.

    :param locations: Feature file locations (FileLocation objects or strings).
    :return: Ordered dict with filename as key and location texts as value.
    """
    grouped = OrderedDict()
    for location in locations:
        filename = getattr(location, "filename", None) or str(location)
        filename = os.path.normpath(filename)
        grouped.setdefault(filename, []).append(str(location))
    return grouped


def is_serial_feature(feature, config=None):
    """Check if a feature must not run together with other features:
    one of its scenarios that run has the tag "@serial" (own tag or
    inherited from its feature, rule or scenario outline).
    """
    return any(SERIAL_TAG in scenario.effective_tags
               and scenario.should_run(config)
               for scenario in feature.walk_scenarios())


def mark_feature_as_errored(feature, error_text, config=None):
    """Mark a feature as errored whose results are lost (like: its worker
    process died). It is unknown how far it came: All its scenarios that
    should run are errored, so that reporters (like: JUnit) show the problem.
    """
    for scenario in feature.walk_scenarios():
        if scenario.should_run(config):
            scenario.set_status(Status.error)
            scenario.error_message = error_text
    feature.set_status(Status.error)


def parse_feature_locations(location_texts, language=None):
    """Parse feature file location texts, like: "alice.feature:12"."""
    locations = []
    for location_text in location_texts:
        location = FileLocationParser.parse(location_text)
        locations.append(FileLocation(os.path.normpath(location.filename),
                                      location.line))
    return parse_features(locations, language=language)


def load_formatter_classes(scoped_class_names):
    """Load built-in formatter classes by their scoped class names.

    HINT: A formatter module that does not exist in the used behave version
    is ignored.
    """
    formatter_classes = []
    for scoped_class_name in scoped_class_names:
        module_name, class_name = scoped_class_name.split(":")
        try:
            module = importlib.import_module(module_name)
            formatter_classes.append(getattr(module, class_name))
        except (ImportError, AttributeError):
            continue
    return tuple(formatter_classes)


def select_complete_testrun_formatter_classes():
    """Select the built-in formatter classes that need the complete test-run."""
    return load_formatter_classes(COMPLETE_TESTRUN_FORMATTER_CLASS_NAMES)


def needs_complete_testrun(formatter_class):
    """Check if a formatter aggregates over the complete test-run
    (or needs one shared output stream), so that it cannot run per worker.

    A formatter class states this with the class attribute
    ``needs_complete_testrun`` (True or False). Without such a statement,
    the built-in formatters of this kind are detected. The most specific
    class decides: a formatter class that is derived from a built-in one
    inherits its needs, unless it states something else itself.
    """
    flag_name = "needs_complete_testrun"
    if not isinstance(formatter_class, type):
        # -- HINT: A formatter needs not be a class (or inherit from Formatter).
        return bool(getattr(formatter_class, flag_name, False))

    builtin_classes = select_complete_testrun_formatter_classes()
    for klass in formatter_class.__mro__:
        if flag_name in vars(klass):
            return bool(vars(klass)[flag_name])
        if klass in builtin_classes:
            return True
    return False


def select_output_merger_name(formatter_class):
    """Select how the parent merges the per-feature output chunks of a
    formatter: "json" for the JSON formatters (and formatter classes that
    are derived from them), otherwise "text" (chunks are appended).
    """
    if isinstance(formatter_class, type):
        json_classes = load_formatter_classes(JSON_FORMATTER_CLASS_NAMES)
        if any(klass in json_classes for klass in formatter_class.__mro__):
            return "json"
    return "text"


def select_outfile_mode(formatter_class):
    """Select how a formatter with an outfile is used by many workers:

    * "merge": The formatter writes to its output stream. Each worker
      collects this output per feature and the parent appends these chunks
      to the outfile.
    * "direct": The formatter does not use its output stream, it writes own
      files by using the name of the outfile (like: a directory with one
      file per test). Each worker gets the real outfile name.
    * None: Unknown.

    A formatter class states this with the class attribute
    ``parallel_outfile``. Without such a statement, the built-in formatters
    of behave use "merge". The most specific class decides.
    """
    attribute_name = "parallel_outfile"
    if not isinstance(formatter_class, type):
        return getattr(formatter_class, attribute_name, None)

    for klass in formatter_class.__mro__:
        if attribute_name in vars(klass):
            return vars(klass)[attribute_name]
        module_name = getattr(klass, "__module__", "") or ""
        if (module_name.startswith("behave.formatter.")
                and module_name not in ABSTRACT_FORMATTER_MODULE_NAMES):
            # -- BUILT-IN FORMATTER: Writes to its output stream.
            return "merge"
    return None


#: Describes one formatter of a worker.
#:
#: * name: Formatter name that the worker uses.
#: * output: None, if the formatter writes into the worker's console output
#:   (together with anything else that the feature prints). Otherwise, the
#:   index of its output: config.outputs[output] (format[i] is paired with
#:   outputs[i]).
#: * merger: How its output gets there. "text" and "json": The formatter
#:   writes into an own buffer and the parent merges these chunks
#:   (see: OUTPUT_MERGER_CLASSES). "direct": The formatter of each worker
#:   gets the real outfile name (see: select_outfile_mode()).
WorkerFormat = namedtuple("WorkerFormat", ["name", "output", "merger"])


def resolve_worker_formats(formats, outfile_bound=None):
    """Compute the formatters that workers should use.

    See :func:`needs_complete_testrun()` for the formatters that are rejected.

    :param formats: Formatter names requested for this test run.
    :param outfile_bound: Flags that tell if format[i] writes to an outfile.
    :return: Tuple (worker_formats, notes) -- a list of :class:`WorkerFormat`
        and the user-facing messages.
    :raises ConfigError: If a formatter cannot be used with "--jobs > 1"
        (or if its formatter class cannot be resolved at all).
    """
    outfile_bound = list(outfile_bound or [])
    worker_formats = []
    notes = []
    for index, name in enumerate(formats):
        is_outfile_bound = (index < len(outfile_bound) and outfile_bound[index])
        try:
            # -- HINT: Resolves aliases and scoped class names (lazy-loaded).
            formatter_class = select_formatter_class(name)
        except (LookupError, ImportError, TypeError, ValueError) as e:
            # -- HINT: Configuration normally rejects unknown formatters
            # earlier -- do not mask such an error here.
            raise ConfigError('PARALLEL: unknown formatter "%s" (%s: %s)'
                              % (name, e.__class__.__name__, e))
        if needs_complete_testrun(formatter_class):
            raise ConfigError(
                'PARALLEL: formatter "%s" is not supported with --jobs > 1 '
                '(it needs the complete test-run). '
                'Use --jobs=1 or drop this formatter.' % name)
        if name == "pretty":
            notes.append(
                'PARALLEL: NOTE -- using "plain" formatter instead of '
                '"pretty" (not usable with --jobs > 1).')
            name = "plain"

        merger = select_output_merger_name(formatter_class)
        if is_outfile_bound:
            outfile_mode = select_outfile_mode(formatter_class)
            if outfile_mode not in ("merge", "direct"):
                # -- HINT: A formatter may use its outfile as directory name.
                raise ConfigError(
                    'PARALLEL: formatter "%s" with --outfile is not supported '
                    'with --jobs > 1: It is unknown how it uses its outfile '
                    '(found: parallel_outfile=%r). Its formatter class can '
                    'state this with the class attribute parallel_outfile = '
                    '"merge" (it writes to its output stream) or "direct" '
                    '(it writes own files, like: into a directory).'
                    % (name, outfile_mode))
            if outfile_mode == "direct":
                merger = "direct"
        if is_outfile_bound or merger != "text":
            # -- OWN OUTPUT: The parent merges the chunks of all workers.
            worker_format = WorkerFormat(name, index, merger)
        else:
            worker_format = WorkerFormat(name, None, merger)
        if worker_format not in worker_formats:
            worker_formats.append(worker_format)
    if not worker_formats:
        worker_formats.append(WorkerFormat("plain", None, "text"))
    return worker_formats, notes


class TextOutputMerger:
    """Merges text chunks: appends them to the output stream."""

    def __init__(self, stream):
        self.stream = stream

    def add(self, text):
        self.stream.write(text)
        self.stream.flush()

    def close(self):
        self.stream.flush()


class JsonOutputMerger(TextOutputMerger):
    """Merges JSON chunks: each chunk is a JSON array (of features) and the
    output stream gets one JSON array with the elements of all chunks.

    HINT: The array elements are copied as text, so that the layout of the
    formatter is kept (like: "json.pretty").
    """
    HEADER = "[\n"
    SEPARATOR = ",\n"
    FOOTER = "\n]\n"

    def __init__(self, stream):
        super(JsonOutputMerger, self).__init__(stream)
        self.count = 0

    def add(self, text):
        text = text.strip()
        try:
            data = json.loads(text)
        except ValueError:
            data = None
        if not isinstance(data, list):
            # -- BROKEN CHUNK: Like a feature whose run ended with an error.
            sys.stderr.write("PARALLEL: WARNING -- JSON output of one "
                             "feature is not usable (ignored).\n")
            return
        if not data:
            return

        self.stream.write(self.SEPARATOR if self.count else self.HEADER)
        self.stream.write(text[1:-1].strip("\n"))
        self.stream.flush()
        self.count += 1

    def close(self):
        if not self.count:
            self.stream.write(self.HEADER)
        self.stream.write(self.FOOTER)
        self.stream.flush()


OUTPUT_MERGER_CLASSES = {
    "text": TextOutputMerger,
    "json": JsonOutputMerger,
}


def select_outfile_bound_formats(config):
    """Determine which formats write into an "--outfile".

    HINT: make_formatters() pairs format[i] with config.outputs[i].
    A stream-opener without filename writes to the console (not an outfile).
    """
    outputs = config.outputs or []
    return [bool(getattr(opener, "name", None)) for opener in outputs]


def merge_status_counts(target, source):
    """Merge one worker's status-count dict into an accumulator dict."""
    for name, count in source.items():
        target[name] = target.get(name, 0) + count


def select_summary_reporter(reporters):
    """Select the summary reporter that the parent merges results into."""
    for reporter in reporters:
        if isinstance(reporter, AbstractSummaryReporter):
            if not isinstance(reporter, SummaryReporterV1):
                raise ConfigError(
                    "PARALLEL: %s is not supported with --jobs > 1 "
                    "(only SummaryReporterV1 counts can be merged)."
                    % type(reporter).__name__)
            return reporter
    return None


def select_picklable_params(params, dropped=None):
    """Select the parameters that can be sent to a worker process.

    :param params: Parameters to check (as dict).
    :param dropped: Optional list that collects the non-picklable names.
    """
    selected = {}
    for name, value in params.items():
        try:
            pickle.dumps(value)
        except Exception:  # pylint: disable=broad-except
            # -- SKIP: Non-picklable parameter.
            if dropped is not None:
                dropped.append(name)
            continue
        selected[name] = value
    return selected


def make_result(filename, **kwargs):
    """Create a feature task result (all values are picklable)."""
    result = {
        "filename": filename,
        "location": filename,
        "failed": True,
        "status": None,
        "feature_summary": {},
        "rule_summary": {},
        "scenario_summary": {},
        "step_summary": {},
        "duration": 0.0,
        "problematic_scenarios": [],
        "undefined_steps": [],
        # -- HOOK-FAILURES: Of this task only.
        "hook_failures": 0,
        # -- HOOK-FAILURES: Of the worker setup (same value in each result
        # of one worker -- the parent counts them once per worker).
        "worker_init_hook_failures": 0,
        "worker_id": None,
        "worker_setup_failed": False,
        # -- HINT: A parse-error aborts the test-run (like: sequential mode).
        "fatal_error": False,
        # -- HINT: This worker's test-run was aborted, like: context.abort()
        "aborted": False,
        # -- HINT: Feature file without any feature (nothing to run/report).
        "no_feature": False,
        # -- HINT: Task failed without any result from the worker.
        "task_errored": False,
        # -- HINT: Feature was not run, it must run alone (tag: @serial).
        "serial_deferred": False,
        "output": "",
        # -- HINT: Output chunks of formatters with an own output,
        # as dict: WorkerFormat.output => text
        "outputs": {},
        "error_text": None,
    }
    result.update(kwargs)
    return result


# -----------------------------------------------------------------------------
# WORKER SIDE (runs in worker processes; must be module-level for pickling):
# -----------------------------------------------------------------------------
class WorkerOutput(io.StringIO):
    """Output stream of one worker process (for its whole lifetime).

    A worker replaces its ``sys.stdout``/``sys.stderr`` with this stream,
    so that anything bound to them (like logging handlers) keeps writing
    into it. The parent collects the text per feature (and prints it).
    """

    def drain(self):
        """Return the collected text and start over (empty again)."""
        text = self.getvalue()
        self.seek(0)
        self.truncate(0)
        return text


class WorkerRunner(Runner):
    """Runner runtime used inside one worker process.

    Lives for the whole worker process and runs its features one by one
    on one Context (so that "before_worker" attributes stay visible),
    without ever running "before_all"/"after_all".
    """

    def load_hooks(self, filename=None):
        super(WorkerRunner, self).load_hooks(filename)
        if "before_worker" not in self.hooks:
            # -- DEFAULT-HOOK (like "before_all"): Setup logging subsystem.
            self.hooks["before_worker"] = self.before_all_default_hook


# -- WORKER-PROCESS GLOBALS:
_worker_runner = None
_worker_output = None
_worker_id = None
_worker_setup_failed = False
_worker_init_hook_failures = 0
_worker_shutdown_failures = None
_worker_formats = None


def _emit_worker_output(text):
    """Write a worker's output chunk to the real process output stream."""
    stream = sys.__stdout__
    if text and stream is not None:
        stream.write(text)
        stream.flush()


def _apply_worker_config_overrides(config, worker_setup):
    """Adjust a worker's rebuilt Configuration for parallel execution."""
    # -- STEP: Re-apply configuration params that the parent may have changed.
    config_params = worker_setup["config_params"]
    if "stage" in config_params:
        config.setup_stage(config_params["stage"])
    if "lang" in config_params:
        config.lang = config_params["lang"]
    if "userdata" in config_params:
        config.userdata.update(config_params["userdata"])
    if "tags" in config_params:
        config.setup_tag_expression(config_params["tags"])

    # -- STEP: Enforce parallel-worker mode.
    config.jobs = 1
    config.runner = DEFAULT_RUNNER_CLASS_NAME
    config.format = [worker_format.name
                     for worker_format in worker_setup["worker_format"]]
    config.default_format = "plain"
    # -- HINT: The parent prints merged undefined-step snippets once.
    config.show_snippets = False
    # -- HINT: No per-worker summary; the parent prints the merged summary.
    config.summary = False
    config.reporters = [reporter for reporter in config.reporters
                        if not isinstance(reporter, AbstractSummaryReporter)]


def _worker_init(worker_setup, worker_id, shutdown_failures):
    """Initialize one worker process."""
    # pylint: disable=global-statement
    global _worker_runner, _worker_output, _worker_id
    global _worker_setup_failed, _worker_init_hook_failures
    global _worker_shutdown_failures, _worker_formats

    # -- SETUP: Use one output stream for the whole worker lifetime,
    # so that logging handlers, etc. keep writing into a collected stream.
    _worker_output = WorkerOutput()
    _worker_shutdown_failures = shutdown_failures
    _worker_id = worker_id
    sys.stdout = _worker_output
    sys.stderr = _worker_output
    try:
        reset_runtime()
        config = Configuration(worker_setup["command_args"],
                               load_config=worker_setup["load_config"],
                               **worker_setup["config_kwargs"])
        _apply_worker_config_overrides(config, worker_setup)

        runner = WorkerRunner(config)
        runner.path_manager.__enter__()  # -- UNDONE: at process exit.
        runner.setup_paths()
        runner.context = Context(runner)
        runner.load_hooks()
        runner.load_step_definitions()
        # -- BIND: Step registry (normally done by: ModelRunner.run_model).
        from behave.runner import the_step_registry
        runner.step_registry = the_step_registry
        runner.context._set_root_attribute("worker_id", _worker_id)
        runner.context._set_root_attribute("jobs", worker_setup["jobs"])
        _worker_runner = runner

        hook_passed = runner.run_hook("before_worker")
        if not hook_passed:
            # -- LIKE: "before_all" hook-error in sequential mode.
            # HINT: No feature is run by this worker (test-run is aborted).
            _worker_setup_failed = True
        _worker_init_hook_failures = runner.hook_failures
        _worker_formats = list(worker_setup["worker_format"])
    except Exception:  # pylint: disable=broad-except
        _worker_setup_failed = True
        _worker_init_hook_failures = max(_worker_init_hook_failures, 1)
        traceback.print_exc()
    finally:
        _emit_worker_output(_worker_output.drain())


def _worker_shutdown():
    """Finalize one worker process (skipped on hard terminate)."""
    runner = _worker_runner
    if runner is None:
        return

    failures = 0
    try:
        if not runner.run_hook("after_worker"):
            failures += 1
        try:
            runner.context._do_remaining_cleanups()
        except Exception:  # pylint: disable=broad-except
            traceback.print_exc()
            failures += 1
    finally:
        if failures and _worker_shutdown_failures is not None:
            # -- REPORT: Shutdown failures to the parent process.
            # HINT: Task results are already sent when this hook runs.
            with _worker_shutdown_failures.get_lock():
                _worker_shutdown_failures.value += failures
        if _worker_output is not None:
            _emit_worker_output(_worker_output.drain())


def _report_errored_feature(runner, feature):
    """Report an errored feature to the reporters (best-effort)."""
    for reporter in runner.config.reporters:
        try:
            reporter.feature(feature)
        except Exception:  # pylint: disable=broad-except
            traceback.print_exc()


def _worker_main(worker_id, worker_setup, connection, shutdown_failures):
    """Main function of one worker process: runs the tasks of the parent.

    PROTOCOL:

    * worker => parent: ("ready", info), when the worker setup is done
      (info: "setup_failed" and "init_hook_failures" of this worker).
    * parent => worker: Task as tuple (location_texts, serial_phase)
      or None (worker should shut down).
    * worker => parent: ("result", result) for each task.
    """
    try:
        _worker_init(worker_setup, worker_id, shutdown_failures)
        connection.send(("ready", {
            "setup_failed": _worker_setup_failed,
            "init_hook_failures": _worker_init_hook_failures,
        }))
        while True:
            task = connection.recv()
            if task is None:
                break
            location_texts, serial_phase = task
            result = _run_feature_task(location_texts, serial_phase)
            _send_result(connection, result)
    except (EOFError, OSError):
        # -- PARENT PROCESS IS GONE: Shut down.
        pass
    except KeyboardInterrupt:
        # -- HARD-STOP: Without shutdown-hooks (the parent terminates workers).
        return
    except Exception:  # pylint: disable=broad-except
        # -- UNEXPECTED: The parent reports the running feature as errored.
        # HINT: sys.stderr is the collected stream, that nobody reads anymore.
        if _worker_output is not None:
            _emit_worker_output(_worker_output.drain())
        if sys.__stderr__ is not None:
            traceback.print_exc(file=sys.__stderr__)
        sys.exit(1)
    _worker_shutdown()


def _send_result(connection, result):
    """Send a task result to the parent process."""
    try:
        connection.send(("result", result))
    except (EOFError, OSError):
        raise
    except Exception as e:  # pylint: disable=broad-except
        # -- TASK ERROR: For example, a result that cannot be pickled.
        # HINT: Nothing was sent yet (pickling is done first).
        filename = result["filename"]
        error_text = "PARALLEL-WORKER FAILURE in %s: %s" % (filename, e)
        connection.send(("result", make_result(
            filename, worker_id=result["worker_id"], task_errored=True,
            error_text=error_text)))


def _make_feature_stream_openers(worker_formats, outputs=None):
    """Create the stream-openers for the formatters of one feature run.

    :param outputs: Stream-openers of the configuration (config.outputs).
    :return: Tuple (stream_openers, buffers) -- buffers of the formatters
        with an own output, as dict: WorkerFormat.output => buffer
    """
    stream_openers = []
    buffers = {}
    for worker_format in worker_formats:
        if worker_format.merger == "direct":
            # -- HINT: This formatter writes own files (uses: outfile name).
            outfile = outputs[worker_format.output].name
            stream_openers.append(StreamOpener(filename=outfile))
            continue

        stream = _worker_output
        if worker_format.output is not None:
            stream = buffers[worker_format.output] = io.StringIO()
        stream_openers.append(StreamOpener(stream=stream))
    return stream_openers, buffers


def _run_feature_task(location_texts, serial_phase=False):
    """Run one feature file in this worker process; returns a result dict.

    :param serial_phase: If true, no other feature runs at the same time
        (otherwise, a "@serial" feature is handed back to the parent).
    """
    # pylint: disable=global-statement
    global _worker_init_hook_failures
    runner = _worker_runner
    filename = FileLocationParser.parse(location_texts[0]).filename
    result = make_result(filename, worker_id=_worker_id)

    # -- STEP: Report worker-setup failures in EACH result of this worker.
    result["worker_init_hook_failures"] = _worker_init_hook_failures
    if _worker_setup_failed or runner is None:
        result["worker_setup_failed"] = True
        result["worker_init_hook_failures"] = max(_worker_init_hook_failures, 1)
        result["error_text"] = ("PARALLEL-WORKER SETUP FAILED: %s "
                                "(feature not run)" % filename)
        result["output"] = (_worker_output.drain()
                            if _worker_output is not None else "")
        return result

    if runner.aborted:
        # -- NOT RUN: The test-run of this worker is aborted.
        # HINT: status stays None -- the parent reports it as untested.
        # HINT: A worker's Context stays aborted (like: sequential mode).
        result.update(failed=False, aborted=runner.aborted,
                      output=_worker_output.drain())
        return result

    hook_failures0 = runner.hook_failures
    undefined_steps0 = len(runner.undefined_steps)
    feature = None
    feature_reported = False
    buffers = {}
    try:
        features = parse_feature_locations(location_texts,
                                           language=runner.config.lang)
        if not features:
            # -- LIKE SEQUENTIAL MODE: parse_features() skips a feature file
            # without any feature (corner case; nothing to run or report).
            result.update(failed=False, no_feature=True,
                          output=_worker_output.drain())
            return result
        if not serial_phase and is_serial_feature(features[0], runner.config):
            # -- NOT RUN: The parent runs it later (without other features).
            result.update(failed=False, serial_deferred=True,
                          output=_worker_output.drain())
            return result
        feature = features[0]

        runner.feature = feature
        worker_formats = _worker_formats
        if worker_formats is None:
            worker_formats = [WorkerFormat(name, None, "text")
                              for name in runner.config.format]
        stream_openers, buffers = _make_feature_stream_openers(
            worker_formats, runner.config.outputs)
        runner.formatters = make_formatters(runner.config, stream_openers)
        try:
            for formatter in runner.formatters:
                formatter.uri(feature.filename)
            failed = feature.run(runner)
            for formatter in runner.formatters:
                formatter.close()
        finally:
            runner.formatters = []
        feature_reported = True
        for reporter in runner.config.reporters:
            reporter.feature(feature)
        result["failed"] = bool(failed)
    except ParserError as e:
        # -- LIKE SEQUENTIAL MODE: A parse-error aborts the test-run.
        # HINT: status stays None -- the parent reports it as untested.
        result["error_text"] = "ParserError: %s" % e
        result["fatal_error"] = True
    except Exception as e:  # pylint: disable=broad-except
        # -- HINT: Without a feature, status stays None (reported as untested).
        result["error_text"] = ("PARALLEL-WORKER ERROR in %s: %s\n%s"
                                % (filename, e, traceback.format_exc()))
        if feature is not None:
            # -- ERRORED FEATURE: Must not be reported as untested.
            feature.set_status(Status.error)
            if not feature_reported:
                _report_errored_feature(runner, feature)

    if feature is not None:
        # -- TALLY: Status counts for this feature (mergeable dicts).
        tally = SummaryReporterV1(runner.config)
        tally.testrun_started()
        tally.process_feature(feature)
        problematic = \
            [("failed", str(scenario.location), scenario.name)
             for scenario in tally.failed_scenarios] + \
            [("errored", str(scenario.location), scenario.name)
             for scenario in tally.errored_scenarios]
        new_undefined = runner.undefined_steps[undefined_steps0:]

        result.update(
            status=feature.status.name,
            location=str(feature.location),
            feature_summary=tally.feature_summary,
            rule_summary=tally.rule_summary,
            scenario_summary=tally.scenario_summary,
            step_summary=tally.step_summary,
            duration=feature.duration,
            problematic_scenarios=problematic,
            undefined_steps=[(step.step_type, step.name)
                             for step in new_undefined])

    result["aborted"] = runner.aborted
    result["hook_failures"] = runner.hook_failures - hook_failures0
    result["output"] = _worker_output.drain()
    result["outputs"] = {output: stream.getvalue()
                         for output, stream in buffers.items()}
    return result


# -----------------------------------------------------------------------------
# PARENT SIDE:
# -----------------------------------------------------------------------------
class WorkerProcess:
    """Parent-side handle of one worker process (and its connection)."""

    def __init__(self, worker_id, mp_context, worker_setup,
                 shutdown_failures):
        self.worker_id = worker_id
        self.ready = False          # -- Worker setup is done.
        self.stopping = False       # -- Worker was told to shut down.
        self.task = None            # -- Filename of the feature that it runs.
        self.process = None
        self.connection = None
        self.start(mp_context, worker_setup, shutdown_failures)

    # -- LOW-LEVEL PART: Process and connection.
    def start(self, mp_context, worker_setup, shutdown_failures):
        self.connection, child_connection = mp_context.Pipe()
        self.process = mp_context.Process(
            target=_worker_main, name="behave-worker-%d" % self.worker_id,
            args=(self.worker_id, worker_setup, child_connection,
                  shutdown_failures))
        self.process.start()
        # -- HINT: Otherwise, the end of the worker process is not seen.
        child_connection.close()

    def send(self, message):
        """Send a message to the worker process.

        :return: True, if the message was sent (false: worker died).
        """
        try:
            self.connection.send(message)
            return True
        except (OSError, ValueError):
            # -- WORKER DIED: Seen by receive(), too.
            return False

    @property
    def wait_objects(self):
        """Objects to wait for: a message or the end of the process."""
        return (self.connection, self.process.sentinel)

    def has_ended(self):
        """Check if the worker process has ended.

        HINT: The wait-objects are not enough to see this. A child process
        that a step has forked inherits the worker's end of the connection
        and of the sentinel, and may keep them open for a long time.
        """
        return not self.process.is_alive()

    def receive(self):
        """Receive the next message of this worker process.

        :return: Message as tuple (kind, payload); None if the process ended.
        """
        try:
            if not self.process.is_alive() and not self.connection.poll():
                # -- PROCESS HAS ENDED: And all its messages were received.
                return None
            return self.connection.recv()
        except (EOFError, OSError):
            return None
        except Exception as e:  # pylint: disable=broad-except
            # -- TASK ERROR: For example, a result that cannot be unpickled.
            return ("error", "%s: %s" % (e.__class__.__name__, e))

    def close(self, timeout=None):
        """Wait until the worker process has ended; returns its exit code.

        :param timeout: Kill the process if it has not ended after this time.
        """
        self.process.join(timeout)
        if self.process.is_alive():
            # -- CASE: Process ignores/handles SIGTERM (or it hangs).
            self.process.kill()
            self.process.join()
        self.connection.close()
        return self.process.exitcode

    def terminate(self):
        self.process.terminate()

    # -- HIGH-LEVEL PART:
    @property
    def idle(self):
        return self.ready and self.task is None and not self.stopping

    def run(self, filename, locations, serial_phase):
        """Let the worker run a feature.

        :return: True, if the worker got this task (false: worker died).
        """
        if not self.send((locations, serial_phase)):
            return False
        self.task = filename
        return True

    def stop(self):
        """Tell the worker to shut down (it runs its shutdown-hooks)."""
        self.stopping = True
        self.send(None)


def wait_for_workers(workers, poll_interval=WORKER_POLL_INTERVAL):
    """Wait until any worker has a message (or its process has ended)."""
    owners = {}
    for worker in workers:
        for wait_object in worker.wait_objects:
            owners[wait_object] = worker

    while True:
        ready = multiprocessing.connection.wait(list(owners),
                                                timeout=poll_interval)
        ready_workers = []
        for wait_object in ready:
            if owners[wait_object] not in ready_workers:
                ready_workers.append(owners[wait_object])
        # -- HINT: The end of a process is not always seen by its
        # wait-objects, see: WorkerProcess.has_ended()
        for worker in workers:
            if worker not in ready_workers and worker.has_ended():
                ready_workers.append(worker)
        if ready_workers:
            return ready_workers


class TaskSchedule:
    """Hands out the work items (feature files) to the workers.

    The "@serial" features are only known after a worker has parsed them:
    they are handed back and run later, one by one, while nothing else runs.
    """

    def __init__(self, work_items):
        self.work_items = work_items
        self.pending = deque(work_items)
        self.serial_pending = deque()
        self.running = 0
        self.cancelled = False

    def has_tasks(self):
        return not self.cancelled and bool(self.pending or self.serial_pending)

    def next_task(self):
        """Select the next task that can run now.

        :return: Task as tuple (filename, locations, serial_phase)
            or None -- HINT: Another worker runs the remaining tasks then.
        """
        serial_phase = False
        if self.cancelled:
            return None
        if self.pending:
            filename = self.pending.popleft()
        elif self.serial_pending and not self.running:
            filename = self.serial_pending.popleft()
            serial_phase = True
        else:
            return None
        self.running += 1
        return (filename, self.work_items[filename], serial_phase)

    def task_done(self):
        self.running -= 1

    def put_back(self, task):
        """Take back a task that its worker did not get (it runs next)."""
        filename, _locations, serial_phase = task
        self.running -= 1
        if serial_phase:
            self.serial_pending.appendleft(filename)
        else:
            self.pending.appendleft(filename)

    def defer_as_serial(self, filename):
        self.serial_pending.append(filename)

    def cancel(self):
        """Fail-early, abort: No further tasks (running features finish)."""
        self.cancelled = True


class ParallelRunner(Runner):
    """Test runner that runs feature files in parallel worker processes.

    Select it with the ``runner`` setting in the behave config-file or with
    ``--runner=behave_parallel_runner:ParallelRunner``; use ``--jobs N``
    (N > 1) to run in parallel. Falls back to normal sequential execution
    for degenerate cases (dry-run or jobs <= 1), including its hooks.
    """

    def __init__(self, config):
        super(ParallelRunner, self).__init__(config)
        self.worker_hook_failures = 0
        self.cleanups_failed = False
        self._worker_init_failures = {}
        self._feature_errors = {}   # -- filename => error_text
        self._idle_worker_deaths = 0
        self._output_mergers = {}

    def load_hooks(self, filename=None):
        super(ParallelRunner, self).load_hooks(filename)
        if "before_parallel" not in self.hooks:
            # -- DEFAULT-HOOK (like "before_all"): Setup logging subsystem.
            # HINT: Not a user-defined hook (see: _has_user_defined_hook).
            self.hooks["before_parallel"] = self.before_all_default_hook

    def run_with_paths(self):
        self.context = Context(self)
        self.load_hooks()

        # -- STEP: Select feature files (parsing is done where it is needed).
        locations = [location for location in self.feature_locations()
                     if not self.config.exclude(location)]
        work_items = group_locations_by_filename(locations)

        if self.config.dry_run or self.config.jobs <= 1:
            # -- DEGENERATE CASE: Run sequentially (like: Runner).
            # HINT: The number of feature files does not matter here,
            # otherwise it would decide which hooks are called.
            self.load_step_definitions()
            self.features.extend(parse_features(locations,
                                                language=self.config.lang))
            self.formatters = make_formatters(self.config, self.config.outputs)
            return self.run_model()

        self._validate_parallel_hooks()
        return self.run_parallel(work_items)

    # -- HOOK SUPPORT:
    def _has_user_defined_hook(self, hook_name):
        hook = self.hooks.get(hook_name)
        if hook is None:
            return False
        # -- EXCLUDE: Injected default hook (see: load_hooks()).
        default_hook_func = Runner.before_all_default_hook
        return getattr(hook, "__func__", hook) is not default_hook_func

    def _validate_parallel_hooks(self):
        """An existing "*_all" hook requires an explicit parallel-mode choice."""
        for all_hook_name, alternatives in PARALLEL_HOOK_REQUIREMENTS.items():
            if not self._has_user_defined_hook(all_hook_name):
                continue
            if not any(self._has_user_defined_hook(hook_name)
                       for hook_name in alternatives):
                raise ConfigError(
                    'PARALLEL: environment file defines "%(all_hook)s", '
                    'which is not called with --jobs > 1. '
                    'Define "%(parent_hook)s" (parent, once) and/or '
                    '"%(worker_hook)s" (per worker) -- e.g. call '
                    '%(all_hook)s(context) from one of them -- to state '
                    'explicitly what should happen.' % dict(
                        all_hook=all_hook_name,
                        parent_hook=alternatives[0],
                        worker_hook=alternatives[1]))

    # -- PARALLEL EXECUTION:
    def select_worker_formats(self):
        """Select the formatter names for the workers (and validate them).

        :raises ConfigError: If a formatter cannot be used with "--jobs > 1".
        """
        config = self.config
        formats = config.format or [config.default_format]
        worker_formats, notes = resolve_worker_formats(
            formats, outfile_bound=select_outfile_bound_formats(config))
        for note in notes:
            print(note)
        return worker_formats

    def make_worker_setup(self, worker_formats):
        """Create the (picklable) setup data for the worker processes.

        HINT: Must be called after the "before_parallel" hook, so that
        configuration changes made by this hook are seen by the workers.

        A worker rebuilds its configuration like the parent process did:
        from the command line and the config-files. If the configuration
        was built programmatically, describe this with these (optional)
        configuration attributes:

        * ``config.command_args``: Command args as list (default: sys.argv).
        * ``config.command_kwargs``: Keyword args of the Configuration.
        * ``config.command_load_config``: If config-files are loaded.
        """
        config = self.config
        dropped = []
        config_kwargs = select_picklable_params(
            getattr(config, "command_kwargs", None) or {}, dropped)

        config_params = {name: getattr(config, name)
                         for name in PROPAGATED_CONFIG_PARAMS
                         if hasattr(config, name)}
        # -- HINT: One non-picklable value should not drop all userdata.
        dropped_userdata = []
        userdata = select_picklable_params(
            dict(config_params.pop("userdata", None) or {}), dropped_userdata)
        config_params = select_picklable_params(config_params, dropped)
        config_params["userdata"] = userdata

        dropped.extend("userdata[%s]" % name for name in dropped_userdata)
        if dropped:
            sys.stderr.write(
                "PARALLEL: WARNING -- not picklable, therefore not sent "
                "to the workers: %s\n" % ", ".join(sorted(dropped)))
        # -- HINT: Without "config.command_args", a worker uses the command
        # line of the parent process ("spawn" provides sys.argv to workers).
        command_args = getattr(config, "command_args", None)
        if isinstance(command_args, (list, tuple)):
            command_args = list(command_args)
        else:
            command_args = None
        return {
            "command_args": command_args,
            "config_kwargs": config_kwargs,
            "load_config": getattr(config, "command_load_config", True),
            "config_params": config_params,
            "worker_format": worker_formats,
            "jobs": config.jobs,
        }

    def open_output_mergers(self, worker_formats):
        """Open the outputs of the formatters with an own output.

        The workers send the output chunks of these formatters (one chunk
        per feature) and this process merges them into the real output.

        :return: Stream-openers that were opened (to close them again).
        """
        outputs = self.config.outputs or []
        stream_openers = []
        for worker_format in worker_formats:
            if worker_format.output is None or worker_format.merger == "direct":
                continue
            if worker_format.output < len(outputs):
                stream_opener = outputs[worker_format.output]
            else:
                # -- LIKE: make_formatters() without a paired output.
                stream_opener = StreamOpener(stream=sys.stdout)
            merger_class = OUTPUT_MERGER_CLASSES[worker_format.merger]
            self._output_mergers[worker_format.output] = \
                merger_class(stream_opener.open())
            stream_openers.append(stream_opener)
        return stream_openers

    def close_output_mergers(self, stream_openers):
        for merger in self._output_mergers.values():
            merger.close()
        self._output_mergers = {}
        for stream_opener in stream_openers:
            stream_opener.close()

    def ensure_junit_directory_exists(self):
        """Create the JUnit report directory before any worker needs it.

        HINT: Workers write their per-feature JUnit reports concurrently and
        the JUnit reporter does not create this directory in a race-free way.
        """
        config = self.config
        junit_directory = getattr(config, "junit_directory", None)
        if getattr(config, "junit", False) and isinstance(junit_directory, str):
            os.makedirs(junit_directory, exist_ok=True)

    def run_parallel(self, work_items):
        config = self.config
        start_time = time.time()
        # -- HINT: Fail early (before any hook runs) on unsupported formats.
        worker_formats = self.select_worker_formats()
        self.ensure_junit_directory_exists()

        self.context._set_root_attribute("jobs", config.jobs)
        summary_reporter = select_summary_reporter(config.reporters)
        if summary_reporter is not None:
            summary_reporter.testrun_started()

        failed_count = 0
        undefined_steps = set()
        processed = set()
        # -- HINT: Fail early (before any hook runs) if an outfile
        # cannot be opened (like: sequential mode).
        output_openers = self.open_output_mergers(worker_formats)
        try:
            if not self.run_hook("before_parallel"):
                self.abort(reason="HOOK-ERROR in hook=before_parallel")
            if work_items and not self.aborted:
                worker_setup = self.make_worker_setup(worker_formats)
                failed_count = self._run_work_items(
                    work_items, worker_setup, summary_reporter,
                    undefined_steps, processed)

            # -- HINT: Worker shutdown-hooks have run now (on process exit).
            self.worker_hook_failures += \
                sum(self._worker_init_failures.values())

            # -- REPORT: Features that never ran (cancelled/aborted)
            # as untested.
            self._report_untested_features(work_items, processed)
        finally:
            # -- ENSURE: The "after_parallel" hook runs if "before_parallel"
            # did (even on an unexpected error), with complete outfiles.
            self.close_output_mergers(output_openers)
            self.run_hook_with_capture("after_parallel")
            try:
                self.context._do_remaining_cleanups()
            except Exception:  # pylint: disable=broad-except
                self.cleanups_failed = True

        if self.aborted:
            print("\nABORTED: By user.")
        if summary_reporter is not None:
            summary_reporter.duration = time.time() - start_time
        for reporter in config.reporters:
            reporter.end()

        self._undefined_steps = sorted(undefined_steps)
        failed = ((failed_count > 0) or self.aborted
                  or (self.hook_failures > 0)
                  or (self.worker_hook_failures > 0)
                  or (len(self._undefined_steps) > 0)
                  or self.cleanups_failed)
        return failed

    def _run_work_items(self, work_items, worker_setup, summary_reporter,
                        undefined_steps, processed):
        """Run the work items (feature files) in worker processes.

        Returns after all worker processes have exited.

        :return: Number of failed work items.
        """
        # pylint: disable=too-many-arguments, too-many-locals
        config = self.config
        mp_context = multiprocessing.get_context("spawn")
        shutdown_failures = mp_context.Value("i", 0)
        schedule = TaskSchedule(work_items)
        workers = []

        def start_worker(worker_id):
            workers.append(WorkerProcess(worker_id, mp_context, worker_setup,
                                         shutdown_failures))

        failed_count = 0
        try:
            for worker_id in range(min(config.jobs, len(work_items))):
                start_worker(worker_id)

            # -- HINT: Ends when all worker processes have exited
            # (their shutdown-hooks have run then).
            while workers:
                for worker in wait_for_workers(workers):
                    filename = worker.task
                    result = self._select_task_result(worker, workers)
                    if result is not None:
                        worker.task = None
                        schedule.task_done()
                    if result is not None and result["serial_deferred"]:
                        self._process_output(result)
                        schedule.defer_as_serial(filename)
                    elif result is not None:
                        if result["status"] is not None or result["no_feature"]:
                            # -- HINT: A feature without status did not run.
                            # It is reported as untested (see below).
                            # BEFORE its counts are merged: An interrupt
                            # must not cause that it is reported twice.
                            processed.add(filename)
                        self._process_result(result, summary_reporter,
                                             undefined_steps)
                        if result["failed"]:
                            failed_count += 1
                        if result["task_errored"]:
                            self._feature_errors[filename] = \
                                result["error_text"]

                        if ((result["worker_setup_failed"]
                                or result["fatal_error"] or result["aborted"])
                                and not self.aborted):
                            # -- LIKE SEQUENTIAL MODE: before_all hook-error,
                            # parse-error and context.abort() in a worker
                            # abort the test-run.
                            self.abort(reason=result["error_text"]
                                       or "Test-run aborted in %s" % filename)
                        if result["failed"] and config.stop:
                            # -- FAIL-EARLY (best-effort): Features that
                            # are already in-flight finish.
                            schedule.cancel()
                    if self.aborted:
                        schedule.cancel()

                    if (worker not in workers and not worker.stopping
                            and schedule.has_tasks()):
                        # -- REPLACE: A worker that died.
                        # HINT: Not if the test-run is aborted/stopped.
                        start_worker(worker.worker_id)
                    self._dispatch_tasks(workers, schedule)
        except KeyboardInterrupt:
            self.abort(reason="KeyboardInterrupt")
            self._terminate_workers(workers)
        except BaseException:
            # -- ENSURE: No worker process is left behind.
            self._terminate_workers(workers)
            raise

        self.worker_hook_failures += shutdown_failures.value
        return failed_count

    @staticmethod
    def _dispatch_tasks(workers, schedule):
        """Give each idle worker its next task (or let it shut down)."""
        for worker in workers:
            if not worker.idle:
                continue
            task = schedule.next_task()
            if task is None:
                # -- HINT: Nothing can run now. Either all is done or another
                # worker is busy (and runs the remaining "@serial" features).
                worker.stop()
            elif not worker.run(*task):
                # -- WORKER DIED while it was idle: Its end is processed soon.
                # HINT: This feature did not run, another worker gets it.
                schedule.put_back(task)

    def _select_task_result(self, worker, workers):
        """Process the next message of a worker process.

        :return: Result of the worker's task (as result dict) or None.
        """
        message = worker.receive()
        filename = worker.task
        if message is None:
            # -- WORKER PROCESS HAS ENDED:
            workers.remove(worker)
            exitcode = worker.close()
            if filename is not None:
                # -- WORKER DIED while it ran a feature: Only this feature
                # is lost. It is reported as errored, the test-run goes on.
                return make_result(
                    filename, worker_id=worker.worker_id, task_errored=True,
                    error_text="PARALLEL-WORKER DIED in %s (exit code: %s)"
                               % (filename, exitcode))
            if not worker.stopping:
                # -- WORKER DIED without a task: No feature is lost.
                self.worker_hook_failures += 1
                self._idle_worker_deaths += 1
                error_text = ("PARALLEL-WORKER %d DIED (exit code: %s)"
                              % (worker.worker_id, exitcode))
                if (not worker.ready
                        or self._idle_worker_deaths > self.config.jobs):
                    # -- DIED DURING ITS SETUP (or: again and again):
                    # A replacement would probably die the same way.
                    self.abort(reason=error_text)
                else:
                    sys.stderr.write(error_text + "\n")
            elif exitcode != 0:
                # -- WORKER DIED while it shut down.
                self.worker_hook_failures += 1
                sys.stderr.write(
                    "PARALLEL-WORKER %d DIED on shutdown (exit code: %s)\n"
                    % (worker.worker_id, exitcode))
            return None

        kind, payload = message
        if kind == "ready":
            # -- HINT: A worker may not get any task (to report this with).
            worker.ready = True
            self._worker_init_failures[worker.worker_id] = \
                payload["init_hook_failures"]
            if payload["setup_failed"]:
                # -- LIKE SEQUENTIAL MODE: before_all hook-error.
                self._worker_init_failures[worker.worker_id] = \
                    max(payload["init_hook_failures"], 1)
                self.abort(reason="PARALLEL-WORKER %d SETUP FAILED"
                                  % worker.worker_id)
            return None
        if kind == "result":
            return payload
        # -- TASK ERROR: No usable result from the worker.
        return make_result(filename, worker_id=worker.worker_id,
                           task_errored=True,
                           error_text="PARALLEL-WORKER FAILURE in %s: %s"
                                      % (filename, payload))

    def _report_untested_features(self, work_items, processed):
        """Report features that did not run to the reporters (as untested).

        A feature whose task errored without any result is reported as
        errored (HINT: :attr:`_feature_errors`).
        """
        for filename, locations in work_items.items():
            if filename in processed:
                continue
            try:
                features = parse_feature_locations(locations,
                                                   language=self.config.lang)
            except Exception:  # pylint: disable=broad-except
                # -- SKIP: Unparsable feature (already reported as error).
                continue

            self.features.extend(features)
            for feature in features:
                if filename in self._feature_errors:
                    mark_feature_as_errored(feature,
                                            self._feature_errors[filename],
                                            self.config)
                for reporter in self.config.reporters:
                    reporter.feature(feature)

    def _process_output(self, result):
        """Print one feature's output chunk (and merge its other outputs)."""
        if result["output"]:
            sys.stdout.write(result["output"])
            sys.stdout.flush()
        if result["error_text"]:
            sys.stderr.write(result["error_text"] + "\n")
            sys.stderr.flush()
        for output, text in result["outputs"].items():
            merger = self._output_mergers.get(output)
            if merger is not None and text:
                merger.add(text)

    def _process_result(self, result, summary_reporter, undefined_steps):
        """Print one feature's output chunk and merge its counts."""
        self._process_output(result)

        worker_id = result["worker_id"]
        if worker_id is not None:
            # -- HINT: Same value in each result of one worker (count once).
            self._worker_init_failures[worker_id] = \
                result["worker_init_hook_failures"]
        self.worker_hook_failures += result["hook_failures"]
        undefined_steps.update(
            UndefinedStepInfo(*info) for info in result["undefined_steps"])

        if summary_reporter is not None and result["status"] is not None:
            merge_status_counts(summary_reporter.feature_summary,
                                result["feature_summary"])
            merge_status_counts(summary_reporter.rule_summary,
                                result["rule_summary"])
            merge_status_counts(summary_reporter.scenario_summary,
                                result["scenario_summary"])
            merge_status_counts(summary_reporter.step_summary,
                                result["step_summary"])
            for kind, location, name in result["problematic_scenarios"]:
                scenario_info = ScenarioInfo(location, name)
                if kind == "failed":
                    summary_reporter.failed_scenarios.append(scenario_info)
                else:
                    summary_reporter.errored_scenarios.append(scenario_info)

    @staticmethod
    def _terminate_workers(workers):
        """Hard-stop of the worker processes (without shutdown-hooks)."""
        for worker in workers:
            try:
                worker.terminate()
            except Exception:  # pylint: disable=broad-except
                pass
        # -- HINT: One deadline for all workers (not: one after the other).
        deadline = time.time() + WORKER_TERMINATE_TIMEOUT
        while workers:
            try:
                workers[-1].close(timeout=max(0.0, deadline - time.time()))
                workers.pop()
            except KeyboardInterrupt:
                continue
