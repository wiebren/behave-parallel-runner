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
  (:class:`concurrent.futures.ProcessPoolExecutor` with "spawn" start-method).
* Work unit: one feature file per task. The parent sends the feature file
  locations (``filename`` or ``filename:line``, so that scenario selection
  is preserved). A worker parses them, runs the feature with a normal
  (sequential) runner runtime and sends back a picklable result
  (status counts, captured output chunk, undefined steps, ...).
* The parent prints each feature's output chunk when its task completes
  (whole chunks, completion order), merges the counts into the summary
  reporter and composes the final exit status like the sequential runner.

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

import atexit
import importlib
import io
import multiprocessing
import os
import pickle
import sys
import time
import traceback
from collections import OrderedDict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool

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
    "behave.formatter.json:JSONFormatter",
    "behave.formatter.rerun:RerunFormatter",
    "behave.formatter.steps:AbstractStepsFormatter",
    "behave.formatter.tags:AbstractTagsFormatter",
)

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


def parse_feature_locations(location_texts, language=None):
    """Parse feature file location texts, like: "alice.feature:12"."""
    locations = []
    for location_text in location_texts:
        location = FileLocationParser.parse(location_text)
        locations.append(FileLocation(os.path.normpath(location.filename),
                                      location.line))
    return parse_features(locations, language=language)


def select_complete_testrun_formatter_classes():
    """Select the built-in formatter classes that need the complete test-run.

    HINT: A formatter module that does not exist in the used behave version
    is ignored.
    """
    formatter_classes = []
    for scoped_class_name in COMPLETE_TESTRUN_FORMATTER_CLASS_NAMES:
        module_name, class_name = scoped_class_name.split(":")
        try:
            module = importlib.import_module(module_name)
            formatter_classes.append(getattr(module, class_name))
        except (ImportError, AttributeError):
            continue
    return tuple(formatter_classes)


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


def resolve_worker_formats(formats, outfile_bound=None):
    """Compute the formatter names that workers should use.

    See :func:`needs_complete_testrun()` for the formatters that are rejected.

    :param formats: Formatter names requested for this test run.
    :param outfile_bound: Flags that tell if format[i] writes to an outfile.
    :return: Tuple (worker_formats, notes) -- notes are user-facing messages.
    :raises ConfigError: If a formatter cannot be used with "--jobs > 1"
        (or if its formatter class cannot be resolved at all).
    """
    outfile_bound = list(outfile_bound or [])
    worker_formats = []
    notes = []
    for index, name in enumerate(formats):
        is_outfile_bound = (index < len(outfile_bound) and outfile_bound[index])
        if is_outfile_bound:
            raise ConfigError(
                'PARALLEL: formatter "%s" with --outfile is not supported '
                'with --jobs > 1 (many workers cannot write one file). '
                'Use --jobs=1 or drop this formatter.' % name)
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
        if name not in worker_formats:
            worker_formats.append(name)
    if not worker_formats:
        worker_formats.append("plain")
    return worker_formats, notes


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
        "output": "",
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
_worker_cancel_event = None


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
    config.format = list(worker_setup["worker_format"])
    config.default_format = "plain"
    # -- HINT: The parent prints merged undefined-step snippets once.
    config.show_snippets = False
    # -- HINT: No per-worker summary; the parent prints the merged summary.
    config.summary = False
    config.reporters = [reporter for reporter in config.reporters
                        if not isinstance(reporter, AbstractSummaryReporter)]


def _worker_init(worker_setup, worker_id_counter, shutdown_failures,
                 cancel_event=None):
    """Initialize one worker process (ProcessPoolExecutor initializer)."""
    # pylint: disable=global-statement
    global _worker_runner, _worker_output, _worker_id
    global _worker_setup_failed, _worker_init_hook_failures
    global _worker_shutdown_failures, _worker_cancel_event

    # -- SETUP: Use one output stream for the whole worker lifetime,
    # so that logging handlers, etc. keep writing into a collected stream.
    _worker_output = WorkerOutput()
    _worker_shutdown_failures = shutdown_failures
    _worker_cancel_event = cancel_event
    sys.stdout = _worker_output
    sys.stderr = _worker_output
    try:
        with worker_id_counter.get_lock():
            _worker_id = worker_id_counter.value
            worker_id_counter.value += 1

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
        atexit.register(_worker_shutdown)
    except Exception:  # pylint: disable=broad-except
        _worker_setup_failed = True
        _worker_init_hook_failures = max(_worker_init_hook_failures, 1)
        traceback.print_exc()
    finally:
        _emit_worker_output(_worker_output.drain())


def _worker_shutdown():
    """Finalize one worker process (atexit; skipped on hard terminate)."""
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


def _run_feature_task(location_texts):
    """Run one feature file in this worker process; returns a result dict."""
    # pylint: disable=global-statement
    global _worker_init_hook_failures
    runner = _worker_runner
    filename = FileLocationParser.parse(location_texts[0]).filename
    result = make_result(filename, worker_id=_worker_id)

    # -- STEP: Report worker-setup failures in EACH result of this worker
    # (a worker may not win any task or its first task may be cancelled).
    result["worker_init_hook_failures"] = _worker_init_hook_failures
    if _worker_setup_failed or runner is None:
        result["worker_setup_failed"] = True
        result["worker_init_hook_failures"] = max(_worker_init_hook_failures, 1)
        result["error_text"] = ("PARALLEL-WORKER SETUP FAILED: %s "
                                "(feature not run)" % filename)
        result["output"] = (_worker_output.drain()
                            if _worker_output is not None else "")
        return result

    cancelled = (_worker_cancel_event is not None
                 and _worker_cancel_event.is_set())
    if cancelled or runner.aborted:
        # -- NOT RUN: The test-run is stopped/aborted (fail-early, abort).
        # HINT: status stays None -- the parent reports it as untested.
        # HINT: A worker's Context stays aborted (like: sequential mode).
        result.update(failed=False, aborted=runner.aborted,
                      output=_worker_output.drain())
        return result

    hook_failures0 = runner.hook_failures
    undefined_steps0 = len(runner.undefined_steps)
    feature = None
    feature_reported = False
    try:
        features = parse_feature_locations(location_texts,
                                           language=runner.config.lang)
        if not features:
            # -- LIKE SEQUENTIAL MODE: parse_features() skips a feature file
            # without any feature (corner case; nothing to run or report).
            result.update(failed=False, no_feature=True,
                          output=_worker_output.drain())
            return result
        feature = features[0]

        runner.feature = feature
        stream_opener = StreamOpener(stream=_worker_output)
        runner.formatters = make_formatters(runner.config, [stream_opener])
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
    return result


# -----------------------------------------------------------------------------
# PARENT SIDE:
# -----------------------------------------------------------------------------
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
        self._worker_pool_broken = False
        self._errored_filenames = set()

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
        if not self.run_hook("before_parallel"):
            self.abort(reason="HOOK-ERROR in hook=before_parallel")
        if work_items and not self.aborted:
            worker_setup = self.make_worker_setup(worker_formats)
            failed_count = self._run_work_items(
                work_items, worker_setup, summary_reporter,
                undefined_steps, processed)

        # -- HINT: Worker shutdown-hooks have run now (on process exit).
        self.worker_hook_failures += sum(self._worker_init_failures.values())

        # -- REPORT: Features that never ran (cancelled/aborted) as untested.
        self._report_untested_features(work_items, processed)

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
        num_workers = min(config.jobs, len(work_items))
        mp_context = multiprocessing.get_context("spawn")
        worker_id_counter = mp_context.Value("i", 0)
        shutdown_failures = mp_context.Value("i", 0)
        # -- HINT: Tasks that a worker has already fetched cannot be
        # cancelled anymore; a worker checks this event before it runs one.
        cancel_event = mp_context.Event()

        failed_count = 0
        cancel_requested = False
        executor = ProcessPoolExecutor(
            max_workers=num_workers, mp_context=mp_context,
            initializer=_worker_init,
            initargs=(worker_setup, worker_id_counter, shutdown_failures,
                      cancel_event))
        try:
            future_to_filename = {}
            pending = set()
            try:
                for filename, locations in work_items.items():
                    future = executor.submit(_run_feature_task, locations)
                    future_to_filename[future] = filename
                    pending.add(future)

                # -- HINT: Do not shutdown the executor inside of this loop.
                # executor.shutdown(cancel_futures=True) never notifies the
                # waiters of the futures that it cancels (waits forever).
                while pending:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        if future.cancelled():
                            continue
                        filename = future_to_filename[future]
                        result = self._select_task_result(future, filename)
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
                            self._errored_filenames.add(filename)

                        if ((result["worker_setup_failed"]
                                or result["fatal_error"] or result["aborted"])
                                and not self.aborted):
                            # -- LIKE SEQUENTIAL MODE: before_all hook-error,
                            # parse-error and context.abort() in a worker
                            # abort the test-run.
                            self.abort(reason=result["error_text"]
                                       or "Test-run aborted in %s" % filename)
                        should_cancel = (self.aborted
                                         or (result["failed"] and config.stop))
                        if should_cancel and not cancel_requested:
                            # -- FAIL-EARLY (best-effort): Cancel pending
                            # tasks; features already in-flight finish.
                            cancel_requested = True
                            cancel_event.set()
                            pending = set(future for future in pending
                                          if not future.cancel())
            except KeyboardInterrupt:
                self._hard_stop_workers(executor, cancel_event, pending)
        finally:
            # -- HINT: Waits until all worker processes have exited
            # (their shutdown-hooks have run then).
            try:
                executor.shutdown(wait=True, cancel_futures=True)
            except KeyboardInterrupt:
                # -- CASE: Interrupted while the workers shut down.
                self._hard_stop_workers(executor, cancel_event, pending)
                executor.shutdown(wait=True, cancel_futures=True)

        self.worker_hook_failures += shutdown_failures.value
        return failed_count

    def _select_task_result(self, future, filename):
        """Select the result of a completed task (as result dict)."""
        error = future.exception()
        if error is None:
            return future.result()

        if isinstance(error, BrokenProcessPool):
            # -- WORKER PROCESS DIED: All remaining tasks fail with this
            # error and it is unknown which feature was running.
            # Abort the test-run and report these features as untested.
            error_text = None
            if not self._worker_pool_broken:
                self._worker_pool_broken = True
                error_text = "PARALLEL-WORKER DIED: %s" % error
            return make_result(filename, fatal_error=True,
                               error_text=error_text)
        # -- TASK ERROR: For example, a result that cannot be unpickled.
        return make_result(filename, task_errored=True,
                           error_text="PARALLEL-WORKER FAILURE in %s: %s"
                                      % (filename, error))

    def _report_untested_features(self, work_items, processed):
        """Report features that did not run to the reporters (as untested).

        A feature whose task errored without any result is reported as
        errored (HINT: :attr:`_errored_filenames`).
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
                if filename in self._errored_filenames:
                    feature.set_status(Status.error)
                for reporter in self.config.reporters:
                    reporter.feature(feature)

    def _process_result(self, result, summary_reporter, undefined_steps):
        """Print one feature's output chunk and merge its counts."""
        if result["output"]:
            sys.stdout.write(result["output"])
            sys.stdout.flush()
        if result["error_text"]:
            sys.stderr.write(result["error_text"] + "\n")
            sys.stderr.flush()

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

    def _hard_stop_workers(self, executor, cancel_event, pending):
        """Abort the test-run on KeyboardInterrupt: Workers stop at once."""
        self.abort(reason="KeyboardInterrupt")
        cancel_event.set()
        for future in pending:
            future.cancel()
        # -- HINT: Must be done before executor.shutdown(),
        # it forgets its worker processes.
        self._terminate_worker_processes(executor)

    @staticmethod
    def _terminate_worker_processes(executor):
        """Best-effort hard-stop of worker processes (uses private API)."""
        processes = getattr(executor, "_processes", None) or {}
        for process in list(processes.values()):
            try:
                process.terminate()
            except Exception:  # pylint: disable=broad-except
                pass

