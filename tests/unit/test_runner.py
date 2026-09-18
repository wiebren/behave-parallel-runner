# -*- coding: UTF-8 -*-
"""
Unit tests for :mod:`behave_parallel_runner.runner` (parallel test runner).

Covers the process-free parts: work-item building, format resolution,
result merging, hook validation, runner-alias wiring and configuration
support.
"""

import re
import threading
from concurrent.futures import Future
from concurrent.futures.process import BrokenProcessPool
from types import SimpleNamespace

import pytest

from behave.configuration import Configuration, DEFAULT_RUNNER_CLASS_NAME
from behave.exception import ConfigError
from behave.formatter import _registry as formatter_registry_module
from behave.formatter._builtins import _BUILTIN_FORMATS
from behave.formatter.base import Formatter
from behave.formatter.json import JSONFormatter
from behave.model_type import FileLocation, Status
from behave.runner import Context
from behave_parallel_runner import runner as runner_parallel
from behave_parallel_runner.runner import (
    ParallelRunner,
    ScenarioInfo,
    UndefinedStepInfo,
    WorkerOutput,
    WorkerRunner,
    _run_feature_task,
    group_locations_by_filename,
    make_result,
    merge_status_counts,
    needs_complete_testrun,
    resolve_worker_formats,
    select_complete_testrun_formatter_classes,
    select_outfile_bound_formats,
    select_picklable_params,
    select_summary_reporter,
)
from behave.runner_plugin import RunnerPlugin
from behave.runner_util import make_undefined_step_snippets
from behave.reporter.summary import SummaryReporterV1, SummaryReporterV2


def make_config(command_args=None, **kwargs):
    return Configuration(command_args or [], load_config=False, **kwargs)


# -----------------------------------------------------------------------------
# WORK ITEMS: Scenario selection by line number must survive (finding 1).
# -----------------------------------------------------------------------------
class TestGroupLocationsByFilename:
    def test_groups_locations_of_one_feature_file(self):
        locations = [FileLocation("alice.feature", 12),
                     FileLocation("bob.feature"),
                     FileLocation("alice.feature", 30)]
        grouped = group_locations_by_filename(locations)
        assert list(grouped.keys()) == ["alice.feature", "bob.feature"]
        assert grouped["alice.feature"] == ["alice.feature:12",
                                            "alice.feature:30"]
        assert grouped["bob.feature"] == ["bob.feature"]

    def test_keeps_line_numbers_in_location_text(self):
        grouped = group_locations_by_filename([FileLocation("a.feature", 5)])
        assert grouped["a.feature"] == ["a.feature:5"]

    def test_works_with_location_strings(self):
        grouped = group_locations_by_filename(["a.feature", "b.feature"])
        assert list(grouped.keys()) == ["a.feature", "b.feature"]


# -----------------------------------------------------------------------------
# WORKER FORMAT RESOLUTION:
# -----------------------------------------------------------------------------
#: Built-in formatters that aggregate over the complete test-run.
AGGREGATING_BUILTIN_FORMATS = [
    "json", "json.pretty", "rerun",
    "sphinx.steps", "steps", "steps.bad", "steps.catalog", "steps.doc",
    "steps.missing", "steps.usage", "tags", "tags.location",
]


class AggregatingUserFormatter(Formatter):
    """User-defined formatter that needs the complete test-run."""
    name = "my.aggregating"
    description = "Aggregates over the complete test-run."
    needs_complete_testrun = True


class MyJSONFormatter(JSONFormatter):
    """User-defined formatter that derives from a built-in one."""
    name = "my.json"
    description = "JSON dump of test run (user-defined variant)."


class ParallelSafeUserFormatter(Formatter):
    """User-defined formatter that can be used per worker."""
    name = "my.steps"
    description = "Parallel-safe user-defined formatter."


class DuckTypedFormatter:
    """User-defined formatter that does not derive from :class:`Formatter`."""
    name = "my.duck_typed"
    description = "Does not inherit from Formatter (duck-typing only)."

    def __init__(self, stream_opener, config):
        self.stream_opener = stream_opener
        self.config = config


@pytest.fixture
def formatter_registry():
    """Provide the formatter registry and restore its entries afterwards."""
    registry = formatter_registry_module._formatter_registry
    saved_entries = dict(registry)
    try:
        yield formatter_registry_module
    finally:
        registry.clear()
        registry.update(saved_entries)


class TestResolveWorkerFormats:
    def test_pretty_is_replaced_by_plain(self):
        worker_formats, notes = resolve_worker_formats(["pretty"])
        assert worker_formats == ["plain"]
        assert any("plain" in note and "pretty" in note for note in notes)

    def test_plain_and_progress_pass_through(self):
        worker_formats, notes = resolve_worker_formats(["plain", "progress"])
        assert worker_formats == ["plain", "progress"]
        assert not notes

    @pytest.mark.parametrize("format_name", AGGREGATING_BUILTIN_FORMATS)
    def test_unsupported_format_is_rejected(self, format_name):
        # -- FINDING 4: Must not silently skip (no output is data-loss).
        with pytest.raises(ConfigError, match=re.escape(format_name)):
            resolve_worker_formats([format_name, "plain"])

    def test_every_other_builtin_format_is_accepted(self):
        # -- FINDING 10: Only aggregating formatters are rejected.
        # HINT: Use the builtin formats (the registry may contain aliases
        # of not-installed formatters, registered by other tests).
        other_formats = [name for name, _class_name in _BUILTIN_FORMATS
                         if name not in AGGREGATING_BUILTIN_FORMATS]
        assert other_formats, "REQUIRE: Some non-aggregating builtin formats"
        for format_name in other_formats:
            worker_formats, _notes = resolve_worker_formats([format_name])
            assert worker_formats  # -- HINT: "pretty" becomes "plain".

    def test_aggregating_user_formatter_is_rejected(self, formatter_registry):
        # -- FINDING 10: Own alias, own class -- the flag decides.
        formatter_registry.register_as("my.aggregating",
                                       AggregatingUserFormatter)
        with pytest.raises(ConfigError, match=re.escape("my.aggregating")):
            resolve_worker_formats(["my.aggregating"])

    def test_derived_json_user_formatter_is_rejected(self, formatter_registry):
        # -- FINDING 10: The flag is inherited from the built-in base class.
        formatter_registry.register_as("my.json", MyJSONFormatter)
        with pytest.raises(ConfigError, match=re.escape("my.json")):
            resolve_worker_formats(["my.json"])

    def test_parallel_safe_formatter_overriding_builtin_name_is_accepted(
            self, formatter_registry):
        # -- FINDING 10: "steps" is only a name -- the registered class counts.
        formatter_registry.register_as("steps", ParallelSafeUserFormatter)
        worker_formats, notes = resolve_worker_formats(["steps"])
        assert worker_formats == ["steps"]
        assert not notes

    def test_formatter_without_the_attribute_is_accepted(self,
                                                         formatter_registry):
        # -- FINDING 10: A formatter needs not inherit from Formatter.
        # HINT: register_as() requires a Formatter subclass -- bypass it here.
        formatter_registry._formatter_registry["my.duck_typed"] = \
            DuckTypedFormatter
        worker_formats, _notes = resolve_worker_formats(["my.duck_typed"])
        assert worker_formats == ["my.duck_typed"]

    def test_unknown_format_is_rejected(self, formatter_registry):
        # -- FINDING 10: Do not mask an unknown/unloadable formatter.
        with pytest.raises(ConfigError, match=re.escape("__unknown__")):
            resolve_worker_formats(["__unknown__"])

    def test_outfile_bound_format_is_rejected(self):
        # -- FINDING 4: An --outfile is never written by workers.
        with pytest.raises(ConfigError, match="outfile"):
            resolve_worker_formats(["plain"], outfile_bound=[True])

    def test_outfile_bound_check_is_positional(self):
        # -- FINDING 5: Only the format bound to the outfile is rejected.
        with pytest.raises(ConfigError, match="progress"):
            resolve_worker_formats(["plain", "progress"],
                                   outfile_bound=[False, True])

    def test_console_bound_formats_are_accepted(self):
        worker_formats, _notes = resolve_worker_formats(
            ["plain", "progress"], outfile_bound=[False, False])
        assert worker_formats == ["plain", "progress"]

    def test_empty_result_falls_back_to_plain(self):
        worker_formats, _notes = resolve_worker_formats([])
        assert worker_formats == ["plain"]

    def test_duplicates_are_removed(self):
        worker_formats, _notes = resolve_worker_formats(["pretty", "plain"])
        assert worker_formats == ["plain"]


class TestNeedsCompleteTestrun:
    """Formatters that cannot run per worker are detected without any
    support of the behave core (no ``Formatter.needs_complete_testrun``).
    """

    @pytest.mark.parametrize("format_name", [
        "json", "json.pretty", "rerun", "steps", "steps.doc",
        "steps.catalog", "steps.usage", "tags", "tags.location",
    ])
    def test_builtin_aggregating_formatters_are_detected(self, format_name):
        formatter_class = formatter_registry_module.select_formatter_class(
            format_name)
        assert needs_complete_testrun(formatter_class) is True

    @pytest.mark.parametrize("format_name", [
        "plain", "pretty", "progress", "progress2", "progress3", "null",
    ])
    def test_builtin_streaming_formatters_are_accepted(self, format_name):
        formatter_class = formatter_registry_module.select_formatter_class(
            format_name)
        assert needs_complete_testrun(formatter_class) is False

    def test_derived_formatter_is_detected(self):
        assert needs_complete_testrun(MyJSONFormatter) is True

    def test_explicit_flag_wins_over_base_class(self):
        class ParallelSafeJSONFormatter(JSONFormatter):
            needs_complete_testrun = False

        assert needs_complete_testrun(ParallelSafeJSONFormatter) is False

    def test_default_of_formatter_base_class_does_not_hide_builtin(
            self, monkeypatch):
        # -- HINT: A behave version may provide this flag in its base class.
        monkeypatch.setattr(Formatter, "needs_complete_testrun", False,
                            raising=False)
        assert needs_complete_testrun(JSONFormatter) is True
        assert needs_complete_testrun(MyJSONFormatter) is True
        assert needs_complete_testrun(ParallelSafeUserFormatter) is False

    def test_explicit_flag_of_user_defined_formatter_is_used(self):
        assert needs_complete_testrun(AggregatingUserFormatter) is True
        assert needs_complete_testrun(ParallelSafeUserFormatter) is False

    def test_duck_typed_formatter_is_accepted(self):
        assert needs_complete_testrun(DuckTypedFormatter) is False

    def test_formatter_factory_function_is_accepted(self):
        # -- HINT: Not a class; issubclass() must not be used with it.
        assert needs_complete_testrun(lambda stream, config: None) is False

    def test_unknown_formatter_module_is_ignored(self, monkeypatch):
        monkeypatch.setattr(
            runner_parallel, "COMPLETE_TESTRUN_FORMATTER_CLASS_NAMES",
            ("behave.formatter.__missing__:MissingFormatter",
             "behave.formatter.json:MissingFormatter",
             "behave.formatter.json:JSONFormatter"))
        formatter_classes = select_complete_testrun_formatter_classes()
        assert formatter_classes == (JSONFormatter,)


class TestEnsureJUnitDirectoryExists:
    def test_creates_nested_directory_if_junit_is_enabled(self, tmp_path):
        junit_directory = tmp_path/"reports"/"junit"
        config = make_config(["--junit",
                              "--junit-directory=%s" % junit_directory])
        ParallelRunner(config).ensure_junit_directory_exists()
        assert junit_directory.is_dir()

    def test_existing_directory_is_tolerated(self, tmp_path):
        config = make_config(["--junit", "--junit-directory=%s" % tmp_path])
        ParallelRunner(config).ensure_junit_directory_exists()
        assert tmp_path.is_dir()

    def test_creates_nothing_if_junit_is_disabled(self, tmp_path):
        junit_directory = tmp_path/"reports"
        config = make_config(["--junit-directory=%s" % junit_directory])
        assert not config.junit
        ParallelRunner(config).ensure_junit_directory_exists()
        assert not junit_directory.exists()


class TestSelectOutfileBoundFormats:
    def test_named_stream_opener_is_outfile_bound(self):
        config = make_config(["-f", "plain", "-o", "out.txt"])
        assert select_outfile_bound_formats(config) == [True]

    def test_default_stdout_opener_is_not_outfile_bound(self):
        config = make_config(["-f", "plain"])
        assert select_outfile_bound_formats(config) == [False]


# -----------------------------------------------------------------------------
# RESULT MERGING:
# -----------------------------------------------------------------------------
class TestMergeStatusCounts:
    def test_merges_counts_key_wise(self):
        target = {"passed": 1, "failed": 0}
        merge_status_counts(target, {"passed": 2, "failed": 1})
        assert target == {"passed": 3, "failed": 1}

    def test_merges_unknown_keys(self):
        target = {"passed": 1}
        merge_status_counts(target, {"error": 2})
        assert target == {"passed": 1, "error": 2}


class TestUndefinedStepInfo:
    def test_works_with_undefined_step_snippets(self):
        snippets = make_undefined_step_snippets(
            [UndefinedStepInfo("given", "some step")])
        assert len(snippets) == 1
        assert "some step" in snippets[0]

    def test_snippet_generation_can_escape_quotes(self):
        # -- HINT: make_undefined_step_snippet() mutates step.name.
        snippets = make_undefined_step_snippets(
            [UndefinedStepInfo("given", "it's quoted")])
        assert r"it\'s quoted" in snippets[0]

    def test_deduplicates_in_set(self):
        infos = {UndefinedStepInfo("given", "a step"),
                 UndefinedStepInfo("given", "a step"),
                 UndefinedStepInfo("when", "a step")}
        assert len(infos) == 2

    def test_is_sortable(self):
        infos = sorted([UndefinedStepInfo("when", "b"),
                        UndefinedStepInfo("given", "a")])
        assert infos[0].step_type == "given"


class TestScenarioInfo:
    def test_duck_types_scenario_for_summary_reporter(self):
        config = make_config()
        reporter = SummaryReporterV1(config)
        reporter.failed_scenarios.append(ScenarioInfo("a.feature:3", "S1"))
        # -- MUST NOT RAISE: Uses scenario.location and scenario.name.
        reporter.print_failing_scenarios()


class TestMakeResult:
    def test_defaults_are_a_not_run_failure(self):
        result = make_result("some.feature")
        assert result["failed"] is True
        assert result["status"] is None
        assert result["worker_setup_failed"] is False
        assert result["fatal_error"] is False
        assert result["hook_failures"] == 0
        assert result["worker_init_hook_failures"] == 0
        assert result["aborted"] is False
        assert result["no_feature"] is False
        assert result["task_errored"] is False


class TestSelectSummaryReporter:
    def test_selects_summary_reporter_v1(self):
        config = make_config()
        reporter = SummaryReporterV1(config)
        assert select_summary_reporter([object(), reporter]) is reporter

    def test_returns_none_without_summary_reporter(self):
        assert select_summary_reporter([object()]) is None

    def test_rejects_unsupported_summary_reporter(self):
        # -- FINDING 10: V2 internals differ; must not crash mid-run.
        config = make_config()
        with pytest.raises(ConfigError, match="SummaryReporterV2"):
            select_summary_reporter([SummaryReporterV2(config)])


class TestSelectPicklableParams:
    def test_keeps_picklable_params(self):
        assert select_picklable_params({"a": 1, "b": "x"}) == {"a": 1, "b": "x"}

    def test_drops_non_picklable_params(self):
        selected = select_picklable_params({"good": 1, "bad": lambda: None})
        assert selected == {"good": 1}

    def test_collects_names_of_dropped_params(self):
        dropped = []
        select_picklable_params({"good": 1, "bad": lambda: None}, dropped)
        assert dropped == ["bad"]


class TestWorkerOutput:
    def test_drain_returns_and_clears_text(self):
        output = WorkerOutput()
        output.write("hello")
        assert output.drain() == "hello"
        assert output.drain() == ""


# -----------------------------------------------------------------------------
# PARENT-SIDE RESULT PROCESSING:
# -----------------------------------------------------------------------------
class TestProcessResult:
    @staticmethod
    def make_runner():
        runner = ParallelRunner(make_config())
        runner.context = None
        return runner

    def test_worker_init_failures_are_counted_once_per_worker(self):
        # -- FINDING 2: Must not be lost, must not be counted per task.
        runner = self.make_runner()
        for _ in range(3):
            result = make_result("a.feature", worker_id=0,
                                 worker_init_hook_failures=1)
            runner._process_result(result, None, set())
        runner.worker_hook_failures += sum(runner._worker_init_failures.values())
        assert runner.worker_hook_failures == 1

    def test_worker_init_failures_are_counted_per_worker(self):
        runner = self.make_runner()
        for worker_id in (0, 1):
            result = make_result("a.feature", worker_id=worker_id,
                                 worker_init_hook_failures=1)
            runner._process_result(result, None, set())
        runner.worker_hook_failures += sum(runner._worker_init_failures.values())
        assert runner.worker_hook_failures == 2

    def test_task_hook_failures_are_counted_per_task(self):
        runner = self.make_runner()
        for _ in range(2):
            result = make_result("a.feature", worker_id=0, hook_failures=1)
            runner._process_result(result, None, set())
        assert runner.worker_hook_failures == 2

    def test_result_without_status_is_not_merged(self):
        # -- FINDING 6: A feature that did not run keeps status=None.
        config = make_config()
        summary_reporter = SummaryReporterV1(config)
        runner = self.make_runner()
        result = make_result("a.feature", worker_id=0,
                             feature_summary={"passed": 1})
        runner._process_result(result, summary_reporter, set())
        assert summary_reporter.feature_summary["passed"] == 0


# -----------------------------------------------------------------------------
# HOOK VALIDATION:
# -----------------------------------------------------------------------------
class TestValidateParallelHooks:
    @staticmethod
    def make_runner(hooks):
        runner = ParallelRunner(make_config())
        runner.hooks = dict(hooks)
        return runner

    @staticmethod
    def hook(context):
        pass

    def test_passes_without_any_hooks(self):
        self.make_runner({})._validate_parallel_hooks()  # -- SHOULD NOT RAISE

    def test_fails_with_before_all_only(self):
        runner = self.make_runner({"before_all": self.hook})
        with pytest.raises(ConfigError, match="before_all"):
            runner._validate_parallel_hooks()

    def test_fails_with_after_all_only(self):
        runner = self.make_runner({"after_all": self.hook})
        with pytest.raises(ConfigError, match="after_all"):
            runner._validate_parallel_hooks()

    @pytest.mark.parametrize("counterpart",
                             ["before_parallel", "before_worker"])
    def test_passes_with_before_all_and_counterpart(self, counterpart):
        runner = self.make_runner({"before_all": self.hook,
                                   counterpart: self.hook})
        runner._validate_parallel_hooks()  # -- SHOULD NOT RAISE

    @pytest.mark.parametrize("counterpart",
                             ["after_parallel", "after_worker"])
    def test_passes_with_after_all_and_counterpart(self, counterpart):
        runner = self.make_runner({"after_all": self.hook,
                                   counterpart: self.hook})
        runner._validate_parallel_hooks()  # -- SHOULD NOT RAISE

    def test_all_hooks_are_checked_independently(self):
        runner = self.make_runner({"before_all": self.hook,
                                   "before_worker": self.hook,
                                   "after_all": self.hook})
        with pytest.raises(ConfigError, match="after_all"):
            runner._validate_parallel_hooks()

    def test_injected_default_hooks_are_not_user_defined(self):
        # -- Runner.load_hooks() injects "before_all",
        # ParallelRunner.load_hooks() injects "before_parallel".
        # Neither may satisfy nor trigger the explicitness rule.
        runner = ParallelRunner(make_config())
        runner.hooks = {"before_all": runner.before_all_default_hook,
                        "before_parallel": runner.before_all_default_hook}
        runner._validate_parallel_hooks()  # -- SHOULD NOT RAISE

    def test_injected_before_parallel_does_not_satisfy_before_all(self):
        runner = ParallelRunner(make_config())
        runner.hooks = {"before_all": self.hook,
                        "before_parallel": runner.before_all_default_hook}
        with pytest.raises(ConfigError, match="before_all"):
            runner._validate_parallel_hooks()


# -----------------------------------------------------------------------------
# WIRING:
# -----------------------------------------------------------------------------
class TestRunnerWiring:
    SCOPED_CLASS_NAME = "behave_parallel_runner:ParallelRunner"

    def test_package_provides_parallel_runner(self):
        import behave_parallel_runner
        assert behave_parallel_runner.ParallelRunner is ParallelRunner

    def test_runner_option_loads_parallel_runner(self):
        config = make_config(["--runner=%s" % self.SCOPED_CLASS_NAME])
        assert isinstance(RunnerPlugin().make_runner(config), ParallelRunner)

    def test_config_file_setting_loads_parallel_runner(self, tmp_path,
                                                       monkeypatch):
        config_file = tmp_path/"behave.ini"
        config_file.write_text("""
[behave]
runner = %s
""" % self.SCOPED_CLASS_NAME)
        monkeypatch.chdir(tmp_path)
        config = Configuration(["--jobs=2"])
        assert isinstance(RunnerPlugin().make_runner(config), ParallelRunner)


# -----------------------------------------------------------------------------
# WORKER SETUP: Made after the "before_parallel" hook (config changes are seen).
# -----------------------------------------------------------------------------
class TestMakeWorkerSetup:
    def test_uses_current_config_values(self):
        # -- HINT: Hook "before_parallel" may rebind these config attributes.
        runner = ParallelRunner(make_config())
        runner.config.tags = "@changed"
        runner.config.lang = "de"
        runner.config.userdata = {"color": "blue"}
        worker_setup = runner.make_worker_setup(["plain"])
        config_params = worker_setup["config_params"]
        assert config_params["tags"] == "@changed"
        assert config_params["lang"] == "de"
        assert config_params["userdata"] == {"color": "blue"}
        assert worker_setup["worker_format"] == ["plain"]

    def test_non_picklable_userdata_value_drops_only_this_value(self, capsys):
        runner = ParallelRunner(make_config())
        runner.config.userdata.update(color="blue", lock=threading.Lock())
        worker_setup = runner.make_worker_setup(["plain"])
        assert worker_setup["config_params"]["userdata"] == {"color": "blue"}
        captured = capsys.readouterr()
        assert "PARALLEL: WARNING" in captured.err
        assert "userdata[lock]" in captured.err

    def test_without_command_args_workers_use_command_line(self):
        # -- HINT: command_args=None means "use sys.argv" in a worker
        # (the "spawn" start-method provides the parent's sys.argv).
        runner = ParallelRunner(make_config(["--jobs=2"]))
        for name in ("command_args", "command_kwargs", "command_load_config"):
            # -- HINT: A behave version may provide these attributes itself.
            if hasattr(runner.config, name):
                delattr(runner.config, name)
        worker_setup = runner.make_worker_setup(["plain"])
        assert worker_setup["command_args"] is None
        assert worker_setup["config_kwargs"] == {}
        assert worker_setup["load_config"] is True

    def test_programmatic_configuration_can_be_described(self):
        # -- HINT: Optional config attributes describe how it was built.
        runner = ParallelRunner(make_config(["--jobs=2"], tags="@one"))
        runner.config.command_args = ("--jobs=2", "features")
        runner.config.command_kwargs = {"tags": "@one"}
        runner.config.command_load_config = False
        worker_setup = runner.make_worker_setup(["plain"])
        assert worker_setup["command_args"] == ["--jobs=2", "features"]
        assert worker_setup["config_kwargs"] == {"tags": "@one"}
        assert worker_setup["load_config"] is False

    def test_empty_command_args_are_kept(self):
        # -- HINT: An empty list must not be replaced by sys.argv.
        runner = ParallelRunner(make_config())
        runner.config.command_args = []
        worker_setup = runner.make_worker_setup(["plain"])
        assert worker_setup["command_args"] == []

    def test_non_picklable_config_kwargs_are_reported(self, capsys):
        runner = ParallelRunner(make_config())
        runner.config.command_kwargs = {"my_param": lambda: None}
        worker_setup = runner.make_worker_setup(["plain"])
        assert "my_param" not in worker_setup["config_kwargs"]
        assert "my_param" in capsys.readouterr().err

    def test_no_warning_if_all_params_are_picklable(self, capsys):
        runner = ParallelRunner(make_config())
        runner.make_worker_setup(["plain"])
        assert "WARNING" not in capsys.readouterr().err

    def test_before_parallel_hook_runs_before_worker_setup_is_made(
            self, monkeypatch):
        runner = ParallelRunner(make_config(["--jobs=2"]))
        runner.context = Context(runner)
        runner.hooks["before_parallel"] = \
            lambda ctx: ctx.config.userdata.update(color="blue")
        seen = {}

        def fake_run_work_items(work_items, worker_setup, *args):
            seen.update(worker_setup["config_params"]["userdata"])
            return 0

        monkeypatch.setattr(runner, "_run_work_items", fake_run_work_items)
        monkeypatch.setattr(runner, "_report_untested_features",
                            lambda *args: None)
        runner.run_parallel({"a.feature": ["a.feature"]})
        assert seen == {"color": "blue"}


# -----------------------------------------------------------------------------
# DEGENERATE CASES: The number of feature files must not select the hooks.
# -----------------------------------------------------------------------------
class TestRunWithPaths:
    @staticmethod
    def make_runner(monkeypatch, command_args, locations):
        runner = ParallelRunner(make_config(command_args))
        calls = []
        monkeypatch.setattr(runner, "load_hooks", lambda: None)
        monkeypatch.setattr(runner, "load_step_definitions", lambda: None)
        monkeypatch.setattr(runner, "feature_locations", lambda: locations)
        monkeypatch.setattr(runner_parallel, "make_formatters",
                            lambda config, openers: [])
        monkeypatch.setattr(runner, "run_model",
                            lambda: calls.append("run_model"))
        monkeypatch.setattr(runner, "run_parallel",
                            lambda work_items: calls.append("run_parallel"))
        return runner, calls

    def test_one_feature_file_runs_in_parallel_mode(self, monkeypatch):
        runner, calls = self.make_runner(monkeypatch, ["--jobs=2"],
                                         [FileLocation("a.feature")])
        runner.run_with_paths()
        assert calls == ["run_parallel"]

    def test_no_feature_file_runs_in_parallel_mode(self, monkeypatch):
        runner, calls = self.make_runner(monkeypatch, ["--jobs=2"], [])
        runner.run_with_paths()
        assert calls == ["run_parallel"]

    @pytest.mark.parametrize("command_args", [
        ["--jobs=1"], ["--jobs=2", "--dry-run"],
    ])
    def test_falls_back_to_sequential_mode(self, monkeypatch, command_args):
        runner, calls = self.make_runner(monkeypatch, command_args, [])
        runner.run_with_paths()
        assert calls == ["run_model"]


# -----------------------------------------------------------------------------
# PARENT RUN LOOP: With a fake executor (process-free).
# -----------------------------------------------------------------------------
class FakeProcess:
    def __init__(self):
        self.terminated = False

    def terminate(self):
        self.terminated = True


class FakeExecutor:
    """Fake ProcessPoolExecutor: outcomes[i] describes the future of task i.

    * result dict or exception: Future is done when it is submitted.
    * ("running", outcome): Future is running (not cancellable), done later.
    * None: Future stays pending (cancellable).
    """
    instance = None
    outcomes = ()

    shutdown_errors = ()
    submit_error = None

    def __init__(self, **kwargs):
        type(self).instance = self
        self.initargs = kwargs["initargs"]
        self.shutdown_errors = list(self.shutdown_errors)
        self.futures = []
        self.timers = []
        self.shutdown_calls = []
        self.process = FakeProcess()
        self._processes = {1: self.process}

    @staticmethod
    def complete(future, outcome):
        if isinstance(outcome, BaseException):
            future.set_exception(outcome)
        else:
            future.set_result(outcome)

    def submit(self, func, *args):
        if self.submit_error and len(self.futures) == 2:
            raise self.submit_error     # pylint: disable=raising-bad-type
        outcome = self.outcomes[len(self.futures)]
        future = Future()
        self.futures.append(future)
        if isinstance(outcome, tuple):
            future.set_running_or_notify_cancel()
            timer = threading.Timer(0.05, self.complete, (future, outcome[1]))
            self.timers.append(timer)
            timer.start()
        elif outcome is not None:
            self.complete(future, outcome)
        return future

    def shutdown(self, wait=True, cancel_futures=False):
        self.shutdown_calls.append((wait, cancel_futures))
        if self.shutdown_errors:
            # -- LIKE: Interrupted while joining (processes are still known).
            raise self.shutdown_errors.pop(0)
        # -- LIKE: ProcessPoolExecutor.shutdown() forgets its processes.
        self._processes = None
        for timer in self.timers:
            timer.join()

    @property
    def cancel_event(self):
        return self.initargs[3]


def passed_result(filename, **kwargs):
    return make_result(filename, failed=False, status="passed", **kwargs)


def failed_result(filename, **kwargs):
    return make_result(filename, failed=True, status="failed", **kwargs)


class TestRunWorkItems:
    FILENAMES = ["a.feature", "b.feature", "c.feature", "d.feature",
                 "e.feature"]

    @pytest.fixture(autouse=True)
    def use_fake_executor(self, monkeypatch):
        monkeypatch.setattr(runner_parallel, "ProcessPoolExecutor",
                            FakeExecutor)
        yield
        FakeExecutor.instance = None
        FakeExecutor.outcomes = ()
        FakeExecutor.shutdown_errors = ()
        FakeExecutor.submit_error = None

    def run_work_items(self, outcomes, command_args=None):
        FakeExecutor.outcomes = outcomes
        runner = ParallelRunner(make_config(["--jobs=2"] + (command_args or [])))
        runner.context = Context(runner)
        work_items = dict((name, [name]) for name in self.FILENAMES)
        processed = set()
        failed_count = runner._run_work_items(work_items, {}, None, set(),
                                              processed)
        return runner, failed_count, processed, FakeExecutor.instance

    def test_runs_all_work_items(self):
        outcomes = [passed_result(name) for name in self.FILENAMES]
        runner, failed_count, processed, executor = \
            self.run_work_items(outcomes)
        assert failed_count == 0
        assert processed == set(self.FILENAMES)
        assert not runner.aborted
        assert not executor.cancel_event.is_set()
        assert executor.shutdown_calls == [(True, True)]

    def test_failure_without_stop_cancels_nothing(self):
        outcomes = [failed_result("a.feature")] + \
                   [passed_result(name) for name in self.FILENAMES[1:]]
        _, failed_count, processed, executor = self.run_work_items(outcomes)
        assert failed_count == 1
        assert processed == set(self.FILENAMES)
        assert not executor.cancel_event.is_set()

    def test_stop_cancels_pending_tasks_and_does_not_hang(self):
        # -- REGRESSION: executor.shutdown(wait=False, cancel_futures=True)
        # inside of the result loop did wait forever for cancelled futures.
        outcomes = [failed_result("a.feature"),
                    ("running", passed_result("b.feature")),
                    None, None, None]
        runner, failed_count, processed, executor = \
            self.run_work_items(outcomes, ["--stop"])
        assert failed_count == 1
        assert processed == {"a.feature", "b.feature"}
        assert [future.cancelled() for future in executor.futures] == \
               [False, False, True, True, True]
        assert executor.cancel_event.is_set()
        assert not runner.aborted
        # -- ONLY ONE SHUTDOWN: That waits for the worker processes. An
        # earlier shutdown(wait=False) turns a later wait=True into a no-op.
        assert executor.shutdown_calls == [(True, True)]

    def test_aborted_worker_aborts_the_testrun(self):
        # -- REGRESSION: context.abort() in a worker was not seen by the
        # parent (exit status: passed, remaining features silently not run).
        outcomes = [make_result("a.feature", failed=False, status="untested",
                                aborted=True),
                    ("running", make_result("b.feature", failed=False,
                                            aborted=True)),
                    None, None, None]
        runner, failed_count, processed, executor = \
            self.run_work_items(outcomes)
        assert runner.aborted
        assert failed_count == 0
        assert processed == {"a.feature"}
        assert executor.cancel_event.is_set()
        assert [future.cancelled() for future in executor.futures[2:]] == \
               [True, True, True]

    @pytest.mark.parametrize("params", [
        dict(fatal_error=True, error_text="ParserError: XFAIL"),
        dict(worker_setup_failed=True, error_text="SETUP FAILED"),
    ])
    def test_fatal_error_aborts_the_testrun(self, params):
        outcomes = [make_result("a.feature", **params), None, None, None, None]
        runner, failed_count, processed, executor = \
            self.run_work_items(outcomes)
        assert runner.aborted
        assert failed_count == 1
        assert processed == set()
        assert executor.cancel_event.is_set()

    def test_died_worker_aborts_the_testrun(self, capsys):
        error = BrokenProcessPool("XFAIL-DIED")
        outcomes = [passed_result("a.feature"), error, error, error, error]
        runner, failed_count, processed, _ = self.run_work_items(outcomes)
        assert runner.aborted
        assert failed_count == 4
        assert processed == {"a.feature"}
        assert runner._errored_filenames == set()
        # -- REPORTED ONCE: Not once per remaining feature.
        assert capsys.readouterr().err.count("PARALLEL-WORKER DIED") == 2
        # -- HINT: error_text + "ABORTED: {reason}" of Context.abort().

    def test_task_error_is_remembered_as_errored_feature(self, capsys):
        outcomes = [passed_result(name) for name in self.FILENAMES]
        outcomes[1] = RuntimeError("XFAIL-TASK")
        runner, failed_count, processed, _ = self.run_work_items(outcomes)
        assert not runner.aborted
        assert failed_count == 1
        assert processed == set(self.FILENAMES) - {"b.feature"}
        assert runner._errored_filenames == {"b.feature"}
        assert "PARALLEL-WORKER FAILURE in b.feature" in capsys.readouterr().err

    def test_feature_file_without_feature_is_processed(self):
        outcomes = [passed_result(name) for name in self.FILENAMES]
        outcomes[0] = make_result("a.feature", failed=False, no_feature=True)
        _, failed_count, processed, _ = self.run_work_items(outcomes)
        assert failed_count == 0
        assert processed == set(self.FILENAMES)

    def test_keyboard_interrupt_terminates_workers_before_shutdown(
            self, monkeypatch):
        # -- REGRESSION: executor.shutdown() forgets the worker processes,
        # workers were never terminated (they ran until their tasks ended).
        def raise_keyboard_interrupt(*args, **kwargs):
            raise KeyboardInterrupt()

        monkeypatch.setattr(runner_parallel, "wait", raise_keyboard_interrupt)
        runner, _, processed, executor = \
            self.run_work_items([None, None, None, None, None])
        assert runner.aborted
        assert processed == set()
        assert executor.process.terminated
        assert executor.cancel_event.is_set()
        assert all(future.cancelled() for future in executor.futures)
        assert executor.shutdown_calls == [(True, True)]

    def test_keyboard_interrupt_while_tasks_are_submitted(self):
        FakeExecutor.submit_error = KeyboardInterrupt()
        runner, _, processed, executor = \
            self.run_work_items([None, None, None, None, None])
        assert runner.aborted
        assert processed == set()
        assert executor.process.terminated
        assert all(future.cancelled() for future in executor.futures)

    def test_keyboard_interrupt_while_workers_shut_down(self):
        # -- HINT: Worker shutdown-hooks may run for a long time.
        FakeExecutor.shutdown_errors = [KeyboardInterrupt()]
        outcomes = [passed_result(name) for name in self.FILENAMES]
        runner, _, processed, executor = self.run_work_items(outcomes)
        assert runner.aborted
        assert processed == set(self.FILENAMES)
        assert executor.process.terminated
        assert executor.shutdown_calls == [(True, True), (True, True)]

    def test_interrupted_result_processing_does_not_report_feature_twice(
            self, monkeypatch):
        # -- HINT: A feature that is not in "processed" is reported as
        # untested. Must not occur if its counts may be merged already.
        def raise_keyboard_interrupt(self, *args):
            raise KeyboardInterrupt()

        monkeypatch.setattr(ParallelRunner, "_process_result",
                            raise_keyboard_interrupt)
        outcomes = [passed_result("a.feature"), None, None, None, None]
        runner, _, processed, _ = self.run_work_items(outcomes)
        assert runner.aborted
        assert processed == {"a.feature"}


class TestReportUntestedFeatures:
    FEATURE_TEXT = u"""
        Feature: {name}
          Scenario: {name}
            Given a step passes
        """

    class FeatureCollector:
        def __init__(self):
            self.features = []

        def feature(self, feature):
            self.features.append(feature)

    def make_runner_and_work_items(self, tmp_path, names):
        work_items = {}
        for name in names:
            feature_file = tmp_path / ("%s.feature" % name)
            feature_file.write_text(self.FEATURE_TEXT.format(name=name))
            work_items[str(feature_file)] = [str(feature_file)]
        runner = ParallelRunner(make_config())
        collector = self.FeatureCollector()
        runner.config.reporters = [collector]
        return runner, work_items, collector

    def test_features_that_did_not_run_are_untested(self, tmp_path):
        runner, work_items, collector = \
            self.make_runner_and_work_items(tmp_path, ["a", "b"])
        processed = set(list(work_items)[:1])
        runner._report_untested_features(work_items, processed)
        assert [f.name for f in collector.features] == ["b"]
        assert collector.features[0].status == Status.untested

    def test_feature_with_task_error_is_errored_not_untested(self, tmp_path):
        runner, work_items, collector = \
            self.make_runner_and_work_items(tmp_path, ["a", "b"])
        runner._errored_filenames.add(list(work_items)[0])
        runner._report_untested_features(work_items, set())
        statuses = dict((f.name, f.status) for f in collector.features)
        assert statuses == {"a": Status.error, "b": Status.untested}


# -----------------------------------------------------------------------------
# WORKER TASK: Runs in this process with a worker runtime (process-free).
# -----------------------------------------------------------------------------
class TestRunFeatureTask:
    @pytest.fixture
    def worker(self, monkeypatch):
        config = make_config()
        config.format = ["plain"]
        config.reporters = []
        runner = WorkerRunner(config)
        runner.context = Context(runner)
        from behave.runner import the_step_registry
        runner.step_registry = the_step_registry
        cancel_event = threading.Event()
        monkeypatch.setattr(runner_parallel, "_worker_runner", runner)
        monkeypatch.setattr(runner_parallel, "_worker_output", WorkerOutput())
        monkeypatch.setattr(runner_parallel, "_worker_id", 0)
        monkeypatch.setattr(runner_parallel, "_worker_setup_failed", False)
        monkeypatch.setattr(runner_parallel, "_worker_cancel_event",
                            cancel_event)
        return SimpleNamespace(runner=runner, cancel_event=cancel_event)

    @staticmethod
    def make_feature_file(tmp_path, text):
        feature_file = tmp_path / "some.feature"
        feature_file.write_text(text)
        return str(feature_file)

    def test_feature_file_without_feature_is_no_failure(self, worker, tmp_path):
        # -- LIKE SEQUENTIAL MODE: parse_features() skips such a file.
        filename = self.make_feature_file(tmp_path, u"# -- EMPTY\n")
        result = _run_feature_task([filename])
        assert result["no_feature"] is True
        assert result["failed"] is False
        assert result["status"] is None
        assert result["error_text"] is None

    def test_cancelled_testrun_does_not_run_feature(self, worker, tmp_path):
        filename = self.make_feature_file(tmp_path, u"THIS IS NOT GHERKIN\n")
        worker.cancel_event.set()
        result = _run_feature_task([filename])  # -- NOT PARSED, NOT RUN.
        assert result["status"] is None
        assert result["failed"] is False
        assert result["aborted"] is False
        assert result["fatal_error"] is False

    def test_aborted_worker_does_not_run_feature(self, worker, tmp_path):
        filename = self.make_feature_file(tmp_path, u"THIS IS NOT GHERKIN\n")
        worker.runner.aborted = True
        result = _run_feature_task([filename])  # -- NOT PARSED, NOT RUN.
        assert result["status"] is None
        assert result["failed"] is False
        assert result["aborted"] is True

    def test_abort_while_feature_runs_is_reported(self, worker, tmp_path,
                                                   monkeypatch):
        filename = self.make_feature_file(
            tmp_path, u"Feature: F\n  Scenario: S\n    Given an unknown step\n")

        class AbortingFormatter:
            def __init__(self, runner):
                self.runner = runner

            def uri(self, uri):
                self.runner.abort(reason="XFAIL-ABORT")

            def __getattr__(self, name):
                return lambda *args, **kwargs: None

        monkeypatch.setattr(
            runner_parallel, "make_formatters",
            lambda config, openers: [AbortingFormatter(worker.runner)])
        result = _run_feature_task([filename])
        assert result["aborted"] is True
        assert result["status"] is not None

    def test_parse_error_is_a_fatal_error(self, worker, tmp_path):
        filename = self.make_feature_file(
            tmp_path, u"Feature: F\n  Scenario: S\n    Given a step\n"
                      u"  NOT GHERKIN\n")
        result = _run_feature_task([filename])
        assert result["fatal_error"] is True
        assert result["failed"] is True
        assert "ParserError" in result["error_text"]

    def test_worker_error_reports_feature_as_errored(self, worker, tmp_path,
                                                     monkeypatch):
        # -- REGRESSION: Feature was reported as untested (not as errored).
        filename = self.make_feature_file(
            tmp_path, u"Feature: F\n  Scenario: S\n    Given an unknown step\n")

        class BadFormatter:
            def close(self):
                raise RuntimeError("XFAIL-FORMATTER")

            def __getattr__(self, name):
                return lambda *args, **kwargs: None

        class FeatureCollector:
            features = []

            def feature(self, feature):
                self.features.append(feature)

        collector = FeatureCollector()
        worker.runner.config.reporters = [collector]
        monkeypatch.setattr(runner_parallel, "make_formatters",
                            lambda config, openers: [BadFormatter()])
        result = _run_feature_task([filename])
        assert result["failed"] is True
        assert result["status"] == "error"
        assert result["feature_summary"]["error"] == 1
        assert "XFAIL-FORMATTER" in result["error_text"]
        assert [f.status for f in collector.features] == [Status.error]

