# -*- coding: UTF-8 -*-
"""
Unit tests for :mod:`behave_parallel_runner.runner` (parallel test runner).

Covers the process-free parts: work-item building, format resolution,
result merging, hook validation, runner-alias wiring and configuration
support.
"""

import io
import json
import re
import threading
from collections import deque
from types import SimpleNamespace

import pytest

from behave.configuration import Configuration, DEFAULT_RUNNER_CLASS_NAME
from behave.exception import ConfigError
from behave.formatter import _registry as formatter_registry_module
from behave.formatter._builtins import _BUILTIN_FORMATS
from behave.formatter.base import Formatter
from behave.formatter.json import JSONFormatter
from behave.formatter.rerun import RerunFormatter
from behave.model_type import FileLocation, Status
from behave.runner import Context
from behave_parallel_runner import runner as runner_parallel
from behave_parallel_runner.runner import (
    JsonOutputMerger,
    ParallelRunner,
    ScenarioInfo,
    TaskSchedule,
    TextOutputMerger,
    UndefinedStepInfo,
    WorkerFormat,
    WorkerOutput,
    WorkerProcess,
    WorkerRunner,
    _run_feature_task,
    group_locations_by_filename,
    is_serial_feature,
    make_result,
    merge_status_counts,
    needs_complete_testrun,
    resolve_worker_formats,
    select_complete_testrun_formatter_classes,
    select_outfile_bound_formats,
    select_output_merger_name,
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
    "rerun",
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


class MyRerunFormatter(RerunFormatter):
    """User-defined formatter that derives from an aggregating one."""
    name = "my.rerun"
    description = "Rerun formatter (user-defined variant)."


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


def console(name):
    return WorkerFormat(name, None, "text")


class TestResolveWorkerFormats:
    def test_pretty_is_replaced_by_plain(self):
        worker_formats, notes = resolve_worker_formats(["pretty"])
        assert worker_formats == [console("plain")]
        assert any("plain" in note and "pretty" in note for note in notes)

    def test_plain_and_progress_pass_through(self):
        worker_formats, notes = resolve_worker_formats(["plain", "progress"])
        assert worker_formats == [console("plain"), console("progress")]
        assert not notes

    @pytest.mark.parametrize("format_name", AGGREGATING_BUILTIN_FORMATS)
    def test_unsupported_format_is_rejected(self, format_name):
        # -- FINDING 4: Must not silently skip (no output is data-loss).
        with pytest.raises(ConfigError, match=re.escape(format_name)):
            resolve_worker_formats([format_name, "plain"])

    def test_unsupported_format_with_outfile_is_rejected(self):
        with pytest.raises(ConfigError, match="rerun"):
            resolve_worker_formats(["rerun"], outfile_bound=[True])

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

    def test_derived_aggregating_user_formatter_is_rejected(
            self, formatter_registry):
        # -- FINDING 10: The flag is inherited from the built-in base class.
        formatter_registry.register_as("my.rerun", MyRerunFormatter)
        with pytest.raises(ConfigError, match=re.escape("my.rerun")):
            resolve_worker_formats(["my.rerun"])

    def test_parallel_safe_formatter_overriding_builtin_name_is_accepted(
            self, formatter_registry):
        # -- FINDING 10: "steps" is only a name -- the registered class counts.
        formatter_registry.register_as("steps", ParallelSafeUserFormatter)
        worker_formats, notes = resolve_worker_formats(["steps"])
        assert worker_formats == [console("steps")]
        assert not notes

    def test_formatter_without_the_attribute_is_accepted(self,
                                                         formatter_registry):
        # -- FINDING 10: A formatter needs not inherit from Formatter.
        # HINT: register_as() requires a Formatter subclass -- bypass it here.
        formatter_registry._formatter_registry["my.duck_typed"] = \
            DuckTypedFormatter
        worker_formats, _notes = resolve_worker_formats(["my.duck_typed"])
        assert worker_formats == [console("my.duck_typed")]

    def test_unknown_format_is_rejected(self, formatter_registry):
        # -- FINDING 10: Do not mask an unknown/unloadable formatter.
        with pytest.raises(ConfigError, match=re.escape("__unknown__")):
            resolve_worker_formats(["__unknown__"])

    def test_outfile_bound_format_gets_an_own_output(self):
        # -- HINT: The parent merges its chunks into config.outputs[0].
        worker_formats, _notes = resolve_worker_formats(
            ["plain"], outfile_bound=[True])
        assert worker_formats == [WorkerFormat("plain", 0, "text")]

    def test_outfile_bound_check_is_positional(self):
        # -- FINDING 5: format[i] is paired with outputs[i].
        worker_formats, _notes = resolve_worker_formats(
            ["plain", "progress"], outfile_bound=[False, True])
        assert worker_formats == [console("plain"),
                                  WorkerFormat("progress", 1, "text")]

    def test_console_bound_formats_are_accepted(self):
        worker_formats, _notes = resolve_worker_formats(
            ["plain", "progress"], outfile_bound=[False, False])
        assert worker_formats == [console("plain"), console("progress")]

    @pytest.mark.parametrize("format_name", ["json", "json.pretty"])
    @pytest.mark.parametrize("outfile_bound", [True, False])
    def test_json_format_is_merged_as_json(self, format_name, outfile_bound):
        # -- HINT: On the console, too (must stay one valid JSON array).
        worker_formats, _notes = resolve_worker_formats(
            ["plain", format_name], outfile_bound=[False, outfile_bound])
        assert worker_formats == [console("plain"),
                                  WorkerFormat(format_name, 1, "json")]

    def test_derived_json_user_formatter_is_merged_as_json(
            self, formatter_registry):
        formatter_registry.register_as("my.json", MyJSONFormatter)
        worker_formats, _notes = resolve_worker_formats(["my.json"])
        assert worker_formats == [WorkerFormat("my.json", 0, "json")]

    def test_same_format_with_two_outfiles_is_kept_twice(self):
        worker_formats, _notes = resolve_worker_formats(
            ["plain", "plain"], outfile_bound=[True, True])
        assert worker_formats == [WorkerFormat("plain", 0, "text"),
                                  WorkerFormat("plain", 1, "text")]

    def test_empty_result_falls_back_to_plain(self):
        worker_formats, _notes = resolve_worker_formats([])
        assert worker_formats == [console("plain")]

    def test_duplicates_are_removed(self):
        worker_formats, _notes = resolve_worker_formats(["pretty", "plain"])
        assert worker_formats == [console("plain")]


class DirectoryUserFormatter(Formatter):
    """User-defined formatter that writes own files into a directory."""
    name = "my.directory"
    description = "Uses its outfile as directory name."
    parallel_outfile = "direct"


class MergeableUserFormatter(Formatter):
    name = "my.stream"
    description = "Writes to its output stream."
    parallel_outfile = "merge"


class TestSelectOutfileMode:
    @pytest.mark.parametrize("format_name", ["plain", "progress", "json"])
    def test_builtin_formatters_are_merged(self, format_name):
        formatter_class = formatter_registry_module.select_formatter_class(
            format_name)
        assert runner_parallel.select_outfile_mode(formatter_class) == "merge"

    def test_formatter_derived_from_builtin_one_is_merged(self):
        assert runner_parallel.select_outfile_mode(MyJSONFormatter) == "merge"

    def test_other_formatter_is_unknown(self):
        # -- REGRESSION: A formatter that uses its outfile as directory
        # (like: allure-behave) got a text buffer instead (and crashed).
        select_outfile_mode = runner_parallel.select_outfile_mode
        assert select_outfile_mode(ParallelSafeUserFormatter) is None
        assert select_outfile_mode(DuckTypedFormatter) is None
        assert select_outfile_mode(lambda stream, config: None) is None

    def test_formatter_can_state_its_mode(self):
        select_outfile_mode = runner_parallel.select_outfile_mode
        assert select_outfile_mode(DirectoryUserFormatter) == "direct"
        assert select_outfile_mode(MergeableUserFormatter) == "merge"

    def test_stated_mode_wins_over_builtin_base_class(self):
        class DirectJSONFormatter(JSONFormatter):
            parallel_outfile = "direct"

        select_outfile_mode = runner_parallel.select_outfile_mode
        assert select_outfile_mode(DirectJSONFormatter) == "direct"


class TestResolveWorkerFormatsWithUserFormatterAndOutfile:
    def test_unknown_outfile_mode_is_rejected(self, formatter_registry):
        formatter_registry.register_as("my.steps", ParallelSafeUserFormatter)
        with pytest.raises(ConfigError, match="parallel_outfile"):
            resolve_worker_formats(["my.steps"], outfile_bound=[True])

    def test_unknown_outfile_mode_is_accepted_on_console(self,
                                                          formatter_registry):
        formatter_registry.register_as("my.steps", ParallelSafeUserFormatter)
        worker_formats, _notes = resolve_worker_formats(["my.steps"])
        assert worker_formats == [console("my.steps")]

    def test_invalid_outfile_mode_is_rejected(self, formatter_registry):
        class BadFormatter(Formatter):
            parallel_outfile = "__invalid__"

        formatter_registry.register_as("my.bad", BadFormatter)
        with pytest.raises(ConfigError, match="__invalid__"):
            resolve_worker_formats(["my.bad"], outfile_bound=[True])

    def test_direct_formatter_gets_the_real_outfile(self, formatter_registry):
        formatter_registry.register_as("my.directory", DirectoryUserFormatter)
        worker_formats, _notes = resolve_worker_formats(
            ["plain", "my.directory"], outfile_bound=[False, True])
        assert worker_formats == [console("plain"),
                                  WorkerFormat("my.directory", 1, "direct")]

    def test_mergeable_formatter_is_merged(self, formatter_registry):
        formatter_registry.register_as("my.stream", MergeableUserFormatter)
        worker_formats, _notes = resolve_worker_formats(
            ["my.stream"], outfile_bound=[True])
        assert worker_formats == [WorkerFormat("my.stream", 0, "text")]


class TestSelectOutputMergerName:
    def test_json_formatters_use_the_json_merger(self):
        assert select_output_merger_name(JSONFormatter) == "json"
        assert select_output_merger_name(MyJSONFormatter) == "json"

    def test_other_formatters_use_the_text_merger(self):
        assert select_output_merger_name(ParallelSafeUserFormatter) == "text"
        assert select_output_merger_name(DuckTypedFormatter) == "text"
        assert select_output_merger_name(lambda stream, config: None) == "text"


class TestOutputMergers:
    def test_text_chunks_are_appended(self):
        stream = io.StringIO()
        merger = TextOutputMerger(stream)
        merger.add("Feature: A\n")
        merger.add("Feature: B\n")
        merger.close()
        assert stream.getvalue() == "Feature: A\nFeature: B\n"

    def test_json_chunks_are_merged_into_one_array(self):
        stream = io.StringIO()
        merger = JsonOutputMerger(stream)
        merger.add('[\n{"name": "A"}\n]\n')
        merger.add('[\n{"name": "B"},\n{"name": "C"}\n]\n')
        merger.close()
        assert stream.getvalue() == \
            '[\n{"name": "A"},\n{"name": "B"},\n{"name": "C"}\n]\n'
        assert [item["name"] for item in json.loads(stream.getvalue())] == \
               ["A", "B", "C"]

    def test_json_layout_of_the_formatter_is_kept(self):
        chunk = json.dumps([{"name": "A", "tags": ["x"]}], indent=2)
        stream = io.StringIO()
        merger = JsonOutputMerger(stream)
        merger.add(chunk)
        merger.close()
        assert '  {\n    "name": "A",' in stream.getvalue()
        assert json.loads(stream.getvalue()) == json.loads(chunk)

    def test_json_without_any_chunk_is_an_empty_array(self):
        stream = io.StringIO()
        JsonOutputMerger(stream).close()
        assert json.loads(stream.getvalue()) == []

    def test_empty_json_chunk_is_ignored(self):
        stream = io.StringIO()
        merger = JsonOutputMerger(stream)
        merger.add("[\n\n]\n")
        merger.add('[\n{"name": "A"}\n]\n')
        merger.close()
        assert json.loads(stream.getvalue()) == [{"name": "A"}]

    @pytest.mark.parametrize("chunk", ['[\n{"name": "A"', "", '{"name": "A"}'])
    def test_broken_json_chunk_is_ignored_with_a_warning(self, chunk, capsys):
        # -- LIKE: A feature run that ended with an error in its formatter.
        stream = io.StringIO()
        merger = JsonOutputMerger(stream)
        merger.add(chunk)
        merger.add('[\n{"name": "B"}\n]\n')
        merger.close()
        assert json.loads(stream.getvalue()) == [{"name": "B"}]
        assert "JSON output of one feature is not usable" in \
               capsys.readouterr().err


class TestNeedsCompleteTestrun:
    """Formatters that cannot run per worker are detected without any
    support of the behave core (no ``Formatter.needs_complete_testrun``).
    """

    @pytest.mark.parametrize("format_name", [
        "rerun", "steps", "steps.doc",
        "steps.catalog", "steps.usage", "tags", "tags.location",
    ])
    def test_builtin_aggregating_formatters_are_detected(self, format_name):
        formatter_class = formatter_registry_module.select_formatter_class(
            format_name)
        assert needs_complete_testrun(formatter_class) is True

    @pytest.mark.parametrize("format_name", [
        "plain", "pretty", "progress", "progress2", "progress3", "null",
        "json", "json.pretty",
    ])
    def test_builtin_streaming_formatters_are_accepted(self, format_name):
        formatter_class = formatter_registry_module.select_formatter_class(
            format_name)
        assert needs_complete_testrun(formatter_class) is False

    def test_derived_formatter_is_detected(self):
        assert needs_complete_testrun(MyRerunFormatter) is True

    def test_explicit_flag_wins_over_base_class(self):
        class ParallelSafeRerunFormatter(RerunFormatter):
            needs_complete_testrun = False

        class AggregatingJSONFormatter(JSONFormatter):
            needs_complete_testrun = True

        assert needs_complete_testrun(ParallelSafeRerunFormatter) is False
        assert needs_complete_testrun(AggregatingJSONFormatter) is True

    def test_default_of_formatter_base_class_does_not_hide_builtin(
            self, monkeypatch):
        # -- HINT: A behave version may provide this flag in its base class.
        monkeypatch.setattr(Formatter, "needs_complete_testrun", False,
                            raising=False)
        assert needs_complete_testrun(RerunFormatter) is True
        assert needs_complete_testrun(MyRerunFormatter) is True
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

    def test_workers_build_the_configuration_like_the_parent(self):
        # -- HINT: The configuration remembers how it was built (behave v1.4.0).
        config = Configuration(["--jobs=2", "features"], load_config=False,
                               tags="@one")
        worker_setup = ParallelRunner(config).make_worker_setup(["plain"])
        assert worker_setup["command_args"] == ["--jobs=2", "features"]
        assert worker_setup["config_kwargs"] == {"tags": "@one"}
        assert worker_setup["load_config"] is False

    def test_empty_command_args_are_kept(self):
        # -- HINT: An empty list must not be replaced by sys.argv.
        worker_setup = ParallelRunner(make_config()).make_worker_setup(["plain"])
        assert worker_setup["command_args"] == []

    def test_behave_without_this_support_is_rejected(self):
        # -- HINT: An older development version has the same version number.
        runner = ParallelRunner(make_config(["--jobs=2"]))
        del runner.config.command_args
        with pytest.raises(ConfigError, match="needs behave >= 1.4.0"):
            runner.make_worker_setup(["plain"])

    def test_non_picklable_config_kwargs_are_reported(self, capsys):
        runner = ParallelRunner(make_config(my_param=lambda: None))
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
# SERIAL FEATURES: Tag "@serial"
# -----------------------------------------------------------------------------
class TestIsSerialFeature:
    @staticmethod
    def parse(text):
        from behave.parser import parse_feature
        return parse_feature(text)

    def test_feature_without_the_tag_is_not_serial(self):
        feature = self.parse(u"@other\nFeature: F\n  Scenario: S\n    Given a step\n")
        assert is_serial_feature(feature, make_config()) is False

    @pytest.mark.parametrize("text", [
        u"@serial\nFeature: F\n  Scenario: S\n    Given a step\n",
        u"Feature: F\n  Scenario: S1\n    Given a step\n"
        u"  @serial\n  Scenario: S2\n    Given a step\n",
        u"Feature: F\n  @serial\n  Rule: R\n    Scenario: S\n      Given a step\n",
        u"Feature: F\n  @serial\n  Scenario Outline: S <x>\n    Given a step\n"
        u"    Examples:\n      | x |\n      | 1 |\n",
        u"Feature: F\n  Scenario Outline: S <x>\n    Given a step\n"
        u"    @serial\n    Examples:\n      | x |\n      | 1 |\n",
    ], ids=["feature", "scenario", "rule", "outline", "examples"])
    def test_tag_is_detected(self, text):
        assert is_serial_feature(self.parse(text), make_config()) is True

    def test_tag_of_scenario_that_should_not_run_is_ignored(self):
        feature = self.parse(
            u"Feature: F\n  Scenario: S1\n    Given a step\n"
            u"  @serial @slow\n  Scenario: S2\n    Given a step\n")
        assert is_serial_feature(feature, make_config(["--tags=not @slow"])) is False
        assert is_serial_feature(feature, make_config(["--tags=@slow"])) is True

    def test_feature_without_scenarios_is_not_serial(self):
        feature = self.parse(u"@serial\nFeature: F\n")
        assert is_serial_feature(feature, make_config()) is False


# -----------------------------------------------------------------------------
# TASK SCHEDULE:
# -----------------------------------------------------------------------------
class TestTaskSchedule:
    @staticmethod
    def make_schedule(names="abc"):
        return TaskSchedule(dict((name, [name + ":1"]) for name in names))

    def test_hands_out_tasks_in_order(self):
        schedule = self.make_schedule("ab")
        assert schedule.next_task() == ("a", ["a:1"], False)
        assert schedule.next_task() == ("b", ["b:1"], False)
        assert schedule.next_task() is None
        assert schedule.running == 2

    def test_serial_task_waits_until_nothing_else_runs(self):
        schedule = self.make_schedule("ab")
        schedule.next_task()
        schedule.next_task()
        schedule.task_done()
        schedule.defer_as_serial("a")
        assert schedule.has_tasks()
        assert schedule.next_task() is None     # -- "b" is still running.
        schedule.task_done()
        assert schedule.next_task() == ("a", ["a:1"], True)

    def test_serial_tasks_run_one_by_one(self):
        schedule = self.make_schedule("ab")
        for name in "ab":
            schedule.next_task()
            schedule.task_done()
            schedule.defer_as_serial(name)
        assert schedule.next_task() == ("a", ["a:1"], True)
        assert schedule.next_task() is None
        schedule.task_done()
        assert schedule.next_task() == ("b", ["b:1"], True)

    def test_other_tasks_run_before_serial_tasks(self):
        schedule = self.make_schedule("ab")
        schedule.next_task()
        schedule.task_done()
        schedule.defer_as_serial("a")
        assert schedule.next_task() == ("b", ["b:1"], False)

    def test_task_that_is_put_back_runs_next(self):
        schedule = self.make_schedule("abc")
        task = schedule.next_task()
        schedule.put_back(task)
        assert schedule.running == 0
        assert schedule.next_task() == ("a", ["a:1"], False)

    def test_serial_task_that_is_put_back_stays_serial(self):
        schedule = self.make_schedule("a")
        schedule.next_task()
        schedule.task_done()
        schedule.defer_as_serial("a")
        task = schedule.next_task()
        schedule.put_back(task)
        assert schedule.running == 0
        assert schedule.next_task() == ("a", ["a:1"], True)

    def test_cancelled_schedule_hands_out_nothing(self):
        schedule = self.make_schedule("ab")
        schedule.cancel()
        assert schedule.next_task() is None
        assert not schedule.has_tasks()


# -----------------------------------------------------------------------------
# WORKER PROCESS HANDLE: Low-level part (with a fake process and connection).
# -----------------------------------------------------------------------------
class FakeConnection:
    def __init__(self, messages=(), send_error=None):
        self.messages = deque(messages)
        self.send_error = send_error
        self.closed = False

    def poll(self):
        return bool(self.messages)

    def recv(self):
        if not self.messages:
            raise EOFError()
        message = self.messages.popleft()
        if isinstance(message, Exception):
            raise message
        return message

    def send(self, message):
        if self.send_error:
            raise self.send_error      # pylint: disable=raising-bad-type

    def close(self):
        self.closed = True


class FakeOsProcess:
    sentinel = "SENTINEL"

    def __init__(self, alive=True, dies_on="terminate"):
        self.alive = alive
        self.dies_on = dies_on
        self.calls = []
        self.exitcode = None

    def is_alive(self):
        return self.alive

    def _signal(self, name):
        self.calls.append(name)
        if name == self.dies_on or name == "kill":
            self.alive = False
            self.exitcode = -9 if name == "kill" else -15

    def terminate(self):
        self._signal("terminate")

    def kill(self):
        self._signal("kill")

    def join(self, timeout=None):
        self.calls.append(("join", timeout))


class TestWorkerProcess:
    @staticmethod
    def make_worker(process, connection):
        class ThisWorkerProcess(WorkerProcess):
            def start(self, mp_context, worker_setup, shutdown_failures):
                self.process = process
                self.connection = connection

        return ThisWorkerProcess(0, None, {}, None)

    def test_waits_for_message_and_for_end_of_process(self):
        # -- REGRESSION: A child process that a step has forked keeps the
        # connection open, the end of the worker process was not seen.
        connection = FakeConnection()
        worker = self.make_worker(FakeOsProcess(), connection)
        assert worker.wait_objects == (connection, "SENTINEL")

    def test_ended_process_without_message_is_seen_without_eof(self):
        class OpenConnection(FakeConnection):
            def recv(self):
                raise AssertionError("BLOCKS: Connection is still open")

        worker = self.make_worker(FakeOsProcess(alive=False), OpenConnection())
        assert worker.receive() is None

    def test_messages_of_ended_process_are_received_first(self):
        connection = FakeConnection([("result", {"filename": "a.feature"})])
        worker = self.make_worker(FakeOsProcess(alive=False), connection)
        assert worker.receive() == ("result", {"filename": "a.feature"})
        assert worker.receive() is None

    def test_end_of_connection_is_end_of_process(self):
        worker = self.make_worker(FakeOsProcess(), FakeConnection())
        assert worker.receive() is None

    def test_message_that_cannot_be_unpickled_is_an_error(self):
        connection = FakeConnection([AttributeError("XFAIL-UNPICKLE")])
        worker = self.make_worker(FakeOsProcess(), connection)
        kind, payload = worker.receive()
        assert kind == "error"
        assert "XFAIL-UNPICKLE" in payload

    def test_task_that_cannot_be_sent_is_not_the_task_of_the_worker(self):
        connection = FakeConnection(send_error=BrokenPipeError())
        worker = self.make_worker(FakeOsProcess(), connection)
        assert worker.run("a.feature", ["a.feature"], False) is False
        assert worker.task is None

    def test_task_that_was_sent_is_the_task_of_the_worker(self):
        worker = self.make_worker(FakeOsProcess(), FakeConnection())
        assert worker.run("a.feature", ["a.feature"], False) is True
        assert worker.task == "a.feature"
        assert not worker.idle

    def test_close_kills_process_that_survives_terminate(self):
        # -- REGRESSION: Parent waited forever (worker handles SIGTERM).
        process = FakeOsProcess(dies_on="kill")
        connection = FakeConnection()
        worker = self.make_worker(process, connection)
        worker.terminate()
        worker.close(timeout=0.1)
        assert process.calls == ["terminate", ("join", 0.1), "kill",
                                 ("join", None)]
        assert connection.closed

    def test_close_does_not_kill_process_that_has_ended(self):
        process = FakeOsProcess()
        worker = self.make_worker(process, FakeConnection())
        worker.terminate()
        worker.close(timeout=0.1)
        assert "kill" not in process.calls


class TestWaitForWorkers:
    class Worker:
        def __init__(self, ended=False):
            import multiprocessing
            self.connection, self.other_end = multiprocessing.Pipe()
            self.ended = ended

        @property
        def wait_objects(self):
            return (self.connection,)

        def has_ended(self):
            return self.ended

    def test_worker_with_message_is_ready(self):
        worker1, worker2 = self.Worker(), self.Worker()
        worker2.other_end.send("hello")
        assert runner_parallel.wait_for_workers([worker1, worker2]) == [worker2]

    def test_ended_worker_is_ready_without_any_event(self):
        # -- REGRESSION: A child process that a step has forked keeps the
        # connection (and the sentinel) of its died worker open.
        worker1, worker2 = self.Worker(), self.Worker(ended=True)
        ready = runner_parallel.wait_for_workers([worker1, worker2],
                                                 poll_interval=0.01)
        assert ready == [worker2]

    def test_ended_worker_with_message_is_ready_once(self):
        worker = self.Worker(ended=True)
        worker.other_end.send("hello")
        assert runner_parallel.wait_for_workers([worker]) == [worker]

    def test_waits_until_a_worker_is_ready(self):
        worker = self.Worker()
        timer = threading.Timer(0.1, setattr, (worker, "ended", True))
        timer.start()
        ready = runner_parallel.wait_for_workers([worker], poll_interval=0.01)
        timer.join()
        assert ready == [worker]


# -----------------------------------------------------------------------------
# PARENT RUN LOOP: With fake worker processes (process-free).
# -----------------------------------------------------------------------------
class FakeWorker(WorkerProcess):
    """Fake worker process: outcomes[filename] describes its task.

    * result dict: Result of the task.
    * function(serial_phase): Provides the outcome.
    * tuple: Message that is sent instead of a result.
    * "die": Worker process ends while it runs the task.
    * "dead": Worker process has ended before it got the task.
    """
    outcomes = {}
    instances = []
    start_messages = ()
    start_error = None
    exitcode = 0

    def start(self, mp_context, worker_setup, shutdown_failures):
        if self.start_error and len(self.instances) == 1:
            raise self.start_error      # pylint: disable=raising-bad-type
        self.instances.append(self)
        self.inbox = deque(self.start_messages or [
            ("ready", {"setup_failed": False, "init_hook_failures": 0})])
        self.tasks = []
        self.dead = False
        self.terminated = False
        self.closed = False

    def send(self, message):
        assert message is None, "REQUIRE: Tasks are sent with run()"
        self.inbox.append(None)
        return True

    def run(self, filename, locations, serial_phase):
        outcome = self.outcomes[filename]
        if callable(outcome):
            outcome = outcome(serial_phase)
        if self.dead:
            return False
        if outcome == "dead":
            if not callable(self.outcomes[filename]):
                # -- ONLY ONCE: Another worker gets this task.
                self.outcomes[filename] = passed_result(filename)
            self.dead = True
            self.exitcode = -9
            self.inbox.append(None)
            return False

        busy_others = [worker.task for worker in self.instances
                       if worker is not self and worker.task is not None
                       and not worker.closed]
        self.task = filename
        self.tasks.append((self.task, serial_phase, busy_others))
        if outcome == "die":
            self.exitcode = 3
            self.inbox.append(None)
        elif isinstance(outcome, tuple):
            self.inbox.append(outcome)
        else:
            self.inbox.append(("result", outcome))
        return True

    def receive(self):
        return self.inbox.popleft()

    def close(self, timeout=None):
        self.closed = True
        self.close_timeout = timeout
        return self.exitcode

    def terminate(self):
        self.terminated = True

    @classmethod
    def all_tasks(cls):
        return [task for worker in cls.instances for task in worker.tasks]


def passed_result(filename, **kwargs):
    return make_result(filename, failed=False, status="passed", **kwargs)


def failed_result(filename, **kwargs):
    return make_result(filename, failed=True, status="failed", **kwargs)


def serial_outcome(filename):
    """Outcome of a "@serial" feature: Handed back unless nothing else runs."""
    def select_outcome(serial_phase):
        if serial_phase:
            return passed_result(filename)
        return make_result(filename, failed=False, serial_deferred=True)
    return select_outcome


class TestRunWorkItems:
    FILENAMES = ["a.feature", "b.feature", "c.feature", "d.feature",
                 "e.feature"]

    @pytest.fixture(autouse=True)
    def use_fake_workers(self, monkeypatch):
        def fake_wait_for_workers(workers):
            if self.interrupt_when(workers):
                raise KeyboardInterrupt()
            ready = [worker for worker in workers if worker.inbox]
            assert ready, "DEADLOCK: No worker has a message"
            return ready

        self.interrupt_when = lambda workers: False
        monkeypatch.setattr(runner_parallel, "WorkerProcess", FakeWorker)
        monkeypatch.setattr(runner_parallel, "wait_for_workers",
                            fake_wait_for_workers)
        monkeypatch.setattr(FakeWorker, "instances", [])
        monkeypatch.setattr(FakeWorker, "outcomes", {})

    def run_work_items(self, outcomes=None, command_args=None):
        all_outcomes = dict((name, passed_result(name))
                            for name in self.FILENAMES)
        all_outcomes.update(outcomes or {})
        FakeWorker.outcomes = all_outcomes
        runner = ParallelRunner(make_config(["--jobs=2"] + (command_args or [])))
        runner.context = Context(runner)
        work_items = dict((name, [name]) for name in self.FILENAMES)
        processed = set()
        failed_count = runner._run_work_items(work_items, {}, None, set(),
                                              processed)
        return runner, failed_count, processed

    @staticmethod
    def assert_all_workers_are_closed():
        assert all(worker.closed for worker in FakeWorker.instances)

    def test_runs_all_work_items(self):
        runner, failed_count, processed = self.run_work_items()
        assert failed_count == 0
        assert processed == set(self.FILENAMES)
        assert not runner.aborted
        assert [worker.worker_id for worker in FakeWorker.instances] == [0, 1]
        assert all(worker.stopping for worker in FakeWorker.instances)
        assert not any(worker.terminated for worker in FakeWorker.instances)
        self.assert_all_workers_are_closed()

    def test_starts_no_more_workers_than_work_items(self, monkeypatch):
        monkeypatch.setattr(self, "FILENAMES", ["a.feature"])
        self.run_work_items(command_args=["--jobs=4"])
        assert len(FakeWorker.instances) == 1

    def test_each_feature_runs_once(self):
        self.run_work_items()
        names = [name for name, _, _ in FakeWorker.all_tasks()]
        assert sorted(names) == self.FILENAMES

    def test_failure_without_stop_cancels_nothing(self):
        _, failed_count, processed = self.run_work_items(
            {"a.feature": failed_result("a.feature")})
        assert failed_count == 1
        assert processed == set(self.FILENAMES)

    def test_stop_hands_out_no_further_tasks(self):
        # -- HINT: "b.feature" is already in-flight (it finishes).
        runner, failed_count, processed = self.run_work_items(
            {"a.feature": failed_result("a.feature")}, ["--stop"])
        assert failed_count == 1
        assert processed == {"a.feature", "b.feature"}
        assert not runner.aborted
        self.assert_all_workers_are_closed()

    def test_aborted_worker_aborts_the_testrun(self):
        # -- REGRESSION: context.abort() in a worker was not seen by the
        # parent (exit status: passed, remaining features silently not run).
        runner, failed_count, processed = self.run_work_items({
            "a.feature": make_result("a.feature", failed=False,
                                     status="untested", aborted=True),
            "b.feature": make_result("b.feature", failed=False, aborted=True),
        })
        assert runner.aborted
        assert failed_count == 0
        assert processed == {"a.feature"}
        self.assert_all_workers_are_closed()

    @pytest.mark.parametrize("params", [
        dict(fatal_error=True, error_text="ParserError: XFAIL"),
        dict(worker_setup_failed=True, error_text="SETUP FAILED"),
    ])
    def test_fatal_error_aborts_the_testrun(self, params):
        runner, failed_count, processed = self.run_work_items(
            {"a.feature": make_result("a.feature", **params)})
        assert runner.aborted
        assert failed_count == 1
        assert processed == {"b.feature"}   # -- HINT: Was in-flight.

    def test_failed_worker_setup_aborts_the_testrun(self, monkeypatch):
        # -- HINT: Reported when the worker is ready, it may never get a task.
        monkeypatch.setattr(FakeWorker, "start_messages", [
            ("ready", {"setup_failed": True, "init_hook_failures": 1})])
        runner, _, processed = self.run_work_items()
        assert runner.aborted
        assert processed == set()
        assert FakeWorker.all_tasks() == []
        assert runner._worker_init_failures == {0: 1, 1: 1}
        self.assert_all_workers_are_closed()

    def test_died_worker_fails_only_its_feature(self, capsys):
        runner, failed_count, processed = self.run_work_items(
            {"b.feature": "die"})
        assert not runner.aborted
        assert failed_count == 1
        assert processed == set(self.FILENAMES) - {"b.feature"}
        assert list(runner._feature_errors) == ["b.feature"]
        assert ("PARALLEL-WORKER DIED in b.feature (exit code: 3)"
                in capsys.readouterr().err)
        self.assert_all_workers_are_closed()

    def test_died_worker_is_replaced_with_the_same_worker_id(self):
        self.run_work_items({"b.feature": "die"})
        assert [worker.worker_id for worker in FakeWorker.instances] == \
               [0, 1, 1]
        assert FakeWorker.instances[2].tasks, "REQUIRE: Replacement is used"

    def test_died_worker_is_not_replaced_without_remaining_tasks(self):
        self.run_work_items({"e.feature": "die"})
        assert len(FakeWorker.instances) == 2

    def test_died_worker_is_not_replaced_with_stop(self):
        _, failed_count, processed = self.run_work_items(
            {"a.feature": "die"}, ["--stop"])
        assert failed_count == 1
        assert processed == {"b.feature"}
        assert len(FakeWorker.instances) == 2

    def test_each_died_worker_fails_its_feature(self):
        outcomes = dict((name, "die") for name in self.FILENAMES)
        runner, failed_count, processed = self.run_work_items(outcomes)
        assert failed_count == 5
        assert processed == set()
        assert sorted(runner._feature_errors) == self.FILENAMES
        assert not runner.aborted

    def test_worker_that_dies_during_its_setup_aborts_the_testrun(
            self, monkeypatch):
        # -- HINT: A replacement would die the same way (no endless loop).
        monkeypatch.setattr(FakeWorker, "start_messages", [None])
        monkeypatch.setattr(FakeWorker, "exitcode", 4)
        runner, _, processed = self.run_work_items()
        assert runner.aborted
        assert processed == set()
        assert len(FakeWorker.instances) == 2
        assert runner.worker_hook_failures == 2

    def test_worker_that_died_while_idle_loses_no_feature(self, capsys):
        # -- REGRESSION: The feature that this worker should run next was
        # reported as errored, although it never ran.
        runner, failed_count, processed = self.run_work_items(
            {"c.feature": "dead"})
        assert failed_count == 0
        assert processed == set(self.FILENAMES)
        assert runner._feature_errors == {}
        assert not runner.aborted
        assert runner.worker_hook_failures == 1   # -- HINT: Test-run fails.
        assert "PARALLEL-WORKER" in capsys.readouterr().err
        names = [name for name, _, _ in FakeWorker.all_tasks()]
        assert sorted(names) == self.FILENAMES

    def test_worker_that_died_while_idle_is_replaced(self):
        self.run_work_items({"c.feature": "dead"})
        assert len(FakeWorker.instances) == 3

    def test_workers_that_die_while_idle_again_and_again_abort_the_testrun(
            self):
        def always_dead(serial_phase):
            return "dead"

        outcomes = dict((name, always_dead) for name in self.FILENAMES)
        runner, _, processed = self.run_work_items(outcomes)
        assert runner.aborted
        assert processed == set()
        assert len(FakeWorker.instances) <= 2 + 2 + 1   # -- jobs=2

    def test_worker_that_dies_on_shutdown_fails_the_testrun(
            self, monkeypatch, capsys):
        monkeypatch.setattr(FakeWorker, "exitcode", 1)
        runner, failed_count, processed = self.run_work_items()
        assert failed_count == 0
        assert processed == set(self.FILENAMES)
        assert runner.worker_hook_failures == 2
        assert "DIED on shutdown" in capsys.readouterr().err

    def test_task_error_is_remembered_as_errored_feature(self, capsys):
        runner, failed_count, processed = self.run_work_items(
            {"b.feature": ("error", "XFAIL-TASK")})
        assert not runner.aborted
        assert failed_count == 1
        assert processed == set(self.FILENAMES) - {"b.feature"}
        assert list(runner._feature_errors) == ["b.feature"]
        assert "PARALLEL-WORKER FAILURE in b.feature" in capsys.readouterr().err
        # -- HINT: This worker is still usable.
        assert len(FakeWorker.instances) == 2

    def test_feature_file_without_feature_is_processed(self):
        _, failed_count, processed = self.run_work_items({
            "a.feature": make_result("a.feature", failed=False,
                                     no_feature=True)})
        assert failed_count == 0
        assert processed == set(self.FILENAMES)

    # -- SERIAL FEATURES:
    def test_serial_features_run_while_nothing_else_runs(self):
        _, failed_count, processed = self.run_work_items({
            "a.feature": serial_outcome("a.feature"),
            "c.feature": serial_outcome("c.feature"),
        })
        assert failed_count == 0
        assert processed == set(self.FILENAMES)
        serial_tasks = [(name, busy_others)
                        for name, serial_phase, busy_others
                        in FakeWorker.all_tasks() if serial_phase]
        assert sorted(serial_tasks) == [("a.feature", []), ("c.feature", [])]
        self.assert_all_workers_are_closed()

    def test_serial_features_run_after_the_other_features(self):
        self.run_work_items({"a.feature": serial_outcome("a.feature")})
        tasks = FakeWorker.all_tasks()
        parallel_names = [name for name, serial_phase, _ in tasks
                          if not serial_phase]
        assert sorted(parallel_names) == self.FILENAMES  # -- "a": Handed back.
        order = [name for worker in FakeWorker.instances
                 for name, serial_phase, _ in worker.tasks if serial_phase]
        assert order == ["a.feature"]

    def test_only_serial_features(self):
        outcomes = dict((name, serial_outcome(name))
                        for name in self.FILENAMES)
        _, failed_count, processed = self.run_work_items(outcomes)
        assert failed_count == 0
        assert processed == set(self.FILENAMES)
        assert all(busy_others == []
                   for _, serial_phase, busy_others in FakeWorker.all_tasks()
                   if serial_phase)

    def test_serial_feature_is_untested_if_testrun_is_stopped(self):
        _, failed_count, processed = self.run_work_items({
            "a.feature": serial_outcome("a.feature"),
            "b.feature": failed_result("b.feature"),
        }, ["--stop"])
        assert failed_count == 1
        assert "a.feature" not in processed

    def test_worker_that_dies_in_serial_feature_is_replaced(self):
        def die_in_serial_phase(serial_phase):
            if serial_phase:
                return "die"
            return make_result("a.feature", failed=False, serial_deferred=True)

        runner, failed_count, processed = self.run_work_items({
            "a.feature": die_in_serial_phase,
            "b.feature": serial_outcome("b.feature"),
        })
        assert failed_count == 1
        assert list(runner._feature_errors) == ["a.feature"]
        assert processed == set(self.FILENAMES) - {"a.feature"}
        assert len(FakeWorker.instances) == 3

    # -- KEYBOARD INTERRUPT:
    def assert_hard_stopped(self, runner):
        assert runner.aborted
        assert FakeWorker.instances
        assert all(worker.terminated for worker in FakeWorker.instances
                   if not worker.closed or worker.terminated)
        self.assert_all_workers_are_closed()

    def test_keyboard_interrupt_terminates_workers(self):
        self.interrupt_when = lambda workers: True
        runner, _, processed = self.run_work_items()
        assert processed == set()
        assert all(worker.terminated for worker in FakeWorker.instances)
        # -- HINT: A worker that survives SIGTERM must not block forever.
        assert all(0 <= worker.close_timeout
                   <= runner_parallel.WORKER_TERMINATE_TIMEOUT
                   for worker in FakeWorker.instances)
        self.assert_hard_stopped(runner)

    def test_keyboard_interrupt_while_workers_are_started(self, monkeypatch):
        monkeypatch.setattr(FakeWorker, "start_error", KeyboardInterrupt())
        runner, _, processed = self.run_work_items()
        assert processed == set()
        assert len(FakeWorker.instances) == 1
        self.assert_hard_stopped(runner)

    def test_keyboard_interrupt_while_workers_shut_down(self):
        # -- HINT: Worker shutdown-hooks may run for a long time.
        self.interrupt_when = lambda workers: all(worker.stopping
                                                  for worker in workers)
        runner, _, processed = self.run_work_items()
        assert processed == set(self.FILENAMES)
        self.assert_hard_stopped(runner)

    def test_interrupted_result_processing_does_not_report_feature_twice(
            self, monkeypatch):
        # -- HINT: A feature that is not in "processed" is reported as
        # untested. Must not occur if its counts may be merged already.
        def raise_keyboard_interrupt(self, *args):
            raise KeyboardInterrupt()

        monkeypatch.setattr(ParallelRunner, "_process_result",
                            raise_keyboard_interrupt)
        runner, _, processed = self.run_work_items()
        assert processed == {"a.feature"}
        self.assert_hard_stopped(runner)

    def test_unexpected_error_leaves_no_worker_process_behind(
            self, monkeypatch):
        def raise_error(self, *args):
            raise RuntimeError("XFAIL-BUG")

        monkeypatch.setattr(ParallelRunner, "_process_result", raise_error)
        with pytest.raises(RuntimeError, match="XFAIL-BUG"):
            self.run_work_items()
        assert all(worker.terminated and worker.closed
                   for worker in FakeWorker.instances)


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
        runner._feature_errors[list(work_items)[0]] = "XFAIL-DIED"
        runner._report_untested_features(work_items, set())
        statuses = dict((f.name, f.status) for f in collector.features)
        assert statuses == {"a": Status.error, "b": Status.untested}

    def test_scenarios_of_errored_feature_are_errored(self, tmp_path):
        # -- HINT: Otherwise, the JUnit report shows them as skipped.
        runner, work_items, collector = \
            self.make_runner_and_work_items(tmp_path, ["a"])
        runner._feature_errors[list(work_items)[0]] = "XFAIL-DIED"
        runner._report_untested_features(work_items, set())
        scenario = collector.features[0].scenarios[0]
        assert scenario.status == Status.error
        assert scenario.error_message == "XFAIL-DIED"


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
        monkeypatch.setattr(runner_parallel, "_worker_runner", runner)
        monkeypatch.setattr(runner_parallel, "_worker_output", WorkerOutput())
        monkeypatch.setattr(runner_parallel, "_worker_id", 0)
        monkeypatch.setattr(runner_parallel, "_worker_setup_failed", False)
        monkeypatch.setattr(runner_parallel, "_worker_formats", None)
        return SimpleNamespace(runner=runner)

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

    SERIAL_FEATURE_TEXT = (u"@serial\nFeature: F\n  Scenario: S\n"
                           u"    Given an unknown step\n")

    def test_serial_feature_is_handed_back(self, worker, tmp_path):
        filename = self.make_feature_file(tmp_path, self.SERIAL_FEATURE_TEXT)
        result = _run_feature_task([filename])  # -- NOT RUN.
        assert result["serial_deferred"] is True
        assert result["status"] is None
        assert result["failed"] is False
        assert result["output"] == ""
        assert worker.runner.undefined_steps == []

    def test_serial_feature_runs_in_serial_phase(self, worker, tmp_path):
        filename = self.make_feature_file(tmp_path, self.SERIAL_FEATURE_TEXT)
        result = _run_feature_task([filename], serial_phase=True)
        assert result["serial_deferred"] is False
        assert result["status"] == "error"     # -- HINT: Undefined step.
        assert result["undefined_steps"] == [("given", "an unknown step")]
        assert "Feature: F" in result["output"]

    def test_formatter_with_own_output_does_not_write_to_the_console(
            self, worker, tmp_path, monkeypatch):
        monkeypatch.setattr(runner_parallel, "_worker_formats", [
            WorkerFormat("progress", None, "text"),
            WorkerFormat("plain", 1, "text"),
            WorkerFormat("json", 2, "json"),
        ])
        worker.runner.config.format = ["progress", "plain", "json"]
        filename = self.make_feature_file(
            tmp_path, u"Feature: F\n  Scenario: S\n    Given an unknown step\n")
        result = _run_feature_task([filename])
        assert sorted(result["outputs"]) == [1, 2]
        assert "Feature: F" in result["outputs"][1]
        assert "Feature: F" not in result["output"]
        assert filename in result["output"]    # -- HINT: progress formatter.
        report = json.loads(result["outputs"][2])
        assert [feature["name"] for feature in report] == ["F"]

    def test_own_outputs_start_empty_for_each_feature(self, worker, tmp_path,
                                                      monkeypatch):
        monkeypatch.setattr(runner_parallel, "_worker_formats",
                            [WorkerFormat("json", 0, "json")])
        worker.runner.config.format = ["json"]
        filename = self.make_feature_file(
            tmp_path, u"Feature: F\n  Scenario: S\n    Given an unknown step\n")
        for _ in range(2):
            result = _run_feature_task([filename])
            assert len(json.loads(result["outputs"][0])) == 1

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



# -----------------------------------------------------------------------------
# RESOURCE TRACKER: A child process that a step has forked must not keep it
# alive (Python 3.12: behave hangs when it exits, see: CPython gh-146313).
# -----------------------------------------------------------------------------
class TestResourceTrackerReleaseAfterFork:
    class TrackerWithoutDel:
        """LIKE: Python < 3.12 (nobody waits for the resource tracker)."""

    class TrackerWithBlockingDel:
        """LIKE: Python 3.12 (waits without timeout)."""
        def __del__(self):
            pass

    class TrackerWithFix(TrackerWithBlockingDel):
        """LIKE: CPython with gh-146313."""
        def _after_fork_in_child(self):
            pass

    @pytest.mark.parametrize("tracker_class, expected", [
        (TrackerWithoutDel, False),
        (TrackerWithBlockingDel, True),
        (TrackerWithFix, False),
    ])
    def test_is_only_needed_if_python_waits_without_the_fix(
            self, monkeypatch, tracker_class, expected):
        from multiprocessing import resource_tracker
        monkeypatch.setattr(resource_tracker, "ResourceTracker", tracker_class)
        assert runner_parallel.needs_resource_tracker_release_after_fork() \
            is expected

    def test_matches_this_python_version(self):
        import sys
        expected = {(3, 10): False, (3, 11): False, (3, 12): True}
        version = sys.version_info[:2]
        if version not in expected:
            pytest.skip("Depends on the patch release of this Python version")
        assert runner_parallel.needs_resource_tracker_release_after_fork() \
            is expected[version]

    @staticmethod
    def use_fake_tracker(monkeypatch, fd, pid=1234):
        from multiprocessing import resource_tracker
        tracker = SimpleNamespace(_fd=fd, _pid=pid)
        monkeypatch.setattr(resource_tracker, "_resource_tracker", tracker)
        return tracker

    def test_release_closes_the_inherited_connection(self, monkeypatch):
        import os
        read_fd, write_fd = os.pipe()
        try:
            tracker = self.use_fake_tracker(monkeypatch, write_fd)
            runner_parallel.release_resource_tracker_in_forked_child()
            assert tracker._fd is None and tracker._pid is None
            # -- CLOSED: The other end sees the end of the stream.
            assert os.read(read_fd, 1) == b""
        finally:
            os.close(read_fd)

    def test_release_without_resource_tracker_does_nothing(self, monkeypatch):
        tracker = self.use_fake_tracker(monkeypatch, None)
        runner_parallel.release_resource_tracker_in_forked_child()
        assert tracker._fd is None and tracker._pid == 1234

    def test_release_tolerates_a_closed_connection(self, monkeypatch):
        import os
        read_fd, write_fd = os.pipe()
        os.close(read_fd)
        os.close(write_fd)
        tracker = self.use_fake_tracker(monkeypatch, write_fd)
        runner_parallel.release_resource_tracker_in_forked_child()
        assert tracker._fd is None
