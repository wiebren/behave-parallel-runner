# -*- coding: UTF-8 -*-
"""
Functional tests for the parallel test runner on the command line
(ported from: "features/runner.parallel_jobs.feature" in the behave fork).

NOTES:
 * The parallel runner is selected in the config-file (see: conftest.py),
   because this package does not auto-select it with "--jobs N".
 * Work unit: one feature file per worker task.
 * Parallel mode never calls "before_all"/"after_all"; it uses
   "before_parallel"/"after_parallel" (parent process, once) and
   "before_worker"/"after_worker" (once per worker process) instead.
 * An environment file that defines "before_all" (or "after_all")
   without a matching parallel-mode hook is rejected with --jobs > 1.
"""

import json
import os
import re
import subprocess
import sys
import time

import pytest

from conftest import run_behave


PARALLEL_RUNNER_NAME = "behave_parallel_runner.runner:ParallelRunner"


def test_run_many_feature_files_in_parallel(workdir):
    """Run many feature files in parallel (case: all passing)"""
    result = run_behave(workdir,
                        "--jobs=2 -f plain --no-color --no-capture-hooks "
                        "features/alice.feature features/bob.feature "
                        "features/charly.feature features/dora.feature")
    assert result.returncode == 0, result.output
    assert "4 features passed, 0 failed, 0 skipped" in result.output
    assert "USING RUNNER: %s" % PARALLEL_RUNNER_NAME in result.output
    assert "4 scenarios passed, 0 failed, 0 skipped" in result.output
    assert "5 steps passed, 0 failed, 0 skipped" in result.output
    assert result.output.count("HOOK: BEFORE-PARALLEL jobs=2") == 1
    assert result.output.count("HOOK: AFTER-PARALLEL") == 1
    assert result.output.count("HOOK: WORKER-STARTED") == 2
    assert result.output.count("HOOK: WORKER-STOPPED") == 2
    assert "HOOK: BEFORE-ALL" not in result.output


def test_sequential_mode_is_used_with_jobs1_and_keeps_before_all(workdir):
    """Sequential mode is used with --jobs=1 and keeps before_all"""
    result = run_behave(workdir, "--jobs=1 -f plain --no-color "
                                 "features/alice.feature")
    assert result.returncode == 0, result.output
    assert "1 feature passed, 0 failed, 0 skipped" in result.output
    # -- ADAPTED: The runner is still the ParallelRunner (selected in
    # behave.ini), which falls back to sequential execution with --jobs=1.
    assert "USING RUNNER: %s" % PARALLEL_RUNNER_NAME in result.output
    assert "HOOK: BEFORE-ALL" in result.output
    assert "HOOK: WORKER-STARTED" not in result.output
    assert "HOOK: BEFORE-PARALLEL" not in result.output


def test_parallel_runner_with_one_feature_file_uses_the_parallel_mode_hooks(workdir):
    """Parallel runner with one feature file uses the parallel-mode hooks

    The number of selected feature files must not decide which hooks run.
    """
    result = run_behave(workdir, "--jobs=2 -f plain --no-color "
                                 "features/alice.feature")
    assert result.returncode == 0, result.output
    assert "1 feature passed, 0 failed, 0 skipped" in result.output
    assert "USING RUNNER: %s" % PARALLEL_RUNNER_NAME in result.output
    assert "HOOK: BEFORE-PARALLEL jobs=2" in result.output
    assert "HOOK: WORKER-STARTED" in result.output
    assert "HOOK: WORKER-STOPPED" in result.output
    assert "HOOK: BEFORE-ALL" not in result.output


def test_a_failing_feature_fails_the_parallel_test_run(workdir):
    """A failing feature fails the parallel test run"""
    workdir.write_file("features/fails.feature", u"""
        Feature: Failing
          Scenario: F1
            Given a step passes
            When a step fails
        """)
    result = run_behave(workdir, "--jobs=2 -f plain --no-color "
                                 "features/alice.feature features/bob.feature "
                                 "features/fails.feature")
    assert result.returncode != 0, result.output
    assert "2 features passed, 1 failed, 0 skipped" in result.output
    assert "Failing scenarios:" in result.output
    assert "features/fails.feature:2  F1" in result.output


def test_undefined_steps_are_reported_once_by_the_parallel_test_run(workdir):
    """Undefined steps are reported once by the parallel test run"""
    workdir.write_file("features/undefined.feature", u"""
        Feature: Undefined
          Scenario: U1
            Given an unknown step is used
        """)
    result = run_behave(workdir, "--jobs=2 -f plain --no-color "
                                 "features/alice.feature features/bob.feature "
                                 "features/undefined.feature")
    assert result.returncode != 0, result.output
    assert ("You can implement step definitions for undefined steps "
            "with these snippets:") in result.output
    # -- HINT: behave 1.3.x generates the snippet with a "u"-prefixed string,
    # newer versions without it: "@given('an unknown step is used')".
    snippets = re.findall(r"@given\(u?'an unknown step is used'\)",
                          result.output)
    assert len(snippets) == 1, result.output


def test_an_environment_file_with_only_a_before_all_hook_is_rejected(workdir):
    """An environment file with only a before_all hook is rejected in parallel mode"""
    workdir.write_file("features/environment.py", u"""
        def before_all(context):
            print("HOOK: BEFORE-ALL")
        """)
    result = run_behave(workdir, "--jobs=2 -f plain --no-color "
                                 "features/alice.feature features/bob.feature")
    assert result.returncode != 0, result.output
    assert ('ConfigError: PARALLEL: environment file defines "before_all", '
            "which is not called with --jobs > 1.") in result.output

    # -- BUT NOTE: The same environment file works in sequential mode.
    result = run_behave(workdir, "--jobs=1 -f plain --no-color "
                                 "features/alice.feature")
    assert result.returncode == 0, result.output
    assert "HOOK: BEFORE-ALL" in result.output


# -----------------------------------------------------------------------------
# RULE: Parallel mode behaves like sequential mode
# -----------------------------------------------------------------------------
def test_scenario_selection_by_line_number_is_preserved(regression_workdir):
    """Scenario selection by line number is preserved"""
    workdir = regression_workdir
    workdir.write_file("features/many.feature", u"""
        Feature: Many
          Scenario: M1
            Given a step passes
          Scenario: M2
            Given a step passes
        """)
    result = run_behave(workdir, "--jobs=1 -f plain --no-color "
                                 "features/many.feature:2 features/alice.feature")
    assert result.returncode == 0, result.output
    assert "2 scenarios passed, 0 failed, 1 skipped" in result.output

    # -- BUT NOTE: The parallel test run must select the same scenarios.
    result = run_behave(workdir, "--jobs=2 -f plain --no-color "
                                 "features/many.feature:2 features/alice.feature")
    assert result.returncode == 0, result.output
    assert "2 scenarios passed, 0 failed, 1 skipped" in result.output


def test_a_failing_before_worker_hook_aborts_the_test_run(regression_workdir):
    """A failing before_worker hook aborts the test run"""
    workdir = regression_workdir
    workdir.write_file("features/environment.py", u"""
        def before_worker(context):
            raise RuntimeError("XFAIL-SETUP")
        """)
    result = run_behave(workdir, "--jobs=2 -f plain --no-color "
                                 "features/alice.feature features/bob.feature")
    assert result.returncode != 0, result.output
    assert "0 features passed, 0 failed, 0 skipped, 2 untested" in result.output
    assert "HOOK-ERROR in before_worker" in result.output


def test_a_failing_after_worker_hook_fails_the_test_run(regression_workdir):
    """A failing after_worker hook fails the test run"""
    workdir = regression_workdir
    workdir.write_file("features/environment.py", u"""
        def after_worker(context):
            raise RuntimeError("XFAIL-TEARDOWN")
        """)
    result = run_behave(workdir, "--jobs=2 -f plain --no-color "
                                 "features/alice.feature features/bob.feature")
    assert result.returncode != 0, result.output
    assert "2 features passed, 0 failed, 0 skipped" in result.output
    assert "HOOK-ERROR in after_worker" in result.output


def test_a_formatter_with_an_outfile_writes_the_output_of_all_workers(regression_workdir):
    """The output chunks of all workers are appended to the outfile"""
    workdir = regression_workdir
    result = run_behave(workdir,
                        "--jobs=2 -f plain -o reports/report.txt --no-color "
                        "features/alice.feature features/bob.feature "
                        "features/charly.feature features/dora.feature")
    assert result.returncode == 0, result.output
    assert "4 features passed, 0 failed, 0 skipped" in result.output
    # -- HINT: The formatter output is in the outfile, not on the console.
    assert "Feature: Alice" not in result.output
    report = (workdir.path / "reports" / "report.txt").read_text()
    for name in ("Alice", "Bob", "Charly", "Dora"):
        assert report.count("Feature: %s" % name) == 1, report


def test_console_and_outfile_formatters_can_be_combined(regression_workdir):
    workdir = regression_workdir
    result = run_behave(workdir,
                        "--jobs=2 -f progress -o progress.txt -f plain "
                        "--no-color features/alice.feature features/bob.feature")
    assert result.returncode == 0, result.output
    assert "Feature: Alice" in result.output
    assert "Feature: Bob" in result.output
    report = (workdir.path / "progress.txt").read_text()
    assert "features/alice.feature" in report
    assert "features/bob.feature" in report
    assert "Feature: Alice" not in report


@pytest.mark.parametrize("format_name", ["json", "json.pretty"])
def test_json_output_of_all_workers_is_merged_into_one_report(
        regression_workdir, format_name):
    """The JSON formatter writes one valid report for the whole test-run"""
    workdir = regression_workdir
    workdir.write_file("features/failing.feature", u"""
        Feature: Failing
          Scenario: F1
            Given a step fails
        """)
    result = run_behave(workdir,
                        "--jobs=2 -f %s -o report.json --no-color features"
                        % format_name)
    assert result.returncode == 1, result.output
    report = json.loads((workdir.path / "report.json").read_text())
    statuses = dict((feature["name"], feature["status"]) for feature in report)
    assert statuses == {"Alice": "passed", "Bob": "passed", "Charly": "passed",
                        "Dora": "passed", "Failing": "failed"}


def test_json_report_is_the_same_as_in_sequential_mode(regression_workdir):
    workdir = regression_workdir

    def make_report(jobs):
        result = run_behave(workdir, "--jobs=%d -f json.pretty -o report.json "
                                     "--no-color features" % jobs)
        assert result.returncode == 0, result.output
        report = json.loads((workdir.path / "report.json").read_text())
        for feature in report:
            for element in feature["elements"]:
                for step in element["steps"]:
                    step["result"].pop("duration")
        return sorted(report, key=lambda feature: feature["name"])

    assert make_report(jobs=2) == make_report(jobs=1)


def test_json_report_on_the_console_is_valid(regression_workdir):
    workdir = regression_workdir
    workdir.write_file("features/environment.py", u"")
    result = run_behave(workdir, "--jobs=2 -f json --no-color --no-summary "
                                 "features/alice.feature features/bob.feature")
    assert result.returncode == 0, result.output
    json_text = result.output[result.output.index("[\n"):]
    report = json.loads(json_text)
    assert sorted(feature["name"] for feature in report) == ["Alice", "Bob"]


def test_json_report_without_any_feature_is_valid(regression_workdir):
    workdir = regression_workdir
    workdir.write_file("features/empty/empty.feature", u"# -- NO FEATURE\n")
    result = run_behave(workdir, "--jobs=2 -f json -o report.json "
                                 "features/empty")
    assert result.returncode == 0, result.output
    assert json.loads((workdir.path / "report.json").read_text()) == []


def test_an_aggregating_formatter_is_rejected(regression_workdir):
    """An aggregating formatter is rejected instead of producing partial output"""
    result = run_behave(regression_workdir,
                        "--jobs=2 -f rerun --no-color "
                        "features/alice.feature features/bob.feature")
    assert result.returncode != 0, result.output
    assert ('ConfigError: PARALLEL: formatter "rerun" is not supported '
            "with --jobs > 1") in result.output


def test_junit_reports_are_written_per_feature(regression_workdir):
    """NEW: JUnit reports are written per feature (nested, missing directory)"""
    workdir = regression_workdir
    result = run_behave(workdir,
                        "--jobs=2 --junit --junit-directory=reports/deep/junit "
                        "-f plain --no-color "
                        "features/alice.feature features/bob.feature "
                        "features/charly.feature features/dora.feature")
    assert result.returncode == 0, result.output
    assert "4 features passed, 0 failed, 0 skipped" in result.output
    junit_dir = workdir.path / "reports" / "deep" / "junit"
    assert junit_dir.is_dir(), result.output
    report_names = sorted(path.name for path in junit_dir.glob("TESTS-*.xml"))
    assert report_names == ["TESTS-alice.xml", "TESTS-bob.xml",
                            "TESTS-charly.xml", "TESTS-dora.xml"]


def test_an_explicitly_selected_runner_is_not_replaced(workdir):
    """An explicitly selected runner is not replaced by the parallel runner

    ADAPTED: Replaces the two scenarios about the auto-selection of the
    parallel runner by "--jobs > 1" (core logic that no longer exists;
    the abbreviated-option scenario tested the same core logic).
    The command line "-r" option must win over the behave.ini runner.
    """
    result = run_behave(workdir, "--jobs=2 -r behave.runner:Runner "
                                 "-f plain --no-color "
                                 "features/alice.feature features/bob.feature")
    assert result.returncode == 0, result.output
    assert "USING RUNNER: behave.runner:Runner" in result.output
    assert "HOOK: BEFORE-ALL" in result.output
    assert "HOOK: WORKER-STARTED" not in result.output
    assert "HOOK: BEFORE-PARALLEL" not in result.output


def test_the_parallel_runner_can_be_selected_without_a_config_file(workdir):
    """NEW: The parallel runner can be selected with "-r" (without behave.ini)"""
    (workdir.path / "behave.ini").unlink()
    result = run_behave(workdir, "--jobs=2 -r behave_parallel_runner:ParallelRunner "
                                 "-f plain --no-color "
                                 "features/alice.feature features/bob.feature")
    assert result.returncode == 0, result.output
    assert "2 features passed, 0 failed, 0 skipped" in result.output
    assert "USING RUNNER: %s" % PARALLEL_RUNNER_NAME in result.output
    assert "HOOK: BEFORE-PARALLEL jobs=2" in result.output
    assert "HOOK: WORKER-STARTED" in result.output
    assert "HOOK: BEFORE-ALL" not in result.output


def test_log_output_of_the_workers_is_not_swallowed(regression_workdir):
    """Log output of the workers is not swallowed"""
    workdir = regression_workdir
    workdir.write_file("features/environment.py", u"""
        def before_parallel(context):
            print("HOOK: BEFORE-PARALLEL")
        """)
    workdir.write_file("features/steps/log_steps.py", u"""
        import logging
        from behave import step

        @step('a step logs')
        def step_logs(context):
            logging.getLogger("demo").warning("LOG-FROM-WORKER")
        """)
    workdir.write_file("features/logging.feature", u"""
        Feature: Logging
          Scenario: L1
            Given a step logs
        """)
    result = run_behave(workdir, "--jobs=1 -f plain --no-color --no-logcapture "
                                 "features/logging.feature features/alice.feature")
    assert result.returncode == 0, result.output
    assert "LOG-FROM-WORKER" in result.output

    # -- BUT NOTE: The parallel test run must not swallow the log output.
    result = run_behave(workdir, "--jobs=2 -f plain --no-color --no-logcapture "
                                 "features/logging.feature features/alice.feature")
    assert result.returncode == 0, result.output
    assert "LOG-FROM-WORKER" in result.output


def test_a_feature_file_that_cannot_be_parsed_aborts_the_test_run(regression_workdir):
    """A feature file that cannot be parsed aborts the test run"""
    workdir = regression_workdir
    workdir.write_file("features/broken.feature", u"""
        Feature: Broken
          Scenario: B1
            Given a step passes
          THIS LINE IS NOT GHERKIN
        """)
    result = run_behave(workdir, "--jobs=2 -f plain --no-color "
                                 "features/broken.feature features/alice.feature")
    assert result.returncode != 0, result.output
    assert "ParserError" in result.output


def test_fail_early_with_stop_ends_the_test_run(regression_workdir):
    """Fail-early with --stop ends the test run (many feature files)

    HINT: Many more feature files than workers.
    """
    workdir = regression_workdir
    workdir.write_file("features/many/m01.feature", u"""
        Feature: M01
          Scenario: M01
            Given a step fails
        """)
    workdir.make_passing_features(11, "features/many")
    result = run_behave(workdir, "--jobs=2 --stop -f plain --no-color "
                                 "features/many")
    assert result.returncode != 0, result.output
    assert "1 failed" in result.output
    assert "untested" in result.output
    assert "HOOK: WORKER-STARTED" in result.output


def test_a_failing_after_worker_hook_fails_the_test_run_with_stop(regression_workdir):
    """A failing after_worker hook fails the test run with --stop"""
    workdir = regression_workdir
    workdir.make_passing_features(11, "features/many")
    workdir.write_file("features/environment.py", u"""
        import time

        def after_worker(context):
            time.sleep(0.3)
            print("HOOK: WORKER-STOPPED")
            raise RuntimeError("XFAIL-TEARDOWN")
        """)
    workdir.write_file("features/many/m01.feature", u"""
        Feature: M01
          Scenario: M01
            Given a step fails
        """)
    result = run_behave(workdir, "--jobs=2 --stop -f plain --no-color "
                                 "features/many")
    assert result.returncode != 0, result.output
    assert "HOOK-ERROR in after_worker" in result.output
    # -- BUT NOTE: The worker shutdown output is seen before the summary.
    assert ("HOOK: WORKER-STOPPED\n"
            "HOOK-ERROR in after_worker: RuntimeError: XFAIL-TEARDOWN"
            ) in result.output


def test_a_test_run_that_is_aborted_in_a_worker_aborts_the_test_run(regression_workdir):
    """A test run that is aborted in a worker aborts the parallel test run"""
    workdir = regression_workdir
    workdir.make_passing_features(11, "features/many")
    workdir.write_file("features/steps/abort_steps.py", u"""
        from behave import step

        @step('the test run is aborted')
        def step_aborts_testrun(context):
            context.abort(reason="XFAIL-ABORT")
        """)
    workdir.write_file("features/many/m01.feature", u"""
        Feature: M01
          Scenario: M01
            Given the test run is aborted
        """)
    result = run_behave(workdir, "--jobs=1 -f plain --no-color features/many")
    assert result.returncode != 0, result.output
    assert "ABORTED: By user." in result.output

    # -- BUT NOTE: The parallel test run must fail in the same way.
    result = run_behave(workdir, "--jobs=2 -f plain --no-color features/many")
    assert result.returncode != 0, result.output
    assert "ABORTED: By user." in result.output
    assert "untested" in result.output


def test_a_feature_file_without_a_feature_is_ignored(regression_workdir):
    """A feature file without a feature is ignored"""
    workdir = regression_workdir
    workdir.make_passing_features(11, "features/many")
    workdir.write_file("features/many/m01.feature", u"""
        # -- EMPTY: Feature file without any feature.
        """)
    result = run_behave(workdir, "--jobs=1 -f plain --no-color features/many")
    assert result.returncode == 0, result.output
    assert "11 features passed, 0 failed, 0 skipped" in result.output

    # -- BUT NOTE: The parallel test run must ignore this file, too.
    result = run_behave(workdir, "--jobs=2 -f plain --no-color features/many")
    assert result.returncode == 0, result.output
    assert "11 features passed, 0 failed, 0 skipped" in result.output
    assert "PARALLEL-WORKER ERROR" not in result.output


def test_config_changes_of_the_before_parallel_hook_are_seen_by_workers(regression_workdir):
    """Configuration changes of the before_parallel hook are seen by workers"""
    workdir = regression_workdir
    workdir.make_passing_features(11, "features/many")
    workdir.write_file("features/environment.py", u"""
        import threading
        from behave.configuration import UserData

        def before_parallel(context):
            context.config.userdata = UserData(color="blue",
                                               lock=threading.Lock())
        """)
    workdir.write_file("features/steps/userdata_steps.py", u"""
        from behave import step

        @step('the userdata color is "{value}"')
        def step_userdata_color_is(context, value):
            assert context.config.userdata.get("color") == value
        """)
    workdir.write_file("features/many/m01.feature", u"""
        Feature: M01
          Scenario: M01
            Given the userdata color is "blue"
        """)
    result = run_behave(workdir, "--jobs=2 -f plain --no-color "
                                 "features/many/m01.feature "
                                 "features/many/p02.feature")
    assert result.returncode == 0, result.output
    assert "2 features passed, 0 failed, 0 skipped" in result.output
    assert ("PARALLEL: WARNING -- not picklable, therefore not sent to the "
            "workers: userdata[lock]") in result.output


DIE_STEPS_TEXT = u"""
    import os
    from behave import step

    @step('the worker process dies')
    def step_worker_dies(context):
        os._exit(3)
    """


def test_a_worker_process_that_dies_fails_only_its_feature(regression_workdir):
    """The feature is reported as errored and the other features still run"""
    workdir = regression_workdir
    workdir.make_passing_features(11, "features/many")
    workdir.write_file("features/steps/die_steps.py", DIE_STEPS_TEXT)
    workdir.write_file("features/many/m01.feature", u"""
        Feature: M01
          Scenario: M01
            Given the worker process dies
        """)
    result = run_behave(workdir, "--jobs=2 -f plain --no-color --junit "
                                 "features/many")
    assert result.returncode == 1, result.output
    assert ("PARALLEL-WORKER DIED in features/many/m01.feature "
            "(exit code: 3)") in result.output
    assert "ABORTED" not in result.output
    assert "11 features passed, 0 failed, 1 error, 0 skipped\n" in result.output
    assert "11 scenarios passed, 0 failed, 1 error, 0 skipped\n" in result.output
    # -- REPLACED WORKER: Three workers were started for --jobs=2.
    assert result.output.count("HOOK: WORKER-STARTED") == 3
    # -- HINT: A died worker must be seen in the JUnit report of its feature.
    junit_report = (workdir.path / "reports" / "TESTS-m01.xml").read_text()
    assert 'errors="1"' in junit_report
    assert "PARALLEL-WORKER DIED in features/many/m01.feature" in junit_report


def test_a_replaced_worker_keeps_the_worker_id(regression_workdir):
    workdir = regression_workdir
    workdir.make_passing_features(6, "features/many")
    workdir.write_file("features/steps/die_steps.py", DIE_STEPS_TEXT)
    workdir.write_file("features/environment.py", u"""
        def before_worker(context):
            print("HOOK: WORKER-STARTED id=%s" % context.worker_id)
        """)
    workdir.write_file("features/many/m01.feature", u"""
        Feature: M01
          Scenario: M01
            Given the worker process dies
        """)
    result = run_behave(workdir, "--jobs=2 -f plain --no-color features/many")
    worker_ids = sorted(re.findall(r"HOOK: WORKER-STARTED id=(\d+)",
                                   result.output))
    assert len(worker_ids) == 3, result.output
    assert set(worker_ids) == {"0", "1"}, result.output


def test_a_worker_process_that_dies_with_stop_ends_the_test_run(regression_workdir):
    workdir = regression_workdir
    workdir.make_passing_features(11, "features/many")
    workdir.write_file("features/steps/die_steps.py", DIE_STEPS_TEXT)
    workdir.write_file("features/many/a00.feature", u"""
        Feature: A00
          Scenario: A00
            Given the worker process dies
        """)
    result = run_behave(workdir, "--jobs=2 --stop -f plain --no-color "
                                 "features/many")
    assert result.returncode == 1, result.output
    assert "PARALLEL-WORKER DIED in features/many/a00.feature" in result.output
    assert re.search(r"1 error, 0 skipped, \d+ untested", result.output), \
        result.output
    # -- NOT REPLACED: No further features should run.
    assert result.output.count("HOOK: WORKER-STARTED") == 2


def test_a_worker_process_that_dies_during_its_setup_aborts_the_test_run(regression_workdir):
    workdir = regression_workdir
    workdir.write_file("features/environment.py", u"""
        import os

        def before_worker(context):
            os._exit(4)
        """)
    result = run_behave(workdir, "--jobs=2 -f plain --no-color "
                                 "features/alice.feature features/bob.feature")
    assert result.returncode == 1, result.output
    assert "DIED (exit code: 4)" in result.output
    assert "0 features passed, 0 failed, 0 skipped, 2 untested" in result.output


@pytest.mark.skipif(not hasattr(os, "fork"), reason="REQUIRES: os.fork()")
def test_a_died_worker_is_seen_while_its_child_process_lives_on(regression_workdir):
    """REGRESSION: A child process that a step has forked keeps the worker's
    connection open. The end of the worker was only seen when this child ended.
    """
    workdir = regression_workdir
    workdir.make_passing_features(3, "features/many")
    workdir.write_file("features/steps/fork_steps.py", u"""
        import os, time
        from behave import step

        @step('the worker process forks a child and dies')
        def step_fork_and_die(context):
            if os.fork() == 0:
                time.sleep(30)
                os._exit(0)
            os._exit(3)

        @step('the worker process leaves a child process behind')
        def step_leave_child_behind(context):
            if os.fork() == 0:
                time.sleep(30)
                os._exit(0)
        """)
    workdir.write_file("features/many/a00.feature", u"""
        Feature: A00
          Scenario: A00
            Given the worker process forks a child and dies
        """)
    workdir.write_file("features/many/a01.feature", u"""
        Feature: A01
          Scenario: A01
            Given the worker process leaves a child process behind
        """)
    # -- HINT: Output goes to a file. The forked child processes inherit the
    # output of behave, a pipe would stay open until they have ended.
    output_file = workdir.path / "output.txt"
    started = time.time()
    with open(str(output_file), "w") as output:
        process = subprocess.run(
            [sys.executable, "-m", "behave", "--jobs=2", "-f", "plain",
             "--no-color", "features/many"],
            cwd=str(workdir.path), stdout=output, stderr=subprocess.STDOUT,
            timeout=25)
    elapsed = time.time() - started
    text = output_file.read_text()
    assert elapsed < 20, "SLOW: %.1fs\n%s" % (elapsed, text)
    assert process.returncode == 1, text
    assert "PARALLEL-WORKER DIED in features/many/a00.feature" in text
    assert "4 features passed, 0 failed, 1 error, 0 skipped" in text


def test_an_outfile_that_cannot_be_opened_fails_before_any_hook_runs(workdir):
    """LIKE SEQUENTIAL MODE: Formatter streams are opened before the hooks."""
    (workdir.path / "report.txt").mkdir()
    result = run_behave(workdir, "--jobs=2 -f plain -o report.txt --no-color "
                                 "features/alice.feature features/bob.feature")
    assert result.returncode != 0, result.output
    assert "HOOK: BEFORE-PARALLEL" not in result.output
    assert "HOOK: WORKER-STARTED" not in result.output


def test_a_user_defined_formatter_with_an_outfile_must_state_how_it_uses_it(regression_workdir):
    """REGRESSION: A formatter that uses its outfile as directory name
    (like: allure-behave) got a text buffer instead of its outfile name.
    """
    workdir = regression_workdir
    workdir.write_file("my_formatters.py", u"""
        import os
        from behave.formatter.base import Formatter

        class UnknownFormatter(Formatter):
            pass

        class DirectoryFormatter(Formatter):
            \"\"\"Writes one file per feature into the outfile directory.\"\"\"
            parallel_outfile = "direct"

            def feature(self, feature):
                directory = self.stream_opener.name
                os.makedirs(directory, exist_ok=True)
                filename = os.path.join(directory, "%s.txt" % feature.name)
                with open(filename, "w") as f:
                    f.write("FEATURE: %s" % feature.name)
        """)
    result = run_behave(workdir, "--jobs=2 -f my_formatters:UnknownFormatter "
                                 "-o results features/alice.feature")
    assert result.returncode != 0, result.output
    assert "parallel_outfile" in result.output
    assert not (workdir.path / "results").exists()

    result = run_behave(workdir, "--jobs=2 -f my_formatters:DirectoryFormatter "
                                 "-o results -f plain --no-color "
                                 "features/alice.feature features/bob.feature")
    assert result.returncode == 0, result.output
    assert "Feature: Alice" in result.output
    names = sorted(path.name for path in (workdir.path / "results").iterdir())
    assert names == ["Alice.txt", "Bob.txt"]


# -----------------------------------------------------------------------------
# SERIAL FEATURES: Tag "@serial"
# -----------------------------------------------------------------------------
SERIAL_STEPS_TEXT = u"""
    import time
    from behave import step

    @step('I work as "{name}"')
    def step_work(context, name):
        with open("work.log", "a") as f:
            f.write("%s %s %.6f\\n" % (name, "START", time.time()))
        time.sleep(0.3)
        with open("work.log", "a") as f:
            f.write("%s %s %.6f\\n" % (name, "END", time.time()))
    """

WORK_FEATURE_TEMPLATE = u"""
    {feature_tags}
    Feature: {name}
      {scenario_tags}
      Scenario: {name}
        Given I work as "{name}"
    """


def read_work_intervals(workdir):
    intervals = {}
    for line in (workdir.path / "work.log").read_text().splitlines():
        name, kind, timestamp = line.split()
        intervals.setdefault(name, {})[kind] = float(timestamp)
    return dict((name, (times["START"], times["END"]))
                for name, times in intervals.items())


def overlaps(interval1, interval2):
    return interval1[0] < interval2[1] and interval2[0] < interval1[1]


def test_serial_features_do_not_run_together_with_other_features(regression_workdir):
    workdir = regression_workdir
    workdir.write_file("features/steps/serial_steps.py", SERIAL_STEPS_TEXT)
    names = ["a_serial_feature", "b_normal", "c_serial_scenario", "d_normal",
             "e_normal", "f_normal"]
    for name in names:
        workdir.write_file("features/work/%s.feature" % name,
                           WORK_FEATURE_TEMPLATE.format(
            name=name,
            feature_tags="@serial" if name == "a_serial_feature" else "",
            scenario_tags="@serial" if name == "c_serial_scenario" else ""))

    result = run_behave(workdir, "--jobs=3 -f plain --no-color features/work")
    assert result.returncode == 0, result.output
    assert "6 features passed, 0 failed, 0 skipped" in result.output
    # -- HINT: A feature that is handed back must not be reported twice.
    assert result.output.count("Feature: a_serial_feature") == 1

    intervals = read_work_intervals(workdir)
    assert sorted(intervals) == names
    for serial_name in ("a_serial_feature", "c_serial_scenario"):
        for name in names:
            if name != serial_name:
                assert not overlaps(intervals[serial_name], intervals[name]), \
                    "%s overlaps with %s" % (serial_name, name)
    # -- SERIAL FEATURES RUN LAST:
    last_normal_end = max(intervals[name][1] for name in names
                          if "normal" in name)
    assert intervals["a_serial_feature"][0] >= last_normal_end


def test_serial_tag_of_a_scenario_that_is_not_selected_is_ignored(regression_workdir):
    workdir = regression_workdir
    workdir.write_file("features/steps/serial_steps.py", SERIAL_STEPS_TEXT)
    for name in ("a_work", "b_work"):
        workdir.write_file("features/work/%s.feature" % name, u"""
            Feature: {name}
              Scenario: {name}
                Given I work as "{name}"

              @serial @slow
              Scenario: Not selected
                Given a step passes
            """.format(name=name))
    result = run_behave(workdir, "--jobs=2 --tags='not @slow' -f plain "
                                 "--no-color features/work")
    assert result.returncode == 0, result.output
    intervals = read_work_intervals(workdir)
    assert overlaps(intervals["a_work"], intervals["b_work"])


def test_serial_features_are_untested_if_the_test_run_is_stopped(regression_workdir):
    workdir = regression_workdir
    workdir.write_file("features/work/a_serial.feature", u"""
        @serial
        Feature: Serial
          Scenario: S1
            Given a step passes
        """)
    workdir.write_file("features/work/b_failing.feature", u"""
        Feature: Failing
          Scenario: F1
            Given a step fails
        """)
    result = run_behave(workdir, "--jobs=2 --stop -f plain --no-color "
                                 "features/work")
    assert result.returncode == 1, result.output
    assert "0 features passed, 1 failed, 0 skipped, 1 untested" in result.output
