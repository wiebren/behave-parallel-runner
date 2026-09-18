# -*- coding: UTF-8 -*-
"""
Test support for the functional tests of the parallel runner command line
(see: tests/functional/test_cli.py).

Ported from the behave fork, where these tests were:

* ``features/runner.parallel_jobs.feature`` (scenarios)
* ``features/steps/behave_parallel_steps.py`` (many-feature-files step)

The "behave4cmd0" step library is not installable, therefore the scenarios
became pytest tests that run the "behave" command in a subprocess.
"""

import shlex
import subprocess
import sys
import textwrap

import pytest


# -----------------------------------------------------------------------------
# TEST DATA: Files of the "Test Setup" scenario (same for each test)
# -----------------------------------------------------------------------------
# -- HINT: This package does not auto-select the parallel runner with
# "--jobs > 1"; the runner must be selected in the config-file (or with "-r").
BEHAVE_INI_TEXT = u"""
[behave]
runner = behave_parallel_runner:ParallelRunner
"""

STEPS_TEXT = u"""
from behave import step

@step('a step passes')
def step_passes(context):
    pass

@step('a step fails')
def step_fails(context):
    assert False, "XFAIL-STEP"
"""

ENVIRONMENT_TEXT = u"""
def before_all(context):
    print("HOOK: BEFORE-ALL")

def before_parallel(context):
    print("HOOK: BEFORE-PARALLEL jobs=%s" % context.jobs)

def after_parallel(context):
    print("HOOK: AFTER-PARALLEL")

def before_worker(context):
    print("HOOK: WORKER-STARTED")

def after_worker(context):
    print("HOOK: WORKER-STOPPED")
"""

# -- HINT: Background of Rule "Parallel mode behaves like sequential mode".
REGRESSION_ENVIRONMENT_TEXT = u"""
import logging

def before_parallel(context):
    print("HOOK: BEFORE-PARALLEL")

def before_worker(context):
    print("HOOK: WORKER-STARTED")
"""

FEATURE_FILES = {
    "features/alice.feature": u"""
        Feature: Alice
          Scenario: A1
            Given a step passes
            When a step passes
        """,
    "features/bob.feature": u"""
        Feature: Bob
          Scenario: B1
            Given a step passes
        """,
    "features/charly.feature": u"""
        Feature: Charly
          Scenario: C1
            Given a step passes
        """,
    "features/dora.feature": u"""
        Feature: Dora
          Scenario: D1
            Given a step passes
        """,
}

PASSING_FEATURE_TEMPLATE = u"""
Feature: {name}
  Scenario: {name}
    Given a step passes
"""


# -----------------------------------------------------------------------------
# TEST SUPPORT
# -----------------------------------------------------------------------------
class CommandResult(object):
    """Result of a "behave" command run (duck-types behave4cmd0's command)."""
    __slots__ = ("returncode", "output")

    def __init__(self, returncode, output):
        self.returncode = returncode
        self.output = output

    def __str__(self):
        return "returncode=%s\n%s" % (self.returncode, self.output)


class Workdir(object):
    """Working directory of one test (replaces: "a new working directory")."""

    def __init__(self, path):
        self.path = path

    def write_file(self, relpath, text):
        """Replaces the step: 'a file named "{relpath}" with: ...'"""
        path = self.path / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(text).lstrip(), encoding="UTF-8")
        return path

    def make_passing_features(self, count, directory):
        """Create many feature files, named: "p01.feature", "p02.feature", ...

        Replaces the step:
        '{count:d} passing feature files in the directory "{directory}"'
        """
        for index in range(1, count + 1):
            name = "p%02d" % index
            self.write_file("%s/%s.feature" % (directory, name),
                            PASSING_FEATURE_TEMPLATE.format(name=name.upper()))


def run_behave(workdir, args, timeout=120):
    """Replaces the step: 'I run "behave {args}"' (uses: no shell)."""
    command = [sys.executable, "-m", "behave"] + shlex.split(args)
    process = subprocess.run(command, cwd=str(workdir.path),
                             capture_output=True, text=True, timeout=timeout)
    # -- HINT: behave4cmd0 checks stdout and stderr together.
    return CommandResult(process.returncode, process.stdout + process.stderr)


# -----------------------------------------------------------------------------
# FIXTURES
# -----------------------------------------------------------------------------
@pytest.fixture
def workdir(tmp_path):
    """Provides a fresh working directory with the "Test Setup" files.

    HINT: The feature file used one working directory for all scenarios;
    each test gets its own one here (and adds only the files it needs).
    """
    this_workdir = Workdir(tmp_path)
    this_workdir.write_file("behave.ini", BEHAVE_INI_TEXT)
    this_workdir.write_file("features/steps/steps.py", STEPS_TEXT)
    this_workdir.write_file("features/environment.py", ENVIRONMENT_TEXT)
    for relpath, text in FEATURE_FILES.items():
        this_workdir.write_file(relpath, text)
    return this_workdir


@pytest.fixture
def regression_workdir(workdir):
    """Working directory for Rule: "Parallel mode behaves like sequential mode"

    HINT: Its Background replaces the environment file (no "*_all" hooks).
    """
    workdir.write_file("features/environment.py", REGRESSION_ENVIRONMENT_TEXT)
    return workdir
