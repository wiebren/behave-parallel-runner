# -*- coding: UTF-8 -*-
"""
Functional test for :mod:`behave_parallel_runner.runner`:
A KeyboardInterrupt (SIGINT) that only the parent process receives
(like: "kill -INT", CI job cancellation, IDE stop button) must hard-stop
the worker processes, instead of waiting until their features are done.
"""

import os
import signal
import subprocess
import sys
import time

import pytest


STEPS_TEXT = u"""
import time
from behave import step

@step('I sleep {seconds:d} seconds')
def step_sleep(context, seconds):
    time.sleep(seconds)
"""

FEATURE_TEXT = u"""
Feature: Slow {index}
  Scenario: Slow {index}
    Given I sleep 60 seconds
"""

BEHAVE_INI_TEXT = u"""
[behave]
runner = behave_parallel_runner:ParallelRunner
"""

ENVIRONMENT_TEXT = u"""
def before_worker(context):
    print("HOOK: WORKER-STARTED")
"""


def wait_for_text(filename, text, timeout):
    deadline = time.time() + timeout
    while time.time() < deadline:
        with open(filename) as f:
            if f.read().count(text) >= 2:
                return True
        time.sleep(0.1)
    return False


SIGTERM_HANDLER_ENVIRONMENT_TEXT = u"""
import signal

def before_worker(context):
    # -- LIKE: Graceful-shutdown handler of the application under test.
    signal.signal(signal.SIGTERM, lambda signum, frame: None)
    print("HOOK: WORKER-STARTED")
"""


@pytest.mark.skipif(sys.platform == "win32", reason="REQUIRES: SIGINT")
@pytest.mark.parametrize("environment_text", [
    ENVIRONMENT_TEXT, SIGTERM_HANDLER_ENVIRONMENT_TEXT,
], ids=["normal_workers", "workers_that_survive_sigterm"])
def test_keyboard_interrupt_in_parent_terminates_workers(tmp_path,
                                                         environment_text):
    features_dir = tmp_path / "features"
    (features_dir / "steps").mkdir(parents=True)
    (features_dir / "steps" / "steps.py").write_text(STEPS_TEXT)
    (features_dir / "environment.py").write_text(environment_text)
    for index in range(1, 5):
        (features_dir / ("slow%d.feature" % index)).write_text(
            FEATURE_TEXT.format(index=index))

    (tmp_path / "behave.ini").write_text(BEHAVE_INI_TEXT)

    output_file = tmp_path / "output.txt"
    with open(str(output_file), "w") as output:
        # -- HINT: Own session, so that only the parent gets the SIGINT.
        process = subprocess.Popen(
            [sys.executable, "-m", "behave", "--jobs=2", "-f", "plain",
             "--no-color", "features"],
            cwd=str(tmp_path), stdout=output,
            stderr=subprocess.STDOUT, start_new_session=True)
        try:
            workers_started = wait_for_text(str(output_file),
                                            "HOOK: WORKER-STARTED", timeout=30)
            time.sleep(0.5)     # -- ENSURE: Workers run their slow steps.
            interrupted_at = time.time()
            os.kill(process.pid, signal.SIGINT)     # -- PARENT ONLY.
            process.wait(timeout=30)
            elapsed = time.time() - interrupted_at
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()

    text = output_file.read_text()
    assert workers_started, text
    assert process.returncode == 1, text
    assert "ABORTED: By user." in text
    assert "4 untested" in text
    assert elapsed < 20, "SLOW: %.1fs (workers not terminated?)" % elapsed
