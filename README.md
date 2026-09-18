# behave-parallel-runner

A parallel test runner for [behave](https://github.com/behave/behave):
with `--jobs N` (N > 1), feature files run concurrently in worker processes.

```console
$ behave --jobs 4
```

It plugs into behave's runner extension-point. No patched behave is needed:
it works with the released `behave >= 1.3.3`.

## Installation

```console
$ pip install git+https://github.com/wiebren/behave-parallel-runner
```

Select the runner in your behave config-file:

```ini
# -- FILE: behave.ini
[behave]
runner = behave_parallel_runner:ParallelRunner
```

or on the command line: `behave -r behave_parallel_runner:ParallelRunner --jobs 4`.

It is safe to keep this runner configured permanently: with `--jobs=1`
(the default) or in dry-run mode, it runs your tests sequentially, exactly
like behave's own runner (including `before_all` / `after_all`).

## Execution model

* One parent process orchestrates up to N **worker processes**
  (`multiprocessing` with the "spawn" start-method on all platforms).
* The **work unit is one feature file**: all rules, scenarios and steps of a
  feature file run inside one worker, in their normal order, with the normal
  feature/rule/scenario/step/tag hooks.
  Scenario selection by line number (`features/alice.feature:12`, `@rerun.txt`)
  selects the same scenarios as in sequential mode.
* Each worker builds a complete, isolated behave runtime: it re-reads the
  configuration files and command-line options, loads `environment.py` and
  the step definitions, and keeps one `Context` for its whole lifetime.
* The parent prints each feature's output as one block when the feature
  finishes (completion order), merges the counts of all workers into one
  summary, and computes the exit status like the sequential runner.
* The number of selected feature files does not matter: with `--jobs > 1`
  even one feature file runs in a worker process, so that the same hooks are
  called in both cases.

## Hooks in parallel mode

Parallel mode **never calls** `before_all` / `after_all`.
It provides its own hooks instead:

| Hook | Sequential mode (jobs=1) | Parallel mode (jobs > 1) |
|---|---|---|
| `before_all` / `after_all` | unchanged | **never called** |
| `before_parallel` / `after_parallel` | ignored | once, in the **parent** process |
| `before_worker` / `after_worker` | ignored | once **per worker process** (startup / shutdown) |
| other hooks | unchanged | run in workers, unchanged |

`before_worker(context)` receives the worker's root context -- attributes set
there are visible to all hooks and steps that the worker later runs, just like
`before_all` attributes in sequential mode. The context also provides
`context.worker_id` (0..N-1) and `context.jobs`.
`before_parallel(context)` runs on the parent context (with `context.jobs`).

**Explicit migration required:** if your `environment.py` defines `before_all`
(or `after_all`) and you run with `--jobs > 1`, the runner refuses to start
unless a matching parallel-mode hook exists (`before_parallel` and/or
`before_worker`; likewise for `after_all`). What happens with your `*_all`
hook under parallel execution must be an explicit choice -- for example:

```python
# -- FILE: features/environment.py
def before_all(context):
    setup_test_database(context)

def before_worker(context):
    # -- EXPLICIT CHOICE: each worker needs its own setup.
    before_all(context)
```

### Per-worker resources

Use `context.worker_id` (0..N-1) to give each worker its own resource,
like a browser port, a test account or a database schema:

```python
# -- FILE: features/environment.py
TEST_ACCOUNTS = ["alice", "bob", "charly", "dora"]

def before_worker(context):
    # -- HINT: Each worker process uses its own test account.
    context.account = TEST_ACCOUNTS[context.worker_id]
    context.server_port = 8080 + context.worker_id

def after_worker(context):
    release_account(context.account)
```

## Formatters and reporters

* `plain` and the `progress` formatters are supported (output arrives in
  whole-feature blocks); `pretty` is replaced by `plain`.
* Formatters that need the complete test-run (`json`, `rerun`, the `steps.*`
  and `tags` formatters, `bad_steps`, and formatter classes derived from them)
  and any formatter bound to an `--outfile` are **rejected** with a
  `ConfigError`: many workers cannot write one file, and silently producing no
  report would be worse than failing. Use `--jobs=1` for those.
* An own formatter states what it needs with the class attribute
  `needs_complete_testrun`: `True` means "reject me with `--jobs > 1`"
  (instead of running once per worker); `False` on a class derived from a
  built-in aggregating formatter means "I am safe to run per worker".
* The **JUnit reporter** is fully supported: workers write their independent
  per-feature XML files.
* Only behave's `SummaryReporterV1` (the default) can be merged;
  `SummaryReporterV2` is rejected.

## Limitations

* **State is not shared between workers.** Module globals, context attributes
  and resources set up in one worker (or in the parent) do not exist in the
  others. Workers rebuild the configuration from the command line and the
  configuration files; changes made to a configuration object after it was
  built are not seen by workers -- except `stage`, `lang`, `tags` and
  `userdata`. These four are sent to the workers after the `before_parallel`
  hook has run, so this hook can still change them. Values that cannot be
  pickled (like a lock in `userdata`) are not sent; a warning names them.
* **Fail-early is best effort:** with `--stop` (or `--wip`), pending feature
  files are cancelled after the first failure, but features already running in
  a worker finish. The same applies if the test-run is aborted in one worker,
  for example with `context.abort()`: the other workers finish their current
  feature, the remaining features are reported as untested and the test-run
  fails (like in sequential mode).
* **KeyboardInterrupt:** the worker processes are terminated at once.
  Their `after_worker` hooks and cleanups are not run in this case.
* If a worker process dies (crash, `os._exit()`), the test-run is aborted.
* The summary duration is the wall-clock time of the whole run.
* Debugging is easier with `--jobs=1`: a debugger cannot be used in a worker
  process and tracebacks cross the process boundary as text.

### Programmatic use

Workers rebuild the configuration from the command line of the parent
process. If you build the `Configuration` yourself, describe how you did it,
so that the workers can do the same:

```python
from behave.configuration import Configuration
from behave.__main__ import run_behave

if __name__ == "__main__":      # -- REQUIRED: "spawn" re-imports __main__.
    args = ["--jobs=4", "-r", "behave_parallel_runner:ParallelRunner", "features"]
    config = Configuration(args, load_config=False)
    config.command_args = args              # default: sys.argv[1:]
    config.command_kwargs = {}              # keyword args of Configuration()
    config.command_load_config = False      # default: True
    raise SystemExit(run_behave(config))
```

## Development

```console
$ pip install -e ".[testing]"
$ pytest
```

## License

BSD-2-Clause, the same license as behave.
