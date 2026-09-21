# Version History

## 0.1.0 (unreleased)

* FIXED: With Python 3.12, behave did not exit after its summary as long as
  a child process lived that a step had forked with `os.fork()` (it waited
  for the resource tracker of `multiprocessing`; CPython gh-146313).
* Requires `behave >= 1.4.0` (until it is released: its development version).
  behave provides there what this package had to work around before:
  a `Configuration` remembers how it was built, so programmatic use needs no
  `config.command_args`, ... description anymore (behave #1349), and the JUnit
  reporter creates its report directory race-free (behave #1345).
* Initial version: `ParallelRunner` runs feature files in worker processes
  with `--jobs N`; parallel-mode hooks `before_parallel` / `after_parallel`
  and `before_worker` / `after_worker`; `context.worker_id`, `context.jobs`.
* A worker process that dies no longer aborts the test-run: the feature that
  it ran is reported as errored (also in its JUnit report), the worker is
  replaced (same `context.worker_id`) and the other features still run.
* Features with the tag `@serial` never run together with another feature:
  they run one by one, after the other features.
* Formatters with an `--outfile` are supported; the `json` and `json.pretty`
  formatters are supported, too (one merged report for the whole test-run).
  An own or third-party formatter with an `--outfile` states how it uses its
  outfile with the class attribute `parallel_outfile` (`"merge"` or
  `"direct"`, like allure-behave with its results directory).
* The end of a worker process is seen even if a child process that a step
  has forked lives on; a worker that survives `SIGTERM` is killed on
  KeyboardInterrupt; `after_parallel` runs whenever `before_parallel` did.
