# Version History

## 0.1.0 (unreleased)

* Initial version: `ParallelRunner` runs feature files in worker processes
  with `--jobs N`; parallel-mode hooks `before_parallel` / `after_parallel`
  and `before_worker` / `after_worker`; `context.worker_id`, `context.jobs`.
