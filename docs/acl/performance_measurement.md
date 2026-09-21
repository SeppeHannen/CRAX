# Measuring training performance

`training/performance` records how fast a training run is and, on request,
where its time goes. It is off by default and has no effect on the compiled
program when on. The reasoning behind the tool choice is in
[`profiling_options.md`](profiling_options.md); measurements so far are under
[`measurements/`](measurements/).

**Weights & Biases is the system of record.** Every training entry point checks
for W&B credentials before compiling anything and aborts with instructions if
they are missing. Throughput numbers are compared in the W&B UI; profiler
traces are stored as W&B artifacts and pulled to a laptop for viewing.

## Quick start

```bash
# Once per machine: API key from https://wandb.ai/authorize in your shell profile
echo 'export WANDB_API_KEY=<key>' >> ~/.bashrc

# Record wall-clock, SPS, compile time/count and compiler program statistics.
python -m training.train_env --env_name safe_goal_point --alg ppo_lag \
    --num_envs 2048 --num_evals 12 --measure_performance

# Same, plus a jax.profiler trace of epoch 6. Trace ONE epoch: it runs ~3x
# slower and produces ~200 MB. Implies --measure_performance.
python -m training.train_env ... --profile_epochs 6

# On your laptop: pull traces from W&B and open XProf on them
python -m training.performance.fetch_traces <run_name>            # one run
python -m training.performance.fetch_traces --filter tags=perf    # every run tagged 'perf'
#   -> downloads into runs/traces/plugins/profile/<run_name>/ and starts
#      xprof --logdir runs/traces  (http://localhost:8791). Add --no-open to skip the UI.

# Terminal alternative to the UI (JSON out):
xprof get_overview runs/traces/plugins/profile/<run_name>
xprof get_kernel_stats runs/traces/plugins/profile/<run_name> --limit 100000
```

Compute nodes without internet: run with `WANDB_MODE=offline` and `wandb sync`
the run directory afterwards; the trace artifact is uploaded during sync.

## Where things end up

| Item | W&B | Local |
|---|---|---|
| per-epoch `epoch_wall_seconds`, `epoch_steps_per_second`, `epoch_compiles`, `epoch_compile_seconds` | run history under `performance/*`, through the trainer's progress callback like every other metric (so the dashboard registry applies) | `runs/performance/<run>/performance.json` → `history` |
| run summary (median steady SPS, IQR, compile time, `program_flops`, `program_bytes_accessed`, `program_peak_memory_bytes`, phase timings, `total_compiles`) | `run.summary["performance/…"]` → columns in the runs table | `performance.json` → `summary` |
| metadata (git SHA/dirty, jax/mujoco versions, GPU, `XLA_FLAGS`) | run config (already) + `performance.json` in run files | `performance.json` → `metadata` |
| profiler trace | artifact `trace-<run_name>` of type `xprof-trace` | `runs/performance/<run>/trace/` (job side), `runs/traces/` (after fetch) |

### Comparing runs in W&B

* **Two runs side by side**: select both in the runs table, open the *Run
  Comparer* panel → all `performance/*` summary values in one table with diffs.
* **Throughput over time**: line plot of `performance/epoch_steps_per_second`
  vs `environment_steps`, grouped by whatever config field you are varying
  (e.g. `num_envs`).  It is also in the dashboard view's Trust section.
* **Regression columns**: add `performance/steady_steps_per_second_median`,
  `performance/total_compiles`, `performance/program_flops` to the runs table.
  Read the deterministic columns first: if `program_flops` /
  `program_bytes_accessed` moved, the compiled program changed; if only
  wall-clock moved, the cause is scheduling, host work, or noise.

## What is recorded and how

| Field | Meaning | Source |
|---|---|---|
| `epochs[].wall_seconds`, `steps_per_second` | one entry per training epoch (one `training_epoch` call) | wall clock around the blocking call |
| `epochs[].compiles_during_epoch`, `compile_seconds_during_epoch` | XLA compilations that finished during that epoch | JAX monitoring hook `/jax/core/compile/backend_compile_duration` |
| `summary.steady_steps_per_second_median`, `_iqr` | throughput over *steady* epochs: not epoch 0, not traced, compile time ≤ 2 % of wall | derived |
| `summary.phase_environment_reset_seconds`, `phase_initial_evaluation_seconds`, `phase_epoch_compile_seconds` | one-off host-side phases | `tracker.phase(...)` |
| `programs[].lower_time_seconds`, `compile_time_seconds` | ahead-of-time lowering and compilation of the epoch program | `jax.jit(...).lower().compile()` |
| `programs[].flops`, `bytes_accessed`, `peak_memory_bytes` | compiler estimate for **one epoch**; deterministic for a given program + shapes | `Compiled.cost_analysis()`, `memory_analysis()` |
| `summary.total_compiles`, `total_compile_seconds` | over the whole run | monitoring hook |

## Measurement protocol

1. Same config, same seed, same GPU, nothing else on the GPU (`nvidia-smi`).
2. Choose `--num_evals` so that an epoch takes 5–10 s and there are at least 8
   steady epochs. Fewer and the IQR is meaningless.
3. Compare median steady SPS, not mean; GPU clocks make the distribution
   right-skewed.
4. `total_compiles` must match between baseline and candidate. If it does not,
   something recompiles; use `--log_compiles` to see what, and fix that before
   comparing anything else.
5. Trace (`--profile_epochs`) in a separate run, or accept that the traced
   epoch is excluded from the throughput statistics.

## How it is wired in

```
training/performance/
    __init__.py       public surface: install / uninstall / get_tracker, NullTracker, PerformanceTracker
    tracker.py        PerformanceTracker (phases, epochs, compile listener, summary), NullTracker
    aot.py            ahead_of_time_compile(): lower/compile timing + cost/memory analysis
    trace.py          trace_window(): jax.profiler.start_trace/stop_trace in XProf's directory layout
    sinks.py          WandbSink (summary, files, trace artifact), JsonSink
    fetch_traces.py   CLI: pull xprof-trace artifacts from W&B into one XProf logdir and open it
```

* `run_utils.require_wandb_login()` runs in every entry point right after
  `setup_gpu_environment`. It reads stored credentials (`WANDB_API_KEY` or
  `~/.netrc`) without a network call, so it is safe on compute nodes; it accepts
  `WANDB_MODE=offline` and refuses `disabled`/`dryrun`. There is no flag to
  run without W&B. Entity and project are pinned in `training/config.py`
  (`WANDB_ENTITY`, `WANDB_PROJECT`) so every machine logs to the same place.
* Entry points call `run_utils.install_performance_tracker(config, run_name, progress_fn)`
  **after** `wandb.init` and `.finish()` after training. `train_curriculum`
  uses one tracker across all stages (stage-boundary recompiles show up in the
  counts); `transfer.benchmark_safety_transfer` installs one per W&B run.
* `training/agents/ppo/train.py` fetches the tracker with
  `performance.get_tracker()` and calls it unconditionally. Without an
  installed tracker it gets a `NullTracker` and every call is a no-op. Hooks:
  `tracker.phase("environment_reset")`, `tracker.phase("initial_evaluation")`,
  `tracker.compile_epoch_function(training_epoch, ...)` before the first epoch,
  `tracker.epoch(index, environment_steps=...)` around each epoch, and
  `tracker.scope(...)` (= `jax.named_scope`) around rollout / normalizer /
  SGD / post-step / the two `debug.callback`s so profiler tables use those
  names.
* No `train()` signature changed. The algorithm wrappers (ppo_lag, focops, …)
  need nothing.
* CLI flags live in `training/config.py`: `--measure_performance`,
  `--profile_epochs`, `--performance_dir`, `--log_compiles`.

## Known gaps / next steps

* Persistent compilation cache (`JAX_COMPILATION_CACHE_DIR`) is not enabled by
  default. It would remove the ~35 s of compilation from repeated benchmark
  runs; it is recorded in `metadata` when set.
* SAC-family trainers are not instrumented yet (only `ppo/train.py`).
* XProf's web UI loads Google Charts from the internet; charts are blank fully
  offline. The `xprof get_*` CLI does not have that dependency.
