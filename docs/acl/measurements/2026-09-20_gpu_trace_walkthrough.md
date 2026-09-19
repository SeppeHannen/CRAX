# 2026-09-20 — GPU trace walkthrough (safe_goal_point L1, 2048 envs)

Run: `safe_goal_point_Level_1_ppo_lag_seed0_20260920_004807_797570` in W&B project
`crax` (tags `perf`, `gpu-trace-demo`). Epoch 3 traced. Epochs here are 1.31 M env
steps (2 training steps), i.e. twice yesterday's; steady SPS 114 k, consistent.

Purpose: a representative GPU trace with the new profiler scopes, and a check that
XProf's aggregate pages work once a step marker is present.

## Two fixes that make XProf readable

1. **Step marker.** XProf's Overview Page, step-time breakdown and `get_avg_step_time`
   are computed per "step" and read **0 ms** without one. `PerformanceTracker.epoch`
   now wraps the epoch in `jax.profiler.StepTraceAnnotation("epoch", step_num=i)`,
   so one XProf step = one epoch. The Overview Page now reports:
   average step time 11 769 ms (the traced epoch, inflated by CUPTI), device
   compute 4 586 ms, "all others" 7 183 ms. That 39 %/61 % split *is* the launch
   overhead finding, read directly off the summary page.
2. **Scopes.** `jax.named_scope` added in `acting.actor_step` (`policy_forward`,
   `environment_step`), `PipelineEnv.pipeline_step` (`physics`, shared by every
   env) and `SafeGoal.step` (`safety_cost`, `observation`), on top of the trainer's
   `rollout` / `sgd` / … scopes. They appear as the "Framework Name Scope" rows in
   the Trace Viewer and in the `tf_op_name` path of every HLO op.

## What the GPU trace looks like

Trace Viewer, whole epoch: one device process `/device:GPU:0` with a single compute
stream (`Stream #14`), the nested scope rows (`jit(training_epoch) → vmap → while →
rollout → … → environment_step → physics`), a `Steps` row with `epoch 3`, and a host
process with the `python` thread. The `rollout` bar is a solid block; the `sgd` bar
is a short gap of large kernels between two rollouts. Zooming to ~100 µs shows the
stream as hundreds of thin slivers separated by white space — kernels shorter than
the gaps between them.

Aggregates (`xprof get_hlo_stats`, self-time by our scope, 87 op groups covering
~197 k of the ~2.05 M launches — the long tail of sub-µs kernels is not in this
table, so treat the shares as "of attributable time"):

| scope | share |
|---|---|
| `physics` (MJX: `triangular-solve`, `cholesky`, solver `while`) | **83 %** |
| `sgd` | 7 % |
| `rollout` outside physics (policy forward, transition bookkeeping) | 6 % |
| `environment_metrics_callback` (`debug.callback`) | 3 % |
| everything else, incl. `safety_cost`, `observation`, `post_step` | < 1 % |

`xprof get_kernel_stats`: **2 051 416 kernel launches**, device busy 4 586 ms →
2.2 µs mean. Same shape as yesterday (1.28 M launches for a half-size epoch).

## Reading

- CRAX's own task code (lidar observation, hazard cost, reward) is negligible on
  the device. The env step is MJX physics; the epoch is MJX physics plus launch
  gaps. Anything we add to the *task* side (context parameters, per-env
  difficulty values) is free at this level as long as it doesn't add kernel
  launches per step.
- The lever remains batch size / kernels-per-step, not our Python.
- `safety_cost`/`observation` showing < 1 % also means the finer scopes cost
  nothing and can stay.

## Reading order for a new trace

1. `xprof get_overview <dir>` (or Overview Page): device compute vs "all others".
   If "all others" dominates, it's launch-bound; if device compute dominates,
   look at ops.
2. `xprof get_hlo_stats <dir> --limit 5000` and aggregate `total_self_time_us`
   by scope substring in `tf_op_name` (snippet in this file's history / the
   `training/performance` docs). Tells you *which part of our code*.
3. `xprof get_kernel_stats <dir> --limit 100000`: launch count and mean kernel
   duration. Tells you *how* it is slow.
4. Trace Viewer only to confirm a hypothesis at a specific location.

## Tooling notes

- Trace size 327 MB for a 1.31 M-step epoch. Fetch took seconds; fine.
- `fetch_traces` puts each run in its own XProf session; the Sessions dropdown
  switches between them. The URL-with-query-string form does not survive VS Code's
  port forwarder; use the dropdown.
- The Trace Viewer tab collapses the left sidebar; switch tools from another tab or
  reload `localhost:8791`.
- Perfetto alternative: `runs/traces/plugins/profile/<run>/*.trace.json.gz` opens
  directly in https://ui.perfetto.dev (area-select → aggregate table is often the
  quickest way to get "time by name in this window").
