# Measurement log — 2026-09-19 — safe_goal_point L1 / PPO-Lag baseline

Machine: RTX 4080 SUPER 16 GB, jax 0.10.1 + cuda12 plugin, mujoco/mjx 3.11.0,
`XLA_FLAGS=--xla_gpu_triton_gemm_any=True`, git `e0c441ce` + uncommitted
`training/performance` instrumentation. Nothing else on the GPU.

Config (both runs): `--alg ppo_lag --difficulty 1 --episode_length 1000
--unroll_length 20 --batch_size 1024 --num_minibatches 32 --num_updates_per_batch 4
--num_timesteps 7e6 --num_evals 12` → 11 epochs × 655,360 env steps.
Runs: `runs/performance/…_20260919_235335_*` (2048 envs, epoch 6 traced) and
`…_20260920_000133_*` (8192 envs).

## Results

| | 2048 envs | 8192 envs |
|---|---|---|
| steady SPS (median) | **112 k** | **228 k** |
| steady SPS (IQR) | 31 k (noisy) | 3 k (stable) |
| steady epoch wall (median) | 5.83 s | 2.88 s |
| epoch program: lower + compile | 3.2 + 14.5 s | ≈ same |
| environment reset (JIT + run) | 7.7 s | 8.3 s |
| initial evaluation (JIT + run) | 24.8 s | 27.3 s |
| total compiles / compile time | 106 / 33.7 s | 96 / 34.8 s |
| program FLOPs per epoch | 7.04e10 | 7.13e10 (+1.3 %) |
| program bytes per epoch | 6.2e9 | 9.6e9 (+54 %) |
| program peak memory | 1.1 GiB | 1.7 GiB |
| total wall (11 epochs) | 218 s | 192 s |

Same amount of work per epoch (FLOPs within 1.3 %), twice the throughput and a
tenth of the variance just by batching 4× more environments per kernel.

## What the trace of epoch 6 (2048 envs, 20 s incl. tracing overhead) says

`xprof get_kernel_stats`: **1,277,421 kernel launches** in one epoch,
554 distinct kernels, **mean kernel duration 2.3 µs**, device busy ≈ 2.97 s.

| kernel duration | launches | device time |
|---|---|---|
| < 2 µs | 1,078,120 | 1.14 s |
| 2–10 µs | 162,367 | 0.63 s |
| 10–50 µs | 33,401 | 0.95 s |
| ≥ 50 µs | 3,533 | 0.26 s |

84 % of launches are sub-2 µs kernels; a CUDA launch costs roughly 2–5 µs of
CPU/driver time, so **the epoch is launch-bound, not compute-bound**. XProf's
`get_top_hlo_ops` reports ~6.0 s IDLE vs ~0.17 s attributable op self-time and a
roofline utilisation of 0.08 %. The GPU is mostly waiting for the host to enqueue
the next tiny kernel.

Where the launches come from (top HLO ops by self time, all inside the
`rollout` scope): MJX `triangular-solve` (smooth.py:380, 18,920 + 18,920 +
7×5,120 calls), `cholesky` (smooth.py:320, 2×5,120), the constraint solver
`while` loop (solver.py:570/602), then PPO loss fusions (losses.py:109/112) and
`acting.py:73` dynamic-update-slices. I.e. the per-step physics — many small
linear-algebra custom calls sequenced inside `scan(actor_step)` — dominates the
launch count; the SGD part shows up as a few large GEMMs (72–82 µs each) that
are the only kernels with meaningful occupancy.

The two `jax.debug.callback`s: `debug_callback.4` (train.py:610, environment
metrics) appears 12× per epoch at 7.4 ms self time total — **negligible** at
this scale. Hypothesis from the design doc is falsified; no action needed on the
callbacks for throughput.

## Consequences for the ACL work

1. **Batch size is the throughput knob.** With ~2 µs kernels, throughput scales
   almost linearly with `num_envs` until kernels get large enough to fill the
   GPU. 8192 envs is still far from that point (1.7 GiB of 16 GiB). Any ACL
   design that shrinks the effective batch (e.g. running one context at a time
   with fewer envs) pays for it directly; one that keeps all `num_envs`
   in a single vmapped program (different contexts as *array values* across the
   batch axis) is free.
2. **Recompiles are expensive but rare**: ~18 s for the epoch program, ~8 s for
   reset, ~25 s for the evaluator, ≈ 50 s per "new program" total. A curriculum
   that changes shapes/structure per stage pays that per stage; one that changes
   only values pays nothing. Both measured, not guessed.
3. The remaining compile count (~100 small compiles, ≈ 0.1 s each) is JAX's
   lazy compilation of tiny helpers; harmless.

## Instrumentation notes

- Tracing inflated the traced epoch from ~5.8 s to 20 s (CUPTI overhead on
  1.3 M launches). Traced epochs are excluded from steady-state statistics; keep
  tracing to one epoch per run.
- Trace size 210 MB for one epoch. Fine locally; don't upload to W&B by default.
- First attempt of the 2048-env run died at process exit with
  `CUDA_ERROR_ILLEGAL_ADDRESS` while destroying CUDA events (58 repeated lines,
  after training had finished). Second identical run was clean. Not
  reproduced yet; note in case it recurs — likely teardown ordering between the
  profiler/CUPTI and XLA, not a training bug.
- `--quiet` also silences the `[performance]` lines; the JSON is the source of
  truth.

## Reproduce

```bash
python -m training.train_env --env_name safe_goal_point --alg ppo_lag --difficulty 1 --seeds 0 \
  --num_envs 2048 --num_timesteps 7_000_000 --num_evals 12 --episode_length 1000 \
  --unroll_length 20 --batch_size 1024 --num_minibatches 32 --num_updates_per_batch 4 \
  --num_eval_envs 128 --store_model false --skip_rollout --skip_video --wandb_tags perf \
  --measure_performance --profile_epochs 6
# compare runs in the W&B Run Comparer (performance/* summary); traces:
python -m training.performance.fetch_traces <run_name> --no-open
xprof get_kernel_stats runs/traces/plugins/profile/<run_name> --limit 100000 | jq ...
```
