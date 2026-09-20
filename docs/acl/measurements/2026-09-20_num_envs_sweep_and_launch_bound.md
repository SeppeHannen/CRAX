# 2026-09-20 — num_envs sweep (Point) and what "launch-bound" means

Status: Point complete; Ant and Humanoid **not yet run** (sweep crashed on a
tracker bug, since fixed; rerun with
`python scripts/sweep_num_envs.py --envs safe_velocity_ant safe_velocity_humanoid`).

Machine: RTX 4080 SUPER 16 GB, jax 0.10.1, mujoco 3.11.0, git `0c9988c`+.
Config: PPO-Lag, `safe_goal_point` L1, `unroll_length=20`, `batch_size=1024`,
`num_minibatches=32`, `num_updates_per_batch=4`, 8 steady epochs of 655 360 env
steps. Runs are in W&B project `crax`, group/tag `sweep-num-envs`.

## Result: Point

| num_envs | unrolls / training step | steady SPS (median) | IQR | peak device mem |
|---:|---:|---:|---:|---:|
| 2048 | 16 | 130 k | 16 k | 1.1 GiB |
| 4096 | 8 | 201 k | 19 k | 1.3 GiB |
| **8192** | **4** | **243 k** | 8 k | 1.7 GiB |
| 16384 | 2 | 232 k | 5 k | 2.8 GiB |
| 32768 | 1 | 203 k | 3 k | 4.2 GiB |

**Operating point for Point-class environments on this card: `num_envs = 8192`.**
Above it throughput *falls*, not just flattens. Memory was never the constraint
(4.2 of 16 GiB at the largest setting).

Cross-check: the CRAX paper (§5.3, Fig. 5) reports ~300 k simulation-only SPS at
~8192 envs for the same task on an **H100**. Same knee, ~25 % more throughput on
a GPU with 2–3× the raw resources. That is itself evidence for the explanation
below: the limit is not GPU compute.

## Why: the epoch is launch-bound

Plain-language version, because it took a while to get right in conversation.

- The GPU cannot run the program by itself. The CPU hands it one small
  instruction at a time; each handoff costs the CPU a fixed ~3 µs regardless of
  how much work the instruction contains.
- One physics tick of MJX is ~30 such instructions that must run in order (step
  t+1 needs step t). Per epoch: 16 unrolls × 20 env steps × 4 physics ticks × ~30
  × 2 training steps ≈ **2 million handoffs**. Measured: 2.05 M kernel launches,
  mean kernel 2.2 µs (trace of 2026-09-20).
- At 2048 envs the GPU finishes each instruction (~2 µs) *before* the CPU has
  finished handing over the next (~3 µs). So the GPU idles ~60 % of the time.
  Measured: device compute 4.6 s vs "all other" 7.2 s in an 11.8 s traced epoch.
- **There is no backlog for the GPU to draw on.** A queue only fills when the
  producer is faster than the consumer; here the CPU is the slower side, and
  the instructions are sequentially dependent anyway.
- Consequently the only free variable is **how many environments ride in each
  instruction**. Going 2048 → 8192 makes each instruction 4× heavier, but the
  GPU was idle, so the epoch takes the same wall time and yields 4× the data.
  Past ~8192 the instruction itself takes longer than the handoff, the GPU
  becomes the slower side, and adding environments adds time.
- Why it *drops* rather than plateaus above the knee is not yet measured (one
  trace at 32768 vs 8192 would settle it). Candidates: memory bandwidth (XProf
  roofline already said "bound by HBM"; `program_bytes_accessed` +54 % from
  2048 → 8192), and loss of overlap with only 1–2 unrolls per training step.

Things this rules out as levers:
- Bigger SGD batch: SGD is ~7 % of the epoch and already GPU-busy; the idling is
  inside the physics chain, where nothing is being "gathered" during the gaps.
- Bigger GPU: an H100 finishes each instruction sooner and waits longer.
- Faster Python: Python is not in the loop; the whole epoch is one compiled
  program.

## Consequences for the ACL work

1. `num_envs = 8192` for Point-class experiments locally; expect the cluster GPU
   to have the same knee (per-GPU speed will be similar; the cluster's value is
   running many runs in parallel). Redo the sweep there once as a check.
2. Contexts must be **array values on the `num_envs` axis**, not shapes. A value
   change is free; a shape change is a new compiled program (~50 s).
3. Adding a *new instruction per physics step* (e.g. an extra `lax.cond` or a
   small per-step op in the env) costs ≈ 3 µs × 2 M ≈ 6 s per epoch — visible.
   Folding a context into an existing operation (multiply into an existing
   array) costs nothing. Check with `--measure_performance`: `program_flops`
   should barely move and steady SPS should not drop.
4. Per the paper's Appendix A.4, five of nine suites vary difficulty through a
   pure value already (Push goal velocity, Velocity threshold, Height max height,
   Pathway max gap, SpiderLegs foot mask). Goal, Reach, Circle vary hazard
   *count* — a shape — and would need "pad to max hazards + mask" to become
   value-parameterised.

## Deferred experiment: CUDA graphs

Giuseppe's question — "if the CPU knows the sequence, why can't the GPU?" — is
exactly what CUDA graphs (XLA: "command buffers") do: record the instruction
sequence once, replay it as a single handoff. Off by default in JAX because not
every instruction type is recordable; MJX's `cholesky`/`triangular-solve` are
library custom calls, historically the awkward case.

Experiment (≈10 min GPU), not yet run:

```bash
XLA_FLAGS="--xla_gpu_triton_gemm_any=True --xla_gpu_enable_command_buffer=FUSION,CUBLAS,CUSTOM_CALL,CUDNN,WHILE" \
python -m training.train_env --env_name safe_goal_point --alg ppo_lag --difficulty 1 \
  --num_envs 8192 --num_timesteps 5_242_880 --num_evals 9 --episode_length 1000 \
  --unroll_length 20 --batch_size 1024 --num_minibatches 32 --num_updates_per_batch 4 \
  --store_model false --skip_rollout --skip_video --measure_performance \
  --wandb_tags command-buffer
```

Compare `performance/steady_steps_per_second_median` against the 8192 sweep run
(243 k). Outcomes: large gain (the single biggest free speedup available), no
change (custom calls not captured), or a compile error (drop `CUSTOM_CALL` from
the list and retry). MuJoCo Warp, which CRAX already supports for vision, is the
other route to fewer handoffs; heavier to adopt.

## Also worth asking Tristan

Paper Appendix B.3.4 lists Point's control timestep as 0.008 s; the code uses
`timestep=0.02 × n_frames=4 = 0.08 s`. One of them is off by 10×.
