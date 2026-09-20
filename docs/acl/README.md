# Automated curriculum learning on CRAX

Giuseppe Hannen's graduation project. Start here.

## The plan

1. **Profile cleanly.** XProf traces and per-round throughput go to W&B for
   every run, so we always know whether training is still fast. Done:
   `training/performance/`, guide in `performance_measurement.md`.
2. **Manual curriculum vs uniform.** On `safe_velocity_ant`, train PPO-Lagrange
   with the three-level curriculum (level 1 → 2 → 3) and with contexts drawn
   uniformly from Ω, and compare their performance on level 3. Code is done and
   tested; the run is next.
3. **Then** actual curriculum methods, and more suites.

## Vocabulary

| term | meaning |
|---|---|
| context ω ∈ Ω | the per-episode task parameters of a suite (Ant: the velocity threshold). A value on the `num_envs` axis, never a shape. |
| distribution | decides which ω each parallel environment gets when its episode starts. `uniform`, `level:3`, `staged:1,2,3`. |
| round | one PPO training step; with a context distribution, one compiled call. |
| sampled vs experienced | which contexts were *chosen* (per episode) vs which the gradient *came from* (per transition). They differ when episode length depends on the context. Always shown together. |

## Running

```bash
# manual curriculum
.venv/bin/python -m training.train_env --env_name safe_velocity_ant --alg ppo_lag \
  --context_distribution staged:1,2,3 --deployment_distribution level:3 \
  --num_envs 8192 --num_timesteps 50_000_000 --num_evals 9 --episode_length 1000 \
  --unroll_length 20 --batch_size 1024 --num_minibatches 32 --num_updates_per_batch 4 \
  --measure_performance --skip_rollout --skip_video --store_model false

# uniform: same, with --context_distribution uniform
```

Every context run is evaluated on the deployment distribution
(`eval/deployment/*`, level 3 by default) and on uniform (`eval/uniform/*`),
and logs the sampled and experienced context distribution every round
(`curriculum/*`, as W&B histograms).

CPU tests: `JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/test_contexts.py tests/test_context_training.py -q -p no:cacheprovider`

## Rules

- No GPU runs without asking; the machine is shared.
- W&B is mandatory (`WANDB_API_KEY` in `~/.bashrc`).
- Contexts are values, not shapes: a new compiled program costs ~50 s.
- Anything that changes the upstream benchmark's behaviour is flagged to Tristan.
- Clear names, no abbreviations, one file per distribution.

## Documents

```
docs/acl/
  README.md                          this file
  performance_measurement.md         how to profile a run (XProf + W&B)
  measurements/                      dated measurement logs
  experiments/                       one file per experiment
  design/
    contexts_package.md              how training/contexts works and how it is wired into the trainer
    intended_vs_realised_curriculum.md   why sampled ≠ experienced, and why both are reported
    training_round.md                why one compiled call is one training step
    per_slot_constraints.md          what may differ between parallel environments
    context_spaces_by_suite.md       Ω for each of the nine suites
  profiling_options.md               tool survey (historical)
```
