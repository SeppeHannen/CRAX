# Automated curriculum learning on CRAX

Giuseppe Hannen's graduation project. Start here.

## The plan

1. **Profile cleanly.** XProf traces and per-round throughput go to W&B for
   every run, so we always know whether training is still fast. Done:
   `training/performance/`, guide in `performance_measurement.md`.
2. **Manual curriculum vs uniform.** On `safe_velocity_ant`, PPO-Lagrange, level
   1 → 2 → 3 vs contexts drawn uniformly from Ω, compared on level 3. Done, one
   seed at 500 M steps: uniform solves level 3, the curriculum collapses when the
   level switches (Lagrange multiplier blow-up) and never recovers. Write-up:
   `experiments/2026-09-20_uniform_vs_staged_velocity_ant.md`.
3. **A W&B dashboard one can read.** The run page has ~50 keys under
   `training_curriculum` and ~80 under `evaluation`. Decide what the few panels
   are that answer "did it work, and what did it train on", and make the rest
   secondary. Open: how exactly.
4. **Make the learner contextual.** Put ω in the observation so the policy knows
   which task it is in (a contextual CMDP). Changes the benchmark's observation
   space — agree with Tristan first.
5. **Repeat 2 with PPO-Saute.** It augments the state with the remaining safety
   budget and enforces the constraint per episode rather than in expectation,
   and has no dual variable that carries the old distribution across a switch.
6. **Then** actual curriculum methods, and more suites.

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
(`evaluation/deployment/*`, level 3 by default) and on uniform (`evaluation/uniform/*`),
and logs the sampled and experienced context distribution every round
(`training_curriculum/*`, as W&B histograms).

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
