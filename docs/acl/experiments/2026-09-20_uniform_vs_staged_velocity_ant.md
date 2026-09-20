# Experiment 1 — uniform $r$ vs manual curriculum on `safe_velocity_ant` (T5)

Status: **ready to launch, not yet run** (GPU runs need Giuseppe's go-ahead).
Code path verified on CPU: `tests/test_context_training.py`.

## Question

Does training on the uniform distribution $r = \mathrm{Uniform}(\Omega)$ reach
the deployment task $w$ (level 3) as well as CRAX's manual three-stage
curriculum, and how do their *realised* curricula $\hat q$ differ?

## Arms

All arms: PPO-Lagrange, same budget, same seeds, same evaluation. The
distribution is the only difference. Ω for the Ant velocity suite is
`velocity_threshold ∈ [1.049, 2.622]` (0.4–1.0 × baseline; L1 = 2.622,
L2 = 1.967, L3 = 1.311).

| arm | `--context_distribution` | $q$ | run-name fragment |
|---|---|---|---|
| direct L3 (paper's "Normal") | `level:3` | point mass at L3 | `ctx_level3` |
| curriculum (paper's "Curriculum") | `staged:1,2,3` | L1 → L2 → L3, equal round split | `ctx_staged123` |
| uniform | `uniform` | $r$ | `ctx_uniform` |
| direct L1 (anchor) | `level:1` | point mass at L1 | `ctx_level1` |

All four use the *same* wrapper (`ContextualAutoResetWrapper`, reset on done),
so the comparison is not confounded by the stock wrapper's frozen-layout
replay. A fifth, optional arm `--context_distribution none --difficulty 3` is
the stock CRAX baseline and doubles as the T4 reset-cost measurement.

## Evaluation

Every arm is evaluated `--num_evals` times on

- `eval/deployment/*` — $w$: `--deployment_distribution level:3` (the default);
- `eval/uniform/*` — $r$.

Report `episode_reward` and `episode_cost` under both, against the cost budget
25 (`--safety_bound 25`). Per round, report $q$ (`curriculum/intended/*`,
`curriculum/sampled/velocity_threshold`) next to $\hat q$
(`curriculum/experienced/velocity_threshold`) and the mechanism behind their
gap (`curriculum/episode_length/velocity_threshold/bin_*`) — see
`docs/acl/design/intended_vs_realised_curriculum.md`. Both `sampled` and
`experienced` are `wandb.Histogram`s per round (heatmap over time in W&B) plus
`.../mean` scalars for arm-vs-arm panels grouped by `training_distribution`.

## Budget and operating point

From the sweep (`measurements/2026-09-20_num_envs_sweep_and_launch_bound.md`):
`num_envs = 8192`, `unroll_length 20`, `batch_size 1024`, `num_minibatches 32`
→ 655 360 env steps per round (`steps_per_round`), 1 000-step episodes. With
one compiled call per round the host loop adds ~ms per round.

Pilot (one seed, ~30 min at ≈ 240 k SPS **on an idle GPU** — the 2026-09-20
smoke run on a shared card got 8–54 k): `--num_timesteps 50_000_000` ≈ 77
rounds; `--num_evals 9` → 8 evaluation blocks of 10 rounds (the total is
rounded up to 80 rounds; the staged switches are placed on the requested 77).
Full run: paper-scale `--num_timesteps 500_000_000`, 3 seeds — cluster.

## Commands (pilot)

```bash
cd ~/tue/graduation/CRAX
COMMON="--env_name safe_velocity_ant --alg ppo_lag --seeds 0 \
  --num_envs 8192 --num_timesteps 50_000_000 --num_evals 9 --episode_length 1000 \
  --unroll_length 20 --batch_size 1024 --num_minibatches 32 --num_updates_per_batch 4 \
  --safety_bound 25 --deployment_distribution level:3 \
  --store_model false --skip_rollout --skip_video --measure_performance \
  --wandb_group exp1_velocity_ant --wandb_tags exp1 pilot"

.venv/bin/python -m training.train_env $COMMON --context_distribution level:3
.venv/bin/python -m training.train_env $COMMON --context_distribution staged:1,2,3
.venv/bin/python -m training.train_env $COMMON --context_distribution uniform
.venv/bin/python -m training.train_env $COMMON --context_distribution level:1
# optional: stock wrapper baseline + T4 reset-cost measurement
.venv/bin/python -m training.train_env $COMMON --context_distribution none --difficulty 3
```

`--measure_performance` costs nothing and gives per-round SPS, so the `level:3`
vs `none` pair answers T4 (reset-every-step + one-call-per-round cost) for
free — provided both run back-to-back on an idle GPU.

## What to look at first

1. `eval/deployment/episode_reward` and `episode_cost` at the final evaluation,
   uniform vs staged vs level 3. Same $w$ for all: directly comparable.
2. `curriculum/experienced/velocity_threshold` (heatmap) and `.../mean` for the
   uniform arm, next to `curriculum/sampled/...` — `sampled` should be flat,
   `experienced` is expected *not* to be: whichever thresholds yield longer
   episodes are over-exposed. That deviation is a result in itself.
3. The staged arm's `experienced` heatmap around rounds 26 and 51 (the
   switches): a blend over ~4 rounds, not a step; `sampled` switches sharply.
4. `eval/uniform/*` for the level-3 arm: how much a policy trained only on $w$
   loses on the rest of Ω (generalisation cost of specialising).
5. `performance/epoch_steps_per_second`, `level 3` vs `none` (T4).

## Results

_(to be filled in after the pilot)_
