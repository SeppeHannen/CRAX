# Manual curriculum vs uniform on `safe_velocity_ant`

Status: 50 M pilot done (one seed, 2026-09-20); 500 M run started 2026-09-21.

## Question

Does training on contexts drawn uniformly from Ω reach level 3 as well as
CRAX's manual three-level curriculum does?

## Setup

Two runs, identical except for the training distribution. PPO-Lagrange,
`safe_velocity_ant`, 8192 envs, 1 000-step episodes, cost budget 25.
Ω = `velocity_threshold ∈ [1.049, 2.622]` (L1 = 2.622, L2 = 1.967, L3 = 1.311).

| arm | `--context_distribution` |
|---|---|
| manual curriculum | `staged:1,2,3` (equal split of the rounds) |
| uniform | `uniform` |

Both are evaluated on level 3 (`evaluation/deployment/*`) and on uniform
(`evaluation/uniform/*`) nine times over the run.

**The policy cannot see the context.** In this benchmark the velocity threshold
enters only the cost; the observation is the plain Ant observation and the
reward is Ant's. A frozen policy therefore produces the same trajectories under
every threshold, so its evaluation *reward* is the same on `deployment` and
`uniform` (up to eval noise) and only its *cost* differs. The student can learn
one speed, not a speed per context. This is CRAX's design (a context-agnostic
student); putting ω in the observation would change the benchmark's observation
space and is a decision to take with Tristan, not something to slip in here.

Budget: 50 M env steps as a pilot (≈ 77 rounds, ~5 min each at ~360 k SPS on an
idle GPU), then paper scale 500 M (≈ 763 rounds, ~40 min each).

```bash
cd ~/tue/graduation/CRAX
COMMON="--env_name safe_velocity_ant --alg ppo_lag --seeds 0 \
  --num_envs 8192 --num_timesteps 500_000_000 --num_evals 21 --episode_length 1000 \
  --unroll_length 20 --batch_size 1024 --num_minibatches 32 --num_updates_per_batch 4 \
  --safety_bound 25 --deployment_distribution level:3 \
  --store_model false --skip_rollout --skip_video --measure_performance \
  --wandb_group velocity_ant_staged_vs_uniform"

.venv/bin/python -m training.train_env $COMMON --context_distribution staged:1,2,3
.venv/bin/python -m training.train_env $COMMON --context_distribution uniform
```

## What to look at

1. `evaluation/deployment/episode_reward` and `episode_cost` over training, both arms
   on one panel.
2. `training_curriculum/experienced/velocity_threshold` (heatmap over rounds) next to
   `training_curriculum/sampled/...`: for uniform, `sampled` should be flat and
   `experienced` probably is not — contexts with longer episodes get more
   gradient. For staged, `experienced` blurs over the two switches while
   `sampled` switches sharply.
3. `performance/epoch_steps_per_second`: are we still fast?

## Results

### 50 M pilot (one seed, W&B group `velocity_ant_staged_vs_uniform`, runs `..._2323..` and `..._2328..`)

Throughput: 359 k / 360 k SPS median (staged / uniform), 80 rounds, no
recompiles. Reset-on-done every step and one compiled call per round cost
nothing measurable.

Evaluation on level 3 (budget 25):

| env steps (M) | staged reward | staged cost | uniform reward | uniform cost |
|---|---|---|---|---|
| 19.7 | 17.0 | 646 | 0.05 | 0.2 |
| 39.3 | 24.0 | 964 | 6.5 | 0.5 |
| 45.9 | 8.1 | 26 | 15.3 | 0.4 |
| 52.4 | 13.2 | 0.6 | 19.3 | 65 |

Not converged; the curves cross at the end. Staged runs fast on L1 for two
thirds of the budget (cost ~950 on L3 because it never sees the L3 threshold),
then the multiplier pulls it back hard after the switch at round 51. Uniform
sits still for 40 M steps (any speed is punished under the strictest sampled
thresholds), then starts moving late. The staged run's `experienced` mean lags
its `sampled` mean by a few rounds at each switch, as predicted.

### 500 M (paper scale, one seed, W&B group `velocity_ant_staged_vs_uniform_500M`)

763 rounds each, 28 min each, 371 k SPS median. Staged switches at rounds 254
and 509 (≈ 167 M and 334 M steps).

Evaluation on level 3 (budget 25), selected points:

| env steps (M) | staged reward | staged cost | λ (staged) | uniform reward | uniform cost |
|---|---|---|---|---|---|
| 51 | 31.7 | 991 | 0.8 | 13.6 | 0.2 |
| 153 | 34.2 | 992 | 1.5 | 19.2 | 0.2 |
| 205 | 0.00 | 3.4 | **405** | 20.6 | 5.2 |
| 306 | 0.03 | 0.4 | 373 | 20.7 | 10.3 |
| 409 | 0.05 | 0.05 | 339 | 18.9 | 0.03 |
| 511 (end) | 0.06 | 0.04 | ~300 | 20.2 | 0.5 |

**Uniform solves level 3; the manual curriculum ends with a policy that does not
move.** Uniform reaches reward ≈ 20 by 100 M and holds it, cost on level 3
between 0 and 18, under budget. Its reward is below the staged run's L1 peak
(34) because it settles on the fastest gait that is safe at the *strict* end of
Ω (thresholds down to 1.05, below level 3).

Why staged fails, from the training-side metrics:

1. **Level 1 teaches the wrong skill.** For 254 rounds the student learns to run
   fast (forward reward 2 400, training-episode cost ≈ 200 under the loose
   threshold). Evaluated on level 3 that gait costs ~980 per 1 000 steps. For a
   policy that cannot see the context, "easier level" is not a sub-task of the
   harder one; it is a contradictory one.
2. **The switch to level 2 blows up the Lagrange multiplier.** The same gait now
   violates almost every step; λ goes 1.5 → 405 within a few rounds. With λ that
   large the only way to lower the loss is to stop: training episodes drop to
   ~10 steps (the Ant falls or freezes), cost 0.
3. **It never recovers.** With cost 0 the violation is −0.025 per step, so λ
   decays at ~0.35 per round: 405 → 320 by the end. Hundreds of rounds too slow.

`sampled` vs `experienced` behaved as predicted: at the first switch
`experienced` drifts from 2.62 to 2.15 over 8 rounds while `sampled` steps
immediately; after that they coincide because training episodes are short.

Two things this is *not*: it is not evidence that curricula don't help (one
seed, one suite, one algorithm); and it is not a curriculum-only effect — it is
the Lagrange multiplier failing under a sudden shift of the cost regime. Note
that CRAX's stock `train_curriculum.py` restarts `train()` per level, which
resets λ to 0 at each switch and hides exactly this; our single-run version does
not. Worth raising with Tristan: the paper's curriculum results include that
reset.

### What this changes in the plan

- **The policy must be able to see the context.** A context-agnostic student can
  only learn one behaviour for all of Ω; every curriculum then trains it on the
  wrong task for part of the budget. Make the learner contextual (ω in the
  observation): a contextual CMDP.
- **PPO-Lagrange enforces the budget in expectation and its dual variable
  carries state about the past distribution.** Try PPO-Saute, which puts the
  remaining safety budget in the state and enforces the constraint per episode.
- **The W&B run page is too dense to read** (50 keys under `training_curriculum`,
  82 under `evaluation`). Needs a deliberate dashboard.
