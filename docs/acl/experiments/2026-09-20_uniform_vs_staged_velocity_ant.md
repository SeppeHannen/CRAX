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

### 500 M (paper scale)

_(running)_
