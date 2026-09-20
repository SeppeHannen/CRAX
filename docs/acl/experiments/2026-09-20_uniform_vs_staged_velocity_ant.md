# Manual curriculum vs uniform on `safe_velocity_ant`

Status: code ready and smoke-tested on the GPU; **not yet run**.

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

Both are evaluated on level 3 (`eval/deployment/*`) and on uniform
(`eval/uniform/*`) nine times over the run.

Budget: 50 M env steps (≈ 77 rounds, ~30 min each on an idle GPU) for a first
look; paper scale is 500 M.

```bash
cd ~/tue/graduation/CRAX
COMMON="--env_name safe_velocity_ant --alg ppo_lag --seeds 0 \
  --num_envs 8192 --num_timesteps 50_000_000 --num_evals 9 --episode_length 1000 \
  --unroll_length 20 --batch_size 1024 --num_minibatches 32 --num_updates_per_batch 4 \
  --safety_bound 25 --deployment_distribution level:3 \
  --store_model false --skip_rollout --skip_video --measure_performance \
  --wandb_group velocity_ant_staged_vs_uniform"

.venv/bin/python -m training.train_env $COMMON --context_distribution staged:1,2,3
.venv/bin/python -m training.train_env $COMMON --context_distribution uniform
```

## What to look at

1. `eval/deployment/episode_reward` and `episode_cost` over training, both arms
   on one panel.
2. `curriculum/experienced/velocity_threshold` (heatmap over rounds) next to
   `curriculum/sampled/...`: for uniform, `sampled` should be flat and
   `experienced` probably is not — contexts with longer episodes get more
   gradient. For staged, `experienced` blurs over the two switches while
   `sampled` switches sharply.
3. `performance/epoch_steps_per_second`: are we still fast?

## Results

_(after the run)_
