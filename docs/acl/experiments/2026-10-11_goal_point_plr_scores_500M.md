# Uniform vs staged vs PLR × {reward, cost, Lagrangian} on `safe_goal_point`, 500 M steps

**Status: planned, awaiting Giuseppe's go. Not run.** W&B group
`goal_point_plr_scores_500M`. One seed. Plan item 2: the three PLR score
variants against the two manual baselines, on the same stack as the
2026-10-08 experiment plus the third evaluation target Ω₂ (placeholder values,
`design/goal_point_out_of_distribution.md`).

## Question

The 2026-10-08 run found the reward critic about equally wrong across Ω
(score max/mean 1.6) while the cost critic's error varied 4×. Does ranking by
the cost critic's error, or by the error of the Lagrangian objective the policy
actually optimises, move PLR's mass further towards the dense end of Ω than
the reward score did — and does that translate into lower cost on level 3
without the return dropping? Secondary: with Ω₂ evaluated on every arm, how do
the five policies rank on contexts none of them could have trained on?

## Setup

Identical to 2026-10-08 except for the three PLR arms and the Ω₂ evaluator.
PPO-Lagrange, `safe_goal_point`, 8192 envs, 1000-step episodes, budget 25,
763 rounds of 655 360 steps, 21 evaluations of 128 episodes on level 3, on
Uniform(Ω) and on Uniform(Ω₂).

| arm | `--context_distribution` | PLR score (mean \|·\| over the episode) |
|---|---|---|
| uniform | `uniform` | — |
| staged | `staged:1,2,3` | — |
| plr | `plr` | reward advantage (the paper) |
| plr:cost | `plr:cost` | cost advantage |
| plr:lagrangian | `plr:lagrangian` | $A_r - \lambda A_c$, both standardised, the round's λ |

PLR hyperparameters as before: buffer 1000, β = 1, ρ = 0.3, p = 0.5.

```bash
COMMON="--env_name safe_goal_point --alg ppo_lag --seeds 0 \
  --num_envs 8192 --num_timesteps 500_000_000 --num_evals 21 --episode_length 1000 \
  --unroll_length 20 --batch_size 1024 --num_minibatches 32 --num_updates_per_batch 4 \
  --safety_bound 25 --deployment_distribution level:3 \
  --store_model false --skip_rollout --skip_video --measure_performance --quiet \
  --wandb_group goal_point_plr_scores_500M"
for ARM in plr:lagrangian plr:cost plr uniform staged:1,2,3; do
  nohup .venv/bin/python -m training.train_env $COMMON --context_distribution $ARM \
    > runs/logs/goal_point_plr_scores_500M_${ARM//[:,]/}.log 2>&1
done
```

Order: the two new PLR variants first (the arms the question is about), then
the reward PLR as the bridge to the previous run, then the baselines. ~2 h per
arm at 85 k SPS; ~10 h total. Each arm `nohup`'d; the chain in one script so a
dropped terminal does not kill it.

Before launching: a 10-round smoke of `plr:lagrangian` to confirm 0 recompiles
and that `training_curriculum/lagrangian_value_loss/<d>` is populated.

## Results

*(to fill: three figures — evaluation on the three targets; training λ / cost;
PLR intended mass and score landscape per variant — each with 3–5 bullets.)*

## What this changes in the plan

*(to fill.)*
