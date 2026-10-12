#!/usr/bin/env bash
# The two new arms of docs/acl/experiments/2026-10-11_goal_point_plr_scores_500M.md,
# run one after the other. The three baseline arms (uniform, staged, plr) are the
# 2026-10-08 v2 runs, moved into this group afterwards (same code path, verified).
#
#   nohup scripts/experiments/goal_point_plr_scores_500M.sh > runs/logs/goal_point_plr_scores_500M_chain.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/../.."

GROUP=goal_point_plr_scores_500M
COMMON=(
  --env_name safe_goal_point --alg ppo_lag --seeds 0
  --num_envs 8192 --num_timesteps 500_000_000 --num_evals 21 --episode_length 1000
  --unroll_length 20 --batch_size 1024 --num_minibatches 32 --num_updates_per_batch 4
  --safety_bound 25 --deployment_distribution level:3
  --store_model false --skip_rollout --skip_video --measure_performance --quiet
  --wandb_group "$GROUP"
)

for ARM in plr:lagrangian plr:cost; do
  LOG="runs/logs/${GROUP}_${ARM//[:,]/}_$(date +%Y%m%d_%H%M%S).log"
  echo "$(date -Is) starting $ARM -> $LOG"
  .venv/bin/python -m training.train_env "${COMMON[@]}" --context_distribution "$ARM" > "$LOG" 2>&1
  echo "$(date -Is) finished $ARM (exit $?)"
done
echo "$(date -Is) chain done"
