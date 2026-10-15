#!/usr/bin/env bash
# Smoke of the contextual learner with λ_ψ(ω) as a function (docs/acl/design/contextual_learner.md):
# three arms at 100 M steps on safe_goal_point, all `network told`, one after the other.
# (Ran 2026-10-14 under the name `per_context`; the `--constraint` value was renamed when the additive form arrived.)
#   .venv/bin/python -m scripts.analysis.read_console_log runs/logs/smoke_contextual_learner_v2_<arm>_*.log
# Run detached:  setsid nohup scripts/experiments/smoke_contextual_learner_v2.sh > runs/logs/smoke_contextual_learner_v2_chain.log 2>&1 < /dev/null & disown
set -euo pipefail
cd "$(dirname "$0")/../.."

GROUP=smoke_contextual_learner_v2
COMMON=(
  --env_name safe_goal_point --alg ppo_lag
  --deployment_distribution level:3
  --constraint network --policy_context told --initial_lambda_lagr 0.1
  --num_envs 8192 --num_timesteps 100_000_000 --num_evals 9 --episode_length 1000
  --unroll_length 20 --batch_size 1024 --num_minibatches 32 --num_updates_per_batch 4
  --lagrangian_coef_rate 1
  --measure_performance --skip_rollout --skip_video --store_model false
  --wandb_group "$GROUP"
)

for distribution in staged:1,2,3 uniform plr; do
  label=$(echo "$distribution" | tr -d ':,')
  log="runs/logs/${GROUP}_${label}_$(date +%Y%m%d_%H%M%S).log"
  echo "$(date +%T) starting $distribution -> $log"
  .venv/bin/python -m training.train_env --context_distribution "$distribution" "${COMMON[@]}" > "$log" 2>&1
  echo "$(date +%T) finished $distribution (exit $?)"
done
