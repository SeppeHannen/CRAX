#!/usr/bin/env bash
# The two function classes of the per-context multiplier λ(ω) against each other
# (docs/acl/design/contextual_learner.md §1): staged and uniform under `network` and under
# `additive`, 200 M steps each on safe_goal_point, policy told its context, one after the other.
#   .venv/bin/python -m scripts.analysis.read_console_log runs/logs/goal_point_lambda_forms_200M_<arm>_*.log
# Run detached:
#   setsid nohup scripts/experiments/goal_point_lambda_forms_200M.sh > runs/logs/goal_point_lambda_forms_200M_chain.log 2>&1 < /dev/null & disown
set -euo pipefail
cd "$(dirname "$0")/../.."

GROUP=goal_point_lambda_forms_200M
COMMON=(
  --env_name safe_goal_point --alg ppo_lag
  --deployment_distribution level:3
  --policy_context told --initial_lambda_lagr 0.1
  --num_envs 8192 --num_timesteps 200_000_000 --num_evals 9 --episode_length 1000
  --unroll_length 20 --batch_size 1024 --num_minibatches 32 --num_updates_per_batch 4
  --lagrangian_coef_rate 1
  --measure_performance --skip_rollout --skip_video --store_model false
  --wandb_group "$GROUP"
)

for constraint in network additive; do
  for distribution in staged:1,2,3 uniform; do
    label="$(echo "$distribution" | tr -d ':,')_${constraint}"
    log="runs/logs/${GROUP}_${label}_$(date +%Y%m%d_%H%M%S).log"
    echo "$(date +%T) starting $distribution $constraint -> $log"
    .venv/bin/python -m training.train_env --context_distribution "$distribution" --constraint "$constraint" "${COMMON[@]}" > "$log" 2>&1
    echo "$(date +%T) finished $distribution $constraint (exit $?)"
  done
done
