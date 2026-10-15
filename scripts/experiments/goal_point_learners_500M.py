"""Five training distributions under three learners on safe_goal_point, 500 M steps:
the scalar PPO-Lagrange baseline with a blind policy, and the per-context multiplier
λ(ω) in its two function classes with the policy told its context
(docs/acl/design/contextual_learner.md). One W&B group; the dashboard groups lines by arm.

    .venv/bin/python -m training.experiments.snellius scripts/experiments/goal_point_learners_500M.py \\
        --partition gpu_a100 --time 05:00:00
"""
from training.experiments import Arm, Experiment

DISTRIBUTIONS = {
    "staged123": "staged:1,2,3",
    "uniform": "uniform",
    "plr": "plr",
    "plrcost": "plr:cost",
    "plrlagrangian": "plr:lagrangian",
}
LEARNERS = {
    "scalar_blind": ("--constraint", "scalar", "--policy_context", "blind"),
    "network_told": ("--constraint", "network", "--policy_context", "told"),
    "additive_told": ("--constraint", "additive", "--policy_context", "told"),
}

EXPERIMENT = Experiment(
    group="goal_point_learners_500M",
    common=(
        "--env_name", "safe_goal_point", "--alg", "ppo_lag",
        "--deployment_distribution", "level:3",
        "--initial_lambda_lagr", "0.1", "--lagrangian_coef_rate", "1",
        "--num_envs", "8192", "--num_timesteps", "500_000_000", "--num_evals", "21", "--episode_length", "1000",
        "--unroll_length", "20", "--batch_size", "1024", "--num_minibatches", "32", "--num_updates_per_batch", "4",
        "--measure_performance", "--skip_rollout", "--skip_video", "--store_model", "true",
    ),
    arms=tuple(
        Arm(f"{distribution_label}_{learner_label}", ("--context_distribution", distribution, *learner_flags))
        for distribution_label, distribution in DISTRIBUTIONS.items()
        for learner_label, learner_flags in LEARNERS.items()
    ),
)
