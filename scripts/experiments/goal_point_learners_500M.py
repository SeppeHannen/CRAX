"""Five training distributions under three learners on safe_goal_point, 500 M steps:
the scalar PPO-Lagrange baseline with a blind policy, and the per-context multiplier
λ(ω) in its two function classes with the policy told its context
(docs/acl/design/contextual_learner.md). One W&B group; the dashboard groups lines by arm.

    .venv/bin/python -m training.experiments.snellius scripts/experiments/goal_point_learners_500M.py \\
        --partition gpu_a100 --time 05:00:00

Four arms added 2026-11-05 complete the 2 × 2 of constraint × policy view on `uniform`
and `plr` (the two off-diagonal learners). Submit them alone with:

    .venv/bin/python -m training.experiments.snellius scripts/experiments/goal_point_learners_500M.py \\
        --partition gpu_a100 --time 05:00:00 \\
        --only uniform_scalar_told plr_scalar_told uniform_additive_blind plr_additive_blind
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
    "scalar_told": ("--constraint", "scalar", "--policy_context", "told"),
    "additive_blind": ("--constraint", "additive", "--policy_context", "blind"),
}
# Every distribution under the three original learners; the two off-diagonal learners on two of them.
ARMS = (
    *((distribution, learner) for distribution in DISTRIBUTIONS for learner in ("scalar_blind", "network_told", "additive_told")),
    *((distribution, learner) for distribution in ("uniform", "plr") for learner in ("scalar_told", "additive_blind")),
)

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
        Arm(f"{distribution}_{learner}", ("--context_distribution", DISTRIBUTIONS[distribution], *LEARNERS[learner]))
        for distribution, learner in ARMS
    ),
)
