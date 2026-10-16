"""The 500 M learners experiment again, with the policy seeing the scene: pixels next to the
sensors (``--vision --vision_obs_mode pixels+state``). Five training distributions under three
learners on safe_goal_point; the per-context multiplier λ(ω) in its two function classes with the
policy told its context, against the scalar PPO-Lagrange baseline with a blind policy
(docs/acl/design/contextual_learner.md). Every evaluation records four episodes with the frames
the policy saw, and every checkpoint carries the whole learner, so an episode can be reviewed or
replayed afterwards (docs/acl/design/episode_review.md).

Batch shape and learning rates are the vision pilot's (group vision_pilot_level3_100M: 1024
environments, 4 minibatches), which learned level 3 from pixels+state in 100 M steps at ~55 min
per 100 M on an A100.

    .venv/bin/python -m training.experiments.snellius scripts/experiments/goal_point_learners_vision_500M.py \\
        --partition gpu_a100 --time 06:00:00
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

_COMMON = (
    "--env_name", "safe_goal_point", "--alg", "ppo_lag",
    "--deployment_distribution", "level:3",
    "--initial_lambda_lagr", "0.1", "--lagrangian_coef_rate", "1",
    "--vision", "--vision_obs_mode", "pixels+state",
    "--num_envs", "1024", "--episode_length", "1000",
    "--unroll_length", "20", "--batch_size", "1024", "--num_minibatches", "4", "--num_updates_per_batch", "4",
    "--record_episodes", "4", "--store_model", "true",
    "--measure_performance", "--skip_rollout", "--skip_video",
)

EXPERIMENT = Experiment(
    group="goal_point_learners_vision_500M",
    common=(*_COMMON, "--num_timesteps", "500_000_000", "--num_evals", "21"),
    arms=tuple(
        Arm(f"{distribution_label}_{learner_label}", ("--context_distribution", distribution, *learner_flags))
        for distribution_label, distribution in DISTRIBUTIONS.items()
        for learner_label, learner_flags in LEARNERS.items()
    ),
)
