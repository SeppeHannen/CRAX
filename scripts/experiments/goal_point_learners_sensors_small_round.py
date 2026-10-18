"""The sensors-only learners experiment at the vision runs' round: does the small round alone make
the policy swing between evaluations?

The pixel runs (2026-10-17, -18) trained at 1024 environments and 4 minibatches — an 81 920-step
round, one eighth of the sensors run's — and their policies moved 2–5 goals between evaluations
25 M apart where the sensors run's moved about one (``design/training_round.md``, the round's
size). This run is the sensors-only observation of 2026-10-15 in exactly the pixel runs' batch
shape and multiplier rate, six arms, 500 M steps. If these swing, the round is the cause; if not,
the camera is. The companion is the pixel run at the sensors run's round
(``goal_point_learners_vision_large_round.py``).

    .venv/bin/python -m training.experiments.snellius scripts/experiments/goal_point_learners_sensors_small_round.py \\
        --partition gpu_a100 --time 05:00:00
"""
from training.experiments import Arm, Experiment

DISTRIBUTIONS = {"staged123": "staged:1,2,3", "uniform": "uniform", "plr": "plr"}
LEARNERS = {
    "scalar_blind": ("--constraint", "scalar", "--policy_context", "blind"),
    "additive_told": ("--constraint", "additive", "--policy_context", "told"),
}

EXPERIMENT = Experiment(
    group="goal_point_learners_sensors_small_round",
    common=(
        "--env_name", "safe_goal_point", "--alg", "ppo_lag",
        "--deployment_distribution", "level:3",
        "--initial_lambda_lagr", "0.1", "--lagrangian_coef_rate", "0.125",  # the 2026-10-15 drift per step at this round size
        "--num_envs", "1024", "--episode_length", "1000",
        "--unroll_length", "20", "--batch_size", "1024", "--num_minibatches", "4", "--num_updates_per_batch", "4",
        "--num_timesteps", "500_000_000", "--num_evals", "21",
        "--record_episodes", "4", "--store_model", "true",
        "--measure_performance", "--skip_rollout", "--skip_video",
    ),
    arms=tuple(
        Arm(f"{distribution_label}_{learner_label}", ("--context_distribution", distribution, *learner_flags))
        for distribution_label, distribution in DISTRIBUTIONS.items()
        for learner_label, learner_flags in LEARNERS.items()
    ),
)
