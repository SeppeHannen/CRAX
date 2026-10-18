"""The learners experiment from pixels + sensors at the sensors run's round: 1024 environments,
32 minibatches, so a round is 655 360 transitions as on 2026-10-15 and every per-round quantity
(the constraint's cost mean, the advantage normalisation, PLR's scores, the additive fit's one
step) is computed from the same amount of data (``design/training_round.md``, the round's size).

The round's frames are held once (the DrQ translation is applied per minibatch since
2026-10-18), ~48 GB at this size, which needs an H100's 80 GB; ``--gpu_memory_fraction 0.85``
leaves the renderer ~12 GB. Six arms, 500 M steps; the multiplier at rate 1 per round is the
2026-10-15 rate per step again, since the round is the same size.

    .venv/bin/python -m training.experiments.snellius scripts/experiments/goal_point_learners_vision_large_round.py \\
        --partition gpu_h100 --time 08:00:00
"""
from training.experiments import Arm, Experiment

DISTRIBUTIONS = {"staged123": "staged:1,2,3", "uniform": "uniform", "plr": "plr"}
LEARNERS = {
    "scalar_blind": ("--constraint", "scalar", "--policy_context", "blind"),
    "additive_told": ("--constraint", "additive", "--policy_context", "told"),
}
COMMON = (
    "--env_name", "safe_goal_point", "--alg", "ppo_lag",
    "--deployment_distribution", "level:3",
    "--initial_lambda_lagr", "0.1", "--lagrangian_coef_rate", "1",
    "--vision", "--vision_obs_mode", "pixels+state", "--gpu_memory_fraction", "0.85",
    "--num_envs", "1024", "--episode_length", "1000",
    "--unroll_length", "20", "--batch_size", "1024", "--num_minibatches", "32", "--num_updates_per_batch", "4",
    "--record_episodes", "4", "--store_model", "true",
    "--measure_performance", "--skip_rollout", "--skip_video",
)

EXPERIMENT = Experiment(
    group="goal_point_learners_vision_large_round",
    common=(*COMMON, "--num_timesteps", "500_000_000", "--num_evals", "21"),
    arms=tuple(
        Arm(f"{distribution_label}_{learner_label}", ("--context_distribution", distribution, *learner_flags))
        for distribution_label, distribution in DISTRIBUTIONS.items()
        for learner_label, learner_flags in LEARNERS.items()
    ),
)
