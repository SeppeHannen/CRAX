"""Three distributions under two learners on safe_goal_point from pixels + sensors, 750 M steps,
with the multiplier's dual ascent at the rate per environment step of the sensors-only run.

The 2026-10-17 run of all fifteen arms from pixels oscillated between evaluations: at 1024
environments a round is 81 920 steps instead of 655 360, and ``--lagrangian_coef_rate`` is
applied once per round, so the same rate 1 moved λ eight times as far per environment step
(``docs/acl/experiments/2026-10-17_goal_point_learners_vision_500M.md``). This run keeps the
batch shape the renderer's memory allows and passes rate 1/8, which is exactly the 2026-10-15
dual ascent per step; everything that still oscillates is then the observation's doing. Six
arms instead of fifteen (uniform, PLR reward, staged; the published scalar-blind learner and
the additive told one), 750 M steps with evaluations 25 M apart as before (31 evaluations).

    .venv/bin/python -m training.experiments.snellius scripts/experiments/goal_point_learners_vision_750M.py \\
        --partition gpu_a100 --time 10:00:00
"""
from training.experiments import Arm, Experiment

DISTRIBUTIONS = {
    "staged123": "staged:1,2,3",
    "uniform": "uniform",
    "plr": "plr",
}
LEARNERS = {
    "scalar_blind": ("--constraint", "scalar", "--policy_context", "blind"),
    "additive_told": ("--constraint", "additive", "--policy_context", "told"),
}

EXPERIMENT = Experiment(
    group="goal_point_learners_vision_750M",
    common=(
        "--env_name", "safe_goal_point", "--alg", "ppo_lag",
        "--deployment_distribution", "level:3",
        # rate 1 per 655 360-step round on 2026-10-15 = rate 1/8 per 81 920-step round here
        "--initial_lambda_lagr", "0.1", "--lagrangian_coef_rate", "0.125",
        "--vision", "--vision_obs_mode", "pixels+state", "--gpu_memory_fraction", "0.75",
        "--num_envs", "1024", "--episode_length", "1000",
        "--unroll_length", "20", "--batch_size", "1024", "--num_minibatches", "4", "--num_updates_per_batch", "4",
        "--num_timesteps", "750_000_000", "--num_evals", "31",
        "--record_episodes", "4", "--store_model", "true",
        "--measure_performance", "--skip_rollout", "--skip_video",
    ),
    arms=tuple(
        Arm(f"{distribution_label}_{learner_label}", ("--context_distribution", distribution, *learner_flags))
        for distribution_label, distribution in DISTRIBUTIONS.items()
        for learner_label, learner_flags in LEARNERS.items()
    ),
)
