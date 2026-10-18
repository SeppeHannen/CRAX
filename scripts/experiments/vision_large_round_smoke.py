"""Three rounds of the large-round pixel experiment on an H100, to measure the epoch program's
peak memory and throughput before the six arms are submitted (one arm, additive told, the larger
of the two learners; ``--measure_performance`` prints the peak).

    .venv/bin/python -m training.experiments.snellius scripts/experiments/vision_large_round_smoke.py --partition gpu_h100 --time 00:30:00
"""
from training.experiments import Arm, Experiment

EXPERIMENT = Experiment(
    group="vision_large_round_smoke_v1",
    common=(
        "--env_name", "safe_goal_point", "--alg", "ppo_lag",
        "--deployment_distribution", "level:3",
        "--initial_lambda_lagr", "0.1", "--lagrangian_coef_rate", "1",
        "--vision", "--vision_obs_mode", "pixels+state", "--gpu_memory_fraction", "0.85",
        "--num_envs", "1024", "--episode_length", "1000",
        "--unroll_length", "20", "--batch_size", "1024", "--num_minibatches", "32", "--num_updates_per_batch", "4",
        "--num_timesteps", "1_966_080", "--num_evals", "2",  # 3 rounds of 655 360 steps
        "--record_episodes", "4", "--store_model", "true",
        "--measure_performance", "--skip_rollout", "--skip_video",
    ),
    arms=(Arm("plr_additive_told", ("--context_distribution", "plr", "--constraint", "additive", "--policy_context", "told")),),
)
