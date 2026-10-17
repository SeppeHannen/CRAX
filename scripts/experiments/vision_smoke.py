"""Ten rounds of the vision learners experiment under two arms, before the 500 M run: the
vision x contextual composition compiles and runs on the GPU (MJWarp renders into the
{sensors, context, pixels} observation, the shared encoder's latent reaches every head, λ(ω)
is updated), the records carry the frames the policy saw, the checkpoint carries the whole
learner, and both reach W&B as run files. Same batch shape and flags as the experiment.

    .venv/bin/python -m training.experiments.snellius scripts/experiments/vision_smoke.py --partition gpu_a100 --time 00:30:00
"""
from training.experiments import Arm, Experiment

EXPERIMENT = Experiment(
    group="vision_smoke_v1",
    common=(
        "--env_name", "safe_goal_point", "--alg", "ppo_lag",
        "--deployment_distribution", "level:3",
        "--initial_lambda_lagr", "0.1", "--lagrangian_coef_rate", "1",
        "--vision", "--vision_obs_mode", "pixels+state", "--gpu_memory_fraction", "0.75",  # the epoch program peaks at 19.3 GiB
        "--num_envs", "1024", "--episode_length", "1000",
        "--unroll_length", "20", "--batch_size", "1024", "--num_minibatches", "4", "--num_updates_per_batch", "4",
        "--record_episodes", "4", "--store_model", "true",
        "--measure_performance", "--skip_rollout", "--skip_video",
        "--num_timesteps", "819_200", "--num_evals", "2",  # 10 rounds of 1024 envs x 20 steps x 4 minibatches
    ),
    arms=(
        Arm("uniform_scalar_blind", ("--context_distribution", "uniform", "--constraint", "scalar", "--policy_context", "blind")),
        Arm("plrlagrangian_additive_told", ("--context_distribution", "plr:lagrangian", "--constraint", "additive", "--policy_context", "told")),
    ),
)
