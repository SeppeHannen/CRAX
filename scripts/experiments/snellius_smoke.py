"""Ten rounds of goal-point under two arms of the 500 M learners experiment, to verify the
pipeline end to end on a new machine: compiles, throughput, W&B reachability, checkpoints
arriving as run files, and two arms starting at once without racing for the view.

    .venv/bin/python -m training.experiments.snellius scripts/experiments/snellius_smoke.py --partition gpu_a100 --time 00:40:00
"""
from training.experiments import Arm, Experiment

EXPERIMENT = Experiment(
    group="snellius_smoke",
    common=(
        "--env_name", "safe_goal_point", "--alg", "ppo_lag",
        "--deployment_distribution", "level:3",
        "--initial_lambda_lagr", "0.1", "--lagrangian_coef_rate", "1",
        "--num_envs", "8192", "--num_timesteps", "6_553_600", "--num_evals", "2", "--episode_length", "1000",
        "--unroll_length", "20", "--batch_size", "1024", "--num_minibatches", "32", "--num_updates_per_batch", "4",
        "--measure_performance", "--skip_rollout", "--skip_video", "--store_model", "true",
    ),
    arms=(
        Arm("uniform_scalar_blind", ("--context_distribution", "uniform", "--constraint", "scalar", "--policy_context", "blind")),
        Arm("plrlagrangian_additive_told", ("--context_distribution", "plr:lagrangian", "--constraint", "additive", "--policy_context", "told")),
    ),
)
