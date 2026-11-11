"""What the experiments of the generalization evaluation share
(``docs/acl/design/generalization_evaluation.md``): the environment, the training budget and
the four learners. Not an experiment itself; the definitions next to it import from here, so
the baselines of the three protocols differ in the protocol and nothing else.

The four learners are the two switches crossed: one multiplier or a Lagrange function of the
context (``--constraint scalar | additive``), a policy that does not or does see the context
(``--policy_context blind | told``; a told policy is never told a hidden coordinate).
"""
from typing import Dict, Tuple

from training.experiments import Arm, Experiment

LEARNERS: Dict[str, Tuple[str, ...]] = {
    "scalar_blind": ("--constraint", "scalar", "--policy_context", "blind"),
    "scalar_told": ("--constraint", "scalar", "--policy_context", "told"),
    "additive_blind": ("--constraint", "additive", "--policy_context", "blind"),
    "additive_told": ("--constraint", "additive", "--policy_context", "told"),
}

# Goal-point with the surface lidar, 500 M steps, scored during training on level 3 and on
# every region of the protocol; checkpoints stored for training/score.py.
COMMON: Tuple[str, ...] = (
    "--env_name", "safe_goal_point", "--alg", "ppo_lag",
    "--env_kwargs", '{"hazard_lidar": "surface_by_kind"}',
    "--deployment_distribution", "level:3",
    "--initial_lambda_lagr", "0.1", "--lagrangian_coef_rate", "1",
    "--num_envs", "8192", "--episode_length", "1000",
    "--unroll_length", "20", "--batch_size", "1024", "--num_minibatches", "32", "--num_updates_per_batch", "4",
    "--num_timesteps", "500_000_000", "--num_evals", "21",
    "--record_episodes", "2", "--store_model", "true",
    "--measure_performance", "--skip_rollout", "--skip_video",
)
SEEDS: Tuple[int, ...] = (0, 1, 2)


def baseline(group: str, protocol: str) -> Experiment:
    """Uniform sampling under ``protocol``: every learner with the protocol's regions held out,
    and as its reference, trained on all of the protocol's contexts. Eight arms, three seeds."""
    return Experiment(
        group=group,
        common=(*COMMON, "--context_distribution", "uniform", "--evaluation_protocol", protocol),
        arms=(
            *(Arm(learner, flags) for learner, flags in LEARNERS.items()),
            *(Arm(f"{learner}_reference", (*flags, "--reference")) for learner, flags in LEARNERS.items()),
        ),
        seeds=SEEDS,
    )
