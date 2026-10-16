"""Replay a run's trained policy on chosen contexts and record the episodes for the reviewer.

    .venv/bin/python -m training.replay --run <W&B run name> --contexts level:3 --episodes 64
    .venv/bin/python -m training.review --records replays/<run name>

The run's latest checkpoint is fetched from its W&B files (``checkpoints/<step>``,
where ``train_env`` puts them) and the networks are rebuilt from the checkpoint's
own config. The policy is then rolled out, ``--episodes`` whole episodes at once per
context, by the same :class:`~training.acting.Evaluator` and evaluation stack the
trainer scored it with, with the same policy (sampling actions, as the run's
``deterministic_eval`` says; ``--deterministic`` overrides). The episodes are written
as :class:`~training.evaluation.record.EpisodeRecord` files, the format the review
page plays (scene, observation, action, reward, cost per step), so a replayed
episode is reviewed exactly like one recorded during training.

**The learner's view comes from the whole learner.** A checkpoint holds the policy
tuple (normaliser, policy, reward critic, encoder) and, next to it, the learner's
state: the cost critic and the constraint's multipliers (``checkpoint.save_learner_state``).
The view of every replayed episode — V, V_c, λ at each step, the three advantages — is
then the agent's own :func:`training.agents.ppo_lag.train.learner_view_function`, the
one the trainer uses for the records it writes during training. A checkpoint from
before learner states were saved is refused: a replay that could only guess at V_c
and λ would be a second, poorer definition of the record.

**A policy is replayed by the code that trained it.** The run's commit is recorded in
W&B. This checkout must be at that commit or descend from it without touching the
code the policy is made of (:data:`POLICY_CODE`): networks, observation, scaling,
physics. Otherwise the replay refuses: a checkpoint restored into different code is
a different policy.
"""
from __future__ import annotations

import argparse
import dataclasses
import functools
import pathlib
import subprocess
from typing import Any, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import wandb

from crax import envs
from crax.envs.limb_colors import colorize_env_limbs
from training import acting
from training import checkpoint as checkpoint_module
from training import contexts
from training.acme.running_statistics import RunningStatisticsState
from training.agents.ppo import networks as ppo_networks
from training.agents.ppo.losses import PPONetworkParams
from training.agents.ppo.train import LEARNER_AUX_STATE, LEARNER_COST_VALUE
from training.agents.ppo_lag.constraint import LagrangeConstraint, ScalarConstraint
from training.agents.ppo_lag.train import learner_view_function
from training.config import WANDB_ENTITY, WANDB_PROJECT
from training.contexts.wrapper import wrap_for_context_evaluation
from training.evaluation import EpisodeRecorder, LearnerView
from training.pixels import PixelObservation
from training.types import Transition

CHECKPOINTS_DIRECTORY = "checkpoints"
NETWORK_CONFIG_FILE = "ppo_network_config.json"
# The code a policy is made of: a commit that differs from the run's in any of these paths
# would restore the checkpoint into a different policy.
POLICY_CODE: Tuple[str, ...] = (
    "crax/",
    "training/acme/",
    "training/networks.py",
    "training/distribution.py",
    "training/agents/ppo/networks.py",
    "training/contexts/networks.py",
    "training/contexts/observation.py",
    "training/contexts/space.py",
    "training/contexts/wrapper.py",
    "training/contexts/distributions/",
)


@dataclasses.dataclass(frozen=True)
class ContextOutcome:
    """Return and cost of the replayed episodes in one context."""

    context_spec: str
    returns: np.ndarray  # [K]
    costs: np.ndarray  # [K]


def fetch_latest_checkpoint(run: wandb.apis.public.Run, cache: pathlib.Path) -> pathlib.Path:
    """Download the run's latest ``checkpoints/<step>`` directory (once) and return it."""
    files = [file for file in run.files(per_page=10_000) if file.name.startswith(f"{CHECKPOINTS_DIRECTORY}/")]
    if not files:
        raise FileNotFoundError(f"run {run.name} has no files under {CHECKPOINTS_DIRECTORY}/; was it trained with --store_model true?")
    steps = sorted({file.name.split("/")[1] for file in files})
    latest = steps[-1]
    target = cache / run.id
    for file in files:
        if file.name.split("/")[1] == latest and not (target / file.name).exists():
            file.download(root=str(target), exist_ok=True)
    return target / CHECKPOINTS_DIRECTORY / latest


def require_the_runs_policy_code(run_commit: str, repository: pathlib.Path) -> None:
    """Refuse unless this checkout is the run's commit, or descends from it with :data:`POLICY_CODE` untouched."""
    git = ["git", "-C", str(repository)]
    head = subprocess.run([*git, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    if head == run_commit:
        return
    is_ancestor = subprocess.run([*git, "merge-base", "--is-ancestor", run_commit, head], capture_output=True).returncode == 0
    if not is_ancestor:
        raise RuntimeError(
            f"this checkout ({head[:12]}) does not descend from the run's commit {run_commit[:12]}; a policy is replayed by the "
            f"code that trained it: `git worktree add ../CRAX-replay {run_commit[:12]}`"
        )
    changed = subprocess.run([*git, "diff", "--name-only", run_commit, head, "--", *POLICY_CODE], capture_output=True, text=True, check=True).stdout
    if changed.strip():
        raise RuntimeError(
            f"this checkout changes the code the policy is made of since the run's commit {run_commit[:12]}:\n{changed}"
            f"restoring the checkpoint here would be a different policy; replay from `git worktree add ../CRAX-replay {run_commit[:12]}`"
        )


@dataclasses.dataclass(frozen=True)
class Learner:
    """A PPO-Lagrange learner rebuilt from a checkpoint: what acts, and what sees an episode."""

    network: ppo_networks.PPONetworks
    normaliser: RunningStatisticsState
    parameters: PPONetworkParams  # every network's parameters, in the loss's layout
    multipliers: Any  # the constraint's state: the trainer's ``aux_state``
    constraint: LagrangeConstraint

    @property
    def policy_tuple(self) -> Tuple[Any, ...]:
        """(normaliser, policy, value[, encoder]): what the inference function consumes, the
        trainer's ``_policy_params_tuple`` (the encoder slot exists iff the network has one)."""
        base = (self.normaliser, self.parameters.policy, self.parameters.value)
        return base + (self.parameters.encoder,) if self.network.encoder_network is not None else base

    def view(self, episodes: Transition, run_config: argparse.Namespace) -> LearnerView:
        """The agent's own view of ``episodes`` (``[K, T, ...]``), as the trainer records it."""
        return learner_view_function(self.constraint).compute(
            self.parameters, self.normaliser, episodes, self.multipliers, ppo_network=self.network,
            discounting=run_config.discounting, reward_scaling=run_config.reward_scaling, gae_lambda=run_config.gae_lambda,
        )


def learner_from_checkpoint(checkpoint_directory: pathlib.Path, run_config: argparse.Namespace) -> Learner:
    """The run's learner: networks from the checkpoint's config under the run's :class:`contexts.Learner`,
    the policy tuple and the learner state from its files, the constraint from the run's flags."""
    config = checkpoint_module.load_config(checkpoint_directory / NETWORK_CONFIG_FILE)
    spec = contexts.Learner(constraint=run_config.constraint, policy_context=run_config.policy_context)
    space = contexts.suite_contexts(run_config.env_name).space
    network = checkpoint_module.get_network(config, spec.train_kwargs(space)["network_factory"].func)
    # PPO-Lagrange's constraint for this learner: λ(ω) if it holds one, else the agent's own scalar λ.
    constraint: LagrangeConstraint = spec.per_context_constraint(space) if spec.holds_constraint_per_context else ScalarConstraint()
    # (normaliser, policy, value[, encoder]): the trainer's policy tuple, the encoder slot iff the
    # network the same config builds has one.
    normaliser, policy, value, *encoder = checkpoint_module.load(checkpoint_directory)
    # The learner state's structure is the agent's own: a fresh learner's, before any training.
    learner_state = checkpoint_module.load_learner_state(checkpoint_directory, like={
        LEARNER_COST_VALUE: network.cost_value_network.init(jax.random.PRNGKey(0)),
        LEARNER_AUX_STATE: constraint.initial_multipliers(0.0),
    })
    parameters = PPONetworkParams(policy=policy, value=value, cost_value=learner_state[LEARNER_COST_VALUE], encoder=encoder[0] if encoder else None)
    return Learner(network, normaliser, parameters, learner_state[LEARNER_AUX_STATE], constraint)


def environment_of(run_config: argparse.Namespace) -> envs.Env:
    """The unwrapped environment the run trained in (``train_env.py``): its level and kwargs; for
    a vision run the MJX backend the renderer needs and, if the run asked, the coloured limbs."""
    env_kwargs = dict(run_config.env_kwargs or {})
    if run_config.vision:
        env_kwargs.setdefault("backend", "mjx")
    env = envs.get_environment(run_config.env_name, level=run_config.difficulty, **env_kwargs)
    if run_config.vision and run_config.vision_limb_colors:
        colorize_env_limbs(env, run_config.env_name)
    return env


def replay_context(
    run_config: argparse.Namespace, learner: Learner, deterministic: bool,
    context_spec: str, episodes: int, seed: int, recorder: EpisodeRecorder, environment_steps: int,
) -> ContextOutcome:
    """Roll the policy out for ``episodes`` whole episodes in one context, at once, and record them."""
    suite = contexts.suite_contexts(run_config.env_name)
    distribution = contexts.parse_distribution(context_spec, suite, total_rounds=1)
    env = wrap_for_context_evaluation(environment_of(run_config), distribution, episode_length=run_config.episode_length)
    pixel_observation = PixelObservation.from_config(run_config)
    if pixel_observation is not None:
        # Outermost, over the fully batched stack, as the trainer's `_maybe_wrap_env` does.
        from crax.envs.wrappers.pixel_observation_gpu import GpuPixelObservationWrapper  # needs Warp and a GPU

        env = GpuPixelObservationWrapper(env, num_envs=episodes, **pixel_observation.wrapper_kwargs())
    evaluator = acting.Evaluator(
        env, functools.partial(ppo_networks.make_inference_fn(learner.network), deterministic=deterministic),
        num_eval_envs=episodes, episode_length=run_config.episode_length, action_repeat=1,
        key=jax.random.PRNGKey(seed), recording=recorder.recording,
    )
    metrics, recorded = evaluator.run_recorded_evaluation(learner.policy_tuple, training_metrics={}, aggregate_episodes=False)
    episode_batches = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), recorded)  # [K, T, ...]
    view = learner.view(episode_batches, run_config)
    recorder.record(contexts.spec_label(context_spec), environment_steps, recorded, view)
    return ContextOutcome(context_spec, np.asarray(metrics["eval/episode_reward"]), np.asarray(metrics["eval/episode_cost"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="W&B run name (== id) of a context run trained with --store_model true")
    parser.add_argument("--contexts", nargs="+", required=True, help="context specs, e.g. level:1 level:2 level:3 or uniform")
    parser.add_argument("--episodes", type=int, default=64, help="episodes per context, rolled out in parallel")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true", help="the mean action instead of the sampling policy the run was evaluated with")
    parser.add_argument("--output_directory", type=pathlib.Path, default=pathlib.Path("replays"),
                        help="records go to <output_directory>/<run name>/seed<seed>/ (one replay never overwrites another)")
    parser.add_argument("--entity", default=WANDB_ENTITY)
    parser.add_argument("--project", default=WANDB_PROJECT)
    arguments = parser.parse_args()

    repository = pathlib.Path(__file__).resolve().parents[1]
    run = wandb.Api().run(f"{arguments.entity}/{arguments.project}/{arguments.run}")
    require_the_runs_policy_code(run.commit, repository)
    run_config = argparse.Namespace(**run.config)
    checkpoint_directory = fetch_latest_checkpoint(run, arguments.output_directory / "checkpoints")
    environment_steps = int(checkpoint_directory.name)
    deterministic = arguments.deterministic or bool(run_config.deterministic_eval)
    print(
        f"{run.name}: arm {run.config['arm']!r}, checkpoint at {environment_steps:,} steps, "
        f"{'deterministic' if deterministic else 'sampling'} policy (the run evaluated with deterministic_eval={run_config.deterministic_eval})"
    )
    learner = learner_from_checkpoint(checkpoint_directory, run_config)
    pixel_observation = PixelObservation.from_config(run_config)
    recorder = EpisodeRecorder(
        directory=arguments.output_directory / run.name / f"seed{arguments.seed}", env_name=run_config.env_name, run_name=run.name,
        arm=run.config["arm"], num_episodes=arguments.episodes,
        sensors_key=contexts.SENSORS_KEY, context_key=contexts.TRANSITION_CONTEXT_KEY,
        image_keys=(pixel_observation.key,) if pixel_observation is not None else (),
    )

    outcomes = [
        replay_context(run_config, learner, deterministic, spec, arguments.episodes, arguments.seed, recorder, environment_steps)
        for spec in arguments.contexts
    ]
    print(f"\n{'context':>10} {'episodes':>8} {'return':>8} {'cost':>8} {'cost sd':>8} {'cost max':>9} {'under budget':>13}   budget {run_config.safety_bound:g}")
    for outcome in outcomes:
        under = int((outcome.costs <= run_config.safety_bound).sum())
        print(
            f"{outcome.context_spec:>10} {len(outcome.costs):8d} {outcome.returns.mean():8.1f} {outcome.costs.mean():8.1f} "
            f"{outcome.costs.std():8.1f} {outcome.costs.max():9.1f} {under:6d} / {len(outcome.costs):<4d}"
        )
    print(f"records in {recorder.directory}; review with: python -m training.review --records {recorder.directory.parent}")


if __name__ == "__main__":
    main()
