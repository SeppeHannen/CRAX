"""Replay a run's trained policy on chosen contexts and watch it.

    .venv/bin/python -m training.replay --run <W&B run name or id> \\
        --contexts level:1 level:2 level:3 --episodes 3 [--output_directory replays]

The run's latest checkpoint is fetched from its W&B files (``checkpoints/<step>``,
where ``train_env`` puts them), the networks are rebuilt from the checkpoint's own
config, and the policy is rolled out for ``--episodes`` full episodes in every
context, through the same evaluation stack the trainer scored it with
(``wrap_for_context_evaluation``) and with the same policy: sampling actions, as
the run's ``deterministic_eval`` flag says (``--deterministic`` overrides, to see the
mean action). Each episode becomes an MP4 with its return and cost in the file
name (``--no-video`` for numbers only, when the question is the distribution over
many episodes rather than one episode's behaviour); return and cost per episode are
printed as a table so the pictures can be read against the numbers.

**A policy is replayed by the code that trained it.** The run's commit is recorded in
W&B; this checkout must be at that commit, or the replay refuses. Networks, the
observation's layout and its scaling are all part of the commit, and a checkpoint
restored into different code is a different policy.
"""
from __future__ import annotations

import argparse
import dataclasses
import pathlib
import subprocess
from typing import List, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import wandb
from imageio import v3 as imageio

from crax import envs
from crax.io import image
from training import checkpoint as checkpoint_module
from training import contexts
from training.agents.ppo import networks as ppo_networks
from training.config import WANDB_ENTITY, WANDB_PROJECT
from training.contexts.wrapper import wrap_for_context_evaluation

COST_KEY = "cost"
CHECKPOINTS_DIRECTORY = "checkpoints"
NETWORK_CONFIG_FILE = "ppo_network_config.json"


@dataclasses.dataclass(frozen=True)
class EpisodeOutcome:
    context_spec: str
    episode: int
    length: int
    episode_return: float
    episode_cost: float
    video: pathlib.Path | None


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


def require_the_runs_commit(run: wandb.apis.public.Run, repository: pathlib.Path) -> None:
    head = subprocess.run(["git", "-C", str(repository), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    if run.commit != head:
        raise RuntimeError(
            f"this checkout is at {head[:12]} but run {run.name} trained at {run.commit[:12]}; a policy is replayed by the code "
            f"that trained it: `git worktree add ../CRAX-replay {run.commit[:12]}` and run from there"
        )


def policy_from_checkpoint(checkpoint_directory: pathlib.Path, factory_arguments: argparse.Namespace, deterministic: bool):
    """The inference function of a checkpoint, with the networks the run used."""
    config = checkpoint_module.load_config(checkpoint_directory / NETWORK_CONFIG_FILE)
    parameters = checkpoint_module.load(checkpoint_directory)
    learner = contexts.Learner(constraint=factory_arguments.constraint, policy_context=factory_arguments.policy_context)
    space = contexts.suite_contexts(factory_arguments.env_name).space
    factory = learner.train_kwargs(space)["network_factory"]
    network = checkpoint_module.get_network(config, factory.func)
    return ppo_networks.make_inference_fn(network)(parameters, deterministic=deterministic)


def replay(
    run_config: argparse.Namespace, policy, context_spec: str, episodes: int, seed: int, output_directory: pathlib.Path | None,
    width: int, height: int, camera: str,
) -> List[EpisodeOutcome]:
    """Roll the policy out for ``episodes`` whole episodes in one fixed context; one MP4 each
    when ``output_directory`` is given."""
    suite = contexts.suite_contexts(run_config.env_name)
    distribution = contexts.parse_distribution(context_spec, suite, total_rounds=1)
    env = envs.get_environment(run_config.env_name, level=run_config.difficulty, **(run_config.env_kwargs or {}))
    env = wrap_for_context_evaluation(env, distribution, episode_length=run_config.episode_length)
    reset, step = jax.jit(env.reset), jax.jit(env.step)
    act = jax.jit(policy)

    outcomes: List[EpisodeOutcome] = []
    key = jax.random.PRNGKey(seed)
    for episode in range(episodes):
        key, reset_key = jax.random.split(key)
        state = reset(jax.random.split(reset_key, 1))
        frames, rewards, costs = [], [], []
        for _ in range(run_config.episode_length):
            key, action_key = jax.random.split(key)
            action, _ = act(state.obs, action_key)
            state = step(state, action)
            rewards.append(float(state.reward[0]))
            costs.append(float(state.info[COST_KEY][0]))
            if output_directory is not None:
                frames.append(image.render_array(env.sys, jax.tree.map(lambda x: x[0], state.pipeline_state), height=height, width=width, camera=camera))
            if bool(state.done[0]):
                break
        video = None
        if output_directory is not None:
            video = output_directory / f"{contexts.spec_label(context_spec)}_episode{episode}_return{sum(rewards):.1f}_cost{sum(costs):.0f}.mp4"
            imageio.imwrite(video, np.stack(frames), fps=25, codec="libx264")
        outcomes.append(EpisodeOutcome(context_spec, episode, len(rewards), sum(rewards), sum(costs), video))
        print(f"{context_spec:>10} episode {episode}: {len(rewards):4d} steps, return {sum(rewards):5.1f}, cost {sum(costs):6.1f}" + (f" -> {video.name}" if video else ""), flush=True)
    return outcomes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="W&B run name (== id) of a context run trained with --store_model true")
    parser.add_argument("--contexts", nargs="+", required=True, help="context specs, e.g. level:1 level:2 level:3 or uniform")
    parser.add_argument("--episodes", type=int, default=3, help="episodes per context")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true", help="the mean action instead of the sampling policy the run was evaluated with")
    parser.add_argument("--no-video", action="store_true", help="numbers only; no rendering (about 50x faster)")
    parser.add_argument("--output_directory", type=pathlib.Path, default=pathlib.Path("replays"))
    parser.add_argument("--camera", default="fixedfar")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--entity", default=WANDB_ENTITY)
    parser.add_argument("--project", default=WANDB_PROJECT)
    arguments = parser.parse_args()

    repository = pathlib.Path(__file__).resolve().parents[1]
    run = wandb.Api().run(f"{arguments.entity}/{arguments.project}/{arguments.run}")
    require_the_runs_commit(run, repository)
    run_config = argparse.Namespace(**run.config)
    checkpoint_directory = fetch_latest_checkpoint(run, repository / "replays" / "checkpoints")
    deterministic = arguments.deterministic or bool(run_config.deterministic_eval)
    print(
        f"{run.name}: arm {run.config['arm']!r}, checkpoint {checkpoint_directory.name} ({int(checkpoint_directory.name):,} steps), "
        f"{'deterministic' if deterministic else 'sampling'} policy (the run evaluated with deterministic_eval={run_config.deterministic_eval})"
    )
    policy = policy_from_checkpoint(checkpoint_directory, run_config, deterministic)

    output_directory = None if arguments.no_video else arguments.output_directory / run.name
    if output_directory is not None:
        output_directory.mkdir(parents=True, exist_ok=True)
    outcomes: List[EpisodeOutcome] = []
    for context_spec in arguments.contexts:
        outcomes.extend(replay(run_config, policy, context_spec, arguments.episodes, arguments.seed, output_directory, arguments.width, arguments.height, arguments.camera))

    print(f"\n{'context':>10} {'episodes':>8} {'mean return':>12} {'mean cost':>10} {'cost sd':>8} {'under budget':>13}   budget {run_config.safety_bound:g}")
    for context_spec in arguments.contexts:
        rows = [outcome for outcome in outcomes if outcome.context_spec == context_spec]
        costs = np.array([r.episode_cost for r in rows])
        print(
            f"{context_spec:>10} {len(rows):8d} {np.mean([r.episode_return for r in rows]):12.1f} {costs.mean():10.1f} {costs.std():8.1f} "
            f"{int((costs <= run_config.safety_bound).sum()):6d} / {len(rows):<4d}"
        )
    if output_directory is not None:
        print(f"videos in {output_directory}")


if __name__ == "__main__":
    main()
