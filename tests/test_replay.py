"""A run's policy is replayed by the code that trained it (or code that leaves the policy's code
untouched), from its latest checkpoint."""
from __future__ import annotations

import argparse
import dataclasses
import pathlib
import shutil
import subprocess
from typing import List

import jax.numpy as jnp
import numpy as np
import pytest

from crax import envs
from training import checkpoint as checkpoint_module
from training import contexts
from training import replay
from training.agents.ppo_lag.train import train as train_ppo_lag
from training.evaluation import EpisodeRecorder, records_under


@dataclasses.dataclass
class _File:
    name: str
    downloaded: List[str]

    def download(self, root: str, exist_ok: bool) -> None:
        self.downloaded.append(self.name)
        path = pathlib.Path(root) / self.name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x")


@dataclasses.dataclass
class _Run:
    id: str
    name: str
    commit: str
    file_names: List[str]
    downloaded: List[str] = dataclasses.field(default_factory=list)

    def files(self, per_page: int) -> List[_File]:
        return [_File(name, self.downloaded) for name in self.file_names]


def test_the_latest_checkpoint_is_fetched_once_and_only_that_one(tmp_path: pathlib.Path):
    run = _Run("r1", "run", "abc", [
        "config.yaml",
        "checkpoints/000000000100/ppo_network_config.json", "checkpoints/000000000100/d/blob",
        "checkpoints/000000000200/ppo_network_config.json", "checkpoints/000000000200/d/blob",
    ])
    directory = replay.fetch_latest_checkpoint(run, tmp_path)
    assert directory == tmp_path / "r1" / "checkpoints" / "000000000200"
    assert sorted(run.downloaded) == ["checkpoints/000000000200/d/blob", "checkpoints/000000000200/ppo_network_config.json"]
    run.downloaded.clear()
    assert replay.fetch_latest_checkpoint(run, tmp_path) == directory and run.downloaded == []  # cached

    with pytest.raises(FileNotFoundError, match="--store_model"):
        replay.fetch_latest_checkpoint(_Run("r2", "bare", "abc", ["config.yaml"]), tmp_path)


def _commit(repository: pathlib.Path, path: str, text: str) -> str:
    git = ["git", "-C", str(repository)]
    (repository / path).parent.mkdir(parents=True, exist_ok=True)
    (repository / path).write_text(text)
    subprocess.run([*git, "add", path], check=True)
    subprocess.run([*git, "commit", "--quiet", "-m", path], check=True)
    return subprocess.run([*git, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()


def test_the_replay_runs_at_the_runs_commit_or_a_descendant_that_leaves_the_policy_code_alone(tmp_path: pathlib.Path):
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    git = ["git", "-C", str(tmp_path)]
    subprocess.run([*git, "config", "user.email", "t@t"], check=True)
    subprocess.run([*git, "config", "user.name", "t"], check=True)
    trained_at = _commit(tmp_path, "training/contexts/networks.py", "networks v1")
    replay.require_the_runs_policy_code(trained_at, tmp_path)  # the run's commit itself

    _commit(tmp_path, "training/review/page.py", "a tool added later")
    replay.require_the_runs_policy_code(trained_at, tmp_path)  # descends, policy code untouched

    with pytest.raises(RuntimeError, match="does not descend"):
        replay.require_the_runs_policy_code("0" * 40, tmp_path)

    _commit(tmp_path, "training/contexts/networks.py", "networks v2")
    with pytest.raises(RuntimeError, match="training/contexts/networks.py"):
        replay.require_the_runs_policy_code(trained_at, tmp_path)


# --------------------------------------------------------------------------- #
# The replayed episodes carry the learner's own view
# --------------------------------------------------------------------------- #

ENV_NAME = "safe_velocity_ant"
TINY = dict(num_envs=8, batch_size=8, num_minibatches=2, unroll_length=5, num_updates_per_batch=1, episode_length=20,
            num_eval_envs=4, num_evals=2, learning_rate=1e-3, normalize_observations=True)
NUM_TIMESTEPS = 8 * 5 * 2 * 4


def _train_and_checkpoint(checkpoints: pathlib.Path, learner: contexts.Learner) -> argparse.Namespace:
    """A tiny PPO-Lagrange run under ``learner`` whose checkpoints go to ``checkpoints``; returns the
    run's config as :func:`replay.main` sees it (the W&B config, every CLI flag by name)."""
    setup = contexts.context_training_setup(ENV_NAME, "uniform", "level:3", learner, environment_overrides={}, num_timesteps=NUM_TIMESTEPS,
                                            num_envs=8, batch_size=8, unroll_length=5, num_minibatches=2)
    train_ppo_lag(environment=envs.get_environment(ENV_NAME, level=1, backend="mjx"), num_timesteps=NUM_TIMESTEPS, seed=0,
                  progress_fn=lambda *_: None, safety_bound=25.0, initial_lambda_lagr=0.1, **TINY, **setup.train_kwargs(),
                  save_checkpoint_path=str(checkpoints))
    return argparse.Namespace(
        env_name=ENV_NAME, difficulty=1, env_kwargs={"backend": "mjx"}, episode_length=TINY["episode_length"],
        constraint=learner.constraint, policy_context=learner.policy_context, memory_time_constant=learner.memory_time_constant,
        discounting=0.97, reward_scaling=1.0, gae_lambda=0.95, vision=False,
    )


def test_a_replayed_episode_carries_the_learners_own_view_from_the_checkpointed_learner_state(tmp_path: pathlib.Path):
    """V_c and λ in a replayed record come from the saved cost critic and multipliers: λ is the
    constraint's lookup at the episode's context, finite and the same throughout the episode."""
    learner_spec = contexts.Learner(constraint="additive", policy_context="told")
    run_config = _train_and_checkpoint(tmp_path / "checkpoints", learner_spec)
    checkpoint_directory = sorted(p for p in (tmp_path / "checkpoints").iterdir() if p.is_dir())[-1]
    learner = replay.learner_from_checkpoint(checkpoint_directory, run_config)
    recorder = EpisodeRecorder(tmp_path / "records", ENV_NAME, "tiny_run", "replay", num_episodes=3,
                               sensors_key=contexts.SENSORS_KEY, context_key=contexts.TRANSITION_CONTEXT_KEY, image_keys=(), written=lambda path: None)
    outcome = replay.replay_context(run_config, learner, False, "level:2", episodes=3, seed=1, recorder=recorder,
                                    environment_steps=int(checkpoint_directory.name))
    assert outcome.costs.shape == (3,)
    records = list(records_under(tmp_path / "records"))
    assert len(records) == 3
    expected = np.asarray(learner.constraint.multipliers_at(learner.multipliers, jnp.asarray([[2.0]])))[0]
    for record in records:
        assert np.isfinite(record.cost_value).all() and np.isfinite(record.cost_advantage).all()
        np.testing.assert_allclose(record.lagrange_multiplier, expected, atol=1e-5)
        np.testing.assert_allclose(record.lagrangian_advantage, record.reward_advantage - record.lagrange_multiplier * record.cost_advantage, atol=1e-5)


def test_a_checkpoint_without_learner_state_is_refused(tmp_path: pathlib.Path):
    run_config = _train_and_checkpoint(tmp_path / "checkpoints", contexts.Learner(constraint="scalar", policy_context="blind"))
    checkpoint_directory = sorted(p for p in (tmp_path / "checkpoints").iterdir() if p.is_dir())[-1]
    shutil.rmtree(checkpoint_directory / checkpoint_module.LEARNER_STATE_DIRECTORY)
    with pytest.raises(ValueError, match="no learner state"):
        replay.learner_from_checkpoint(checkpoint_directory, run_config)
