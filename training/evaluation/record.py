"""One recorded evaluation episode: what happened, what the policy saw and did,
and what the learner believed at every step (``docs/acl/design/episode_review.md`` §2).

An :class:`EpisodeRecord` is written once by :class:`~training.evaluation.recorder.EpisodeRecorder`
and read by the viewer (``training/review``). It is one ``.npz`` file: the
per-step arrays as NumPy arrays and the identifying facts as one JSON string, so
a record is self-describing without the run that produced it.

The episode's steps run from the first observation to the step the episode
ended, ``length`` steps; the arrays are that long. These are the frozen policy's
evaluation episodes, scored but never trained on: the values and advantages are
the learner's view of behaviour that was not in any gradient.
"""
from __future__ import annotations

import dataclasses
import json
import pathlib
from typing import Iterator, Tuple

import numpy as np

SUFFIX = ".episode.npz"
# The subdirectory of a run's W&B files its records are written to and read from.
RECORDS_DIRECTORY = "episodes"
_METADATA_KEY = "metadata"


@dataclasses.dataclass(frozen=True)
class EpisodeRecord:
    """One episode, one row per step; identifying facts first, per-step arrays after."""

    # Which episode this is
    env_name: str
    run_name: str
    arm: str  # the arm label the dashboard groups by
    evaluation_name: str  # the target the episode was scored on: deployment, uniform, …
    environment_steps: int  # the training progress at which the frozen policy was evaluated
    episode_index: int  # position among the K recorded episodes of that evaluation
    context: np.ndarray  # [D]  the context ω the episode ran in
    # The scene, for the renderer (training/evaluation/scene.py)
    qpos: np.ndarray  # [T, nq]
    qvel: np.ndarray  # [T, nv]
    mocap_pos: np.ndarray  # [T, nmocap, 3]
    mocap_quat: np.ndarray  # [T, nmocap, 4]
    # What the policy saw and did, and what happened
    observation: np.ndarray  # [T, obs]
    action: np.ndarray  # [T, A]
    reward: np.ndarray  # [T]  raw environment reward
    cost: np.ndarray  # [T]
    # The learner's view (training/evaluation/learner_view.py)
    value: np.ndarray  # [T]
    cost_value: np.ndarray  # [T]
    lagrange_multiplier: np.ndarray  # [T]
    reward_advantage: np.ndarray  # [T]
    cost_advantage: np.ndarray  # [T]
    lagrangian_advantage: np.ndarray  # [T]

    def __post_init__(self) -> None:
        length = self.length
        if length == 0:
            raise ValueError("an EpisodeRecord needs at least one step")
        for name in _PER_STEP_FIELDS:
            array = getattr(self, name)
            if array.shape[:1] != (length,):
                raise ValueError(f"EpisodeRecord.{name} has shape {array.shape}; expected a leading axis of {length} steps")
        for name in ("reward", "cost", "value", "cost_value", "lagrange_multiplier",
                     "reward_advantage", "cost_advantage", "lagrangian_advantage"):
            if getattr(self, name).ndim != 1:
                raise ValueError(f"EpisodeRecord.{name} must be one value per step, has shape {getattr(self, name).shape}")
        if self.context.ndim != 1:
            raise ValueError(f"EpisodeRecord.context must be one vector, has shape {self.context.shape}")

    @property
    def length(self) -> int:
        return int(self.reward.shape[0])

    @property
    def episode_return(self) -> float:
        return float(self.reward.sum())

    @property
    def episode_cost(self) -> float:
        return float(self.cost.sum())

    @property
    def key(self) -> Tuple[str, int, str, int]:
        """What identifies the episode within a group: arm, evaluation step, target, index."""
        return (self.arm, self.environment_steps, self.evaluation_name, self.episode_index)

    def filename(self) -> str:
        """Unique within one run's records: evaluation step, target, index."""
        return f"{self.environment_steps:012d}_{self.evaluation_name}_{self.episode_index:02d}{SUFFIX}"

    def save(self, directory: pathlib.Path) -> pathlib.Path:
        """Write the record under ``directory`` (created if needed); returns the path."""
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / self.filename()
        metadata = {
            "env_name": self.env_name,
            "run_name": self.run_name,
            "arm": self.arm,
            "evaluation_name": self.evaluation_name,
            "environment_steps": self.environment_steps,
            "episode_index": self.episode_index,
        }
        arrays = {name: getattr(self, name) for name in _PER_STEP_FIELDS + ("context",)}
        np.savez_compressed(path, **arrays, **{_METADATA_KEY: np.asarray(json.dumps(metadata))})
        return path

    @classmethod
    def load(cls, path: pathlib.Path) -> "EpisodeRecord":
        with np.load(path) as contents:
            metadata = json.loads(str(contents[_METADATA_KEY]))
            arrays = {name: contents[name] for name in _PER_STEP_FIELDS + ("context",)}
        return cls(
            env_name=str(metadata["env_name"]),
            run_name=str(metadata["run_name"]),
            arm=str(metadata["arm"]),
            evaluation_name=str(metadata["evaluation_name"]),
            environment_steps=int(metadata["environment_steps"]),
            episode_index=int(metadata["episode_index"]),
            **arrays,
        )


_PER_STEP_FIELDS: Tuple[str, ...] = (
    "qpos", "qvel", "mocap_pos", "mocap_quat",
    "observation", "action", "reward", "cost",
    "value", "cost_value", "lagrange_multiplier", "reward_advantage", "cost_advantage", "lagrangian_advantage",
)


def records_under(root: pathlib.Path) -> Iterator[EpisodeRecord]:
    """Every record below ``root``, in path order."""
    for path in sorted(root.rglob(f"*{SUFFIX}")):
        yield EpisodeRecord.load(path)
