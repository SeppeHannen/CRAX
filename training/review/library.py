"""Where a group's episode records are, and what is in them, without loading
every array.

A run's records live in its W&B files under ``episodes/`` (``train_env.py``
writes them there). While the run is local the same directory is on disk under
``wandb/run-*-<run id>/files/episodes``; the library reads that when it exists,
so a running run can be reviewed, and downloads the files into a cache
otherwise. Opening the library reads each record's identifying facts and its
per-step reward and cost (small arrays); the scene and the learner's view are
loaded when an episode is opened.
"""
from __future__ import annotations

import dataclasses
import json
import pathlib
from typing import Dict, List, Sequence, Tuple

import numpy as np
import wandb

from training.evaluation.record import RECORDS_DIRECTORY, SUFFIX, EpisodeRecord


@dataclasses.dataclass(frozen=True)
class EpisodeSummary:
    """What the episode list shows, and where the whole record is."""

    index: int  # position in the library, the id the page uses
    path: pathlib.Path
    env_name: str
    run_name: str
    arm: str
    evaluation_name: str
    environment_steps: int
    episode_index: int
    length: int
    episode_return: float
    episode_cost: float
    context: Tuple[float, ...]

    @classmethod
    def read(cls, index: int, path: pathlib.Path) -> "EpisodeSummary":
        with np.load(path) as contents:
            metadata = json.loads(str(contents["metadata"]))
            reward, cost, context = contents["reward"], contents["cost"], contents["context"]
        return cls(
            index=index, path=path, env_name=str(metadata["env_name"]), run_name=str(metadata["run_name"]),
            arm=str(metadata["arm"]), evaluation_name=str(metadata["evaluation_name"]),
            environment_steps=int(metadata["environment_steps"]), episode_index=int(metadata["episode_index"]),
            length=int(reward.shape[0]), episode_return=float(reward.sum()), episode_cost=float(cost.sum()),
            context=tuple(float(value) for value in context),
        )

    def as_json(self) -> Dict[str, object]:
        """The summary for the page; the path is the server's business."""
        return {name: value for name, value in dataclasses.asdict(self).items() if name != "path"}


@dataclasses.dataclass(frozen=True)
class Library:
    """Every recorded episode of a group, summarised, in a fixed order."""

    group: str
    episodes: Tuple[EpisodeSummary, ...]

    @classmethod
    def from_directories(cls, group: str, directories: Sequence[pathlib.Path]) -> "Library":
        paths = sorted(path for directory in directories for path in directory.rglob(f"*{SUFFIX}"))
        if not paths:
            raise FileNotFoundError(f"no episode records (*{SUFFIX}) under {[str(d) for d in directories]}")
        return cls(group, tuple(EpisodeSummary.read(index, path) for index, path in enumerate(paths)))

    def summary(self, index: int) -> EpisodeSummary:
        if not 0 <= index < len(self.episodes):
            raise IndexError(f"no episode {index}; the library has {len(self.episodes)}")
        return self.episodes[index]

    def record(self, index: int) -> EpisodeRecord:
        return EpisodeRecord.load(self.summary(index).path)

    @property
    def env_names(self) -> Tuple[str, ...]:
        return tuple(sorted({episode.env_name for episode in self.episodes}))


def group_record_directories(
    entity: str, project: str, group: str, wandb_directory: pathlib.Path, cache_directory: pathlib.Path
) -> List[pathlib.Path]:
    """One ``episodes`` directory per run of the group that has records: the run's local
    W&B files directory when this machine ran it, else a download into ``cache_directory``."""
    api = wandb.Api(timeout=120)
    directories: List[pathlib.Path] = []
    for run in api.runs(f"{entity}/{project}", filters={"group": group}):
        local = _local_run_files(wandb_directory, run.id)
        if local is not None and (local / RECORDS_DIRECTORY).is_dir():
            directories.append(local / RECORDS_DIRECTORY)
            continue
        remote = [file for file in run.files(per_page=10_000) if file.name.startswith(f"{RECORDS_DIRECTORY}/") and file.name.endswith(SUFFIX)]
        if not remote:
            continue
        target = cache_directory / group / run.id
        for file in remote:
            if not (target / file.name).exists():
                file.download(root=str(target), exist_ok=True)
        directories.append(target / RECORDS_DIRECTORY)
    if not directories:
        raise FileNotFoundError(f"no run of group {group!r} has episode records; was it trained with --record_episodes > 0?")
    return directories


def _local_run_files(wandb_directory: pathlib.Path, run_id: str) -> pathlib.Path | None:
    matches = sorted(wandb_directory.glob(f"run-*-{run_id}/files"))
    return matches[-1] if matches else None
