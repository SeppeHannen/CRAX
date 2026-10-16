"""A run's policy is replayed by the code that trained it, from its latest checkpoint."""
from __future__ import annotations

import dataclasses
import pathlib
import subprocess
from typing import List

import pytest

from training import replay


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


def test_the_replay_refuses_a_checkout_at_another_commit(tmp_path: pathlib.Path):
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    git = ["git", "-C", str(tmp_path)]
    subprocess.run([*git, "config", "user.email", "t@t"], check=True)
    subprocess.run([*git, "config", "user.name", "t"], check=True)
    (tmp_path / "a").write_text("a")
    subprocess.run([*git, "add", "a"], check=True)
    subprocess.run([*git, "commit", "--quiet", "-m", "first"], check=True)
    head = subprocess.run([*git, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()

    replay.require_the_runs_commit(_Run("r", "run", head, []), tmp_path)
    with pytest.raises(RuntimeError, match="replayed by the code that trained it"):
        replay.require_the_runs_commit(_Run("r", "run", "0" * 40, []), tmp_path)
