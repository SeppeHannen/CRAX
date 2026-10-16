"""A run's policy is replayed by the code that trained it (or code that leaves the policy's code
untouched), from its latest checkpoint."""
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
