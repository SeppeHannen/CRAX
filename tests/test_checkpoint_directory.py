"""A run's checkpoints live under its W&B run directory, so they travel with the run."""
from __future__ import annotations

import pathlib

import pytest
import wandb

from training import checkpoint


def test_checkpoints_go_under_the_active_run_directory(tmp_path: pathlib.Path):
    with pytest.raises(RuntimeError, match="initialise the run first"):
        checkpoint.run_checkpoint_directory()
    run = wandb.init(mode="offline", dir=tmp_path, project="test")
    try:
        directory = checkpoint.run_checkpoint_directory()
        assert pathlib.Path(directory) == pathlib.Path(run.dir) / checkpoint.RUN_CHECKPOINTS_DIRECTORY
        assert directory.is_dir()
        assert checkpoint.run_checkpoint_directory() == directory  # idempotent
    finally:
        run.finish()
