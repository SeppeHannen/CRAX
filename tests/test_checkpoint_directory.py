"""A run's checkpoints live under its W&B run directory and every one written there is
handed to the run for upload, so the policy travels with the run."""
from __future__ import annotations

import pathlib
from typing import List
from unittest import mock

import jax.numpy as jnp
import pytest
import wandb
from ml_collections import config_dict

from training import checkpoint

PARAMS = ({"weight": jnp.ones((2, 2))},)
CONFIG = config_dict.ConfigDict({"note": "test"})


def test_a_checkpoint_saved_in_the_run_directory_is_handed_to_the_run_and_one_elsewhere_is_not(tmp_path: pathlib.Path):
    run = wandb.init(mode="offline", dir=tmp_path, project="test")
    saved: List[dict] = []
    try:
        with mock.patch.object(run, "save", side_effect=lambda glob, **kwargs: saved.append({"glob": glob, **kwargs})):
            checkpoint.save(checkpoint.run_checkpoint_directory(), 7, PARAMS, CONFIG)
            checkpoint.save(tmp_path / "elsewhere", 7, PARAMS, CONFIG)
    finally:
        run.finish()
    assert len(saved) == 1
    written = pathlib.Path(run.dir) / checkpoint.RUN_CHECKPOINTS_DIRECTORY / "000000000007"
    assert saved[0] == {"glob": str(written / "**"), "base_path": str(pathlib.Path(run.dir).resolve()), "policy": "now"}
    assert (written / "config.json").is_file() and (tmp_path / "elsewhere" / "000000000007" / "config.json").is_file()


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
