"""A checkpoint loads as NumPy on any machine, with the normaliser's state rebuilt."""
from __future__ import annotations

import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ml_collections import config_dict

from training import checkpoint
from training.acme import running_statistics

CONFIG = config_dict.ConfigDict({"note": "test"})


def test_a_saved_checkpoint_loads_as_numpy_with_the_normaliser_state_rebuilt(tmp_path: pathlib.Path):
    """The saved tree is (normaliser state, policy, value); `load` restores every array as NumPy
    (a GPU-trained checkpoint must open on a CPU machine) and rebuilds the normaliser's dataclass."""
    normaliser = running_statistics.init_state(jnp.zeros(3))
    policy = {"dense": {"kernel": jnp.arange(6.0).reshape(2, 3), "bias": jnp.ones(3)}}
    value = {"dense": {"kernel": jnp.full((3, 1), 2.0)}}
    checkpoint.save(tmp_path, 42, (normaliser, policy, value), CONFIG)

    restored = checkpoint.load(tmp_path / "000000000042")
    assert isinstance(restored[0], running_statistics.RunningStatisticsState)
    assert all(isinstance(leaf, np.ndarray) for leaf in jax.tree_util.tree_leaves(restored))
    np.testing.assert_array_equal(restored[1]["dense"]["kernel"], np.arange(6.0).reshape(2, 3))
    np.testing.assert_array_equal(restored[2]["dense"]["kernel"], np.full((3, 1), 2.0))
    np.testing.assert_array_equal(restored[0].std, np.ones(3))
    assert checkpoint.load_config(tmp_path / "000000000042" / "config.json").note == "test"
    with pytest.raises(ValueError, match="does not exist"):
        checkpoint.load(tmp_path / "000000000043")
