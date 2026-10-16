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
from training.contexts import suite_contexts
from training.contexts.constraints.additive import AdditiveConstraint, AdditiveMultiplierState

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


def test_the_learner_state_round_trips_in_its_own_containers_next_to_the_policy_tuple(tmp_path: pathlib.Path):
    """`save_learner_state` writes the agent's extra state under `<step>/learner`, leaving the
    policy tuple as `load` expects it; `load_learner_state` restores into the structure the reader
    hands it (a dataclass multiplier state stays a dataclass), every array NumPy."""
    constraint = AdditiveConstraint(suite_contexts("safe_goal_point").space)
    multipliers = constraint.initial_multipliers(0.3)
    cost_value = {"dense": {"kernel": jnp.full((3, 1), 2.0)}}
    normaliser = running_statistics.init_state(jnp.zeros(3))
    checkpoint.save(tmp_path, 7, (normaliser, {}, {}), CONFIG)
    checkpoint.save_learner_state(tmp_path, 7, {"cost_value": cost_value, "aux_state": multipliers})

    assert len(checkpoint.load(tmp_path / "000000000007")) == 3, "the policy tuple keeps its shape"
    like = {"cost_value": cost_value, "aux_state": constraint.initial_multipliers(0.0)}
    restored = checkpoint.load_learner_state(tmp_path / "000000000007", like)
    assert isinstance(restored["aux_state"], AdditiveMultiplierState)
    assert all(isinstance(leaf, np.ndarray) for leaf in jax.tree_util.tree_leaves(restored))
    np.testing.assert_array_equal(restored["cost_value"]["dense"]["kernel"], np.full((3, 1), 2.0))
    np.testing.assert_allclose(restored["aux_state"].baseline, 0.3)
    checkpoint.save(tmp_path, 8, (normaliser, {}, {}), CONFIG)  # a policy-only checkpoint
    with pytest.raises(ValueError, match="no learner state"):
        checkpoint.load_learner_state(tmp_path / "000000000008", like)
