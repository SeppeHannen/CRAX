"""The DrQ random translation of pixel observations (``training/agents/ppo/train.py``): one
shift per trajectory held across its steps, applied to each minibatch inside the SGD scan.

Run with: JAX_PLATFORMS=cpu pytest tests/test_pixel_augmentation.py -v
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from training.agents.ppo.train import _random_translate_pixels

PADDING = 4


def _frames(batch: int, steps: int, size: int) -> jnp.ndarray:
    """Distinct frames: pixel value = row index, so a shift is readable from the result."""
    rows = jnp.arange(size, dtype=jnp.uint8)[:, None, None]
    return jnp.broadcast_to(rows, (batch, steps, size, size, 3))


def test_translation_shifts_each_trajectory_by_its_own_offset_and_leaves_non_pixel_parts_alone():
    observation = {"pixels/vision": _frames(batch=6, steps=3, size=16), "sensors": jnp.ones((6, 3, 5))}
    out = _random_translate_pixels(observation, jax.random.PRNGKey(0))
    np.testing.assert_array_equal(out["sensors"], observation["sensors"])
    shifted = np.asarray(out["pixels/vision"])
    assert shifted.shape == (6, 3, 16, 16, 3) and shifted.dtype == np.uint8
    # the same shift for every step of a trajectory: the frames within a trajectory are identical
    for k in range(6):
        for t in range(1, 3):
            np.testing.assert_array_equal(shifted[k, t], shifted[k, 0])
    # a vertical shift moves the row-index pattern by a whole number of rows within the padding
    offsets = {int(shifted[k, 0, PADDING, 0, 0]) - PADDING for k in range(6)}
    assert offsets <= set(range(-PADDING, PADDING + 1))
    assert len(offsets) > 1, "six trajectories drew the same shift"


def test_translating_a_minibatch_draws_from_the_same_distribution_as_translating_the_round():
    """The shift is drawn per trajectory from one key split per trajectory; whether the
    trajectories arrive as one round or as its minibatches changes which key each gets, not
    the distribution: the set of offsets over many trajectories matches."""
    frames = _frames(batch=4096, steps=1, size=16)
    whole = np.asarray(_random_translate_pixels({"pixels/vision": frames}, jax.random.PRNGKey(1))["pixels/vision"])
    pieces = [
        np.asarray(_random_translate_pixels({"pixels/vision": frames[i : i + 1024]}, jax.random.PRNGKey(10 + i))["pixels/vision"])
        for i in range(0, 4096, 1024)
    ]
    by_minibatch = np.concatenate(pieces)
    offset = lambda shifted: shifted[:, 0, PADDING, 0, 0].astype(int) - PADDING  # vertical offset per trajectory
    histogram = lambda offsets: np.bincount(offsets + PADDING, minlength=2 * PADDING + 1) / offsets.size
    np.testing.assert_allclose(histogram(offset(whole)), histogram(offset(by_minibatch)), atol=0.03)
    np.testing.assert_allclose(histogram(offset(whole)), np.full(2 * PADDING + 1, 1 / (2 * PADDING + 1)), atol=0.03)
