"""The frame the policy itself saw at a step, from the record's image parts.

A vision run's record keeps, per step, the camera frames the policy was given
(``EpisodeRecord.images``): the wrapper's frame stack, ``frame_stack`` RGB frames
side by side on the channel axis, oldest first. The page shows them as one strip,
oldest on the left, newest on the right, enlarged by a whole factor so a 64-pixel
frame is legible next to the scene renderer's 640-pixel frames. PNG, not JPEG: the
point is to see exactly what the network got.
"""
from __future__ import annotations

import io

import numpy as np
from PIL import Image

from training.evaluation.record import EpisodeRecord

CHANNELS = 3  # RGB per stacked frame
SCALE = 4


def policy_view(record: EpisodeRecord, step: int, part: str) -> bytes:
    """The PNG of what ``record``'s policy saw through image part ``part`` at ``step``."""
    if part not in record.images:
        raise ValueError(f"the record has no image part {part!r}; it has {tuple(record.images)}")
    if not 0 <= step < record.length:
        raise IndexError(f"step {step} outside the episode's {record.length} steps")
    stacked = record.images[part][step]  # [H, W, CHANNELS * frame_stack]
    if stacked.shape[-1] % CHANNELS:
        raise ValueError(f"image part {part!r} has {stacked.shape[-1]} channels, not a multiple of {CHANNELS}")
    frames = np.split(stacked, stacked.shape[-1] // CHANNELS, axis=-1)  # oldest first
    strip = np.concatenate(frames, axis=1)
    enlarged = np.repeat(np.repeat(strip, SCALE, axis=0), SCALE, axis=1)
    buffer = io.BytesIO()
    Image.fromarray(enlarged).save(buffer, format="PNG")
    return buffer.getvalue()
