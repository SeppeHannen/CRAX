"""A run's pixel observation: which camera the policy sees, at what size, under what name.

``GpuPixelObservationWrapper`` (``crax/envs/wrappers/pixel_observation_gpu.py``) renders a
camera with MJWarp and puts the frames under ``pixels/<camera>`` in the observation dict;
the vision encoders (``training/networks.py``, ``training/contexts/networks.py``) pick the
pixel parts out by that prefix. The wrapper module needs Warp and a GPU to import, so what
can be said about a run's pixels without rendering — the part's name, the wrapper's
arguments — is said here, from the run's flags, for the entry point, the recorder and the
replay alike.
"""
from __future__ import annotations

import argparse
import dataclasses
from typing import Any, Dict, Optional

PIXELS_PREFIX = "pixels/"


def pixel_observation_key(camera: str) -> str:
    """The observation part holding ``camera``'s frames: ``pixels/<camera>``."""
    return f"{PIXELS_PREFIX}{camera}"


def is_pixel_part(part: str) -> bool:
    return part.startswith(PIXELS_PREFIX)


@dataclasses.dataclass(frozen=True)
class PixelObservation:
    """What the policy sees through the camera, as the run's ``--vision_*`` flags set it."""

    camera: str
    height: int
    width: int
    obs_mode: str  # 'pixels' or 'pixels+state'
    frame_stack: int

    @classmethod
    def from_config(cls, config: argparse.Namespace) -> Optional["PixelObservation"]:
        """From the run's flags (the CLI's namespace, or a W&B run's config as a namespace);
        ``None`` for a run without ``--vision``."""
        if not config.vision:
            return None
        return cls(
            camera=config.vision_camera, height=config.vision_height, width=config.vision_width,
            obs_mode=config.vision_obs_mode, frame_stack=config.vision_frame_stack,
        )

    @property
    def key(self) -> str:
        """The observation part the frames are under."""
        return pixel_observation_key(self.camera)

    def wrapper_kwargs(self) -> Dict[str, Any]:
        """``GpuPixelObservationWrapper``'s arguments except ``num_envs``, which the trainer
        threads in per environment stack."""
        return dict(camera=self.camera, height=self.height, width=self.width, obs_mode=self.obs_mode, frame_stack=self.frame_stack)
