"""One frame of a recorded episode, from its scene, through MuJoCo.

The record holds ``qpos``, ``qvel`` and the mocap bodies' poses per step
(``training/evaluation/scene.py``); ``mj_forward`` on the suite's model rebuilds
every body from them, and MuJoCo's renderer draws it from one of the model's
cameras. The model is the one the environment builds — for a context suite the
union model, the same for every level — so one renderer serves every record of
an environment. Frames are JPEG bytes for the page.
"""
from __future__ import annotations

import io
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Tuple

import mujoco
import numpy as np
from PIL import Image

from crax import envs
from training.evaluation.record import EpisodeRecord

DEFAULT_CAMERA = "fixedfar"
WIDTH, HEIGHT = 640, 480


class SceneRenderer:
    """Renders frames of one environment's records.

    MuJoCo's renderer owns an OpenGL context, and a context can be current in one
    thread only; the HTTP server answers requests from many. Everything that touches
    the context, construction included, therefore runs on one dedicated thread.
    """

    def __init__(self, env_name: str, width: int = WIDTH, height: int = HEIGHT):
        self.env_name = env_name
        environment = envs.get_environment(env_name, level=1)
        self.model: mujoco.MjModel = environment.sys.mj_model
        self.control_dt = float(environment.dt)
        self.cameras: Tuple[str, ...] = tuple(
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_CAMERA, index) for index in range(self.model.ncam)
        )
        if DEFAULT_CAMERA not in self.cameras:
            raise ValueError(f"{env_name} has no camera {DEFAULT_CAMERA!r}; it has {self.cameras}")
        self._data = mujoco.MjData(self.model)
        self._render_thread = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"render-{env_name}")
        self._renderer: mujoco.Renderer = self._render_thread.submit(mujoco.Renderer, self.model, height=height, width=width).result()

    def frame(self, record: EpisodeRecord, step: int, camera: str) -> bytes:
        """The JPEG of ``record`` at ``step`` from ``camera``."""
        if record.env_name != self.env_name:
            raise ValueError(f"this renderer draws {self.env_name}; the record is of {record.env_name}")
        if not 0 <= step < record.length:
            raise IndexError(f"step {step} outside the episode's {record.length} steps")
        if camera not in self.cameras:
            raise ValueError(f"no camera {camera!r}; {self.env_name} has {self.cameras}")
        pixels = self._render_thread.submit(self._render, record, step, camera).result()
        buffer = io.BytesIO()
        Image.fromarray(pixels).save(buffer, format="JPEG", quality=88)
        return buffer.getvalue()

    def _render(self, record: EpisodeRecord, step: int, camera: str) -> np.ndarray:
        self._data.qpos[:] = record.qpos[step]
        self._data.qvel[:] = record.qvel[step]
        self._data.mocap_pos[:] = record.mocap_pos[step]
        self._data.mocap_quat[:] = record.mocap_quat[step]
        mujoco.mj_forward(self.model, self._data)
        self._renderer.update_scene(self._data, camera=camera)
        return np.array(self._renderer.render())

    def close(self) -> None:
        self._render_thread.submit(self._renderer.close).result()
        self._render_thread.shutdown()


class Renderers:
    """A :class:`SceneRenderer` per environment, built on first use."""

    def __init__(self) -> None:
        self._renderers: Dict[str, SceneRenderer] = {}
        self._lock = threading.Lock()

    def for_environment(self, env_name: str) -> SceneRenderer:
        with self._lock:
            if env_name not in self._renderers:
                self._renderers[env_name] = SceneRenderer(env_name)
            return self._renderers[env_name]

    def close(self) -> None:
        for renderer in self._renderers.values():
            renderer.close()
