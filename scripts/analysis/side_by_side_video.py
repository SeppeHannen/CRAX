"""Two recorded episodes, side by side, as one MP4 with a running caption.

    .venv/bin/python -m scripts.analysis.side_by_side_video \\
        --left  "scalar λ, blind policy"            <record.episode.npz> \\
        --right "λ(ω) additive, told, surface lidar" <record.episode.npz> \\
        --output <video.mp4> [--camera fixedfar] [--fps 25] [--stride 2]

Both records are drawn by the review page's renderer (``training/review/renderer.py``), frame
for frame, from each camera in ``--cameras`` stacked top to bottom (default: the arena overview
``fixedfar`` and ``above``, a free camera that follows the agent from steeply above, so an agent
wedged between tall solids is not hidden behind them as it is from the model's ``track``
camera); a shorter episode holds its last frame. The agent is drawn blue so it stands out from
the red hazards. Each column carries its label, the step, the
goals reached so far and the cost so far; a step that costs 3 or more (solid contact on
goal-point) flashes the frame's border red. ``--stride`` keeps every n-th step (1000-step
episodes at stride 2 and 25 fps play in 20 s).

For the presentation of 2026-10-21: the published baseline against the best contextual
policy on the same level, the episodes chosen from the replay tables.
"""
from __future__ import annotations

import argparse
import io
import pathlib
from typing import Sequence, Tuple

import mujoco
import numpy as np
from imageio import v3 as iio
from PIL import Image, ImageDraw, ImageFont

from training.evaluation import EpisodeRecord
from training.review.renderer import SceneRenderer

SOLID_CONTACT_COST = 3.0
CAPTION_HEIGHT = 56
AGENT_GEOMS = ("agent", "pointarrow")
AGENT_RGBA = (0.12, 0.35, 0.95, 1.0)
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"  # has λ and ω
ABOVE = "above"  # the follow-from-above view this script adds to the model's cameras
ABOVE_DISTANCE, ABOVE_ELEVATION, ABOVE_AZIMUTH = 2.6, -80.0, 90.0


def follow_from_above(record: EpisodeRecord, step: int) -> mujoco.MjvCamera:
    """A free camera looking down at the agent (the point agent's x, y are ``qpos[0:2]``)."""
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = (float(record.qpos[step, 0]), float(record.qpos[step, 1]), 0.0)
    camera.distance, camera.elevation, camera.azimuth = ABOVE_DISTANCE, ABOVE_ELEVATION, ABOVE_AZIMUTH
    return camera


def colour_the_agent(renderer: SceneRenderer) -> None:
    """Draw the agent's geoms in :data:`AGENT_RGBA`; the hazards are red and so is the agent by default."""
    for name in AGENT_GEOMS:
        geom = mujoco.mj_name2id(renderer.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if geom < 0:
            raise ValueError(f"{renderer.env_name} has no geom {name!r}")
        renderer.model.geom_rgba[geom] = AGENT_RGBA


def caption(frames: Sequence[np.ndarray], label: str, record: EpisodeRecord, step: int, flash: bool) -> Image.Image:
    """One policy's column: a caption band, then its camera frames stacked; a red border on a costly step."""
    goals = int(round(float(record.reward[: step + 1].sum())))
    cost = float(record.cost[: step + 1].sum())
    width, height = frames[0].shape[1], sum(frame.shape[0] for frame in frames)
    image = Image.new("RGB", (width, height + CAPTION_HEIGHT), "#111")
    y = CAPTION_HEIGHT
    for frame in frames:
        image.paste(Image.fromarray(frame), (0, y))
        y += frame.shape[0]
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(FONT, 18)
    draw.text((10, 6), label, fill="#fff", font=font)
    draw.text((10, 30), f"step {step:4d}   goals {goals:2d}   cost {cost:6.1f}", fill="#ddd", font=font)
    if flash:
        draw.rectangle([0, CAPTION_HEIGHT, width - 1, image.height - 1], outline="#e03030", width=6)
    return image


def side_by_side(
    renderer: SceneRenderer, left: Tuple[str, EpisodeRecord], right: Tuple[str, EpisodeRecord], cameras: Sequence[str], stride: int
) -> np.ndarray:
    """``[T, H, W, 3]`` frames, both episodes at the same steps, the shorter holding its last frame."""
    steps = range(0, max(left[1].length, right[1].length), stride)
    frames = []
    for step in steps:
        columns = []
        for label, record in (left, right):
            own_step = min(step, record.length - 1)
            views = [
                np.asarray(Image.open(io.BytesIO(renderer.frame(
                    record, own_step, follow_from_above(record, own_step) if camera == ABOVE else camera
                ))))
                for camera in cameras
            ]
            flash = bool(record.cost[own_step] >= SOLID_CONTACT_COST)
            columns.append(np.asarray(caption(views, label, record, own_step, flash)))
        frames.append(np.concatenate(columns, axis=1))
    return np.stack(frames)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--left", nargs=2, metavar=("LABEL", "RECORD"), required=True)
    parser.add_argument("--right", nargs=2, metavar=("LABEL", "RECORD"), required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--cameras", nargs="+", default=("fixedfar", ABOVE),
                        help=f"the model's cameras or {ABOVE!r}, stacked top to bottom per policy")
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--stride", type=int, default=2, help="keep every n-th step")
    arguments = parser.parse_args()
    left = (arguments.left[0], EpisodeRecord.load(pathlib.Path(arguments.left[1])))
    right = (arguments.right[0], EpisodeRecord.load(pathlib.Path(arguments.right[1])))
    if left[1].env_name != right[1].env_name:
        raise ValueError(f"the records are of different environments: {left[1].env_name} and {right[1].env_name}")
    renderer = SceneRenderer(left[1].env_name)
    try:
        colour_the_agent(renderer)
        frames = side_by_side(renderer, left, right, tuple(arguments.cameras), arguments.stride)
    finally:
        renderer.close()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(arguments.output, frames, fps=arguments.fps, codec="libx264")
    print(f"wrote {arguments.output}: {frames.shape[0]} frames, {frames.shape[2]}x{frames.shape[1]}, "
          f"left {left[1].episode_return:.0f} goals / cost {left[1].episode_cost:.0f}, right {right[1].episode_return:.0f} / {right[1].episode_cost:.0f}")


if __name__ == "__main__":
    main()
