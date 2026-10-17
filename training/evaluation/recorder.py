"""From an evaluation's recorded transitions and the learner's view of them to
:class:`EpisodeRecord` files on disk.

The Evaluator (``training/acting.py``) keeps the first ``num_episodes`` of its
environments' unrolls when given an :class:`EvaluationRecording`; the trainer
applies the agent's :class:`~training.evaluation.learner_view.LearnerViewFunction`
to them and hands both to :meth:`EpisodeRecorder.record`, which cuts each unroll
at the step its episode ended and writes one record per episode. The directory
is the one door: whoever owns it (the entry point puts it inside the W&B run
directory) decides where the records end up.

The recorder sits below the trainers (``training/acting.py`` imports this package),
so it knows nothing of the contexts package or the pixel wrapper: the labels it needs —
which part of the observation dict is the sensors, which parts are the camera frames
the policy saw, which ``state.info`` field holds the transition's context — are handed
in by whoever builds it, who knows the run.
"""
from __future__ import annotations

import dataclasses
import pathlib
from typing import Callable, List, Tuple

import jax
import numpy as np

from training.evaluation.learner_view import LearnerView
from training.evaluation.record import EpisodeRecord
from training.evaluation.scene import TRANSITION_SCENE_KEY
from training.types import Transition

COST_KEY = "cost"
TRUNCATION_KEY = "truncation"


@dataclasses.dataclass(frozen=True)
class EvaluationRecording:
    """What the Evaluator must keep of its unroll: how many environments, which ``state.info`` fields."""

    num_episodes: int
    extra_fields: Tuple[str, ...]

    def __post_init__(self) -> None:
        if self.num_episodes < 1:
            raise ValueError(f"EvaluationRecording.num_episodes must be at least 1, got {self.num_episodes}")


@dataclasses.dataclass(frozen=True)
class EpisodeRecorder:
    """Writes the first ``num_episodes`` episodes of every evaluation under ``directory``."""

    directory: pathlib.Path
    env_name: str
    run_name: str
    arm: str
    num_episodes: int
    sensors_key: str  # the part of the observation dict the record keeps as `observation`
    context_key: str  # the `state.info` field holding the context the transition started from
    image_keys: Tuple[str, ...]  # the parts of the observation dict that are camera frames; () for a state-based run
    written: Callable[[pathlib.Path], None]  # told of every record file once it exists (the run hands it to W&B)

    @property
    def recording(self) -> EvaluationRecording:
        """What the Evaluator has to keep for :meth:`record`: the cost and truncation the
        advantages need, and, both as the transition started from them, the scene for the
        renderer and the context the episode ran in."""
        return EvaluationRecording(self.num_episodes, (COST_KEY, TRUNCATION_KEY, TRANSITION_SCENE_KEY, self.context_key))

    def record(self, evaluation_name: str, environment_steps: int, unroll: Transition, view: LearnerView) -> List[pathlib.Path]:
        """``unroll``: the Evaluator's recorded transitions, leading axes ``[T, K]``;
        ``view``: the learner's view of the same transitions, ``[K, T]``. Writes one
        record per episode, cut at the step the episode ended, and returns the paths."""
        unroll = jax.tree_util.tree_map(np.asarray, unroll)
        view = jax.tree_util.tree_map(np.asarray, view)
        num_recorded = unroll.reward.shape[1]
        if num_recorded != self.num_episodes:
            raise ValueError(f"the Evaluator recorded {num_recorded} episodes; this recorder expects {self.num_episodes}")
        state_extras = unroll.extras["state_extras"]
        scene = state_extras[TRANSITION_SCENE_KEY]
        paths = []
        for k in range(num_recorded):
            length = _episode_length(unroll.discount[:, k])
            steps = slice(0, length)
            record = EpisodeRecord(
                env_name=self.env_name,
                run_name=self.run_name,
                arm=self.arm,
                evaluation_name=evaluation_name,
                environment_steps=environment_steps,
                episode_index=k,
                context=state_extras[self.context_key][0, k],  # one episode, one context: its first transition's
                qpos=scene.qpos[steps, k],
                qvel=scene.qvel[steps, k],
                mocap_pos=scene.mocap_pos[steps, k],
                mocap_quat=scene.mocap_quat[steps, k],
                observation=unroll.observation[self.sensors_key][steps, k],  # the context is its own field
                images={key: unroll.observation[key][steps, k] for key in self.image_keys},
                action=unroll.action[steps, k],
                reward=unroll.reward[steps, k],
                cost=state_extras[COST_KEY][steps, k],
                value=view.value[k, steps],
                cost_value=view.cost_value[k, steps],
                lagrange_multiplier=view.lagrange_multiplier[k, steps],
                reward_advantage=view.reward_advantage[k, steps],
                cost_advantage=view.cost_advantage[k, steps],
                lagrangian_advantage=view.lagrangian_advantage[k, steps],
            )
            path = record.save(self.directory)
            self.written(path)
            paths.append(path)
        return paths


def _episode_length(discount: np.ndarray) -> int:
    """Steps up to and including the first ``done`` (``discount == 0``); all of them if none."""
    ended = np.flatnonzero(discount == 0.0)
    return int(ended[0]) + 1 if ended.size else int(discount.shape[0])
