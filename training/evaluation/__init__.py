"""Recording the evaluation episodes a run already simulates, with the learner's
view of every step (``docs/acl/design/episode_review.md``).

* :mod:`scene` — the simulator fields that reconstruct a frame, written into the
  state by :class:`SceneWrapper` so the trainer records them like any other field.
* :mod:`learner_view` — what the agent can say about an episode: its critics'
  values, its multiplier and the advantages its loss would compute, as a typed
  struct and the function an agent hands the trainer to produce it.
* :mod:`record` — :class:`EpisodeRecord`, the artifact one recorded episode becomes.
* :mod:`recorder` — :class:`EpisodeRecorder`, which turns an evaluation's recorded
  transitions and the learner's view into records on disk.
"""

from training.evaluation.learner_view import LearnerView, LearnerViewFunction
from training.evaluation.record import RECORDS_DIRECTORY, EpisodeRecord, records_under
from training.evaluation.recorder import EpisodeRecorder, EvaluationRecording
from training.evaluation.scene import TRANSITION_SCENE_KEY, Scene, SceneWrapper

__all__ = [
    "EpisodeRecord",
    "EpisodeRecorder",
    "EvaluationRecording",
    "LearnerView",
    "LearnerViewFunction",
    "RECORDS_DIRECTORY",
    "TRANSITION_SCENE_KEY",
    "Scene",
    "SceneWrapper",
    "records_under",
]
