"""The simulator fields that reconstruct a frame, carried in the state so the
trainer records them like any other ``state.info`` field.

MuJoCo rebuilds every body's pose from the generalised coordinates ``qpos`` and
the mocap bodies' ``mocap_pos`` / ``mocap_quat`` with one ``mj_forward``; that
is what the renderer (``crax/io/image.py``) does from a pipeline state. The
CRAX suites place every goal and hazard as a mocap body, so these four arrays
are the whole scene.

:class:`SceneWrapper` writes into ``state.info[TRANSITION_SCENE_KEY]`` the scene
of the state the **last transition started from** — after a reset, the reset
state's own. The trainer records a transition's fields from the state *after*
the step, so this is how the scene of ``s_t`` lines up with the observation
``o_t``, the action ``a_t`` and the critics' ``V(s_t)`` in the same row of a
record; the convention is the context wrapper's ``transition_context``.

The four arrays are fields of the MJX pipeline state. Brax's other backends
(``generalized``, ``spring``, ``positional``) have no mocap bodies, so a suite on
them has no scene in this sense; the wrapper refuses such an environment.
"""
from __future__ import annotations

import flax.struct
import jax
from crax.envs.base import Env, State, Wrapper

TRANSITION_SCENE_KEY = "transition_scene"
MJX_BACKEND = "mjx"


@flax.struct.dataclass
class Scene:
    """What ``mj_forward`` needs to rebuild one frame. Batched like the state: ``[..., n]``."""

    qpos: jax.Array  # [..., nq]
    qvel: jax.Array  # [..., nv]
    mocap_pos: jax.Array  # [..., nmocap, 3]
    mocap_quat: jax.Array  # [..., nmocap, 4]

    @classmethod
    def of(cls, state: State) -> "Scene":
        pipeline = state.pipeline_state
        return cls(qpos=pipeline.qpos, qvel=pipeline.qvel, mocap_pos=pipeline.mocap_pos, mocap_quat=pipeline.mocap_quat)


class SceneWrapper(Wrapper):
    """After a reset, ``info[TRANSITION_SCENE_KEY]`` is the reset state's scene; after a
    step, the scene of the state the step was taken from.

    Outermost in the stack, so the scene it reads is the one the observation describes:
    after an auto-reset the inner wrapper has already swapped in the fresh episode's state.
    """

    def __init__(self, env: Env):
        super().__init__(env)
        if env.backend != MJX_BACKEND:
            raise ValueError(
                f"recording a scene needs the {MJX_BACKEND!r} backend (its pipeline state carries qpos, qvel and the "
                f"mocap bodies); {type(env.unwrapped).__name__} runs on {env.backend!r}"
            )

    def reset(self, rng: jax.Array) -> State:
        state = self.env.reset(rng)
        return _with_scene(state, Scene.of(state))

    def step(self, state: State, action: jax.Array) -> State:
        info = dict(state.info)
        del info[TRANSITION_SCENE_KEY]  # the inner stack never sees it, as it never wrote it
        return _with_scene(self.env.step(state.replace(info=info), action), Scene.of(state))


def _with_scene(state: State, scene: Scene) -> State:
    return state.replace(info={**state.info, TRANSITION_SCENE_KEY: scene})
