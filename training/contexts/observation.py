"""The observation of a context run: the environment's sensors and the episode's
context, as two labelled parts.

A contextual CMDP puts the context ω in the state. Whether a network is allowed
to read it is the learner's decision, taken per network (the critics always,
the policy only when told; ``docs/acl/design/contextual_learner.md`` §2), so the
observation must carry ω as a part a network can name rather than as columns
appended to the sensor vector. :class:`ContextObservationWrapper` makes the
observation the dict

    {SENSORS_KEY: the wrapped environment's observation, CONTEXT_OBSERVATION_KEY: state.info["context"]}

at reset and after every step. It sits outermost in both the training and the
evaluation stack (``wrapper.py``), outside the auto-reset: on the step an
episode ends, the inner wrapper has already replaced the observation with the
fresh episode's and ``state.info["context"]`` with the fresh episode's ω, so the
two parts always describe the same episode. The environment itself is untouched
and a stock run still observes the flat vector.
"""
from __future__ import annotations

import jax
from crax.envs.base import State, Wrapper
from crax.envs.context import CONTEXT_KEY

SENSORS_KEY = "sensors"
CONTEXT_OBSERVATION_KEY = "context"


class ContextObservationWrapper(Wrapper):
    """``state.obs`` becomes ``{"sensors": inner obs, "context": state.info["context"]}``.

    The wrapped stack observes a flat vector (every registered suite does); the
    inner stack is stepped with that vector, as it was before the labelling.
    """

    def reset(self, rng: jax.Array) -> State:
        return _with_context_observation(self.env.reset(rng))

    def step(self, state: State, action: jax.Array) -> State:
        inner = state.replace(obs=state.obs[SENSORS_KEY])
        return _with_context_observation(self.env.step(inner, action))


def _with_context_observation(state: State) -> State:
    return state.replace(obs={SENSORS_KEY: state.obs, CONTEXT_OBSERVATION_KEY: state.info[CONTEXT_KEY]})
