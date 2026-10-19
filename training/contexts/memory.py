"""A fixed-form memory for a feed-forward policy: an exponential trace of what the
agent sensed, did and was charged, as a third labelled part of the observation.

The failure this is for (``docs/acl/README.md``, plan task 1b): on goal-point
the catastrophic episode is a limit cycle in a gap too narrow to pass — the
agent touches a solid hazard, is pushed back, re-approaches, touches again,
for hundreds of steps. Measured on 384 replayed level-3 episodes (2026-10-19):
contact is intermittent (the longest contiguous contact in the worst episode is
10 steps; the median interval between touches is 32 steps), so the evidence
that the agent is stuck is never in the present observation. A reactive policy
re-decides on every step on which the bump has passed, and decides the same.

What is missing is **accumulation**: that there have been many touches, that
the lidar has not changed, that the commanded speed has not become actual
speed. One exponential trace carries all three. With ``x_t`` the vector
``(sensors_t, action_{t−1}, cost_t)`` — the observation the step produced, the
action that produced it, the cost it was charged — and time constant ``τ``,

    m_t = (1 − 1/τ) m_{t−1} + (1/τ) x_t,        m_0 = 0 at the start of an episode.

A single event decays as ``exp(−t/τ)``, so ``τ`` is the trace's half-weight
point, not a window: five touches within 50 steps leave the cost trace at
``τ = 32`` ten times above the running disc-proximity floor, and a lone touch is
back at the floor after ~100 steps, which is the measured distinction between
a brush and a trap. A frame stack of the last ``k`` observations is what this
replaces: a stack forgets at step ``k + 1``, and at ``k = 8`` would hold the
previous touch in one re-approach of three.

:class:`MemoryObservationWrapper` sits outermost in both the training and the
evaluation stack (``wrapper.py``), outside the context labelling: it reads the
inner stack's ``obs["sensors"]``, the step's action and ``state.info["cost"]``,
carries the trace as ``obs["memory"]`` per slot (the observation is its only
home), and zeroes it where an episode ended (the inner auto-reset has already
replaced that slot's observation with a fresh episode's first). The policy and
the critics read it like any other part (``networks.py``); the trainer's running
normaliser scales it like the sensors.

The trace includes the cost, which a deployed robot is not told; its physical
proxy, the accelerometer's spike at a touch, is in the sensors block of the same
trace. Which of the two the policy uses is a question for the records, and the
inclusion is flagged (``docs/acl/README.md``, *To raise with Tristan*).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from crax.envs.base import Env, State, Wrapper

from training.contexts.observation import SENSORS_KEY

MEMORY_OBSERVATION_KEY = "memory"
COST_KEY = "cost"


class MemoryObservationWrapper(Wrapper):
    """``state.obs`` gains ``"memory"``: the exponential trace of ``(sensors, action, cost)``.

    ``env`` is the batched context stack whose observation is already the dict
    ``{"sensors": ..., "context": ...}``; the leading axis of every array is the slot axis.
    """

    def __init__(self, env: Env, time_constant: int):
        super().__init__(env)
        if time_constant < 2:
            raise ValueError(f"a memory trace needs a time constant of at least 2 steps, got {time_constant}")
        self.time_constant = time_constant

    @property
    def decay(self) -> float:
        return 1.0 - 1.0 / self.time_constant

    def reset(self, rng: jax.Array) -> State:
        state = self.env.reset(rng)
        trace = jnp.zeros(self._trace_shape(state), dtype=jnp.float32)
        return self._with_memory(state, trace)

    def step(self, state: State, action: jax.Array) -> State:
        inner_obs = {part: values for part, values in state.obs.items() if part != MEMORY_OBSERVATION_KEY}
        stepped = self.env.step(state.replace(obs=inner_obs), action)
        observed = self._trace_input(stepped.obs[SENSORS_KEY], action, stepped.info[COST_KEY])
        trace = self.decay * state.obs[MEMORY_OBSERVATION_KEY] + (1.0 - self.decay) * observed
        # Where the episode ended the inner stack has started a fresh one; its memory starts empty.
        done = jnp.reshape(stepped.done, (-1, 1)).astype(bool)
        trace = jnp.where(done, jnp.zeros_like(trace), trace)
        return self._with_memory(stepped, trace)

    @staticmethod
    def _trace_input(sensors: jax.Array, action: jax.Array, cost: jax.Array) -> jax.Array:
        """``x_t = (sensors_t, action_{t−1}, cost_t)`` per slot, ``[N, S + A + 1]``."""
        return jnp.concatenate([sensors, action, cost[:, None]], axis=-1).astype(jnp.float32)

    def _trace_shape(self, state: State) -> tuple[int, int]:
        num_slots, sensors = state.obs[SENSORS_KEY].shape
        return num_slots, sensors + self.env.action_size + 1

    @staticmethod
    def _with_memory(state: State, trace: jax.Array) -> State:
        return state.replace(obs={**state.obs, MEMORY_OBSERVATION_KEY: trace})
