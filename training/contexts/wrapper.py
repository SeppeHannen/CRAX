"""Environment wrapper that gives every parallel slot its own context and a fresh
episode on termination.

Replaces the Brax ``AutoResetWrapper`` for context-conditioned training. The
stock wrapper never calls ``reset()`` after the first one: it caches the initial
state per slot and copies it back whenever ``done`` fires, so a slot replays
the *same* layout for the whole run. This wrapper instead, on every step:

1. draws a fresh context for every slot from the distribution parameters φ
   that were handed in for this round (``state.info["distribution_parameters"]``,
   one value for the whole population — lifted out of the state while the inner,
   vmapped stack steps, since every array that stack sees must have the slot
   axis first; see ``DistributionParameters`` in ``distribution.py``);
2. runs the environment's ``reset()`` for every slot with that context;
3. keeps the fresh state where ``done`` is set and the stepped state elsewhere
   (``jnp.where``), and likewise keeps the fresh context only for done slots.

Every slot therefore starts each episode in a newly sampled ω with a newly
sampled layout, and ``state.info["context"]`` always holds the context of the
episode currently running in that slot. ``state.info["transition_context"]``
holds the context the *last transition* was generated in: the two differ only
on the step an episode ends, where ``context`` is already the next episode's ω
while ``transition_context`` is the finished episode's. The trainer collects
``transition_context`` per transition, so the realised curriculum q̂ and the
per-episode feedback (read at ``episode_done`` transitions, alongside
``episode_metrics``) both refer to the context the data actually came from.

Cost: the reset runs for all slots on every step whether or not any slot is
done — that is how per-slot conditionals work on a GPU (see
``docs/acl/design/per_slot_constraints.md``). How expensive that is depends on
the suite's ``reset()``; it is the first thing to measure with
``--measure_performance``.

Phase (**training only**): the first reset gives every slot a random head
start on its step counter, so the first episodes end at different rounds and
the population is spread over the episode from then on. Without it, on a suite
whose episodes always run to the limit, all slots reset together for the whole
run. An *evaluation* episode must run its full length from step 0, so the
evaluation stack (:func:`wrap_for_context_evaluation`) never spreads; the two
stacks differ in nothing else. Three things follow from spreading, and one does
not:

- every round's PPO batch is a sample of the episode, not one slice of it;
- a new distribution parameter acts at once: ~``num_slots × steps_per_round /
  episode_length`` slots reset every round and draw from it, so q̂ ramps
  towards q from the next round instead of jumping to it up to an episode later;
- completed-episode feedback arrives every round in a steady stream instead of
  all at once every ``episode_length / steps_per_round`` rounds;
- feedback is **not** fresher: an episode's outcome is known when it ends, one
  episode length after its context was drawn, in either regime.

Decision of 2026-10-06; the lockstep run is
``docs/acl/experiments/2026-10-06_goal_point_staged_vs_uniform.md``.

Wrapper order follows Brax exactly, with this class in place of
``AutoResetWrapper``::

    AutoReset( Episode( Vmap( env ) ) )   ->   ContextualAutoReset( Episode( Vmap( env ) ) )

so the batched ``done`` (including truncation at ``episode_length``) and the
episode bookkeeping (``episode_metrics``, ``episode_done``) that the trainer and
PPO-Lagrange read are untouched. The env's ``reset`` is therefore called via
``jax.vmap`` here, with one key and one context per slot. See
:func:`wrap_for_context_training`.
"""
from __future__ import annotations

import functools
from typing import Callable, Optional

import jax
import jax.numpy as jnp
from crax.envs.base import Env, State, Wrapper
from crax.envs.wrappers.training import EpisodeWrapper, VmapWrapper

from crax.envs.context import CONTEXT_KEY

from training.contexts.distribution import ContextDistribution, DistributionParameters
from training.contexts.rollout import SLOT_INDEX_KEY, TRANSITION_CONTEXT_KEY
from training.contexts.space import Contexts

DISTRIBUTION_PARAMETERS_KEY = "distribution_parameters"
RNG_KEY = "context_rng"


class ContextualAutoResetWrapper(Wrapper):
    """Per-slot context + reset-on-done, on a *batched* environment.

    Expects the wrapped env to be ``EpisodeWrapper(VmapWrapper(env))``; the
    leading axis of every array is the slot axis.
    """

    def __init__(self, env: Env, distribution: ContextDistribution, spread_initial_phase: bool):
        """``spread_initial_phase``: give each slot a random head start on its step
        counter at the first reset (module docstring). True for the training
        population, False wherever every episode must run its full length."""
        super().__init__(env)
        self.distribution = distribution
        self.space = distribution.space
        self.spread_initial_phase = spread_initial_phase
        # The single-slot env that VmapWrapper batches (adapters included, so
        # fields like `cost` are still added). We vmap its reset ourselves so a
        # per-slot context can travel alongside the per-slot key.
        self._single_slot_env = _find_vmapped_env(env)
        self._episode_length = _find_episode_wrapper(env).episode_length

    # ---- reset ------------------------------------------------------------- #

    def reset(self, rng: jax.Array) -> State:
        """Initial reset with contexts drawn from ``distribution.initialise``.

        ``rng`` has shape ``[num_slots, 2]`` (as VmapWrapper expects). The
        round-0 parameters are attached to the state; the trainer replaces them
        between rounds with :func:`attach_parameters`.
        """
        return self.reset_with_parameters(rng, self.distribution.initialise())

    def reset_with_parameters(self, rng: jax.Array, parameters: DistributionParameters) -> State:
        num_slots = rng.shape[0]
        keys = jax.vmap(lambda k: jax.random.split(k, 4))(rng)  # [N, 4, 2]
        sample_key, reset_keys, next_rng, phase_key = keys[0, 0], keys[:, 1], keys[:, 2], keys[0, 3]
        contexts = self.distribution.sample(parameters, sample_key, num_slots)
        state = self._reset_in_contexts(reset_keys, contexts)
        if self.spread_initial_phase:
            # Slot i's first episode is cut short by its offset, after which the
            # population stays desynchronised (module docstring).
            state.info["steps"] = jax.random.randint(phase_key, (num_slots,), 0, self._episode_length).astype(state.info["steps"].dtype)
        state.info[CONTEXT_KEY] = contexts
        state.info[TRANSITION_CONTEXT_KEY] = contexts
        # Each slot carries its own index so the host can lay the recorded transitions
        # out per slot without knowing how the trainer orders them (rollout.py).
        state.info[SLOT_INDEX_KEY] = jnp.arange(num_slots, dtype=jnp.int32)
        state.info[DISTRIBUTION_PARAMETERS_KEY] = parameters  # one copy per run, not per slot (see `step`)
        state.info[RNG_KEY] = next_rng
        return state

    def _reset_in_contexts(self, keys: jax.Array, contexts: Contexts) -> State:
        """Batched reset with one context per slot: the suite's ``reset_with_context``
        under ``vmap`` (crax/envs/context.py). EpisodeWrapper's bookkeeping fields
        are recreated explicitly, exactly as its ``reset`` would."""
        state = jax.vmap(self._single_slot_env.reset_with_context)(keys, contexts)
        return _add_episode_fields(state, keys)

    # ---- step -------------------------------------------------------------- #

    def step(self, state: State, action: jax.Array) -> State:
        # Identical to AutoResetWrapper.step up to the point of choosing the
        # replacement state: reset step counters for slots that were done, clear
        # `done`, step.
        if "steps" in state.info:
            steps = state.info["steps"]
            state.info.update(steps=jnp.where(state.done, jnp.zeros_like(steps), steps))
        # Every array in the state has the slot axis first except φ, which is one
        # value for the whole population (DistributionParameters). VmapWrapper runs
        # the single-slot step once per row of every array it is given, so φ must
        # not be in the state it sees. Nothing below this wrapper reads φ.
        info_for_step = dict(state.info)
        parameters = info_for_step.pop(DISTRIBUTION_PARAMETERS_KEY)
        state = state.replace(done=jnp.zeros_like(state.done), info=info_for_step)
        stepped = self.env.step(state, action)

        # Fresh contexts + fresh episodes, computed for every slot every step.
        num_slots = stepped.done.shape[0]
        keys = jax.vmap(lambda k: jax.random.split(k, 3))(state.info[RNG_KEY])
        sample_key, reset_keys, next_rng = keys[0, 0], keys[:, 1], keys[:, 2]
        new_contexts = self.distribution.sample(parameters, sample_key, num_slots)
        fresh = self._reset_in_contexts(reset_keys, new_contexts)

        done = stepped.done

        def where_done(fresh_leaf, stepped_leaf):
            d = jnp.reshape(done, (num_slots,) + (1,) * (jnp.ndim(stepped_leaf) - 1))
            return jnp.where(d, fresh_leaf, stepped_leaf)

        pipeline_state = jax.tree_util.tree_map(where_done, fresh.pipeline_state, stepped.pipeline_state)
        obs = jax.tree_util.tree_map(where_done, fresh.obs, stepped.obs)

        # Per-episode info the env wrote at reset (its view of the context, e.g.
        # `velocity_threshold`, plus `cost`, `step_count`, ...) must also switch
        # to the fresh episode's values where done. Episode bookkeeping owned by
        # EpisodeWrapper (`steps`, `truncation`, `episode_done`, `episode_metrics`)
        # is deliberately kept from the stepped state: the trainer reads the
        # *finished* episode's metrics there, and `steps` is zeroed on the next
        # step above.
        info = dict(stepped.info)
        for key, fresh_value in fresh.info.items():
            if key in _EPISODE_BOOKKEEPING or key in _WRAPPER_OWNED:
                continue
            if key in info and _same_structure(fresh_value, info[key]):
                info[key] = jax.tree_util.tree_map(where_done, fresh_value, info[key])
        # The transition just taken happened in the context that was running
        # before any reset; the slot's context switches where done.
        info[TRANSITION_CONTEXT_KEY] = stepped.info[CONTEXT_KEY]
        info[CONTEXT_KEY] = where_done(new_contexts, stepped.info[CONTEXT_KEY])
        info[DISTRIBUTION_PARAMETERS_KEY] = parameters
        info[RNG_KEY] = next_rng
        return stepped.replace(pipeline_state=pipeline_state, obs=obs, info=info)


_EPISODE_BOOKKEEPING = frozenset({"steps", "truncation", "episode_done", "episode_metrics"})
_WRAPPER_OWNED = frozenset({CONTEXT_KEY, TRANSITION_CONTEXT_KEY, SLOT_INDEX_KEY, DISTRIBUTION_PARAMETERS_KEY, RNG_KEY})


def _same_structure(a, b) -> bool:
    la, ta = jax.tree_util.tree_flatten(a)
    lb, tb = jax.tree_util.tree_flatten(b)
    return ta == tb and all(jnp.shape(x) == jnp.shape(y) for x, y in zip(la, lb))


def _find_vmapped_env(env: Env) -> Env:
    """The env directly inside the VmapWrapper in a wrapper stack."""
    current = env
    while isinstance(current, Wrapper):
        if isinstance(current, VmapWrapper):
            return current.env
        current = current.env
    raise ValueError("ContextualAutoResetWrapper expects a VmapWrapper somewhere inside its stack")


def _find_episode_wrapper(env: Env) -> EpisodeWrapper:
    """The EpisodeWrapper in a wrapper stack (it owns the step counter and the episode length)."""
    current = env
    while isinstance(current, Wrapper):
        if isinstance(current, EpisodeWrapper):
            return current
        current = current.env
    raise ValueError("ContextualAutoResetWrapper expects an EpisodeWrapper somewhere inside its stack")


def _add_episode_fields(state: State, keys: jax.Array) -> State:
    """EpisodeWrapper.reset's bookkeeping, for states produced by a vmapped
    ``reset_with_context`` that bypassed the wrapper stack."""
    zeros = jnp.zeros(keys.shape[:-1])
    state.info["steps"] = zeros
    state.info["truncation"] = zeros
    state.info["episode_done"] = zeros
    episode_metrics = {"sum_reward": zeros, "length": zeros}
    for name in state.metrics:
        episode_metrics[name] = zeros
    state.info["episode_metrics"] = episode_metrics
    return state


# --------------------------------------------------------------------------- #
# Host-side helpers
# --------------------------------------------------------------------------- #
#
# Inside the wrapper φ has no batch axis. The trainer's state has one leading
# axis over devices (size 1 on a single GPU; ``jax.vmap`` / ``pmap`` of the
# wrapper's reset and step add it to every leaf, φ included), so these helpers
# add and remove that one axis, derived from ``state.done``.


def _device_axes(state: State) -> int:
    return state.done.ndim - 1  # done is [devices..., num_slots] on the host


def attach_parameters(state: State, parameters: DistributionParameters) -> State:
    """Hand the distribution parameters for the coming round to the trainer's state.

    Call between training rounds, after ``distribution.update``. ``parameters`` is
    broadcast over the device axis only, which keeps the pytree *structure and
    shapes* identical to what the compiled program was traced with, so no
    recompile.
    """
    batch_shape = state.done.shape[: _device_axes(state)]
    info = dict(state.info)
    info[DISTRIBUTION_PARAMETERS_KEY] = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, batch_shape + jnp.shape(x)), parameters)
    return state.replace(info=info)


def current_parameters(state: State) -> DistributionParameters:
    """The parameters currently attached to the trainer's state, without the device axis."""
    return jax.tree_util.tree_map(lambda x: x[(0,) * _device_axes(state)], state.info[DISTRIBUTION_PARAMETERS_KEY])


def current_contexts(state: State) -> Contexts:
    """``[..., num_slots, D]`` contexts of the episodes currently running."""
    return state.info[CONTEXT_KEY]


def _wrap_with_contexts(
    env: Env,
    distribution: ContextDistribution,
    episode_length: int,
    action_repeat: int,
    randomization_fn: Optional[Callable],
    spread_initial_phase: bool,
) -> Wrapper:
    if randomization_fn is not None:
        raise NotImplementedError("randomization_fn is not supported with context distributions")
    env = VmapWrapper(env)
    env = EpisodeWrapper(env, episode_length, action_repeat)
    return ContextualAutoResetWrapper(env, distribution, spread_initial_phase=spread_initial_phase)


def wrap_for_context_training(
    env: Env,
    distribution: ContextDistribution,
    episode_length: int,
    action_repeat: int = 1,
    randomization_fn: Optional[Callable] = None,
) -> Wrapper:
    """The **training** stack: per-slot contexts, slots spread over the episode.

    Same as ``crax.envs.training.wrap`` with ``ContextualAutoResetWrapper`` in
    place of ``AutoResetWrapper``::

        ContextualAutoResetWrapper(EpisodeWrapper(VmapWrapper(env)))

    ``randomization_fn`` (Brax domain randomisation of the physics `System`) is
    accepted for signature compatibility with ``wrap`` but not supported
    together with contexts: contexts are the mechanism for varying the task.
    """
    return _wrap_with_contexts(env, distribution, episode_length, action_repeat, randomization_fn, spread_initial_phase=True)


def wrap_for_context_evaluation(
    env: Env,
    distribution: ContextDistribution,
    episode_length: int,
    action_repeat: int = 1,
    randomization_fn: Optional[Callable] = None,
) -> Wrapper:
    """The **evaluation** stack: the training stack without the phase spread, so
    every evaluation episode starts at step 0 and runs its full length. The
    Evaluator sums an episode's reward and cost until its first ``done``; a
    head start would cut that sum short by the head start."""
    return _wrap_with_contexts(env, distribution, episode_length, action_repeat, randomization_fn, spread_initial_phase=False)


def make_wrap_env_fn(distribution: ContextDistribution) -> Callable[..., Wrapper]:
    """A training ``wrap_env_fn`` for ``train(...)`` that installs this distribution.

    Every CRAX trainer accepts ``wrap_env_fn(env, episode_length=, action_repeat=,
    randomization_fn=)`` in place of the default ``crax.envs.training.wrap``. This
    is the only hook needed to switch the auto-reset wrapper::

        train(environment=env, wrap_env_fn=make_wrap_env_fn(distribution), ...)
    """
    return functools.partial(wrap_for_context_training, distribution=distribution)


def make_evaluation_wrap_env_fn(distribution: ContextDistribution) -> Callable[..., Wrapper]:
    """An evaluation ``wrap_env_fn`` for ``train(evaluation_wrap_env_fns={name: ...})``."""
    return functools.partial(wrap_for_context_evaluation, distribution=distribution)
