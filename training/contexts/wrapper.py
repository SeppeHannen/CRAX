"""Environment wrapper that gives every parallel slot its own context and a fresh
episode on termination.

Replaces the Brax ``AutoResetWrapper`` for context-conditioned training. The
stock wrapper never calls ``reset()`` after the first one: it caches the initial
state per slot and copies it back whenever ``done`` fires, so a slot replays
the *same* layout for the whole run. This wrapper instead, on every step:

1. draws a fresh context for every slot from the distribution parameters that
   were handed in for this round (``state.info["distribution_params"]``);
2. runs the environment's ``reset()`` for every slot with that context;
3. keeps the fresh state where ``done`` is set and the stepped state elsewhere
   (``jnp.where``), and likewise keeps the fresh context only for done slots.

Every slot therefore starts each episode in a newly sampled ω with a newly
sampled layout, and ``state.info["context"]`` always holds the context of the
episode currently running in that slot.

Cost: the reset runs for all slots on every step whether or not any slot is
done — that is how per-slot conditionals work on a GPU (see
``docs/acl/design/per_slot_constraints.md``). How expensive that is depends on
the suite's ``reset()``; it is the first thing to measure with
``--measure_performance``.

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

from training.contexts.distribution import ContextDistribution, Params
from training.contexts.space import Contexts

CONTEXT_KEY = "context"
PARAMS_KEY = "distribution_params"
RNG_KEY = "context_rng"


class ContextualAutoResetWrapper(Wrapper):
    """Per-slot context + reset-on-done, on a *batched* environment.

    Expects the wrapped env to be ``EpisodeWrapper(VmapWrapper(env))``; the
    leading axis of every array is the slot axis.
    """

    def __init__(self, env: Env, distribution: ContextDistribution):
        super().__init__(env)
        self.distribution = distribution
        self.space = distribution.space
        # The single-slot env that VmapWrapper batches (adapters included, so
        # fields like `cost` are still added). We vmap its reset ourselves so a
        # per-slot context can travel alongside the per-slot key.
        self._single_slot_env = _find_vmapped_env(env)

    # ---- reset ------------------------------------------------------------- #

    def reset(self, rng: jax.Array) -> State:
        """Initial reset with contexts drawn from ``distribution.initialise``.

        ``rng`` has shape ``[num_slots, 2]`` (as VmapWrapper expects). The
        round-0 parameters are attached to the state; the trainer replaces them
        between rounds with :func:`attach_parameters`.
        """
        params = self.distribution.initialise(rng[0])
        return self.reset_with_parameters(rng, params)

    def reset_with_parameters(self, rng: jax.Array, params: Params) -> State:
        num_slots = rng.shape[0]
        keys = jax.vmap(lambda k: jax.random.split(k, 3))(rng)  # [N, 3, 2]
        sample_key, reset_keys, next_rng = keys[0, 0], keys[:, 1], keys[:, 2]
        contexts = self.distribution.sample(params, sample_key, num_slots)
        state = self._reset_in_contexts(reset_keys, contexts)
        state.info[CONTEXT_KEY] = contexts
        state.info[PARAMS_KEY] = _broadcast_params(params, num_slots)
        state.info[RNG_KEY] = next_rng
        return state

    def _reset_in_contexts(self, keys: jax.Array, contexts: Contexts) -> State:
        """Batched reset with one context per slot, visible to the env's reset.

        The unwrapped env's ``reset(key)`` cannot take a context argument, so
        we give each slot its context through ``reset_with_context(key, ω)``
        when the suite defines it, and otherwise call ``reset(key)`` and inject
        the context afterwards (correct for suites that only read the context
        in ``step``; the value-typed suites all do). EpisodeWrapper's bookkeeping
        fields are recreated explicitly, exactly as its ``reset`` would.
        """
        state = jax.vmap(_reset_single_slot, in_axes=(None, 0, 0))(self._single_slot_env, keys, contexts)
        state.info[CONTEXT_KEY] = contexts
        return _add_episode_fields(state, keys)

    # ---- step -------------------------------------------------------------- #

    def step(self, state: State, action: jax.Array) -> State:
        # Identical to AutoResetWrapper.step up to the point of choosing the
        # replacement state: reset step counters for slots that were done, clear
        # `done`, step.
        if "steps" in state.info:
            steps = state.info["steps"]
            state.info.update(steps=jnp.where(state.done, jnp.zeros_like(steps), steps))
        state = state.replace(done=jnp.zeros_like(state.done))
        stepped = self.env.step(state, action)

        # Fresh contexts + fresh episodes, computed for every slot every step.
        num_slots = stepped.done.shape[0]
        keys = jax.vmap(lambda k: jax.random.split(k, 3))(state.info[RNG_KEY])
        sample_key, reset_keys, next_rng = keys[0, 0], keys[:, 1], keys[:, 2]
        params = _unbroadcast_params(state.info[PARAMS_KEY])
        new_contexts = self.distribution.sample(params, sample_key, num_slots)
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
            if key in _EPISODE_BOOKKEEPING or key in (PARAMS_KEY, RNG_KEY):
                continue
            if key in info and _same_structure(fresh_value, info[key]):
                info[key] = jax.tree_util.tree_map(where_done, fresh_value, info[key])
        info[CONTEXT_KEY] = where_done(new_contexts, stepped.info[CONTEXT_KEY])
        info[PARAMS_KEY] = state.info[PARAMS_KEY]
        info[RNG_KEY] = next_rng
        return stepped.replace(pipeline_state=pipeline_state, obs=obs, info=info)


_EPISODE_BOOKKEEPING = frozenset({"steps", "truncation", "episode_done", "episode_metrics"})


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


def _reset_single_slot(env: Env, key: jax.Array, context: jax.Array) -> State:
    """Reset one slot with its context.

    Protocol: an env (or adapter) that can use the context at reset time exposes
    ``reset_with_context(key, context)``; adapters forward it inward (see
    ``UnifiedEnvAdapter``). Otherwise the plain ``reset`` is used and the context
    is attached afterwards, which is correct for suites that read it in ``step``.
    """
    reset_with_context = getattr(env, "reset_with_context", None)
    if reset_with_context is not None:
        return reset_with_context(key, context)
    state = env.reset(key)
    state.info[CONTEXT_KEY] = context
    return state


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


def _broadcast_params(params: Params, num_slots: int) -> Params:
    return jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (num_slots,) + jnp.shape(x)), params)


def _unbroadcast_params(params: Params) -> Params:
    return jax.tree_util.tree_map(lambda x: x[0], params)


# --------------------------------------------------------------------------- #
# Host-side helpers
# --------------------------------------------------------------------------- #


def attach_parameters(state: State, params: Params) -> State:
    """Hand the distribution parameters for the coming round to the batched state.

    ``params`` is broadcast over the slot axis so every slot samples from the
    same φ_k. Call between training rounds, after ``distribution.update``.
    Broadcasting keeps the pytree *structure* identical to what the compiled
    program was traced with, so no recompile.
    """
    info = dict(state.info)
    info[PARAMS_KEY] = _broadcast_params(params, state.done.shape[0])
    return state.replace(info=info)


def current_parameters(state: State) -> Params:
    """The (un-batched) parameters currently attached to a batched state."""
    return _unbroadcast_params(state.info[PARAMS_KEY])


def current_contexts(state: State) -> Contexts:
    """``[num_slots, D]`` contexts of the episodes currently running."""
    return state.info[CONTEXT_KEY]


def wrap_for_context_training(
    env: Env,
    distribution: ContextDistribution,
    episode_length: int,
    action_repeat: int = 1,
    randomization_fn: Optional[Callable] = None,
) -> Wrapper:
    """Training wrapper stack with per-slot contexts.

    Same as ``crax.envs.training.wrap`` with ``ContextualAutoResetWrapper`` in
    place of ``AutoResetWrapper``::

        ContextualAutoResetWrapper(EpisodeWrapper(VmapWrapper(env)))

    ``randomization_fn`` (Brax domain randomisation of the physics `System`) is
    accepted for signature compatibility with ``wrap`` but not supported
    together with contexts: contexts are the mechanism for varying the task.
    """
    if randomization_fn is not None:
        raise NotImplementedError("randomization_fn is not supported with context distributions")
    env = VmapWrapper(env)
    env = EpisodeWrapper(env, episode_length, action_repeat)
    env = ContextualAutoResetWrapper(env, distribution)
    return env


def make_wrap_env_fn(distribution: ContextDistribution) -> Callable[..., Wrapper]:
    """A ``wrap_env_fn`` for ``train(...)`` that installs this distribution.

    Every CRAX trainer accepts ``wrap_env_fn(env, episode_length=, action_repeat=,
    randomization_fn=)`` in place of the default ``crax.envs.training.wrap``. This
    is the only hook needed to switch the auto-reset wrapper::

        train(environment=env, wrap_env_fn=make_wrap_env_fn(distribution), ...)
    """
    return functools.partial(wrap_for_context_training, distribution=distribution)
