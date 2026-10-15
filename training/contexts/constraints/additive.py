"""The per-context multiplier as a monotone price of an additive difficulty.

Two assumptions about the factors of ω, stated by Giuseppe on 2026-10-15, and a
third about what a Lagrange multiplier is:

* **each factor is monotone in difficulty**: more hazards of a kind never make
  the task easier; a smaller goal never makes it easier. Which way difficulty
  runs along each coordinate is a fact of the suite, ``Dimension.harder_when``.
* **the factors do not interact**: difficulty is a sum of per-factor terms.
* **λ is the shadow price of the budget**: 0 where the constraint is slack and
  rising where it binds, and it binds *more* the harder the context is.

Together these give a single-index model

    λ(ω) = f( D(ω) ),      D(ω) = Σ_i h_i(ω_i),

with ``h_i`` nondecreasing in factor ``i``'s difficulty direction and ``f ≥ 0``
nondecreasing. The slack region is then a *level set of difficulty* — every
context easier than some threshold has λ = 0 — which a sum of per-factor
multipliers cannot express (it has to start rising at the same value of each
factor whatever the others are; measured 2026-10-15: on a cost linear in the
total hazard count the sum-of-factors form missed the integrator's λ by 0.19
where this form is exact).

**Parameterisation.** ``h_i`` is a step function: a cumulative sum of
nonnegative increments over factor ``i``'s values (every integer for a count,
``CONTINUOUS_BINS`` equal bins for a continuous coordinate); the difficulty
``D`` is the increments passed by ``ω``, as a fraction of all of them, so it is
0 at the easiest corner of Ω and 1 at the hardest. ``f`` is piecewise linear on
``[0, 1]`` over ``PRICE_SEGMENTS`` equal segments with a nonnegative slope on
each, plus a baseline ``b ≥ 0``: any nondecreasing shape, and differentiable in
``D`` so the fit reaches the increments. Every parameter is nonnegative.
``initial_multipliers`` sets ``b`` to the initial λ, every slope to 0 and every
increment to the same value: the constant function, with "difficulty = share of
factor steps passed" as the prior the first rounds reshape.

**Fit.** The integrator names a target per transition (``fitted.py``); the
parameters are fitted to the targets by projected gradient on the squared error
(``SOLVE_ITERATIONS`` steps, warm-started). The problem is bilinear in
``(h, f)`` and so not convex, but it is tiny (tens of parameters), bounded, and
every step keeps the parameters nonnegative, so monotonicity holds at every
round by construction. ``f`` and ``h`` are only identified jointly where
difficulty has been observed; beyond the hardest observed difficulty ``f``
continues with its last slope — an unseen harder context gets at least the
hardest seen λ, which is what monotonicity guarantees.

What the structure buys and what it cannot do is in
``docs/acl/design/contextual_learner.md`` §1. In one line: a per-factor
difficulty learnt at any value of the other factors transfers to all of Ω, so
an unseen context has a multiplier determined by its neighbours, not by a seed;
and if the assumptions are false the fit error against the targets' spread says
so.
"""
from __future__ import annotations

import dataclasses
from typing import Dict, List, Tuple

import flax.struct
import jax
import jax.numpy as jnp
import numpy as np
import optax

from training.agents.ppo_lag.constraint import StateExtras
from training.contexts.constraints.fitted import integrator_targets, multiplier_metrics
from training.contexts.rollout import TRANSITION_CONTEXT_KEY
from training.contexts.space import ContextSpace, Dimension

CONTINUOUS_BINS = 8
PRICE_SEGMENTS = 16
# Projected Adam steps of the fit per round, warm-started from the previous round. Adam rather than
# plain projected gradient because the three parameter groups (baseline, slopes, increments) differ
# in scale by orders of magnitude; one step size for all either stalls the slopes or kills the baseline.
SOLVE_ITERATIONS = 300
SOLVE_LEARNING_RATE = 0.01


@flax.struct.dataclass
class AdditiveMultiplierState:
    """What ``TrainingState.aux_state`` holds. All nonnegative.

    ``difficulty_increments``: ``[M]`` the increments of every factor's ``h_i``, concatenated in
    factor order; ``price_slopes``: ``[PRICE_SEGMENTS]`` the slope of ``f`` on each segment of
    ``[0, 1]``; ``baseline``: ``[]`` the λ of the easiest context, ``f(0)``. ``optimizer_state``
    is the fit's Adam state, carried across rounds like the parameters.
    """

    difficulty_increments: jax.Array
    price_slopes: jax.Array
    baseline: jax.Array
    optimizer_state: optax.OptState

    @property
    def parameters(self) -> "_Parameters":
        return _Parameters(self.difficulty_increments, self.price_slopes, self.baseline)


@flax.struct.dataclass
class _Parameters:
    """The fitted part of the state, as the optimiser sees it."""

    difficulty_increments: jax.Array
    price_slopes: jax.Array
    baseline: jax.Array


@dataclasses.dataclass(frozen=True)
class _Factor:
    """One coordinate's share of the difficulty features: the values at which ``h_i`` may step."""

    dimension: Dimension
    index: int  # column of ω
    thresholds: np.ndarray  # [m] values of ω_i past which an increment counts, in difficulty order
    first_feature: int  # offset of this factor's increments

    @property
    def num_features(self) -> int:
        return int(self.thresholds.shape[0])

    def features(self, values: jax.Array) -> jax.Array:
        """``[..., m]`` 0/1: which thresholds ``ω_i`` has passed in the direction of difficulty."""
        thresholds = jnp.asarray(self.thresholds, jnp.float32)
        if self.dimension.harder_when == "higher":
            return (values[..., None] >= thresholds).astype(jnp.float32)
        return (values[..., None] <= thresholds).astype(jnp.float32)


def _factor_thresholds(dimension: Dimension) -> np.ndarray:
    """The values at which ``h_i`` may step up, ordered from the easiest end towards the
    hardest. A count steps at every value past the easiest one; a continuous coordinate at
    the inner edges of ``CONTINUOUS_BINS`` equal bins. The easiest value passes no threshold
    and contributes 0 difficulty."""
    if dimension.kind == "integer":
        low, high = int(dimension.low), int(dimension.high)
        if dimension.harder_when == "higher":
            return np.arange(low + 1, high + 1, dtype=np.float32)  # passed when ω_i ≥ t
        return np.arange(high - 1, low - 1, -1, dtype=np.float32)  # passed when ω_i ≤ t
    inner_edges = np.linspace(dimension.low, dimension.high, CONTINUOUS_BINS + 1, dtype=np.float32)[1:-1]
    return inner_edges if dimension.harder_when == "higher" else inner_edges[::-1]


@dataclasses.dataclass(frozen=True)
class AdditiveConstraint:
    """``λ(ω) = f(Σ_i h_i(ω_i))`` over ``space``, ``h_i`` and ``f`` nondecreasing
    (a :class:`~training.agents.ppo_lag.constraint.LagrangeConstraint`)."""

    space: ContextSpace

    @property
    def extra_fields(self) -> Tuple[str, ...]:
        return (TRANSITION_CONTEXT_KEY,)

    @property
    def factors(self) -> Tuple[_Factor, ...]:
        factors: List[_Factor] = []
        offset = 0
        for index, dimension in enumerate(self.space.dimensions):
            if dimension.is_degenerate:
                continue
            thresholds = _factor_thresholds(dimension)
            factors.append(_Factor(dimension, index, thresholds, offset))
            offset += thresholds.shape[0]
        return tuple(factors)

    @property
    def num_difficulty_features(self) -> int:
        return sum(factor.num_features for factor in self.factors)

    def initial_multipliers(self, initial_value: float) -> AdditiveMultiplierState:
        """The constant function ``initial_value``: ``b = initial_value``, every slope 0, and
        every difficulty increment equal (``1 / M``), so ``D`` starts as the share of factor steps
        passed. ``initial_value`` may be 0."""
        if initial_value < 0.0:
            raise ValueError(f"a Lagrange multiplier cannot start below 0, got {initial_value}")
        parameters = _Parameters(
            difficulty_increments=jnp.full((self.num_difficulty_features,), 1.0 / self.num_difficulty_features, jnp.float32),
            price_slopes=jnp.zeros((PRICE_SEGMENTS,), jnp.float32),
            baseline=jnp.float32(initial_value),
        )
        return _with_parameters(parameters, _optimizer().init(parameters))

    def per_transition(self, multipliers: AdditiveMultiplierState, state_extras: StateExtras) -> jax.Array:
        return self.multipliers_at(multipliers, state_extras[TRANSITION_CONTEXT_KEY])

    def multipliers_at(self, multipliers: AdditiveMultiplierState, contexts: jax.Array) -> jax.Array:
        """``λ`` at arbitrary contexts ``[..., D]``."""
        return _price(multipliers, self._difficulty_features(contexts))

    def difficulty_at(self, multipliers: AdditiveMultiplierState, contexts: jax.Array) -> jax.Array:
        """``D(ω)`` in ``[0, 1]``: 0 at the easiest corner of Ω, 1 at the hardest."""
        return _difficulty(multipliers, self._difficulty_features(contexts))

    def factor_shares(self, multipliers: AdditiveMultiplierState) -> Dict[str, jax.Array]:
        """Per factor, its share of the total difficulty range: ``Σ increments_i / Σ increments``."""
        total = jnp.maximum(jnp.sum(multipliers.difficulty_increments), 1e-8)
        return {
            factor.dimension.name: jnp.sum(
                multipliers.difficulty_increments[factor.first_feature : factor.first_feature + factor.num_features]
            ) / total
            for factor in self.factors
        }

    def update(
        self, multipliers: AdditiveMultiplierState, costs: jax.Array, state_extras: StateExtras,
        budget_per_step: float, rate: float,
    ) -> Tuple[AdditiveMultiplierState, Dict[str, jax.Array]]:
        """Fit the parameters to the integrator's targets by projected gradient (module
        docstring). Metrics as in :func:`multiplier_metrics`, plus ``lambda_lagr/baseline``
        (``f(0)``, the λ of the easiest context) and ``lambda_lagr/difficulty_share/<factor>``
        (:meth:`factor_shares`)."""
        features = self._difficulty_features(state_extras[TRANSITION_CONTEXT_KEY]).reshape(-1, self.num_difficulty_features)
        targets = integrator_targets(_price(multipliers, features), costs.reshape(-1), budget_per_step, rate)

        def squared_error(parameters: _Parameters) -> jax.Array:
            return jnp.mean(jnp.square(_price(parameters, features) - targets))

        def step(carry: Tuple[_Parameters, optax.OptState], _: None) -> Tuple[Tuple[_Parameters, optax.OptState], None]:
            parameters, optimizer_state = carry
            updates, optimizer_state = _optimizer().update(jax.grad(squared_error)(parameters), optimizer_state, parameters)
            moved = jax.tree_util.tree_map(lambda p, u: jnp.maximum(p + u, 0.0), parameters, updates)
            return (moved, optimizer_state), None

        (parameters, optimizer_state), _ = jax.lax.scan(
            step, (multipliers.parameters, multipliers.optimizer_state), None, length=SOLVE_ITERATIONS
        )
        updated = _with_parameters(parameters, optimizer_state)
        metrics = multiplier_metrics(_price(updated, features), targets)
        metrics["lambda_lagr/baseline"] = updated.baseline
        for name, share in self.factor_shares(updated).items():
            metrics[f"lambda_lagr/difficulty_share/{name}"] = share
        return updated, metrics

    def _difficulty_features(self, contexts: jax.Array) -> jax.Array:
        """``φ(ω)``: ``[..., M]`` 0/1 for contexts ``[..., D]``."""
        return jnp.concatenate([factor.features(contexts[..., factor.index]) for factor in self.factors], axis=-1)


def _optimizer() -> optax.GradientTransformation:
    return optax.adam(SOLVE_LEARNING_RATE)


def _with_parameters(parameters: _Parameters, optimizer_state: optax.OptState) -> AdditiveMultiplierState:
    return AdditiveMultiplierState(
        difficulty_increments=parameters.difficulty_increments, price_slopes=parameters.price_slopes,
        baseline=parameters.baseline, optimizer_state=optimizer_state,
    )


def _difficulty(parameters: _Parameters, features: jax.Array) -> jax.Array:
    """``D = φ · h / Σ h`` in ``[0, 1]``; 0 everywhere while no increment has been learnt."""
    total = jnp.sum(parameters.difficulty_increments)
    return features @ parameters.difficulty_increments / jnp.maximum(total, 1e-8)


def _price(parameters: _Parameters, features: jax.Array) -> jax.Array:
    """``f(D)``: the baseline plus, for each segment ``[k/S, (k+1)/S]`` of ``[0, 1]``, its slope
    times the length of the segment that ``D`` has covered. ``f(0) = b``; nondecreasing because
    every slope is."""
    segment = 1.0 / PRICE_SEGMENTS
    starts = jnp.arange(PRICE_SEGMENTS, dtype=jnp.float32) * segment
    covered = jnp.clip(_difficulty(parameters, features)[..., None] - starts, 0.0, segment)
    return parameters.baseline + covered @ parameters.price_slopes
