"""What the two per-context multiplier forms share: the integrator's targets and
the metrics that say how well a function followed them.

A per-context multiplier is a function λ(ω) moved by PPO-Lagrange's integrator,
projected onto a function class: after a round every transition names where its
context's multiplier should go,

    λ*_t = relu( λ(ω_t) + rate · (c_t − d) ),

and the function is fitted to those targets. ``network.py`` fits a small MLP by
gradient steps; ``additive.py`` fits a monotone additive model by a convex
solve. Both report the same metrics so that the two forms compare on one panel
(``docs/acl/design/contextual_learner.md`` §1).
"""
from __future__ import annotations

from typing import Dict

import jax
import jax.numpy as jnp


def integrator_targets(current: jax.Array, costs: jax.Array, budget_per_step: float, rate: float) -> jax.Array:
    """Where the integrator asks each transition's multiplier to move: ``relu(λ + rate·(c − d))``."""
    return jax.nn.relu(current + rate * (costs - budget_per_step))


def multiplier_metrics(per_transition: jax.Array, targets: jax.Array) -> Dict[str, jax.Array]:
    """The ``training/*`` metrics of a fitted multiplier, over the round's transitions under the
    *updated* function (so ``lambda_lagr`` is the λ the batch was weighted by on average,
    comparable to a scalar arm's).

    ``lambda_lagr`` mean and ``lambda_lagr/{std,min,max}``, the spread across contexts;
    ``lambda_lagr/fit_error`` the root mean squared distance from the targets after the fit,
    and ``lambda_lagr/target_spread`` the targets' standard deviation over the batch. The
    targets vary for two reasons: between contexts, because their multipliers differ, and
    within a context, because a target is one transition's cost and costs scatter by
    ``rate · std(c)`` around their mean. The fit resolves the first and lands on the mean of
    the second, so ``fit_error`` well below ``target_spread`` means the function holds the
    contexts' multipliers apart; ``fit_error`` near ``target_spread`` means it does not (one
    context, or a function class that cannot separate the contexts mixed).
    """
    return {
        "lambda_lagr": jnp.mean(per_transition),
        "lambda_lagr/std": jnp.std(per_transition),
        "lambda_lagr/min": jnp.min(per_transition),
        "lambda_lagr/max": jnp.max(per_transition),
        "lambda_lagr/fit_error": jnp.sqrt(jnp.mean(jnp.square(per_transition - targets))),
        "lambda_lagr/target_spread": jnp.std(targets),
    }
