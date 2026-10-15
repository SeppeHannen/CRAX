"""The per-context forms of PPO-Lagrange's multiplier: λ as a function of the context.

Each module is one function class, fitted each round to the integrator's targets
(``fitted.py``). ``--constraint`` on the CLI names one of them
(``training/contexts/learner.py``):

    network     λ_ψ(ω) = softplus(MLP(ω̃)); free shape, seeded extrapolation
    additive    λ(ω) = b + Σ_i g_i(ω_i), each g_i monotone in difficulty; determined extrapolation
"""
from training.contexts.constraints.additive import AdditiveConstraint, AdditiveMultiplierState
from training.contexts.constraints.network import MultiplierFunctionState, NetworkConstraint

__all__ = ["AdditiveConstraint", "AdditiveMultiplierState", "MultiplierFunctionState", "NetworkConstraint"]
