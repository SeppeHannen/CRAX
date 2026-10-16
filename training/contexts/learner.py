"""The two switches that make the learner contextual, from the CLI to ``train(...)``.

A context run's learner is named by how it holds the **constraint** — one λ for
the whole training distribution (``scalar``), or a function of the context
λ(ω) in one of two function classes (``network``, ``additive``;
``training/contexts/constraints/``) — and by what the **policy** observes — its
sensors only (blind), or its context too (told). The critics observe the
context in every context run. The reasons are
``docs/acl/design/contextual_learner.md``; this module turns the two words into
the trainer arguments that realise them and into the arm label the dashboard
groups runs by.
"""
from __future__ import annotations

import dataclasses
import functools
from typing import Any, Dict, Literal, Tuple, get_args

from training.agents.ppo_lag.constraint import LagrangeConstraint
from training.contexts.constraints import AdditiveConstraint, NetworkConstraint
from training.contexts.networks import make_contextual_ppo_networks
from training.contexts.space import ContextSpace

Constraint = Literal["scalar", "network", "additive"]
PolicyContext = Literal["blind", "told"]
CONSTRAINTS: Tuple[str, ...] = get_args(Constraint)
POLICY_CONTEXTS: Tuple[str, ...] = get_args(PolicyContext)


@dataclasses.dataclass(frozen=True)
class Learner:
    """How the constraint is held and what the policy observes."""

    constraint: Constraint
    policy_context: PolicyContext

    def __post_init__(self) -> None:
        if self.constraint not in CONSTRAINTS:
            raise ValueError(f"constraint must be one of {CONSTRAINTS}, got {self.constraint!r}")
        if self.policy_context not in POLICY_CONTEXTS:
            raise ValueError(f"policy_context must be one of {POLICY_CONTEXTS}, got {self.policy_context!r}")

    @property
    def label(self) -> str:
        """The two words, as they appear in the arm label and the run name."""
        return f"{self.constraint} {self.policy_context}"

    @property
    def holds_constraint_per_context(self) -> bool:
        return self.constraint != "scalar"

    def per_context_constraint(self, space: ContextSpace) -> LagrangeConstraint:
        """The λ(ω) this learner holds over ``space``; raises for the scalar learner, whose
        constraint is the agent's own default."""
        if self.constraint == "network":
            return NetworkConstraint(space)
        if self.constraint == "additive":
            return AdditiveConstraint(space)
        raise ValueError(f"the {self.constraint!r} learner holds no per-context constraint")

    def train_kwargs(self, space: ContextSpace) -> Dict[str, Any]:
        """``network_factory`` always (every context run observes ``{sensors, context}``);
        ``lagrange_constraint`` only for λ(ω), since a scalar λ is every constrained agent's own
        default and only PPO-Lagrange takes a ``lagrange_constraint``. The key differs from the
        CLI flag ``--constraint`` on purpose: the entry point forwards every CLI flag that shares
        a name with a ``train()`` parameter, and the flag's value is this learner's spec string,
        not the constraint object."""
        kwargs: Dict[str, Any] = {
            "network_factory": functools.partial(
                make_contextual_ppo_networks, policy_observes_context=self.policy_context == "told"
            ),
        }
        if self.holds_constraint_per_context:
            kwargs["lagrange_constraint"] = self.per_context_constraint(space)
        return kwargs
