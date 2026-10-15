"""The two switches that make the learner contextual, from the CLI to ``train(...)``.

A context run's learner is named by how it holds the **constraint** — one λ for
the whole training distribution, or one per context λ(ω) — and by what the
**policy** observes — its sensors only (blind), or its context too (told). The
critics observe the context in every context run. The reasons are
``docs/acl/design/contextual_learner.md``; this module turns the two words into
the trainer arguments that realise them and into the arm label the dashboard
groups runs by.
"""
from __future__ import annotations

import dataclasses
import functools
from typing import Any, Dict, Literal, Tuple, get_args

from training.contexts.constraint import PerContextConstraint
from training.contexts.networks import make_contextual_ppo_networks
from training.contexts.space import ContextSpace

Constraint = Literal["scalar", "per_context"]
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

    def train_kwargs(self, space: ContextSpace) -> Dict[str, Any]:
        """``network_factory`` always (every context run observes ``{sensors, context}``);
        ``constraint`` only for λ(ω), since a scalar λ is every constrained agent's own default
        and only PPO-Lagrange takes a ``constraint``. λ(ω) is a function over ``space``, so it
        needs nothing of Ω beyond its box."""
        kwargs: Dict[str, Any] = {
            "network_factory": functools.partial(
                make_contextual_ppo_networks, policy_observes_context=self.policy_context == "told"
            ),
        }
        if self.constraint == "per_context":
            kwargs["constraint"] = PerContextConstraint(space)
        return kwargs
