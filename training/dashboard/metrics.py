"""Which metrics a CRAX run logs to Weights & Biases, and what each one means.

This is the single source of truth behind the dashboard
(``docs/acl/design/dashboard.md`` §7): every scalar and histogram a trainer
reports passes through :func:`select_for_logging` before it reaches W&B, and the
saved view (:mod:`training.dashboard.view`) is built from the same registry.

A key is one of three things, and nothing else:

* **kept** — it matches a :class:`Metric`; the registry says what it measures,
  in which unit, and which review question it serves;
* **dropped** — it matches a :class:`Dropped` entry that says why (a duplicate
  of another key, a meaningless episode sum, ...);
* **unregistered** — :func:`select_for_logging` raises. A new suite or
  algorithm therefore fails loudly at its first log until its metrics are
  registered here, which is where the reviewer's explanation has to be written
  anyway.

Patterns use two placeholders: ``{evaluation}`` stands for the evaluation
prefix (``eval`` for a stock run, ``evaluation/<distribution>`` for a context
run) and ``{dimension}`` for a context dimension of the suite's Ω.
"""
from __future__ import annotations

import dataclasses
import enum
import re
from typing import Any, Dict, Mapping, Optional, Tuple

PLACEHOLDERS = {
    "{evaluation}": r"(eval|evaluation/[^/]+)",
    "{dimension}": r"[^/]+",
}

# Logged next to every per-episode cost so the budget is a line on the same panel.
BUDGET_SUFFIX = "_budget"


class Group(enum.Enum):
    """The review question a metric answers; the order is the order on the page."""

    VERDICT = "Verdict"
    MECHANISM = "Mechanism"
    TRUST = "Trust"
    DETAIL = "Detail"


def _pattern_to_regex(pattern: str) -> "re.Pattern[str]":
    regex = re.escape(pattern)
    for placeholder, expansion in PLACEHOLDERS.items():
        regex = regex.replace(re.escape(placeholder), expansion)
    return re.compile(f"^{regex}$")


@dataclasses.dataclass(frozen=True)
class Metric:
    """A key we keep, with the words a reviewer needs to read its panel.

    Metrics of one group with the same ``panel`` are drawn as series of one
    panel (e.g. the sampled, experienced and intended mean context). By default
    a metric has its own panel, titled by ``title``.
    """

    pattern: str
    title: str
    unit: str
    group: Group
    panel: str = ""
    histogram: bool = False  # logged as wandb.Histogram; W&B renders the sequence over rounds as a heatmap

    @property
    def panel_title(self) -> str:
        return self.panel or self.title

    def matches(self, key: str) -> bool:
        return _pattern_to_regex(self.pattern).match(key) is not None

    def key(self, **fills: str) -> str:
        """The concrete key for one suite/evaluation, e.g. ``key(evaluation="evaluation/deployment")``."""
        key = self.pattern
        for placeholder, value in fills.items():
            key = key.replace("{" + placeholder + "}", value)
        if "{" in key:
            raise ValueError(f"unfilled placeholder in {key!r}")
        return key


@dataclasses.dataclass(frozen=True)
class Dropped:
    """A key the trainer or environment produces that we do not forward, and why."""

    pattern: str
    reason: str

    def matches(self, key: str) -> bool:
        return _pattern_to_regex(self.pattern).match(key) is not None


MEAN_CONTEXT_PANEL = "Mean context: sampled vs experienced (vs intended)"
CONTEXT_SPREAD_PANEL = "Spread of contexts: sampled vs experienced"
FORWARD_REWARD_PANEL = "Training forward reward per episode"

KEPT: Tuple[Metric, ...] = (
    # Verdict: the frozen policy on the evaluation distributions.
    Metric("{evaluation}/episode_reward", "Return per episode", "return (suite's scaled reward)", Group.VERDICT),
    Metric("{evaluation}/episode_cost", "Cost per episode", "violating steps per episode", Group.VERDICT),
    # Mechanism: what the student trained on and how the constraint reacted.
    Metric("training_curriculum/experienced/{dimension}", "Contexts experienced (per transition)", "share of the round's transitions per bin of Ω", Group.MECHANISM, histogram=True),
    Metric("training_curriculum/sampled/{dimension}", "Contexts sampled (per completed episode)", "share of the round's completed episodes per bin of Ω", Group.MECHANISM, histogram=True),
    Metric("training_curriculum/experienced/{dimension}/mean", "Mean context experienced", "context value", Group.MECHANISM, panel=MEAN_CONTEXT_PANEL),
    Metric("training_curriculum/sampled/{dimension}/mean", "Mean context sampled", "context value", Group.MECHANISM, panel=MEAN_CONTEXT_PANEL),
    Metric("training_curriculum/intended/context/{dimension}", "Intended context (staged / fixed distributions)", "context value", Group.MECHANISM, panel=MEAN_CONTEXT_PANEL),
    Metric("training_curriculum/intended/stage", "Stage of the staged curriculum", "stage index", Group.MECHANISM),
    Metric("training/lambda_lagr", "Lagrange multiplier λ", "—", Group.MECHANISM),
    Metric("episodic/cost", "Training cost per episode", "violating steps per episode", Group.MECHANISM),
    Metric("episodic/sum_reward", "Training return per episode", "return (suite's scaled reward)", Group.MECHANISM),
    # Trust: is the run healthy enough to believe the above.
    Metric("episodic/length", "Training episode length", "steps", Group.TRUST),
    Metric("{evaluation}/episode_reward_std", "Return spread over evaluation episodes", "std of return", Group.TRUST),
    Metric("{evaluation}/episode_cost_std", "Cost spread over evaluation episodes", "std of violating steps", Group.TRUST),
    Metric("{evaluation}/avg_episode_length", "Evaluation episode length", "steps", Group.TRUST),
    Metric("training_curriculum/num_completed_episodes", "Completed training episodes per round", "episodes", Group.TRUST),
    Metric("performance/epoch_steps_per_second", "Throughput", "environment steps per second (round wall-clock)", Group.TRUST),
    Metric("performance/epoch_compiles", "Compiles per round", "compiled programs (expect 0 after the first round)", Group.TRUST),
    # Detail: kept for when the above raise a question.
    Metric("training/cost_violation", "Batch cost per transition minus budget per step", "cost per transition (× episode length = per episode)", Group.DETAIL),
    Metric("training/mean_cost", "Batch cost per transition", "cost per transition", Group.DETAIL),
    Metric("training/total_loss", "PPO total loss", "loss", Group.DETAIL),
    Metric("training/policy_loss", "PPO policy loss", "loss", Group.DETAIL),
    Metric("training/v_loss", "Value loss", "loss", Group.DETAIL),
    Metric("training/cost_v_loss", "Cost-value loss", "loss", Group.DETAIL),
    Metric("training/entropy_loss", "Entropy loss", "loss", Group.DETAIL),
    Metric("episodic/forward_reward", "Training forward reward per episode", "sum of forward velocity", Group.DETAIL, panel=FORWARD_REWARD_PANEL),
    Metric("episodic/reward_forward", "Training forward reward per episode (Brax's name; other suites log this one)", "sum of forward velocity", Group.DETAIL, panel=FORWARD_REWARD_PANEL),
    Metric("episodic/reward_ctrl", "Training control penalty per episode", "−Σ ‖action‖²·weight", Group.DETAIL),
    Metric("episodic/velocity_value", "Training constrained speed, summed over the episode", "speed × steps (÷ length = mean speed)", Group.DETAIL),
    Metric("{evaluation}/episode_forward_reward", "Evaluation forward reward per episode", "sum of forward velocity", Group.DETAIL),
    Metric("{evaluation}/episode_velocity_value", "Evaluation constrained speed, summed over the episode", "speed × steps", Group.DETAIL),
    Metric("{evaluation}/episode_reward_ctrl", "Evaluation control penalty per episode", "−Σ ‖action‖²·weight", Group.DETAIL),
    Metric("{evaluation}/std_episode_length", "Evaluation episode length spread", "std of steps", Group.DETAIL),
    Metric("training_curriculum/experienced/{dimension}/std", "Spread of contexts experienced", "context value", Group.DETAIL, panel=CONTEXT_SPREAD_PANEL),
    Metric("training_curriculum/sampled/{dimension}/std", "Spread of contexts sampled", "context value", Group.DETAIL, panel=CONTEXT_SPREAD_PANEL),
    Metric("training_curriculum/episode_length/{dimension}", "Mean training episode length per bin of Ω", "steps (0 = no episode ended in that bin)", Group.DETAIL, histogram=True),
    Metric("training_curriculum/num_transitions", "Transitions per round", "transitions", Group.DETAIL),
    Metric("performance/epoch_wall_seconds", "Wall-clock per round", "seconds", Group.DETAIL),
    Metric("performance/epoch_compile_seconds", "Compile time per round", "seconds", Group.DETAIL),
)

DROPPED: Tuple[Dropped, ...] = (
    Dropped("episodic/velocity_cost", "identical to episodic/cost (cost weight 1)"),
    Dropped("episodic/velocity_violation", "identical to episodic/cost (binary cost)"),
    Dropped("episodic/reward_survive", "identical to episodic/length (1 per step alive)"),
    Dropped("episodic/velocity_magnitude", "identical to episodic/velocity_value"),
    Dropped("episodic/x_velocity", "identical to episodic/forward_reward"),
    Dropped("episodic/reward_unscaled", "episodic/sum_reward ÷ reward_scaler, a constant factor"),
    Dropped("episodic/reward_contact", "contact forces are disabled: always 0"),
    Dropped("episodic/velocity_threshold", "threshold × episode length; the context is reported in training_curriculum"),
    Dropped("episodic/x_position", "a position summed over the episode is meaningless"),
    Dropped("episodic/y_position", "a position summed over the episode is meaningless"),
    Dropped("episodic/distance_from_origin", "a distance summed over the episode is meaningless"),
    Dropped("episodic/y_velocity", "sideways drift summed over the episode; no question needs it"),
    Dropped("{evaluation}/episode_velocity_cost", "identical to episode_cost"),
    Dropped("{evaluation}/episode_velocity_cost_std", "identical to episode_cost_std"),
    Dropped("{evaluation}/episode_velocity_violation", "identical to episode_cost"),
    Dropped("{evaluation}/episode_velocity_violation_std", "identical to episode_cost_std"),
    Dropped("{evaluation}/episode_reward_survive", "identical to avg_episode_length"),
    Dropped("{evaluation}/episode_reward_survive_std", "identical to std_episode_length"),
    Dropped("{evaluation}/episode_velocity_magnitude", "identical to episode_velocity_value"),
    Dropped("{evaluation}/episode_velocity_magnitude_std", "identical to episode_velocity_value_std"),
    Dropped("{evaluation}/episode_reward_forward", "identical to episode_forward_reward"),
    Dropped("{evaluation}/episode_reward_forward_std", "identical to episode_forward_reward_std"),
    Dropped("{evaluation}/episode_x_velocity", "identical to episode_forward_reward"),
    Dropped("{evaluation}/episode_x_velocity_std", "identical to episode_forward_reward_std"),
    Dropped("{evaluation}/episode_reward_unscaled", "episode_reward ÷ reward_scaler"),
    Dropped("{evaluation}/episode_reward_unscaled_std", "episode_reward_std ÷ reward_scaler"),
    Dropped("{evaluation}/episode_reward_contact", "always 0"),
    Dropped("{evaluation}/episode_reward_contact_std", "always 0"),
    Dropped("{evaluation}/episode_velocity_threshold", "threshold × episode length"),
    Dropped("{evaluation}/episode_velocity_threshold_std", "spread of threshold × episode length"),
    Dropped("{evaluation}/episode_x_position", "position summed over the episode"),
    Dropped("{evaluation}/episode_x_position_std", "position summed over the episode"),
    Dropped("{evaluation}/episode_y_position", "position summed over the episode"),
    Dropped("{evaluation}/episode_y_position_std", "position summed over the episode"),
    Dropped("{evaluation}/episode_distance_from_origin", "distance summed over the episode"),
    Dropped("{evaluation}/episode_distance_from_origin_std", "distance summed over the episode"),
    Dropped("{evaluation}/episode_y_velocity", "sideways drift summed over the episode"),
    Dropped("{evaluation}/episode_y_velocity_std", "sideways drift summed over the episode"),
    Dropped("{evaluation}/episode_forward_reward_std", "spread of a Detail metric"),
    Dropped("{evaluation}/episode_velocity_value_std", "spread of a Detail metric"),
    Dropped("{evaluation}/episode_reward_ctrl_std", "spread of a Detail metric"),
    Dropped("{evaluation}/epoch_eval_time", "evaluation wall-clock is not a research quantity"),
    Dropped("{evaluation}/sps", "evaluation throughput is not a research quantity"),
    Dropped("{evaluation}/walltime", "evaluation wall-clock is not a research quantity"),
    Dropped("training/sps", "performance/epoch_steps_per_second, from the performance tracker, is the same number"),
    Dropped("training/walltime", "performance/ has wall-clock per round"),
    Dropped("training/final_step", "only emitted when no evaluation ran; equals num_timesteps"),
)


class UnregisteredMetric(KeyError):
    """A trainer reported a key that is neither kept nor dropped."""

    def __init__(self, key: str):
        super().__init__(
            f"metric {key!r} is not registered: add it to KEPT (with title and unit) or to DROPPED "
            f"(with a reason) in training/dashboard/metrics.py"
        )


def registered(key: str) -> Optional[Metric]:
    """The :class:`Metric` a key is kept under, ``None`` if it is dropped; raises if unregistered."""
    for metric in KEPT:
        if metric.matches(key):
            return metric
    for dropped in DROPPED:
        if dropped.matches(key):
            return None
    raise UnregisteredMetric(key)


# Per-episode costs whose panel carries the budget as a second line.
COSTS_WITH_BUDGET: Tuple[str, ...] = ("episodic/cost", "{evaluation}/episode_cost")


def budget_key(cost_key: str) -> str:
    return cost_key + BUDGET_SUFFIX


def has_budget(key: str) -> bool:
    """Whether this per-episode cost is logged with the budget as a second series."""
    return any(_pattern_to_regex(pattern).match(key) for pattern in COSTS_WITH_BUDGET)


def select_for_logging(metrics: Mapping[str, Any], safety_bound: float) -> Dict[str, Any]:
    """The subset of ``metrics`` that goes to W&B, plus a budget line next to each per-episode cost.

    Raises :class:`UnregisteredMetric` for a key the registry does not know.
    """
    selected: Dict[str, Any] = {}
    for key, value in metrics.items():
        if registered(key) is None:
            continue
        selected[key] = value
        if has_budget(key):
            selected[budget_key(key)] = safety_bound
    return selected
