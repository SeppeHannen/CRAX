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

# Facts of the run a description may also refer to; the view fills them from ``RunFacts``.
FACT_PLACEHOLDERS = ("episode_length", "num_eval_envs", "budget")

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

    ``description`` says where the numbers come from — the population, and
    what was done to it — in one sentence a reviewer can read on the section's
    text panel. It may use the placeholders of :data:`PLACEHOLDERS` and
    :data:`FACT_PLACEHOLDERS`.

    Metrics of one group with the same ``panel`` are drawn as series of one
    panel. By default a metric has its own panel, titled by ``title``.
    """

    pattern: str
    title: str
    unit: str
    group: Group
    description: str
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


MEAN_CONTEXT_PANEL = "{dimension}: mean over the round's training data"
CONTEXT_SPREAD_PANEL = "{dimension}: spread over the round's training data"
FORWARD_VELOCITY_PANEL = "Training forward velocity, summed over the episode"

# Wording shared by several descriptions.
EVALUATION_POPULATION = "{num_eval_envs} episodes of the frozen policy on {evaluation}"
TRAINING_EPISODES = "the training episodes that ended in the round (exploration noise on, contexts as the training distribution drew them)"

KEPT: Tuple[Metric, ...] = (
    # Verdict: the frozen policy on the evaluation distributions.
    Metric("{evaluation}/episode_reward", "Return per episode", "return (the suite's scaled reward)", Group.VERDICT,
           f"the reward (defined above) summed over one episode (≤ {{episode_length}} steps), mean over {EVALUATION_POPULATION}."),
    Metric("{evaluation}/episode_cost", "Cost per episode", "cost per episode", Group.VERDICT,
           f"the cost (defined above) summed over one episode, mean over {EVALUATION_POPULATION}."),
    # Mechanism: what the student trained on and how the constraint reacted.
    Metric("training_curriculum/experienced/{dimension}", "Contexts experienced (per transition)", "share of the round's transitions per bin", Group.MECHANISM,
           "heatmap over rounds: the share of the round's *transitions* whose episode ran at each value of {dimension} (q̂, the realised curriculum).", histogram=True),
    Metric("training_curriculum/sampled/{dimension}", "Contexts sampled (per completed episode)", "share of the round's completed episodes per bin", Group.MECHANISM,
           "heatmap over rounds: the share of the round's *completed episodes* that ran at each value of {dimension} (the empirical q).", histogram=True),
    Metric("training_curriculum/experienced/{dimension}/mean", "experienced", "{dimension}", Group.MECHANISM,
           "**experienced** = mean {dimension} over the round's transitions: every step counts once, so contexts with long episodes weigh more — the data the gradient came from.", panel=MEAN_CONTEXT_PANEL),
    Metric("training_curriculum/sampled/{dimension}/mean", "sampled", "{dimension}", Group.MECHANISM,
           "**sampled** = mean {dimension} over the round's completed episodes: every episode counts once — what the training distribution chose. Sampled and experienced are the same contexts counted per episode vs per step; they differ when episode length depends on the context.", panel=MEAN_CONTEXT_PANEL),
    Metric("training_curriculum/intended/context/{dimension}", "intended", "{dimension}", Group.MECHANISM,
           "**intended** = the {dimension} a fixed or staged distribution states for the round (absent for uniform).", panel=MEAN_CONTEXT_PANEL),
    Metric("training_curriculum/intended/stage", "Stage of the staged curriculum", "stage index", Group.MECHANISM,
           "which stage of a staged curriculum the round sampled from (absent for other distributions)."),
    Metric("training/lambda_lagr", "Lagrange multiplier λ", "λ", Group.MECHANISM,
           "PPO-Lagrange's multiplier after the round's update; it grows while the batch's cost per step exceeds the budget per step and shrinks otherwise."),
    Metric("episodic/cost", "Training cost per episode", "cost per episode", Group.MECHANISM,
           f"the cost (defined in the Verdict section) summed over one episode, mean over {TRAINING_EPISODES}. Not "
           f"comparable to Verdict's cost."),
    Metric("episodic/sum_reward", "Training return per episode", "return (the suite's scaled reward)", Group.MECHANISM,
           f"reward summed over one episode, mean over {TRAINING_EPISODES}."),
    # Trust: is the run healthy enough to believe the above.
    Metric("episodic/length", "Training episode length", "steps", Group.TRUST,
           f"steps per episode, mean over {TRAINING_EPISODES}. An episode lasts {{episode_length}} steps unless the "
           f"agent falls, so this is a survival measure: {{episode_length}} = stayed upright throughout; a drop to a "
           f"few steps = fell almost immediately."),
    Metric("{evaluation}/episode_reward_std", "Return spread over evaluation episodes", "std of return", Group.TRUST,
           f"standard deviation of the return over the {EVALUATION_POPULATION} — how noisy the Verdict return is."),
    Metric("{evaluation}/episode_cost_std", "Cost spread over evaluation episodes", "std of cost per episode", Group.TRUST,
           f"standard deviation of the cost over the {EVALUATION_POPULATION} — how noisy the Verdict cost is."),
    Metric("{evaluation}/avg_episode_length", "Evaluation episode length", "steps", Group.TRUST,
           f"steps per episode, mean over {EVALUATION_POPULATION}; below {{episode_length}} means the agent fell before "
           f"the step limit."),
    Metric("training_curriculum/num_completed_episodes", "Completed training episodes per round", "episodes", Group.TRUST,
           "how many training episodes ended in the round — the sample size behind *sampled*."),
    Metric("performance/epoch_steps_per_second", "Throughput", "environment steps per second", Group.TRUST,
           "environment steps of the round divided by its wall-clock, from the performance tracker."),
    Metric("performance/epoch_compiles", "Compiles per round", "compiled programs", Group.TRUST,
           "number of XLA compilations that finished during the round. JAX compiles the training step into one GPU "
           "program before round 0 (reported in the run summary, not here) and reuses it; a value changing between "
           "rounds (a context, λ) never recompiles, only a change of array shape or structure does. The expected "
           "reading is 0 everywhere; any non-zero value is a recompile costing tens of seconds and pointing at a "
           "shape change."),
    # Detail: kept for when the above raise a question.
    Metric("training/cost_violation", "Batch cost per transition minus budget per step", "cost per transition", Group.DETAIL,
           "mean cost per *transition* in the round's PPO batch minus the budget per step ({budget} / {episode_length}); this is the signal that moves λ. × {episode_length} gives per-episode units."),
    Metric("training/mean_cost", "Batch cost per transition", "cost per transition", Group.DETAIL,
           "mean cost per transition in the round's PPO batch."),
    Metric("training/total_loss", "PPO total loss", "loss", Group.DETAIL, "sum of the losses below, mean over the round's minibatches."),
    Metric("training/policy_loss", "PPO policy loss", "loss", Group.DETAIL, "clipped surrogate on the Lagrangian advantage, mean over the round's minibatches."),
    Metric("training/v_loss", "Value loss", "loss", Group.DETAIL,
           "reward value function regression loss. Tiny in absolute terms because it is computed on rewards scaled by "
           "`reward_scaling` on top of the suite's own reward scaling (per-step values of order 1e-3); judge it relative "
           "to its own start, not against the cost-value loss."),
    Metric("training/cost_v_loss", "Cost-value loss", "loss", Group.DETAIL,
           "cost value function regression loss; cost is unscaled (0 or 1 per step), hence the much larger scale."),
    Metric("training/entropy_loss", "Entropy loss", "loss", Group.DETAIL,
           "−entropy_cost × the policy's action entropy, mean over the round's minibatches: more negative = the policy "
           "is still exploring. It rises as the policy commits to a gait; a fast rise early on means exploration "
           "stopped before anything was learned."),
    Metric("episodic/forward_reward", "forward_reward", "velocity × steps", Group.DETAIL,
           f"the agent's velocity along the x-axis (forward), summed over the episode — the main term of the **reward**, "
           f"so higher is better; negative = walked backwards. Mean over {TRAINING_EPISODES}.", panel=FORWARD_VELOCITY_PANEL),
    Metric("episodic/reward_forward", "reward_forward", "velocity × steps", Group.DETAIL,
           "the same number under Brax's key name (suites other than the velocity ones log only this one).", panel=FORWARD_VELOCITY_PANEL),
    Metric("episodic/reward_ctrl", "Training control penalty, summed over the episode", "−Σ ‖action‖² × weight", Group.DETAIL,
           f"the reward's penalty for large actions (−weight × ‖action‖² per step), summed over the episode, mean over {TRAINING_EPISODES}."),
    Metric("episodic/velocity_value", "Training speed in any direction, summed over the episode", "speed × steps", Group.DETAIL,
           f"the agent's speed regardless of direction (for Ant: √(vₓ²+vᵧ²) of the torso), summed over the episode — the "
           f"quantity the **cost** compares against the threshold: a step is a violation when this exceeds the episode's "
           f"velocity_threshold. Mean over {TRAINING_EPISODES}; ÷ the training episode length = mean speed, to compare "
           f"with the thresholds in Ω."),
    Metric("{evaluation}/episode_forward_reward", "Evaluation forward velocity, summed over the episode", "velocity × steps", Group.DETAIL,
           f"the agent's velocity along the x-axis (forward), summed over the episode — the main term of the reward. "
           f"Mean over {EVALUATION_POPULATION}."),
    Metric("{evaluation}/episode_velocity_value", "Evaluation speed in any direction, summed over the episode", "speed × steps", Group.DETAIL,
           f"the agent's speed regardless of direction, summed over the episode — the quantity the cost compares against "
           f"the threshold. Mean over {EVALUATION_POPULATION}; ÷ the evaluation episode length = mean speed."),
    Metric("{evaluation}/episode_reward_ctrl", "Evaluation control penalty, summed over the episode", "−Σ ‖action‖² × weight", Group.DETAIL,
           f"the reward's penalty for large actions, summed over the episode, mean over {EVALUATION_POPULATION}."),
    Metric("{evaluation}/std_episode_length", "Evaluation episode length spread", "std of steps", Group.DETAIL,
           f"standard deviation of the episode length over the {EVALUATION_POPULATION}."),
    Metric("training_curriculum/experienced/{dimension}/std", "experienced", "{dimension}", Group.DETAIL,
           "standard deviation of {dimension} over the round's transitions.", panel=CONTEXT_SPREAD_PANEL),
    Metric("training_curriculum/sampled/{dimension}/std", "sampled", "{dimension}", Group.DETAIL,
           "standard deviation of {dimension} over the round's completed episodes.", panel=CONTEXT_SPREAD_PANEL),
    Metric("training_curriculum/episode_length/{dimension}", "Mean training episode length per value of {dimension}", "steps", Group.DETAIL,
           "heatmap over rounds: mean length of the round's completed episodes per value of {dimension} (0 = none ended there) — why sampled ≠ experienced.", histogram=True),
    Metric("training_curriculum/num_transitions", "Transitions per round", "transitions", Group.DETAIL,
           "environment steps in the round — the sample size behind *experienced*."),
    Metric("performance/epoch_wall_seconds", "Wall-clock per round", "seconds", Group.DETAIL, "wall-clock of the round."),
    Metric("performance/epoch_compile_seconds", "Compile time per round", "seconds", Group.DETAIL, "seconds spent compiling during the round."),
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
