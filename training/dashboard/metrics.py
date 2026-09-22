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

Nothing here knows which agent a run uses. The words that depend on the agent
(what it is, whether it can fall, how its speed is measured) are the suite's
``TaskDescription`` (``training/contexts/registry.py``), which the view quotes
once at the top; descriptions here refer to "the reward", "the cost" and "the
Environment definition" instead of restating them.

Patterns use three placeholders: ``{evaluation}`` stands for the evaluation
prefix (``eval`` for a stock run, ``evaluation/<distribution>`` for a context
run), ``{dimension}`` for a context dimension of the suite's Ω, and
``{reward_term}`` for one of the names Brax's agents give the terms of their own
reward.
"""
from __future__ import annotations

import dataclasses
import enum
import re
from typing import Any, Dict, Mapping, Optional, Tuple

# The names Brax's MuJoCo agents give the terms of their reward (each agent its own subset):
# the forward term, the per-step upright bonus, the control penalty, the contact penalty.
# Named here so the registry can drop them all under one reason without naming agents.
# `reward_forward` is not among them: results/common.py reads it for suites outside
# velocity, so it stays a kept key below.
REWARD_TERMS = ("run", "fwd", "linvel", "survive", "healthy", "alive", "ctrl", "quadctrl", "contact", "dist", "bonus")

PLACEHOLDERS = {
    "{evaluation}": r"(eval|evaluation/[^/]+)",
    "{dimension}": r"[^/]+",
    "{reward_term}": "(" + "|".join(REWARD_TERMS) + ")",
}

# Facts of the run a description may also refer to; the view fills them from ``RunFacts``.
# ``episode_length_reading`` is the sentence that says how to read an episode length for
# this suite (whether the environment can end an episode early).
FACT_PLACEHOLDERS = ("episode_length", "num_eval_envs", "budget", "episode_length_reading")

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
FORWARD_REWARD_PANEL = "Training forward reward, summed over the episode"
EVALUATION_FORWARD_REWARD_PANEL = "Evaluation forward reward, summed over the episode"
COST_PARTS_PANEL = "Training cost per episode, by part"
EVALUATION_COST_PARTS_PANEL = "Evaluation cost per episode, by part"

# Wording shared by several descriptions.
EVALUATION_POPULATION = "{num_eval_envs} episodes of the frozen policy on {evaluation}"
TRAINING_EPISODES = "the training episodes that ended in the round (exploration noise on, contexts as the training distribution drew them)"
EPISODE_LENGTH_READING = "{episode_length_reading}"
FORWARD_REWARD = (
    "the forward term of the **reward** (its weight × the forward velocity vₓ), summed over the episode — higher is "
    "better, negative = moved backwards. The number the CRAX paper's figures report (`results/common.py`), under the "
    "key they read"
)

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
           f"steps per episode, mean over {TRAINING_EPISODES}. {EPISODE_LENGTH_READING}"),
    Metric("{evaluation}/episode_reward_std", "Return spread over evaluation episodes", "std of return", Group.TRUST,
           f"standard deviation of the return over the {EVALUATION_POPULATION} — how noisy the Verdict return is."),
    Metric("{evaluation}/episode_cost_std", "Cost spread over evaluation episodes", "std of cost per episode", Group.TRUST,
           f"standard deviation of the cost over the {EVALUATION_POPULATION} — how noisy the Verdict cost is."),
    Metric("{evaluation}/avg_episode_length", "Evaluation episode length", "steps", Group.TRUST,
           f"steps per episode, mean over {EVALUATION_POPULATION}. {EPISODE_LENGTH_READING}"),
    Metric("training_curriculum/num_completed_episodes", "Completed training episodes per round", "episodes", Group.TRUST,
           "how many training episodes ended in the round — the sample size behind *sampled*."),
    Metric("performance/epoch_steps_per_second", "Throughput", "environment steps per second", Group.TRUST,
           "environment steps of the round divided by its wall-clock, from the performance tracker."),
    Metric("performance/epoch_compiles", "Substantial compiles per round", "compiled programs ≥ 1 s", Group.TRUST,
           "number of XLA programs taking at least one second to compile that finished during the round. JAX compiles "
           "the training step into one GPU program before round 0 (reported in the run summary, not here) and reuses "
           "it; a value changing between rounds (a context, λ) never recompiles, only a change of array shape or "
           "structure does. The few-millisecond helper programs JAX builds for host-side glue every round are not "
           "counted. The expected reading is 0 everywhere; a non-zero value is a rebuilt program pointing at a shape "
           "change."),
    Metric("performance/epoch_compile_seconds", "Compile time per round", "seconds", Group.TRUST,
           "seconds spent compiling anything during the round, helpers included; ~0.1 s per round is the normal cost of "
           "the host-side glue. Seconds, not tenths, means a program was rebuilt."),
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
    Metric("episodic/forward_reward", "forward_reward", "weight × velocity × steps", Group.DETAIL,
           f"**forward_reward** = {FORWARD_REWARD}. Mean over {TRAINING_EPISODES}.", panel=FORWARD_REWARD_PANEL),
    Metric("episodic/reward_forward", "reward_forward", "weight × velocity × steps", Group.DETAIL,
           "**reward_forward** = the same term under Brax's other name; suites whose agent logs only this one "
           "(`results/common.py` reads it for them) show this series instead.", panel=FORWARD_REWARD_PANEL),
    Metric("episodic/velocity_value", "Training speed as the cost measures it, summed over the episode", "speed × steps", Group.DETAIL,
           f"the agent's speed as the **cost** defines it, summed over the episode — a step is a violation when this exceeds "
           f"the episode's velocity_threshold. Mean over {TRAINING_EPISODES}; ÷ the training episode length = mean speed, "
           f"to compare with the thresholds in Ω."),
    Metric("episodic/head_height", "Training head height, summed over the episode", "metres × steps", Group.DETAIL,
           f"the height of the top of the head as the **cost** defines it, summed over the episode. Mean over "
           f"{TRAINING_EPISODES}; ÷ the training episode length = mean head height, to compare with the ceilings in Ω."),
    Metric("episodic/hazard_cost", "hazard_cost", "cost per episode", Group.DETAIL,
           f"**hazard_cost** = the part of the **cost** incurred by feet inside hazards, summed over the episode; mean over "
           f"{TRAINING_EPISODES}. Suites whose cost has several parts log each part; they sum to the cost.", panel=COST_PARTS_PANEL),
    Metric("episodic/terminal_cost", "terminal_cost", "cost per episode", Group.DETAIL,
           "**terminal_cost** = the part of the cost charged on the step the episode ends early (a fall); 0 for episodes "
           "that reached the step limit.", panel=COST_PARTS_PANEL),
    Metric("episodic/boundary_cost", "boundary_cost", "cost per episode", Group.DETAIL,
           f"**boundary_cost** = the part of the **cost** charged for being outside the episode's boundary rectangle, summed "
           f"over the episode (the rest is hazard proximity); mean over {TRAINING_EPISODES}.", panel=COST_PARTS_PANEL),
    Metric("episodic/radial_error", "Training distance from the ring, summed over the episode", "metres × steps", Group.DETAIL,
           f"|the robot's distance from the circle's centre − the ring's radius|, summed over the episode. Mean over "
           f"{TRAINING_EPISODES}; ÷ the training episode length = how far off the ring it drove on average."),
    Metric("episodic/tangent_velocity", "Training speed around the ring, summed over the episode", "m/s × steps", Group.DETAIL,
           f"the robot's velocity component tangential to the ring (counter-clockwise positive), summed over the episode — "
           f"the **reward** is this ÷ (1 + radial_error) × 0.1 per step. Mean over {TRAINING_EPISODES}; ÷ the training "
           f"episode length = mean orbital speed."),
    Metric("{evaluation}/episode_forward_reward", "forward_reward", "weight × velocity × steps", Group.DETAIL,
           f"**forward_reward** = {FORWARD_REWARD}. Mean over {EVALUATION_POPULATION}.", panel=EVALUATION_FORWARD_REWARD_PANEL),
    Metric("{evaluation}/episode_reward_forward", "reward_forward", "weight × velocity × steps", Group.DETAIL,
           "**reward_forward** = the same term under Brax's other name; suites whose agent logs only this one show this "
           "series instead.", panel=EVALUATION_FORWARD_REWARD_PANEL),
    Metric("{evaluation}/episode_velocity_value", "Evaluation speed as the cost measures it, summed over the episode", "speed × steps", Group.DETAIL,
           f"the agent's speed as the cost defines it, summed over the episode. Mean over {EVALUATION_POPULATION}; "
           f"÷ the evaluation episode length = mean speed."),
    Metric("{evaluation}/episode_head_height", "Evaluation head height, summed over the episode", "metres × steps", Group.DETAIL,
           f"the height of the top of the head as the cost defines it, summed over the episode. Mean over "
           f"{EVALUATION_POPULATION}; ÷ the evaluation episode length = mean head height."),
    Metric("{evaluation}/episode_hazard_cost", "hazard_cost", "cost per episode", Group.DETAIL,
           f"**hazard_cost** = the part of the cost incurred by feet inside hazards, summed over the episode; mean over "
           f"{EVALUATION_POPULATION}.", panel=EVALUATION_COST_PARTS_PANEL),
    Metric("{evaluation}/episode_terminal_cost", "terminal_cost", "cost per episode", Group.DETAIL,
           "**terminal_cost** = the part of the cost charged on the step the episode ends early (a fall).", panel=EVALUATION_COST_PARTS_PANEL),
    Metric("{evaluation}/episode_boundary_cost", "boundary_cost", "cost per episode", Group.DETAIL,
           f"**boundary_cost** = the part of the cost charged for being outside the episode's boundary rectangle, summed "
           f"over the episode; mean over {EVALUATION_POPULATION}.", panel=EVALUATION_COST_PARTS_PANEL),
    Metric("{evaluation}/episode_radial_error", "Evaluation distance from the ring, summed over the episode", "metres × steps", Group.DETAIL,
           f"|distance from the circle's centre − the ring's radius|, summed over the episode. Mean over "
           f"{EVALUATION_POPULATION}; ÷ the evaluation episode length = mean distance off the ring."),
    Metric("{evaluation}/episode_tangent_velocity", "Evaluation speed around the ring, summed over the episode", "m/s × steps", Group.DETAIL,
           f"the robot's tangential velocity around the ring (counter-clockwise positive), summed over the episode. Mean over "
           f"{EVALUATION_POPULATION}; ÷ the evaluation episode length = mean orbital speed."),
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
)

# Why the agents' own reward terms are not forwarded: they decompose the reward the way each
# Brax agent names its parts; the reward itself is sum_reward, its forward term (the one the
# paper reports) is forward_reward, and the upright bonus is a constant per step alive, i.e.
# the episode length again.
AGENTS_OWN_REWARD_TERMS = (
    "a term of the agent's reward under the agent's own name; the reward is sum_reward, its forward term forward_reward, "
    "and an upright bonus is a constant × the episode length"
)

DROPPED: Tuple[Dropped, ...] = (
    Dropped("episodic/velocity_cost", "identical to episodic/cost (cost weight 1)"),
    Dropped("episodic/velocity_violation", "identical to episodic/cost (binary cost)"),
    Dropped("episodic/reward_{reward_term}", AGENTS_OWN_REWARD_TERMS),
    Dropped("episodic/reward_unscaled", "episodic/sum_reward ÷ the suite's reward scale, a constant factor"),
    Dropped("episodic/velocity_magnitude", "identical to episodic/velocity_value"),
    Dropped("episodic/height_cost", "identical to episodic/cost"),
    Dropped("episodic/height_violation", "episodic/cost × a constant (the hinge width ÷ the cost weight)"),
    Dropped("episodic/reward", "identical to episodic/sum_reward (the point-robot suites log the per-step reward under this name)"),
    Dropped("episodic/feet_on_ground", "episodic/cost ÷ the cost scale (1 by default): restricted feet on the ground per step, summed"),
    Dropped("episodic/ctrl_cost", "‖action‖² × a weight, summed; part of neither the reward nor the cost in the point-robot suites"),
    Dropped("episodic/distance_to_goal", "a distance summed over the episode is meaningless"),
    Dropped("episodic/dist", "a distance summed over the episode is meaningless"),
    Dropped("episodic/last_dist_goal", "the previous step's distance_to_goal, summed"),
    Dropped("episodic/radius", "the distance from the circle's centre, summed; episodic/radial_error is |this − the ring's radius|"),
    Dropped("episodic/out_of_boundary", "steps outside the boundary rectangle: episodic/boundary_cost ÷ the boundary cost weight (0.1)"),
    Dropped("episodic/x_velocity", "the forward velocity summed over the episode; forward_reward is this × the reward's weight"),
    Dropped("episodic/velocity_threshold", "threshold × episode length; the context is reported in training_curriculum"),
    Dropped("episodic/x_position", "a position summed over the episode is meaningless"),
    Dropped("episodic/y_position", "a position summed over the episode is meaningless"),
    Dropped("episodic/distance_from_origin", "a distance summed over the episode is meaningless"),
    Dropped("episodic/y_velocity", "sideways drift summed over the episode; no question needs it"),
    Dropped("{evaluation}/episode_velocity_cost", "identical to episode_cost"),
    Dropped("{evaluation}/episode_velocity_cost_std", "identical to episode_cost_std"),
    Dropped("{evaluation}/episode_velocity_violation", "identical to episode_cost"),
    Dropped("{evaluation}/episode_velocity_violation_std", "identical to episode_cost_std"),
    Dropped("{evaluation}/episode_reward_{reward_term}", AGENTS_OWN_REWARD_TERMS),
    Dropped("{evaluation}/episode_reward_{reward_term}_std", "spread of a dropped metric"),
    Dropped("{evaluation}/episode_reward_forward_std", "spread of a Detail metric"),
    Dropped("{evaluation}/episode_reward_unscaled", "episode_reward ÷ the suite's reward scale"),
    Dropped("{evaluation}/episode_reward_unscaled_std", "episode_reward_std ÷ the suite's reward scale"),
    Dropped("{evaluation}/episode_velocity_magnitude", "identical to episode_velocity_value"),
    Dropped("{evaluation}/episode_velocity_magnitude_std", "identical to episode_velocity_value_std"),
    Dropped("{evaluation}/episode_height_cost", "identical to episode_cost"),
    Dropped("{evaluation}/episode_height_cost_std", "identical to episode_cost_std"),
    Dropped("{evaluation}/episode_height_violation", "episode_cost × a constant"),
    Dropped("{evaluation}/episode_height_violation_std", "episode_cost_std × a constant"),
    Dropped("{evaluation}/episode_head_height_std", "spread of a Detail metric"),
    Dropped("{evaluation}/episode_hazard_cost_std", "spread of a Detail metric"),
    Dropped("{evaluation}/episode_terminal_cost_std", "spread of a Detail metric"),
    Dropped("{evaluation}/episode_feet_on_ground", "episode_cost ÷ the cost scale"),
    Dropped("{evaluation}/episode_feet_on_ground_std", "episode_cost_std ÷ the cost scale"),
    Dropped("{evaluation}/episode_ctrl_cost", "‖action‖² × a weight, summed; part of neither the reward nor the cost"),
    Dropped("{evaluation}/episode_ctrl_cost_std", "spread of a dropped metric"),
    Dropped("{evaluation}/episode_dist", "distance summed over the episode"),
    Dropped("{evaluation}/episode_dist_std", "distance summed over the episode"),
    Dropped("{evaluation}/episode_distance_to_goal", "distance summed over the episode"),
    Dropped("{evaluation}/episode_distance_to_goal_std", "distance summed over the episode"),
    Dropped("{evaluation}/episode_last_dist_goal", "the previous step's distance_to_goal, summed"),
    Dropped("{evaluation}/episode_last_dist_goal_std", "the previous step's distance_to_goal, summed"),
    Dropped("{evaluation}/episode_radius", "distance from the circle's centre, summed; episode_radial_error is |this − the ring's radius|"),
    Dropped("{evaluation}/episode_radius_std", "spread of a dropped metric"),
    Dropped("{evaluation}/episode_out_of_boundary", "episode_boundary_cost ÷ the boundary cost weight (0.1)"),
    Dropped("{evaluation}/episode_out_of_boundary_std", "spread of a dropped metric"),
    Dropped("{evaluation}/episode_boundary_cost_std", "spread of a Detail metric"),
    Dropped("{evaluation}/episode_radial_error_std", "spread of a Detail metric"),
    Dropped("{evaluation}/episode_tangent_velocity_std", "spread of a Detail metric"),
    Dropped("{evaluation}/episode_x_velocity", "the forward velocity summed over the episode; episode_forward_reward is this × the reward's weight"),
    Dropped("{evaluation}/episode_x_velocity_std", "spread of a dropped metric"),
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
