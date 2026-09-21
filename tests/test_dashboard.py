"""The dashboard registry and view (docs/acl/design/dashboard.md §7).

* every key a PPO-Lagrange velocity run produces is registered — kept with a
  title and unit, or dropped with a reason — and the funnel raises on a key
  that is neither;
* per-episode costs are logged with their budget so the panel has the line;
* the saved view has the four review sections in order, each opening with a
  text panel that states population, unit and cadence, and no smoothing.

Run with: JAX_PLATFORMS=cpu pytest tests/test_dashboard.py -v
"""
from __future__ import annotations

import dataclasses
from typing import Dict, List

import pytest
import wandb

from training import dashboard
from training.dashboard.metrics import Group, has_budget
from training.dashboard.view import plots_for

# Every key of the two 500 M runs of 2026-09-21 that the current code still
# produces (the old bin_NN / completed_episodes / round keys are gone).
VELOCITY_ANT_KEYS: List[str] = [
    *[f"episodic/{name}" for name in (
        "cost", "distance_from_origin", "forward_reward", "length", "reward_contact", "reward_ctrl", "reward_forward",
        "reward_survive", "reward_unscaled", "sum_reward", "velocity_cost", "velocity_magnitude", "velocity_threshold",
        "velocity_value", "velocity_violation", "x_position", "x_velocity", "y_position", "y_velocity",
    )],
    *[
        f"evaluation/{distribution}/{name}{suffix}"
        for distribution in ("deployment", "uniform")
        for name in (
            "episode_cost", "episode_distance_from_origin", "episode_forward_reward", "episode_reward",
            "episode_reward_contact", "episode_reward_ctrl", "episode_reward_forward", "episode_reward_survive",
            "episode_reward_unscaled", "episode_velocity_cost", "episode_velocity_magnitude", "episode_velocity_threshold",
            "episode_velocity_value", "episode_velocity_violation", "episode_x_position", "episode_x_velocity",
            "episode_y_position", "episode_y_velocity",
        )
        for suffix in ("", "_std")
    ],
    *[f"evaluation/{d}/{name}" for d in ("deployment", "uniform") for name in ("avg_episode_length", "std_episode_length", "epoch_eval_time", "sps", "walltime")],
    *[f"training/{name}" for name in ("cost_v_loss", "cost_violation", "entropy_loss", "lambda_lagr", "mean_cost", "policy_loss", "sps", "total_loss", "v_loss", "walltime")],
    "training_curriculum/experienced/velocity_threshold",
    "training_curriculum/experienced/velocity_threshold/mean",
    "training_curriculum/experienced/velocity_threshold/std",
    "training_curriculum/sampled/velocity_threshold",
    "training_curriculum/sampled/velocity_threshold/mean",
    "training_curriculum/sampled/velocity_threshold/std",
    "training_curriculum/episode_length/velocity_threshold",
    "training_curriculum/intended/context/velocity_threshold",
    "training_curriculum/intended/stage",
    "training_curriculum/num_completed_episodes",
    "training_curriculum/num_transitions",
    # reported by the performance tracker through the same progress callback
    *[f"performance/{name}" for name in ("epoch_steps_per_second", "epoch_compiles", "epoch_wall_seconds", "epoch_compile_seconds")],
]

STOCK_RUN_KEYS = [key.replace("evaluation/deployment", "eval") for key in VELOCITY_ANT_KEYS if key.startswith("evaluation/deployment")]

FACTS = dashboard.RunFacts(
    num_timesteps=500_000_000,
    num_evals=21,
    num_eval_envs=128,
    episode_length=1000,
    safety_bound=25.0,
    deployment_distribution="level:3",
    task=dashboard.view.TaskDescription(
        agent="A MuJoCo ant that has to run forward along the x-axis.",
        reward="Per step: forward velocity + 1 for being upright − a penalty on large torques.",
        cost="Per step, 1 if the agent's speed exceeds the episode's velocity_threshold, else 0.",
    ),
    context_space=(dashboard.view.ContextDimension("velocity_threshold", 1.049, 2.622, "the speed above which a step counts as a violation"),),
    num_rounds=763,
    environment_steps_per_round=655_360,
)


def test_every_produced_key_is_registered():
    for key in VELOCITY_ANT_KEYS + STOCK_RUN_KEYS:
        dashboard.registered(key)  # raises UnregisteredMetric otherwise


def test_unregistered_key_raises_with_instructions():
    with pytest.raises(dashboard.UnregisteredMetric, match="training/dashboard/metrics.py"):
        dashboard.registered("training/some_new_thing")


def test_kept_metrics_have_words_for_the_reviewer():
    for metric in dashboard.KEPT:
        assert metric.title and metric.unit and metric.description, metric.pattern
    for dropped in dashboard.DROPPED:
        assert dropped.reason, dropped.pattern


def test_every_panel_is_described_with_no_placeholder_left():
    for group in Group:
        text = dashboard.view.section_text(group, FACTS)
        assert "{" not in text and "}" not in text, text
        for plot in plots_for(group, FACTS):
            assert f"**{plot.title}**" in text, plot.title
            for _, description in plot.described_series:
                assert description in text


def test_select_for_logging_drops_duplicates_and_adds_budgets():
    metrics: Dict[str, object] = {key: 1.0 for key in VELOCITY_ANT_KEYS}
    metrics["training_curriculum/experienced/velocity_threshold"] = wandb.Histogram([1.0, 2.0])
    selected = dashboard.select_for_logging(metrics, safety_bound=25.0)

    kept = [key for key in VELOCITY_ANT_KEYS if dashboard.registered(key) is not None]
    assert set(selected) == set(kept) | {"episodic/cost_budget", "evaluation/deployment/episode_cost_budget", "evaluation/uniform/episode_cost_budget"}
    assert selected["episodic/cost_budget"] == 25.0
    assert isinstance(selected["training_curriculum/experienced/velocity_threshold"], wandb.Histogram)
    for duplicate in ("episodic/velocity_cost", "episodic/reward_survive", "evaluation/deployment/episode_reward_unscaled", "training/sps"):
        assert duplicate not in selected
    assert len(selected) < len(metrics) / 2, "the dashboard should remove most of what the environment logs"


def test_budget_only_on_per_episode_costs():
    assert has_budget("episodic/cost") and has_budget("evaluation/deployment/episode_cost") and has_budget("eval/episode_cost")
    assert not has_budget("training/mean_cost") and not has_budget("evaluation/deployment/episode_cost_std")


def test_view_has_the_four_review_sections_with_text_first_and_no_smoothing():
    view = dashboard.build_view("entity", "crax", "some_group", FACTS)
    sections = view._spec["spec"]["section"]["panelBankConfig"]["sections"]
    assert [section["name"] for section in sections] == ["Verdict", "Mechanism", "Trust", "Detail"]
    assert [section["isOpen"] for section in sections] == [True, True, False, False]
    for section in sections:
        text, *plots = section["panels"]
        assert text["viewType"] == "Markdown Panel" and len(text["config"]["value"]) > 100
        assert plots, section["name"]
        for plot in plots:
            assert plot["viewType"] == "Run History Line Plot"
            assert plot["config"]["smoothingType"] == "none"
            assert plot["config"]["xAxis"] == "environment_steps"
            assert plot["config"]["chartTitle"] and plot["config"]["yAxisTitle"]
    runset = view._spec["spec"]["section"]["runSets"][0]
    assert runset["grouping"][0]["name"].startswith("training_distribution")
    assert "some_group" in str(runset["filters"])


def test_text_panels_state_population_unit_cadence_and_budget():
    verdict, mechanism = (section["panels"][0]["config"]["value"] for section in
                          dashboard.build_view("e", "p", "g", FACTS)._spec["spec"]["section"]["panelBankConfig"]["sections"][:2])
    for fact in ("21 evaluations", "128 episodes", "level:3", "25,000,000", "not data", "velocity_threshold",
                 "**Environment.** A MuJoCo ant", "**Reward.** Per step: forward velocity", "**Cost.** Per step, 1 if",
                 "budget is 25", "128 episodes of the frozen policy on the deployment task (level:3)", "training_distribution"):
        assert fact in verdict, fact
    for fact in ("One point per round", "655,360", "763 rounds", "**sampled**", "**experienced**"):
        assert fact in mechanism, fact


def test_verdict_panels_are_return_and_cost_with_budget_on_both_evaluations():
    plots = plots_for(Group.VERDICT, FACTS)
    assert [plot.title for plot in plots] == [
        "Return per episode on the deployment task (level:3)", "Return per episode on all of Ω (uniform)",
        "Cost per episode on the deployment task (level:3)", "Cost per episode on all of Ω (uniform)",
    ]
    assert plots[2].series == ("evaluation/deployment/episode_cost", "evaluation/deployment/episode_cost_budget")


def test_trust_shows_throughput_and_compiles():
    titles = [plot.title for plot in plots_for(Group.TRUST, FACTS)]
    assert "Throughput" in titles and "Compiles per round" in titles


def test_mechanism_merges_sampled_experienced_intended_means_into_one_panel():
    plots = plots_for(Group.MECHANISM, FACTS)
    mean_context = plots[0]
    assert mean_context.title == "velocity_threshold: mean over the round's training data"
    assert mean_context.series == (
        "training_curriculum/experienced/velocity_threshold/mean",
        "training_curriculum/sampled/velocity_threshold/mean",
        "training_curriculum/intended/context/velocity_threshold",
    )
    assert not any("training_curriculum/experienced/velocity_threshold" == key for plot in plots for key in plot.series), "histograms are heatmaps, not line panels"


def test_run_facts_round_trip_through_json():
    """The saved view stores the facts it was built from; equality after the round-trip is the group invariant."""
    assert dashboard.RunFacts.from_json(FACTS.to_json()) == FACTS
    assert dashboard.RunFacts.from_json(dataclasses.replace(FACTS, safety_bound=10.0).to_json()) != FACTS


def test_run_facts_require_dashboard_config_keys():
    with pytest.raises(KeyError, match="num_rounds"):
        dashboard.RunFacts.from_wandb_config({"num_timesteps": 1, "num_evals": 2, "num_eval_envs": 1,
                                               "episode_length": 1, "safety_bound": 1, "deployment_distribution": "level:3",
                                               "task": {"agent": "a", "reward": "r", "cost": "c"},
                                               "context_space": {"a": {"low": 0, "high": 1, "description": "a"}}})
    with pytest.raises(ValueError, match="num_evals"):
        dataclasses.replace(FACTS, num_evals=1)
