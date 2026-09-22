"""The saved W&B view for one experiment group: what a reviewer sees, in the
order they need it, with the words that make each panel readable without the
code (``docs/acl/design/dashboard.md`` §7).

Four sections, one per review question — Verdict, Mechanism, Trust, Detail.
Each opens with a text panel, then the line plots for the kept metrics of that
group (:mod:`training.dashboard.metrics`). The text is generated, not written:
a preamble that states what is common to the section (cadence, populations,
distributions, what the legend means) and then one line per panel, built from
the registry's ``description``, so every graph says where its numbers come
from. Runs are grouped by training distribution so arms overlay with a min–max
band over seeds. Smoothing is off everywhere: a point is drawn only where a
number was computed.

Numbers and names come from the group's ``wandb.config`` (:class:`RunFacts`),
so the same builder serves every suite.
"""
from __future__ import annotations

import dataclasses
import json
from typing import Dict, List, Mapping, Sequence, Tuple

import wandb_workspaces.reports.v2 as panels
import wandb_workspaces.workspaces as workspaces
from wandb_workspaces import expr

from training.dashboard.metrics import FACT_PLACEHOLDERS, KEPT, Group, Metric, budget_key, has_budget

X_AXIS = "environment_steps"
# The arm of the experiment. This config key names what a line *is* (the
# distribution the run trained on), so the legend reads `training_distribution: uniform`.
GROUP_RUNS_BY = "training_distribution"

# The two evaluation distributions every context run is scored on, as they appear in keys.
DEPLOYMENT_EVALUATION = "evaluation/deployment"
UNIFORM_EVALUATION = "evaluation/uniform"

# Panel grid: 24 columns wide; a text panel spans the width, plots are half width.
TEXT_PANEL_HEIGHT = 9
PLOT_WIDTH, PLOT_HEIGHT = 12, 8


@dataclasses.dataclass(frozen=True)
class ContextDimension:
    """One coordinate of Ω, with the words the suite uses for it."""

    name: str
    low: float
    high: float
    description: str

    def describe(self) -> str:
        return f"`{self.name}` ∈ [{self.low:.3g}, {self.high:.3g}] — {self.description}"


@dataclasses.dataclass(frozen=True)
class TaskDescription:
    """The suite's task as its registry states it: agent, reward and cost in words, and
    whether an episode can end before the step limit (decides how episode length reads)."""

    agent: str
    reward: str
    cost: str
    episode_ends_early: bool


def _context_dimension(name: str, bounds: object) -> ContextDimension:
    """One entry of ``wandb.config["context_space"]`` as written by ``ContextTrainingSetup.wandb_config``."""
    if not isinstance(bounds, Mapping) or set(bounds) != {"low", "high", "description"}:
        raise ValueError(
            f"context_space[{name!r}] must be {{low, high, description}}, got {bounds!r}; "
            "the run predates the dashboard's config format"
        )
    return ContextDimension(name, float(bounds["low"]), float(bounds["high"]), str(bounds["description"]))


@dataclasses.dataclass(frozen=True)
class RunFacts:
    """The facts about a run that the text panels state, read from ``wandb.config``."""

    num_timesteps: int
    num_evals: int
    num_eval_envs: int
    episode_length: int
    safety_bound: float
    deployment_distribution: str
    task: TaskDescription
    context_space: Tuple[ContextDimension, ...]
    num_rounds: int
    environment_steps_per_round: int

    def __post_init__(self) -> None:
        if self.num_evals < 2:
            raise ValueError("num_evals must be at least 2 (an initial evaluation and one after training)")

    @classmethod
    def from_wandb_config(cls, config: Mapping[str, object]) -> "RunFacts":
        """Raises ``KeyError`` naming the missing key when a run predates the dashboard."""
        return cls(
            num_timesteps=int(float(config["num_timesteps"])),
            num_evals=int(config["num_evals"]),
            num_eval_envs=int(config["num_eval_envs"]),
            episode_length=int(config["episode_length"]),
            safety_bound=float(config["safety_bound"]),
            deployment_distribution=str(config["deployment_distribution"]),
            task=TaskDescription(**config["task"]),
            context_space=tuple(_context_dimension(name, bounds) for name, bounds in config["context_space"].items()),
            num_rounds=int(config["num_rounds"]),
            environment_steps_per_round=int(config["environment_steps_per_round"]),
        )

    @property
    def steps_between_evaluations(self) -> int:
        return self.num_timesteps // (self.num_evals - 1)

    @property
    def context_dimensions(self) -> Sequence[str]:
        return tuple(dimension.name for dimension in self.context_space)

    @property
    def evaluations(self) -> Tuple[Tuple[str, str], ...]:
        """``(key prefix, label)`` of each evaluation distribution; the label says what it is and its value."""
        return (
            (DEPLOYMENT_EVALUATION, f"the deployment task ({self.deployment_distribution})"),
            (UNIFORM_EVALUATION, "all of Ω (uniform)"),
        )

    def words(self) -> Dict[str, str]:
        """The values for the registry's :data:`FACT_PLACEHOLDERS`."""
        if self.task.episode_ends_early:
            episode_length_reading = (
                f"An episode lasts {self.episode_length} steps unless the environment ends it early (see the Environment "
                f"definition for when), so a value below {self.episode_length} means episodes were cut short."
            )
        else:
            episode_length_reading = (
                f"In this suite every episode lasts exactly {self.episode_length} steps; this panel is a constant and "
                f"carries no information."
            )
        words = {
            "episode_length": f"{self.episode_length}",
            "num_eval_envs": f"{self.num_eval_envs}",
            "budget": f"{self.safety_bound:g}",
            "episode_length_reading": episode_length_reading,
        }
        if set(words) != set(FACT_PLACEHOLDERS):
            raise RuntimeError(f"RunFacts.words supplies {sorted(words)} but the registry declares {sorted(FACT_PLACEHOLDERS)}")
        return words

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "RunFacts":
        fields = json.loads(text)
        fields["task"] = TaskDescription(**fields["task"])
        fields["context_space"] = tuple(ContextDimension(**dimension) for dimension in fields["context_space"])
        return cls(**fields)


# --------------------------------------------------------------------------- #
# Text panels
# --------------------------------------------------------------------------- #

LEGEND = (
    "**Legend.** Lines are grouped by `training_distribution`, the distribution over contexts that the run trained "
    "on (e.g. `uniform`, `staged:1,2,3`): one line per training distribution, mean over its seeds, band = min–max "
    "over seeds. The legend does not indicate what a line was evaluated on; the panel title does."
)


def _environment(facts: RunFacts) -> str:
    omega = "\n".join(f"- {dimension.describe()}" for dimension in facts.context_space)
    return (
        f"**Environment.** {facts.task.agent}\n\n"
        f"**Reward.** {facts.task.reward}\n\n"
        f"**Cost.** {facts.task.cost} Episodes have at most {facts.episode_length} steps. The budget is "
        f"{facts.safety_bound:g}: a policy satisfies the constraint when its mean cost per episode is at most "
        f"{facts.safety_bound:g}, i.e. its cost line lies below the budget line.\n\n"
        f"**Context space Ω.** Each episode is run at one context ω ∈ Ω, drawn by the distribution in force:\n{omega}"
    )


def _preamble(group: Group, facts: RunFacts) -> str:
    if group is Group.VERDICT:
        return (
            f"### Verdict: performance and constraint satisfaction of the learned policy\n\n"
            f"{_environment(facts)}\n\n"
            f"**Measurement.** The policy is frozen (no exploration noise) and run for {facts.num_eval_envs} episodes "
            f"per evaluation; {facts.num_evals} evaluations over the run, at step 0 and then every "
            f"{facts.steps_between_evaluations:,} environment steps. Each panel shows one point per evaluation; the "
            f"segments between points are interpolation, not data.\n\n"
            f"**Evaluation distributions.** Independently of what it trained on, every policy is evaluated on two "
            f"distributions over Ω, one panel each:\n"
            f"- *the deployment task* (`{facts.deployment_distribution}`, set by `--deployment_distribution`): the "
            f"task the policy is trained for;\n"
            f"- *all of Ω (uniform)*: contexts drawn uniformly from the whole context space, measuring how the policy "
            f"generalises beyond the deployment task.\n\n{LEGEND}"
        )
    if group is Group.MECHANISM:
        return (
            f"### Mechanism: the training distribution as delivered, and the constraint's response\n\n"
            f"**Cadence.** One point per round; a round is one PPO training step of "
            f"{facts.environment_steps_per_round:,} environment steps, {facts.num_rounds} rounds per run.\n\n"
            f"**Population.** Each point summarises the round's own training data — all of its transitions, or the "
            f"training episodes that ended in it — collected under the contexts the training distribution drew and "
            f"with exploration noise. These values are therefore not comparable to the Verdict section's.\n\n"
            f"**Heatmaps.** The distributions over Ω of the sampled and experienced contexts, and of episode length "
            f"per context, are logged as histograms; W&B renders them as heatmaps over rounds in its auto-generated "
            f"section `training_curriculum`.\n\n{LEGEND}"
        )
    if group is Group.TRUST:
        return (
            f"### Trust: indicators of run health\n\n"
            f"Panels on training data show one point per round; panels on evaluation data show one point per "
            f"evaluation ({facts.num_evals} over the run), computed over the same {facts.num_eval_envs} frozen-policy "
            f"episodes as the Verdict section.\n\n{LEGEND}"
        )
    return (
        f"### Detail: remaining diagnostics\n\n"
        f"Cadences and populations as above: `training/*` and `episodic/*` once per round, `evaluation/*` once per "
        f"evaluation. The reward and cost referred to below are defined in the Verdict section.\n\n{LEGEND}"
    )


def _panel_lines(group: Group, facts: RunFacts) -> str:
    """One bullet per panel of the section: its title, then what each series is."""
    lines: List[str] = []
    for plot in plots_for(group, facts):
        if len(plot.described_series) == 1:
            lines.append(f"- **{plot.title}** — {plot.described_series[0][1]}")
        else:
            lines.append(f"- **{plot.title}**:")
            lines.extend(f"  - {description}" for _, description in plot.described_series)
    return "\n".join(lines)


def section_text(group: Group, facts: RunFacts) -> str:
    return f"{_preamble(group, facts)}\n\n**Panels**\n\n{_panel_lines(group, facts)}"


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class Plot:
    """One line panel: several series on one y-axis, each with the sentence that says what it is."""

    title: str
    described_series: Tuple[Tuple[str, str], ...]  # (key, description)
    unit: str

    @property
    def series(self) -> Tuple[str, ...]:
        return tuple(key for key, _ in self.described_series)


@dataclasses.dataclass(frozen=True)
class Instance:
    """One concrete panel of a metric pattern: the key it expands to and the words that name it.

    A pattern with ``{evaluation}`` has one instance per evaluation distribution
    (``in_keys`` = the key prefix, ``in_words`` = its label, shown in the title);
    a pattern with ``{dimension}`` has one per dimension of Ω; a plain key has
    one instance with nothing to fill.
    """

    placeholder: str
    in_keys: str
    in_words: str

    @classmethod
    def plain(cls) -> "Instance":
        return cls(placeholder="", in_keys="", in_words="")

    def key(self, metric: Metric) -> str:
        return metric.key(**{self.placeholder: self.in_keys}) if self.placeholder else metric.pattern

    def words(self, text: str, facts: RunFacts) -> str:
        """``text`` with the run's facts and this instance's words filled in."""
        fills = {**facts.words(), self.placeholder: self.in_words} if self.placeholder else facts.words()
        for placeholder, value in fills.items():
            text = text.replace("{" + placeholder + "}", value)
        return text

    def title(self, metric: Metric, facts: RunFacts) -> str:
        title = self.words(metric.panel_title, facts)
        return f"{title} on {self.in_words}" if self.placeholder == "evaluation" else title


def _instances(metric: Metric, facts: RunFacts) -> List[Instance]:
    if "{evaluation}" in metric.pattern:
        return [Instance("evaluation", prefix, label) for prefix, label in facts.evaluations]
    if "{dimension}" in metric.pattern:
        return [Instance("dimension", name, name) for name in facts.context_dimensions]
    return [Instance.plain()]


def plots_for(group: Group, facts: RunFacts) -> List[Plot]:
    """The line panels of one section, in registry order; metrics sharing a ``panel`` share one.

    Histogram metrics are not line panels: W&B renders them as heatmaps in its
    auto-generated `training_curriculum` section.
    """
    series_by_title: Dict[str, List[Tuple[str, str]]] = {}
    unit_by_title: Dict[str, str] = {}
    for metric in KEPT:
        if metric.group is not group or metric.histogram:
            continue
        for instance in _instances(metric, facts):
            title = instance.title(metric, facts)
            key = instance.key(metric)
            series = series_by_title.setdefault(title, [])
            series.append((key, instance.words(metric.description, facts)))
            if has_budget(key):
                series.append((budget_key(key), f"the budget, {facts.safety_bound:g} per episode."))
            unit_by_title[title] = instance.words(metric.unit, facts)
    return [Plot(title, tuple(series), unit_by_title[title]) for title, series in series_by_title.items()]


def _line_panel(plot: Plot, layout: panels.Layout) -> panels.LinePlot:
    return panels.LinePlot(
        title=plot.title,
        x=X_AXIS,
        y=list(plot.series),
        title_x="environment steps",
        title_y=plot.unit,
        smoothing_type="none",
        groupby=panels.Config(GROUP_RUNS_BY),
        groupby_aggfunc="mean",
        groupby_rangefunc="minmax",
        layout=layout,
    )


OPEN_SECTIONS = (Group.VERDICT, Group.MECHANISM)


def _section(group: Group, facts: RunFacts) -> workspaces.Section:
    text = panels.MarkdownPanel(markdown=section_text(group, facts), layout=panels.Layout(x=0, y=0, w=24, h=TEXT_PANEL_HEIGHT))
    line_panels = [
        _line_panel(
            plot,
            panels.Layout(x=(index % 2) * PLOT_WIDTH, y=TEXT_PANEL_HEIGHT + (index // 2) * PLOT_HEIGHT, w=PLOT_WIDTH, h=PLOT_HEIGHT),
        )
        for index, plot in enumerate(plots_for(group, facts))
    ]
    return workspaces.Section(
        name=group.value,
        panels=[text, *line_panels],
        is_open=group in OPEN_SECTIONS,
        pinned=group is Group.VERDICT,
        panel_settings=workspaces.SectionPanelSettings(x_axis=X_AXIS, smoothing_type="none"),
    )


def build_view(entity: str, project: str, group_name: str, facts: RunFacts) -> workspaces.Workspace:
    """The workspace for one experiment group, ready to ``save()``."""
    return workspaces.Workspace(
        entity=entity,
        project=project,
        name=group_name,
        sections=[_section(group, facts) for group in Group],
        settings=workspaces.WorkspaceSettings(
            x_axis=X_AXIS,
            smoothing_type="none",
            point_visualization_method="bucketing",
            group_by_prefix="first",
        ),
        runset_settings=workspaces.RunsetSettings(
            filters=f"Group = '{group_name}'",
            groupby=[expr.Config(GROUP_RUNS_BY)],
        ),
    )
