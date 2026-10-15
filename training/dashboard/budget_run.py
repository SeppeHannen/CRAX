"""The budget line on the cost panels: a reference run, one per experiment group.

W&B line panels cannot draw a horizontal reference line — a constant custom
expression renders nothing, and adding any expression hides the logged series.
What a panel can draw is a run. So each group holds one *reference run* whose
only history is the budget, logged at the first and last environment step of
the experiment, under every per-episode cost key the group's panels show. W&B
joins the two points into one flat line across the panel, and the run's
``arm`` value puts it in the legend as ``budget`` (grey, ``view.ARM_COLOURS``).

The run is the group's object, like its view: :func:`save.ensure_view` creates
it with the view on the group's first run and never again. Its facts are the
group's facts, so a group never has two budgets.
"""
from __future__ import annotations

from typing import Dict

import wandb

from training.dashboard.metrics import COSTS_WITH_BUDGET, Metric
from training.dashboard.view import BUDGET_ARM, GROUP_RUNS_BY, X_AXIS, RunFacts

JOB_TYPE = "reference"


def budget_run_id(group: str) -> str:
    """The run id, derived from the group so a second creation is refused by W&B, not duplicated."""
    return f"{group}__{BUDGET_ARM}"


def budget_run_exists(api: wandb.Api, entity: str, project: str, group: str) -> bool:
    try:
        api.run(f"{entity}/{project}/{budget_run_id(group)}")
    except wandb.errors.CommError:
        return False
    return True


def _cost_keys(facts: RunFacts) -> Dict[str, float]:
    """Every per-episode cost key the group's panels carry, each at the budget."""
    keys = []
    for pattern in COSTS_WITH_BUDGET:
        if "{evaluation}" in pattern:
            keys.extend(Metric.fill(pattern, evaluation=prefix) for prefix, _ in facts.evaluation_prefixes)
        else:
            keys.append(pattern)
    return {key: facts.safety_bound for key in keys}


def create_budget_run(entity: str, project: str, group: str, facts: RunFacts) -> str:
    """Log the budget at step 0 and at the last step under every cost key; returns the run's URL.

    The run is finished immediately: it has nothing else to say.
    """
    run = wandb.init(
        entity=entity,
        project=project,
        group=group,
        id=budget_run_id(group),
        name=BUDGET_ARM,
        job_type=JOB_TYPE,
        config={GROUP_RUNS_BY: BUDGET_ARM, "safety_bound": facts.safety_bound},
        resume="never",
    )
    costs = _cost_keys(facts)
    for step in (0, facts.num_timesteps):
        run.log({**costs, X_AXIS: step}, step=step)
    url = run.url
    run.finish()
    return url
