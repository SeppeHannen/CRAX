"""Rebuild the dashboard view of an experiment group.

    .venv/bin/python -m training.dashboard --group velocity_ant_staged_vs_uniform_750M

Every context run with a named ``--wandb_group`` creates the group's view at
start-up (see :mod:`training.dashboard.save`). This command is for the other
case: the dashboard code changed and an existing group's view should be
rebuilt. It takes the facts from the group's runs — which agree, or the runs
would not have been allowed into the group — and overwrites the view.
"""
from __future__ import annotations

import argparse

import wandb

from training.config import WANDB_ENTITY, WANDB_PROJECT
from training.dashboard.save import rebuild_view
from training.dashboard.view import RunFacts


def facts_of_group(api: wandb.Api, entity: str, project: str, group: str) -> RunFacts:
    """The facts the group's runs share; raises if two runs disagree (the text panels would lie for one of them)."""
    runs = list(api.runs(f"{entity}/{project}", filters={"group": group}))
    if not runs:
        raise SystemExit(f"no runs in group {group!r} of {entity}/{project}")
    facts_by_run = {run.name: RunFacts.from_wandb_config(run.config) for run in runs}
    distinct = set(facts_by_run.values())
    if len(distinct) > 1:
        raise SystemExit(
            f"runs in group {group!r} differ in the facts the dashboard states; one view cannot describe them all:\n"
            + "\n".join(f"  {name}: {facts}" for name, facts in facts_by_run.items())
        )
    print(f"{len(runs)} runs in group {group!r}")
    return distinct.pop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--group", required=True, help="W&B group of the experiment (--wandb_group of its runs)")
    parser.add_argument("--entity", default=WANDB_ENTITY)
    parser.add_argument("--project", default=WANDB_PROJECT)
    arguments = parser.parse_args()

    facts = facts_of_group(wandb.Api(), arguments.entity, arguments.project, arguments.group)
    rebuild_view(arguments.entity, arguments.project, arguments.group, facts)


if __name__ == "__main__":
    main()
