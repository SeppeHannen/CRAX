"""Save the dashboard view for one experiment group.

    .venv/bin/python -m training.dashboard --group velocity_ant_staged_vs_uniform_500M

Reads the group's runs from W&B, takes the run facts from the first run's
config (all runs of a group share them by construction), builds the view and
saves it. Saving the same group again updates the existing view.
"""
from __future__ import annotations

import argparse

import wandb
import wandb_workspaces.workspaces as workspaces

from training.config import WANDB_ENTITY, WANDB_PROJECT
from training.dashboard.view import RunFacts, build_view


def facts_of_group(api: wandb.Api, entity: str, project: str, group_name: str) -> RunFacts:
    """The facts the group's runs share; raises if two runs disagree (the text panels would lie for one of them)."""
    runs = list(api.runs(f"{entity}/{project}", filters={"group": group_name}))
    if not runs:
        raise SystemExit(f"no runs in group {group_name!r} of {entity}/{project}")
    facts_by_run = {run.name: RunFacts.from_wandb_config(run.config) for run in runs}
    distinct = set(facts_by_run.values())
    if len(distinct) > 1:
        raise SystemExit(
            f"runs in group {group_name!r} differ in the facts the dashboard states; one view cannot describe them all:\n"
            + "\n".join(f"  {name}: {facts}" for name, facts in facts_by_run.items())
        )
    print(f"{len(runs)} runs in group {group_name!r}")
    return distinct.pop()


def point_at_existing_view(view: workspaces.Workspace, api: wandb.Api) -> None:
    """Make ``view.save()`` update the saved view of the same name instead of creating a second one.

    The workspaces SDK creates a new view unless the workspace carries the
    saved view's id and internal name; those are only public via GraphQL.
    """
    query = """query Views($entity: String!, $project: String!) {
      project(name: $project, entityName: $entity) {
        allViews(viewType: "project-view") { edges { node { id name displayName } } } } }"""
    result = api.client.execute(wandb.apis.public.gql(query), {"entity": view.entity, "project": view.project})
    for edge in result["project"]["allViews"]["edges"]:
        if edge["node"]["displayName"] == view.name:
            view._internal_id = edge["node"]["id"]
            view._internal_name = edge["node"]["name"]
            print(f"updating existing view {view.name!r}")
            return
    print(f"creating view {view.name!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--group", required=True, help="W&B group of the experiment (--wandb_group of its runs)")
    parser.add_argument("--entity", default=WANDB_ENTITY)
    parser.add_argument("--project", default=WANDB_PROJECT)
    arguments = parser.parse_args()

    api = wandb.Api()
    facts = facts_of_group(api, arguments.entity, arguments.project, arguments.group)
    view = build_view(arguments.entity, arguments.project, arguments.group, facts)
    point_at_existing_view(view, api)
    view.save()


if __name__ == "__main__":
    main()
