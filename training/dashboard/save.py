"""Saving the view of an experiment group, and the invariant it protects.

A W&B group is one experiment: every run in it shares the facts the dashboard
states (budget, evaluations, episode length, Ω, deployment task, ...). The view
is the group's one shared object, so it is also the group's **record of those
facts**: it stores the :class:`RunFacts` it was built from in its description.
:func:`ensure_view` runs at the start of every run with a named group, before
any compute, and raises when the run's facts differ from the group's.

The workspaces SDK creates a new view on every ``save()`` and cannot write a
description, so the two GraphQL calls a view needs — find by name, upsert —
live here, and nowhere else.
"""
from __future__ import annotations

import dataclasses
from typing import Optional

import wandb
import wandb_workspaces.workspaces as workspaces
from wandb_workspaces._graphql import execute_graphql
from wandb_workspaces.workspaces.internal import _generate_view_name, _internal_name_to_url_query_str

from training.dashboard.view import RunFacts, build_view


@dataclasses.dataclass(frozen=True)
class SavedView:
    id: str
    url: str
    facts: RunFacts


def _view_url(entity: str, project: str, internal_name: str) -> str:
    return f"https://wandb.ai/{entity}/{project}?nw={_internal_name_to_url_query_str(internal_name)}"


class FactsMismatch(RuntimeError):
    """This run's facts differ from those the group's view (and hence its other runs) were built from."""


_FIND = """query Views($entity: String!, $project: String!) {
  project(name: $project, entityName: $entity) {
    allViews(viewType: "project-view") { edges { node { id name displayName description } } } } }"""

_UPSERT = """mutation UpsertView($id: ID, $entity: String, $project: String, $name: String, $displayName: String,
                                $description: String, $spec: String) {
  upsertView(input: {id: $id, entityName: $entity, projectName: $project, name: $name, displayName: $displayName,
                     description: $description, type: "project-view", spec: $spec, createdUsing: WANDB_SDK}) {
    view { id name } } }"""


def find_view(api: wandb.Api, entity: str, project: str, group: str) -> Optional[SavedView]:
    """The project's view named after the group, or ``None``.

    Raises if a view of that name exists that this module did not save (no
    facts in its description): it cannot serve as the group's record.
    """
    result = execute_graphql(api, _FIND, {"entity": entity, "project": project})
    for edge in result["project"]["allViews"]["edges"]:
        node = edge["node"]
        if node["displayName"] != group:
            continue
        if not node["description"]:
            raise RuntimeError(
                f"view {group!r} exists but was not saved by training.dashboard (it carries no run facts); "
                "delete it in W&B or use another --wandb_group"
            )
        return SavedView(id=node["id"], url=_view_url(entity, project, node["name"]), facts=RunFacts.from_json(node["description"]))
    return None


def _upsert(api: wandb.Api, view: workspaces.Workspace, facts: RunFacts, existing: Optional[SavedView]) -> str:
    """Create or overwrite the view; returns its URL."""
    spec = view._to_model().spec.model_dump_json(by_alias=True, exclude_none=True)
    result = execute_graphql(
        api,
        _UPSERT,
        {
            "id": existing.id if existing else None,
            "entity": view.entity,
            "project": view.project,
            "name": _generate_view_name(),  # ignored by W&B when `id` is given
            "displayName": view.name,
            "description": facts.to_json(),
            "spec": spec,
        },
    )
    return _view_url(view.entity, view.project, result["upsertView"]["view"]["name"])


def _differences(expected: RunFacts, actual: RunFacts) -> str:
    return "\n".join(
        f"  {field.name}: group has {getattr(expected, field.name)!r}, this run has {getattr(actual, field.name)!r}"
        for field in dataclasses.fields(RunFacts)
        if getattr(expected, field.name) != getattr(actual, field.name)
    )


def ensure_view(entity: str, project: str, group: str, facts: RunFacts) -> None:
    """Create the group's view from ``facts`` if the group is new; otherwise require the facts to match.

    Raises :class:`FactsMismatch` for a run whose flags differ from the
    experiment's: fix the flags, or use another ``--wandb_group``.
    """
    api = wandb.Api()
    existing = find_view(api, entity, project, group)
    if existing is None:
        url = _upsert(api, build_view(entity, project, group, facts), facts, None)
        print(f"dashboard: created view {group!r}: {url}")
    elif existing.facts != facts:
        raise FactsMismatch(
            f"group {group!r} is one experiment and this run's flags differ from it:\n{_differences(existing.facts, facts)}\n"
            "Fix the flags so they match, or start a new experiment with another --wandb_group."
        )
    else:
        print(f"dashboard: {existing.url}")


def rebuild_view(entity: str, project: str, group: str, facts: RunFacts) -> None:
    """Overwrite the group's view (or create it) from ``facts`` — the deliberate path after dashboard code changed."""
    api = wandb.Api()
    existing = find_view(api, entity, project, group)
    url = _upsert(api, build_view(entity, project, group, facts), facts, existing)
    print(f"dashboard: {'updated' if existing else 'created'} view {group!r}: {url}")
