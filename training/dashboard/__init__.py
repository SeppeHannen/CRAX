"""The W&B dashboard: which metrics a run logs, what they mean, and the saved
view that presents them by review question (``docs/acl/design/dashboard.md``).

* :mod:`metrics` — the registry every logged key passes through.
* :mod:`view` — the workspace built from the registry and a run's facts.
* :mod:`save` — the group's view as the record of its facts: created by the
  first run of a group, enforced on every later one (``ensure_view``), rebuilt
  deliberately with ``python -m training.dashboard --group <name>``.
"""

from training.dashboard.metrics import KEPT, DROPPED, Group, Metric, UnregisteredMetric, registered, select_for_logging
from training.dashboard.save import FactsMismatch, ensure_view, rebuild_view
from training.dashboard.view import RunFacts, build_view

__all__ = [
    "DROPPED",
    "FactsMismatch",
    "Group",
    "KEPT",
    "Metric",
    "RunFacts",
    "UnregisteredMetric",
    "build_view",
    "ensure_view",
    "rebuild_view",
    "registered",
    "select_for_logging",
]
