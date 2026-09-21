"""The W&B dashboard: which metrics a run logs, what they mean, and the saved
view that presents them by review question (``docs/acl/design/dashboard.md``).

* :mod:`metrics` — the registry every logged key passes through.
* :mod:`view` — the workspace built from the registry and a run's facts.
* ``python -m training.dashboard --group <name>`` saves the view for a group.
"""

from training.dashboard.metrics import KEPT, DROPPED, Group, Metric, UnregisteredMetric, registered, select_for_logging
from training.dashboard.view import RunFacts, build_view

__all__ = [
    "DROPPED",
    "Group",
    "KEPT",
    "Metric",
    "RunFacts",
    "UnregisteredMetric",
    "build_view",
    "registered",
    "select_for_logging",
]
