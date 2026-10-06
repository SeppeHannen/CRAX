"""The union of a suite's hazard specs across its difficulty levels.

A hazard count is the size of MuJoCo arrays, so the levels of a hazard suite
are, as written in ``difficulty.py``, three different compiled programs. To
make them three *contexts* of one program (``docs/acl/design/hazard_activation.md``),
the environment is built from the **union**: every ``(type, collidable)`` group
any level uses, at that group's maximum count. A level is then the per-group
active counts, read from the context, with everything else parked.

Size and height differ between levels within a group (goal's cylinders are 0.4 m
at levels 1–2 and 0.35 m at level 3). The union takes the largest of each, so a
level's hazards may be slightly bigger than in the stock benchmark. Flagged to
Tristan; a per-slot ``geom_size`` would remove the difference.
"""
from __future__ import annotations

import dataclasses
from typing import Dict, List, Mapping, Sequence, Tuple

HazardSpec = Dict[str, object]
GroupKind = Tuple[str, bool]  # (hazard type, collidable)

# Spec keys that identify a group; every other key is a per-group parameter merged by max.
_KIND_KEYS = ("type", "collidable")


def _kind(spec: HazardSpec) -> GroupKind:
    return str(spec["type"]), bool(spec["collidable"])


@dataclasses.dataclass(frozen=True)
class HazardUnion:
    """The union model's hazard specs and the per-level active counts within it."""

    specs: List[HazardSpec]  # one spec per group, in kind order
    kinds: Tuple[GroupKind, ...]  # the groups, in the order they appear in ``specs``
    counts_by_level: Mapping[int, Tuple[int, ...]]  # per level, the active count of each kind

    def counts(self, level: int) -> Tuple[int, ...]:
        return self.counts_by_level[level]


def union_of_levels(specs_by_level: Mapping[int, Sequence[HazardSpec]]) -> HazardUnion:
    """Merge each level's hazard specs into one list covering all of them.

    Groups are keyed by ``(type, collidable)``. For each group, the union spec
    has the maximum ``count`` over levels and, for every other numeric field,
    the maximum over the levels that have the group.
    """
    per_kind: Dict[GroupKind, HazardSpec] = {}
    order: List[GroupKind] = []
    for level in sorted(specs_by_level):
        for spec in specs_by_level[level]:
            kind = _kind(spec)
            if kind not in per_kind:
                per_kind[kind] = dict(spec)
                order.append(kind)
                continue
            merged = per_kind[kind]
            for key, value in spec.items():
                if key in _KIND_KEYS:
                    continue
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    merged[key] = max(merged[key], value)
                elif merged[key] != value:
                    raise ValueError(f"hazard group {kind}: field {key!r} differs between levels ({merged[key]!r} vs {value!r}) and is not numeric")

    counts_by_level = {}
    for level, specs in specs_by_level.items():
        counts = {kind: 0 for kind in order}
        for spec in specs:
            counts[_kind(spec)] += int(spec["count"])
        counts_by_level[level] = tuple(counts[kind] for kind in order)

    return HazardUnion(specs=[per_kind[kind] for kind in order], kinds=tuple(order), counts_by_level=counts_by_level)
