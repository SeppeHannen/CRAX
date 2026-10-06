"""The arena fence: four static walls around the placement square.

The walls keep the robot in the arena and nothing else. They are not hazards:
they appear in no lidar, are no compass target, incur no cost and take part in
no placement (docs/acl/README.md, Decisions, 2026-10-06). The stock benchmark
built them through the hazard pipeline, which gave them all four of those roles.
"""
from __future__ import annotations

import dataclasses
from typing import List, Tuple

PlacementExtents = Tuple[float, float, float, float]  # (min_x, min_y, max_x, max_y)


@dataclasses.dataclass(frozen=True)
class ArenaWalls:
    """Four axis-aligned box geoms fencing a placement square.

    The wall centre lines sit ``offset`` outside the placement square, so
    nothing is ever placed against a wall; ``thickness`` is the wall's half
    thickness; ``height`` its full height. Same geometry as the stock
    benchmark's walls, so the physics is unchanged.
    """

    offset: float
    thickness: float
    height: float

    def geoms(self, placement_extents: PlacementExtents) -> List[str]:
        """The four walls as MuJoCo ``<geom>`` strings for the world body."""
        min_x, min_y, max_x, max_y = placement_extents
        half_x = 0.5 * (max_x - min_x) + self.offset
        half_y = 0.5 * (max_y - min_y) + self.offset
        centre_x = 0.5 * (min_x + max_x)
        centre_y = 0.5 * (min_y + max_y)
        half_height = 0.5 * self.height
        # (name, centre, half-extents): the two side walls run along y, the two end walls along x.
        walls = [
            ("wall_west", (centre_x - half_x, centre_y), (self.thickness, half_y)),
            ("wall_east", (centre_x + half_x, centre_y), (self.thickness, half_y)),
            ("wall_south", (centre_x, centre_y - half_y), (half_x, self.thickness)),
            ("wall_north", (centre_x, centre_y + half_y), (half_x, self.thickness)),
        ]
        return [
            f'        <geom name="{name}" type="box" pos="{x} {y} {half_height}" '
            f'size="{size_x} {size_y} {half_height}" rgba="0.9 0.3 0.3 0.8" '
            f'contype="1" conaffinity="1" condim="3" friction="1 .03 .003" solref="0.01 1"/>'
            for name, (x, y), (size_x, size_y) in walls
        ]
