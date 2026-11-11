"""The arena fence: four static walls around the placement square.

The walls keep the robot in the arena and nothing else. They are not hazards:
they appear in no lidar, are no compass target, incur no cost and take part in
no placement (docs/acl/decisions/standing_decisions.md, 2026-10-06). The stock benchmark
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

    def centre_lines(self, placement_extents: PlacementExtents) -> PlacementExtents:
        """Where the walls' centre lines run, as ``(min_x, min_y, max_x, max_y)``: a point beyond
        them is outside the arena."""
        min_x, min_y, max_x, max_y = placement_extents
        return (min_x - self.offset, min_y - self.offset, max_x + self.offset, max_y + self.offset)

    def geoms(self, placement_extents: PlacementExtents) -> List[str]:
        """The four walls as MuJoCo ``<geom>`` strings for the world body."""
        west, south, east, north = self.centre_lines(placement_extents)
        half_x = 0.5 * (east - west)
        half_y = 0.5 * (north - south)
        centre_x = 0.5 * (west + east)
        centre_y = 0.5 * (south + north)
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

