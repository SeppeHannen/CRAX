"""Per-limb geom colouring for pixel observations.

Why? A symmetric agent rendered in one flat colour is ambiguous from a camera that
does not rotate with the body, i.e., front-left and front-right legs project to the
same silhouette once the agent turns, so the policy cannot tell which limb it
is looking at. Here we therefore give each limb its own colour. This removes that
ambiguity without touching physics. Intended for `--vision` runs, enabled with
`--vision_limb_colors`. Apply to the model before the wrapper is constructed.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

LIMB_PALETTE: List[Tuple[float, float, float, float]] = [
    (0.85, 0.15, 0.15, 1.0),  # red
    (0.15, 0.35, 0.85, 1.0),  # blue
    (0.15, 0.70, 0.25, 1.0),  # green
    (0.95, 0.80, 0.10, 1.0),  # yellow
    (0.60, 0.25, 0.75, 1.0),  # purple
    (0.95, 0.55, 0.10, 1.0),  # orange
    (0.10, 0.75, 0.80, 1.0),  # cyan
    (0.95, 0.45, 0.70, 1.0),  # pink
]

# Geoms making up each limb, per agent
LIMB_GEOMS: Dict[str, List[List[str]]] = {
    "ant": [
        ["aux_1_geom", "left_leg_geom", "left_ankle_geom", "left_foot_geom"],
        ["aux_2_geom", "right_leg_geom", "right_ankle_geom", "right_foot_geom"],
        ["aux_3_geom", "back_leg_geom", "third_ankle_geom", "third_foot_geom"],
        ["aux_4_geom", "rightback_leg_geom", "fourth_ankle_geom", "fourth_foot_geom"],
    ],
}


def limb_geoms_for_env(env_name: str) -> Optional[List[List[str]]]:
    """Limb geom groups for `env_name`, by first substring match, else None.

    Matches the `morphology_override` convention used for vision cameras, so
    e.g., `safe_velocity_ant` and `ant` both resolve to the ant's limbs.
    """
    for agent, limbs in LIMB_GEOMS.items():
        if agent in env_name:
            return limbs
    return None


def colorize_limbs(mj_model, limbs: Sequence[Sequence[str]],
                   palette: Sequence[Tuple[float, float, float, float]] = LIMB_PALETTE) -> int:
    """Recolour each limb of `mj_model` in place. Returns geoms changed.

    Geoms outside any limb (torso, floor) are left alone, so the body stays
    visually distinct from the limbs.
    """
    import mujoco

    changed = 0
    for limb_index, geom_names in enumerate(limbs):
        color = np.asarray(palette[limb_index % len(palette)],
                           dtype=mj_model.geom_rgba.dtype)
        for geom_name in geom_names:
            geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
            if geom_id == -1:
                available = [mj_model.geom(i).name for i in range(mj_model.ngeom)]
                raise ValueError(
                    f"Limb geom '{geom_name}' not found in model. Available: {available}"
                )
            mj_model.geom_rgba[geom_id] = color
            changed += 1
    return changed


def colorize_env_limbs(env, env_name: str) -> int:
    """Colour `env`'s limbs in place from its name. Returns geoms changed.

    No-op (returns 0) for an agent with no entry in `LIMB_GEOMS`.
    """
    limbs = limb_geoms_for_env(env_name)
    if not limbs:
        return 0
    return colorize_limbs(env.sys.mj_model, limbs)
