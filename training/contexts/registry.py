"""Per-suite context spaces and the mapping from CRAX difficulty levels to contexts.

This is the single place that knows *what* varies in each suite. Environments
know how to *read* a context (see :func:`training.contexts.wrapper`), and
distributions know how to *choose* one; neither needs to know the suite.

Adding a suite: write a :class:`SuiteContexts` (Ω and the task in words) and
register it. The level contexts are not written down: level ``n`` is whatever
``get_environment(env, level=n)`` resets into by default (``default_context()``,
``crax/envs/context.py``), so ``FixedContext(level(3))`` is the level-3 task by
construction and cannot drift from ``crax/envs/difficulty.py``.

Every suite of the benchmark is registered. Value-typed knobs (a threshold, a
radius) are read from the context where the constant used to be; count-typed
knobs (how many hazards) are per-group active counts in a union model of the
levels (``docs/acl/design/hazard_activation.md``).
"""
from __future__ import annotations

import dataclasses
from typing import Callable, Dict, Mapping, Tuple

import jax.numpy as jnp

from training.contexts.space import Context, ContextSpace, Dimension

LEVELS = (1, 2, 3)


@dataclasses.dataclass(frozen=True)
class TaskDescription:
    """The suite's task in words, for a reader of the dashboard who has not seen the environment.

    ``agent``, ``reward`` and ``cost`` are one or two sentences each; the
    dashboard quotes them and refers to "the reward" and "the cost" defined here
    in every panel description. ``episode_ends_early`` is the one fact the
    dashboard needs as a value rather than as words: whether the environment can
    end an episode before the step limit (a fall, a stall), which decides how the
    episode-length panels are to be read. ``tests/test_suites.py`` checks it
    against the environment.
    """

    agent: str
    reward: str
    cost: str
    episode_ends_early: bool


@dataclasses.dataclass(frozen=True)
class SuiteContexts:
    """Everything the context machinery needs to know about one environment."""

    env_name: str
    # Ω. Its dimension names are, in order, the environment's ``CONTEXT_PARAMETERS``
    # (crax/envs/context.py); tests/test_suites.py checks the two agree.
    space: ContextSpace
    level_contexts: Mapping[int, Context]
    task: TaskDescription

    def __post_init__(self) -> None:
        for level, context in self.level_contexts.items():
            if context.shape != (self.space.size,):
                raise ValueError(f"{self.env_name}: level {level} context has shape {context.shape}, Ω has {self.space.size} dimensions")
            if not bool(self.space.contains(context[None])[0]):
                raise ValueError(f"{self.env_name}: level {level} context {self.space.decode(context)} lies outside Ω\n{self.space.describe()}")

    def level(self, level: int) -> Context:
        if level not in self.level_contexts:
            raise KeyError(f"{self.env_name}: no level {level}; have {sorted(self.level_contexts)}")
        return self.level_contexts[level]

    def levels(self) -> jnp.ndarray:
        """All difficulty levels as a ``[3, D]`` array (for StagedContexts)."""
        return jnp.stack([self.level(l) for l in LEVELS])


_REGISTRY: Dict[str, Callable[[], SuiteContexts]] = {}


def register(env_name: str):
    def decorator(factory: Callable[[], SuiteContexts]):
        _REGISTRY[env_name] = factory
        return factory

    return decorator


def suite_contexts(env_name: str) -> SuiteContexts:
    """The registered :class:`SuiteContexts` for ``env_name``."""
    try:
        return _REGISTRY[env_name]()
    except KeyError:
        raise KeyError(f"No context space registered for {env_name!r}. Registered: {sorted(_REGISTRY)}.") from None


def registered_environments() -> Tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def _level_contexts(env_name: str) -> Dict[int, Context]:
    """The three difficulty levels as contexts: what an environment built at each level resets into by default."""
    from crax import envs

    return {level: envs.get_environment(env_name, level=level).unwrapped.default_context() for level in LEVELS}


# --------------------------------------------------------------------------- #
# Safe Velocity: one dimension, the speed limit.
# --------------------------------------------------------------------------- #

# Ω spans from a bit below the hardest level to the easiest level. The lower
# bound is deliberately not 0: a threshold of 0 makes every step a violation
# and the task degenerate. 0.4 × baseline sits below level 3 (0.5 ×) so the
# uniform distribution also covers "harder than any level".
_VELOCITY_LOW_FRACTION = 0.4
_VELOCITY_HIGH_FRACTION = 1.0

# How each agent's speed is measured for the constraint (SafeVelocity*.velocity_mode).
_VELOCITY_MEASURE = {
    "ant": "the torso's speed in the horizontal plane, √(vₓ² + vᵧ²)",
    "halfcheetah": "the forward velocity vₓ",
    "hopper": "the forward velocity vₓ",
    "humanoid": "the centre of mass's speed in the horizontal plane",
    "swimmer": "the forward velocity vₓ",
    "walker2d": "the forward velocity vₓ",
}

# Agents whose episode ends early when they leave their healthy posture
# (`terminate_when_unhealthy` in crax/envs/<agent>.py); the planar halfcheetah and
# the swimmer cannot fall. Checked against the environments in tests/test_suites.py.
_AGENTS_THAT_CAN_FALL = frozenset({"ant", "hopper", "humanoid", "walker2d"})


def _velocity_task(agent: str) -> TaskDescription:
    # The reward is the agent's stock Brax locomotion reward; its weights differ per agent
    # (crax/envs/<agent>.py) and the dashboard does not need them, so it is described by structure.
    falls = agent in _AGENTS_THAT_CAN_FALL
    if falls:
        termination = (
            "An episode ends either at the step limit or earlier when the agent falls (it leaves its healthy posture "
            "range); violating the constraint does not end the episode."
        )
        upright = ", plus a constant for every step it has not fallen"
    else:
        termination = "An episode ends at the step limit; the agent cannot fall, and violating the constraint does not end it."
        upright = ""
    return TaskDescription(
        agent=f"A MuJoCo {agent} that has to move forward along the x-axis. Actions are joint torques in [−1, 1]. {termination}",
        reward=f"The {agent}'s stock locomotion reward per step: a weight × the forward velocity vₓ{upright}, minus a "
        f"penalty proportional to ‖action‖² (large torques), with the agent's own weights, multiplied by the suite's "
        f"reward scale of 0.01. The return is the sum over the episode; higher means it travelled further with less effort.",
        cost=f"Per step, 1 if the agent's speed — {_VELOCITY_MEASURE[agent]} — exceeds the episode's velocity_threshold, "
        f"else 0. The episode's cost is the number of steps over the limit; the budget bounds its mean.",
        episode_ends_early=falls,
    )


def _velocity_suite(agent: str) -> Callable[[], SuiteContexts]:
    def factory() -> SuiteContexts:
        from crax.envs.safe_velocity import DEFAULT_THRESHOLDS

        env_name = f"safe_velocity_{agent}"
        baseline = DEFAULT_THRESHOLDS[agent]
        space = ContextSpace((
            Dimension(
                "velocity_threshold",
                low=_VELOCITY_LOW_FRACTION * baseline,
                high=_VELOCITY_HIGH_FRACTION * baseline,
                description=f"the speed above which a step counts as a violation (the {agent}'s speed limit)",
            ),
        ))
        return SuiteContexts(
            env_name=env_name,
            space=space,
            level_contexts=_level_contexts(env_name),
            task=_velocity_task(agent),
        )

    return factory


for _agent in ("ant", "halfcheetah", "hopper", "humanoid", "swimmer", "walker2d"):
    register(f"safe_velocity_{_agent}")(_velocity_suite(_agent))


# --------------------------------------------------------------------------- #
# Safe Height: one dimension, the ceiling.
# --------------------------------------------------------------------------- #

# Levels are 1.20 / 1.10 / 1.00 m. Ω extends 0.1 m beyond both, so the uniform
# distribution covers "lower than any level" without reaching the humanoid's
# healthy z-range floor (1.0 m for the torso; the head tip sits ~0.25 m above).
_HEIGHT_LOW = 0.9
_HEIGHT_HIGH = 1.3


@register("safe_height_humanoid")
def _height_humanoid() -> SuiteContexts:
    env_name = "safe_height_humanoid"
    space = ContextSpace((
        Dimension(
            "max_height",
            low=_HEIGHT_LOW,
            high=_HEIGHT_HIGH,
            description="the ceiling height in metres; the head above it is a violation",
        ),
    ))
    return SuiteContexts(
        env_name=env_name,
        space=space,
        level_contexts=_level_contexts(env_name),
        task=TaskDescription(
            agent="A MuJoCo humanoid that has to move forward along the x-axis under a ceiling. Actions are joint "
            "torques in [−1, 1]. An episode ends at the step limit, or earlier when the humanoid stops making forward "
            "progress (less than 0.5 m in 200 steps); it does not end on a fall or on a violation.",
            reward="Per step: 5 × the forward velocity of the centre of mass, plus 5 while the torso is within its healthy "
            "height range, minus 0.1 × ‖action‖². The return is the sum over the episode.",
            cost="Per step, 0.1 × max(0, head_height − max_height) / 0.08: zero while the top of the head is under the "
            "episode's max_height, then growing linearly with how far above it is (a soft hinge, 0.08 m wide). The "
            "episode's cost is the sum; the budget bounds its mean.",
            episode_ends_early=True,
        ),
    )


# --------------------------------------------------------------------------- #
# Safe Push: one dimension, how fast the goal moves.
# --------------------------------------------------------------------------- #

# Levels are 0 / 0.3 / 0.6 m/s. Ω starts at 0 (a stationary goal is a legitimate
# task, level 1) and extends a third beyond the hardest level.
_PUSH_GOAL_VELOCITY_HIGH = 0.8


@register("safe_push_point")
def _push_point() -> SuiteContexts:
    env_name = "safe_push_point"
    space = ContextSpace((
        Dimension(
            "goal_velocity",
            low=0.0,
            high=_PUSH_GOAL_VELOCITY_HIGH,
            description="the speed in m/s at which the goal drifts across the arena, bouncing off its edges; 0 = stationary",
        ),
    ))
    return SuiteContexts(
        env_name=env_name,
        space=space,
        level_contexts=_level_contexts(env_name),
        task=TaskDescription(
            agent="A wheeled point robot in a walled arena that has to push a block into a goal disc, among six "
            "solid cube hazards it can bump into and six flat cube hazards it can drive over. Actions are two "
            "wheel torques in [−1, 1]. When the block reaches the goal, the goal respawns elsewhere and the "
            "episode continues; an episode ends at the step limit, or earlier if the robot flips or leaves its "
            "healthy height range.",
            reward="Per step: how much closer the block got to the goal (1 × the distance decrease), plus 0.1 × how "
            "much closer the robot got to the block, plus 1 each time the block enters a goal. The return is the "
            "sum over the episode; higher means more goals reached with less wandering.",
            cost="Per step, the sum over hazards of a proximity penalty (2 × (1 − distance/size), zero beyond one "
            "hazard size away) for the flat hazards and 3 for each solid hazard the robot is in contact with. The "
            "episode's cost is the sum; the budget bounds its mean.",
            episode_ends_early=True,
        ),
    )


# --------------------------------------------------------------------------- #
# Safe Lift: one 0/1 dimension per foot — which feet must stay off the ground.
# --------------------------------------------------------------------------- #
#
# Ω is the discrete set of foot masks, {0,1}^feet, as integer dimensions; the
# three levels are three particular masks. "Uniform over Ω" therefore draws each
# foot independently with probability ½, which includes the empty mask (no
# constraint) and the full one (every foot restricted, an impossible task for a
# walker). Both are legitimate corners of the space; a distribution that wants
# to avoid them says so, the space does not.


def _lift_suite(env_name: str, agent: str, detection: str) -> Callable[[], SuiteContexts]:
    def factory() -> SuiteContexts:
        from crax import envs

        feet = envs.get_environment(env_name, level=1).unwrapped.foot_names
        space = ContextSpace(tuple(
            Dimension(f"restrict_{foot}", low=0.0, high=1.0, kind="integer",
                      description=f"1 if the {foot.replace('_', ' ')} foot must stay off the ground this episode, else 0")
            for foot in feet
        ))
        return SuiteContexts(
            env_name=env_name,
            space=space,
            level_contexts=_level_contexts(env_name),
            task=TaskDescription(
                agent=f"A MuJoCo {agent} that has to walk forward along the x-axis while keeping the episode's restricted "
                f"feet off the ground. Actions are joint torques in [−1, 1]. An episode ends either at the step limit "
                f"or earlier when the agent falls (its torso leaves the healthy height range); violating the "
                f"constraint does not end the episode.",
                reward=f"The {agent}'s stock locomotion reward per step — the forward velocity vₓ, plus 1 for every step "
                f"it has not fallen, minus 0.5 × ‖action‖² — multiplied by the suite's reward scale of 0.01. The return "
                f"is the sum over the episode.",
                cost=f"Per step, the number of restricted feet touching the ground ({detection}). The episode's cost is "
                f"the sum; the budget bounds its mean.",
                episode_ends_early=True,
            ),
        )

    return factory


register("safe_lift_ant")(_lift_suite("safe_lift_ant", "ant with four legs", "a foot counts as touching when its tip is below 0.15 m"))
register("safe_lift_spider")(_lift_suite("safe_lift_spider", "six-legged spider", "a foot counts as touching when MuJoCo reports it in contact with the floor"))


# --------------------------------------------------------------------------- #
# Safe Pathway: one dimension, the widest gap between consecutive hazards.
# --------------------------------------------------------------------------- #

# Levels are max_gap 6 / 4 / 2 m with min_gap fixed at 1 m; a smaller max_gap
# packs the hazards tighter. Ω extends below the hardest level (down to 1.5 m,
# above min_gap so the draw is well-defined) and above the easiest.
_PATHWAY_MAX_GAP_LOW = 1.5
_PATHWAY_MAX_GAP_HIGH = 8.0


@register("safe_pathway_walker2d")
def _pathway_walker2d() -> SuiteContexts:
    env_name = "safe_pathway_walker2d"
    space = ContextSpace((
        Dimension(
            "max_gap",
            low=_PATHWAY_MAX_GAP_LOW,
            high=_PATHWAY_MAX_GAP_HIGH,
            description="the upper bound in metres of the gap between consecutive hazards along the corridor "
            "(each gap is drawn uniformly from [1, max_gap] when the episode starts); smaller = denser hazards",
        ),
    ))
    return SuiteContexts(
        env_name=env_name,
        space=space,
        level_contexts=_level_contexts(env_name),
        task=TaskDescription(
            agent="A planar MuJoCo walker2d that has to walk forward along a corridor of 100 flat hazards (alternating "
            "discs and squares, 0.25 m / 0.2 m) laid out ahead of it when the episode starts, with random gaps and "
            "lateral jitter. Actions are joint torques in [−1, 1]. An episode ends either at the step limit or "
            "earlier when the walker falls (its torso leaves the healthy height or angle range); violating the "
            "constraint does not end the episode.",
            reward="Per step: the forward velocity vₓ, plus 0.3 while the walker has not fallen, multiplied by the "
            "suite's reward scale of 0.01. The return is the sum over the episode.",
            cost="Per step, 1.5 × the sum over hazards of how deeply a grounded foot is inside it ((1 − distance/size)², "
            "0 outside), plus 5 on the step the walker falls. The episode's cost is the sum; the budget bounds its mean.",
            episode_ends_early=True,
        ),
    )


# --------------------------------------------------------------------------- #
# Safe Goal: how many hazards of each kind are active, and the goal's radius.
# --------------------------------------------------------------------------- #
#
# The environment is the union model of the three levels (crax/envs/hazard_union.py,
# docs/acl/design/hazard_activation.md): every hazard kind any level uses, at its
# maximum count. Ω has one integer dimension per kind — how many are active this
# episode — from 0 to that maximum, plus the goal radius. The levels are three
# points: L1 = (12, 0, 0, 0, 0.20), L2 = (8, 8, 0, 0, 0.18), L3 = (6, 4, 6, 4, 0.16).
# Uniform over Ω therefore includes layouts no level has (many cubes *and* many
# cylinders); that is the point of Ω, and is said in the thesis.

_GOAL_SIZE_LOW = 0.14
_GOAL_SIZE_HIGH = 0.22

_HAZARD_KIND_WORDS = {
    ("cylinder", False): "flat discs the robot can drive over",
    ("cylinder", True): "solid cylinders the robot bumps into",
    ("cube", False): "flat squares the robot can drive over",
    ("cube", True): "solid cubes the robot bumps into",
}


@register("safe_goal_point")
def _goal_point() -> SuiteContexts:
    from crax import envs

    env_name = "safe_goal_point"
    environment = envs.get_environment(env_name, level=1).unwrapped
    dimensions = []
    for group, name in zip(environment.hazard_groups, environment.CONTEXT_PARAMETERS):
        dimensions.append(Dimension(
            name, low=0.0, high=float(group.count), kind="integer",
            description=f"how many of the model's {group.count} {_HAZARD_KIND_WORDS[(group.hazard_type, group.collidable)]} "
            f"are in the arena this episode (the rest are parked out of reach)",
        ))
    dimensions.append(Dimension(
        "goal_size", low=_GOAL_SIZE_LOW, high=_GOAL_SIZE_HIGH,
        description="the radius in metres of the goal discs the robot has to reach; smaller = harder to hit",
    ))
    return SuiteContexts(
        env_name=env_name,
        space=ContextSpace(tuple(dimensions)),
        level_contexts=_level_contexts(env_name),
        task=TaskDescription(
            agent="A wheeled point robot in a 5 m × 5 m walled arena that has to drive to one of two goal discs, "
            "avoiding hazards placed at random when the episode starts: flat discs and squares it can drive over "
            "(a proximity cost) and solid cylinders and cubes it bumps into (a contact cost). Actions are two "
            "wheel torques in [−1, 1]. When the robot reaches a goal, that goal respawns elsewhere and the episode "
            "continues; an episode ends at the step limit, or earlier if the robot flips or leaves its healthy "
            "height range.",
            reward="Per step: 1 each time the robot enters a goal disc (the dense distance term is off). The return "
            "is the number of goals reached in the episode.",
            cost="Per step, the sum over the active hazards of 2 × (1 − distance/size) for each flat hazard the robot "
            "is within one size of, plus 3 for each solid hazard it is in contact with. The episode's cost is the "
            "sum; the budget bounds its mean.",
            episode_ends_early=True,
        ),
    )


# --------------------------------------------------------------------------- #
# Safe Circle: how many hazards are in the arena, and the boundary half-widths.
# --------------------------------------------------------------------------- #

# The levels are (1.125, none), (1.05, 1.05), (0.975, 0.975) around a circle of
# radius 1.5 m, so the robot always has to cross the boundary to follow the circle
# in one direction. Ω reaches a bit below level 3 and, in y, up to the arena
# edge (3 m), which is what "no boundary" encodes to (crax/envs/safe_circle.py).
_CIRCLE_BOUNDARY_LOW = 0.9
_CIRCLE_BOUNDARY_X_HIGH = 1.2
_CIRCLE_BOUNDARY_Y_HIGH = 3.0


@register("safe_circle_point")
def _circle_point() -> SuiteContexts:
    from crax import envs

    env_name = "safe_circle_point"
    environment = envs.get_environment(env_name, level=1).unwrapped
    dimensions = []
    for group in environment.hazard_groups:
        dimensions.append(Dimension(
            group.context_name, low=0.0, high=float(group.count), kind="integer",
            description=f"how many of the model's {group.count} {_HAZARD_KIND_WORDS[(group.hazard_type, group.collidable)]} "
            f"are in the arena this episode (the rest are parked out of reach)",
        ))
    dimensions.append(Dimension(
        "boundary_x", low=_CIRCLE_BOUNDARY_LOW, high=_CIRCLE_BOUNDARY_X_HIGH,
        description="half-width in metres of the allowed strip |x| ≤ boundary_x around the circle's centre; "
        "smaller = the robot has to leave the circle more",
    ))
    dimensions.append(Dimension(
        "boundary_y", low=_CIRCLE_BOUNDARY_LOW, high=_CIRCLE_BOUNDARY_Y_HIGH,
        description="half-height in metres of the allowed strip |y| ≤ boundary_y; 3 m is the arena edge, i.e. no "
        "y-boundary",
    ))
    return SuiteContexts(
        env_name=env_name,
        space=ContextSpace(tuple(dimensions)),
        level_contexts=_level_contexts(env_name),
        task=TaskDescription(
            agent="A wheeled point robot in a 6 m × 6 m arena that has to circle a marked ring of radius 1.5 m as fast "
            "as it can while staying inside a rectangle |x| ≤ boundary_x, |y| ≤ boundary_y that is narrower than the "
            "ring, and clear of flat hazard discs placed at random inside that rectangle when the episode starts. "
            "Actions are thrust and yaw torque in [−1, 1]. An episode ends at the step limit, or earlier if the robot "
            "flips or leaves its healthy height range.",
            reward="Per step: 0.1 × the robot's tangential velocity around the ring's centre divided by (1 + its "
            "distance from the ring), so fast, on the ring and counter-clockwise scores most. The return is the sum "
            "over the episode.",
            cost="Per step: 0.1 if the robot is outside the rectangle, plus the sum over the active hazards of "
            "3 × (1 − distance/size) for each disc the robot is within one size of. The episode's cost is the sum; "
            "the budget bounds its mean.",
            episode_ends_early=True,
        ),
    )


# --------------------------------------------------------------------------- #
# Safe Button: how many hazards of each kind, the gremlins' orbit radius, the layout square.
# --------------------------------------------------------------------------- #

# The levels: 4/8/12 cylinders, 4/6/8 gremlins, orbit radius 0.35/0.35/0.45 m, layout
# half-width 2/2.5/3 m. Ω spans the levels with a little room around the orbit radius.
_GREMLIN_TRAVEL_LOW = 0.3
_GREMLIN_TRAVEL_HIGH = 0.5
_BUTTON_LAYOUT_LOW = 2.0
_BUTTON_LAYOUT_HIGH = 3.0

_BUTTON_HAZARD_WORDS = {
    ("cube", True): "solid blocks (the benchmark's cylinders, boxes under MJX) the robot bumps into",
    ("gremlin", True): "gremlins, small solid boxes that orbit their spot, which the robot bumps into",
}


@register("safe_button_point")
def _button_point() -> SuiteContexts:
    from crax import envs

    env_name = "safe_button_point"
    environment = envs.get_environment(env_name, level=1).unwrapped
    dimensions = []
    for group in environment.hazard_groups:
        dimensions.append(Dimension(
            group.context_name, low=0.0, high=float(group.count), kind="integer",
            description=f"how many of the model's {group.count} {_BUTTON_HAZARD_WORDS[(group.hazard_type, group.collidable)]} "
            f"are in the arena this episode (the rest are parked out of reach)",
        ))
    dimensions.append(Dimension(
        "gremlin_travel", low=_GREMLIN_TRAVEL_LOW, high=_GREMLIN_TRAVEL_HIGH,
        description="the radius in metres of every gremlin's circular orbit; larger = each gremlin sweeps more floor",
    ))
    dimensions.append(Dimension(
        "placement_extent", low=_BUTTON_LAYOUT_LOW, high=_BUTTON_LAYOUT_HIGH,
        description="half-width in metres of the square the buttons and hazards are laid out in; smaller = the same "
        "objects packed closer together",
    ))
    return SuiteContexts(
        env_name=env_name,
        space=ContextSpace(tuple(dimensions)),
        level_contexts=_level_contexts(env_name),
        task=TaskDescription(
            agent="A wheeled point robot on an open floor with four buttons (spheres of radius 0.1 m) laid out at random "
            "when the episode starts, one of which is the goal; solid blocks and orbiting gremlins are laid out with "
            "them. Actions are thrust and yaw torque in [−1, 1]. When the robot presses the goal button another button "
            "becomes the goal and, for 10 steps, the buttons vanish from the lidar and pressing a wrong one is free. "
            "An episode ends at the step limit, or earlier if the robot flips or leaves its healthy height range.",
            reward="Per step: the decrease in distance to the goal button since the last step, plus 1 when it is pressed. "
            "The return over the episode is roughly the number of buttons pressed.",
            cost="Per step: 3 for each active block or gremlin the robot is in contact with, plus 1 for pressing a "
            "button that is not the goal (outside the 10-step grace). The episode's cost is the sum; the budget bounds "
            "its mean.",
            episode_ends_early=True,
        ),
    )


# --------------------------------------------------------------------------- #
# Safe Reach: one dimension, how many of the model's hazards are in the arm's workspace.
# --------------------------------------------------------------------------- #


@register("safe_reacher")
def _reacher() -> SuiteContexts:
    from crax import envs

    env_name = "safe_reacher"
    num_hazards = envs.get_environment(env_name, level=1).unwrapped._num_hazards
    space = ContextSpace((
        Dimension(
            "active_hazards", low=0.0, high=float(num_hazards), kind="integer",
            description=f"how many of the model's {num_hazards} flat hazards (alternating discs and squares) are in the "
            f"arm's workspace this episode; the rest are parked out of reach",
        ),
    ))
    return SuiteContexts(
        env_name=env_name,
        space=space,
        level_contexts=_level_contexts(env_name),
        task=TaskDescription(
            agent="A two-link planar reacher arm (MuJoCo Reacher) that has to bring its fingertip to a target dot, "
            "placed at random within reach when the episode starts, while flat hazards — discs of radius 3.5 cm and "
            "squares of half-width 3 cm — lie at random in and around its workspace. Actions are the two joint "
            "torques in [−1, 1]. An episode ends at the step limit.",
            reward="Per step: 0.1 × (1 − distance/(2 × reach))², i.e. more the closer the fingertip is to the target, "
            "plus 1 while it is within 2 cm of it. The return is the sum over the episode.",
            cost="Per step, the number of active hazards that any point of the arm (five samples per link and the "
            "fingertip) is inside. The episode's cost is the sum; the budget bounds its mean.",
            episode_ends_early=False,
        ),
    )
