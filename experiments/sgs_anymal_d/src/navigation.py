"""Goal-navigation labels and Appendix A.3.1 reward coefficients."""

import math


GOAL_RADIUS = .40
GOAL_HEIGHT_TOLERANCE = .36
BASE_HEIGHT = .60
MIN_CONTACT_FORCE = 15.
BASE_CONTACT_FORCE = 30.
DROP_HEIGHT = -1.05
ACTION_SCALE = .50
SUCCESS_HOLD = 20
CONTROL_DT = .02
MAX_STEPS = 500
WEIGHTS = (50., -50., -.001, -.05, -.25, -.05, .3)
FORWARD_WARMUP_PPO_UPDATES = 500


def reward_weights(iteration):
    return WEIGHTS if iteration < FORWARD_WARMUP_PPO_UPDATES else (*WEIGHTS[:-1], 0.)


def terminal(root_z, base_force, near_steps, steps):
    if root_z < DROP_HEIGHT:
        return "drop"
    if base_force > BASE_CONTACT_FORCE:
        return "base_contact"
    if near_steps >= SUCCESS_HOLD:
        return "success"
    if steps >= MAX_STEPS:
        return "timeout"
    return None


def goal_metrics(root, goal):
    dx, dy = goal[0] - root[0], goal[1] - root[1]
    distance = math.hypot(dx, dy)
    height_error = abs(goal[2] + BASE_HEIGHT - root[2])
    return distance, height_error, distance <= GOAL_RADIUS and height_error <= GOAL_HEIGHT_TOLERANCE


def controlled_cases():
    # The tensorized runtime reads the same scalar constants. These cases exercise
    # the label interpretation and priority independently of physical actuation.
    cases = [
        ("inside", (0.20, 0.10, 0.61), (0.,0.,0.), SUCCESS_HOLD-1, 0., 19, "success"),
        ("outside_xy", (0.41, 0., 0.60), (0.,0.,0.), SUCCESS_HOLD-1, 0., 19, None),
        ("under_goal", (0., 0., 0.22), (0.,0.,0.), SUCCESS_HOLD-1, 0., 19, None),
        ("short_hold", (0., 0., 0.6), (0.,0.,0.), SUCCESS_HOLD-2, 0., 18, None),
        ("base_collision", (0., 0., 0.6), (0.,0.,0.), SUCCESS_HOLD-1, 42., 19, "base_contact"),
        ("drop", (0., 0., -1.1), (0.,0.,0.), SUCCESS_HOLD-1, 0., 19, "drop"),
        ("timeout", (0.41, 0., 0.6), (0.,0.,0.), 4, 0., MAX_STEPS, "timeout"),
    ]
    results = []
    for name, root, target, prior, force, steps, expected in cases:
        dist, hz, close = goal_metrics(root, target)
        observed = terminal(root[2], force, prior + 1 if close else 0, steps)
        assert observed == expected, (name, observed, expected)
        results.append(dict(case=name, root=root, goal=target, distance=dist, height_error=hz, near=close,
                            result=observed, base_force=force))
    return results
