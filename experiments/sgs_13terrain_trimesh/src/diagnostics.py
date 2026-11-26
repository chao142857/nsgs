"""The shared, frozen training-style arrival predicate. Strict routes use separate counters."""


def stable_arrival(near, gravity_z, planar_speed, angular_speed, stance_feet,
                   *, min_stance_feet=2, nonfoot_contact=None):
    ok = near & (gravity_z < -.8660254) & (planar_speed < .45) & (angular_speed < 1.5) & (stance_feet >= min_stance_feet)
    if nonfoot_contact is not None:
        ok = ok & (nonfoot_contact == 0)
    return ok
