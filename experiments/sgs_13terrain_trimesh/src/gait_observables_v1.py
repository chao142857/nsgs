"""Descriptive strict-route observables & sequential obstacle-traversal predicates.
CPU only; never used to score or train a policy during PPO rollouts."""
import math
import numpy as np
from scipy.spatial.transform import Rotation


SCHEMA = "gait-observables-20261124.3"
FOOT_NAMES = ("LF_FOOT", "LH_FOOT", "RF_FOOT", "RH_FOOT")
SHANK_NAMES = ("LF_SHANK", "LH_SHANK", "RF_SHANK", "RH_SHANK")
REWARD_STANCE_COS_MIN = 0.6088  # cos(52.5 deg)
REWARD_SWING_COS_MIN = 0.3746   # cos(68.0 deg)


def mesh_clearance(xyz, vertices, triangles):
    """Vertical distance to highest triangle at or below each base XY. No hit -> NaN."""
    xyz = np.asarray(xyz, dtype=np.float64)
    out = np.full(len(xyz), np.nan)
    tri = np.asarray(vertices, dtype=np.float64)[np.asarray(triangles, dtype=np.int64).reshape(-1, 3)]
    p, e, f = tri[:, 0], tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    det = e[:, 0] * f[:, 1] - e[:, 1] * f[:, 0]
    flat = abs(det) > 1e-12  # vertical walls are not downward-ray support
    mn, mx = tri[:, :, :2].min(1), tri[:, :, :2].max(1)
    cells, indexes = np.unique(np.floor(xyz[:, :2]).astype(int), axis=0, return_inverse=True)
    for k, cell in enumerate(cells):
        rows = np.flatnonzero(indexes == k)
        use = flat & (mx >= cell - 1e-7).all(1) & (mn <= cell + 1 + 1e-7).all(1)
        if not use.any():
            continue
        q, a, b, d = p[use], e[use], f[use], det[use]
        for begin in range(0, len(rows), 32):
            ix = rows[begin:begin + 32]
            dx = xyz[ix, 0, None] - q[:, 0]
            dy = xyz[ix, 1, None] - q[:, 1]
            u = (dx * b[:, 1] - dy * b[:, 0]) / d
            v = (a[:, 0] * dy - a[:, 1] * dx) / d
            z = q[:, 2] + u * a[:, 2] + v * b[:, 2]
            dz = xyz[ix, 2, None] - z
            inside = (u >= -1e-7) & (v >= -1e-7) & (u + v <= 1 + 1e-7) & (dz >= -1e-7)
            closest = np.where(inside, dz, np.inf).min(1)
            out[ix] = np.where(np.isfinite(closest), np.maximum(closest, 0), np.nan)
    return out


def med(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if len(values) else None


def qtile(values, q):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.quantile(values, q)) if len(values) else None


def verify_obstacle_traversal(terrain_name, start_xyz, goal_xyz, base_xyz, foot_xyz, foot_loaded):
    """Verify sequential start_support -> mid_obstacle_support -> goal_support progression.

    base_xyz:    (M, 3) world/tile-local base positions on uniform post-reset ticks
    foot_xyz:    (M, 4, 3) foot positions on the same ticks
    foot_loaded: (M, 4) bool array (Fz > 15 N) on the same ticks
    """
    if len(base_xyz) == 0:
        return dict(traversed=False, t_start=None, t_mid=None, t_goal=None)
    sx, sy, sz = float(start_xyz[0]), float(start_xyz[1]), float(start_xyz[2])
    gx, gy, gz = float(goal_xyz[0]), float(goal_xyz[1]), float(goal_xyz[2])
    bx, by, bz = base_xyz[:, 0], base_xyz[:, 1], base_xyz[:, 2]
    fx, fy, fz = foot_xyz[:, :, 0], foot_xyz[:, :, 1], foot_xyz[:, :, 2]
    any_load = foot_loaded.any(axis=1)
    two_load = foot_loaded.sum(axis=1) >= 2

    start_mask = (np.hypot(bx - sx, by - sy) <= 0.90) & any_load
    goal_mask = (np.hypot(bx - gx, by - gy) <= 0.55) & two_load

    if terrain_name == "radial_beam":
        fr = np.hypot(fx, fy)
        mid_mask = ((fr >= 1.35) & (fr <= 3.45) & foot_loaded).any(axis=1)
    elif terrain_name == "balance_beam":
        mid_mask = ((fx >= -0.45) & (fx <= 3.55) & (np.abs(fy) <= 0.50) & foot_loaded).any(axis=1)
    elif terrain_name == "stairs":
        z_lo, z_hi = min(sz, gz) + 0.14, max(sz, gz) - 0.14
        mid_mask = ((fz >= z_lo) & (fz <= z_hi) & foot_loaded).any(axis=1) if z_hi > z_lo else any_load
    elif terrain_name == "inverted_slope":
        z_lo, z_hi = min(sz, gz) + 0.07, max(sz, gz) - 0.07
        f_box = np.maximum(np.abs(fx), np.abs(fy))
        mid_mask = ((fz >= z_lo) & (fz <= z_hi) & (f_box >= 1.00) & (f_box <= 3.65) & foot_loaded).any(axis=1)
    elif terrain_name == "pit":
        in_pit = ((fz < -0.10) & foot_loaded).any(axis=1)
        on_rim = ((fz > -0.04) & (np.maximum(np.abs(fx), np.abs(fy)) > 1.75) & foot_loaded).any(axis=1)
        mid_mask = on_rim if sz < -0.08 else in_pit
    elif terrain_name == "gap":
        # Crossing the moat requires loaded support on island (max(|x|,|y|)<1.35) and outer (>1.65)
        on_isl = ((np.maximum(np.abs(fx), np.abs(fy)) < 1.32) & foot_loaded).sum(axis=1) >= 2
        on_out = ((np.maximum(np.abs(fx), np.abs(fy)) > 1.65) & foot_loaded).sum(axis=1) >= 2
        mid_mask = on_out if max(abs(sx), abs(sy)) < 1.4 else on_isl
    elif terrain_name == "maze":
        rx = np.clip(np.floor((bx + 4.8) / 2.4).astype(int), 0, 3)
        ry = np.clip(np.floor((by + 4.8) / 2.4).astype(int), 0, 3)
        rooms = rx * 4 + ry
        sr = int(np.clip(math.floor((sx + 4.8) / 2.4), 0, 3)) * 4 + int(np.clip(math.floor((sy + 4.8) / 2.4), 0, 3))
        gr = int(np.clip(math.floor((gx + 4.8) / 2.4), 0, 3)) * 4 + int(np.clip(math.floor((gy + 4.8) / 2.4), 0, 3))
        mid_mask = (rooms != sr) & (rooms != gr) & any_load
    else:
        vx, vy = gx - sx, gy - sy
        vv = max(vx * vx + vy * vy, 1e-6)
        prog = ((bx - sx) * vx + (by - sy) * vy) / vv
        mid_mask = (prog >= 0.35) & (prog <= 0.65) & any_load

    ts = np.flatnonzero(start_mask)
    if len(ts) == 0:
        return dict(traversed=False, t_start=None, t_mid=None, t_goal=None)
    t0 = int(ts[0])
    tm = np.flatnonzero(mid_mask & (np.arange(len(base_xyz)) >= t0))
    if len(tm) == 0:
        return dict(traversed=False, t_start=t0, t_mid=None, t_goal=None)
    t1 = int(tm[0])
    tg = np.flatnonzero(goal_mask & (np.arange(len(base_xyz)) >= t1))
    if len(tg) == 0:
        return dict(traversed=False, t_start=t0, t_mid=t1, t_goal=None)
    return dict(traversed=True, t_start=t0, t_mid=t1, t_goal=int(tg[0]))


def route_metrics(poses, ticks, force_z, foot_ids, foot_names, end_tick, success, vertices, faces,
                  shank_ids=None, shank_names=None, terrain_name=None, start_xyz=None, goal_xyz=None):
    """One mask for moving/pre-hold data. Frame 0 = reset; force row 0 = tick 1."""
    ticks = np.asarray(ticks)
    force_z = np.asarray(force_z)
    if len(force_z) != end_tick or force_z.shape[1] != 4:
        raise ValueError("Force trace and episode clock do not agree")
    if set(foot_names) != set(FOOT_NAMES) or len(set(foot_ids)) != 4:
        raise ValueError("Expected four named, distinct feet")
    even = (ticks % 2 == 0) & (ticks < end_tick)
    t = ticks[even]
    if len(t) == 0 or t[0] != 0 or not np.all(np.diff(t) == 2):
        raise ValueError("25 Hz pose grid has a missing or duplicated tick")
    a = np.asarray(poses[even], dtype=float)
    base = a[:, 0, :3]
    rot = Rotation.from_quat(a[:, 0, [4, 5, 6, 3]]).as_matrix()
    ord_foot_cols = [list(foot_names).index(fn) for fn in FOOT_NAMES]
    ord_foot_ids = [foot_ids[c] for c in ord_foot_cols]
    rel = np.einsum("tij,tki->tkj", rot, a[:, foot_ids, :3] - base[:, None])
    vel_xy = np.diff(base[:, :2], axis=0) / .04
    speed = np.linalg.norm(vel_xy, axis=1)
    cutoff = end_tick - 150 if success else end_tick
    move = (speed > .25) & (t[1:] <= cutoff)
    accel = np.diff(rel, n=2, axis=0) / .04**2
    chosen = accel[move[:-1] & move[1:]]
    r = rel[1:][move]
    clear = mesh_clearance(base[1:][move], vertices, faces)

    # Support-relative loaded height (exact counterpart of reward h_stance)
    post_ticks = t[1:]
    fc_even = (force_z[post_ticks - 1][:, ord_foot_cols] > 15.0)
    foot_w_post = a[1:][:, ord_foot_ids, :3]
    fc_mov = fc_even[move]
    fw_mov = foot_w_post[move, :, 2]
    bw_mov = base[1:][move, 2]
    cnt_mov = fc_mov.sum(axis=1)
    h_stance_arr = np.where(
        cnt_mov > 0,
        bw_mov - (fw_mov * fc_mov).sum(axis=1) / np.maximum(cnt_mov, 1),
        np.nan,
    )

    if move.any():
        fwd_xy = rot[:-1, :2, 0][move]
        fwd_xy = fwd_xy / np.maximum(np.linalg.norm(fwd_xy, axis=1, keepdims=True), 1e-6)
        vel_u = vel_xy[move] / np.maximum(speed[move, None], 1e-6)
        crab_deg = med(np.degrees(np.arccos(np.clip((fwd_xy * vel_u).sum(1), -1.0, 1.0))))
    else:
        crab_deg = None

    # Both WORLD-vertical (primary, matching reward!) and BODY-vertical shank angles, split by stance/swing
    shank_stats = dict(
        shank_vert_world_deg_med=None, shank_vert_world_deg_p95=None, shank_vert_world_max_leg_med=None,
        shank_vert_world_stance_deg_med=None, shank_vert_world_stance_deg_p95=None,
        shank_vert_world_stance_max_leg_med=None, shank_world_stance_above_deadzone_pct=None,
        shank_vert_world_swing_deg_med=None, shank_vert_world_swing_deg_p95=None,
        shank_world_swing_above_deadzone_pct=None,
        shank_vert_body_deg_med=None, shank_vert_body_stance_deg_med=None, shank_vert_body_swing_deg_med=None,
        shank_pairwise_spread_deg_med=None, inter_shank_deg_med=None,
    )
    if shank_ids is not None and move.any():
        s_names = list(shank_names) if shank_names is not None else list(SHANK_NAMES)
        ord_s = [shank_ids[s_names.index(sn)] for sn in SHANK_NAMES]
        v_sh_w = a[1:][move][:, ord_foot_ids, :3] - a[1:][move][:, ord_s, :3]
        v_sh_b = np.einsum("tij,tki->tkj", rot[1:][move], v_sh_w)
        u_w = v_sh_w / np.maximum(np.linalg.norm(v_sh_w, axis=-1, keepdims=True), 1e-6)
        u_b = v_sh_b / np.maximum(np.linalg.norm(v_sh_b, axis=-1, keepdims=True), 1e-6)
        cos_w = np.clip(-u_w[..., 2], -1.0, 1.0)
        cos_b = np.clip(-u_b[..., 2], -1.0, 1.0)
        deg_w = np.degrees(np.arccos(cos_w))
        deg_b = np.degrees(np.arccos(cos_b))
        leg_w_meds = [med(deg_w[:, j]) for j in range(4)]
        leg_w_st_meds = [med(deg_w[:, j][fc_mov[:, j]]) for j in range(4)]
        st_vals_w = deg_w[fc_mov]
        sw_vals_w = deg_w[~fc_mov]
        # Pairwise sagittal pitch spread across all 4 individual legs in body sagittal plane
        sag_pitch_deg = np.degrees(np.arctan2(v_sh_b[..., 0], np.maximum(-v_sh_b[..., 2], 1e-6)))
        pairwise_spread = sag_pitch_deg.max(axis=1) - sag_pitch_deg.min(axis=1)
        fv = 0.5 * (v_sh_b[:, 0, [0, 2]] + v_sh_b[:, 2, [0, 2]])
        hv = 0.5 * (v_sh_b[:, 1, [0, 2]] + v_sh_b[:, 3, [0, 2]])
        fv /= np.maximum(np.linalg.norm(fv, axis=-1, keepdims=True), 1e-6)
        hv /= np.maximum(np.linalg.norm(hv, axis=-1, keepdims=True), 1e-6)
        inter_deg = np.degrees(np.arccos(np.clip((fv * hv).sum(-1), -1.0, 1.0)))
        shank_stats.update(
            shank_vert_world_deg_med=med(deg_w),
            shank_vert_world_deg_p95=qtile(deg_w, 0.95),
            shank_vert_world_max_leg_med=max((x for x in leg_w_meds if x is not None), default=None),
            shank_vert_world_stance_deg_med=med(st_vals_w),
            shank_vert_world_stance_deg_p95=qtile(st_vals_w, 0.95),
            shank_vert_world_stance_max_leg_med=max((x for x in leg_w_st_meds if x is not None), default=None),
            shank_world_stance_above_deadzone_pct=(float(100.0 * np.mean(cos_w[fc_mov] < REWARD_STANCE_COS_MIN)) if len(st_vals_w) else None),
            shank_vert_world_swing_deg_med=med(sw_vals_w),
            shank_vert_world_swing_deg_p95=qtile(sw_vals_w, 0.95),
            shank_world_swing_above_deadzone_pct=(float(100.0 * np.mean(cos_w[~fc_mov] < REWARD_SWING_COS_MIN)) if len(sw_vals_w) else None),
            shank_vert_body_deg_med=med(deg_b),
            shank_vert_body_stance_deg_med=med(deg_b[fc_mov]),
            shank_vert_body_swing_deg_med=med(deg_b[~fc_mov]),
            shank_pairwise_spread_deg_med=med(pairwise_spread),
            inter_shank_deg_med=med(inter_deg),
        )

    trav = None
    if terrain_name is not None and start_xyz is not None and goal_xyz is not None and len(post_ticks) > 0:
        trav = verify_obstacle_traversal(terrain_name, start_xyz, goal_xyz, base[1:], foot_w_post, fc_even)

    min_xy_r = float(np.min(np.hypot(base[:, 0], base[:, 1])))
    end = t[1:][move]
    fi = np.column_stack((end - 2, end - 1)).ravel()
    fc = force_z[fi] > 15
    lf, lh, rf, rh = ord_foot_cols
    diag = (fc.sum(1) == 2) & ((fc[:, lf] & fc[:, rh]) | (fc[:, rf] & fc[:, lh]))
    tail = rel[t > end_tick - 50]
    return dict(
        schema=SCHEMA, moving_intervals=len(end), moving_force_ticks=len(fi),
        moving_speed_m_s=med(speed[move]), crab_deg_med=crab_deg,
        min_xy_radius_m=min_xy_r, obstacle_traversal=trav,
        accel_triplets=len(chosen),
        foot_acc_rms_m_s2=float(np.sqrt(np.sum(chosen**2, axis=-1).mean())) if len(chosen) else None,
        diagonal_only_pct=float(100 * diag.mean()) if len(diag) else None,
        ground_hits=int(np.isfinite(clear).sum()),
        stance_rel_height_m=med(h_stance_arr),
        mesh_clearance_below_base_m=med(clear),
        base_to_mean_foot_depth_body_m=med(-r[:, :, 2].mean(1)),
        hind_width_body_m=med(r[:, lh, 1] - r[:, rh, 1]),
        hind_center_offset_body_m=med((r[:, lh, 1] + r[:, rh, 1]) / 2),
        last1s_even_hind_width_m=med(tail[:, lh, 1] - tail[:, rh, 1]),
        last1s_even_hind_center_m=med((tail[:, lh, 1] + tail[:, rh, 1]) / 2),
        last1s_even_rh_y_m=med(tail[:, rh, 1]),
        **shank_stats,
    )
