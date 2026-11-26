"""Goal-navigation labels, Appendix A.3.1 reward coefficients, and gait recipes."""

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

TERMS = (
    "success", "failure", "work", "posture", "touchdown", "nonfoot_contact", "forward",
    "action_rate", "flat_orientation", "low_height",
)

RECIPES = {
    # Arm 1: Strict Appendix A.3.1 7-term reward structure only (extra 3 weights = 0.0).
    # Fixes: arrival height tolerance (0.15m), zero nonfoot contact + >=3 feet on arrival,
    # decoupled nonfoot_contact (-0.50), speed-capped forward reward, friction=1.0, entropy=0.002.
    "paper7-decoupled": {
        "gamma": 0.997,
        "entropy_coef": 0.002,
        "hygiene": 1,
        "fwd_mode": "first500",
        "fwd_speed_cap": 0.8,
        "goal_height_tolerance": 0.15,
        "min_arrival_stance_feet": 3,
        "require_zero_nonfoot_arrival": True,
        "terrain_friction": 1.0,
        "weights": (50.0, -50.0, -0.00015, -0.012, -0.025, -0.50, 0.06, 0.0, 0.0, 0.0),
    },
    # Arm 2: Paper 7-term + minimal terrain-agnostic locomotion recipe
    # (action_rate_l2, flat_orientation_l2, one-sided low_height_hinge below 0.48m).
    "recipe-smooth-upright": {
        "gamma": 0.997,
        "entropy_coef": 0.002,
        "hygiene": 1,
        "fwd_mode": "first500",
        "fwd_speed_cap": 0.8,
        "goal_height_tolerance": 0.15,
        "min_arrival_stance_feet": 3,
        "require_zero_nonfoot_arrival": True,
        "terrain_friction": 1.0,
        "weights": (50.0, -50.0, -0.00012, -0.010, -0.025, -0.50, 0.06, -0.004, -0.15, -1.5),
    },
    # Arm 3: Same minimal recipe + gamma=0.999 + firmer energy/smoothness regularization
    # to remove time-discount sprint pressure and preserve strong return on 6-8m 13-terrain courses.
    "recipe-smooth-gamma999": {
        "gamma": 0.999,
        "entropy_coef": 0.002,
        "hygiene": 1,
        "fwd_mode": "first500",
        "fwd_speed_cap": 0.8,
        "goal_height_tolerance": 0.15,
        "min_arrival_stance_feet": 3,
        "require_zero_nonfoot_arrival": True,
        "terrain_friction": 1.0,
        "weights": (50.0, -50.0, -0.00020, -0.012, -0.030, -0.50, 0.06, -0.006, -0.15, -1.5),
    },
    "recipe-13t-gamma998": {
        "gamma": 0.998,
        "entropy_coef": 0.002,
        "hygiene": 1,
        "fwd_mode": "first500",
        "fwd_speed_cap": 0.8,
        "goal_height_tolerance": 0.15,
        "min_arrival_stance_feet": 3,
        "require_zero_nonfoot_arrival": True,
        "terrain_friction": 1.0,
        "reg_floor": -0.09,
        "weights": (50.0, -50.0, -0.00016, -0.010, -0.025, -0.50, 0.20, -0.005, -0.15, -1.5),
    },
    "recipe-13t-v3": {
        "gamma": 0.998,
        "entropy_coef": 0.002,
        "hygiene": 1,
        "fwd_mode": "first500",
        "fwd_speed_cap": 0.8,
        "fwd_kind": "potential_progress",
        "goal_height_tolerance": 0.15,
        "min_arrival_stance_feet": 3,
        "require_zero_nonfoot_arrival": True,
        "vertical_foot_contact": True,
        "terrain_friction": 1.0,
        "reg_soft_cap": 0.095,
        "weights": (50.0, -50.0, -0.00016, -0.010, -0.025, -0.50, 0.15, -0.005, -0.15, -1.5),
    },
    # G2 changes both the compression function and all seven original regularization weights.
    # Leaky linear cap: exact 1:1 linear for x <= 0.10, slope 0.10 above 0.10, hard ceiling 0.25.
    # The resulting penalty depends on the observed state/action distribution.
    "recipe-13t-v4-linreg": {
        "gamma": 0.998,
        "entropy_coef": 0.002,
        "hygiene": 1,
        "fwd_mode": "first500",
        "fwd_speed_cap": 0.8,
        "fwd_kind": "potential_progress",
        "goal_height_tolerance": 0.15,
        "min_arrival_stance_feet": 3,
        "require_zero_nonfoot_arrival": True,
        "vertical_foot_contact": True,
        "terrain_friction": 1.0,
        "reg_soft_cap": 0.10,
        "reg_cap_mode": "leaky",
        "reg_leaky_slope": 0.10,
        "reg_hard_ceiling": 0.25,
        "posture_mode": "l1",
        "low_height_thresh": 0.48,
        "weights": (50.0, -50.0, -0.00003, -0.004, -0.010, -0.20, 0.15, -0.003, -0.10, -0.80),
    },
    # G3: Full gait-fix candidate (moderate): leaky linear reg + L2 posture (2x HAA) +
    # Isaac-Lab dof_acc_l2 + ang_vel_xy_l2 + low_height@0.53m + progress-gated feet_air_time + heading align.
    "recipe-13t-v4-gait": {
        "gamma": 0.998,
        "entropy_coef": 0.002,
        "hygiene": 1,
        "fwd_mode": "first500",
        "fwd_speed_cap": 0.8,
        "fwd_kind": "potential_progress",
        "goal_height_tolerance": 0.15,
        "min_arrival_stance_feet": 3,
        "require_zero_nonfoot_arrival": True,
        "vertical_foot_contact": True,
        "terrain_friction": 1.0,
        "reg_soft_cap": 0.10,
        "reg_cap_mode": "leaky",
        "reg_leaky_slope": 0.10,
        "reg_hard_ceiling": 0.25,
        "posture_mode": "l2_haa2",
        "low_height_thresh": 0.53,
        "dof_acc_w": -1.5e-7,
        "ang_vel_xy_w": -0.015,
        "heading_w": -0.010,
        "feet_air_time_w": 0.06,
        "feet_air_time_target": 0.30,
        "weights": (50.0, -50.0, -0.000025, -0.010, -0.010, -0.20, 0.15, -0.004, -0.12, -1.20),
    },
    # G4: firmer multi-weight bracket (ratios relative to G3 vary by term).
    "recipe-13t-v4-gait-firm": {
        "gamma": 0.998,
        "entropy_coef": 0.002,
        "hygiene": 1,
        "fwd_mode": "first500",
        "fwd_speed_cap": 0.8,
        "fwd_kind": "potential_progress",
        "goal_height_tolerance": 0.15,
        "min_arrival_stance_feet": 3,
        "require_zero_nonfoot_arrival": True,
        "vertical_foot_contact": True,
        "terrain_friction": 1.0,
        "reg_soft_cap": 0.10,
        "reg_cap_mode": "leaky",
        "reg_leaky_slope": 0.10,
        "reg_hard_ceiling": 0.25,
        "posture_mode": "l2_haa2",
        "low_height_thresh": 0.53,
        "dof_acc_w": -2.5e-7,
        "ang_vel_xy_w": -0.025,
        "heading_w": -0.015,
        "feet_air_time_w": 0.08,
        "feet_air_time_target": 0.30,
        "weights": (50.0, -50.0, -0.000030, -0.015, -0.012, -0.20, 0.15, -0.006, -0.18, -1.80),
    },
    # v5-upright (V5.3): Zero-offset standard-scale (V in [-0.8, +1.3], matching v3/G3 value_loss ~ 4e-4!)
    # ANYmal-D upright parallel-shank & straight-ahead locomotion recipe:
    # - gamma = 0.998, entropy_coef = 0.002, alive_bonus_w = 0.0, success = +65.0, failure = -65.0:
    #   * Zero critic offset eliminates the 27x terminal TD-jump variance inflation that stalled wave1/wave2!
    #   * High-headroom leaky linear zone m_cap = 0.105 (4.4x above official 10-reset upright walking x ~ 0.024,
    #     so standing & normal walking have exact scale = 1.000 1:1 linear gradients!), hard_ceiling = 0.135
    #     so V_1000(u0) ~ -52 > -65 = R_fail (zero u0 suicide without alive_bonus!)
    #   * Uncompressed quadratic overspeed brake outside reg_cap (-0.50 * (v - 0.78)_+^2) blocks fast squat sprints
    # - support-relative two-sided base_height_l2 around 0.560m (10-reset loaded median = 0.554-0.559m, w = -1.50)
    # - non-redundant phase-separated 3D WORLD-vertical shank constraint (swing deadzone gated on planar_spd>=0.18):
    #   * STANCE (Fz > 15N or stationary): cos_min_stance = 0.6088 (52.5 deg, 10-reset P95 = 52.0 deg), w = 0.80
    #   * SWING  (Fz <=15N & moving):      cos_min_swing  = 0.3746 (68.0 deg, 10-reset P95 = 67.6 deg), w = 0.35
    #   * HAA splay: |dq_HAA| > 0.26 rad (10-reset P95 = 0.258 rad), w = 0.30
    #   * Smooth C1 near-goal 12-joint stance + >=3.5-foot planting ramp inside dist < 0.55m
    # - non-negative feet_air_time bonus calibrated to 10-reset median 0.18s (target = 0.12s, w = +0.05)
    # - zero-arbitrage potential-difference warmup (w_fwd = +0.20, fwd_mode="decay1000" -> 0 after u1000)
    "recipe-13t-v5-upright": {
        "family": "v5_upright",
        "gamma": 0.998,
        "entropy_coef": 0.0006,
        "hygiene": 1,
        "alive_bonus_w": 0.0,
        "fwd_mode": "decay1000",
        "fwd_speed_cap": 0.65,
        "yaw_rate_cap": 1.20,
        "yaw_prog_coef": 0.15,
        "overspeed_thresh": 0.78,
        "overspeed_w": -0.50,
        "step_urgency_w": -0.003,
        "goal_height_tolerance": 0.15,
        "min_arrival_stance_feet": 3,
        "require_zero_nonfoot_arrival": True,
        "vertical_foot_contact": True,
        "terrain_friction": 1.0,
        "reg_soft_cap": 0.095,
        "reg_cap_mode": "leaky",
        "reg_leaky_slope": 0.06,
        "reg_hard_ceiling": 0.115,
        "default_haa_limit_pen_weight": 2.0,
        "default_haa_soft_factor": 0.90,
        "stance_height_min": 0.548,
        "stance_height_max": 0.590,
        "air_crouch_min_height": 0.46,
        "uncomp_height_wall_w": 8.0,
        "uncomp_pose_cap": 0.16,
        "uncomp_action_rate_w": -0.0012,
        "global_posture_l2_w": 0.005,
        "haa_deadzone_rad": 0.24,
        "haa_posture_w": 0.30,
        "shank_world_cos_min_stance": 0.7193,  # cos(44.0 deg) — tight upright shank!
        "shank_world_stance_w": 1.60,
        "shank_world_cos_min_swing": 0.5299,   # cos(58.0 deg)
        "shank_world_swing_w": 0.50,
        "fh_shank_cos_min": -0.05,             # cos(93 deg) — forbids obtuse / \ splay!
        "fh_shank_splay_w": 0.32,
        "near_ramp_outer_m": 0.55,
        "near_ramp_inner_m": 0.25,
        "near_stand_posture_w": 0.12,
        "near_min_feet_w": 0.06,
        "near_vel_damp_w": 0.05,
        "hfe_kfe_soft_limit_w": -0.25,
        "lat_vel_w": -0.18,
        "lat_vel_cap": 0.64,
        "back_vel_w": -0.15,
        "lin_vel_z_w": -0.12,
        "lin_vel_z_cap": 0.50,
        "ang_vel_xy_w": -0.015,
        "ang_vel_xy_cap": 4.0,
        "dof_acc_w": -1.5e-7,
        "dof_acc_cap": 2.0e5,
        "action_rate_cap": 20.0,
        "torque_sq_cap": 25000.0,
        "feet_slide_w": -0.012,
        "feet_slide_cap": 1.5,
        "feet_air_time_w": 0.05,
        "feet_air_time_target": 0.12,
        "weights": (135.0, -135.0, -1.5e-6, -1.0, -0.010, -0.20, 0.25, -0.0028, -0.15, 0.0),
    },
    # V5b (firm): Strictly acute front-hind inter-shank angle (<= 85 deg) + 41.5 deg world-vertical stance shank:
    "recipe-13t-v5-upright-firm": {
        "family": "v5_upright",
        "gamma": 0.998,
        "entropy_coef": 0.0006,
        "hygiene": 1,
        "alive_bonus_w": 0.0,
        "fwd_mode": "decay1000",
        "fwd_speed_cap": 0.65,
        "yaw_rate_cap": 1.20,
        "yaw_prog_coef": 0.15,
        "overspeed_thresh": 0.78,
        "overspeed_w": -0.50,
        "step_urgency_w": -0.003,
        "goal_height_tolerance": 0.15,
        "min_arrival_stance_feet": 3,
        "require_zero_nonfoot_arrival": True,
        "vertical_foot_contact": True,
        "terrain_friction": 1.0,
        "reg_soft_cap": 0.095,
        "reg_cap_mode": "leaky",
        "reg_leaky_slope": 0.06,
        "reg_hard_ceiling": 0.115,
        "default_haa_limit_pen_weight": 2.0,
        "default_haa_soft_factor": 0.90,
        "stance_height_min": 0.552,
        "stance_height_max": 0.590,
        "air_crouch_min_height": 0.46,
        "uncomp_height_wall_w": 10.0,
        "uncomp_pose_cap": 0.16,
        "uncomp_action_rate_w": -0.0012,
        "global_posture_l2_w": 0.005,
        "haa_deadzone_rad": 0.22,
        "haa_posture_w": 0.35,
        "shank_world_cos_min_stance": 0.7490,  # cos(41.5 deg) — near-nominal vertical shank!
        "shank_world_stance_w": 2.00,
        "shank_world_cos_min_swing": 0.5878,   # cos(54.0 deg)
        "shank_world_swing_w": 0.60,
        "fh_shank_cos_min": 0.0872,            # cos(85.0 deg) — strictly acute/parallel front-hind shanks!
        "fh_shank_splay_w": 0.38,
        "near_ramp_outer_m": 0.55,
        "near_ramp_inner_m": 0.25,
        "near_stand_posture_w": 0.15,
        "near_min_feet_w": 0.06,
        "near_vel_damp_w": 0.05,
        "hfe_kfe_soft_limit_w": -0.25,
        "lat_vel_w": -0.22,
        "lat_vel_cap": 0.64,
        "back_vel_w": -0.18,
        "lin_vel_z_w": -0.15,
        "lin_vel_z_cap": 0.50,
        "ang_vel_xy_w": -0.015,
        "ang_vel_xy_cap": 4.0,
        "dof_acc_w": -1.5e-7,
        "dof_acc_cap": 2.0e5,
        "action_rate_cap": 20.0,
        "torque_sq_cap": 25000.0,
        "feet_slide_w": -0.015,
        "feet_slide_cap": 1.5,
        "feet_air_time_w": 0.05,
        "feet_air_time_target": 0.12,
        "weights": (135.0, -135.0, -1.5e-6, -1.0, -0.010, -0.20, 0.26, -0.0028, -0.20, 0.0),
    },
}

# Clean nested single-variable ablation arms derived strictly from recipe-13t-v5-upright:
# A1: ONLY fix raycast miss (+inf) sign from -1.0 (wall alias) to +1.0 (deep drop); keep 1.6x1.0 scan (obs_dim=235)
RECIPES["recipe-13t-v5a-voidpos"] = {
    **RECIPES["recipe-13t-v5-upright"],
    "void_mode": "drop_pos",
    "scan_offset_x": 0.0,
    "scan_size": (1.6, 1.0),
}
# A2: A1 + ONLY expand forward scan to [-0.75, +1.45]m x [-0.70, +0.70]m (23x15 = 345 rays, obs_dim=393)
RECIPES["recipe-13t-v5a-voidpos-wide"] = {
    **RECIPES["recipe-13t-v5-upright"],
    "void_mode": "drop_pos",
    "scan_offset_x": 0.35,
    "scan_size": (2.2, 1.4),
}
# A3: A2 + ONLY append explicit binary ray-validity mask isfinite(scan) (obs_dim=738)
RECIPES["recipe-13t-v5a-wide-validmask"] = {
    **RECIPES["recipe-13t-v5-upright"],
    "void_mode": "drop_pos_with_mask",
    "scan_offset_x": 0.35,
    "scan_size": (2.2, 1.4),
}
# A4: A2 + ONLY relax dynamic speed & vertical-velocity soft penalties (obs_dim=393, same init weights as A2)
RECIPES["recipe-13t-v5a-wide-relaxdyn"] = {
    **RECIPES["recipe-13t-v5-upright"],
    "void_mode": "drop_pos",
    "scan_offset_x": 0.35,
    "scan_size": (2.2, 1.4),
    "overspeed_thresh": 1.35,
    "lin_vel_z_w": -0.02,
}

# V6 Agile (UWLab-aligned):
# - Asymmetric terminal scale: +150.0 arrival vs -22.0 drop (decoupled from -55.0 base_contact),
#   with step_urgency_w = -0.020 -> 1000-step standstill return ~ -9.5 > -21.6 u0 suicide (zero suicide!),
#   yet crossing break-even drops from 50% down to ~8% initial success!
# - Whole-body gait freedom: ZERO shank-angle / front-hind obtuse / HAA splay / lateral-crab locks;
#   ONLY support-relative torso height floor (0.515m with step-straddle compliance) + flat orientation + nonfoot contact deterrent.
# - Exact telescoping potential-difference navigation shaping with 20% floor after u1000 (zero closed-loop oscillation).
# - Wide forward scan [-0.75, +1.45]m x [-0.70, +0.70]m with +1.0 drop encoding + relaxed 1.45 m/s dynamic speed.
# - Bounded state-dependent heteroscedastic Gaussian policy + Markovian Critic (with time_left_frac, hold_frac, foot_contact).
RECIPES["recipe-13t-v6-agile"] = {
    "family": "v5_upright",
    "gamma": 0.998,
    "entropy_coef": 0.0015,
    "dist_mode": "heteroscedastic",
    "init_std": 0.85,
    "std_bounds": (0.06, 1.35),
    "use_critic_priv": True,
    "use_clipped_value_loss": True,
    "hygiene": 1,
    "alive_bonus_w": 0.0,
    "fwd_mode": "floor_decay1000",
    "fwd_floor_ratio": 0.20,
    "exact_telescoping_fwd": True,
    "fwd_speed_cap": 0.85,
    "yaw_rate_cap": 1.40,
    "yaw_prog_coef": 0.12,
    "overspeed_thresh": 1.45,
    "overspeed_w": -0.20,
    "step_urgency_w": -0.020,
    "goal_height_tolerance": 0.16,
    "min_arrival_stance_feet": 3,
    "require_zero_nonfoot_arrival": True,
    "vertical_foot_contact": True,
    "terrain_friction": 1.0,
    "void_mode": "drop_pos",
    "scan_offset_x": 0.35,
    "scan_size": (2.2, 1.4),
    "reg_soft_cap": 0.110,
    "reg_cap_mode": "leaky",
    "reg_leaky_slope": 0.08,
    "reg_hard_ceiling": 0.145,
    "default_haa_limit_pen_weight": 0.25,
    "default_haa_soft_factor": 0.95,
    "stance_height_min": 0.515,
    "stance_height_max": 0.640,
    "step_straddle_compliance": True,
    "air_crouch_min_height": 0.40,
    "uncomp_height_wall_w": 5.0,
    "uncomp_pose_cap": 0.14,
    "uncomp_action_rate_w": -0.0008,
    "global_posture_l2_w": 0.003,
    "haa_deadzone_rad": 0.50,
    "haa_posture_w": 0.0,
    "shank_world_cos_min_stance": 0.0,
    "shank_world_stance_w": 0.0,
    "shank_world_cos_min_swing": 0.0,
    "shank_world_swing_w": 0.0,
    "fh_shank_cos_min": -1.0,
    "fh_shank_splay_w": 0.0,
    "near_ramp_outer_m": 0.50,
    "near_ramp_inner_m": 0.22,
    "near_stand_posture_w": 0.08,
    "near_min_feet_w": 0.05,
    "near_vel_damp_w": 0.04,
    "hfe_kfe_soft_limit_w": -0.15,
    "lat_vel_w": 0.0,
    "lat_vel_cap": 0.64,
    "back_vel_w": -0.04,
    "lin_vel_z_w": -0.015,
    "lin_vel_z_cap": 0.80,
    "ang_vel_xy_w": -0.012,
    "ang_vel_xy_cap": 4.0,
    "dof_acc_w": -1.0e-7,
    "dof_acc_cap": 2.5e5,
    "action_rate_cap": 25.0,
    "torque_sq_cap": 30000.0,
    "feet_slide_w": -0.008,
    "feet_slide_cap": 1.5,
    "feet_air_time_w": 0.04,
    "feet_air_time_target": 0.12,
    "base_contact_weight": -55.0,
    "weights": (150.0, -22.0, -1.0e-6, -1.0, -0.008, -0.25, 0.30, -0.0020, -0.14, 0.0),
}


def get_recipe(name):
    if name is None:
        return None
    if name not in RECIPES:
        raise ValueError(f"Unknown recipe: {name}")
    return RECIPES[name]


def reward_weights(iteration, forward_weight=None, *, step_scale=1., fwd_mode="first500", recipe=None):
    if recipe is not None:
        cfg = get_recipe(recipe)
        base = list(cfg["weights"])
        mode = fwd_mode if fwd_mode is not None else cfg["fwd_mode"]
        if forward_weight is not None:
            base[6] = forward_weight
        elif mode == "decay1000":
            if iteration < 500:
                scale_f = 1.0
            elif iteration < 1000:
                scale_f = (1000 - iteration) / 500.0
            else:
                scale_f = 0.0
            base[6] = base[6] * scale_f
        elif mode == "floor_decay1000":
            floor_r = float(cfg.get("fwd_floor_ratio", 0.25))
            if iteration < 500:
                scale_f = 1.0
            elif iteration < 1000:
                t = (iteration - 500) / 500.0
                scale_f = 1.0 - (1.0 - floor_r) * t
            else:
                scale_f = floor_r
            base[6] = base[6] * scale_f
        elif mode == "matched_decay1000":
            # Total integrated shaping weight = (500 + 250.5) * (1000/1501) * 0.15 = 75.000000 exact == first500 (500 * 0.15)!
            if iteration < 500:
                scale_f = 1000.0 / 1501.0
            elif iteration < 1000:
                scale_f = (1000.0 / 1501.0) * (1000 - iteration) / 500.0
            else:
                scale_f = 0.0
            base[6] = base[6] * scale_f
        else:
            warmup = iteration < FORWARD_WARMUP_PPO_UPDATES
            enabled = {
                "first500": warmup, "after500": not warmup, "always": True, "never": False,
            }[mode]
            if not enabled:
                base[6] = 0.0
        return tuple(base)
    if forward_weight is None:
        warmup = iteration < FORWARD_WARMUP_PPO_UPDATES
        enabled = {
            "first500": warmup, "after500": not warmup, "always": True, "never": False,
        }[fwd_mode]
        forward_weight = WEIGHTS[-1] if enabled else 0.
    return (*WEIGHTS[:2], *(step_scale * weight for weight in (*WEIGHTS[2:-1], forward_weight)), 0., 0., 0.)


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


def goal_metrics(root, goal, *, height_tolerance=GOAL_HEIGHT_TOLERANCE):
    dx, dy = goal[0] - root[0], goal[1] - root[1]
    distance = math.hypot(dx, dy)
    height_error = abs(goal[2] + BASE_HEIGHT - root[2])
    return distance, height_error, distance <= GOAL_RADIUS and height_error <= height_tolerance
