"""Full audit evaluator for HAA single-variable experiments (Arm A / Arm B / Baseline).

Uses the exact shared TrimeshBatchedScene.step() (auto-loading haa_target_map from checkpoint).
Reports:
  - 2,042 zero-overlap common panel (equal-terrain mean & raw count, SHA256 verified)
  - 2,080 heldout (per-terrain & aggregate) + heldout LH_FOOT midline-crossing rate + all-4 HAA lower/upper 1deg pin rates
  - 16 strict 3.0s-hold routes:
      * Per-joint (LF/LH/RF/RH_HAA) lower-1deg and upper-1deg hit rates (full ep & last-1s hold)
      * Per-joint longest consecutive pinned ticks (lower & upper, full ep & last-1s hold)
      * q_raw / q_cmd / q_meas (50 Hz traces saved to haa_trace_50hz.npz)
      * LH_FOOT body-frame y_b median & midline-crossing rate (y_b <= 0)
      * Strict uniform 0.04s (even tick % 2 == 0 only, excluding trailing odd terminal frame)
        walking-phase (v > 0.25 m/s) crab_deg, foot_rad_m, foot_acc_rms_m_s2, plus 50 Hz diag_pair_pct
      * prepared/ Metal render bundle for Mac video rendering
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
from time import monotonic
import numpy as np

HERE = Path(__file__).resolve().parent
EXP_ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from trimesh_terrains import TERRAIN_NAMES, terrain_arrays
from eval_common_panel import load_common_panel
from gait_observables_v1 import SCHEMA, route_metrics


def quat_to_rot_np(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
    R[..., 0, 0] = 1 - 2 * (y * y + z * z)
    R[..., 0, 1] = 2 * (x * y - z * w)
    R[..., 0, 2] = 2 * (x * z + y * w)
    R[..., 1, 0] = 2 * (x * y + z * w)
    R[..., 1, 1] = 1 - 2 * (x * x + z * z)
    R[..., 1, 2] = 2 * (y * z - x * w)
    R[..., 2, 0] = 2 * (x * z - y * w)
    R[..., 2, 1] = 2 * (y * z + x * w)
    R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def max_consecutive_true(arr: np.ndarray) -> int:
    if len(arr) == 0 or not np.any(arr):
        return 0
    best = cur = 0
    for v in arr:
        if v:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    return int(best)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--catalog", type=Path, default=EXP_ROOT / "catalog_v2")
    p.add_argument("--panel-mode", choices=["common_manifest", "catalog_holdout"], default="common_manifest",
                   help="Explicit panel mode: common_manifest hard-verifies panel-json SHA; catalog_holdout scores native holdout.")
    p.add_argument("--panel-json", type=Path, default=EXP_ROOT / "common_panel_v1_v2_2042.json")
    p.add_argument("--allow-cross-catalog", action="store_true",
                   help="Allow evaluating a checkpoint on a different catalog holdout than it was trained with.")
    p.add_argument("--haa-target-map", choices=["auto", "none", "soft_tanh"], default="auto")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = monotonic()

    from isaaclab.app import AppLauncher
    launcher = AppLauncher(headless=True)
    _app = launcher.app

    import torch
    import navigation as nav
    from ppo_smoke import make_algorithm, obs, fingerprint
    from trimesh_scene import TrimeshBatchedScene

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    init = ck["initial"]
    rcfg_init = nav.get_recipe(init["recipe"]) if init.get("recipe") in nav.RECIPES else {}
    haa_map = init.get("haa_target_map", "none") if args.haa_target_map == "auto" else args.haa_target_map
    pen_w = float(init.get("haa_limit_pen_weight", 0.0))
    soft_fac = float(init.get("haa_soft_factor", 0.95))
    scan_offset_x = float(init.get("scan_offset_x", rcfg_init.get("scan_offset_x", 0.0)))
    scan_size_raw = init.get("scan_size", rcfg_init.get("scan_size", (1.6, 1.0)))
    scan_size = (float(scan_size_raw[0]), float(scan_size_raw[1]))
    void_mode = str(init.get("void_mode", rcfg_init.get("void_mode", "legacy_neg")))
    dist_mode = str(init.get("dist_mode", rcfg_init.get("dist_mode", "gaussian")))
    init_std = float(init.get("init_std", rcfg_init.get("init_std", 1.0)))
    std_bounds_raw = init.get("std_bounds", rcfg_init.get("std_bounds", (0.06, 1.35)))
    std_bounds = (float(std_bounds_raw[0]), float(std_bounds_raw[1]))
    use_critic_priv = bool(init.get("use_critic_priv", rcfg_init.get("use_critic_priv", False)))
    use_clipped_vl = init.get("use_clipped_value_loss", rcfg_init.get("use_clipped_value_loss", None))

    hold_arr = np.load(args.catalog / "holdout_tasks.npy", allow_pickle=False)
    strict_cfg = json.loads((args.catalog / "strict_routes.json").read_text())
    n_ho = len(hold_arr)
    n_st = len(strict_cfg)
    total_envs = n_ho + n_st

    import hashlib
    hold_sha = hashlib.sha256(np.ascontiguousarray(hold_arr).tobytes()).hexdigest()
    strict_sha = hashlib.sha256((args.catalog / "strict_routes.json").read_bytes()).hexdigest()
    if not args.allow_cross_catalog:
        if "catalog_holdout_sha256" in init and init["catalog_holdout_sha256"] != hold_sha:
            raise ValueError(f"Holdout SHA mismatch vs checkpoint initial ({hold_sha} != {init['catalog_holdout_sha256']}); pass --allow-cross-catalog if intentional.")
        if "catalog_strict_sha256" in init and init["catalog_strict_sha256"] != strict_sha:
            raise ValueError(f"Strict routes SHA mismatch vs checkpoint initial ({strict_sha} != {init['catalog_strict_sha256']}).")

    if args.panel_mode == "common_manifest":
        if not args.panel_json.exists():
            raise FileNotFoundError(f"common_manifest mode requires existing --panel-json: {args.panel_json}")
        common_arr, keep_idx, panel_meta = load_common_panel(EXP_ROOT, args.panel_json)
        actual_common_sha = panel_meta["actual_panel_sha256"]
        if not np.array_equal(hold_arr[keep_idx], common_arr):
            raise ValueError("hold_arr[keep_idx] != common_arr in common_manifest mode!")
    else:
        keep_idx = np.arange(n_ho, dtype=np.int64)
        common_arr = None
        actual_common_sha = None

    strict_rows = np.array([
        [r["tile_idx"], r["start"][0], r["start"][1], r["start"][2], r["start_yaw"],
         r["goal"][0], r["goal"][1], r["goal"][2]]
        for r in strict_cfg
    ], dtype=np.float32)
    eval_bank = np.vstack([hold_arr, strict_rows])

    world = TrimeshBatchedScene(
        args.catalog, total_envs,
        terrain_friction=init["terrain_friction"],
        max_episode_steps=init["max_episode_steps"],
        haa_target_map=haa_map,
        haa_limit_pen_weight=pen_w,
        haa_soft_factor=soft_fac,
        scan_offset_x=scan_offset_x,
        scan_size=scan_size,
        void_mode=void_mode,
    )
    assert world.catalog_meta["collision_sha256"] == init["catalog_collision_sha256"]
    world.banks["eval"] = torch.tensor(eval_bank, device=world.device, dtype=torch.float32)
    world.select_bank("eval")
    world.place(world.env_ids, world.env_ids)

    readings = obs(world, clip_height=bool(init["hygiene"]), include_critic=use_critic_priv)
    obs_dim_eval = int(readings["policy"].shape[1])
    if "obs_dim" in init and int(init["obs_dim"]) != obs_dim_eval:
        raise ValueError(f"Eval obs_dim {obs_dim_eval} != checkpoint initial obs_dim {init['obs_dim']}")
    alg = make_algorithm(
        world, readings, init["updates"], gamma=init["gamma"],
        hygiene=bool(init["hygiene"]), entropy_coef=init["entropy_coef"],
        dist_mode=dist_mode, init_std=init_std, std_bounds=std_bounds,
        use_critic_priv=use_critic_priv, use_clipped_value_loss=use_clipped_vl,
    )
    alg.load(ck["ppo"], load_cfg=None, strict=True)
    alg.eval_mode()
    actor_sha = fingerprint(alg.actor.state_dict())

    deadline = int(init["max_episode_steps"])
    STRICT_HOLD_TICKS = 150
    done_label = np.zeros(total_envs, dtype=np.int8)
    done_tick = np.full(total_envs, deadline, dtype=np.int32)
    min_dist = np.full(total_envs, np.inf, dtype=np.float32)
    max_hold = np.zeros(total_envs, dtype=np.int32)

    st_slices = slice(n_ho, total_envs)
    st_origins = world.origins[st_slices].detach().cpu().numpy()
    st_poses = [[] for _ in range(n_st)]
    st_ticks = [[] for _ in range(n_st)]
    st_holds = [[] for _ in range(n_st)]

    # 50 Hz per-tick strict telemetry
    st_q_raw = [[] for _ in range(n_st)]
    st_q_cmd = [[] for _ in range(n_st)]
    st_q_meas = [[] for _ in range(n_st)]
    st_lo1 = [[] for _ in range(n_st)]
    st_hi1 = [[] for _ in range(n_st)]
    st_soft = [[] for _ in range(n_st)]
    st_fz = [[] for _ in range(n_st)]
    st_lh_yb = [[] for _ in range(n_st)]

    # Held-out aggregate telemetry (separate lo/hi + LH_FOOT midline crossing + max consecutive per env)
    ho_act_ticks = 0
    ho_lo1_ticks = np.zeros(4, dtype=np.int64)
    ho_hi1_ticks = np.zeros(4, dtype=np.int64)
    ho_near_ticks = 0
    ho_near_lo1_ticks = np.zeros(4, dtype=np.int64)
    ho_near_hi1_ticks = np.zeros(4, dtype=np.int64)
    ho_near_lh_cross_ticks = 0
    ho_cur_lo_run = np.zeros((n_ho, 4), dtype=np.int32)
    ho_max_lo_run = np.zeros((n_ho, 4), dtype=np.int32)
    ho_cur_hi_run = np.zeros((n_ho, 4), dtype=np.int32)
    ho_max_hi_run = np.zeros((n_ho, 4), dtype=np.int32)
    ho_raw_reg_sum = 0.0
    ho_reg_scale_sum = 0.0
    ho_exc_sum = 0.0
    # Separate signed, post-compression components. Each is accumulated over the SAME active rows.
    ho_env_sum = ho_main_sum = ho_extra_sum = ho_air_sum = ho_alive_sum = ho_limit_sum = 0.0
    # Per-route last-tick LH_FOOT y_b for held-out routes that succeed
    ho_end_lh_yb = np.full(n_ho, np.nan, dtype=np.float32)
    ho_end_rh_yb = np.full(n_ho, np.nan, dtype=np.float32)
    ho_end_haa_meas_deg = np.full((n_ho, 4), np.nan, dtype=np.float32)

    # Attempt & far-side foot support telemetry across all 2,080 held-out routes
    ho_start_xy_t = world.robot.data.root_pos_w[:n_ho, :2].clone()
    ho_goal_xy_t = world.targets[:n_ho, :2].clone()
    ho_vec_t = ho_goal_xy_t - ho_start_xy_t
    ho_tot_dist_t = ho_vec_t.norm(dim=-1).clamp_min(1e-4)
    ho_unit_dir_t = ho_vec_t / ho_tot_dist_t.unsqueeze(-1)
    ho_max_fwd_proj_t = torch.zeros(n_ho, device=world.device, dtype=torch.float32)
    ho_ever_goal_xy_t = torch.zeros(n_ho, device=world.device, dtype=torch.bool)
    ho_far_foot_loaded_t = torch.zeros(n_ho, device=world.device, dtype=torch.bool)

    RAD2DEG = 180.0 / math.pi
    lh_foot_bi = world.lh_foot_body_idx
    rh_foot_bi = world.foot_bodies[list(world.foot_names).index("RH_FOOT")]

    def foot_yb_np(bp_np: np.ndarray, body_index: int) -> np.ndarray:
        base_p = bp_np[:, 0, :3].astype(np.float64)
        base_R = quat_to_rot_np(bp_np[:, 0, 3:7].astype(np.float64))
        lh_p = bp_np[:, body_index, :3].astype(np.float64)
        rel_b = np.einsum("nij,ni->nj", base_R, lh_p - base_p)
        return rel_b[:, 1]

    def record_strict_pose(tick: int, hc: np.ndarray, active=None):
        bp = world.robot.data.body_pose_w[st_slices].detach().cpu().numpy()
        for i in range(n_st):
            if active is not None and not active[i]:
                continue
            p_loc = bp[i].copy()
            p_loc[:, :3] -= st_origins[i]
            st_poses[i].append(p_loc.astype("<f4"))
            st_ticks[i].append(tick)
            st_holds[i].append(int(hc[i]))

    st_hold_live = np.zeros(n_st, dtype=np.int32)
    record_strict_pose(0, st_hold_live)

    with torch.no_grad():
        for tick in range(1, deadline + STRICT_HOLD_TICKS + 1):
            action = alg.actor(obs(world, clip_height=bool(init["hygiene"])))
            labels, sig = world.step(action, training_iteration=int(ck["update"]),
                                     recipe=init["recipe"], fwd_mode=init["fwd_mode"])
            lbl_np = labels.cpu().numpy()
            dist_np = sig["distance"].cpu().numpy()
            hold_np = world.hold.cpu().numpy()
            stable_np = sig["stable"].cpu().numpy()
            near_np = sig["near"].cpu().numpy()
            q_raw_deg = sig["q_raw_haa"].cpu().numpy() * RAD2DEG
            q_cmd_deg = sig["q_cmd_haa"].cpu().numpy() * RAD2DEG
            q_meas_deg = sig["q_meas_haa"].cpu().numpy() * RAD2DEG
            lo1_np = sig["near_phys_lo_1deg"].cpu().numpy()
            hi1_np = sig["near_phys_hi_1deg"].cpu().numpy()
            soft_np = sig["near_soft"].cpu().numpy()
            fz_np = sig["foot_force_z"].cpu().numpy()

            # Held-out (first 2,080)
            if tick <= deadline:
                act_ho = (done_label[:n_ho] == 0)
                n_act = int(act_ho.sum())
                if n_act > 0:
                    act_ho_t = torch.as_tensor(act_ho, device=world.device, dtype=torch.bool)
                    cur_xy_t = world.robot.data.root_pos_w[:n_ho, :2]
                    fwd_proj_t = ((cur_xy_t - ho_start_xy_t) * ho_unit_dir_t).sum(dim=-1)
                    ho_max_fwd_proj_t = torch.where(act_ho_t, torch.maximum(ho_max_fwd_proj_t, fwd_proj_t), ho_max_fwd_proj_t)
                    ho_ever_goal_xy_t = ho_ever_goal_xy_t | (act_ho_t & (sig["distance"][:n_ho] <= 0.40))
                    foot_xy_t = world.robot.data.body_pos_w[:n_ho, world.foot_bodies, :2]
                    foot_proj_t = ((foot_xy_t - ho_start_xy_t.unsqueeze(1)) * ho_unit_dir_t.unsqueeze(1)).sum(dim=-1)
                    far_hit_t = ((sig["foot_force_z"][:n_ho] > 15.0) & (foot_proj_t >= 0.55 * ho_tot_dist_t.unsqueeze(1))).any(dim=1)
                    ho_far_foot_loaded_t = ho_far_foot_loaded_t | (act_ho_t & far_hit_t)

                    ho_act_ticks += n_act
                    ho_lo1_ticks += lo1_np[:n_ho][act_ho].sum(axis=0)
                    ho_hi1_ticks += hi1_np[:n_ho][act_ho].sum(axis=0)
                    ho_raw_reg_sum += float(sig["raw_reg_x"][:n_ho][act_ho].sum().item())
                    ho_reg_scale_sum += float(sig["reg_scale"][:n_ho][act_ho].sum().item())
                    ho_exc_sum += float(sig["haa_limit_excess"][:n_ho][act_ho].sum().item())
                    def active_sum(key):
                        return float(sig[key][:n_ho][act_ho].sum(dtype=torch.float64).item())
                    ho_env_sum += active_sum("reward")
                    ho_main_sum += active_sum("reward_terms")
                    ho_extra_sum += active_sum("extra_reg_scaled")
                    ho_air_sum += active_sum("air_bonus")
                    ho_alive_sum += active_sum("alive_bonus")
                    ho_limit_sum += active_sum("haa_limit_pen")
                    # Update longest consecutive pin per env & joint
                    act_col = act_ho[:, None]
                    ho_cur_lo_run = np.where(act_col & lo1_np[:n_ho], ho_cur_lo_run + 1, np.where(act_col, 0, ho_cur_lo_run))
                    ho_max_lo_run = np.maximum(ho_max_lo_run, ho_cur_lo_run)
                    ho_cur_hi_run = np.where(act_col & hi1_np[:n_ho], ho_cur_hi_run + 1, np.where(act_col, 0, ho_cur_hi_run))
                    ho_max_hi_run = np.maximum(ho_max_hi_run, ho_cur_hi_run)

                    nm = act_ho & near_np[:n_ho]
                    new_ho = act_ho & (lbl_np[:n_ho] != 0)
                    if np.any(nm) or np.any(new_ho):
                        bp_ho = world.robot.data.body_pose_w[:n_ho].detach().cpu().numpy()
                        yb_ho = foot_yb_np(bp_ho, lh_foot_bi)
                        if np.any(nm):
                            ho_near_ticks += int(nm.sum())
                            ho_near_lo1_ticks += lo1_np[:n_ho][nm].sum(axis=0)
                            ho_near_hi1_ticks += hi1_np[:n_ho][nm].sum(axis=0)
                            ho_near_lh_cross_ticks += int((yb_ho[nm] <= 0.0).sum())
                        if np.any(new_ho):
                            ho_end_lh_yb[new_ho] = yb_ho[new_ho].astype(np.float32)
                            ho_end_rh_yb[new_ho] = foot_yb_np(bp_ho[new_ho], rh_foot_bi)
                            ho_end_haa_meas_deg[new_ho] = q_meas_deg[:n_ho][new_ho].astype(np.float32)
                min_dist[:n_ho][act_ho] = np.minimum(min_dist[:n_ho][act_ho], dist_np[:n_ho][act_ho])
                max_hold[:n_ho][act_ho] = np.maximum(max_hold[:n_ho][act_ho], hold_np[:n_ho][act_ho])
                new_ho = act_ho & (lbl_np[:n_ho] != 0)
                if np.any(new_ho):
                    done_label[:n_ho][new_ho] = lbl_np[:n_ho][new_ho]
                    done_tick[:n_ho][new_ho] = tick
            elif tick == deadline + 1:
                rem_ho = (done_label[:n_ho] == 0)
                if np.any(rem_ho):
                    bp_ho = world.robot.data.body_pose_w[:n_ho].detach().cpu().numpy()
                    yb_ho = foot_yb_np(bp_ho, lh_foot_bi)
                    ho_end_lh_yb[rem_ho] = yb_ho[rem_ho].astype(np.float32)
                    ho_end_rh_yb[rem_ho] = foot_yb_np(bp_ho[rem_ho], rh_foot_bi)
                    ho_end_haa_meas_deg[rem_ho] = q_meas_deg[:n_ho][rem_ho].astype(np.float32)
                    done_label[:n_ho][rem_ho] = 4

            # Strict (last 16)
            act_st = (done_label[n_ho:] == 0)
            if np.any(act_st):
                bp_st = world.robot.data.body_pose_w[st_slices].detach().cpu().numpy()
                yb_st = foot_yb_np(bp_st, lh_foot_bi)
                for i in range(n_st):
                    if act_st[i]:
                        st_q_raw[i].append(q_raw_deg[n_ho + i].astype(np.float32))
                        st_q_cmd[i].append(q_cmd_deg[n_ho + i].astype(np.float32))
                        st_q_meas[i].append(q_meas_deg[n_ho + i].astype(np.float32))
                        st_lo1[i].append(lo1_np[n_ho + i].astype(np.bool_))
                        st_hi1[i].append(hi1_np[n_ho + i].astype(np.bool_))
                        st_soft[i].append(soft_np[n_ho + i].astype(np.bool_))
                        st_fz[i].append(fz_np[n_ho + i].astype(np.float32))
                        st_lh_yb[i].append(float(yb_st[i]))

            st_hold_live = np.where(stable_np[n_ho:], st_hold_live + 1, 0)
            min_dist[n_ho:][act_st] = np.minimum(min_dist[n_ho:][act_st], dist_np[n_ho:][act_st])
            max_hold[n_ho:][act_st] = np.maximum(max_hold[n_ho:][act_st], st_hold_live[act_st])
            st_fail = act_st & ((lbl_np[n_ho:] == 2) | (lbl_np[n_ho:] == 3))
            st_win = act_st & (~st_fail) & (st_hold_live >= STRICT_HOLD_TICKS)
            st_to = act_st & (~st_fail) & (~st_win) & (tick >= deadline) & (st_hold_live == 0)
            if np.any(st_fail):
                done_label[n_ho:][st_fail] = lbl_np[n_ho:][st_fail]
                done_tick[n_ho:][st_fail] = tick
            if np.any(st_win):
                done_label[n_ho:][st_win] = 1
                done_tick[n_ho:][st_win] = tick
            if np.any(st_to):
                done_label[n_ho:][st_to] = 4
                done_tick[n_ho:][st_to] = tick
            newly_st = st_fail | st_win | st_to
            if tick % 2 == 0 or np.any(newly_st):
                record_strict_pose(tick, st_hold_live, act_st)
            if np.all(done_label != 0):
                break

    done_label[done_label == 0] = 4
    actor_sha_after = fingerprint(alg.actor.state_dict())
    assert actor_sha == actor_sha_after, f"Actor mutated: {actor_sha} != {actor_sha_after}"

    ho_max_fwd_proj = ho_max_fwd_proj_t.detach().cpu().numpy()
    ho_ever_goal_xy = ho_ever_goal_xy_t.detach().cpu().numpy()
    ho_far_foot_loaded = ho_far_foot_loaded_t.detach().cpu().numpy()

    label_names = {1: "success", 2: "drop", 3: "base_contact", 4: "timeout"}
    ho_kinds = hold_arr[:, 0].astype(np.int64) // 20
    by_terrain = []
    for k, name in enumerate(TERRAIN_NAMES):
        m = (ho_kinds == k)
        lbls = done_label[:n_ho][m]
        by_terrain.append(dict(
            terrain=name, n=int(m.sum()),
            success=int((lbls == 1).sum()),
            drop=int((lbls == 2).sum()),
            base_contact=int((lbls == 3).sum()),
            timeout=int((lbls == 4).sum()),
            rate=round(float((lbls == 1).mean()), 4),
            attempt_gt_0p5m=int((ho_max_fwd_proj[m] > 0.50).sum()),
            far_foot_loaded=int(ho_far_foot_loaded[m].sum()),
            entered_goal_xy=int(ho_ever_goal_xy[m].sum()),
            med_max_fwd_proj_m=round(float(np.median(ho_max_fwd_proj[m])), 3),
        ))
    ho_succ = int((done_label[:n_ho] == 1).sum())
    ho_succ_mask = (done_label[:n_ho] == 1)
    ho_lh_left_mask = (ho_end_lh_yb > 0.0)
    ho_no_sustained_pin = (np.maximum(ho_max_lo_run, ho_max_hi_run).max(axis=1) < 20)
    ho_succ_and_lh_left = int((ho_succ_mask & ho_lh_left_mask).sum())
    ho_succ_clean = int((ho_succ_mask & ho_lh_left_mask & ho_no_sustained_pin).sum())
    ho_succ_lh_cross_rate = (
        round(float(100.0 * np.mean(ho_end_lh_yb[ho_succ_mask] <= 0.0)), 2)
        if np.any(ho_succ_mask) else None
    )
    ho_succ_lh_yb_med = (
        round(float(np.median(ho_end_lh_yb[ho_succ_mask])), 4)
        if np.any(ho_succ_mask) else None
    )

    c_lbls = done_label[:n_ho][keep_idx]
    c_kinds = ho_kinds[keep_idx]
    c_ter_rates = [float((c_lbls[c_kinds == k] == 1).mean()) for k in range(13)]
    panel_eq_rate = float(np.mean(c_ter_rates))
    panel_raw = int((c_lbls == 1).sum())
    c_succ_left = (ho_succ_mask & ho_lh_left_mask)[keep_idx]
    c_succ_clean = (ho_succ_mask & ho_lh_left_mask & ho_no_sustained_pin)[keep_idx]
    panel_succ_and_lh_left = int(c_succ_left.sum())
    panel_succ_and_lh_left_eq_rate = float(np.mean([float(c_succ_left[c_kinds == k].mean()) for k in range(13)]))
    panel_succ_clean = int(c_succ_clean.sum())
    panel_succ_clean_eq_rate = float(np.mean([float(c_succ_clean[c_kinds == k].mean()) for k in range(13)]))

    is_common2042 = np.zeros(n_ho, dtype=np.bool_)
    if args.panel_mode == "common_manifest":
        is_common2042[keep_idx] = True
        common2042_raw = panel_raw
        common2042_eq_rate = round(panel_eq_rate, 4)
        common2042_succ_and_lh_left = panel_succ_and_lh_left
        common2042_succ_and_lh_left_eq_rate = round(panel_succ_and_lh_left_eq_rate, 4)
        common2042_succ_clean = panel_succ_clean
        common2042_succ_clean_eq_rate = round(panel_succ_clean_eq_rate, 4)
    else:
        common2042_raw = None
        common2042_eq_rate = None
        common2042_succ_and_lh_left = None
        common2042_succ_and_lh_left_eq_rate = None
        common2042_succ_clean = None
        common2042_succ_clean_eq_rate = None

    np.savez_compressed(
        args.out / "heldout_routes.npz",
        source_row=np.arange(n_ho, dtype=np.int64),
        panel_mode=np.array(args.panel_mode),
        is_common2042=is_common2042,
        terrain_id=ho_kinds.astype(np.int32),
        label=done_label[:n_ho].astype(np.int8),
        end_tick=done_tick[:n_ho].astype(np.int32),
        min_dist_m=min_dist[:n_ho].astype(np.float32),
        max_fwd_proj_m=ho_max_fwd_proj.astype(np.float32),
        ever_entered_goal_xy=ho_ever_goal_xy.astype(np.bool_),
        far_foot_loaded=ho_far_foot_loaded.astype(np.bool_),
        max_hold_ticks=max_hold[:n_ho].astype(np.int32),
        end_lh_foot_y_b_m=ho_end_lh_yb.astype(np.float32),
        end_rh_foot_y_b_m=ho_end_rh_yb,
        end_haa_meas_deg=ho_end_haa_meas_deg.astype(np.float32),
        max_consec_lo1=ho_max_lo_run.astype(np.int32),
        max_consec_hi1=ho_max_hi_run.astype(np.int32),
        haa_names=np.array(world.haa_names),
    )

    # Strict routes: exact uniform 0.04s (even ticks only!) walking metrics + per-joint lo/hi/consec/y_b
    prep = args.out / "prepared"
    prep.mkdir(parents=True, exist_ok=True)
    strict_out = []
    haa_report = []
    bank = []
    fn = list(world.foot_names)
    lf_i, lh_i, rf_i, rh_i = [fn.index(x) for x in ("LF_FOOT", "LH_FOOT", "RF_FOOT", "RH_FOOT")]

    for i, r in enumerate(strict_cfg):
        lbl = int(done_label[n_ho + i])
        state = label_names[lbl]
        folder = f"{i:02d}-{r['terrain']}-s{r['seed']}l{r['level']}"
        fdir = prep / folder
        fdir.mkdir(parents=True, exist_ok=True)
        v, f = terrain_arrays(r["terrain"], r["seed"], r["level"], terrain_set=world.terrain_set)
        (fdir / "ground.mesh").write_bytes(v.tobytes() + f.tobytes())
        poses_arr = np.asarray(st_poses[i], dtype="<f4")
        (fdir / "poses.f32").write_bytes(poses_arr.tobytes())

        qr = np.asarray(st_q_raw[i], dtype=np.float32)
        qc = np.asarray(st_q_cmd[i], dtype=np.float32)
        qm = np.asarray(st_q_meas[i], dtype=np.float32)
        lo1 = np.asarray(st_lo1[i], dtype=np.bool_)
        hi1 = np.asarray(st_hi1[i], dtype=np.bool_)
        ls = np.asarray(st_soft[i], dtype=np.bool_)
        fz = np.asarray(st_fz[i], dtype=np.float32)
        lhy = np.asarray(st_lh_yb[i], dtype=np.float32)
        np.savez_compressed(
            fdir / "haa_trace_50hz.npz",
            haa_names=np.array(world.haa_names),
            q_raw_deg=qr, q_cmd_deg=qc, q_meas_deg=qm,
            near_phys_lo_1deg=lo1, near_phys_hi_1deg=hi1,
            near_soft=ls, foot_force_z=fz, lh_foot_y_b_m=lhy,
            foot_names=np.asarray(world.foot_names), foot_body_ids=np.asarray(world.foot_bodies, dtype=np.int32),
        )

        # Filter poses_arr to STRICT uniform 0.04s even ticks (tick % 2 == 0)
        ticks_arr = np.asarray(st_ticks[i], dtype=np.int32)
        # Preserve historical legacy windows (old snapshots omitted the terminal pose).
        even_mask = (ticks_arr % 2 == 0) & (ticks_arr < done_tick[n_ho + i])
        p_even = poses_arr[even_mask]
        dt = 0.04
        if len(p_even) > 4:
            base_p = p_even[:, 0, :3].astype(np.float64)
            base_R = quat_to_rot_np(p_even[:, 0, 3:7].astype(np.float64))
            rel_b = np.einsum("tij,tki->tkj", base_R, p_even[:, :, :3].astype(np.float64) - base_p[:, None, :])
            foot_idx = world.foot_bodies
            vel_xy = np.diff(base_p[:, :2], axis=0) / dt
            spd = np.linalg.norm(vel_xy, axis=1)
            moving = spd > 0.25
            if moving.sum() > 3:
                fwd_xy = base_R[:-1, :2, 0]
                fn_v = fwd_xy[moving] / np.maximum(1e-6, np.linalg.norm(fwd_xy[moving], axis=1, keepdims=True))
                vn = vel_xy[moving] / spd[moving, None]
                crab_deg = float(np.median(np.degrees(np.arccos(np.clip(np.sum(fn_v * vn, axis=1), -1.0, 1.0)))))
                walk_rad = float(np.median(np.linalg.norm(rel_b[:-1][moving][:, foot_idx, :2], axis=2)))
                walk_hbase = float(np.median(-rel_b[:-1][moving][:, foot_idx, 2].mean(axis=1)))
            else:
                crab_deg, walk_rad, walk_hbase = float("nan"), float("nan"), float("nan")
            f_vel = np.diff(rel_b[:, foot_idx], axis=0) / dt
            f_acc = np.diff(f_vel, axis=0) / dt
            mv2 = moving[1:]
            foot_acc_rms = float(np.sqrt(np.mean(np.sum(f_acc[mv2] ** 2, axis=-1)))) if mv2.sum() > 2 else float("nan")
        else:
            crab_deg = walk_rad = walk_hbase = foot_acc_rms = float("nan")

        T50 = len(fz)
        walk_end = max(1, T50 - 150) if state == "success" else T50
        fc = fz[:walk_end] > 15.0
        d1 = fc[:, lf_i] & fc[:, rh_i] & (~fc[:, rf_i]) & (~fc[:, lh_i])
        d2 = fc[:, rf_i] & fc[:, lh_i] & (~fc[:, lf_i]) & (~fc[:, rh_i])
        diag_frac = float(np.mean(d1 | d2)) if walk_end > 0 else 0.0

        last50 = slice(max(0, T50 - 50), T50)
        per_joint_hold1s = {}
        per_joint_ep = {}
        for j, nm in enumerate(world.haa_names):
            per_joint_hold1s[nm] = dict(
                raw_med_deg=round(float(np.median(qr[last50, j])), 2) if T50 else None,
                cmd_med_deg=round(float(np.median(qc[last50, j])), 2) if T50 else None,
                meas_med_deg=round(float(np.median(qm[last50, j])), 2) if T50 else None,
                lo1_pct=round(float(100.0 * np.mean(lo1[last50, j])), 1) if T50 else 0.0,
                hi1_pct=round(float(100.0 * np.mean(hi1[last50, j])), 1) if T50 else 0.0,
                max_consec_lo1=max_consecutive_true(lo1[last50, j]) if T50 else 0,
                max_consec_hi1=max_consecutive_true(hi1[last50, j]) if T50 else 0,
            )
            per_joint_ep[nm] = dict(
                lo1_pct=round(float(100.0 * np.mean(lo1[:, j])), 2) if T50 else 0.0,
                hi1_pct=round(float(100.0 * np.mean(hi1[:, j])), 2) if T50 else 0.0,
                max_consec_lo1=max_consecutive_true(lo1[:, j]) if T50 else 0,
                max_consec_hi1=max_consecutive_true(hi1[:, j]) if T50 else 0,
            )

        lh_yb_hold = lhy[last50] if T50 else np.array([], dtype=np.float32)
        hrec = dict(
            folder=folder, terrain=r["terrain"], result=state,
            end_tick=int(done_tick[n_ho + i]), max_hold_ticks=int(max_hold[n_ho + i]),
            lh_foot_hold1s_yb_med_m=round(float(np.median(lh_yb_hold)), 4) if len(lh_yb_hold) else None,
            lh_foot_hold1s_cross_pct=round(float(100.0 * np.mean(lh_yb_hold <= 0.0)), 1) if len(lh_yb_hold) else None,
            any_haa_hold1s_max_consec_pin=max(
                max(v["max_consec_lo1"], v["max_consec_hi1"]) for v in per_joint_hold1s.values()
            ),
            per_joint_hold1s=per_joint_hold1s,
            per_joint_ep=per_joint_ep,
            walk_even_0p04s_crab_deg=round(crab_deg, 1) if np.isfinite(crab_deg) else None,
            walk_even_0p04s_foot_rad_m=round(walk_rad, 3) if np.isfinite(walk_rad) else None,
            walk_even_0p04s_h_base_m=round(walk_hbase, 3) if np.isfinite(walk_hbase) else None,
            # Compatibility alias above is foot depth in the body frame, not terrain clearance.
            walk_even_0p04s_base_to_mean_foot_depth_body_m=round(walk_hbase, 3) if np.isfinite(walk_hbase) else None,
            moving_prehold=route_metrics(
                poses_arr, ticks_arr, fz, world.foot_bodies, world.foot_names,
                int(done_tick[n_ho + i]), state == "success", v, f,
                shank_ids=world.leg_shank_bodies,
                shank_names=[f"{lg}_SHANK" for lg in world.leg_order],
                terrain_name=r["terrain"], start_xyz=r["start"], goal_xyz=r["goal"],
            ),
            walk_even_0p04s_foot_acc_rms=round(foot_acc_rms, 1) if np.isfinite(foot_acc_rms) else None,
            walk_diag_pair_pct=round(100.0 * diag_frac, 1),
        )
        haa_report.append(hrec)
        strict_out.append(dict(**r, result=state, end_tick=int(done_tick[n_ho + i]),
                               min_dist_m=round(float(min_dist[n_ho + i]), 3),
                               max_hold_ticks=int(max_hold[n_ho + i]),
                               checkpoint=int(ck["update"]), method=ck["method"]))
        bank.append(dict(
            folder=folder, terrain=r["terrain"], id=int(r["tile_idx"]),
            ticks=st_ticks[i], hold=st_holds[i],
            start=r["start"], goal=r["goal"], state=state,
            seed=int(ck["seed"]), update=int(ck["update"]),
            mesh=dict(file="ground.mesh", vertices=int(len(v)), triangles=int(len(f))),
            bodies=int(poses_arr.shape[1]),
        ))

    (prep / "segments.json").write_text(json.dumps(bank, indent=2) + "\n")
    st_succ = sum(1 for r in strict_out if r["result"] == "success")
    st_pass_h = [h for h in haa_report if h["result"] == "success"]
    st_traversed_succ = sum(
        1 for h in st_pass_h
        if (h["moving_prehold"].get("obstacle_traversal") or {}).get("traversed", False)
    )
    st_clean_hold = sum(1 for h in st_pass_h if h["any_haa_hold1s_max_consec_pin"] == 0 and (h["lh_foot_hold1s_cross_pct"] or 0.0) == 0.0)
    st_clean_bilateral_traversed = sum(
        1 for h in st_pass_h
        if h["any_haa_hold1s_max_consec_pin"] == 0
        and (h["lh_foot_hold1s_cross_pct"] or 0.0) == 0.0
        and (h["moving_prehold"].get("last1s_even_rh_y_m") is not None and h["moving_prehold"]["last1s_even_rh_y_m"] < 0.0)
        and (h["moving_prehold"].get("obstacle_traversal") or {}).get("traversed", False)
    )
    ho_any_pin = np.maximum(ho_max_lo_run, ho_max_hi_run).max(axis=1)
    def _med(vals):
        v = [x for x in vals if x is not None and np.isfinite(x)]
        return round(float(np.median(v)), 3) if v else None
    summary = dict(
        method=ck["method"],
        recipe=init["recipe"],
        panel_mode=args.panel_mode,
        catalog_path=str(args.catalog.resolve()),
        catalog_version=world.catalog_meta.get("version", "unknown"),
        catalog_holdout_sha256=hold_sha,
        catalog_strict_sha256=strict_sha,
        haa_target_map=haa_map,
        haa_limit_pen_weight=pen_w,
        haa_soft_factor=soft_fac,
        scan_offset_x=scan_offset_x,
        scan_size=list(scan_size),
        void_mode=void_mode,
        obs_dim=obs_dim_eval,
        haa_limits_info=world.haa_limits_info,
        update=int(ck["update"]),
        seed=int(ck["seed"]),
        gait_measurement_schema=SCHEMA,
        hind_foot_terminal_telemetry_version=1,
        heldout_attempt_gt_0p5m=int((ho_max_fwd_proj > 0.50).sum()),
        heldout_far_foot_loaded=int(ho_far_foot_loaded.sum()),
        heldout_entered_goal_xy=int(ho_ever_goal_xy.sum()),
        heldout_succ_end_both_hind_on_own_sides=int((ho_succ_mask & ho_lh_left_mask & (ho_end_rh_yb < 0)).sum()),
        heldout_succ_end_hind_width_med_m=_med((ho_end_lh_yb-ho_end_rh_yb)[ho_succ_mask]),
        heldout_succ_end_hind_center_med_m=_med(((ho_end_lh_yb+ho_end_rh_yb)*0.5)[ho_succ_mask]),
        metric_aliases={"strict_pass_med_h_base_m": "legacy body-frame base-to-four-foot depth; not terrain clearance",
                        "strict_pass_med_diag_pct": "pre-hold contact fraction; no speed filter"},
        heldout_active_ticks=int(ho_act_ticks), heldout_near_active_ticks=int(ho_near_ticks),
        heldout_reward_ledger={
            k: v / ho_act_ticks for k, v in (
                ("env_reward_mean", ho_env_sum), ("original_ten_scaled_mean", ho_main_sum),
                ("extra_reg_scaled_mean", ho_extra_sum), ("air_bonus_mean", ho_air_sum),
                ("alive_bonus_mean", ho_alive_sum), ("haa_limit_pen_mean", ho_limit_sum),
                ("residual_mean", ho_env_sum-ho_main_sum-ho_extra_sum-ho_air_sum-ho_alive_sum-ho_limit_sum))
        },
        strict_pass_moving_prehold_medians={
            k: _med([h["moving_prehold"][k] for h in st_pass_h])
            for k in (
                "foot_acc_rms_m_s2", "diagonal_only_pct",
                "stance_rel_height_m", "mesh_clearance_below_base_m", "base_to_mean_foot_depth_body_m",
                "moving_speed_m_s", "crab_deg_med",
                "shank_vert_world_deg_med", "shank_vert_world_stance_deg_med",
                "shank_vert_world_stance_max_leg_med", "shank_world_stance_above_deadzone_pct",
                "shank_vert_world_swing_deg_med", "shank_vert_body_deg_med",
                "shank_pairwise_spread_deg_med", "inter_shank_deg_med",
            )
        },
        actor_sha256=actor_sha,
        actor_sha256_after=actor_sha_after,
        common_panel_sha256=actual_common_sha,
        n_heldout=n_ho,
        heldout_success=ho_succ,
        heldout_rate=round(ho_succ / n_ho, 4),
        heldout_equal_terrain_rate=round(float(np.mean([r["rate"] for r in by_terrain])), 4),
        heldout_succ_and_lh_left=ho_succ_and_lh_left,
        heldout_succ_and_lh_left_rate=round(ho_succ_and_lh_left / n_ho, 4),
        heldout_succ_clean=ho_succ_clean,
        heldout_succ_clean_rate=round(ho_succ_clean / n_ho, 4),
        common2042_raw_success=common2042_raw,
        common2042_equal_terrain_rate=common2042_eq_rate,
        common2042_succ_and_lh_left=common2042_succ_and_lh_left,
        common2042_succ_and_lh_left_eq_rate=common2042_succ_and_lh_left_eq_rate,
        common2042_succ_clean=common2042_succ_clean,
        common2042_succ_clean_eq_rate=common2042_succ_clean_eq_rate,
        n_strict=n_st,
        strict_success=st_succ,
        strict_traversed_and_held_16=st_traversed_succ,
        strict_clean_hold1s=st_clean_hold,
        strict_clean_bilateral_traversed_16=st_clean_bilateral_traversed,
        strict_pass_med_acc_0p04s=_med([h["walk_even_0p04s_foot_acc_rms"] for h in st_pass_h]),
        strict_pass_med_crab_deg=_med([h["walk_even_0p04s_crab_deg"] for h in st_pass_h]),
        strict_pass_med_foot_rad_m=_med([h["walk_even_0p04s_foot_rad_m"] for h in st_pass_h]),
        strict_pass_med_h_base_m=_med([h["walk_even_0p04s_h_base_m"] for h in st_pass_h]),
        strict_pass_med_diag_pct=_med([h["walk_diag_pair_pct"] for h in st_pass_h]),
        heldout_clean_fixed_pct=round(100.0 * ho_succ_clean / n_ho, 2),
        heldout_any_haa_p95_max_consec=round(float(np.percentile(ho_any_pin, 95)), 2),
        heldout_any_haa_ge20_routes=int((ho_any_pin >= 20).sum()),
        heldout_succ_any_haa_ge20_routes=int((ho_any_pin[ho_succ_mask] >= 20).sum()),
        heldout_all_lo1_pct={world.haa_names[j]: round(100.0 * float(ho_lo1_ticks[j]) / max(1, ho_act_ticks), 2) for j in range(4)},
        heldout_all_hi1_pct={world.haa_names[j]: round(100.0 * float(ho_hi1_ticks[j]) / max(1, ho_act_ticks), 2) for j in range(4)},
        heldout_near_lo1_pct={world.haa_names[j]: round(100.0 * float(ho_near_lo1_ticks[j]) / ho_near_ticks, 2) if ho_near_ticks else None for j in range(4)},
        heldout_near_hi1_pct={world.haa_names[j]: round(100.0 * float(ho_near_hi1_ticks[j]) / ho_near_ticks, 2) if ho_near_ticks else None for j in range(4)},
        heldout_p95_max_consec_lo1={world.haa_names[j]: round(float(np.percentile(ho_max_lo_run[:, j], 95)), 2) for j in range(4)},
        heldout_p95_max_consec_hi1={world.haa_names[j]: round(float(np.percentile(ho_max_hi_run[:, j], 95)), 2) for j in range(4)},
        heldout_max_consec_lo1={world.haa_names[j]: int(ho_max_lo_run[:, j].max()) for j in range(4)},
        heldout_max_consec_hi1={world.haa_names[j]: int(ho_max_hi_run[:, j].max()) for j in range(4)},
        heldout_near_lh_cross_pct=round(100.0 * float(ho_near_lh_cross_ticks) / ho_near_ticks, 2) if ho_near_ticks else None,
        heldout_succ_lh_cross_pct=ho_succ_lh_cross_rate,
        heldout_succ_lh_yb_med_m=ho_succ_lh_yb_med,
        heldout_raw_reg_mean=round(ho_raw_reg_sum / max(1, ho_act_ticks), 4),
        heldout_reg_scale_mean=round(ho_reg_scale_sum / max(1, ho_act_ticks), 4),
        heldout_haa_limit_excess_mean=round(ho_exc_sum / max(1, ho_act_ticks), 6),
        wall_s=round(monotonic() - t0, 1),
    )
    (args.out / "heldout_by_terrain.json").write_text(json.dumps(by_terrain, indent=2) + "\n")
    (args.out / "strict_routes.json").write_text(json.dumps(strict_out, indent=2) + "\n")
    (args.out / "haa_summary.json").write_text(json.dumps(haa_report, indent=2) + "\n")
    tmp_s = args.out / "summary.json.tmp"
    tmp_s.write_text(json.dumps(summary, indent=2) + "\n")
    tmp_s.replace(args.out / "summary.json")

    panel_tag = (
        f"common2042={100*common2042_eq_rate:.1f}%({common2042_raw}/2042)"
        if args.panel_mode == "common_manifest"
        else f"catalog_holdout={100*panel_eq_rate:.1f}%({panel_raw}/{n_ho})"
    )
    print(f"[HAA-FULL] {ck['method']} mode={args.panel_mode} map={haa_map} pen={pen_w} u{int(ck['update']):04d}: "
          f"{panel_tag} strict_hold={st_succ}/{n_st} strict_trav={st_traversed_succ}/{n_st} "
          f"succ&LH_L={ho_succ_and_lh_left}/{n_ho} clean={ho_succ_clean}/{n_ho} ho_succ_lh_cross={ho_succ_lh_cross_rate}% (med_y={ho_succ_lh_yb_med}m) "
          f"near_lo1={summary['heldout_near_lo1_pct']} near_hi1={summary['heldout_near_hi1_pct']}", flush=True)
    for h in haa_report:
        if h["result"] == "success":
            pj = h["per_joint_hold1s"]
            jstr = " ".join(f"{nm[:2]}:{pj[nm]['meas_med_deg']:+5.1f}°(L{pj[nm]['max_consec_lo1']}/H{pj[nm]['max_consec_hi1']})" for nm in world.haa_names)
            print(f"  PASS {h['folder']:22s} | {jstr} | LH_y={h['lh_foot_hold1s_yb_med_m']:+.3f}m(cross={h['lh_foot_hold1s_cross_pct']}%) | 0.04s acc={h['walk_even_0p04s_foot_acc_rms']}", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
