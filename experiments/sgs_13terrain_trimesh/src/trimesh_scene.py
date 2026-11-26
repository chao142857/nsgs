"""Batched Isaac Lab GPU scene for the 260-tile 1:1 3D Trimesh 13-terrain benchmark.

Uses the verified recipe-13t-v3 locomotion physics & reward:
- terrain_friction = 1.0 (PhysX static & dynamic)
- smooth rational penalty saturation M = 0.095 (zero saturation cliff, V_alive > -50)
- potential-based signed progress reward (first 500 updates, zero oscillation arbitrage)
- vertical foot contact F_z > 15 N, >= 3 stance feet, zero non-foot contact arrival
- goal_height_tolerance = 0.15 m, 3.0s (150-tick) hold for success, 20.0s (1000-tick) limit
"""
from __future__ import annotations

import hashlib
import json
import math


def map_haa_targets_soft_tanh(q_raw_h, q0_h, soft_lo, soft_hi):
    """C1-smooth asymmetric tanh mapping anchored at q0_h strictly inside (soft_lo, soft_hi)."""
    import torch
    span_pos = (soft_hi - q0_h).clamp_min(1e-4)
    span_neg = (q0_h - soft_lo).clamp_min(1e-4)
    dq = q_raw_h - q0_h
    return torch.where(
        dq >= 0.0,
        q0_h + span_pos * torch.tanh(dq / span_pos),
        q0_h - span_neg * torch.tanh((-dq) / span_neg),
    )


def compute_soft_limits_from_factor(phys_lo, phys_hi, factor: float):
    mid = 0.5 * (phys_lo + phys_hi)
    half = 0.5 * (phys_hi - phys_lo)
    return mid - factor * half, mid + factor * half
from pathlib import Path
import numpy as np

import navigation as nav
from diagnostics import stable_arrival
from trimesh_terrains import TERRAIN_NAMES, terrain_arrays

TERRAIN_PRIM = "/World/SGSTerrainTrimesh"
PHYSICS_DT = 0.005
CONTROL_SUBSTEPS = 4
TILE_SPACING = 14.0
N_TILES = 260  # 13 terrains x 2 seeds x 10 levels


def tile_origins() -> np.ndarray:
    """World origin (x, y, z) for each of the 260 tiles (k*20 + s*10 + l)."""
    origins = np.zeros((N_TILES, 3), dtype=np.float32)
    for k in range(len(TERRAIN_NAMES)):
        for s in range(2):
            for l in range(10):
                tidx = k * 20 + s * 10 + l
                origins[tidx, 0] = l * TILE_SPACING
                origins[tidx, 1] = (k * 2 + s) * TILE_SPACING
                origins[tidx, 2] = 0.0
    return origins


class TrimeshBatchedScene:
    def __init__(self, catalog_dir: Path, num_envs: int, *, terrain_friction: float = 1.0, max_episode_steps: int = 1000, haa_target_map: str = "none", haa_limit_pen_weight: float = 0.0, haa_soft_factor: float = 0.95, scan_offset_x: float = 0.0, scan_size: tuple[float, float] = (1.6, 1.0), void_mode: str = "legacy_neg"):
        import torch
        import omni.usd
        from pxr import Gf, Sdf, UsdGeom, UsdPhysics, Vt
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg
        from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
        from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
        from isaaclab.utils import configclass
        from isaaclab_assets.robots.anymal import ANYMAL_D_CFG

        if void_mode not in ("legacy_neg", "drop_pos", "drop_pos_with_mask"):
            raise ValueError(f"Unsupported void_mode: {void_mode}")
        self.scan_offset_x = float(scan_offset_x)
        self.scan_size = (float(scan_size[0]), float(scan_size[1]))
        self.void_mode = str(void_mode)
        sx_off = self.scan_offset_x
        sx_sz, sy_sz = self.scan_size

        self.torch = torch
        self.device = "cuda:0"
        self.n = int(num_envs)
        self.num_envs = self.n
        self.num_actions = 12
        self.max_episode_steps = int(max_episode_steps)
        self.catalog_dir = Path(catalog_dir)

        meta = json.loads((self.catalog_dir / "catalog_meta.json").read_text())
        self.catalog_meta = meta
        self.terrain_set = str(meta.get("terrain_set", "v3"))
        train_arr = np.load(self.catalog_dir / "train_tasks.npy", allow_pickle=False)
        hold_arr = np.load(self.catalog_dir / "holdout_tasks.npy", allow_pickle=False)
        assert hashlib.sha256(train_arr.tobytes()).hexdigest() == meta["train_sha256"]
        assert hashlib.sha256(hold_arr.tobytes()).hexdigest() == meta["holdout_sha256"]

        # Build & verify combined 260-tile USD trimesh
        origins_np = tile_origins()
        all_v, all_f, ghashes = [], [], []
        offset = 0
        for tidx in range(N_TILES):
            k = tidx // 20
            s = (tidx % 20) // 10
            l = tidx % 10
            v, f = terrain_arrays(k, s, l, terrain_set=self.terrain_set)
            ghashes.append(hashlib.sha256(v.tobytes() + f.tobytes()).hexdigest())
            wv = v.copy()
            wv[:, 0] += origins_np[tidx, 0]
            wv[:, 1] += origins_np[tidx, 1]
            wv[:, 2] += origins_np[tidx, 2]
            all_v.append(wv)
            all_f.append(f.astype(np.int32).reshape(-1) + offset)
            offset += len(v)
        collision_sha = hashlib.sha256("".join(ghashes).encode()).hexdigest()
        assert collision_sha == meta["collision_sha256"], (
            f"Collision mesh hash mismatch! built={collision_sha} catalog={meta[collision_sha256]}"
        )
        vertices = np.concatenate(all_v, axis=0)
        indices = np.concatenate(all_f, axis=0)

        @configclass
        class SceneCfg(InteractiveSceneCfg):
            robot = ANYMAL_D_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
            contacts = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", update_period=0.0, history_length=4)
            scan = RayCasterCfg(
                prim_path="{ENV_REGEX_NS}/Robot/base",
                offset=RayCasterCfg.OffsetCfg(pos=(sx_off, 0.0, 3.0)),
                ray_alignment="yaw",
                update_period=nav.CONTROL_DT,
                pattern_cfg=patterns.GridPatternCfg(resolution=0.10, size=(sx_sz, sy_sz)),
                mesh_prim_paths=[TERRAIN_PRIM],
            )
            light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2300.0))

        sim_cfg = sim_utils.SimulationCfg(
            dt=PHYSICS_DT, render_interval=CONTROL_SUBSTEPS, device=self.device, use_fabric=True,
            physx=sim_utils.PhysxCfg(
                gpu_collision_stack_size=(256 * 2**20 if self.n >= 16384 else 2**26),
                gpu_max_rigid_patch_count=max(5 * 2**15, 512 * self.n),
                gpu_max_rigid_contact_count=max(2**23, 1024 * self.n),
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=float(terrain_friction),
                dynamic_friction=float(terrain_friction),
                restitution=0.0,
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
            ),
        )
        self.sim = sim_utils.SimulationContext(sim_cfg)
        stage = omni.usd.get_context().get_stage()
        mesh_prim = UsdGeom.Mesh.Define(stage, Sdf.Path(TERRAIN_PRIM))
        mesh_prim.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(vertices))
        mesh_prim.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(indices))
        mesh_prim.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(indices) // 3, 3, dtype=np.int32)))
        mesh_prim.CreateSubdivisionSchemeAttr("none")
        mesh_prim.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(0.39, 0.52, 0.64)]))
        mesh_prim.CreateDoubleSidedAttr(True)
        UsdPhysics.CollisionAPI.Apply(mesh_prim.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(mesh_prim.GetPrim()).CreateApproximationAttr("none")
        mat_cfg = sim_utils.RigidBodyMaterialCfg(
            static_friction=float(terrain_friction),
            dynamic_friction=float(terrain_friction),
            restitution=0.0,
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
        )
        mat_path = f"{TERRAIN_PRIM}/physicsMaterial"
        mat_cfg.func(mat_path, mat_cfg)
        sim_utils.bind_physics_material(TERRAIN_PRIM, mat_path, stage=stage)

        self.scene = InteractiveScene(SceneCfg(num_envs=self.n, env_spacing=TILE_SPACING))
        self.sim.reset()
        self.robot = self.scene["robot"]
        self.contact = self.scene["contacts"]
        self.scan = self.scene["scan"]
        self.env_ids = torch.arange(self.n, device=self.device, dtype=torch.long)

        # Store task banks on GPU: [tile_idx, sx, sy, sz, syaw, gx, gy, gz]
        self.tile_origins_t = torch.tensor(origins_np, device=self.device, dtype=torch.float32)
        self.banks = {
            "train": torch.tensor(train_arr, device=self.device, dtype=torch.float32),
            "holdout": torch.tensor(hold_arr, device=self.device, dtype=torch.float32),
        }
        self.active_bank_name = "train"
        self.active_bank = self.banks["train"]

        # Optional precomputed 2D shortest-path geodesic fields for the 20 maze tiles (used ONLY for
        # training warmup potential shaping u0..999; NEVER fed into observations or evaluation!)
        maze_npz = self.catalog_dir / "maze_geodesic.npz"
        if maze_npz.exists():
            mg = np.load(maze_npz, allow_pickle=False)
            self.maze_fields = torch.tensor(mg["fields"], device=self.device, dtype=torch.float32)
            self.maze_bank_gids = {
                "train": torch.tensor(mg["train_gid"], device=self.device, dtype=torch.long),
                "holdout": torch.tensor(mg["holdout_gid"], device=self.device, dtype=torch.long),
            }
        else:
            self.maze_fields = None
            self.maze_bank_gids = {}
        self.env_maze_gid = torch.full((self.n,), -1, device=self.device, dtype=torch.long)

        self.task_ids = torch.zeros(self.n, device=self.device, dtype=torch.long)
        self.origins = torch.zeros((self.n, 3), device=self.device, dtype=torch.float32)
        self.targets = torch.zeros((self.n, 3), device=self.device, dtype=torch.float32)
        self.steps = torch.zeros(self.n, device=self.device, dtype=torch.long)
        self.hold = torch.zeros(self.n, device=self.device, dtype=torch.long)
        self.prev_dist = torch.zeros(self.n, device=self.device, dtype=torch.float32)
        self.last_action = torch.zeros((self.n, 12), device=self.device, dtype=torch.float32)
        self.previous_contact = torch.zeros((self.n, 4), device=self.device, dtype=torch.bool)
        self.last_reward = torch.zeros((self.n, len(nav.TERMS)), device=self.device, dtype=torch.float32)

        self.foot_bodies, self.foot_names = self.robot.find_bodies(".*_FOOT")
        self.foot_sensors, sensor_names = self.contact.find_bodies(self.foot_names, preserve_order=True)
        if list(sensor_names) != list(self.foot_names) or len(self.foot_bodies) != 4:
            raise RuntimeError((self.foot_names, sensor_names))
        self.base_sensor = self.contact.find_bodies("base")[0][0]
        self.non_feet = [i for i in range(len(self.contact.body_names)) if i not in self.foot_sensors]

        ray_starts = getattr(self.scan, "ray_starts", getattr(self.scan, "_ray_starts", None))
        if ray_starts is not None:
            xy_norm = ray_starts[0, :, :2].norm(dim=-1)
            self.center_ray = int(xy_norm.argmin().item())
        else:
            self.center_ray = self.scan.data.ray_hits_w.shape[1] // 2

        if haa_target_map not in ("none", "soft_tanh"):
            raise ValueError(f"Unsupported haa_target_map: {haa_target_map}")
        if float(haa_limit_pen_weight) < 0.0:
            raise ValueError(f"haa_limit_pen_weight must be >= 0, got {haa_limit_pen_weight}")
        if not (0.5 <= float(haa_soft_factor) <= 1.0):
            raise ValueError(f"haa_soft_factor must be in [0.5, 1.0], got {haa_soft_factor}")
        self.haa_target_map = str(haa_target_map)
        self.haa_limit_pen_weight = float(haa_limit_pen_weight)
        self.haa_soft_factor = float(haa_soft_factor)
        self.prev_joint_vel = torch.zeros((self.n, 12), device=self.device, dtype=torch.float32)
        self.feet_air_time = torch.zeros((self.n, 4), device=self.device, dtype=torch.float32)

        haa_ids, haa_names = self.robot.find_joints(".*HAA")
        if len(haa_ids) != 4:
            raise RuntimeError(f"Expected 4 HAA joints, got {haa_names} ({haa_ids})")
        order = sorted(range(4), key=lambda i: haa_names[i])
        self.haa_ids = [int(haa_ids[i]) for i in order]
        self.haa_names = [str(haa_names[i]) for i in order]
        self.haa_ids_t = torch.tensor(self.haa_ids, device=self.device, dtype=torch.long)
        self.haa_joint_weights_l2 = torch.ones((1, 12), device=self.device, dtype=torch.float32)
        self.haa_joint_weights_l2[0, self.haa_ids_t] = 2.0

        # Explicit per-leg matched joint & link indices in canonical order (LF, LH, RF, RH)
        self.leg_order = ("LF", "LH", "RF", "RH")
        leg_haa, leg_hfe, leg_kfe, leg_shank, leg_foot = [], [], [], [], []
        for lg in self.leg_order:
            j_haa, _ = self.robot.find_joints(f"{lg}_HAA")
            j_hfe, _ = self.robot.find_joints(f"{lg}_HFE")
            j_kfe, _ = self.robot.find_joints(f"{lg}_KFE")
            b_shk, _ = self.robot.find_bodies(f"{lg}_SHANK")
            b_ft, _ = self.robot.find_bodies(f"{lg}_FOOT")
            if len(j_haa) != 1 or len(j_hfe) != 1 or len(j_kfe) != 1 or len(b_shk) != 1 or len(b_ft) != 1:
                raise RuntimeError(f"Failed to bind unique leg entities for {lg}")
            leg_haa.append(int(j_haa[0]))
            leg_hfe.append(int(j_hfe[0]))
            leg_kfe.append(int(j_kfe[0]))
            leg_shank.append(int(b_shk[0]))
            leg_foot.append(int(b_ft[0]))
        self.leg_haa_ids_t = torch.tensor(leg_haa, device=self.device, dtype=torch.long)
        self.leg_hfe_ids_t = torch.tensor(leg_hfe, device=self.device, dtype=torch.long)
        self.leg_kfe_ids_t = torch.tensor(leg_kfe, device=self.device, dtype=torch.long)
        self.leg_hfe_kfe_ids_t = torch.cat((self.leg_hfe_ids_t, self.leg_kfe_ids_t), dim=0)
        self.leg_shank_bodies = leg_shank
        self.leg_foot_bodies = leg_foot

        lh_ids, lh_names = self.robot.find_bodies("LH_FOOT")
        if len(lh_ids) != 1:
            raise RuntimeError(f"Expected 1 LH_FOOT body, got {lh_names}")
        self.lh_foot_body_idx = int(lh_ids[0])
        phys_t = self.robot.data.joint_pos_limits[0, self.haa_ids_t]
        if abs(self.haa_soft_factor - 0.95) < 1e-6:
            soft_lo_t = self.robot.data.soft_joint_pos_limits[0, self.haa_ids_t, 0]
            soft_hi_t = self.robot.data.soft_joint_pos_limits[0, self.haa_ids_t, 1]
        else:
            soft_lo_t, soft_hi_t = compute_soft_limits_from_factor(phys_t[:, 0], phys_t[:, 1], self.haa_soft_factor)
        phys = phys_t.detach().cpu().numpy()
        s_lo = soft_lo_t.detach().cpu().numpy()
        s_hi = soft_hi_t.detach().cpu().numpy()
        q0_h = self.robot.data.default_joint_pos[0, self.haa_ids_t].detach().cpu().numpy()
        self.haa_limits_info = {
            nm: dict(
                joint_idx=int(self.haa_ids[i]),
                default_deg=round(float(math.degrees(q0_h[i])), 3),
                phys_lo_deg=round(float(math.degrees(phys[i, 0])), 3),
                phys_hi_deg=round(float(math.degrees(phys[i, 1])), 3),
                soft_lo_deg=round(float(math.degrees(s_lo[i])), 3),
                soft_hi_deg=round(float(math.degrees(s_hi[i])), 3),
                soft_factor=self.haa_soft_factor,
            )
            for i, nm in enumerate(self.haa_names)
        }

    def select_bank(self, name: str):
        self.active_bank_name = name
        self.active_bank = self.banks[name]

    def place(self, slots, task_ids):
        torch = self.torch
        slots = torch.as_tensor(slots, device=self.device, dtype=torch.long)
        tids = torch.as_tensor(task_ids, device=self.device, dtype=torch.long)
        rows = self.active_bank[tids]  # (M, 8): [tile_idx, sx, sy, sz, syaw, gx, gy, gz]
        tile_idx = rows[:, 0].long()
        orig = self.tile_origins_t[tile_idx]  # (M, 3)
        if self.maze_fields is not None and self.active_bank_name in self.maze_bank_gids:
            self.env_maze_gid[slots] = self.maze_bank_gids[self.active_bank_name][tids]
        else:
            self.env_maze_gid[slots] = -1

        root_state = self.robot.data.default_root_state[slots].clone()
        root_state[:, 0] = orig[:, 0] + rows[:, 1]
        root_state[:, 1] = orig[:, 1] + rows[:, 2]
        root_state[:, 2] = orig[:, 2] + rows[:, 3] + nav.BASE_HEIGHT + 0.02
        half_yaw = 0.5 * rows[:, 4]
        root_state[:, 3] = torch.cos(half_yaw)
        root_state[:, 4] = 0.0
        root_state[:, 5] = 0.0
        root_state[:, 6] = torch.sin(half_yaw)
        root_state[:, 7:] = 0.0

        self.robot.write_root_state_to_sim(root_state, env_ids=slots)
        jpos = self.robot.data.default_joint_pos[slots].clone()
        jvel = torch.zeros_like(jpos)
        self.robot.write_joint_state_to_sim(jpos, jvel, env_ids=slots)
        self.robot.set_joint_position_target(jpos, env_ids=slots)

        self.task_ids[slots] = tids
        self.origins[slots] = orig
        self.targets[slots, 0] = orig[:, 0] + rows[:, 5]
        self.targets[slots, 1] = orig[:, 1] + rows[:, 6]
        self.targets[slots, 2] = orig[:, 2] + rows[:, 7]
        self.steps[slots] = 0
        self.hold[slots] = 0
        self.prev_dist[slots] = (self.targets[slots, :2] - root_state[:, :2]).norm(dim=-1)
        self.last_action[slots] = 0.0
        self.previous_contact[slots] = False
        self.prev_joint_vel[slots] = 0.0
        self.feet_air_time[slots] = 0.0

        if slots.numel() == self.n:
            self.scene.reset()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(PHYSICS_DT)

    def observations(self, *, clip_height: bool = True):
        from isaaclab.utils.math import quat_apply_inverse
        torch = self.torch
        bot = self.robot.data
        target = self.targets - bot.root_pos_w
        target[:, 2] += nav.BASE_HEIGHT
        goal = quat_apply_inverse(bot.root_quat_w, target)
        raw_scan = self.scan.data.ray_hits_w[:, :, 2]
        valid = torch.isfinite(raw_scan)
        rel_h = bot.root_pos_w[:, 2:3] - raw_scan - nav.BASE_HEIGHT
        if self.void_mode == "legacy_neg":
            height = torch.nan_to_num(rel_h, nan=0.0, posinf=2.0, neginf=-2.0)
        else:
            # Warp ray miss returns +inf; map missing floor to +2.0 drop (same sign as pit, never -1.0 high wall)
            clean_rel = torch.nan_to_num(rel_h, nan=0.0, posinf=2.0, neginf=-2.0)
            height = torch.where(valid, clean_rel, torch.full_like(clean_rel, 2.0))
        if clip_height:
            height = height.clamp(-1.0, 1.0)
        if self.void_mode == "drop_pos_with_mask":
            height = torch.cat((height, valid.float()), dim=1)
        return torch.cat((
            bot.root_lin_vel_b, bot.root_ang_vel_b, bot.projected_gravity_b,
            bot.joint_pos - bot.default_joint_pos, bot.joint_vel,
            self.last_action, goal, height,
        ), dim=1)

    def critic_observations(self, policy_obs):
        """Stationary Markovian privileged Critic features (arrival hold counter + 4-foot vertical forces + stance height)."""
        torch = self.torch
        bot = self.robot.data
        root = bot.root_pos_w
        dist = (self.targets[:, :2] - root[:, :2]).norm(dim=-1)
        z_err = (self.targets[:, 2] + nav.BASE_HEIGHT - root[:, 2]).abs()
        near_f = ((dist <= nav.GOAL_RADIUS) & (z_err <= 0.16)).float().unsqueeze(1)
        hold_frac = (self.hold.float() / float(nav.SUCCESS_HOLD)).clamp(0.0, 1.0).unsqueeze(1)
        fz = self.contact.data.net_forces_w[:, self.foot_sensors, 2]
        fc = (fz > nav.MIN_CONTACT_FORCE).float()
        fz_norm = (fz / 250.0).clamp(-0.5, 2.5)
        foot_z = bot.body_pos_w[:, self.foot_bodies, 2]
        cnt = fc.sum(dim=1).clamp_min(1.0)
        h_st = ((root[:, 2] - (foot_z * fc).sum(dim=1) / cnt) - 0.55).clamp(-0.5, 0.5).unsqueeze(1)
        return torch.cat((policy_obs, near_f, hold_frac, fc, fz_norm, h_st), dim=1)

    def _nav_potential_and_heading(self, root_pos_w, root_quat_w):
        """Compute feasible potential distance d_pot and signed yaw error e_yaw in [-pi, pi]."""
        torch = self.torch
        dx = self.targets[:, 0] - root_pos_w[:, 0]
        dy = self.targets[:, 1] - root_pos_w[:, 1]
        d_euc = torch.hypot(dx, dy)
        dir_x, dir_y, d_pot = dx, dy, d_euc

        maze_fields = getattr(self, "maze_fields", None)
        env_gid = getattr(self, "env_maze_gid", None)
        if maze_fields is not None and env_gid is not None:
            use_m = (env_gid >= 0) & (d_euc > 0.70)
            if use_m.any():
                idx = use_m.nonzero(as_tuple=False).squeeze(-1)
                gid = env_gid[idx]
                g_size = maze_fields.shape[-1]
                lx = root_pos_w[idx, 0] - self.origins[idx, 0]
                ly = root_pos_w[idx, 1] - self.origins[idx, 1]
                u = ((lx + 4.95) / 0.10).clamp(0.0, float(g_size) - 1.001)
                v = ((ly + 4.95) / 0.10).clamp(0.0, float(g_size) - 1.001)
                i0 = u.long()
                j0 = v.long()
                fu = u - i0.float()
                fv = v - j0.float()
                f00 = maze_fields[gid, i0, j0]
                f10 = maze_fields[gid, i0 + 1, j0]
                f01 = maze_fields[gid, i0, j0 + 1]
                f11 = maze_fields[gid, i0 + 1, j0 + 1]
                d_geo = (1.0 - fu) * (1.0 - fv) * f00 + fu * (1.0 - fv) * f10 + (1.0 - fu) * fv * f01 + fu * fv * f11
                gx_m = -((1.0 - fv) * (f10 - f00) + fv * (f11 - f01)) / 0.10
                gy_m = -((1.0 - fu) * (f01 - f00) + fu * (f11 - f10)) / 0.10
                gnorm = torch.hypot(gx_m, gy_m)
                valid_g = gnorm > 1e-3
                d_pot = d_pot.clone()
                dir_x = dir_x.clone()
                dir_y = dir_y.clone()
                d_pot[idx] = torch.where(valid_g, d_geo, d_euc[idx])
                dir_x[idx] = torch.where(valid_g, gx_m, dx[idx])
                dir_y[idx] = torch.where(valid_g, gy_m, dy[idx])

        qw, qx, qy, qz = root_quat_w.unbind(dim=1)
        yaw = torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        cy, sy = torch.cos(yaw), torch.sin(yaw)
        gx_yaw = cy * dir_x + sy * dir_y
        gy_yaw = -sy * dir_x + cy * dir_y
        heading_err = torch.atan2(gy_yaw, gx_yaw)
        return d_pot, heading_err, cy, sy

    def step(self, action, *, training_iteration: int = 0, recipe: str = "recipe-13t-v3", fwd_mode: str = "first500"):
        torch = self.torch
        cfg = nav.get_recipe(recipe)
        is_v5 = (cfg.get("family") == "v5_upright")
        if is_v5:
            d_pot_before, herr_before, _, _ = TrimeshBatchedScene._nav_potential_and_heading(
                self, self.robot.data.root_pos_w, self.robot.data.root_quat_w
            )
            abs_herr_before = herr_before.abs()

        action_rate = (action - self.last_action).square().sum(dim=1)
        action_rate = torch.where(self.steps > 0, action_rate, torch.zeros_like(action_rate))
        self.last_action.copy_(action)
        q0_all = self.robot.data.default_joint_pos
        q_raw = q0_all + nav.ACTION_SCALE * action
        if self.haa_target_map == "soft_tanh":
            h = self.haa_ids_t
            if abs(self.haa_soft_factor - 0.95) < 1e-6:
                soft_lo_t = self.robot.data.soft_joint_pos_limits[:, h, 0]
                soft_hi_t = self.robot.data.soft_joint_pos_limits[:, h, 1]
            else:
                soft_lo_t, soft_hi_t = compute_soft_limits_from_factor(
                    self.robot.data.joint_pos_limits[:, h, 0],
                    self.robot.data.joint_pos_limits[:, h, 1],
                    self.haa_soft_factor,
                )
            q_cmd = q_raw.clone()
            q_cmd[:, h] = map_haa_targets_soft_tanh(q_raw[:, h], q0_all[:, h], soft_lo_t, soft_hi_t)
        else:
            q_cmd = q_raw
        self.robot.set_joint_position_target(q_cmd)
        for _ in range(CONTROL_SUBSTEPS):
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(PHYSICS_DT)

        height_tol = cfg["goal_height_tolerance"]
        min_stance = cfg["min_arrival_stance_feet"]
        req_zero_nf = cfg["require_zero_nonfoot_arrival"]
        fwd_cap = cfg["fwd_speed_cap"]

        state = self.robot.data
        root = state.root_pos_w
        xy = self.targets[:, :2] - root[:, :2]
        dist = xy.norm(dim=-1)
        z_error = (self.targets[:, 2] + nav.BASE_HEIGHT - root[:, 2]).abs()
        near = (dist <= nav.GOAL_RADIUS) & (z_error <= height_tol)
        self.steps += 1

        force = self.contact.data.net_forces_w.norm(dim=-1)
        max_force = self.contact.data.net_forces_w_history.norm(dim=-1).amax(dim=1)
        foot = force[:, self.foot_sensors]
        foot_force_z = self.contact.data.net_forces_w[:, self.foot_sensors, 2]
        foot_contact = foot_force_z > nav.MIN_CONTACT_FORCE
        nonfoot_hits = (max_force[:, self.non_feet] > nav.MIN_CONTACT_FORCE).sum(dim=1)

        stable = stable_arrival(
            near, state.projected_gravity_b[:, 2],
            state.root_lin_vel_w[:, :2].norm(dim=-1),
            state.root_ang_vel_w.norm(dim=-1),
            foot_contact.sum(dim=1),
            min_stance_feet=min_stance,
            nonfoot_contact=(nonfoot_hits if req_zero_nf else None),
        )
        self.hold = torch.where(stable, self.hold + 1, 0)
        touchdown = foot_contact & ~self.previous_contact
        self.previous_contact.copy_(foot_contact)

        result = torch.zeros(self.n, dtype=torch.int8, device=self.device)
        result[self.steps >= self.max_episode_steps] = 4
        result[self.hold >= nav.SUCCESS_HOLD] = 1
        result[max_force[:, self.base_sensor] > nav.BASE_CONTACT_FORCE] = 3
        result[root[:, 2] - self.origins[:, 2] < nav.DROP_HEIGHT] = 2

        vel_feet = state.body_lin_vel_w[:, self.foot_bodies, 2]
        flat_orient = state.projected_gravity_b[:, :2].square().sum(dim=1)
        dq_posture = state.joint_pos - state.default_joint_pos

        # Update joint_acc & feet_air_time state
        dof_acc_sq = ((state.joint_vel - self.prev_joint_vel) / nav.CONTROL_DT).square().sum(dim=1)
        dof_acc_sq = torch.where(self.steps > 1, dof_acc_sq, torch.zeros_like(dof_acc_sq))
        self.prev_joint_vel.copy_(state.joint_vel)
        air_before_td = self.feet_air_time.clone()
        self.feet_air_time = torch.where(foot_contact, torch.zeros_like(self.feet_air_time), self.feet_air_time + nav.CONTROL_DT)

        v5_diag = None
        if is_v5:
            # Compute potential progress & body-yaw velocities first so posture/air gates use exact motion state
            d_pot_after, heading_err, cy, sy = TrimeshBatchedScene._nav_potential_and_heading(
                self, root, state.root_quat_w
            )
            vwx, vwy = state.root_lin_vel_w[:, 0], state.root_lin_vel_w[:, 1]
            vx_yaw = cy * vwx + sy * vwy
            vy_yaw = -sy * vwx + cy * vwy
            planar_spd = torch.hypot(vwx, vwy)

            # 1. Support-relative upright height wall [h_min, h_max] (applied UNCOMPRESSED outside reg_soft_cap!)
            foot_pos_w = state.body_pos_w[:, self.foot_bodies, :]
            foot_z = foot_pos_w[:, :, 2]
            stance_f = foot_contact.float()
            stance_cnt = stance_f.sum(dim=1)
            z_stance = (foot_z * stance_f).sum(dim=1) / stance_cnt.clamp_min(1.0)
            h_stance = root[:, 2] - z_stance
            h_min = float(cfg.get("stance_height_min", 0.548))
            h_max = float(cfg.get("stance_height_max", 0.590))
            h_air_min = float(cfg.get("air_crouch_min_height", 0.46))
            if bool(cfg.get("step_straddle_compliance", False)):
                foot_span_z = (foot_z.max(dim=1).values - foot_z.min(dim=1).values - 0.10).clamp(0.0, 0.14)
                eff_h_min = h_min - 0.65 * foot_span_z
            else:
                eff_h_min = h_min
            h_err = (eff_h_min - h_stance).clamp(0.0, 0.18).square() + 0.5 * (h_stance - h_max).clamp(0.0, 0.15).square()
            air_crouch = (h_air_min - (root[:, 2] - foot_z.mean(dim=1))).clamp(0.0, 0.18).square()
            h_wall_p = float(cfg.get("uncomp_height_wall_w", 8.0)) * torch.where(stance_cnt >= 1.0, h_err, air_crouch)
            height_term = torch.zeros_like(h_stance)

            # 2. Hard kinematic upright-parallel-shank invariants (applied UNCOMPRESSED outside reg_soft_cap!):
            #    (a) World-vertical shank cone (44 deg V5a / 41.5 deg V5b)
            #    (b) Same-side front-hind 3D shank dot-product wall (forbids obtuse / \ sprawl!)
            dq_haa = dq_posture[:, self.leg_haa_ids_t]
            haa_dz = float(cfg.get("haa_deadzone_rad", 0.24))
            haa_err_sq = (dq_haa.abs() - haa_dz).clamp_min(0.0).square().sum(dim=1)

            v_shank_w = state.body_pos_w[:, self.leg_foot_bodies, :] - state.body_pos_w[:, self.leg_shank_bodies, :]
            u_shank_w = v_shank_w / v_shank_w.norm(dim=-1, keepdim=True).clamp_min(1e-4)
            cos_world = -u_shank_w[:, :, 2]
            leg_fc = torch.stack(
                [foot_contact[:, list(self.foot_names).index(f"{lg}_FOOT")] for lg in self.leg_order], dim=1
            ).float() if hasattr(self, "foot_names") else stance_f
            moving_mask = ((planar_spd >= 0.18) & (~near)).float().unsqueeze(1)
            eff_stance_fc = torch.where(moving_mask > 0.5, leg_fc, torch.ones_like(leg_fc))
            eff_swing_fc = 1.0 - eff_stance_fc

            cos_min_st = float(cfg.get("shank_world_cos_min_stance", 0.7193))
            cos_min_sw = float(cfg.get("shank_world_cos_min_swing", 0.5299))
            sh_st_err = ((cos_min_st - cos_world).clamp(0.0, 0.45).square() * eff_stance_fc).sum(dim=1)
            sh_sw_err = ((cos_min_sw - cos_world).clamp(0.0, 0.45).square() * eff_swing_fc).sum(dim=1)

            # Same-side front-hind unit shank dot products: LF(0)·LH(1) and RF(2)·RH(3)
            cos_fh_l = (u_shank_w[:, 0, :] * u_shank_w[:, 1, :]).sum(dim=-1)
            cos_fh_r = (u_shank_w[:, 2, :] * u_shank_w[:, 3, :]).sum(dim=-1)
            fh_cos_min = float(cfg.get("fh_shank_cos_min", -0.05))
            fh_splay_p = float(cfg.get("fh_shank_splay_w", 0.30)) * (
                (fh_cos_min - cos_fh_l).clamp(0.0, 0.90).square() + (fh_cos_min - cos_fh_r).clamp(0.0, 0.90).square()
            )

            r_out = float(cfg.get("near_ramp_outer_m", 0.55))
            r_in = float(cfg.get("near_ramp_inner_m", 0.25))
            elev_ok = (z_error <= height_tol).float()
            near_ramp = (((r_out - dist) / max(r_out - r_in, 1e-4)).clamp(0.0, 1.0).square()) * elev_ok

            haa_p = float(cfg.get("haa_posture_w", 0.30)) * haa_err_sq
            sh_st_p = float(cfg.get("shank_world_stance_w", 1.50)) * sh_st_err
            sh_sw_p = float(cfg.get("shank_world_swing_w", 0.50)) * sh_sw_err
            uncomp_pose_pen = (h_wall_p + sh_st_p + sh_sw_p + fh_splay_p).clamp_max(float(cfg.get("uncomp_pose_cap", 0.34)))

            glob_l2_p = float(cfg.get("global_posture_l2_w", 0.005)) * dq_posture.square().sum(dim=1)
            near_feet_def = float(cfg.get("near_min_feet_w", 0.06)) * near_ramp * (3.5 - stance_cnt).clamp_min(0.0)
            near_vel_damp = float(cfg.get("near_vel_damp_w", 0.05)) * near_ramp * (
                planar_spd.square() + 0.25 * state.root_ang_vel_w.square().sum(dim=1)
            )
            near_st_p = float(cfg.get("near_stand_posture_w", 0.12)) * near_ramp * dq_posture.square().sum(dim=1) + near_feet_def + near_vel_damp
            posture_term = haa_p + glob_l2_p + near_st_p

            # 3. Exact telescoping potential-difference navigation shaping (zero closed-loop oscillation!)
            d_prog = ((d_pot_before - d_pot_after) / (nav.CONTROL_DT * fwd_cap)).clamp(-1.0, 1.0)
            nose_cos = torch.cos(heading_err).clamp(0.0, 1.0)
            if bool(cfg.get("exact_telescoping_fwd", False)):
                fwd_prog = d_prog
            else:
                fwd_prog = torch.where(d_prog > 0.0, d_prog * nose_cos, d_prog)

            yaw_cap = float(cfg.get("yaw_rate_cap", 1.20))
            yaw_prog = ((abs_herr_before - heading_err.abs()) / (nav.CONTROL_DT * yaw_cap)).clamp(-1.0, 1.0)
            yaw_prog = torch.where(dist > 0.30, yaw_prog, torch.zeros_like(yaw_prog))
            yaw_coef = float(cfg.get("yaw_prog_coef", 0.20))
            fwd_term = torch.where(near, torch.zeros_like(dist), fwd_prog + yaw_coef * yaw_prog)

            tau_cap = float(cfg.get("torque_sq_cap", 25000.0))
            act_cap = float(cfg.get("action_rate_cap", 20.0))
            acc_cap = float(cfg.get("dof_acc_cap", 2.0e5))
            effort_term = state.applied_torque.square().sum(dim=1).clamp_max(tau_cap)
            act_rate_term = action_rate.clamp_max(act_cap)
            v5_diag = dict(
                h_stance_mean=h_stance,
                stance_feet_mean=stance_cnt,
                shank_stance_pen=sh_st_p,
                shank_swing_pen=sh_sw_p,
                fh_splay_pen=fh_splay_p,
                height_wall_pen=h_wall_p,
                uncomp_pose_pen=uncomp_pose_pen,
                haa_splay_pen=haa_p,
                near_stand_pen=near_st_p,
                fwd_prog_raw=fwd_prog,
                yaw_prog_raw=yaw_prog,
                act_rate_clipped=(action_rate > act_cap).float(),
                dof_acc_clipped=(dof_acc_sq > acc_cap).float(),
            )
        else:
            progress = ((self.prev_dist - dist) / (nav.CONTROL_DT * fwd_cap)).clamp(-1.0, 1.0)
            fwd_term = torch.where(self.steps > 1, progress, torch.zeros_like(progress))
            center_z = self.scan.data.ray_hits_w[:, self.center_ray, 2]
            h_base = torch.where(torch.isfinite(center_z), root[:, 2] - center_z,
                                 torch.full_like(center_z, nav.BASE_HEIGHT))
            lh_thresh = float(cfg.get("low_height_thresh", 0.48))
            height_term = (lh_thresh - h_base).clamp_min(0.0).square()
            posture_mode = cfg.get("posture_mode", "l1")
            if posture_mode == "l2_haa2":
                posture_term = (dq_posture.square() * self.haa_joint_weights_l2).sum(dim=1)
            else:
                posture_term = dq_posture.abs().sum(dim=1)
            effort_term = (state.applied_torque * state.joint_vel).abs().sum(dim=1)
            act_rate_term = action_rate

        bc_override = cfg.get("base_contact_weight", None)
        w1_base = float(cfg["weights"][1]) if "weights" in cfg else -50.0
        if bc_override is not None and abs(w1_base) > 1e-8:
            fail_term = (result == 2).float() + (float(bc_override) / w1_base) * (result == 3).float()
        else:
            fail_term = ((result == 2) | (result == 3)).float()

        terms = torch.stack((
            (result == 1).float(),
            fail_term,
            effort_term,
            posture_term,
            (touchdown * vel_feet.square()).sum(dim=1),
            nonfoot_hits,
            fwd_term,
            act_rate_term,
            flat_orient,
            height_term,
        ), dim=1)
        self.last_reward.copy_(terms)
        self.prev_dist.copy_(dist)

        weights = terms.new_tensor(nav.reward_weights(training_iteration, fwd_mode=fwd_mode, recipe=recipe))
        weighted = terms * weights
        reg_idx = (2, 3, 4, 5, 7, 8, 9)
        m_cap = float(cfg["reg_soft_cap"])
        cap_mode = cfg.get("reg_cap_mode", "hyperbolic")

        # Extra gait-quality penalties (zero for recipe-13t-v3!)
        dof_acc_w = float(cfg.get("dof_acc_w", 0.0))
        ang_vel_xy_w = float(cfg.get("ang_vel_xy_w", 0.0))
        heading_w = float(cfg.get("heading_w", 0.0))
        air_time_w = float(cfg.get("feet_air_time_w", 0.0))
        extra_reg_pen = torch.zeros(self.n, device=self.device, dtype=torch.float32)
        if dof_acc_w != 0.0:
            acc_in = dof_acc_sq.clamp_max(float(cfg.get("dof_acc_cap", 2.0e5))) if is_v5 else dof_acc_sq
            extra_reg_pen = extra_reg_pen + (-dof_acc_w) * acc_in
        if ang_vel_xy_w != 0.0:
            wxy_sq = state.root_ang_vel_b[:, :2].square().sum(dim=1)
            if is_v5:
                wxy_sq = wxy_sq.clamp_max(float(cfg.get("ang_vel_xy_cap", 4.0)))
            extra_reg_pen = extra_reg_pen + (-ang_vel_xy_w) * wxy_sq
        if heading_w != 0.0:
            v_xy = state.root_lin_vel_w[:, :2]
            v_spd = v_xy.norm(dim=-1)
            q_w = state.root_quat_w
            # Body +x unit vector in world xy from quaternion (w, x, y, z)
            fx = 1.0 - 2.0 * (q_w[:, 2] * q_w[:, 2] + q_w[:, 3] * q_w[:, 3])
            fy = 2.0 * (q_w[:, 1] * q_w[:, 2] + q_w[:, 3] * q_w[:, 0])
            f_norm = torch.hypot(fx, fy).clamp_min(1e-6)
            cos_h = ((fx * v_xy[:, 0] + fy * v_xy[:, 1]) / (f_norm * v_spd.clamp_min(1e-6))).clamp(-1.0, 1.0)
            moving_far = (v_spd > 0.25) & (dist > 0.80)
            head_misalign = torch.where(moving_far, 1.0 - cos_h, torch.zeros_like(cos_h))
            extra_reg_pen = extra_reg_pen + (-heading_w) * head_misalign

        if is_v5:
            lin_vel_z_w = float(cfg.get("lin_vel_z_w", 0.0))
            lat_vel_w = float(cfg.get("lat_vel_w", 0.0))
            back_vel_w = float(cfg.get("back_vel_w", 0.0))
            overspeed_w = float(cfg.get("overspeed_w", 0.0))
            overspeed_th = float(cfg.get("overspeed_thresh", 0.80))
            feet_slide_w = float(cfg.get("feet_slide_w", 0.0))
            hk_lim_w = float(cfg.get("hfe_kfe_soft_limit_w", 0.0))
            urgency_w = float(cfg.get("step_urgency_w", 0.0))
            if urgency_w != 0.0:
                extra_reg_pen = extra_reg_pen + (-urgency_w) * (~near).float()
            if lin_vel_z_w != 0.0:
                vz_sq = state.root_lin_vel_b[:, 2].square().clamp_max(float(cfg.get("lin_vel_z_cap", 0.50)))
                extra_reg_pen = extra_reg_pen + (-lin_vel_z_w) * vz_sq
            if lat_vel_w != 0.0:
                vy_sq = vy_yaw.square().clamp_max(float(cfg.get("lat_vel_cap", 0.64)))
                extra_reg_pen = extra_reg_pen + (-lat_vel_w) * (~near).float() * vy_sq
            if back_vel_w != 0.0:
                vb_sq = (-vx_yaw).clamp_min(0.0).square().clamp_max(0.64)
                extra_reg_pen = extra_reg_pen + (-back_vel_w) * (dist > 0.50).float() * vb_sq
            if feet_slide_w != 0.0:
                foot_vel_xy = state.body_lin_vel_w[:, self.foot_bodies, :2].norm(dim=-1)
                slide_sum = (foot_vel_xy * stance_f).sum(dim=1).clamp_max(float(cfg.get("feet_slide_cap", 1.5)))
                extra_reg_pen = extra_reg_pen + (-feet_slide_w) * slide_sum
            if hk_lim_w != 0.0:
                hk = self.leg_hfe_kfe_ids_t
                q_hk = state.joint_pos[:, hk]
                hk_lo = state.soft_joint_pos_limits[:, hk, 0]
                hk_hi = state.soft_joint_pos_limits[:, hk, 1]
                hk_exc = (-(q_hk - hk_lo).clamp_max(0.0)) + (q_hk - hk_hi).clamp_min(0.0)
                extra_reg_pen = extra_reg_pen + (-hk_lim_w) * hk_exc.sum(dim=1)

        base_reg_x = (-weighted[:, reg_idx].sum(dim=1)).clamp_min(0.0)
        x = base_reg_x + extra_reg_pen
        if cap_mode == "none":
            scale = torch.ones_like(x)
            extra_reg_scaled = -extra_reg_pen
        elif cap_mode == "leaky":
            slope = float(cfg.get("reg_leaky_slope", 0.10))
            ceil = float(cfg.get("reg_hard_ceiling", 0.25))
            y = torch.where(x <= m_cap, x, m_cap + slope * (x - m_cap)).clamp_max(ceil)
            scale = torch.where(x > 1e-8, y / x.clamp_min(1e-8), torch.ones_like(x))
            weighted = weighted.clone()
            weighted[:, reg_idx] = weighted[:, reg_idx] * scale.unsqueeze(1)
            extra_reg_scaled = -extra_reg_pen * scale
        else:
            scale = m_cap / (m_cap + x)
            weighted = weighted.clone()
            weighted[:, reg_idx] = weighted[:, reg_idx] * scale.unsqueeze(1)
            extra_reg_scaled = -extra_reg_pen * scale

        if is_v5:
            # Add uncompressed hard kinematic pose invariants (height wall + 3D shank + anti-obtuse front-hind splay),
            # constant sigma^2 smoothness curvature, and overspeed brake OUTSIDE reg_soft_cap!
            uncomp_ar_w = float(cfg.get("uncomp_action_rate_w", 0.0))
            extra_reg_scaled = extra_reg_scaled - uncomp_pose_pen + uncomp_ar_w * act_rate_term
            if float(cfg.get("overspeed_w", 0.0)) != 0.0:
                ov_w = float(cfg.get("overspeed_w", 0.0))
                ov_th = float(cfg.get("overspeed_thresh", 0.78))
                extra_reg_scaled = extra_reg_scaled + ov_w * (planar_spd - ov_th).clamp_min(0.0).square()
        # gated strictly on genuine forward potential progress + body forward speed so baby steps are never taxed
        # and stationary tapping gets 0.
        if air_time_w != 0.0:
            air_target = float(cfg.get("feet_air_time_target", 0.12))
            td_f = touchdown.float()
            if is_v5:
                air_rew = ((air_before_td - air_target).clamp(0.0, 0.16) * td_f).sum(dim=1)
                air_gate = (d_prog > 0.15) & (vx_yaw > 0.18) & (~near)
            else:
                v_xy = state.root_lin_vel_w[:, :2]
                goal_dir = xy / dist.clamp_min(1e-6).unsqueeze(1)
                v_toward = (v_xy * goal_dir).sum(dim=1)
                air_rew = ((air_before_td - air_target).clamp(-air_target, air_target) * td_f).sum(dim=1)
                air_gate = (v_toward > 0.20) & (dist > 0.60)
            air_bonus = torch.where(air_gate, air_time_w * air_rew, torch.zeros_like(air_rew))
        else:
            air_bonus = torch.zeros(self.n, device=self.device, dtype=torch.float32)

        # HAA joint telemetry + optional measured soft-limit penalty
        h = self.haa_ids_t
        q_raw_h = q_raw[:, h]
        q_cmd_h = q_cmd[:, h]
        q_meas_h = state.joint_pos[:, h]
        phys_lo = state.joint_pos_limits[:, h, 0]
        phys_hi = state.joint_pos_limits[:, h, 1]
        if abs(self.haa_soft_factor - 0.95) < 1e-6:
            soft_lo = state.soft_joint_pos_limits[:, h, 0]
            soft_hi = state.soft_joint_pos_limits[:, h, 1]
        else:
            soft_lo, soft_hi = compute_soft_limits_from_factor(phys_lo, phys_hi, self.haa_soft_factor)
        tol_1deg = math.radians(1.0)
        near_phys_lo_1deg = q_meas_h <= (phys_lo + tol_1deg)
        near_phys_hi_1deg = q_meas_h >= (phys_hi - tol_1deg)
        near_soft = (q_meas_h <= soft_lo) | (q_meas_h >= soft_hi)
        out_of_limits = (-(q_meas_h - soft_lo).clamp_max(0.0)) + (q_meas_h - soft_hi).clamp_min(0.0)
        haa_limit_excess = out_of_limits.sum(dim=1)

        if self.haa_limit_pen_weight > 0.0:
            haa_limit_pen = -self.haa_limit_pen_weight * haa_limit_excess
        else:
            haa_limit_pen = torch.zeros_like(haa_limit_excess)

        alive_w = float(cfg.get("alive_bonus_w", 0.0))
        if alive_w != 0.0:
            # Exact Bellman survival baseline on non-terminal steps (result == 0 or timeout == 4);
            # success (result == 1) carries its continuation b0/(1-gamma) inside weights[0],
            # and failure (result == 2 or 3) receives 0.0!
            alive_bonus = alive_w * ((result == 0) | (result == 4)).float()
        else:
            alive_bonus = torch.zeros_like(haa_limit_excess)

        total_reward = weighted.sum(dim=1) + extra_reg_scaled + air_bonus + haa_limit_pen + alive_bonus

        return result, dict(
            reward_terms=weighted, reward=total_reward, distance=dist,
            height_error=z_error, near=near, stable=stable, foot_force=foot,
            foot_force_z=foot_force_z, nonfoot_hits=nonfoot_hits,
            base_force=max_force[:, self.base_sensor],
            raw_reg_x=x, reg_scale=scale,
            extra_reg_scaled=extra_reg_scaled, air_bonus=air_bonus, alive_bonus=alive_bonus,
            q_raw_haa=q_raw_h, q_cmd_haa=q_cmd_h, q_meas_haa=q_meas_h,
            near_phys_lo_1deg=near_phys_lo_1deg, near_phys_hi_1deg=near_phys_hi_1deg,
            near_soft=near_soft, haa_limit_excess=haa_limit_excess, haa_limit_pen=haa_limit_pen,
            v5_diag=v5_diag,
        )
