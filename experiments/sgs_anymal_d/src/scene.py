"""ANYmal-D terrain mesh, goal observations, physics and the shared locomotion reward."""
from pathlib import Path

import navigation as nav
from terrain import mesh, tile

TERRAIN_PRIM = "/World/SGSTerrain"
PHYSICS_DT = .005
CONTROL_SUBSTEPS = 4
LABELS = ("running", "success", "drop", "base_contact", "timeout")


def stage_terrain(stage, combinations, origins):
    import numpy as np
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, Vt

    vertices, indices = [], []
    offset = 0
    for (kind, seed, level), (x, y, z) in zip(combinations, origins, strict=True):
        points, faces = mesh(tile(kind, seed, level).grid)
        world_points = np.asarray(points, dtype=np.float32)
        world_points += np.asarray((x,y,z), dtype=np.float32)
        vertices.append(world_points)
        indices.append(np.asarray(faces, dtype=np.int32).reshape(-1) + offset)
        offset += len(points)
    vertices = np.concatenate(vertices, axis=0)
    indices = np.concatenate(indices, axis=0)
    terrain = UsdGeom.Mesh.Define(stage, Sdf.Path(TERRAIN_PRIM))
    terrain.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(vertices))
    terrain.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(indices))
    terrain.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(indices)//3, 3, dtype=np.int32)))
    terrain.CreateSubdivisionSchemeAttr("none")
    terrain.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(.39, .52, .64)]))
    terrain.CreateDoubleSidedAttr(True)
    UsdPhysics.CollisionAPI.Apply(terrain.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(terrain.GetPrim()).CreateApproximationAttr("none")
    return dict(points=len(vertices), triangles=len(indices)//3, prim=TERRAIN_PRIM)


class GoalScene:
    def __init__(self, combinations, render, *, tile_bank=()):
        import omni.usd
        import torch
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg
        from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
        from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
        from isaaclab.utils import configclass
        from isaaclab_assets.robots.anymal import ANYMAL_D_CFG

        @configclass
        class SceneCfg(InteractiveSceneCfg):
            robot = ANYMAL_D_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
            contacts = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", update_period=0., history_length=4)
            scan = RayCasterCfg(
                prim_path="{ENV_REGEX_NS}/Robot/base",
                offset=RayCasterCfg.OffsetCfg(pos=(0.,0.,3.)),
                ray_alignment="yaw", update_period=nav.CONTROL_DT,
                pattern_cfg=patterns.GridPatternCfg(resolution=.10, size=(1.6,1.)),
                mesh_prim_paths=[TERRAIN_PRIM],
            )
            light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2300.))

        self.torch = torch
        self.device = "cuda:0"
        self.combinations = list(combinations)
        self.n = len(combinations)
        self.sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
            dt=PHYSICS_DT, render_interval=CONTROL_SUBSTEPS, device=self.device, use_fabric=True,
            physx=sim_utils.PhysxCfg(gpu_collision_stack_size=(256*2**20 if self.n==32768 else 2**26),
                                     gpu_max_rigid_patch_count=max(5*2**15, 512*self.n),
                                     gpu_max_rigid_contact_count=max(2**23, 1024*self.n))))
        self.scene = InteractiveScene(SceneCfg(num_envs=self.n, env_spacing=15., filter_collisions=bool(tile_bank),
                                               lazy_sensor_update=False))
        stage = omni.usd.get_context().get_stage()
        tile_bank = tuple(dict.fromkeys(tile_bank))
        bank_origins = [((i%17)*15.-120., (i//17)*15.-120., 0.) for i in range(len(tile_bank))]
        self.bank = dict(zip(tile_bank, bank_origins, strict=True))
        if self.bank:
            self.geometry = stage_terrain(stage, tile_bank, bank_origins)
        else:
            self.geometry = stage_terrain(stage, combinations, self.scene.env_origins.cpu().tolist())
        self.stage = stage
        self.sim.reset()
        self.robot = self.scene["robot"]
        self.contact = self.scene["contacts"]
        self.scan = self.scene["scan"]
        self.origins = self.scene.env_origins.clone()
        self.env_ids = torch.arange(self.n, device=self.device)
        self.ids = [-1] * self.n
        self.steps = torch.zeros(self.n, dtype=torch.int64, device=self.device)
        self.hold = self.steps.clone()
        self.targets = torch.zeros((self.n, 3), device=self.device)
        self.start_locations = self.targets.clone()
        self.last_action = torch.zeros((self.n, 12), device=self.device)
        self.previous_contact = torch.zeros((self.n, 4), dtype=torch.bool, device=self.device)
        self.last_reward = torch.zeros((self.n, 7), device=self.device)
        self.foot_bodies, self.foot_names = self.robot.find_bodies(".*_FOOT")
        self.foot_sensors, sensor_names = self.contact.find_bodies(self.foot_names, preserve_order=True)
        if list(sensor_names) != list(self.foot_names) or len(self.foot_bodies) != 4:
            raise RuntimeError((self.foot_names, sensor_names))
        self.base_sensor = self.contact.find_bodies("base")[0][0]
        self.non_feet = [i for i in range(len(self.contact.body_names)) if i not in self.foot_sensors]
        if self.robot.num_joints != 12 or self.robot.root_physx_view.get_root_transforms().device.type != "cuda":
            raise RuntimeError("Expected the 12-joint CUDA ANYmal-D articulation")
        self.render = render

    def ray_hits(self, positions, offset=2.):
        from isaaclab.utils.warp import raycast_mesh

        sources = positions.clone()
        sources[..., 2] += offset
        directions = self.torch.zeros_like(sources)
        directions[..., 2] = -1.
        hits, _, _, _ = raycast_mesh(sources, directions, self.scan.meshes[TERRAIN_PRIM])
        return hits

    def observations(self):
        from isaaclab.utils.math import quat_apply_inverse

        torch = self.torch
        bot = self.robot.data
        target = self.targets - bot.root_pos_w
        target[:, 2] += nav.BASE_HEIGHT
        goal = quat_apply_inverse(bot.root_quat_w, target)
        scan = self.scan.data.ray_hits_w[:, :, 2]
        height = torch.nan_to_num(bot.root_pos_w[:, 2:3] - scan - nav.BASE_HEIGHT, nan=0., posinf=2., neginf=-2.)
        return torch.cat((bot.root_lin_vel_b, bot.root_ang_vel_b, bot.projected_gravity_b,
                          bot.joint_pos - bot.default_joint_pos, bot.joint_vel, self.last_action, goal, height), dim=1)

    def step(self, action, *, training_iteration=0, after_physics=None):
        torch = self.torch
        if tuple(action.shape) != (self.n, 12):
            raise ValueError(action.shape)
        self.last_action.copy_(action)
        self.robot.set_joint_position_target(self.robot.data.default_joint_pos + nav.ACTION_SCALE * action)
        for substep in range(CONTROL_SUBSTEPS):
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(PHYSICS_DT)
            if after_physics and substep==0:
                after_physics()
        state = self.robot.data
        root = state.root_pos_w
        xy = self.targets[:, :2] - root[:, :2]
        dist = xy.norm(dim=-1)
        z_error = (self.targets[:, 2] + nav.BASE_HEIGHT - root[:, 2]).abs()
        near = (dist <= nav.GOAL_RADIUS) & (z_error <= nav.GOAL_HEIGHT_TOLERANCE)
        self.steps += 1
        force = self.contact.data.net_forces_w.norm(dim=-1)
        max_force = self.contact.data.net_forces_w_history.norm(dim=-1).amax(dim=1)
        foot = force[:, self.foot_sensors]
        foot_contact = foot > nav.MIN_CONTACT_FORCE
        from diagnostics import stable_arrival
        stable = stable_arrival(near, state.projected_gravity_b[:,2],
                                state.root_lin_vel_w[:,:2].norm(dim=-1),
                                state.root_ang_vel_w.norm(dim=-1), foot_contact.sum(dim=1))
        self.hold = torch.where(stable,self.hold+1,0)
        touchdown = foot_contact & ~self.previous_contact
        self.previous_contact.copy_(foot_contact)
        result = torch.zeros(self.n, dtype=torch.int8, device=self.device)
        result[self.steps >= nav.MAX_STEPS] = 4
        result[self.hold >= nav.SUCCESS_HOLD] = 1
        result[max_force[:, self.base_sensor] > nav.BASE_CONTACT_FORCE] = 3
        result[root[:, 2] - self.origins[:, 2] < nav.DROP_HEIGHT] = 2
        forward_xy = state.root_lin_vel_b[:, :2]
        from isaaclab.utils.math import quat_apply_inverse
        goal_vector = torch.column_stack((xy, torch.zeros_like(dist)))
        goal_body = quat_apply_inverse(state.root_quat_w, goal_vector)[:, :2]
        align = ((forward_xy * goal_body).sum(dim=1) /
                 (forward_xy.norm(dim=1).clamp_min(1e-6) * goal_body.norm(dim=1).clamp_min(1e-6))).clamp_min(0.)
        vel_feet = state.body_lin_vel_w[:, self.foot_bodies, 2]
        terms = torch.stack((result == 1, (result == 2) | (result == 3),
                             (state.applied_torque * state.joint_vel).abs().sum(dim=1),
                             (state.joint_pos - state.default_joint_pos).abs().sum(dim=1),
                             (touchdown * vel_feet.square()).sum(dim=1),
                             (max_force[:, self.non_feet] > nav.MIN_CONTACT_FORCE).sum(dim=1), align), dim=1)
        self.last_reward.copy_(terms)
        if not torch.isfinite(root).all() or not torch.isfinite(terms).all():
            raise RuntimeError("Invalid navigation tensors")
        weighted = terms * terms.new_tensor(nav.reward_weights(training_iteration))
        return result, dict(reward_terms=weighted, distance=dist, height_error=z_error, near=near, foot_force=foot,
                            base_force=max_force[:, self.base_sensor],
                            reward=terms @ terms.new_tensor(nav.reward_weights(training_iteration)))

