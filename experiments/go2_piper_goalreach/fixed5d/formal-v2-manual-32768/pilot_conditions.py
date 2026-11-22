"""Execute the condition attached to each requested native GoalReach task."""
import math

import numpy as np
import torch
from goalreach import mdp
from goalreach.p3s1_grid import cell_ids
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_mul

SLOPE_DEG = 6.0
PLATFORM = (7.8, 7.8, .4)
DECK_Z = 4.0
LOW_FRICTION = (.40, .32)
FEET = ('FL_foot', 'FR_foot', 'RL_foot', 'RR_foot')


def ramp_rotation():
    half_angle = math.radians(SLOPE_DEG) / 2
    return (math.cos(half_angle), 0., -math.sin(half_angle), 0.)


def configure(cfg):
    from isaaclab import sim as sim_utils
    from isaaclab.assets import AssetBaseCfg

    angle = math.radians(SLOPE_DEG)
    material = cfg.scene.terrain.physics_material
    assert material.static_friction == material.dynamic_friction == 1.
    assert material.friction_combine_mode == 'multiply'
    cfg.scene.pilot_ramp = AssetBaseCfg(
        prim_path='{ENV_REGEX_NS}/PilotRamp',
        spawn=sim_utils.CuboidCfg(size=PLATFORM, collision_props=sim_utils.CollisionPropertiesCfg(),
                                 physics_material=material),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(PLATFORM[2] / 2 * math.sin(angle), 0., DECK_Z - PLATFORM[2] / 2 * math.cos(angle)),
            rot=ramp_rotation()))
    cfg.sim.enable_scene_query_support = True
    cfg.events.reset_base.func = reset_robot


class FootMaterials:
    def __init__(self, env):
        self.robot = env.scene['robot']
        self.view = self.robot.root_physx_view
        counts = [self.robot._physics_sim_view.create_rigid_body_view(path).max_shapes
                  for path in self.view.link_paths[0]]
        assert len(counts) == len(self.robot.body_names) and sum(counts) == self.view.max_shapes
        feet = [self.robot.body_names.index(name) for name in FEET]
        assert all(counts[body] > 0 for body in feet)
        self.indices = [shape for body in feet
                        for shape in range(sum(counts[:body]), sum(counts[:body+1]))]
        self.native = self.view.get_material_properties().clone()
        assert self.native.device.type == 'cpu'
        self.current = self.native.clone()
        self.low_values = self.native.clone()
        self.low_values[:, self.indices, :2] = torch.tensor(LOW_FRICTION, dtype=self.native.dtype)
        self.is_low = torch.zeros(env.num_envs, dtype=torch.bool)

    def apply(self, ids, low):
        ids, low = ids.cpu(), low.cpu()
        changed = self.is_low[ids] != low
        if not changed.any():
            return
        apply_ids, requested_low = ids[changed], low[changed]
        self.current[apply_ids] = torch.where(requested_low[:, None, None],
                                              self.low_values[apply_ids], self.native[apply_ids])
        self.view.set_material_properties(self.current, apply_ids)
        measured = self.view.get_material_properties()
        if not torch.allclose(measured[apply_ids], self.current[apply_ids], atol=1e-6, rtol=0.):
            raise RuntimeError('The live foot material differs from the requested task condition')
        self.is_low[apply_ids] = requested_low


def reset_robot(env, env_ids):
    ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    if not len(ids):
        return
    rt = mdp.runtime(env)
    tasks = rt.assign(ids)
    condition = rt.condition[tasks]
    if not hasattr(env, 'pilot_materials'):
        env.pilot_materials = FootMaterials(env)
    env.pilot_materials.apply(ids, condition.remainder(2).bool())
    robot = env.scene['robot']
    root = rt.initial_root_poses[tasks].clone()
    root[:, :3] += env.scene.env_origins[ids]
    q = robot.data.default_joint_pos[ids].clone()
    q[:, rt.action_ids] = rt.initial_joint_positions[tasks]
    robot.write_root_pose_to_sim(root, env_ids=ids)
    robot.write_root_velocity_to_sim(root.new_zeros((len(ids), 6)), env_ids=ids)
    robot.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=ids)
    rt.applied[ids] = torch.stack((tasks, rt.state.generation[ids], condition), dim=1)
    rt.executed.scatter_add_(0, condition, torch.ones_like(condition))


class PilotRuntime(mdp.TaskRuntime):
    def __init__(self, env):
        super().__init__(env)
        self.condition = torch.as_tensor(self.pool['condition_id'], device=env.device, dtype=torch.long)
        assert len(self.condition) == len(self.targets) and self.condition.unique().tolist() == list(range(4))
        assert np.array_equal(cell_ids(self.pool, 16), self.pool['projection_cell_id'])
        if self.p3b and not self.grid_bins:
            self.sampler.set_probabilities(np.ones(len(self.targets), dtype=np.float64))
        rotation = self.targets.new_tensor(ramp_rotation())
        slope = self.condition >= 2
        q = rotation.expand(int(slope.sum()), 4)
        self.targets[slope] = quat_apply(q, self.targets[slope])
        self.initial_ee[slope] = quat_apply(q, self.initial_ee[slope])
        self.initial_root_poses[slope, :3] = quat_apply(q, self.initial_root_poses[slope, :3])
        self.initial_root_poses[slope, 3:] = quat_mul(q, self.initial_root_poses[slope, 3:])
        for xyz in (self.targets, self.initial_ee, self.initial_root_poses):
            xyz[slope, 2] += DECK_Z
        self.plane_quat = self.targets.new_zeros((4, 4))
        self.plane_quat[:2, 0] = 1.
        self.plane_quat[2:] = rotation
        self.gravity_up = self.targets.new_tensor([0., 0., 1.]).expand(env.num_envs, 3)
        self.normal = quat_apply(self.plane_quat, self.gravity_up[:1].expand(4, 3))
        self.applied = torch.full((env.num_envs, 3), -1, device=env.device, dtype=torch.long)
        self.requested = torch.zeros(4, device=env.device, dtype=torch.long)
        self.executed = torch.zeros_like(self.requested)
        self.ends = torch.zeros_like(self.requested)
        self.successes = torch.zeros_like(self.requested)
        self.explicit_resets = torch.zeros_like(self.requested)
        self.edge_ends = torch.zeros_like(self.requested)
        self.foot_ids = [env.scene['robot'].body_names.index(name) for name in FEET]
        self.sensor_ids = [env.scene.sensors['contact_forces'].body_names.index(name) for name in FEET]
        # Peaks (world/surface tilt), sum of support slip, support controls,
        # platform edge exposure, and flat clearance beneath the other deck.
        self.motion = self.targets.new_zeros((env.num_envs, 6))

    def assign(self, ids):
        tasks = super().assign(ids)
        condition = self.condition[tasks]
        self.requested.scatter_add_(0, condition, torch.ones_like(condition))
        self.motion[ids] = 0.
        return tasks

    @torch.inference_mode()
    def discard_at_evaluation_boundary(self):
        live = self.state.task >= 0
        self.explicit_resets.scatter_add_(0, self.applied[live, 2], torch.ones_like(self.applied[live, 2]))
        self.state.reason.zero_()
        self.state.task.fill_(-1)
        self.applied.fill_(-1)

    def tick(self):
        if self.last_tick == self.env.common_step_counter:
            return
        super().tick()
        data = self.env.scene['robot'].data
        forces = self.env.scene.sensors['contact_forces'].data.net_forces_w[:, self.sensor_ids]
        mask = self.applied[:, 2] >= 0
        condition = self.applied[:, 2].clamp_min(0)
        normal = self.normal[condition]
        root_up = quat_apply(data.root_link_quat_w, self.gravity_up)
        world_tilt = torch.rad2deg(torch.acos(root_up[:, 2].clamp(-1., 1.)))
        surface_tilt = torch.rad2deg(torch.acos((root_up * normal).sum(1).clamp(-1., 1.)))
        velocity = data.body_link_lin_vel_w[:, self.foot_ids]
        tangent = velocity - (velocity * normal[:, None]).sum(2, keepdim=True) * normal[:, None]
        support = ((forces * normal[:, None]).sum(2) >= 3.)
        local = data.root_link_pos_w - self.env.scene.env_origins
        local[:, 2] -= DECK_Z * (condition >= 2)
        in_plane = quat_apply_inverse(self.plane_quat[condition], local)
        flat_body_height = (data.body_link_pos_w[:, :, 2] - self.env.scene.env_origins[:, 2:]).amax(1)
        self.motion[:, 0] = torch.maximum(self.motion[:, 0], world_tilt * mask)
        self.motion[:, 1] = torch.maximum(self.motion[:, 1], surface_tilt * mask)
        self.motion[:, 2] += (tangent.norm(dim=2) * support).sum(1) / support.sum(1).clamp_min(1) * mask
        self.motion[:, 3] += support.any(1) & mask
        edge = (condition >= 2) & ((in_plane[:, :2].abs().amax(1) >= 3.75) | (in_plane[:, 2] < -.10))
        self.motion[:, 4] = torch.maximum(self.motion[:, 4], (edge & mask).to(self.motion.dtype))
        self.motion[:, 5] = torch.maximum(self.motion[:, 5], flat_body_height * mask * (condition < 2))

    def finish(self, ids):
        ended = ids[self.state.reason[ids] != 0]
        if not len(ended):
            return
        keys = self.applied[ended]
        tasks = self.state.task[ended]
        condition = self.condition[tasks]
        if not torch.equal(keys, torch.stack((tasks, self.state.generation[ended], condition), 1)):
            raise RuntimeError('Natural ending does not match the executed assignment and generation')
        material = self.env.pilot_materials.is_low[ended.cpu()]
        if not torch.equal(material, condition.remainder(2).bool().cpu()):
            raise RuntimeError('Foot material changed during an in-flight episode')
        success = self.state.reason[ended] == 1
        self.ends.scatter_add_(0, condition, torch.ones_like(condition))
        self.successes.scatter_add_(0, condition, success.long())
        self.edge_ends.scatter_add_(0, condition, self.motion[ended, 4].long())
        motion = self.motion[ended].cpu().tolist()
        bindings = keys.cpu().tolist()
        start = len(self.last_rows)
        super().finish(ids)
        for row, (task, generation, applied), values in zip(self.last_rows[start:], bindings, motion, strict=True):
            assert row['task_id'] == task and row['generation'] == generation
            row.update(condition_id=applied, source_task_id=int(self.pool['source_task_id'][task]),
                       peak_tilt_world_deg=values[0], peak_tilt_plane_deg=values[1],
                       slip_support_m_s=values[2]/max(values[3], 1.), support_controls=int(values[3]),
                       slope_edge=bool(values[4]), flat_peak_body_height_m=values[5])

    def counts(self):
        return {'requested': self.requested.tolist(), 'executed': self.executed.tolist(),
                'ended': self.ends.tolist(), 'successes': self.successes.tolist(),
                'discarded_at_evaluation_boundary': self.explicit_resets.tolist(),
                'edge_exposed_endings': self.edge_ends.tolist()}


def install():
    # This isolated entry uses the native environment, reward, and feedback machinery.
    mdp.TaskRuntime = PilotRuntime
