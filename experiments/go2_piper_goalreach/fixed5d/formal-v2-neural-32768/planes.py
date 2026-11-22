"""Update one persistent collision body per slot before the original robot reset."""
import torch
from isaaclab.utils.math import quat_apply, quat_mul

from protocol import PLATFORM_METRES, WIDE


def slope_quat(alpha_deg):
    half = torch.deg2rad(alpha_deg) * .5
    zero = torch.zeros_like(half)
    return torch.stack((half.cos(),zero,-half.sin(),zero),dim=-1)


class PersistentPlanes:
    def __init__(self, env):
        self.env = env
        self.body = env.scene['episode_plane']
        self.view = self.body.root_physx_view
        self.transforms = self.view.get_transforms().clone()
        assert list(self.view.prim_paths)==[f'{path}/EpisodePlane' for path in env.scene.env_prim_paths]
        slot_ids = torch.arange(env.num_envs,device='cpu')
        self.view.set_masses(torch.zeros_like(self.view.get_masses()),slot_ids)
        self.view.set_inertias(torch.zeros_like(self.view.get_inertias()),slot_ids)
        assert torch.count_nonzero(self.view.get_masses())==0
        assert torch.count_nonzero(self.view.get_inertias())==0
        self.velocities = torch.zeros_like(self.view.get_velocities())
        self.material = self.view.get_material_properties().clone()
        assert self.material.device.type == 'cpu' and self.material.shape == (env.num_envs,1,3)
        assert torch.all(self.material[..., :2]==1.) and torch.isfinite(self.material[...,2]).all()
        self.keys = [None]*env.num_envs
        self.counters = dict(batches=0, assignments=0, distinct_collision_bodies=env.num_envs)

    def assign(self, ids, mu, alpha, rotations):
        origin = self.env.scene.env_origins[ids]
        offset = torch.zeros_like(origin)
        offset[:,2] = -.5*PLATFORM_METRES[2]
        local = quat_apply(rotations,offset)
        center = origin + local
        pose = torch.cat((center,rotations),dim=-1)
        host_ids = ids.cpu()
        slot_ids = host_ids.tolist()
        condition = list(zip(mu.tolist(),alpha.tolist()))
        # Zero inverse mass and inertia keep the solver pose fixed under contact.
        pose_ids = host_ids.to(self.transforms.device)
        self.transforms[pose_ids] = pose[:,[0,1,2,4,5,6,3]].to(self.transforms.device)
        self.view.set_transforms(self.transforms,pose_ids)
        self.view.set_velocities(self.velocities,host_ids.to(self.velocities.device))
        self.material[host_ids,:,:2] = mu.cpu().to(self.material.dtype).reshape(-1,1,1)
        self.view.set_material_properties(self.material,host_ids)
        for slot,physical in zip(slot_ids,condition):
            self.keys[slot] = physical
        self.counters['batches'] += 1
        self.counters['assignments'] += len(ids)


def orient_births(runtime):
    runtime.mu = torch.as_tensor(runtime.pool['mu'],device=runtime.env.device)
    runtime.alpha = torch.as_tensor(runtime.pool['alpha_deg'],device=runtime.env.device)
    rotations = slope_quat(runtime.alpha.to(runtime.initial_root_poses.dtype))
    runtime.initial_ee = quat_apply(rotations,runtime.initial_ee)
    root = runtime.initial_root_poses.clone()
    root[:,:3] = quat_apply(rotations,root[:,:3])
    root[:,3:] = quat_mul(rotations,root[:,3:])
    runtime.initial_root_poses = root
    runtime.plane_rotations = rotations


def configure_planes(cfg, *, scene_queries=True, probe=False):
    from isaaclab import sim as sim_utils
    from isaaclab.assets import RigidObjectCfg

    material = cfg.scene.terrain.physics_material
    assert material.static_friction == material.dynamic_friction == 1.
    assert material.friction_combine_mode == 'multiply'
    cfg.scene.terrain = None
    # Clone independent material handles for the CPU solver. The probe checks that
    # writes to one live slot do not change its neighbors' solver-side values.
    cfg.scene.replicate_physics = False
    cfg.scene.filter_collisions = False
    if WIDE:
        cfg.scene.env_spacing = 11.
    cfg.scene.episode_plane = RigidObjectCfg(prim_path='{ENV_REGEX_NS}/EpisodePlane',
        spawn=sim_utils.CuboidCfg(size=PLATFORM_METRES,physics_material=material,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=False,disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.),
            collision_props=sim_utils.CollisionPropertiesCfg()),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.,0.,-.5*PLATFORM_METRES[2])))
    if probe:
        cfg.scene.friction_witness = RigidObjectCfg(prim_path='{ENV_REGEX_NS}/FrictionWitness',
            spawn=sim_utils.CuboidCfg(size=(.25,.25,.025),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.),
                physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.,dynamic_friction=1.,
                                                              friction_combine_mode='multiply'),
                collision_props=sim_utils.CollisionPropertiesCfg(),activate_contact_sensors=True),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(1.7,1.7,1.)))
        for name,prim,mu,y in (('static_friction_witness','StaticFrictionWitness',.8,-4.15),
                               ('unit_static_witness','UnitStaticWitness',1.,4.15)):
            setattr(cfg.scene,name,RigidObjectCfg(prim_path=f'{{ENV_REGEX_NS}}/{prim}',
                spawn=sim_utils.SphereCfg(radius=.08,rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.),
                    physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=mu,dynamic_friction=mu,
                                                                  friction_combine_mode='multiply'),
                    collision_props=sim_utils.CollisionPropertiesCfg(),activate_contact_sensors=True),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.,y,1.))))
    cfg.sim.enable_scene_query_support = scene_queries
