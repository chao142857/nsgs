"""Measure live support, friction, slot isolation and a frozen walking policy."""
from argparse import ArgumentParser
from hashlib import sha256
import json
import math
from pathlib import Path
from time import monotonic
import traceback

from protocol import ALPHA_BOUNDS_DEG, MU_BOUNDS, file_sha


HISTORICAL_SHA = '9d1a806e38008b6766a0511c4f7e9899af5c299a0f4c07949170e285c05671fe'
HISTORICAL_EMA_CONTENT = 'd1717be1979bf68545a7b2ade53f7f08d9ec0a0591c8567a08254a86bc8b8884'


def panel(source, path, *, flat_reference, wide):
    import numpy as np
    from build_pools import NATIVE_META

    with np.load(source,allow_pickle=False) as upstream:
        safe = upstream['target'][:,2] > abs(upstream['target'][:,0])*np.tan(np.deg2rad(6.))+.002
        originals = []
        for region in range(4):
            candidates = np.flatnonzero((upstream['region']==region)&safe)[::2]
            originals.extend(candidates[np.linspace(4,len(candidates)-5,16,dtype=np.intp)])
        ids = np.asarray(originals,dtype=np.intp)
        angles,mu = np.tile([0.,0.,6.,-6.],64),np.tile([1.,.4,.4,1.],64)
        indices = np.repeat(ids,4)
        if wide:
            extreme=ALPHA_BOUNDS_DEG[-1]
            clearance = upstream['target'][:,2] > abs(upstream['target'][:,0])*math.tan(math.radians(extreme))+.003
            paired = []
            for region in range(4):
                possible = np.flatnonzero((upstream['region']==region)&clearance)[::2]
                paired.extend(possible[[len(possible)//4,3*len(possible)//4]])
            baseline_pool = np.repeat(ids.reshape(4,16)[:,[3,11]].ravel(),4)
            indices = np.r_[baseline_pool,np.repeat(paired,28)]
            angles = np.r_[angles[:32],np.tile([-extreme,-14.,-6.5,0.,6.5,14.,extreme],32)]
            centers=[.475,.625,.775,.925] if MU_BOUNDS[0]==.4 else [.275,.45,.65,.875]
            mu = np.r_[mu[:32],np.tile(np.repeat(centers,7),8)]
        result = {name:upstream[name] if name in NATIVE_META else upstream[name][indices]
                  for name in upstream.files}
    result['task_id'] = result['config_id'] = np.arange(256,dtype='<i4')
    result['target_index'] = indices.astype('<i4')//2
    result['mu'] = np.ones(256,dtype='<f8') if flat_reference else mu
    result['alpha_deg'] = np.zeros(256,dtype='<f8') if flat_reference else angles
    np.savez(path,**result)
    return result


def anonymous_bytes():
    fields = Path('/proc/self/status').read_text().splitlines()
    return int(next(line for line in fields if line.startswith('RssAnon:')).split()[1])*1024


def record_pose_diagnostics(env, rt, out):
    import carb
    from isaacsim.core.simulation_manager import SimulationManager
    from omni.physx import get_physx_interface, get_physx_scene_query_interface
    from pxr import PhysxSchema, UsdPhysics
    from spatial_train import put

    api = get_physx_interface()
    settings = carb.settings.get_settings()
    scene = PhysxSchema.PhysxSceneAPI(env.scene.stage.GetPrimAtPath(env.sim.cfg.physics_prim_path))
    simview = SimulationManager.get_physics_sim_view()
    mode=dict(device=str(simview.device),readback_suppressed=bool(api.is_readback_suppressed()),
        fabric=env.sim.cfg.use_fabric,gpu_dynamics=scene.GetEnableGPUDynamicsAttr().Get(),
        broadphase=str(scene.GetBroadphaseTypeAttr().Get()),update_to_usd=settings.get('/physics/updateToUsd'))
    query = get_physx_scene_query_interface()
    traces=[]
    for step in range(5):
        measured=rt.planes.view.get_transforms()
        for slot in (2,3,32,38):
            origin=env.scene.env_origins[slot].tolist()
            task=int(rt.state.task[slot]); fraction=rt.alpha[task].item()
            x=1.6
            hit=query.raycast_closest((origin[0]+x,origin[1]+.7,origin[2]+2.),(0.,0.,-1.),4.)
            desired=rt.planes.transforms[slot].tolist()
            prim=env.scene.stage.GetPrimAtPath(rt.planes.view.prim_paths[slot])
            traces.append(dict(step=step,slot=slot,config_id=int(rt.pool['config_id'][task]),
                alpha_deg=fraction,request_world_pose=desired,solver_pose=measured[slot].tolist(),
                kinematic_enabled=UsdPhysics.RigidBodyAPI(prim).GetKinematicEnabledAttr().Get(),
                kinematic_target_sent=None,solver_mass=float(rt.planes.view.get_masses()[slot]),
                ray_height_m=float(hit['position'][2]-origin[2]) if hit['hit'] else None,
                expected_ray_height_m=x*math.tan(math.radians(fraction)),
                ray_collider=str(hit['collision']) if hit['hit'] else None))
        if step<4:
            env.sim.step(render=False)
            env.scene.update(env.physics_dt)
    put(out/'pose_diagnostic.json',dict(mode=mode,rows=traces,physics_dt=env.physics_dt))


def verify_contact_points(env, rt, out):
    import torch
    from isaacsim.core.simulation_manager import SimulationManager
    from isaaclab.utils.math import quat_apply
    from spatial_train import put

    slots=(2,3,32,38)
    ids=torch.tensor(slots,device=env.device)
    sv=SimulationManager.get_physics_sim_view()
    probes=[sv.create_rigid_contact_view(f'/World/envs/env_{slot}/FrictionWitness',
                [f'/World/envs/env_{slot}/EpisodePlane'],max_contact_data_count=32) for slot in slots]
    assert all(view.sensor_count==view.filter_count==1 for view in probes)
    origin=env.scene.env_origins[ids]
    row_ids=rt.state.task[ids]
    rotations=rt.plane_rotations[row_ids]
    offset=origin.new_tensor([1.6,1.4,.045]).expand(len(ids),3)
    stance=origin+quat_apply(rotations,offset)
    witness=env.scene['friction_witness']
    witness.write_root_pose_to_sim(torch.cat((stance,rotations),dim=-1),env_ids=ids)
    witness.write_root_velocity_to_sim(stance.new_zeros((len(ids),6)),env_ids=ids)
    normals=quat_apply(rotations,origin.new_tensor([0.,0.,1.]).expand(len(ids),3))
    rows=[]
    for step in range(40):
        env.sim.step(render=False)
        env.scene.update(env.physics_dt)
        for i,view in enumerate(probes):
            forces,points,_,separations,counts,starts=view.get_contact_data(env.physics_dt)
            count=int(counts[0,0]); start=int(starts[0,0])
            if not count:
                continue
            assert 0<=start<32 and 0<count<32 and start+count<=32
            for j in range(start,start+count):
                point=points[j]
                rows.append(dict(step=step,slot=slots[i],config_id=int(rt.pool['config_id'][row_ids[i]]),
                    point_world_m=point.tolist(),force_n=float(forces[j]),separation_m=float(separations[j]),
                    residual_to_requested_plane_m=abs(float((point-origin[i])@normals[i])),
                    residual_to_flat_plane_m=abs(float(point[2]-origin[i,2]))))
    good={row['slot'] for row in rows if row['force_n']>1. and abs(row['separation_m'])<.005 and
          row['residual_to_requested_plane_m']<.005 and row['residual_to_flat_plane_m']>.05}
    put(out/'contact_diagnostic.json',dict(rows=rows,support_verified_slots=sorted(good)))
    assert good==set(slots),good
    assert torch.allclose(rt.planes.view.get_transforms()[ids],rt.planes.transforms[ids],rtol=0.,atol=1.e-5)


def material_bindings(stage, body_path):
    from pxr import PhysxSchema, Usd, UsdPhysics, UsdShade

    root=stage.GetPrimAtPath(body_path)
    assert root.IsValid(),body_path
    colliders=[]
    for prim in Usd.PrimRange(root,Usd.TraverseInstanceProxies()):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        material,relationship=UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial('physics')
        path=str(material.GetPath()) if material else None
        if material:
            mat=material.GetPrim()
            mode=PhysxSchema.PhysxMaterialAPI(mat).GetFrictionCombineModeAttr().Get()
            usd=UsdPhysics.MaterialAPI(mat)
            coefficients=[usd.GetStaticFrictionAttr().Get(),usd.GetDynamicFrictionAttr().Get()]
        else:
            mode=None
            coefficients=None
        colliders.append(dict(collider_path=str(prim.GetPath()),usd_material_path=path,
            binding_relationship=str(relationship.GetPath()) if relationship else None,
            authored_static_dynamic=coefficients,friction_combine=str(mode) if mode else None,
            binding_kind='bound' if path else 'solver_default_material'))
    return colliders


def solver_evidence(env, rt, ids, *, cycle=0):
    import torch
    from omni.physx import get_physx_scene_query_interface

    query = get_physx_scene_query_interface()
    poses = rt.planes.view.get_transforms().cpu()
    materials = rt.planes.view.get_material_properties()
    wanted = rt.mu[rt.state.task].to(materials.dtype).cpu()
    assert torch.allclose(materials[:,0,0],wanted,atol=1.e-6,rtol=0.)
    assert torch.allclose(materials[:,0,1],wanted,atol=1.e-6,rtol=0.)
    probes = []
    for slot in (0,1,2,3,124,125,126,127,252,253,254,255):
        x0,y0,z0 = env.scene.env_origins[slot].cpu().tolist()
        alpha = torch.deg2rad(rt.alpha[rt.state.task[slot]]).item()
        normal = (-math.sin(alpha),0.,math.cos(alpha))
        for x,y in ((-1.6,.7),(1.6,.7),(1.2,-.8)):
            hit = query.raycast_closest((x0+x,y0+y,z0+2.),(0.,0.,-1.),4.)
            assert hit['hit'] and f'/env_{slot}/EpisodePlane' in str(hit['collision']), (slot,hit)
            expected = x*math.tan(alpha)
            assert abs(hit['position'][2]-z0-expected)<.002,(slot,hit,expected,
                poses[slot].tolist(),rt.planes.transforms[slot].tolist())
            assert max(abs(float(a)-b) for a,b in zip(hit['normal'],normal))<.001,(slot,hit,normal)
            probes.append(dict(slot=slot,x=x,y=y,height=float(hit['position'][2]-z0),
                               normal=list(map(float,hit['normal'])),mu=float(materials[slot,0,1])))
    target = rt.targets[rt.state.task] + env.scene.env_origins
    assert torch.equal(rt.state.target,target)
    return dict(cycle=cycle,probes=probes,solver_poses=poses[ids[:4]].tolist())


def friction_evidence(env, rt, out, *, wide):
    import torch
    from pilot_conditions import FootMaterials
    from spatial_train import put

    body = env.scene['friction_witness']
    origins = env.scene.env_origins
    ids = torch.arange(env.num_envs,device=env.device)
    pose = origins.new_zeros((len(ids),7))
    pose[:,:3] = origins
    pose[:,1] += 3.18
    pose[:,2] += .032
    pose[:,3] = 1.
    body.write_root_pose_to_sim(pose,env_ids=ids)
    speed = pose.new_zeros((len(ids),6))
    body.write_root_velocity_to_sim(speed,env_ids=ids)
    for _ in range(30):
        env.sim.step(render=False)
        env.scene.update(env.physics_dt)
    speed[:,1] = 1.4
    body.write_root_velocity_to_sim(speed,env_ids=ids)
    steps = 18
    velocity = []
    for _ in range(steps):
        env.sim.step(render=False)
        env.scene.update(env.physics_dt)
        velocity.append(body.data.root_lin_vel_w[:,1].clone())
    sample = torch.stack(velocity)
    measured = (sample[2]-sample[-1])/((steps-3)*env.physics_dt*9.81)
    paired = [measured[i:32:4].median().item() for i in (0,1)]
    assert abs(paired[0]-1.)<.23 and abs(paired[1]-.4)<.15 and paired[0]>1.6*paired[1],paired
    feet = FootMaterials(env)
    native = feet.native[:,feet.indices,:2]
    assert torch.equal(native,feet.view.get_material_properties()[:,feet.indices,:2])
    robot=env.scene['robot']
    counts=[robot._physics_sim_view.create_rigid_body_view(path).max_shapes for path in feet.view.link_paths[0]]
    bindings={}
    for name in ('FL_foot','FR_foot','RL_foot','RR_foot'):
        link=robot.body_names.index(name)
        path=feet.view.link_paths[0][link]
        begin=sum(counts[:link]); end=begin+counts[link]
        measured=feet.native[:,begin:end,:2]
        collisions=material_bindings(env.scene.stage,path)
        bindings[name]=dict(link_path=path,solver_shape_indexes=list(range(begin,end)),colliders=collisions,
            solver_static_dynamic_by_slot=measured.tolist())
    sources={name:material_bindings(env.scene.stage,asset.root_physx_view.prim_paths[0]) for name,asset in (
        ('ground',rt.planes.body),('dynamic_box',env.scene['friction_witness']),
        ('calibrated_sphere',env.scene['static_friction_witness']),('unit_sphere',env.scene['unit_static_witness']))}
    material=dict(slider_effective_dynamic_mu=paired,friction_witness_material=[1.,1.],
                solver_foot_material_combine_by_foot=bindings,
                solver_sliders_and_ground_material_bindings=sources,
                foot_shape_index_order=feet.indices,
                robot_foot_material_ranges=[native.amin(dim=(0,1)).tolist(),native.amax(dim=(0,1)).tolist()],
                robot_effective_mu_at_ground_min=(native[0]*MU_BOUNDS[0]).tolist(),
                robot_effective_mu_at_ground_max=(native[0]*MU_BOUNDS[1]).tolist())
    put(out/'dynamic_friction_and_feet.json',material)
    assert all(row['colliders'] and len(row['colliders'])==len(row['solver_shape_indexes']) for row in bindings.values()),[
        (name,row['link_path'],len(row['colliders']),len(row['solver_shape_indexes'])) for name,row in bindings.items()]
    assert all(len(rows)==1 and rows[0]['friction_combine']=='multiply' for rows in sources.values()),sources
    if wide:
        material['slider_static_threshold']=static_witness(env,rt,out)
    return material


def flat_policy_evaluation(env, wrapped, runtime, policy, args, count, actor_sha):
    import numpy as np
    import torch
    from spatial_train import put

    finished=[]
    paired_births=[]
    default_material=None
    if args.save_flat_birth:
        from pxr import PhysxSchema, UsdShade

        scene=env.scene.stage.GetPrimAtPath(env.sim.cfg.physics_prim_path)
        material,relationship=UsdShade.MaterialBindingAPI(scene).ComputeBoundMaterial('physics')
        assert material and relationship
        mode=PhysxSchema.PhysxMaterialAPI(material.GetPrim()).GetFrictionCombineModeAttr().Get()
        assert mode in ('average','min','multiply','max')
        default_material=dict(path=str(material.GetPath()),binding=str(relationship.GetPath()),friction_combine=str(mode))
    with torch.inference_mode():
        for offset in range(0,count,env.num_envs):
            tasks=list(range(offset,min(count,offset+env.num_envs)))
            forced=tasks+[tasks[-1]]*(env.num_envs-len(tasks))
            if hasattr(runtime,'planes'):
                runtime.discard_at_evaluation_boundary()
                runtime.eval_repeat=torch.as_tensor(forced,device=env.device)
            runtime.forced_ids=forced
            wrapped.reset()
            obs=wrapped.get_observations()
            assert obs['policy'].shape==(256,201)
            if args.save_flat_birth:
                from pilot_conditions import FootMaterials

                robot=env.scene['robot']
                view=robot.root_physx_view
                feet=FootMaterials(env)
                size=len(tasks)
                values=zip(tasks,env.scene.env_origins[:size].tolist(),runtime.state.target[:size].tolist(),
                    view.get_root_transforms()[:size].tolist(),view.get_dof_positions()[:size].tolist(),
                    view.get_material_properties()[:size,feet.indices].tolist())
                for task,origin,target,root,joints,material in values:
                    assert int(runtime.pool['task_id'][task])==task
                    paired_births.append(dict(task=task,origin=origin,target_world=target,root_solver_xyz_xyzw=root,
                                               joint_solver_positions=joints,foot_solver_material=material,
                                               foot_shape_indexes=feet.indices))
            generations=runtime.state.generation.tolist()
            got={}
            for _ in range(500):
                obs,_,_,_=wrapped.step(policy(obs.to('cuda:0')).to(env.device))
                for row in runtime.last_rows:
                    slot=row['env_id']
                    if slot<len(tasks) and row['generation']==generations[slot]:
                        assert row['task_id']==tasks[slot] and slot not in got
                        got[slot]=row
                if len(got)==len(tasks):
                    break
            assert len(got)==len(tasks)
            finished.extend(got[i] for i in range(len(tasks)))
    assert [row['task_id'] for row in finished]==list(range(count))
    assert not any(runtime.sampler.completed_counts)
    success=np.array([row['success'] for row in finished],dtype=np.int8)
    region=runtime.pool['region']
    cells=runtime.pool['cell_id']
    cell_counts=np.bincount(cells,minlength=42)
    assert len(cell_counts)==42 and cell_counts.tolist()==[12]*42
    result=dict(passed=True,baseline=args.baseline,ppo_updates=0,n=count,flat_panel=True,
        native_student_observation=201,successes=int(success.sum()),actor_sha256=actor_sha,
        success_by_task=success.tolist(),walk_wins=int(success[region==3].sum()),
        coverage=float((np.bincount(cells,weights=success,minlength=42)/cell_counts).mean()))
    put(args.out/'rows.json',finished)
    if args.save_flat_birth:
        assert len(paired_births)==count
        put(args.out/'flat_birth_material.json',paired_births)
        put(args.out/'scene_default_material.json',default_material)
    put(args.out/'passed.json',result)


def static_witness(env,rt,out):
    import torch
    from isaaclab.utils.math import quat_apply
    from omni.physx import get_physx_scene_query_interface
    from pxr import PhysxSchema
    from planes import slope_quat
    from spatial_train import put

    mu=.40
    calibration=.80
    critical=math.degrees(math.atan(mu*calibration))
    tests=[0.,16.,20.,-16.,-20.]
    ids=torch.arange(20,device=env.device)
    angles=env.scene.env_origins.new_tensor(tests).repeat_interleave(4)
    friction=torch.full_like(angles,mu)
    rotation=slope_quat(angles)
    rt.planes.assign(ids,friction,angles,rotation)
    ground=rt.planes.view.get_material_properties()[ids].clone()
    assert torch.allclose(ground[:,:,:2],ground.new_full((20,1,2),mu),rtol=0.,atol=1.e-6)
    bodies=[]
    for name,witness_mu,y in (('static_friction_witness',calibration,-3.6),('unit_static_witness',1.,3.75)):
        body=env.scene[name]
        view=body.root_physx_view
        materials=view.get_material_properties()[ids].clone()
        assert torch.allclose(materials[:,:,:2],materials.new_full((20,1,2),witness_mu),rtol=0.,atol=1.e-6)
        inertia=view.get_inertias().clone()
        fixed_inertia=inertia.clone(); fixed_inertia[ids]=0.
        view.set_inertias(fixed_inertia,ids.cpu())
        assert torch.count_nonzero(view.get_inertias()[ids])==0
        mass=view.get_masses()[ids].clone()
        assert torch.allclose(mass,torch.ones_like(mass),rtol=0.,atol=1.e-6)
        origin=env.scene.env_origins[ids]
        local=origin.new_tensor([0.,y,.0801]).expand(len(ids),3)
        contact=origin+quat_apply(rotation,local)
        pose=torch.cat((contact,rotation),dim=-1)
        body.write_root_pose_to_sim(pose,env_ids=ids)
        body.write_root_velocity_to_sim(pose.new_zeros((len(ids),6)),env_ids=ids)
        release=view.get_velocities()[ids].clone()
        assert torch.count_nonzero(release)==0
        bodies.append(dict(body=body,view=view,contact=contact,material=materials,mass=mass,
                           release=release,inertia=inertia))
    down=quat_apply(rotation,rotation.new_tensor([1.,0.,0.]).expand(len(ids),3))
    down*=torch.where(angles>=0,-1.,1.)[:,None]
    steps=round(1./env.physics_dt)
    stride=round(.1/env.physics_dt)
    trace=[]
    ray_heights = []
    query=get_physx_scene_query_interface()
    for step in range(steps):
        env.sim.step(render=False)
        env.scene.update(env.physics_dt)
        if not step:
            for slot in range(0,20,4):
                x,y,z=env.scene.env_origins[slot].tolist()
                hit=query.raycast_closest((x+1.6,y-.7,z+2.),(0.,0.,-1.),4.)
                expected=1.6*math.tan(math.radians(angles[slot].item()))
                assert hit['hit'] and f'/env_{slot}/EpisodePlane' in str(hit['collision'])
                assert abs(float(hit['position'][2]-z)-expected)<.002
                ray_heights.append(dict(slot=slot,expected_height_m=expected,actual_height_m=float(hit['position'][2]-z),
                                        normal=list(map(float,hit['normal']))))
        if (step+1)%stride==0:
            displacement=torch.cat([((r['body'].data.root_pos_w[ids]-r['contact'])*down).sum(1) for r in bodies])
            speed=torch.cat([(r['body'].data.root_lin_vel_w[ids]*down).sum(1) for r in bodies])
            trace.append(dict(time_s=(step+1)*env.physics_dt,
                signed_downslope_displacement_m=displacement.reshape(10,4).tolist(),
                signed_downslope_speed_m_s=speed.reshape(10,4).tolist()))
    assert len(trace)>=9
    flat=ids[:4]
    for index,r in enumerate(bodies):
        velocities=rotation.new_zeros((4,6))
        velocities[:,1]=1.2 if index else -1.2
        r['body'].write_root_velocity_to_sim(velocities,env_ids=flat)
    moving=[]
    for _ in range(16):
        env.sim.step(render=False)
        env.scene.update(env.physics_dt)
        moving.append(torch.stack([r['body'].data.root_lin_vel_w[flat,1].abs() for r in bodies]))
    dyn_mu=((moving[1]-moving[-1])/(14*env.physics_dt*9.81)).tolist()
    scene=PhysxSchema.PhysxSceneAPI(env.scene.stage.GetPrimAtPath(env.sim.cfg.physics_prim_path))
    result=dict(ground_mu=mu,calibration_slider_material_mu=calibration,unit_slider_material_mu=1.,
        predicted_calibration_angle_deg=critical,predicted_unit_angle_deg=math.degrees(math.atan(mu)),
        test_angles_deg=tests+tests,solver_witness_static_dynamic=[r['material'][:,:,:2].tolist() for r in bodies],
        solver_witness_mass=[r['mass'].tolist() for r in bodies],
        solver_ground_static_dynamic=ground[:,:,:2].tolist(),friction_model=str(scene.GetFrictionTypeAttr().Get()),
        single_sphere_contact_with_zero_angular_inverse_inertia=True,
        ray_heights=ray_heights,initial_velocity=[r['release'].tolist() for r in bodies],
        effective_dynamic_mu_after_static_trial=dyn_mu,observation_seconds=1.,traces=trace)
    put(out/'static_friction_diagnostic.json',result)
    lower=(0,1,3,5,6,7,8,9)
    upper=(2,4)
    for group in lower:
        assert max(abs(v) for v in trace[-1]['signed_downslope_displacement_m'][group])<.03,trace[-1]
    for group in upper:
        assert min(trace[-1]['signed_downslope_displacement_m'][group])>.05,trace[-1]
        assert all(min(row['signed_downslope_speed_m_s'][group])>.06 for row in trace[-3:]),trace[-3:]
    tasks=rt.state.task[ids]
    rt.planes.assign(ids,rt.mu[tasks],rt.alpha[tasks],rt.plane_rotations[tasks])
    for r in bodies:
        r['view'].set_inertias(r['inertia'],ids.cpu())
    return result


def run(args):
    import numpy as np
    import torch
    import warp as wp
    from goalreach import mdp
    from goalreach.config import agent_cfg, build_cfg, make_env
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner
    from spatial_runtime import configure
    from spatial_train import actor_digest, put

    assert file_sha(args.checkpoint)==HISTORICAL_SHA
    wp.config.kernel_cache_dir=str(args.out/'warp_cache')
    dataset=args.out/'representative_panel.npz'
    if args.flat_panel:
        with np.load(args.development,allow_pickle=False) as source:
            pool=dict(source)
        assert len(pool['task_id'])==504
        pool.update(mu=np.ones(504,dtype='<f8'),alpha_deg=np.zeros(504,dtype='<f8'))
        np.savez(dataset,**pool)
    else:
        pool=panel(args.development,dataset,flat_reference=args.baseline,wide=args.wide)
    cfg=build_cfg(dataset,args.out/'eval',256,seed=43,reward='R0',sampler='evaluation',device='cpu',timeout_penalty=-50.)
    cfg.goalreach.update(success_criterion='hold_only',legacy_jsonl=False,spatial_method='evaluation')
    if not args.baseline:
        configure(cfg,probe=not args.flat_panel)
    elif args.wide:
        cfg.scene.env_spacing=11.
    env=make_env(cfg)
    try:
        agent=agent_cfg(43,.3,.001)
        wrapped=RslRlVecEnvWrapper(env,clip_actions=agent.clip_actions)
        runner=OnPolicyRunner(wrapped,agent.to_dict(),log_dir=None,device='cuda:0')
        stored=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
        assert stored['completed_updates']==8000
        runner.alg.actor.load_state_dict(stored['ema'])
        actual=runner.alg.actor.state_dict()
        ema_content=sha256(b''.join(value.detach().cpu().contiguous().numpy().tobytes()
            for _,value in sorted(actual.items()))).hexdigest()
        assert ema_content==HISTORICAL_EMA_CONTENT
        historical_actor=actor_digest(runner.alg.actor)
        del stored
        rt=mdp.runtime(env)
        policy=runner.get_inference_policy(device='cuda:0')
        if args.flat_panel:
            flat_policy_evaluation(env,wrapped,rt,policy,args,504,historical_actor)
            return
        ids=torch.arange(256,device=env.device)
        with torch.inference_mode():
            if not args.baseline:
                rt.discard_at_evaluation_boundary()
                rt.forced_ids=ids.tolist()
                wrapped.reset()
                record_pose_diagnostics(env,rt,args.out)
                verify_contact_points(env,rt,args.out)
                first=solver_evidence(env,rt,ids)
                friction=friction_evidence(env,rt,args.out,wide=args.wide)
                same_world=rt.state.target.clone()
                other=ids[::4]
                rest=ids[ids.remainder(4)!=0]
                before=rt.planes.view.get_material_properties()[rest].clone()
                started=monotonic()
                memory=[]
                for cycle in range(100):
                    redrawn=other+(cycle+1)%4
                    if args.wide:
                        redrawn[8:]=other[8:]
                    rt.forced_ids=redrawn.tolist()
                    mdp.reset_robot(env,other)
                    env.sim.step(render=False)
                    env.scene.update(env.physics_dt)
                    if cycle%10==9:
                        memory.append(anonymous_bytes())
                elapsed=monotonic()-started
                assert torch.equal(rt.state.target,same_world)
                assert torch.equal(rt.planes.view.get_material_properties()[rest],before)
                assert memory[-1]-min(memory[2:])<512*2**20,memory
                last=solver_evidence(env,rt,ids,cycle=100)
                put(args.out/'physical_readback.json',dict(first=first,last=last,friction=friction,
                    resets=100*len(other),unchanged_neighbors=len(rest),world_targets_bitwise_fixed=True,
                    bytes_anonymous=memory,resets_per_s=100*len(other)/elapsed,
                    rigid_bodies=rt.planes.counters['distinct_collision_bodies']))
                rt.discard_at_evaluation_boundary()
                rt.eval_repeat=ids
            rt.forced_ids=ids.tolist()
            wrapped.reset()
            obs=wrapped.get_observations()
            assert obs['policy'].shape==(256,201)
            generations=rt.state.generation.tolist()
            seen={}
            stance=[]
            for control in range(500):
                obs,_,_,_=wrapped.step(policy(obs.to('cuda:0')).to(env.device))
                if not args.baseline and control<50:
                    robot=env.scene['robot']
                    sensor=env.scene.sensors['contact_forces']
                    feet=[i for i,name in enumerate(robot.body_names) if name.endswith('_foot')]
                    feet_sensor=[i for i,name in enumerate(sensor.body_names) if name.endswith('_foot')]
                    from isaaclab.utils.math import quat_apply
                    normals=quat_apply(rt.plane_rotations[ids],rt.targets.new_tensor([0.,0.,1.]).expand(256,3))
                    force=(sensor.data.net_forces_w[:,feet_sensor]*normals[:,None]).sum(2)
                    vel=robot.data.body_lin_vel_w[:,feet]
                    tangent=vel-(vel*normals[:,None]).sum(2,keepdim=True)*normals[:,None]
                    stance.append(torch.stack(((force>3.).sum(1),
                        torch.where(force>3.,tangent.norm(dim=-1),0.).amax(1),
                        robot.data.root_lin_vel_w[:,:2].norm(dim=1)),dim=1).cpu())
                for row in rt.last_rows:
                    slot=row['env_id']
                    if row['generation']==generations[slot]:
                        assert row['task_id']==slot and slot not in seen
                        seen[slot]=row
                if len(seen)==256:
                    break
        assert len(seen)==256 and not any(rt.sampler.completed_counts)
        basic=32 if args.wide else 256
        results=np.array([bool(seen[i]['success']) for i in range(256)])
        wins=results[:basic].reshape(-1,4)
        walk=pool['region'][:basic:4]==3
        strata=None
        if args.wide and not args.baseline:
            recorded=torch.stack(stance).numpy()[:,32:]
            stance_mask=(recorded[:,:,0]>=2)&(recorded[:,:,2]<=.15)
            stance_result=np.divide((recorded[:,:,1]*stance_mask).sum(0),stance_mask.sum(0),
                out=np.zeros(224),where=stance_mask.any(0))
            strata=dict(mu=pool['mu'][32:60].tolist(),alpha_deg=pool['alpha_deg'][32:60].tolist(),
                target_pairs_per_stratum=8,successes=results[32:].reshape(8,28).sum(0).tolist(),
                controls_with_support=stance_mask.reshape(-1,28).sum(0).tolist(),
                mean_max_support_foot_slip_m_s_per_target=stance_result.reshape(8,28).tolist())
        put(args.out/'rows.json',[seen[i] for i in range(256)])
        put(args.out/'passed.json',dict(passed=True,baseline=args.baseline,ppo_updates=0,n=256,
            wins=wins.sum(axis=0).tolist(),walk_wins=wins[walk].sum(axis=0).tolist(),
            flat_successes=wins[:,0].tolist(),historical_actor_sha256=historical_actor,
            total_successes=int(results.sum()),wide_risk_strata=strata,
            native_student_observation=201,physical_readback=(not args.baseline)))
    except BaseException:
        put(args.out/'error.json',dict(traceback=traceback.format_exc()))
        raise
    finally:
        env.close()


if __name__=='__main__':
    from isaaclab.app import AppLauncher
    cli=ArgumentParser(description=__doc__)
    cli.add_argument('out',type=Path)
    cli.add_argument('--development',type=Path,required=True)
    cli.add_argument('--checkpoint',type=Path,required=True)
    cli.add_argument('--kit-args',default='')
    cli.add_argument('--baseline',action='store_true')
    cli.add_argument('--wide',action='store_true')
    cli.add_argument('--flat-panel',action='store_true')
    cli.add_argument('--save-flat-birth',action='store_true')
    args=cli.parse_args()
    args.out.mkdir()
    app=AppLauncher(headless=True,device='cuda:0',enable_cameras=False,livestream=0,kit_args=args.kit_args).app
    try:
        run(args)
    except BaseException:
        if not (args.out/'error.json').is_file():
            (args.out/'error.json').write_text(json.dumps(dict(traceback=traceback.format_exc())))
        print(traceback.format_exc(),flush=True)
        raise
    finally:
        app.close(wait_for_replicator=False,skip_cleanup=True)
