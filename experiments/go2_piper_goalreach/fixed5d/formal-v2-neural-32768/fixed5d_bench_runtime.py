"""Bounded engineering observation around the signed fixed-plane PPO path."""
import json
import math
from pathlib import Path
from time import monotonic


WARM=32
WINDOW=16
THROUGH=64
CAPACITY_NAMES=('gpu_collision_stack_size','gpu_max_rigid_contact_count','gpu_max_rigid_patch_count',
                'gpu_found_lost_pairs_capacity','gpu_found_lost_aggregate_pairs_capacity',
                'gpu_total_aggregate_pairs_capacity','gpu_heap_capacity','gpu_temp_buffer_capacity')


def cpu_anon():
    fields=Path('/proc/self/status').read_text().splitlines()
    return int(next(line for line in fields if line.startswith('RssAnon:')).split()[1])*1024


def configure_bench(cfg, direct=False, fabric=False):
    from isaaclab.sim import SimulationCfg

    assert not fabric or direct
    cfg.goalreach['benchmark_fabric_defaults']=dict(isaac_lab=bool(SimulationCfg().use_fabric),
                                                   task=bool(cfg.sim.use_fabric))
    cfg.sim.use_fabric=fabric
    if str(cfg.sim.device).startswith('cuda') and not direct:
        from isaaclab.managers import EventTermCfg

        # Request CPU tensor and scene-query readback without changing the scene's GPU solver.
        cfg.sim.device='cpu'
        cfg.events.benchmark_gpu_with_readback=EventTermCfg(
            func=enable_gpu_solver_with_readback,mode='prestartup')


def enable_gpu_solver_with_readback(env, env_ids):
    import carb
    from omni.physx import get_physx_interface
    from pxr import PhysxSchema

    assert not get_physx_interface().is_running()
    scene=PhysxSchema.PhysxSceneAPI(env.scene.stage.GetPrimAtPath(env.sim.cfg.physics_prim_path))
    assert scene
    scene.CreateEnableGPUDynamicsAttr().Set(True)
    scene.CreateBroadphaseTypeAttr().Set('GPU')
    scene.CreateEnableCCDAttr().Set(False)
    carb.settings.get_settings().set_bool('/physics/suppressReadback',False)


def read_births(rt, env):
    import torch

    robot=env.scene['robot']
    view=robot.root_physx_view
    root=view.get_root_transforms()
    expected=rt.initial_root_poses[rt.state.task].clone()
    expected[:,:3]+=env.scene.env_origins
    expected=expected[:,[0,1,2,4,5,6,3]].to(root.device)
    root_error=(root[:,:3]-expected[:,:3]).abs().max().item()
    quat_error=torch.minimum((root[:,3:]-expected[:,3:]).abs().amax(1),
                             (root[:,3:]+expected[:,3:]).abs().amax(1)).max().item()
    joints=view.get_dof_positions()[:,rt.action_ids]
    expected_joints=rt.initial_joint_positions[rt.state.task].to(joints.device)
    joint_error=(joints-expected_joints).abs().max().item()
    assert max(root_error,quat_error,joint_error)<5.e-5,(root_error,quat_error,joint_error)
    assert torch.equal(rt.state.target,rt.targets[rt.state.task]+env.scene.env_origins)
    return dict(slots=env.num_envs,root_max_error_m=root_error,quat_max_error=quat_error,
                joint_max_error_rad=joint_error,world_targets_exact=True)


def effective_fabric(env, expected):
    import carb
    import omni.kit.app

    settings=carb.settings.get_settings()
    manager=omni.kit.app.get_app().get_extension_manager()
    configured=bool(env.sim.cfg.use_fabric)
    enabled=settings.get('/physics/fabricEnabled')
    has_interface=env.sim._fabric_iface is not None
    state=dict(configured=configured,enabled=enabled,has_interface=has_interface,
               defaults=env.cfg.goalreach['benchmark_fabric_defaults'],
               extension=manager.get_enabled_extension_id('omni.physx.fabric'),
               update_transforms=settings.get('/physics/fabricUpdateTransformations'),
               update_velocities=settings.get('/physics/fabricUpdateVelocities'),
               gpu_interop=settings.get('/physics/fabricUseGPUInterop'))
    assert configured==enabled==has_interface==expected,state
    return state


class Evidence:
    def __init__(self, rt, env, wrapped, requested_gpu, direct, fabric, warm):
        import torch
        import carb
        from isaacsim.core.simulation_manager import SimulationManager
        from omni.physx import get_physx_interface
        from pxr import PhysxSchema

        self.torch=torch
        self.rt=rt
        self.env=env
        self.wrapped=wrapped
        self.n=env.num_envs
        self.direct=direct
        self.geometry_steps=(24,384,24*warm)
        scene=PhysxSchema.PhysxSceneAPI(env.scene.stage.GetPrimAtPath(env.sim.cfg.physics_prim_path))
        view=SimulationManager.get_physics_sim_view()
        gpu=scene.GetEnableGPUDynamicsAttr().Get()
        assert gpu==requested_gpu
        self.mode=dict(simulator=str(env.device),simulation_view=str(view.device),gpu_dynamics=gpu,
            broadphase=str(scene.GetBroadphaseTypeAttr().Get()),fabric=env.sim.cfg.use_fabric,
            readback_suppressed=bool(get_physx_interface().is_readback_suppressed()),
            buffers={name:getattr(env.sim.cfg.physx,name,None) for name in CAPACITY_NAMES},
            pose_tensor=str(rt.planes.transforms.device),velocity_tensor=str(rt.planes.velocities.device),
            material_tensor=str(rt.planes.material.device),world_task_tensor=str(rt.state.target.device))
        settings=carb.settings.get_settings()
        self.mode['readback_setting']=settings.get('/physics/suppressReadback')
        self.mode['update_to_usd']=settings.get('/physics/updateToUsd')
        self.mode['effective_fabric']=effective_fabric(env,fabric)
        assert self.mode['readback_suppressed']==direct
        assert not requested_gpu or self.mode['broadphase']=='GPU'
        assert self.mode['material_tensor']=='cpu'
        if direct:
            from fixed5d_bench_native_contacts import NativeContacts

            assert str(view.device)==str(env.device)=='cuda:0'
            self.native_contacts=NativeContacts(rt,env)
        self.neighbor_rows=[]
        self.foot_rows={}
        self.checks=[]
        self.original_assign=rt.planes.assign
        self.original_step=wrapped.step
        self.steps=0
        rt.planes.assign=self.assign
        wrapped.step=self.step

    def assign(self, ids, mu, alpha, rotations):
        if 0<len(ids)<self.n and len(self.neighbor_rows)<8:
            slots=set(ids.cpu().tolist())
            candidates={i+delta for i in slots for delta in (-1,1)
                        if 0<=i+delta<self.n and i+delta not in slots}
            assert candidates
            neighbors=sorted(candidates)[:8]
            view=self.rt.planes.view
            pose=view.get_transforms().cpu()[neighbors].clone()
            mat=view.get_material_properties().cpu()[neighbors].clone()
            self.original_assign(ids,mu,alpha,rotations)
            assert self.torch.equal(view.get_transforms().cpu()[neighbors],pose)
            assert self.torch.equal(view.get_material_properties().cpu()[neighbors],mat)
            self.neighbor_rows.append(dict(control_before=self.steps,reset_slots=len(slots),
                selected_example=sorted(slots)[:8],neighbor_slots=neighbors,mu_min=float(mu.min()),
                mu_max=float(mu.max()),alpha_min=float(alpha.min()),alpha_max=float(alpha.max())))
        else:
            self.original_assign(ids,mu,alpha,rotations)

    def step(self, actions):
        sample=self.steps+1 in (1,2,3,4,8,16,24,48,96,192,384,768)
        if self.direct and sample:
            previous=self.rt.state.generation.clone()
        out=self.original_step(actions)
        self.steps+=1
        if sample:
            self.foot_sample()
            if self.direct:
                self.native_contacts.sample(self.steps,previous)
        if self.steps in self.geometry_steps:
            self.checks.append(self.geometry())
        return out

    def foot_sample(self):
        from isaaclab.utils.math import quat_apply

        rt=self.rt
        sensor=self.env.scene.sensors['contact_forces']
        feet=[i for i,name in enumerate(sensor.body_names) if name.endswith('_foot')]
        assert len(feet)==4
        slope=rt.alpha[rt.state.task]
        normals=quat_apply(rt.plane_rotations[rt.state.task],
                          rt.targets.new_tensor([0.,0.,1.]).expand(self.n,3))
        force=(sensor.data.net_forces_w[:,feet]*normals[:,None]).sum(2)
        real=(force>3.).any(1)
        mu=rt.mu[rt.state.task]
        for name,mask in (('negative_low_mu',(slope<-3)&(mu<.55)),
                          ('negative_high_mu',(slope<-3)&(mu>.85)),
                          ('positive_low_mu',(slope>3)&(mu<.55)),
                          ('positive_high_mu',(slope>3)&(mu>.85))):
            rows=self.torch.where(mask&real)[0]
            if len(rows) and name not in self.foot_rows:
                slot=int(rows[0])
                self.foot_rows[name]=dict(control=self.steps,slot=slot,alpha_deg=float(slope[slot]),
                    mu=float(mu[slot]),foot_normal_forces_n=force[slot].tolist())

    def geometry(self):
        from omni.physx import get_physx_scene_query_interface

        t,rt,env=self.torch,self.rt,self.env
        view=rt.planes.view
        actual=view.get_transforms().cpu()
        assert t.allclose(actual,rt.planes.transforms.cpu(),rtol=0.,atol=5e-5)
        mat=view.get_material_properties().cpu()
        mu=rt.mu[rt.state.task].to(mat.dtype).cpu()
        assert t.allclose(mat[:,0,0],mu,atol=1.e-6,rtol=0.) and t.allclose(mat[:,0,1],mu,atol=1.e-6,rtol=0.)
        assert t.equal(rt.state.target,rt.targets[rt.state.task]+env.scene.env_origins)
        assert t.count_nonzero(view.get_masses())==t.count_nonzero(view.get_inertias())==0
        alpha=rt.alpha[rt.state.task].cpu()
        slots={0,self.n//2,self.n-1,int(alpha.argmin()),int(alpha.argmax())}
        geometry=dict(control=self.steps,pose_max_abs_delta=float((actual-rt.planes.transforms.cpu()).abs().max()),
                      world_targets_exact=True,all_slot_mu_readback=True)
        if self.direct:
            from fixed5d_bench_native_contacts import gpu_plane_intersections

            geometry['solver_pose_intersections']=gpu_plane_intersections(rt,env,actual,slots)
            return geometry
        query=get_physx_scene_query_interface()
        results=[]
        for slot in sorted(slots):
            x,y,z=env.scene.env_origins[slot].cpu().tolist()
            radians=math.radians(float(alpha[slot]))
            for offset in (-1.6,1.6):
                hit=query.raycast_closest((x+offset,y+.7,z+2.),(0.,0.,-1.),4.)
                assert hit['hit'] and f'/env_{slot}/EpisodePlane' in str(hit['collision']),(slot,hit)
                assert abs(hit['position'][2]-z-offset*math.tan(radians))<.002,(slot,hit)
                normal=(-math.sin(radians),0.,math.cos(radians))
                assert max(abs(float(a)-b) for a,b in zip(hit['normal'],normal))<.001
                results.append(dict(slot=slot,alpha_deg=float(alpha[slot]),mu=float(mu[slot]),
                                    dx=offset,height=float(hit['position'][2]-z),normal=list(map(float,hit['normal']))))
        geometry['rays']=results
        return geometry

    def finish(self):
        self.rt.planes.assign=self.original_assign
        self.wrapped.step=self.original_step
        assert len(self.neighbor_rows)>=2 and len(self.checks)==3
        assert len(self.foot_rows)==4, self.foot_rows
        result=dict(mode=self.mode,neighbor_partial_resets=self.neighbor_rows,
                    solver_checks=self.checks,actual_robot_foot_support=self.foot_rows)
        if self.direct:
            assert len(self.native_contacts.rows)==4,self.native_contacts.rows
            result['native_filtered_contact_points']=self.native_contacts.rows
            del self.native_contacts
        return result


def run_bench(args,out,rt,env,wrapped,runner,ema,spec):
    import torch
    import warp as wp
    from spatial_train import evaluate, put

    assert len(rt.pool['task_id'])==1048576 and env.num_envs==spec['num_envs']
    assert spec['learner_device']==str(next(runner.alg.actor.parameters()).device)=='cuda:0'
    assert str(rt.neural.estimator.device)=='cuda:0' and not runner.alg.optimizer.state
    budget=out/'ppo_budget.json'
    direct=spec.get('benchmark_direct_gpu',False)
    fabric=spec.get('benchmark_fabric',False)
    profile=spec.get('benchmark_profile','full')
    warm,width,through=(24,8,32) if profile=='capacity' else (WARM,WINDOW,THROUGH)
    evidence=Evidence(rt,env,wrapped,str(spec['simulator_device']).startswith('cuda'),direct,fabric,warm)
    put(out/'benchmark_entry.json',dict(mode=evidence.mode,slots=env.num_envs,learner='cuda:0',scorer='cuda:0',
                                       warp_cache=str(wp.config.kernel_cache_dir),
                                       profile=profile,training_parallel_envs=env.num_envs,steps_per_env_per_update=24,
                                       training_interactions_per_update=24*env.num_envs,
                                       startup_s=monotonic()-args.benchmark_started,training_pool_rows=1048576))
    birth=read_births(rt,env)
    put(out/'birth_readback.json',birth)
    fit=rt.neural.end_update
    score=[]

    def timed_score(update,timings):
        value=fit(update,timings)
        if warm<update<=through:
            score.append(dict(update=update,fit_s=timings['network_fit_s'],
                              inference_s=timings['full_pool_inference_s']))
        return value

    rt.neural.end_update=timed_score
    runner.learn(num_learning_iterations=warm,init_at_random_ep_len=False)
    physical=evidence.finish()
    physical['initial_births']=birth
    put(out/'physical_passed.json',physical)
    windows=[]
    for start in range(warm,through,width):
        assert json.loads(budget.read_bytes())['ppo_updates_completed']==start
        before_control=env.common_step_counter
        before_ends=int(rt.sampler.completed_counts.sum())
        before_fit=rt.neural.estimator.fit_steps
        torch.cuda.synchronize()
        wall=monotonic()
        runner.current_learning_iteration=start
        runner.learn(num_learning_iterations=width,init_at_random_ep_len=False)
        torch.cuda.synchronize()
        wall=monotonic()-wall
        after=json.loads(budget.read_bytes())
        stop=start+width
        assert after['ppo_updates_completed']==after['scorer_updates_passed']==stop and after['optimizer_calls']==20*stop
        assert env.common_step_counter-before_control==24*width
        ends=int(rt.sampler.completed_counts.sum())-before_ends
        assert ends>0 and rt.neural.estimator.fit_steps-before_fit==100*width
        timings=[item for item in score if start<item['update']<=stop]
        assert len(timings)==width and all(q['fit_s']>0 and q['inference_s']>0 for q in timings)
        interactions=width*24*env.num_envs
        windows.append(dict(first_update=start+1,last_update=stop,interactions=interactions,wall_s=wall,
                            interactions_per_s=interactions/wall,natural_ends=ends,
                            rss_anon_bytes=cpu_anon(),scorer_timings=timings,budget=after))
        put(out/'throughput_windows.json',windows)

    if profile!='full':
        rt.episode_log.flush(through,final=True)
        interactions=through*24*env.num_envs
        put(out/'benchmark_result.json',dict(passed=True,engineering_only=True,profile=profile,
            slots=env.num_envs,mode=physical['mode'],budget=after,training_interactions=interactions,
            training_parallel_envs=env.num_envs,steps_per_env_per_update=24,
            training_interactions_per_update=24*env.num_envs,single_policy_training_interactions=interactions,
            windows=windows,evaluation_parallel_envs=0,representative_evaluation=None))
        return

    before=json.loads(budget.read_bytes())
    before_controls=env.common_step_counter
    start=monotonic()
    evaluated=evaluate(env,wrapped,runner,ema,args.compact_panel,THROUGH,out,rt,spec,fixed_ema_only=True)
    seconds=monotonic()-start
    steps=env.common_step_counter-before_controls
    assert evaluated['rows']==504 and json.loads(budget.read_bytes())==before
    restore_start=monotonic()
    runner.current_learning_iteration=THROUGH
    runner.learn(num_learning_iterations=1,init_at_random_ep_len=False)
    torch.cuda.synchronize()
    resumed_s=monotonic()-restore_start
    after=json.loads(budget.read_bytes())
    assert after['ppo_updates_completed']==after['scorer_updates_passed']==65 and after['optimizer_calls']==1300
    rt.episode_log.flush(65,final=True)
    put(out/'benchmark_result.json',dict(passed=True,engineering_only=True,slots=env.num_envs,
        mode=physical['mode'],budget=after,training_interactions=65*24*env.num_envs,
        training_parallel_envs=env.num_envs,steps_per_env_per_update=24,
        training_interactions_per_update=24*env.num_envs,single_policy_training_interactions=65*24*env.num_envs,
        windows=windows,representative_evaluation=dict(rows=504,wall_s=seconds,
        evaluation_parallel_envs=env.num_envs,
        control_steps=steps,padded_slot_interactions=steps*env.num_envs,
        first_post_evaluation_update=65,first_post_evaluation_wall_s=resumed_s),
        startup_marker=out.joinpath('benchmark_entry.json').name))
