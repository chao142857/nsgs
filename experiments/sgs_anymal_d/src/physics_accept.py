"""Single-GPU preflight for the reconstructed course, zero PPO."""
import argparse
import json
from pathlib import Path
from dataclasses import asdict
import numpy as np

from catalog import ROOT
from course import decode, patchset, TERRAIN_NAMES, region_at, surface_at, surface, region_grid
from route_events import Clock, Event, clock_checks, sample_mesh, rasters, LOAD_N, FOOT_MESH_TOL_M
from strict_eval import snapshot, native_hit, write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--device-index',type=int,default=0)
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    from isaac_entry import open_app
    app=open_app(args.device_index,args.out)
    try:verify(args)
    finally:app.close(wait_for_replicator=False,skip_cleanup=True)


def verify(args):
    import torch
    from native import BatchedScene
    from pilot_config import git_commit
    from navigation import WEIGHTS
    out=args.out
    catalog=json.loads((ROOT/'catalog.json').read_text())
    routes=json.loads((ROOT/'fixed_routes.json').read_text())
    checks={};details={}; cpu=lambda x:x.detach().cpu().numpy()
    def record():
        write(out/'physics_acceptance.json',dict(passed=bool(checks) and all(checks.values()) and
            checks.get('complete',False),zero_ppo=True,geometry_hashes=catalog['geometry_sha256'],
            git_commit=git_commit(ROOT),checks=checks,details=details))
    def require(name,ok,detail=None):
        checks[name]=bool(ok)
        if detail is not None: details[name]=detail
        record()
        if not ok:raise RuntimeError('failed physical acceptance: '+name)
    def tensor(x):return torch.as_tensor(x,device=world.device)
    # 2 physical spawn tests per selected route, plus three controlled pose fixtures.
    m=len(routes);n=2*m+3
    world=BatchedScene(n)
    world.select_bank('holdout')
    require('all_layout_raycasts',bool(world.probe_geometry()['ray_xyz_max_error_m']<.00015))
    safe=next(r for r in routes if r['terrain']=='islands' and r['route']=='level_neighbors')
    fixed=[r['id'] for r in routes]
    ids=tensor(fixed+fixed+[safe['id']]*3)
    world.place(world.env_ids,ids,True)
    # For the goal-side physical test use its own yaw and clearance independently of birth.
    lanes=world.env_ids[m:2*m]
    roots=world.robot.data.default_root_state[lanes].clone()
    roots[:,:3]=world.targets[lanes]
    for i,r in enumerate(routes):
        k,seed,lev,s,g=decode(r['id']);gs=patchset(k,seed,lev,1,1)
        yaw=gs.yaw[g];roots[i,2]+=.60+gs.lift[g]
        roots[i,3:7]=tensor([np.cos(yaw/2),0.,0.,np.sin(yaw/2)])
    world.robot.write_root_pose_to_sim(roots[:,:7],env_ids=lanes)
    world.sim.forward()
    zero=torch.zeros_like(world.last_action)
    idx=world.env_ids[:2*m]
    frames=[];named=[]
    for i in range(2*m):
        r=routes[i%m]
        named.append([r['terrain'], r['route'], ('start' if i<m else 'goal'), r['id']])
    for t in range(101):
        if t:lab,signal=world.step(zero)
        raw=snapshot(world,idx);frames.append(raw)
        if t==0:
            scan=world.scan.data.ray_hits_w[idx,:,2]
            require('finite_height_observation',bool(torch.isfinite(world.observations()[idx]).all()),
                    dict(n_rays=scan.shape[1],finite_per_lane=cpu(torch.isfinite(scan).sum(1)).tolist()))
        if t==1:
            expect=world.last_reward @ world.last_reward.new_tensor(WEIGHTS)
            require('reward_live_units_no_dt',bool(torch.allclose(expect,signal['reward'])),dict(weights=list(WEIGHTS),
                    term_contributions=cpu(signal['reward_terms'].sum(0)).tolist()))
    arrays={k:np.stack([x[k] for x in frames]) for k in frames[0]}
    np.savez_compressed(out/'starts_and_targets_physics.npz',**arrays,names=np.asarray(named,dtype=str))
    supported=((arrays['foot_force'][:,:,:,2]>LOAD_N)&
               (np.abs(arrays['feet'][:,:,:,2]-arrays['ground_z'])<=FOOT_MESH_TOL_M))
    starts=[]
    for i in range(2*m):
        r=routes[i%m];k,s,l,_,_=decode(r['id'])
        reg=region_at(k,s,l,arrays['feet'][:,i,:,0],arrays['feet'][:,i,:,1])
        legal=supported[:,i]& (reg>=0)
        penetration=arrays['ground_z'][0,i]-arrays['feet'][0,i,:,2]
        starts.append(dict(terrain=r['terrain'],id=r['id'],side=named[i][2],
            max_initial_foot_link_under_surface_m=float(np.max(penetration)),
            two_supported_steps=int((legal.sum(-1)>=2).sum()),
            two_supported_final10=int((legal[-10:].sum(-1)>=2).sum()),
            max_base_contact=float(arrays['base_force'][:,i].max()),
            tilt_peak=float(arrays['tilt_deg'][:,i].max())))
    require('real_birth_goal_support',all(r['two_supported_steps']>=20 and r['two_supported_final10']>=5
            and r['max_initial_foot_link_under_surface_m']<.035 and r['max_base_contact']<=30 for r in starts),starts)
    # Distinct body-height probes at actual obstacle interiors in PhysX's authored mesh.
    from terrain_compat import RESOLUTION,HALF_WIDTH
    probes=[]
    for i,r in enumerate(routes):
        grid,rr=surface(r['terrain'],r['terrain_seed'],r['level']),region_grid(r['terrain'],r['terrain_seed'],r['level'])
        regions=sorted(set(rr.ravel().tolist()))
        for rid in regions:
            mask=np.argwhere(rr==rid)
            if rid==-1:
                interior=mask[(abs(mask[:,0]-grid.shape[0]/2)<21)&(abs(mask[:,1]-grid.shape[1]/2)<18)]
                if len(interior):mask=interior
            u,v=mask[len(mask)//2]; xy=(np.array([u,v])+.5)*RESOLUTION-HALF_WIDTH
            z=grid[u,v]
            point=world.origins[i]+tensor([*xy,float(z)]).float()
            actual=world.ray_hits(point[None].float())[0]-world.origins[i]
            probes.append(dict(terrain=r['terrain'],region=rid,xy=xy.tolist(),expected=float(z),actual=float(actual[2])))
    require('surface_profiles',all(abs(r['expected']-r['actual'])<.0002 for r in probes),probes)
    require('clock_boundary_unit_vectors',clock_checks())
    # Physically stable terminals and tilt/slide interventions have separate provenance from policy traces.
    # Local route tokens are marked as complete only to independently challenge the terminal verifier.
    # Another lane is kept at the start with the original route intact throughout.
    ctrls=world.env_ids[-3:]
    world.place(ctrls,ids[-3:],True)
    original=safe
    anchor={**safe,'start':safe['start'],'goal':safe['start'],'goal_support':safe['start_support']}
    world.targets[ctrls[1:]]=world.start_locations[ctrls[1:]].clone()
    idle=Event(original); positive=Event(anchor);positive.event_tick=0
    wrong=Event({**anchor,'goal_support':safe['goal_support']});wrong.event_tick=0
    c=Clock();early=Clock()
    rows=[]; stable_run=[]; replay_hold=[]
    for t in range(1,241):
        labels,signals=world.step(zero)
        raw=snapshot(world,ctrls)
        a0,s0,f0,_,_=native_hit(idle,t,raw,0,labels[ctrls[0]])
        a,s,f,_,_=native_hit(positive,t,raw,1,labels[ctrls[1]])
        aw,sw,fw,_,_=native_hit(wrong,t,raw,1,labels[ctrls[1]])
        c.observe(t,a,s,f);early.observe(t,a0,s0,f0)
        rows.append(dict(t=t,arrived=a,stable=s,failure=f,wrong_arrived=aw,idle_arrived=a0,
            training_label=int(labels[ctrls[1]]),hold=c.hold,tilt=float(raw['tilt_deg'][1]),
            foot_normal_n=raw['foot_force'][1,:,2].tolist(),base_n=float(raw['base_force'][1])))
        if s and not f:stable_run.append((a,s,f))
        else:stable_run.clear()
        if len(stable_run)>=150:break
    write(out/'controlled_stand.json',rows)
    require('true_stand_150',c.done=='success' and len(stable_run)>=150 and c.end-c.hold_start+1==150)
    require('idle_does_not_cross',not any(r['idle_arrived'] for r in rows))
    require('wrong_island_support_is_rejected',not any(r['wrong_arrived'] for r in rows),
            'Same measured root, forces and in-radius target XY. Substitute the other island support ID to isolate the surface check.')
    require('training_hold_positive',any(r['training_label']==1 for r in rows))
    now=cpu(world.robot.data.root_pos_w[ctrls[1]]-world.origins[ctrls[1]])
    # In a controlled terminal lane give a velocity impulse while in the target.
    slide_lane=ctrls[2:]; velocity=world.robot.data.root_vel_w[slide_lane].clone()
    velocity[:,:3]=tensor([1.4,0.,0.]).float();world.robot.write_root_velocity_to_sim(velocity,env_ids=slide_lane)
    world.sim.forward()
    slide=Event(anchor);slide.event_tick=0
    sl=[]; fall=[]
    # Physically lower and roll the native base into its support, after a stable controlled prefix.
    fall_lane=ctrls[1:2];root=world.robot.data.root_pos_w[fall_lane].clone()
    root[:,2]=world.targets[fall_lane,2]+.10
    pose=torch.cat([root,tensor([[.70710678,.70710678,0.,0.]]).float()],dim=1)
    world.robot.write_root_pose_to_sim(pose,env_ids=fall_lane)
    world.robot.write_root_velocity_to_sim(torch.zeros((1,6),device=world.device),env_ids=fall_lane)
    world.sim.forward()
    for j in range(18):
        labels,signals=world.step(zero)
        raw=snapshot(world,ctrls)
        a,s,f,_,_=native_hit(slide,j+1,raw,2,labels[ctrls[2]])
        sl.append(dict(tick=j+1,near=bool(signals['near'][ctrls[2]]),speed=float(raw['linear_speed'][2]),stable=s))
        _,_,ff,_,_=native_hit(positive,j+1,raw,1,labels[ctrls[1]])
        fall.append(dict(tick=j+1,failure=ff,tilt=float(raw['tilt_deg'][1]),base_force=float(raw['base_force'][1]),
                         training_label=int(labels[ctrls[1]])))
    require('slide_through_target',any(r['near'] and r['speed']>.45 and not r['stable'] for r in sl),sl)
    require('real_base_contact_failure',any(r['failure']=='base_contact' and r['training_label']==3 for r in fall),fall)
    # Use one measured sequence and a measured failure as boundary inputs; these are explicitly replay checks.
    actual_fall=next(r['failure'] for r in fall if r['failure']=='base_contact')
    measured=stable_run[-150:]
    cases={}
    def replay_case(key,prefix,tail,expected):
        ck=Clock()
        for t,triple in enumerate(prefix+tail,1):ck.observe(t,*triple)
        cases[key]=asdict(ck);assert ck.done==expected,(key,asdict(ck))
    replay_case('149_measured_controls',[],measured[:149],None)
    replay_case('150_measured_controls',[],measured,'success')
    replay_case('fall_after_2p9s',measured[:145],[(False,False,actual_fall)],'base_contact')
    replay_case('failure_wins_on_success_tick',measured[:149],[(True,True,actual_fall)],'base_contact')
    replay_case('arrival_at_9p5_seconds',[(False,False,None)]*474,measured,'success')
    replay_case('interrupted_after_deadline',[(False,False,None)]*474,measured[:55]+[(False,False,None)],'hold_interrupted')
    require('measured_clock_replay',True,cases)

    # Three persistent real solver lanes. No episode reset between comparator ticks.
    world.place(ctrls,ids[-3:],True)
    world.targets[ctrls]=world.start_locations[ctrls].clone()
    for warm in range(65):
        world.step(zero)
    # Lane 0 gets a real physical fall at t146. Lane 1 receives a controlled, late
    # target command at t475. Lane 2 checks the unperturbed 149/150 boundary.
    world.targets[ctrls[1]]=world.origins[ctrls[1]]+tensor(safe['goal']).float()
    curves=[]; local_clocks=[Clock() for _ in range(3)]
    markers=[Event(anchor) for _ in range(3)]
    for e in markers:e.event_tick=0
    for tick in range(1,626):
        if tick==146:
            assert local_clocks[0].hold==145,asdict(local_clocks[0])
            lane=ctrls[:1]; rr=world.robot.data.root_pos_w[lane].clone()
            rr[:,2]=world.targets[lane,2]+.10
            pose=torch.cat([rr,tensor([[.70710678,.70710678,0.,0.]]).float()],dim=1)
            world.robot.write_root_pose_to_sim(pose,env_ids=lane)
            world.robot.write_root_velocity_to_sim(torch.zeros((1,6),device=world.device),env_ids=lane)
            world.sim.forward()
        if tick==475:
            world.targets[ctrls[1]]=world.start_locations[ctrls[1]].clone()
        labels,sig=world.step(zero)
        raw=snapshot(world,ctrls)
        qr=[]
        for j in range(3):
            ev=markers[j];ev.route={**anchor,'goal':cpu(world.targets[ctrls[j]]-world.origins[ctrls[j]]).tolist(),
                'goal_support':safe['goal_support'] if j==1 and tick<475 else safe['start_support']}
            a,s,f,_,_=native_hit(ev,tick,raw,j,labels[ctrls[j]])
            local_clocks[j].observe(tick,a,s,f)
            qr.append([a,s,f,int(local_clocks[j].hold),float(raw['base_force'][j]),
                       float(raw['tilt_deg'][j]),float(raw['linear_speed'][j]),raw['foot_force'][j,:,2].tolist()])
        curves.append(qr)
    write(out/'continuous_controlled_clocks.json',dict(fields=['arrival','stable','failure','hold','base_force_N',
          'tilt_deg','speed_mps','foot_normal_N'],lanes=['perturb_after145','change_command475','unperturbed'],
          rows=curves))
    require('physical_fall_after145',local_clocks[0].done=='base_contact' and 146<=local_clocks[0].end<150
            and local_clocks[0].best_hold==145,asdict(local_clocks[0]))
    require('physical_149_150',local_clocks[2].end==150 and curves[148][2][3]==149
            and curves[149][2][3]==150,asdict(local_clocks[2]))
    require('physical_late_arrival',local_clocks[1].first_arrival==475 and local_clocks[1].end==624
            and local_clocks[1].done=='success',asdict(local_clocks[1]))
    require('complete',True)


if __name__=='__main__':main()
