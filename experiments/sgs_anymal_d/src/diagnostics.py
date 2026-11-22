"""On-device qualification and held-out behavioral evaluation for the SGS replication."""
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
from course import TERRAIN_NAMES
from ppo_smoke import obs, save, fingerprint


def stable_arrival(near, gravity_z, planar_speed, angular_speed, stance_feet):
    return near & (gravity_z < -.8660254) & (planar_speed < .45) & (angular_speed < 1.5) & (stance_feet >= 2)


def qualify(world, out):
    # Every tile is represented; index 10 is the first goal on the opposite apron.
    ids=(world.env_ids % 260)*400+torch.where((world.env_ids//260)%2==0,10,200)
    world.place(world.env_ids,ids,True)
    start=world.start_locations.clone()
    goals=world.targets.clone()
    offsets=goals-start
    assert (offsets[:,0].abs()>4.).all()
    points=torch.cat([start[:260],goals[:260]])
    surface_error=(world.ray_hits(points)-points).abs().max().item()
    expected=world.lookup[ids]
    assert (goals-world.origins-expected[:,3:6]).abs().max()<2.e-5
    assert surface_error<.0002
    stairs=(ids//8000==3)
    assert offsets[stairs,2].abs().min()>.39
    center=world.origins[:260].clone()
    # The narrowest gap occupies one raster cell. x=0 is a cell edge shared
    # with the apron for several layouts; cast at a known interior point.
    layout=torch.arange(260,device=world.device)
    center[:,0]+=torch.where((layout//10)%2==0,-.075,.075)
    center[:,1]+=.075
    center[:,2]+=3.
    measured=world.ray_hits(center)[:,2]-world.origins[:260,2]
    save(out/'gap-probes.json',dict(offset_x_by_seed=[-.075,.075],offset_y=.075,
                                   hits_m=measured[:20].cpu().tolist()))
    assert measured[:20].max() < -1., 'gap meshes lack a trench'
    # Spatially representative physics check on all 260 layouts, with a controlled
    # positive at the spawn and a true far-goal negative on another set of slots.
    world.targets[:520]=start[:520]
    positive=torch.zeros(520,device=world.device,dtype=torch.bool)
    far_positive=torch.zeros_like(positive)
    zero=torch.zeros_like(world.last_action)
    max_tilt=torch.zeros(520,device=world.device)
    for tick in range(80):
        labels,_=world.step(zero)
        positive |= labels[:520]==1
        far_positive |= labels[520:1040]==1
        up=-world.robot.data.projected_gravity_b[:520,2]
        max_tilt=torch.maximum(max_tilt,torch.rad2deg(torch.acos(up.clamp(-1,1))))
    save(out/'stand-probes.json',dict(n=int(positive.numel()),positive=int(positive.sum()),
         missing_positive_slots=(~positive).nonzero().flatten().cpu().tolist(),
         false_far_arrivals=int(far_positive.sum()),peak_tilt_deg=float(max_tilt.max())))
    assert positive.all() and not far_positive.any(), (int(positive.sum()),int(far_positive.sum()))
    assert float(max_tilt.max())<30, float(max_tilt.max())
    # Challenge the final hold tick with an actually tilted native articulation.
    cut=world.env_ids[:13]
    roots=world.robot.data.root_state_w[cut,:7].clone()
    roots[:,3:7]=roots.new_tensor([.70710678,.70710678,0,0])
    world.robot.write_root_pose_to_sim(roots,env_ids=cut)
    world.sim.forward()
    world.hold[cut]=19
    labels,_=world.step(zero)
    assert not (labels[cut]==1).any()
    save(out/'physics-and-standing.json',dict(layouts=260,envs=world.n,
         under_spawn_goal_ray_error_m=surface_error,stairs_min_endpoint_delta_m=float(offsets[stairs,2].abs().min()),
         gap_highest_middle_surface_m=float(measured[:20].max()),standing_positive=int(positive.sum()),
         far_stationary_false_successes=int(far_positive.sum()),sideways_false_successes=int((labels[cut]==1).sum()),
         largest_native_standing_tilt_deg=float(max_tilt.max()),student_updates=0))


def evaluate(world,alg,out,node,seed):
    """New reset coordinates, uniform within each layout; same panel for the four students."""
    root=Path(__file__).resolve().parent.parent
    rng=np.random.default_rng(881924)
    # Eight unselected tasks in each of the 260 layouts. The first row for each
    # terrain in the separate replay subset is preselected to traverse its obstacle.
    ids=np.concatenate([k*400+rng.choice(400,size=8,replace=False) for k in range(260)])
    trace_ids=np.array([(k*20+4)*400+2*20+17 for k in range(13)])
    fixed=np.concatenate([ids,trace_ids])
    world_ids=np.resize(fixed,world.n)
    save_to=out/'evaluations'/f'u{node:04d}'
    save_to.mkdir(parents=True,exist_ok=True)
    n=len(fixed)
    ix=torch.as_tensor(world_ids,device=world.device)
    capture=torch.arange(len(ids),n,device=world.device)
    torch_before=torch.get_rng_state()
    cuda_before=torch.cuda.get_rng_state(world.device)
    original=world.lookup
    try:
        world.lookup=torch.as_tensor(np.load(root/'holdout.npy'),device=world.device)
        world.place(world.env_ids,ix)
        alg.eval_mode()
        first=torch.zeros(n,device=world.device,dtype=torch.int8)
        tilt=torch.zeros(n,device=world.device); peak=torch.zeros_like(tilt)
        ticks=torch.zeros(n,device=world.device,dtype=torch.int32)
        min_x=torch.full_like(tilt,float('inf'));max_x=-min_x
        positions=[];snap_flags=[]
        # Isaac lazily creates mutable articulation caches while stepping. Inference
        # tensors cannot be written by the next reset outside this block.
        with torch.no_grad():
            for t in range(1,501):
                labels,_=world.step(alg.actor(obs(world)),training_iteration=node)
                active=first==0
                a=torch.rad2deg(torch.acos((-world.robot.data.projected_gravity_b[:n,2]).clamp(-1,1)))
                x=world.robot.data.root_pos_w[:n,0]-world.origins[:n,0]
                min_x=torch.where(active,torch.minimum(min_x,x),min_x)
                max_x=torch.where(active,torch.maximum(max_x,x),max_x)
                peak=torch.where(active,torch.maximum(peak,a),peak)
                fresh=active & (labels[:n]!=0)
                first[fresh]=labels[:n][fresh];tilt[fresh]=a[fresh];ticks[fresh]=t
                if t==1 or t%5==0:
                    positions.append((world.robot.data.body_link_pose_w[capture].clone()).cpu().numpy())
                    snap_flags.append((active[capture].clone()).cpu().numpy())
                if (first!=0).all():break
        assert (first!=0).all()
        labels=first.cpu().numpy();angle=tilt.cpu().numpy()
        endpoints=world.lookup[ix[:n]].cpu().numpy()
        cross=endpoints[:,0]*endpoints[:,3]<0
        traversed=(min_x.cpu().numpy() < -1.9) & (max_x.cpu().numpy() > 1.9)
        success=labels==1
        assert not np.any(success & (angle>=30.01))
        assert not np.any(success & cross & ~traversed)
        np.savez_compressed(save_to/'outcomes.npz',ids=fixed,labels=labels,tilt_deg=angle,
            peak_tilt_deg=peak.cpu().numpy(),controls=ticks.cpu().numpy(),cross=cross,traversed=traversed)
        np.savez_compressed(save_to/'unselected_terrain_traces.npz',ids=trace_ids,
            body_names=world.robot.body_names,origins=world.origins[capture].cpu().numpy(),
            goal=world.targets[capture].cpu().numpy(),poses=np.asarray(positions),active=np.asarray(snap_flags))
        scored=np.arange(len(ids))
        def counts(indices):
            c=indices[cross[indices]]
            return dict(n=len(indices),success=int(success[indices].sum()),cross_n=len(c),
                        cross_stable_success=int(success[c].sum()),base_contacts=int((labels[indices]==3).sum()),
                        drops=int((labels[indices]==2).sum()),timeouts=int((labels[indices]==4).sum()))
        result=counts(scored)
        result.update(seed=seed,update=node,actor_sha256=fingerprint(alg.actor.state_dict()),
            by_terrain={name:counts(scored[ids//8000==k]) for k,name in enumerate(TERRAIN_NAMES)},
            preselected_crossing_13=counts(np.arange(len(ids),n)))
        save(save_to/'summary.json',result)
    finally:
        world.lookup=original
        torch.set_rng_state(torch_before)
        torch.cuda.set_rng_state(cuda_before,world.device)
        alg.train_mode()
    return result
