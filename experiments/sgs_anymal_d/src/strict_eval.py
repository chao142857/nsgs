"""Independent held-out policy replay. Does not construct or update a SGS sampler."""
import argparse
from collections import defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import numpy as np

from catalog import ROOT
from course import TERRAIN_NAMES
from route_events import Clock, Event, DT, DEADLINE, HOLD, sample_mesh, rasters


def write(path,data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def file_sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def snapshot(world,indices):
    """Copy measured solver state once per control tick for the illustrated route lanes."""
    torch=world.torch
    data=world.robot.data
    ids=indices
    pose=data.body_link_pose_w[ids].clone()
    origin=world.origins[ids]
    foot_world=pose[:,world.foot_bodies,:3]
    surface=world.ray_hits(foot_world.reshape(-1,3)).reshape_as(foot_world)
    pose[:,:,:3]-=origin[:,None,:]
    contact=world.contact.data
    feet=pose[:,world.foot_bodies,:3]
    root=data.root_pos_w[ids]-origin
    variables=dict(body_poses=pose,root=root,feet=feet,
        ground_z=surface[:,:,2]-origin[:,2:3],foot_force=contact.net_forces_w[ids][:,world.foot_sensors],
        contact_forces=contact.net_forces_w[ids],nonfoot_force=contact.net_forces_w[ids][:,world.non_feet].norm(dim=-1).amax(dim=-1),base_force=contact.net_forces_w_history[ids,:,
            world.base_sensor].norm(dim=-1).amax(dim=1),
        tilt_deg=torch.rad2deg(torch.acos((-data.projected_gravity_b[ids,2]).clamp(-1,1))),
        linear_speed=data.root_lin_vel_w[ids,:2].norm(dim=-1),angular_speed=data.root_ang_vel_w[ids].norm(dim=-1),
        joint_pos=data.joint_pos[ids],joint_vel=data.joint_vel[ids],action=world.last_action[ids])
    return {k:v.detach().cpu().numpy() for k,v in variables.items()}


def native_hit(event,t,raw,i,label):
    args=[raw[k][i] for k in ('root','feet','foot_force','tilt_deg','linear_speed','angular_speed','base_force')]
    return event.observe(t,*args,int(label),ray_heights=raw['ground_z'][i],nonfoot_force=raw['nonfoot_force'][i])


def replay(args):
    import torch
    from native import BatchedScene
    from ppo_smoke import make_algorithm, obs, fingerprint
    from visual_assets import export_visuals
    from pilot_config import git_commit
    cpu=lambda v:v.detach().cpu().numpy()
    out=args.out
    routes=json.loads((ROOT/'fixed_routes.json').read_text())
    cat=json.loads((ROOT/'catalog.json').read_text())
    ck=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    assert list(ck['identity']['geometry_hashes'])==cat['geometry_sha256']
    rng=np.random.default_rng(881924)
    panel=np.concatenate([k*400+rng.choice(400,8,replace=False) for k in range(260)])
    np.testing.assert_equal(np.unique(panel).size,len(panel))
    names=TERRAIN_NAMES
    sample_ids=np.r_[panel,[r['id'] for r in routes]]
    n=len(panel);m=len(routes)
    world=BatchedScene(len(sample_ids))
    world.num_envs,world.num_actions=world.n,12
    world.select_bank('holdout')
    ids=torch.as_tensor(sample_ids,device=world.device,dtype=torch.long)
    world.place(world.env_ids,ids,True)
    export_visuals(world,out/'visuals')
    for r,g in zip(routes,cpu(world.lookup[ids[n:]])):
        np.testing.assert_allclose(g[:6],r['start']+r['goal'],atol=1e-5,rtol=0.)
    alg=make_algorithm(world,obs(world))
    alg.load(ck['ppo'],load_cfg=None,strict=True)
    alg.eval_mode()
    actor_sha=fingerprint(alg.actor.state_dict())
    seed=int(ck['seed']);node=int(ck['update'])
    print('Independent replay',seed,node,'held-out',n,'strict',m,flush=True)
    clocks=[Clock() for _ in routes];events=[Event(r) for r in routes]
    traces=[defaultdict(list) for _ in routes]
    selected=world.env_ids[n:]
    origins=cpu(world.origins[selected])
    hist=[]
    panel_labels=np.zeros(n,dtype='i1');panel_end=np.zeros(n,dtype='i2')
    rows=[]
    def record(i,t,raw,regions,arrived=False,stable=False,short=0):
        tr=traces[i];ev=events[i];clock=clocks[i]
        for k,v in raw.items():tr[k].append(v[i].copy())
        values=dict(t=t,foot_region=regions,accepted_arrival=bool(arrived),
                    stable=bool(stable),hold_count=int(clock.hold),
                    event=bool(ev.event_tick is not None),training_label=int(short))
        for k,v in values.items():tr[k].append(v)
    raw=snapshot(world,selected)
    for i in range(m):record(i,0,raw,np.full(4,-1,dtype='i2'))
    with torch.no_grad():
        for tick in range(1,DEADLINE+HOLD+1):
            labels,_=world.step(alg.actor(obs(world)),training_iteration=node)
            labels=cpu(labels)
            new=(panel_labels==0)&(labels[:n]!=0)
            panel_end[new]=tick;panel_labels[new]=labels[:n][new]
            if any(c.done is None for c in clocks):
                raw=snapshot(world,selected)
                for i,c in enumerate(clocks):
                    if c.done is not None:continue
                    arrived,stable,failure,regions,z=native_hit(events[i],tick,raw,i,labels[n+i])
                    c.observe(tick,arrived,stable,failure)
                    record(i,tick,raw,regions,arrived,stable,labels[n+i])
            if (panel_labels!=0).all() and all(c.done is not None for c in clocks):break
    assert (panel_labels!=0).all() and all(c.done for c in clocks)
    assert actor_sha==fingerprint(alg.actor.state_dict())
    (out/'traces').mkdir(exist_ok=True)
    for i,(r,ev,c,tr) in enumerate(zip(routes,events,clocks,traces)):
        path=Path('traces')/f"{i:02d}-{r['terrain']}-{r['id']}.npz"
        assert len(tr['t'])==c.end+1
        np.savez_compressed(out/path,**{k:np.asarray(v) for k,v in tr.items()},
            seed=seed,checkpoint=node,task_id=r['id'],start=r['start'],goal=r['goal'],
            world_origin=origins[i],body_names=world.robot.body_names,foot_names=world.foot_names,
            contact_body_names=world.contact.body_names,dt=DT)
        rows.append(dict(**r,seed=seed,checkpoint=node,result=c.done,end_tick=c.end,
                         path=str(path),**ev.row(),clock=asdict(c)))
    classes=np.load(ROOT/'holdout_classes.npy')[panel]
    for name in names:
        mask=panel//8000==names.index(name)
        def counts(sub):
            vals=panel_labels[sub]
            return dict(n=int(sub.sum()),success=int((vals==1).sum()),drop=int((vals==2).sum()),
                        base_contact=int((vals==3).sum()),timeout=int((vals==4).sum()))
        hist.append(dict(terrain=name,**counts(mask),by_class={
            key:counts(mask&(classes==code)) for code,key in enumerate(('birth_near','same_platform','obstacle'))}))
    strict=[]
    for name in names:
        rr=[r for r in rows if r['terrain']==name]
        strict.append(dict(terrain=name,n=len(rr),event=sum(r['event_tick'] is not None for r in rr),
            arrival=sum(r['clock']['first_arrival'] is not None for r in rr),
            stable_success=sum(r['result']=='success' for r in rr),
            fall=sum(r['result'] in ('drop','base_contact','rollover') for r in rr),
            timeout=sum(r['result']=='navigation_timeout' for r in rr),
            hold_interrupted=sum(r['result']=='hold_interrupted' for r in rr),
            first_transfer=sum(r['first_support_transfer'] is not None for r in rr),
            max_progress_m=max((r['max_progress_m'] for r in rr),default=None)))
    np.savez_compressed(out/'heldout_rows.npz',id=panel,classification=classes,first_label=panel_labels,
                        first_terminal_tick=panel_end)
    write(out/'strict_routes.json',rows)
    write(out/'heldout_by_terrain.json',hist)
    write(out/'strict_by_terrain.json',strict)
    write(out/'summary.json',dict(seed=seed,checkpoint=node,checkpoint_sha256=file_sha(args.checkpoint),
        git_commit=git_commit(ROOT),actor_sha256=actor_sha,geometry_hashes=cat['geometry_sha256'],
        panel_ids_sha256=hashlib.sha256(panel.astype('<i8').tobytes()).hexdigest(),
        fixed_routes_sha256=file_sha(ROOT/'fixed_routes.json'),independent_scene=True,ppo_updates=0,
        sampling_feedback=0,n_heldout=n,n_strict=m,dt=DT,final_tick=tick))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device-index',type=int,default=0)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    if (args.out/'summary.json').exists():raise FileExistsError(args.out)
    from isaac_entry import open_app
    app=open_app(args.device_index,args.out)
    try:replay(args)
    finally:app.close(wait_for_replicator=False,skip_cleanup=True)

if __name__=='__main__':main()
