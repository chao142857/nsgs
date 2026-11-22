"""Versioned equilibrium-entry pools; never overwrites the original generator/pools."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from goalreach.entry_kinematics import EntryArm,RobotTree
from tools.build_pools import COUNTS,REGIONS,content_hash

BANDS=[(0.,.02),(.02,.05),(.05,.10)]


def make_entry_pool(path,reference,seed,evaluation=False):
    arm=EntryArm(path,reference);rng=np.random.default_rng(seed)
    targets=[];regions=[];heights=[];bands=[];witness=[]
    for region,count in enumerate([252]*4 if evaluation else COUNTS):
        levels=range(3) if evaluation or region==0 else [-1]
        for level in levels:
            n=count//3+(level<count%3) if region==0 else count//3 if evaluation else count
            accepted=[];qs=[]
            while sum(len(v) for v in accepted)<n:
                size=max(2048,3*n);q=np.zeros((size,6))
                if region==0:
                    lo,hi=BANDS[level];direction=rng.normal(size=(size,3))
                    direction/=np.linalg.norm(direction,axis=1)[:,None]
                    radius=rng.uniform(lo**3,hi**3,size)**(1/3)
                    xyz=arm.nominal+radius[:,None]*direction
                    valid=np.ones(size,bool)
                elif region==1:
                    q=rng.uniform(arm.lo,arm.hi,(size,6))
                    q[:size//2]=arm.reference_q+(q[:size//2]-arm.reference_q)*rng.random((size//2,1))
                    xyz=arm.fk(q)
                    valid=arm.admissible(q)&(np.linalg.norm(xyz-arm.nominal,axis=1)>.105)
                else:
                    lo,hi=(arm.radial_bound+.01,1.5) if region==2 else (1.5,3.)
                    radius=np.sqrt(rng.uniform(lo**2,hi**2,size));angle=rng.uniform(-np.pi,np.pi,size)
                    xyz=np.c_[radius*np.cos(angle),radius*np.sin(angle),rng.uniform(.10,.85,size)]
                    valid=np.ones(size,bool)
                valid&=(xyz[:,2]>=.10)&(xyz[:,2]<=.85)
                if evaluation and region!=0:
                    valid&=(xyz[:,2]>=.10+.75*level/3)&(xyz[:,2]<.10+.75*(level+1)/3)
                accepted.append(xyz[valid]);qs.append(q[valid])
            targets.append(np.concatenate(accepted)[:n]);witness.append(np.concatenate(qs)[:n])
            regions.extend([region]*n);bands.extend([level if region==0 else -1]*n)
            heights.extend([level if evaluation and region!=0 else -1]*n)
    xyz=np.concatenate(targets).astype(np.float32);n=len(xyz)
    delta=rng.uniform(-.01,.01,(n,18))
    roots,q,initial_ee,lift=RobotTree(path).reset_geometry(reference,delta)
    return dict(task_id=np.arange(n,dtype=np.int32),region=np.array(regions,np.int8),
        target=xyz,horizontal_distance=np.linalg.norm(xyz[:,:2],axis=1),
        azimuth=np.round(np.arctan2(xyz[:,1].astype(float),xyz[:,0].astype(float)),12).astype(np.float32),
        height=xyz[:,2].copy(),nominal_distance=np.linalg.norm(xyz-arm.nominal,axis=1).astype(np.float32),
        near_distance_band=np.array(bands,np.int8),height_stratum=np.array(heights,np.int8),
        initial_joint_delta=(q-np.array(reference['joint_position'])).astype(np.float32),
        initial_joint_position=q.astype(np.float32),initial_root_pose=roots.astype(np.float32),
        initial_ee=initial_ee.astype(np.float32),reset_clearance_lift=lift.astype(np.float32),
        action_joint_names=np.array(reference['action_joint_names']),
        reference_root_pose=np.array(reference['root_pose']),reference_joint_position=np.array(reference['joint_position']),
        initial_yaw=np.zeros(n,np.float32),episode_seconds=np.full(n,10.,np.float32),
        inplace_joint_witness=np.concatenate(witness).astype(np.float32),seed=np.array(seed),
        nominal_ee=arm.nominal.astype(np.float32),near_distance_bounds=np.array(BANDS),
        schema_version=np.array('goalreach-entry-v4'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--reference',type=Path,default=Path('configs/equilibrium_candidate_v1.json'))
    p.add_argument('--kinematics',type=Path,default=Path('reference/native_kinematics.json'));args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False);reference=json.loads(args.reference.read_text())
    result=dict(schema='goalreach-entry-v4',reference=reference,
        reference_file_sha256=hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        native_pd_targets_unchanged=True,reset_minimum_foot_clearance_m=.001,
        evaluation_strata='near: three distance shells, 84 each; other regions: absolute height thirds, 84 each',
        height_limits_m=[.10,.85],stats={})
    pools={}
    for name,seed in [('train',2026111217),('evaluation',2026111218)]:
        d=make_entry_pool(args.kinematics,reference,seed,name=='evaluation');pools[name]=d
        assert content_hash(d)==content_hash(make_entry_pool(args.kinematics,reference,seed,name=='evaluation'))
        np.savez_compressed(args.out/(name+'.npz'),**d)
        result['stats'][name]=dict(n=len(d['target']),seed=seed,content_sha256=content_hash(d),
            file_sha256=hashlib.sha256((args.out/(name+'.npz')).read_bytes()).hexdigest(),
            reference_ee_m=d['nominal_ee'].tolist(),reset_lift_range_m=[float(d['reset_clearance_lift'].min()),float(d['reset_clearance_lift'].max())],
            regions={r:dict(n=int((d['region']==i).sum()),xyz_min=d['target'][d['region']==i].min(0).tolist(),
                xyz_max=d['target'][d['region']==i].max(0).tolist(),
                distance_bands=[int(((d['region']==i)&(d['near_distance_band']==j)).sum()) for j in range(3)],
                height_strata=[int(((d['region']==i)&(d['height_stratum']==j)).sum()) for j in range(3)]) for i,r in enumerate(REGIONS)})
        assert len(np.unique(d['target'],axis=0))==len(d['target'])
    assert not set(map(tuple,pools['train']['target']))&set(map(tuple,pools['evaluation']['target']))
    (args.out/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['stats'],indent=2))


if __name__=='__main__':main()
