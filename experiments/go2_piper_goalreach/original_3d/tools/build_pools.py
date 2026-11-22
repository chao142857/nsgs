"""Generate seeds, task features, FK witnesses and reproducible admission evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from goalreach.kinematics import Arm

COUNTS = [4915, 9830, 8192, 9831]
REGIONS = ['near', 'inplace', 'step', 'walk']


def make_pool(arm, seed, evaluation=False):
    rng = np.random.default_rng(seed)
    counts = [252]*4 if evaluation else COUNTS
    targets, regions, strata, witness = [], [], [], []
    bounds = [(arm.nominal[2]-.05, arm.nominal[2]+.05), (.10,.85), (.10,.85), (.10,.85)]
    for region, count in enumerate(counts):
        for level in range(3) if evaluation else [-1]:
            n = count//3 if evaluation else count
            accepted, qs = [], []
            while sum(len(a) for a in accepted) < n:
                size = max(n*3, 2048)
                q = np.zeros((size, 6))
                if region == 0:
                    v = rng.normal(size=(size, 3)); v /= np.linalg.norm(v,axis=1)[:,None]
                    xyz = arm.nominal + v * (.05*rng.random(size)**(1/3))[:,None]
                    valid = np.ones(size,bool)
                elif region == 1:
                    # Mix broad joint sampling with a continuous bridge from nominal.
                    q = rng.uniform(arm.lo, arm.hi, (size,6))
                    q[:size//2] *= rng.random((size//2,1))
                    xyz = arm.fk(q)
                    valid = arm.admissible(q) & (np.linalg.norm(xyz-arm.nominal,axis=1) > .055)
                else:
                    # Outward bound is rigorously beyond the fixed-base arm workspace;
                    # body shift/tilt without foot motion is not certified impossible.
                    lo, hi = (arm.radial_bound+.01, 1.5) if region==2 else (1.5,3.)
                    assert lo < hi
                    radius = np.sqrt(rng.uniform(lo**2,hi**2,size))
                    angle = rng.uniform(-np.pi,np.pi,size)
                    xyz = np.c_[radius*np.cos(angle),radius*np.sin(angle),rng.uniform(.10,.85,size)]
                    valid = np.ones(size,bool)
                if evaluation:
                    zlo,zhi = bounds[region]
                    lower,upper = zlo+(zhi-zlo)*level/3, zlo+(zhi-zlo)*(level+1)/3
                    valid &= (xyz[:,2] >= lower) & (xyz[:,2] < upper)
                accepted.append(xyz[valid]); qs.append(q[valid])
            xyz=np.concatenate(accepted)[:n]; q=np.concatenate(qs)[:n]
            targets.append(xyz); witness.append(q)
            regions.extend([region]*n); strata.extend([level]*n)
    xyz=np.concatenate(targets).astype(np.float32)
    region=np.array(regions,np.int8)
    delta=rng.uniform(-.01,.01,(len(xyz),18))
    # Native joint2=0 is its lower hard limit; joint3=0 is its upper hard limit.
    # Clamp the stored configuration itself, not just the runtime reset.
    delta[:,-6:]=np.clip(delta[:,-6:],arm.lo-1e-4,arm.hi+1e-4)
    return dict(task_id=np.arange(len(xyz),dtype=np.int32), region=region,
        target=xyz, horizontal_distance=np.linalg.norm(xyz[:,:2],axis=1),
        azimuth=np.round(np.arctan2(xyz[:,1].astype(np.float64),xyz[:,0].astype(np.float64)),12).astype(np.float32), height=xyz[:,2].copy(),
        nominal_distance=np.linalg.norm(xyz-arm.nominal,axis=1).astype(np.float32),
        height_stratum=np.array(strata,np.int8),
        initial_joint_delta=delta.astype(np.float32),
        initial_yaw=np.zeros(len(xyz),np.float32), episode_seconds=np.full(len(xyz),10.,np.float32),
        inplace_joint_witness=np.concatenate(witness).astype(np.float32),
        seed=np.array(seed), nominal_ee=arm.nominal.astype(np.float32), height_bounds=np.round(np.array(bounds),12))


def content_hash(data):
    h=hashlib.sha256()
    for k,v in sorted(data.items()):
        h.update(k.encode());h.update(str(v.dtype).encode());h.update(str(v.shape).encode());h.update(v.tobytes())
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--kinematics',type=Path,default=Path('reference/native_kinematics.json'))
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    arm=Arm(args.kinematics)
    pools={k:make_pool(arm,s,k=='evaluation') for k,s in [('train',2026111207),('evaluation',2026111208)]}
    stats={}
    for name,d in pools.items():
        assert content_hash(d)==content_hash(make_pool(arm,int(d['seed']),name=='evaluation'))
        xyz=d['target'];assert len(np.unique(xyz,axis=0))==len(xyz)
        np.savez_compressed(args.out/(name+'.npz'),**d)
        stats[name]=dict(n=len(xyz),seed=int(d['seed']),content_sha256=content_hash(d),
            file_sha256=hashlib.sha256((args.out/(name+'.npz')).read_bytes()).hexdigest(),
            regions={r:dict(n=int((d['region']==i).sum()),
                xyz_min=xyz[d['region']==i].min(0).tolist(),xyz_max=xyz[d['region']==i].max(0).tolist(),
                radius_min=float(d['horizontal_distance'][d['region']==i].min()),
                radius_max=float(d['horizontal_distance'][d['region']==i].max()),
                strata=[int(((d['region']==i)&(d['height_stratum']==j)).sum()) for j in range(3)])
                for i,r in enumerate(REGIONS)})
    a,b=pools.values()
    assert not set(map(tuple,a['target'])) & set(map(tuple,b['target']))
    audit=dict(nominal_ee=arm.nominal.tolist(), fixed_base_horizontal_upper_bound_m=arm.radial_bound,
        step_radius_min_m=arm.radial_bound+.01, height_limits_m=[.1,.85],
        chassis_exclusion=dict(abs_x_lt=.32,abs_y_lt=.20,z_lt=.48),
        method='Continuous FK sampling within USD joint limits, fixed nominal base/feet; positions carry joint witnesses; no orientation goal. Segment/chassis conservative screen.',
        limitations=['Not an exhaustive whole-body no-step workspace.',
            'Step region is beyond fixed-base arm upper bound; no proof that every task requires foot lift when body shifts/tilts.',
            'Native self-collisions disabled. Segment screen is not a mesh collision or dynamic reachability certificate.'],
        evaluation_strata='Equal-width thirds of each region height range; 84 tasks per cell, user approved',
        stats=stats, checks=dict(seed_reproduction=True,no_duplicates=True,train_eval_disjoint=True))
    (args.out/'manifest.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
