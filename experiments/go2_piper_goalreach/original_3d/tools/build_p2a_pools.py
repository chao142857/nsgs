"""Separate v5 generator. Original v3/v4 generators remain byte-for-byte unchanged."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from goalreach.entry_kinematics import EntryArm, RobotTree
from tools.build_entry_pools import BANDS, make_entry_pool
from tools.build_pools import content_hash

NATURAL_COUNTS = [656, 1966, 6554, 23592]
NATURAL_SEED = 2026111301
EVALUATION_SEED = 2026111302
SECTORS = ['front', 'left', 'right', 'back']


def sector(angle):
    a = np.asarray(angle)
    return np.where(abs(a) <= np.pi/4, 0,
           np.where((a > np.pi/4) & (a <= 3*np.pi/4), 1,
           np.where((a < -np.pi/4) & (a >= -3*np.pi/4), 2, 3)))


def cells():
    rows = []
    for region in range(4):
        for distance in (range(3) if region == 0 else range(2) if region == 3 else [-1]):
            for az in (range(4) if region >= 2 else [-1]):
                for height in ([-1] if region == 0 else range(3)):
                    rows.append(dict(cell_id=len(rows), region=region, distance_band=distance,
                                     azimuth_sector=az, height_stratum=height,
                                     count=84 if region < 2 else 42))
    return rows


def make_pool(path, reference, evaluation=False):
    seed = EVALUATION_SEED if evaluation else NATURAL_SEED
    rng = np.random.default_rng(seed); arm = EntryArm(path, reference)
    specifications = cells() if evaluation else [dict(region=r, count=n, distance_band=b,
        azimuth_sector=-1, height_stratum=-1, cell_id=-1)
        for r in range(4) for b,n in (list(enumerate([219,219,218])) if r==0 else [(-1,NATURAL_COUNTS[r])])]
    targets=[]; witnesses=[]; metadata=[]
    for cell in specifications:
        r=cell['region']; n=cell['count']; accepted=[]; qs=[]; count=0
        while count < n:
            size=max(2048,3*n); q=np.zeros((size,6))
            if r==0:
                lo,hi=BANDS[cell['distance_band']]
                v=rng.normal(size=(size,3));v/=np.linalg.norm(v,axis=1)[:,None]
                xyz=arm.nominal+v*rng.uniform(lo**3,hi**3,size)[:,None]**(1/3)
                valid=np.ones(size,bool)
            elif r==1:
                q=rng.uniform(arm.lo,arm.hi,(size,6))
                q[:size//2]=arm.reference_q+(q[:size//2]-arm.reference_q)*rng.random((size//2,1))
                xyz=arm.fk(q);valid=arm.admissible(q)&(np.linalg.norm(xyz-arm.nominal,axis=1)>.105)
            else:
                lo,hi=(arm.radial_bound+.01,1.5) if r==2 else (1.5,3.)
                if evaluation and r==3:lo,hi=[(1.5,2.25),(2.25,3.)][cell['distance_band']]
                radius=np.sqrt(rng.uniform(lo**2,hi**2,size));angle=rng.uniform(-np.pi,np.pi,size)
                xyz=np.c_[radius*np.cos(angle),radius*np.sin(angle),rng.uniform(.1,.85,size)]
                valid=np.ones(size,bool)
            valid&=(xyz[:,2]>=.1)&(xyz[:,2]<=.85)
            if evaluation and r!=0:
                h=cell['height_stratum'];valid&=(xyz[:,2]>=.1+.25*h)&(xyz[:,2]<.1+.25*(h+1))
            if evaluation and r>=2:
                valid&=sector(np.arctan2(xyz[:,1],xyz[:,0]))==cell['azimuth_sector']
            accepted.append(xyz[valid]);qs.append(q[valid]);count+=int(valid.sum())
        targets.append(np.concatenate(accepted)[:n]);witnesses.append(np.concatenate(qs)[:n])
        metadata.extend([cell]*n)
    xyz=np.concatenate(targets).astype(np.float32); n=len(xyz)
    delta=rng.uniform(-.01,.01,(n,18))
    roots,q,ee,lift=RobotTree(path).reset_geometry(reference,delta)
    col=lambda key:np.array([c[key] for c in metadata],np.int16)
    return dict(task_id=np.arange(n,dtype=np.int32),region=col('region').astype(np.int8),target=xyz,
        horizontal_distance=np.linalg.norm(xyz[:,:2],axis=1),
        azimuth=np.round(np.arctan2(xyz[:,1].astype(float),xyz[:,0].astype(float)),12).astype(np.float32),
        height=xyz[:,2].copy(),nominal_distance=np.linalg.norm(xyz-arm.nominal,axis=1).astype(np.float32),
        near_distance_band=np.where(col('region')==0,col('distance_band'),-1).astype(np.int8),
        height_stratum=col('height_stratum').astype(np.int8),cell_id=col('cell_id'),
        azimuth_sector=sector(np.arctan2(xyz[:,1],xyz[:,0])).astype(np.int8),
        distance_band=col('distance_band').astype(np.int8),
        initial_joint_delta=(q-np.array(reference['joint_position'])).astype(np.float32),
        initial_joint_position=q.astype(np.float32),initial_root_pose=roots.astype(np.float32),
        initial_ee=ee.astype(np.float32),reset_clearance_lift=lift.astype(np.float32),
        action_joint_names=np.array(reference['action_joint_names']),reference_root_pose=np.array(reference['root_pose']),
        reference_joint_position=np.array(reference['joint_position']),initial_yaw=np.zeros(n,np.float32),
        episode_seconds=np.full(n,10.,np.float32),inplace_joint_witness=np.concatenate(witnesses).astype(np.float32),
        seed=np.array(seed),nominal_ee=arm.nominal.astype(np.float32),near_distance_bounds=np.array(BANDS),
        schema_version=np.array('goalreach-p2a-v5'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=Path('pools_v5_natural'));a=p.parse_args()
    a.out.mkdir(exist_ok=False,parents=True)
    path=Path('reference/native_kinematics.json');ref=json.loads(Path('configs/equilibrium_candidate_v1.json').read_text())
    old=make_entry_pool(path,ref,2026111217)
    assert content_hash(old)=='64d0c6c391c2e3dfcb228d038e12d3d04c36936afe935717c229e694e79c13ad'
    manifest=dict(schema='goalreach-p2a-v5',cells=cells(),stats={},reference=ref,
                  old_training_content_sha256=content_hash(old))
    pools=[old]
    for name,ev in [('train',False),('evaluation',True)]:
        d=make_pool(path,ref,ev);assert content_hash(d)==content_hash(make_pool(path,ref,ev))
        assert len(np.unique(d['target'],axis=0))==len(d['target'])
        assert all(not set(map(tuple,d['target']))&set(map(tuple,o['target'])) for o in pools)
        pools.append(d);np.savez_compressed(a.out/(name+'.npz'),**d)
        manifest['stats'][name]=dict(n=len(d['target']),seed=int(d['seed']),content_sha256=content_hash(d),
            file_sha256=hashlib.sha256((a.out/(name+'.npz')).read_bytes()).hexdigest(),
            region_counts=np.bincount(d['region']).tolist(),
            cell_counts=np.bincount(d['cell_id']).tolist() if ev else None)
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest['stats'],indent=2))

if __name__=='__main__':main()
