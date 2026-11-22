"""Deterministic natural P3B1 pool. Run from the frozen goalreach project."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np

from goalreach.entry_kinematics import EntryArm, RobotTree
from tools.build_entry_pools import BANDS
from tools.build_p2a_pools import sector
from tools.build_pools import content_hash

SEED = 2026111501
COUNTS = (20992, 62912, 209728, 754944)
NEAR = (7008, 7008, 6976)
SIZE = sum(COUNTS)
CHUNK = 16384
VERSION = 'goalreach-P3B1-pool-v1'
KINEMATICS = Path('reference/native_kinematics.json')
REFERENCE = Path('configs/equilibrium_candidate_v1.json')
EVALUATION = Path('pools_v5_natural/evaluation.npz')
EVAL_SIZE = 2016


def sha256(path):
    with Path(path).open('rb') as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
        return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def target_keys(xyz):
    return np.ascontiguousarray(xyz).view([('x','<f4'),('y','<f4'),('z','<f4')]).ravel()


def draw_candidates(rng, arm, region, band):
    witness = np.zeros((CHUNK, 6))
    if region == 0:
        lo, hi = BANDS[band]
        direction = rng.normal(size=(CHUNK, 3))
        direction /= np.linalg.norm(direction, axis=1)[:, None]
        xyz = arm.nominal + direction * rng.uniform(lo**3, hi**3, CHUNK)[:, None]**(1/3)
        valid = np.ones(CHUNK, bool)
    elif region == 1:
        witness = rng.uniform(arm.lo, arm.hi, (CHUNK, 6))
        witness[:CHUNK//2] = arm.reference_q + (witness[:CHUNK//2]-arm.reference_q)*rng.random((CHUNK//2, 1))
        xyz = arm.fk(witness)
        valid = arm.admissible(witness) & (np.linalg.norm(xyz-arm.nominal, axis=1) > .105)
    else:
        lo, hi = (arm.radial_bound + .01, 1.5) if region == 2 else (1.5, 3.)
        radius = np.sqrt(rng.uniform(lo**2, hi**2, CHUNK))
        angle = rng.uniform(-np.pi, np.pi, CHUNK)
        xyz = np.column_stack((radius*np.cos(angle), radius*np.sin(angle), rng.uniform(.1, .85, CHUNK)))
        valid = np.ones(CHUNK, bool)
    valid &= (xyz[:, 2] >= .1) & (xyz[:, 2] <= .85)
    return xyz[valid].astype('<f4'), witness[valid].astype('<f4')


def build(reference, evaluation):
    rng = np.random.Generator(np.random.PCG64(SEED))
    arm, tree = EntryArm(KINEMATICS, reference), RobotTree(KINEMATICS)
    xyz, witness = np.empty((SIZE, 3), '<f4'), np.empty((SIZE, 6), '<f4')
    region, band = np.empty(SIZE, np.int8), np.full(SIZE, -1, np.int8)
    eval_bytes = target_keys(evaluation).tobytes()
    seen = {eval_bytes[i:i+12] for i in range(0, len(eval_bytes), 12)}
    rejected = 0
    start = 0
    for r, b, count in [(0, b, n) for b, n in enumerate(NEAR)] + [(r, -1, COUNTS[r]) for r in (1, 2, 3)]:
        stop = start + count
        region[start:stop] = r
        band[start:stop] = b
        while start < stop:
            targets, joints = draw_candidates(rng, arm, r, b)
            for target, q in zip(targets, joints):
                key = target.tobytes()
                if key in seen:
                    rejected += 1
                    continue
                seen.add(key)
                xyz[start], witness[start] = target, q
                start += 1
                if start == stop:
                    break
    del seen

    q = np.empty((SIZE, 18), '<f4')
    delta = np.empty_like(q)
    roots = np.empty((SIZE, 7), '<f4')
    ee = np.empty_like(xyz)
    lift = np.empty(SIZE, '<f4')
    for start in range(0, SIZE, CHUNK):
        sl = slice(start, min(start+CHUNK, SIZE))
        d = rng.uniform(-.01, .01, (sl.stop-sl.start, 18))
        root, position, initial_ee, clearance = tree.reset_geometry(reference, d)
        roots[sl], q[sl], ee[sl], lift[sl] = root, position, initial_ee, clearance
        delta[sl] = position - np.asarray(reference['joint_position'])
    angle = np.arctan2(xyz[:, 1].astype(np.float64), xyz[:, 0].astype(np.float64))
    data = dict(task_id=np.arange(SIZE, dtype='<i4'), region=region, target=xyz,
        horizontal_distance=np.linalg.norm(xyz[:, :2], axis=1), azimuth=np.round(angle, 12).astype('<f4'),
        height=xyz[:, 2].copy(), nominal_distance=np.linalg.norm(xyz-arm.nominal, axis=1).astype('<f4'),
        near_distance_band=band, height_stratum=np.full(SIZE,-1,np.int8), cell_id=np.full(SIZE,-1,'<i2'),
        azimuth_sector=sector(angle).astype(np.int8), distance_band=band,
        initial_joint_delta=delta, initial_joint_position=q, initial_root_pose=roots,
        initial_ee=ee, reset_clearance_lift=lift, action_joint_names=np.array(reference['action_joint_names']),
        reference_root_pose=np.asarray(reference['root_pose']), reference_joint_position=np.asarray(reference['joint_position']),
        initial_yaw=np.zeros(SIZE,'<f4'), episode_seconds=np.full(SIZE,10.,'<f4'), inplace_joint_witness=witness,
        seed=np.array(SEED), nominal_ee=arm.nominal.astype('<f4'), near_distance_bounds=np.array(BANDS),
        schema_version=np.array(VERSION))
    checks = validate(data, evaluation, arm, tree, reference)
    return data, dict(checks, duplicate_candidates_rejected=rejected)


def validate(data, evaluation, arm, tree, reference):
    xyz, region, band = (data[k] for k in ('target','region','near_distance_band'))
    require(np.array_equal(data['task_id'],np.arange(SIZE)), 'Task ID coverage')
    require(tuple(np.bincount(region)) == COUNTS, 'Region quota')
    require(tuple(np.bincount(band[region==0])) == NEAR and np.all(band[region!=0]==-1), 'Near quota')
    require(all(np.isfinite(v).all() for v in data.values() if v.dtype.kind=='f'), 'Nonfinite pool field')
    keys = target_keys(xyz)
    require(len(np.unique(keys))==SIZE and not np.intersect1d(keys,target_keys(evaluation)).size, 'Repeated target')
    require(np.all((xyz[:,2]>=.1)&(xyz[:,2]<=.85)), 'Height bounds')
    tol = 5e-7
    for b, (lo, hi) in enumerate(BANDS):
        distances = data['nominal_distance'][band==b]
        require(np.all((distances>=lo-tol)&(distances<=hi+tol)), 'Near band bounds')
    for r, lo, hi in [(2,arm.radial_bound+.01,1.5),(3,1.5,3.)]:
        distances=data['horizontal_distance'][region==r]
        require(np.all((distances>=lo-tol)&(distances<=hi+tol)), 'Horizontal bounds')
    mask = region==1
    for indices in np.array_split(np.flatnonzero(mask), 8):
        witness = data['inplace_joint_witness'][indices]
        require(np.all((witness>=arm.lo-tol)&(witness<=arm.hi+tol)), 'Witness joint bounds')
        require(np.max(abs(arm.fk(witness.astype(float))-xyz[indices])) < tol, 'Witness FK')
        require(np.all(data['nominal_distance'][indices]>.105-tol), 'In-place separation')
    q, roots = data['initial_joint_position'], data['initial_root_pose']
    joints = {j['name']:j for j in tree.joints}
    for i, name in enumerate(reference['action_joint_names']):
        lo, hi = (joints[name][key] for key in ('lower','upper'))
        require(np.all((q[:,i]>=lo-tol)&(q[:,i]<=hi+tol)), 'Reset joint bounds')
    require(np.max(abs(data['initial_joint_delta']))<=.01+tol, 'Reset perturbation')
    require(np.allclose(q-data['reference_joint_position'],data['initial_joint_delta'],rtol=0,atol=tol), 'Reset delta')
    fixed = [0, 1, 3, 4, 5, 6]
    require(np.allclose(roots[:,fixed],np.array(reference['root_pose'])[fixed],rtol=0,atol=tol), 'Root pose')
    require(np.all(data['reset_clearance_lift'] >= 0) and np.allclose(
        roots[:,2]-reference['root_pose'][2],data['reset_clearance_lift'],rtol=0,atol=tol), 'Root clearance lift')
    minimum_clearance = np.inf
    max_ee_error = 0.
    for start in range(0, SIZE, CHUNK):
        sl = slice(start, start+CHUNK)
        poses = tree.poses(roots[sl],q[sl],reference['action_joint_names'])
        center = np.array([-.002, 0, 0])
        for leg in ('FL','FR','RL','RR'):
            foot = poses['/go2_piper/'+leg+'_foot']
            bottom = (foot[:,:3,:3]@center+foot[:,:3,3])[:,2]-.022
            minimum_clearance = min(minimum_clearance, float(bottom.min()))
        max_ee_error = max(max_ee_error,float(abs(poses['/go2_piper/end_effector'][:,:3,3]-data['initial_ee'][sl]).max()))
    require(minimum_clearance>=.001-tol and max_ee_error<tol, 'Reconstructed reset geometry')
    return dict(unique_targets=True,evaluation_disjoint=True,finite_numeric_fields=True,validated_records=SIZE,
                region_counts=list(COUNTS),near_counts=list(NEAR),minimum_foot_clearance_m=minimum_clearance,
                max_ee_reconstruction_error_m=max_ee_error,
                xyz_min=xyz.min(axis=0).tolist(),xyz_max=xyz.max(axis=0).tolist())


def write_pool(path, data):
    # Stored ZIP members, fixed names/date/order, no architecture-specific zlib stream.
    with zipfile.ZipFile(path,'x',compression=zipfile.ZIP_STORED) as archive:
        for key, value in sorted(data.items()):
            member = zipfile.ZipInfo(key+'.npy',date_time=(1980,1,1,0,0,0))
            member.create_system = 3
            member.external_attr = 0o100644 << 16
            with archive.open(member,'w') as stream:
                np.lib.format.write_array(stream, np.ascontiguousarray(value) if value.ndim else value, allow_pickle=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--expected',type=Path)
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    path=args.out/'train.npz'
    require(not path.exists(),'Output already exists')
    reference=json.loads(REFERENCE.read_text())
    with np.load(EVALUATION,allow_pickle=False) as data:
        evaluation=dict(data)
    require(len(evaluation['target'])==EVAL_SIZE and len(np.unique(evaluation['cell_id']))==42, 'Fixed evaluation geometry')
    data, audit=build(reference,evaluation['target'])
    temporary = args.out/'train.partial.npz'
    write_pool(temporary,data)
    manifest=dict(schema=VERSION,seed=SEED,block_size=CHUNK,numpy=np.__version__,
        draw_order='PCG64. Ordered near bands 0/1/2 then inplace/step/walk. Full blocked candidate draws, preserve accepted row order, skip float32 duplicates. Then reset uniforms in task order.',
        generator_files={name:sha256(name) for name in ('tools/build_p3b_pool.py','tools/build_entry_pools.py',
            'tools/build_p2a_pools.py','tools/build_pools.py','goalreach/entry_kinematics.py','goalreach/kinematics.py')},
        reference_files={str(k):sha256(k) for k in (KINEMATICS,REFERENCE,Path('reference/source_manifest.json'))},
        training=dict(path=str(path),size=SIZE,bytes=temporary.stat().st_size,file_sha256=sha256(temporary),
                      content_sha256=content_hash(data)),
        evaluation=dict(path=str(EVALUATION),size=EVAL_SIZE,cells=42,file_sha256=sha256(EVALUATION),
                        content_sha256=content_hash(evaluation)),audit=audit)
    if args.expected:
        expected=json.loads(args.expected.read_text())
        for role in ('training','evaluation'):
            for key in ('size','file_sha256','content_sha256'):
                require(expected[role][key]==manifest[role][key],f'Mismatched {role} {key}')
        for key in ('schema','seed','block_size','generator_files','reference_files'):
            require(expected[key]==manifest[key],f'Mismatched generator binding {key}')
    temporary.replace(path)
    (args.out/'build_manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    print(json.dumps(manifest,indent=2,allow_nan=False))


if __name__=='__main__':
    main()
