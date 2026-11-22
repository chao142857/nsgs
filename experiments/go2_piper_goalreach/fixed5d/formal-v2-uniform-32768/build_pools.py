"""Materialize fixed world targets and independently drawn physical conditions."""
from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from protocol import (ALPHA_BOUNDS_DEG, AXES, BOUNDS, DRAW_KEYS, MU_BOUNDS, PLATFORM_METRES,
                      TARGET_CLEARANCE_M, VERSION, file_sha, schedule)


NATIVE_SHA = {
    'train': '37e66100194b2e00058f299ab8af85e6a75c324e8b06f48c0df7701dbe90862b',
    'evaluation': 'd50591e1af543a1f3ff1a2630d624e457a02bd7ea015a5160d0afab55b837edf',
}
OLD_MAP_SHA = 'f78adac2e76b82bd4b53581cf403b87f3ef1033ea3643a75c04d51420379a97a'
NATIVE_META = {'nominal_ee', 'near_distance_bounds', 'schema_version', 'seed', 'action_joint_names',
               'reference_root_pose', 'reference_joint_position'}
COUNTS = {'train': (1048576,1), 'development': (2016,2), 'test': (2016,4)}


def put(path, value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def random_units(ids, key, draw, stream):
    # SplitMix64 with separate counter coordinates for each split and variable.
    x = np.asarray(ids,dtype=np.uint64) + np.uint64(key << 27)
    x ^= np.uint64((2*draw + stream) * 0x9E3779B97F4A7C15 & ((1 << 64)-1))
    x = (x ^ (x >> 30)) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> 27)) * np.uint64(0x94D049BB133111EB)
    return ((x ^ (x >> 31)) >> 11).astype(np.float64) * (2. ** -53)


def draw_conditions(targets, ids, key):
    mu = MU_BOUNDS[0] + (MU_BOUNDS[1]-MU_BOUNDS[0])*random_units(ids,key,0,1)
    alpha = np.empty(len(ids),dtype='<f8')
    rejections = np.zeros(len(ids),dtype='<u2')
    pending = np.arange(len(ids))
    for draw in range(64):
        unit = random_units(ids[pending],key,draw,0)
        alpha[pending] = ALPHA_BOUNDS_DEG[0] + (ALPHA_BOUNDS_DEG[1]-ALPHA_BOUNDS_DEG[0])*unit
        xyz = targets[pending]
        height = xyz[:,2] - xyz[:,0]*np.tan(np.deg2rad(alpha[pending]))
        pending = pending[height <= TARGET_CLEARANCE_M]
        if not len(pending):
            return mu, alpha, rejections
        rejections[pending] += 1
    raise RuntimeError('A fixed target could not be drawn above its support plane')


def occupancy(pool, dimensions, bins):
    ids = np.zeros(len(pool['task_id']),dtype=np.int32)
    for key,(lo,hi) in zip(AXES[:dimensions],BOUNDS):
        values = np.clip(pool[key].astype(np.float64),lo,hi)
        if key == 'azimuth':
            values[values == hi] = lo
        index = np.minimum(np.floor((values-lo)*(bins/(hi-lo))).astype(np.int32),bins-1)
        ids = ids*bins + index
    _, counts = np.unique(ids,return_counts=True)
    return dict(theoretical_cells=bins**dimensions,occupied_cells=len(counts),
        configurations_per_occupied_cell=np.quantile(counts,[0,.25,.5,.75,1]).tolist(),
        row_cell_sha256=sha256(ids.astype('<i4',copy=False).tobytes()).hexdigest())


def materialize(native, split):
    length, copies = COUNTS[split]
    assert np.array_equal(native['task_id'],np.arange(length))
    original = np.repeat(np.arange(length,dtype='<i4'),copies)
    ids = np.arange(len(original),dtype='<i4')
    pool = {name:native[name] if name in NATIVE_META else native[name][original]
            for name in native.files if name!='task_id'}
    target = pool['target']
    assert np.array_equal(target,native['target'][original])
    assert np.all(pool['initial_root_pose'][:,:2]==0.)
    assert np.linalg.norm(target[:,:2],axis=1).max() < PLATFORM_METRES[0]*.5-.5
    mu,alpha,rejected = draw_conditions(target,ids,DRAW_KEYS[split])
    pool.update(task_id=ids, config_id=ids.copy(), target_index=original,
                mu=mu, alpha_deg=alpha, alpha_rejections=rejected)
    return pool


def build(train, evaluation, old_mapping, directory):
    assert {'train':file_sha(train),'evaluation':file_sha(evaluation)} == NATIVE_SHA
    assert file_sha(old_mapping)==OLD_MAP_SHA
    directory.mkdir(parents=True,exist_ok=False)
    old_rows = json.loads(old_mapping.read_bytes())['rows']
    manifest = dict(version=VERSION,native_input_sha256=NATIVE_SHA,axes=list(AXES),
        bounds=[list(row) for row in BOUNDS],draw_keys=DRAW_KEYS,platform_m=list(PLATFORM_METRES),
        friction=dict(static='mu',dynamic='mu',restitution=0.,combine='multiply'),
        slope=dict(degrees=True,positive='rise along world +x',world_gravity_unchanged=True),
        world_target_rule='native xyz plus environment origin, unchanged across slopes',
        minimum_vertical_clearance_m=TARGET_CLEARANCE_M,
        rejection='Only redraw alpha when the native world target is on or beneath the support plane. Keep mu.',
        generator_sha256=file_sha(Path(__file__)),panels={})
    indexes = set()
    for split in COUNTS:
        with np.load(train if split=='train' else evaluation,allow_pickle=False) as native:
            pool = materialize(native,split)
        if split!='train':
            n = COUNTS[split][1]
            assert np.bincount(pool['cell_id'],minlength=42).tolist()==[84*n]*6+[42*n]*36
        path = directory/f'{split}.npz'
        np.savez(path,**pool)
        manifest['panels'][split] = dict(sha256=file_sha(path), rows=len(pool['task_id']),
            region_counts=np.bincount(pool['region'],minlength=4).tolist(),
            rejections=int(pool['alpha_rejections'].sum()),
            redrawn_configurations=int(np.count_nonzero(pool['alpha_rejections'])),
            maximum_redraws=int(pool['alpha_rejections'].max()),
            minimum_clearance_m=float((pool['target'][:,2]-pool['target'][:,0]*
                                       np.tan(np.deg2rad(pool['alpha_deg']))).min()),
            friction_percentiles=np.quantile(pool['mu'],[0,.25,.5,.75,1]).tolist(),
            slope_percentiles=np.quantile(pool['alpha_deg'],[0,.25,.5,.75,1]).tolist())
        pair_ids = set(zip(pool['mu'].tolist(),pool['alpha_deg'].tolist())) if split!='train' else set()
        assert not indexes.intersection(pair_ids)
        indexes.update(pair_ids)
        if split=='train':
            manifest['train_grid']={f'{dims}d_{bins}':occupancy(pool,dims,bins)
                                    for dims in (3,5) for bins in (16,32)}
        if split=='development':
            samples = np.array([row['original_task_id'] for row in old_rows],dtype=np.intp)
            compact = {name:values if name in NATIVE_META else values[samples] for name,values in pool.items()}
            compact['task_id'] = np.arange(504,dtype='<i4')
            compact['target_index'] = np.array([row['target_index'] for row in old_rows],dtype='<i4')
            assert np.bincount(compact['cell_id'],minlength=42).tolist()==[12]*42
            assert np.all(np.bincount(compact['target_index'])==2)
            path = directory/'development_504.npz'
            np.savez(path,**compact)
            rows = [dict(task_id=i,original_task_id=int(src),target_index=int(compact['target_index'][i]),
                         original_target_index=int(pool['target_index'][src]),config_id=int(compact['config_id'][i]),
                         cell_id=int(compact['cell_id'][i]),mu=float(compact['mu'][i]),
                         alpha_deg=float(compact['alpha_deg'][i])) for i,src in enumerate(samples)]
            put(directory/'original_index_mapping.json',rows)
            manifest['mapping_sha256']=file_sha(directory/'original_index_mapping.json')
            manifest['source_target_subset_sha256']=OLD_MAP_SHA
            manifest['panels']['development_504']=dict(sha256=file_sha(path), rows=504)
    put(directory/'pool_manifest.json',manifest)
    put(directory/'sparse_schedule.json',schedule(manifest['panels']['development_504']['sha256']))
    return manifest


if __name__=='__main__':
    CLI = ArgumentParser(description=__doc__)
    for name in ('train','evaluation','old_mapping','directory'):
        CLI.add_argument(name,type=Path)
    opts = CLI.parse_args()
    print(json.dumps(build(opts.train,opts.evaluation,opts.old_mapping,opts.directory)))
