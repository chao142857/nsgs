"""Build and hash the active development pools identically on every host."""
from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile, ZipInfo

import numpy as np

from build_pools import COUNTS, NATIVE_META, NATIVE_SHA, OLD_MAP_SHA, put, random_units
from protocol import BOUNDS, file_sha, schedule


def store_pool(path, pool):
    with ZipFile(path,'x',compression=ZIP_STORED) as archive:
        for name,array in sorted(pool.items()):
            entry=ZipInfo(name+'.npy',date_time=(1980,1,1,0,0,0))
            entry.create_system=3
            entry.external_attr=0o600 << 16
            with archive.open(entry,'w',force_zip64=False) as stream:
                np.lib.format.write_array(stream,array,version=(1,0),allow_pickle=False)


def bin_index(values, edges, *, periodic=False):
    values = np.clip(values,edges[0],edges[-1])
    if periodic:
        values = np.where(values==edges[-1],edges[0],values)
    return np.minimum(np.searchsorted(edges,values,side='right')-1,len(edges)-2)


def draw(pool, key, plan):
    ids = pool['task_id']
    lo,hi = plan['mu_uniform_proposal']
    mu = lo + (hi-lo)*random_units(ids,key,0,1)
    lower,upper = plan['alpha_deg_uniform_proposal']
    analysis_edges = plan['geometry_report_edges']
    groups = [bin_index(pool['horizontal_distance'],analysis_edges['distance_m']),
              bin_index(pool['azimuth'],analysis_edges['azimuth_rad'],periodic=True),
              bin_index(pool['height'],analysis_edges['target_height_m'])]
    alpha_edges = plan['alpha_strata_deg']
    cube = tuple(len(edges)-1 for edges in analysis_edges.values()) + (len(alpha_edges)-1,)
    proposals,rejected,accepted = (np.zeros(cube,dtype=np.int64) for _ in range(3))
    initial_proposals,initial_rejected = np.zeros_like(proposals),np.zeros_like(proposals)
    alpha = np.empty(len(ids),dtype='<f8')
    retries = np.zeros(len(ids),dtype='<u2')
    first_alpha = np.empty_like(alpha)
    pending = np.arange(len(ids))
    for attempt in range(64):
        candidate = lower+(upper-lower)*random_units(ids[pending],key,attempt,0)
        alpha[pending] = candidate
        angles = bin_index(candidate,alpha_edges)
        strata = tuple(group[pending] for group in groups)+(angles,)
        xyz = pool['target'][pending]
        misses = xyz[:,2]-xyz[:,0]*np.tan(np.deg2rad(candidate)) <= plan['valid_target_vertical_clearance_m']
        np.add.at(proposals,strata,1)
        np.add.at(rejected,tuple(group[misses] for group in strata),1)
        np.add.at(accepted,tuple(group[~misses] for group in strata),1)
        if not attempt:
            first_alpha[:] = candidate
            initial_proposals[:],initial_rejected[:] = proposals,rejected
        pending = pending[misses]
        retries[pending] += 1
        if not len(pending):
            break
    else:
        raise ValueError('Could not place a fixed target above the plane')
    np.testing.assert_equal(proposals,rejected+accepted)
    assert accepted.sum()==len(ids)
    pool.update(mu=mu,alpha_deg=alpha,alpha_rejections=retries,first_proposed_alpha_deg=first_alpha)
    mu_strata=bin_index(mu,plan['mu_strata'])
    alpha_strata=bin_index(alpha,alpha_edges)
    observed=np.zeros((len(plan['mu_strata'])-1,len(alpha_edges)-1),dtype=np.int64)
    np.add.at(observed,(mu_strata,alpha_strata),1)
    return dict(axis_order=['distance','azimuth','height','proposed_or_accepted_alpha'],
        initial_proposals=initial_proposals.tolist(),initial_rejected=initial_rejected.tolist(),
        all_proposals=proposals.tolist(),all_rejected=rejected.tolist(),accepted=accepted.tolist(),
        initial_slope_totals=initial_proposals.sum(axis=(0,1,2)).tolist(),
        accepted_slope_totals=accepted.sum(axis=(0,1,2)).tolist(),
        initial_rejections=int(initial_rejected.sum()),total_rejections=int(rejected.sum()),
        redrawn_configurations=int(np.count_nonzero(retries)),
        accepted_mu_by_alpha_counts=observed.tolist(),median_proposal_mu=float(np.median(mu)))


def grid_map(pool, bins, plan):
    axes=['horizontal_distance','azimuth','height','mu','alpha_deg']
    bounds=(*BOUNDS[:3],plan['mu_uniform_proposal'],plan['alpha_deg_uniform_proposal'])
    ids=np.zeros(len(pool['task_id']),dtype=np.int32)
    for n,axis,(lo,hi) in zip(bins,axes,bounds):
        values=np.clip(pool[axis].astype(np.float64),lo,hi)
        if axis=='azimuth':
            values[values==hi]=lo
        bucket=np.minimum(np.floor((values-lo)*(n/(hi-lo))).astype(np.int32),n-1)
        ids=ids*n+bucket
    _,counts=np.unique(ids,return_counts=True)
    return dict(bins=bins,total=int(np.prod(bins)),occupied=len(counts),
        configs_per_occupied_cell=np.quantile(counts,[0,.25,.5,.75,1]).tolist(),
        row_cell_sha256=sha256(ids.astype('<i4',copy=False).tobytes()).hexdigest())


def build(train,evaluation,old_mapping,preregistration,directory):
    assert {'train':file_sha(train),'evaluation':file_sha(evaluation)}==NATIVE_SHA
    assert file_sha(old_mapping)==OLD_MAP_SHA
    plan=json.loads(preregistration.read_bytes())
    directory.mkdir(parents=True,exist_ok=False)
    old=json.loads(old_mapping.read_bytes())['rows']
    manifest=dict(version=plan['version'],authorized_to_dispatch_formal=False,
        preregistration_sha256=file_sha(preregistration),generator_sha256=file_sha(Path(__file__)),
        native_input_sha256=NATIVE_SHA,panels={})
    geometry={}
    for split,(count,copies) in COUNTS.items():
        original=np.repeat(np.arange(count,dtype='<i4'),copies)
        with np.load(train if split=='train' else evaluation,allow_pickle=False) as native:
            pool={name:native[name] if name in NATIVE_META else native[name][original]
                  for name in native.files if name!='task_id'}
        ids=np.arange(len(original),dtype='<i4')
        pool.update(task_id=ids,config_id=ids.copy(),target_index=original)
        geometry[split]=draw(pool,plan['panel_proposal_streams'][split],plan)
        path=directory/f'{split}.npz'
        store_pool(path,pool)
        native_region=np.bincount(pool['region'],minlength=4)
        redrawn=np.bincount(pool['region'][pool['alpha_rejections']>0],minlength=4)
        manifest['panels'][split]=dict(sha256=file_sha(path),rows=len(ids),
            region_counts=native_region.tolist(),redrawn_region_counts=redrawn.tolist(),
            redrawn_region_rate=(redrawn/native_region).tolist(),
            redrawn_configurations=geometry[split]['redrawn_configurations'],
            condition_quantiles={axis:np.quantile(pool[axis],[0,.01,.25,.5,.75,.99,1]).tolist()
                for axis in ('mu','alpha_deg')})
        if split=='train':
            grid=plan['grid_selection']
            dims=[*grid['candidates'],grid['double_budget_sensitivity'],[16]*3,[16]*5,[32]*3,[32]*5]
            manifest['train_grid']={','.join(map(str,bins)):grid_map(pool,bins,plan) for bins in dims}
            proposed=dict(pool,alpha_deg=pool['first_proposed_alpha_deg'])
            manifest['uniform_proposal_grid']={','.join(map(str,bins)):grid_map(proposed,bins,plan) for bins in dims}
        if split=='development':
            pick=np.array([row['original_task_id'] for row in old],dtype=np.intp)
            compact={name:values if name in NATIVE_META else values[pick] for name,values in pool.items()}
            compact['task_id']=np.arange(504,dtype='<i4')
            compact['target_index']=np.array([row['target_index'] for row in old],dtype='<i4')
            assert np.bincount(compact['cell_id'],minlength=42).tolist()==[12]*42
            path=directory/'development_504.npz'
            store_pool(path,compact)
            manifest['panels']['development_504']=dict(sha256=file_sha(path),rows=504)
            rows=[dict(task_id=i,original_task_id=int(large),original_target_index=int(pool['target_index'][large]),
                   target_index=int(compact['target_index'][i]),config_id=int(compact['config_id'][i]),
                   cell_id=int(compact['cell_id'][i]),mu=float(compact['mu'][i]),
                   alpha_deg=float(compact['alpha_deg'][i])) for i,large in enumerate(pick)]
            put(directory/'original_index_mapping.json',rows)
            manifest['mapping_sha256']=file_sha(directory/'original_index_mapping.json')
    put(directory/'geometry_density.json',geometry)
    manifest['geometry_density_sha256']=file_sha(directory/'geometry_density.json')
    put(directory/'pool_manifest.json',manifest)
    registered=schedule(manifest['panels']['development_504']['sha256'])
    registered['version']=plan['version']
    put(directory/'sparse_schedule.json',registered)
    return manifest


if __name__=='__main__':
    parser=ArgumentParser(description=__doc__)
    for key in ('train','evaluation','old_mapping','preregistration','directory'):
        parser.add_argument(key,type=Path)
    args=parser.parse_args()
    print(json.dumps(build(args.train,args.evaluation,args.old_mapping,args.preregistration,args.directory)))
