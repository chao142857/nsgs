"""Rebuild archived V1 and independently count V3 positions and mesh properties."""
from collections import Counter
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import base_course
import course as c

def profile(row, samples=4001):
    k=c.number(row['terrain']);seed,lev=row['terrain_seed'],row['level']
    p,q=np.array(row['start']),np.array(row['goal']);t=np.linspace(0,1,samples)
    xy=p[:2]+t[:,None]*(q[:2]-p[:2])
    z=c.surface_at(k,seed,lev,xy[:,0],xy[:,1]); ids=c.region_at(k,seed,lev,xy[:,0],xy[:,1])
    dist=float(np.linalg.norm(q[:2]-p[:2]))
    bad=np.flatnonzero(ids<0)
    sample_length=dist/(samples-1)
    return dict(id=row['id'],terrain=row['terrain'],route=row['route'],min_m=float(z.min()),max_m=float(z.max()),
                horizontal_m=dist,delta_z_m=float(q[2]-p[2]),delta_xy_section_void_m=float(np.count_nonzero(ids==-1)*sample_length),
                straight_line_intersects_wall=bool(np.any(ids==-2)),
                supports_along_chord=[int(x) for x in ids[np.r_[True,ids[1:]!=ids[:-1]]]],
                first_intersection_distance=None if not len(bad) else float(t[bad[0]]*dist))

def audit(bank_dir,output_dir):
    output_dir.mkdir(exist_ok=True,parents=True)
    catalog=json.loads((bank_dir/'catalog.json').read_text())
    with TemporaryDirectory() as tmp:
        old_info=base_course.build(Path(tmp))
        assert old_info['geometry_sha']==['d6d21d6bb94dcdb6ef2c968abd3ffed54751ca2b039ee61d53dec64547ea568f','6447f24738a21925a0bc92ee7dbab4ad6ebe18a781dbc34c2a2bb023f53e7148']
        result={}
        for index,(name,oldfile) in enumerate((('train','geometry.npy'),('holdout','holdout.npy'))):
            original=np.load(Path(tmp)/oldfile)
            current=np.load(bank_dir/oldfile); split=index
            assert hashlib.sha256(current.tobytes()).hexdigest()==catalog['geometry_sha256'][index]
            old_delta=original[:,3:6]-original[:,:3]
            old_near=(np.linalg.norm(old_delta[:,:2],axis=1)<=.4)&(abs(old_delta[:,2])<=.36)
            old_cross=original[:,0]*original[:,3]<0
            new_delta=current[:,3:6]-current[:,:3]
            new_distance=np.linalg.norm(new_delta[:,:2],axis=1)
            new_near=(new_distance<=.4)&(abs(new_delta[:,2])<=.36)
            classes=np.load(bank_dir/(name+'_classes.npy'))
            assert np.array_equal((classes==0),new_near)
            yaws=np.load(bank_dir/(name+'_yaw.npy')).reshape(260,20,20)
            assert (yaws==yaws[:,:,:1]).all()
            data=current.reshape(260,20,20,7)
            assert np.all(data[:,:,:,:3]==data[:,:,:1,:3]) and np.all(data[:,:,:,3:6]==data[:,:1,:,3:6])
            cnt=dict(zip(c.CLASS_NAMES,np.bincount(classes,minlength=3).tolist()))
            result[name]=dict(old=dict(configurations=len(original),born_within_position_threshold=int(old_near.sum()),
                                  cross_aprons=int(old_cross.sum()),same_apron_non_near=int((~old_cross&~old_near).sum())),
                         new=dict(classes=cnt,fraction_born_near=float(new_near.mean()),
                             distance_quantiles_m=np.quantile(new_distance,[0,.25,.5,.75,1]).tolist(),
                             delta_height_quantiles_m=np.quantile(new_delta[:,2],[0,.25,.5,.75,1]).tolist(),
                             exact_sources_cartesian_20_by_20=True, birth_yaw_independent_of_goal=True))
    routes=json.loads((bank_dir/'fixed_routes.json').read_text())
    islands=[r for r in routes if r['terrain']=='islands']
    def island_gap(r):
        ids=c.region_grid('islands',r['terrain_seed'],r['level'])
        src=np.argwhere(ids==r['start_support']);dst=np.argwhere(ids==r['goal_support'])
        dxy=np.maximum(0,np.abs(src[:,None]-dst)-1)*c.RESOLUTION
        return float(np.linalg.norm(dxy,axis=2).min())
    section=[profile(row) for row in routes]
    result.update(section=section,source_region_edge_gaps_m={str(r['id']):island_gap(r) for r in islands},
                  foot_offset_m=c.FOOT.tolist(),sample_disk_radius_m=.035,
                  source_geometry_v1=old_info['geometry_sha'],source_geometry_v3=catalog['geometry_sha256'])
    wall=next(x for x in section if x['terrain']=='maze');assert wall['straight_line_intersects_wall']
    for route,section_row in zip(routes,section):
        if route['terrain']=='islands':
            assert section_row['delta_xy_section_void_m']>.14
        if route['terrain']=='stairs':
            assert set(range(9)).issubset(section_row['supports_along_chord'])
    # The connected wall blocks the direct chord; an actual traversable path on
    # the mesh remains. Erode floor by an inscribed 0.3-m radius before the check.
    maze=next(r for r in routes if r['terrain']=='maze')
    m=c.region_grid('maze',maze['terrain_seed'],maze['level'])>=0
    disk=[(i,j) for i in range(-2,3) for j in range(-2,3) if i*i+j*j<=4]
    clearance=np.logical_and.reduce([np.roll(m,(dx,dy),(0,1)) for dx,dy in disk])
    def cell(p):return tuple(np.floor((np.asarray(p[:2])+c.HALF_WIDTH)/c.RESOLUTION).astype(int))
    start,goal=cell(maze['start']),cell(maze['goal']);stack=[start]
    visited={start}
    while goal not in visited and stack:
        i,j=stack.pop()
        for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
            n=(i+dx,j+dy)
            if 0<=n[0]<c.GRID and 0<=n[1]<c.GRID and n not in visited and clearance[n]:
                visited.add(n);stack.append(n)
    assert goal in visited
    result['maze_has_floor_path_after_0p30m_erosion']=True
    (output_dir/'geometry_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('bank',type=Path);p.add_argument('output',type=Path);args=p.parse_args()
    print(audit(args.bank,args.output)['train'])
