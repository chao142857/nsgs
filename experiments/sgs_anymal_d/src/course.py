"""Figure-7 task geometry, births and catalogs. Units are metres, yaw is radians.

All geometric values and patch priors below are local reconstruction choices.
Each bank uses the full Cartesian product of two independently drawn sets.
"""
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np

from base_course import Tile, height_at, surface as previous_surface
from terrain_compat import (TERRAIN_NAMES, TERRAIN_SEEDS, LEVELS, PATCHES, TILE_COUNT,
                           TASK_COUNT, GRID, DROP, HALF_WIDTH, RESOLUTION, encode, decode, mesh)

GEOMETRY_VERSION = "sgs-fig7-physical-patches-v3"
BIRTH_REPLACEMENTS = {}
ISLAND = TERRAIN_NAMES.index("islands")
axis = (np.arange(GRID) + .5) * RESOLUTION - HALF_WIDTH
X,Y = np.meshgrid(axis, axis, indexing="ij")
# Native ANYmal-D nominal reset foot-link offsets, rounded from articulation readback.
FOOT = np.array([[.507,.318],[-.507,.318],[.507,-.318],[-.507,-.318]])
TOE = np.array([[0,0],[.035,0],[-.035,0],[0,.035],[0,-.035]])
FOOT_SAMPLES = (FOOT[:,None,:] + TOE).reshape(-1,2)
BASE_CLEARANCE = .035
MAX_FOOT_RELIEF = .13
SETTLE_MAX = .16
POSITION_RADIUS, POSITION_DZ = .40,.36
CLASS_NAMES = ["birth_near","same_platform","obstacle"]
DOMAINS = {
    'gap':'jump gap', 'pit':'enter and leave depression', 'radial_beam':'cross spoke hub',
    'stairs':'traverse several risers', 'parallel_box':'cross parallel boxes',
    'balance_beam':'cross narrow beam', 'stepping_stones':'cross three isolated footholds',
    'jump_box':'jump between disconnected boxes', 'maze':'navigate around walls', 'contour':'traverse uneven relief',
    'islands':'transfer between disconnected islands', 'climb_box':'climb or descend elevated boxes',
    'inverted_slope':'descend and ascend the trough',
}

def number(kind):
    return TERRAIN_NAMES.index(kind) if isinstance(kind,str) else int(kind)

def islands(seed,level):
    rng=np.random.default_rng([10227,seed,level])
    pitch=2.10+.045*level
    sites=[]
    for row in range(3):
        for col in range(3):
            x,y=np.array([(col-1)*pitch,(row-1)*pitch])+rng.uniform(-.045,.045,2)
            z=.18+(.36+.045*level if (row,col)==(1,1) else 0.)
            sites.append((x,y,z,.87+rng.uniform(0,.06),row,col))
    return sites

@lru_cache(maxsize=260)
def maps(kind,seed,level):
    k=number(kind); name=TERRAIN_NAMES[k]; d=level/9
    a=np.full((GRID,GRID),DROP); ids=np.full(a.shape,-1,np.int16)
    def paint(mask,z,region):
        a[mask]=z if np.isscalar(z) else z[mask]
        ids[mask]=region
    if name=='islands':
        for i,(x,y,z,h,_,_) in enumerate(islands(seed,level)):
            paint((abs(X-x)<=h)&(abs(Y-y)<=h),z,i)
    elif name in ('stepping_stones','jump_box','climb_box','parallel_box'):
        # Three longitudinal rows of genuinely disconnected boxes, including wide
        # end platforms. Some tread interiors can accommodate a full nominal stance.
        rng=np.random.default_rng([171,k,seed,level])
        lim=np.abs(Y)<2.75
        paint((X< -2.52)&(X> -5.25)&lim,0.,0)
        finish=.38+.38*d if name=='climb_box' else 0.
        paint((X>2.52)&(X<5.25)&lim,finish,1)
        for i in range(3):
            for j in range(3):
                x=(i-1)*1.62
                y=(j-1)*1.68
                if name!='parallel_box':x+=rng.uniform(-.09,.09);y+=rng.uniform(-.10,.10)
                hx=.68-.065*d; hy=.67-.075*d
                if name=='stepping_stones':hx=max(.645,hx-.01);hy-=.055
                if name=='climb_box': z=(i+1)*finish/3
                elif name=='parallel_box':z=.10+(.06+.22*d)*(i%2)
                elif name=='jump_box':z=.12+rng.uniform(0,.22+.20*d)
                else:z=.07+rng.uniform(0,.08+.075*d)
                paint((abs(X-x)<=hx)&(abs(Y-y)<=hy),z,10+3*i+j)
    elif name=='radial_beam':
        r=np.hypot(X,Y); phase=np.arctan2(Y,X); step=np.pi/4
        spoke=np.rint(phase/step).astype(int)%8
        delta=phase-np.rint(phase/step)*step
        for j in range(8):
            theta=j*step
            cap=(abs(X-3.18*np.cos(theta))<.78)&(abs(Y-3.18*np.sin(theta))<.78)
            stick=(r<3.18)&(spoke==j)&(r*np.abs(np.sin(delta))<.51-.10*d)
            paint(cap|stick,.04,j+1)
        paint(r<.85,.04,0)
    elif name=='stairs':
        rise=.075+.095*d
        steps=np.clip(np.floor((X+3.15)/.90).astype(int)+1,0,8)
        inside=(abs(Y)<2.7)&(abs(X)<5.4)
        a[inside]=steps[inside]*rise; ids[inside]=steps[inside]
    else:
        a=previous_surface(k,seed,level)
        if name in ('contour','inverted_slope'):
            # Isolated rest patches in a side lane, embedded at local elevation.
            # The centre lane and its full section retain their original relief.
            for cx in (-1.25,0.,1.25):
                cy=(1.42 if cx==0 else -1.42) * (-1 if seed else 1)
                h=height_at(a,cx,cy)
                a[(abs(X-cx)<.71)&(abs(Y-cy)<.67)]=h
        valid=a>DROP+.01
        if name=='gap': ids[valid]=(X[valid]>0).astype(int)
        elif name=='pit':ids[valid]=np.where(a[valid]<-.1,1,np.where(X[valid]<0,0,2))
        elif name=='maze':
            ids[valid]=np.searchsorted([-.98,0,.98],X[valid])
            ids[(a>.6)&valid]=-2
        elif name in ('inverted_slope','contour'):
            ids[valid]=np.searchsorted([-1.9,-.38,.38,1.9],X[valid])
        elif name=='balance_beam':
            # The native nominal paw centres span 0.636 m. The narrow middle
            # is an evaluated traversal; not every beam location is a safe birth.
            ids[valid]=np.where(X[valid]<-1.9,0,np.where(X[valid]>1.9,2,1))
        else:raise KeyError(name)
    a=np.round(a,5);a.setflags(write=False);ids.setflags(write=False)
    return a,ids

def surface(kind,seed,level):return maps(number(kind),seed,level)[0]
def region_grid(kind,seed,level):return maps(number(kind),seed,level)[1]

def lookup(grid,x,y,missing):
    x,y=np.broadcast_arrays(x,y)
    i=np.floor((x+HALF_WIDTH)/RESOLUTION).astype(int)
    j=np.floor((y+HALF_WIDTH)/RESOLUTION).astype(int)
    val=np.where((i>=0)&(i<GRID)&(j>=0)&(j<GRID),grid[i.clip(0,GRID-1),j.clip(0,GRID-1)],missing)
    return val.item() if val.ndim==0 else val

def region_at(kind,seed,level,x,y): return lookup(region_grid(kind,seed,level),x,y,-1)
def surface_at(kind,seed,level,x,y): return lookup(surface(kind,seed,level),x,y,DROP)

def footprint(x,y,yaw):
    u,v=FOOT_SAMPLES.T
    co,si=np.cos(yaw),np.sin(yaw)
    return x+co*u-si*v,y+si*u+co*v

def base_yaw(name,zone,x,y,seed):
    if name=='radial_beam' and zone>0:return (zone-1)*np.pi/4
    if name=='stairs' and 0<zone<8:return np.pi/2
    if name=='maze' and zone in (1,2):return np.pi/2
    if name in ('contour','inverted_slope') and zone in (1,3):return np.pi/2
    if name=='balance_beam' and zone==1:return 0.
    return (seed%2)*np.pi

@dataclass(frozen=True)
class PatchSet:
    xyz: np.ndarray
    region: np.ndarray
    yaw: np.ndarray
    lift: np.ndarray
    relief: np.ndarray
    attempts: int

def _zone_order(name,available):
    # The roster controls only the region of each birth/goal, never which
    # pairs are discarded. Wide approach/exit aprons are prevented from
    # drowning out a central feature solely because they have more area.
    if name=='balance_beam':return [0,2] # middle births need verified available paw width
    return sorted(available)

@lru_cache(maxsize=1040)
def patchset(kind,seed,level,split,side):
    k=number(kind);name=TERRAIN_NAMES[k];grid,regions=maps(k,seed,level)
    rng=np.random.default_rng([17221,k,seed,level,split,side])
    ids=_zone_order(name,np.unique(regions[regions>=0]).tolist())
    if name=='balance_beam' and level<=2:ids=[0,1,2]
    # A uniform pass per region, with independently permuted remainder.
    schedule=[]
    while len(schedule)<20:schedule.extend(rng.permutation(ids).tolist())
    # Pick actual raster footprints; cap dead approach space and wall proximity.
    sites={z:np.argwhere((regions==z)&(abs(X)<4.45)&(abs(Y)<3.72)) for z in ids}
    points=[];yaws=[];lifts=[];reliefs=[];attempts=0
    for zone in schedule[:20]:
        cells=sites[zone]
        for it in range(2000):
            attempts+=1;i,j=cells[rng.integers(len(cells))]
            x=axis[i]+rng.uniform(-.06,.06);y=axis[j]+rng.uniform(-.06,.06)
            yaw=base_yaw(name,zone,x,y,seed)+rng.uniform(-.12,.12)
            fx,fy=footprint(x,y,yaw)
            foot=lookup(grid,fx,fy,DROP); support=lookup(regions,fx,fy,-1)
            h=grid[i,j]
            if (support<0).any() or foot.max()-foot.min()>MAX_FOOT_RELIEF:continue
            if max(abs(foot-h))>SETTLE_MAX:continue
            if name in ('islands','jump_box','parallel_box','climb_box','stepping_stones') and (support!=zone).any():continue
            if name=='maze' and (support!=zone).any():continue
            points.append([x,y,h]);yaws.append(yaw);lifts.append(max(0.,float(foot.max()-h))+BASE_CLEARANCE)
            reliefs.append(float(np.ptp(foot)));break
        else:raise RuntimeError((name,seed,level,split,side,zone,'no valid paw footprint'))
    return PatchSet(np.array(points),np.asarray(schedule[:20]),np.asarray(yaws),np.asarray(lifts),np.asarray(reliefs),attempts)

def points(kind,seed,level,split):
    return [(p.xyz,p.region) for p in (patchset(kind,seed,level,split,v) for v in (0,1))]

def endpoints(grid,kind,seed,level,split):
    return tuple(tuple(map(tuple,xyz)) for xyz,_ in points(kind,seed,level,split))

@lru_cache(maxsize=260)
def tile(kind,seed,level):
    s,g=endpoints(None,kind,seed,level,0)
    return Tile(tuple(map(tuple,surface(kind,seed,level))),s,g)

def birth_lift(task_id):
    k,seed,level,i,_=decode(task_id)
    return float(patchset(k,seed,level,0,0).lift[i])

def island_class(a,b):
    if a==b:return 'same_island'
    if b==4:return 'up_to_center'
    if a==4:return 'down_from_center'
    if abs(a//3-b//3)+abs(a%3-b%3)==1:return 'level_neighbors'
    return 'other_separate_outer'

def pair_class(name,zone0,zone1,near):
    if near:return 'birth_near'
    if zone0==zone1:return 'same_platform'
    return 'obstacle'

def route_type(name,a,b):
    if name=='islands':return island_class(a,b)
    if a==b:return 'same_zone'
    if name=='stairs':return 'stairs_up' if b>a else 'stairs_down'
    if name=='radial_beam':return 'spokes_via_hub' if a*b and min((a-b)%8,(b-a)%8)>=2 else 'hub_or_adjacent'
    if name in ('stepping_stones','jump_box','parallel_box','climb_box'):
        if a>=10 and b>=10:return 'object_to_object'
        if {a,b}=={0,1}:return 'through_object_field'
        return 'object_to_apron'
    if name in ('inverted_slope','contour'):return 'full_relief' if {a,b}=={0,4} else 'local_relief'
    if name=='maze':return 'multi_gate' if abs(a-b)>=2 else 'single_gate'
    if name in ('pit','balance_beam'):return 'full_crossing' if {a,b}=={0,2} else 'partial_crossing'
    return 'cross_gap'

def fixed_pairs(name,seed,level,starts,goals,grid):
    """Select routes by geometry only before seeing any policy."""
    if level!=4:return []
    if seed and name!='stairs':return []
    a,b=starts.region,goals.region
    candidates=[]
    for i,p in enumerate(starts.xyz):
        for j,q in enumerate(goals.xyz):
            delta=q-p
            length=np.linalg.norm(delta[:2]);u,v=int(a[i]),int(b[j])
            if length<.9:continue
            line=p[None,:2]+np.linspace(0,1,120)[:,None]*(q-p)[None,:2]
            zones=region_at(name,seed,level,line[:,0],line[:,1])
            center=float(abs(p[1])+abs(q[1]))
            candidates.append((i,j,u,v,length,delta[2],center,zones))
    wanted={
        'islands':['level_neighbors','up_to_center','down_from_center'],
        'stairs':['stairs_up'] if not seed else ['stairs_down'],
    }.get(name,[name])
    out=[]
    for which in wanted:
        matches=[]
        for i,j,u,v,length,dz,center,zones in candidates:
            typ=route_type(name,u,v)
            if name=='islands':ok=typ==which and (which=='level_neighbors' or abs(u//3-v//3)+abs(u%3-v%3)==1)
            elif name=='stairs':ok=(which=='stairs_up' and u==0 and v==8) or (which=='stairs_down' and u==8 and v==0)
            elif name=='radial_beam':ok=u*v and min((u-v)%8,(v-u)%8) in (2,3) and -1 in zones
            elif name=='maze':ok={u,v}=={0,3} and -2 in zones
            elif name in ('balance_beam','pit'):ok=(u,v)==(0,2)
            elif name in ('contour','inverted_slope'):ok=(u,v)==(0,4)
            elif name=='jump_box':ok=(u,v)==(11,14) and -1 in zones
            elif name=='climb_box':ok=u==0 and v>=16 and dz>.32
            elif name in ('stepping_stones','parallel_box'):ok=(u,v)==(0,1)
            else:ok=(u,v)==(0,1)
            if ok:
                # Shorter representative crossing first, then smaller lateral excursion.
                matches.append((length+.16*center,i,j,u,v))
        if not matches:raise ValueError(('fixed route absent',name,which,seed))
        _,i,j,u,v=min(matches)
        k=TERRAIN_NAMES.index(name);p=starts.xyz[i];q=goals.xyz[j]
        out.append(dict(terrain=name,route=which,id=encode(k,seed,level,i,j),terrain_seed=seed,level=level,
             start_index=i,goal_index=j,start_support=u,goal_support=v,start=p.tolist(),goal=q.tolist(),
             horizontal_m=float(np.linalg.norm(q[:2]-p[:2])),delta_z_m=float(q[2]-p[2]),local_event=DOMAINS[name]))
    return out

def check_feature(grid,name,regions):
    assert all(np.all(row==DROP) for row in (grid[0],grid[-1],grid[:,0],grid[:,-1]))
    if name=='islands':
        assert set(np.unique(regions))=={-1,*range(9)}
        # No horizontal, vertical, or diagonal contact between island surfaces.
        for sx,sy in ((1,0),(0,1),(1,1),(1,-1)):
            other=np.roll(regions,(sx,sy),axis=(0,1))
            assert not np.any((regions>=0)&(other>=0)&(other!=regions))
        assert grid[regions==4].min()-grid[(regions>=0)&(regions!=4)].max()>=.35
    elif name in ('jump_box','stepping_stones','parallel_box','climb_box'):
        assert set(range(10,19))<=set(np.unique(regions))
        assert np.any(np.all(grid[abs(axis)<1.8]==DROP,axis=1))
    elif name=='maze':
        for x in [-.98,0,.98]:
            col=regions[np.argmin(abs(axis-x)),abs(axis)<2.25]
            assert -2 in col and np.any(col>=0)
    elif name=='stairs':assert len(np.unique(grid[regions>=0]))==9
    elif name=='radial_beam':assert set(range(9))<=set(np.unique(regions))
    elif name=='gap':assert np.any(np.all(grid[abs(axis)<1.5]==DROP,axis=1))
    elif name in ('contour','pit','inverted_slope'):assert np.ptp(grid[regions>=0])>.15

def build(root):
    root=Path(root);root.mkdir(exist_ok=True,parents=True)
    banks=[np.empty((TASK_COUNT,7),np.float32) for _ in range(2)]
    ys=[np.empty(TASK_COUNT,np.float32) for _ in range(2)]
    cs=[np.empty(TASK_COUNT,np.uint8) for _ in range(2)]
    chains=[np.empty(TASK_COUNT,np.uint8) for _ in range(2)]
    names=[];per_layout=[];routes=[];allpoints=[];summaries=[];geom_hash=[]
    for k,name in enumerate(TERRAIN_NAMES):
        aggregate=[Counter(),Counter()]
        for seed in range(2):
            for level in range(10):
                grid,regions=maps(k,seed,level)
                check_feature(grid,name,regions);geom_hash.append(hashlib.sha256(grid.tobytes()).hexdigest())
                for split in range(2):
                    s=patchset(k,seed,level,split,0);g=patchset(k,seed,level,split,1)
                    first=encode(k,seed,level,0,0); sl=slice(first,first+400)
                    a=np.repeat(s.xyz,20,axis=0);b=np.tile(g.xyz,(20,1))
                    banks[split][sl]=np.c_[a,b,np.repeat(s.lift,20)]
                    ys[split][sl]=np.repeat(s.yaw,20)
                    dist=np.linalg.norm(b[:,:2]-a[:,:2],axis=1);dz=b[:,2]-a[:,2]
                    same=np.repeat(s.region,20)==np.tile(g.region,20)
                    near=(dist<=POSITION_RADIUS)&(abs(dz)<=POSITION_DZ)
                    cls=np.where(near,0,np.where(same,1,2))
                    cs[split][sl]=cls
                    route_keys=[route_type(name,int(u),int(v)) for u in s.region for v in g.region]
                    for key in route_keys:
                        if key not in names:names.append(key)
                    chains[split][sl]=[names.index(x) for x in route_keys]
                    classes=dict(zip(CLASS_NAMES,np.bincount(cls,minlength=3).tolist()))
                    per_layout.append(dict(terrain=name,terrain_seed=seed,level=level,split=split,classes=classes,
                           same_support_including_near=int(same.sum()),routes=dict(Counter(route_keys)),born_near=int(near.sum()),
                           dist_min_m=float(dist.min()),dist_median_m=float(np.median(dist)),dist_max_m=float(dist.max()),
                           dz_min_m=float(dz.min()),dz_max_m=float(dz.max()),proposal_count=s.attempts+g.attempts,
                           max_foot_relief_m=float(max(s.relief.max(),g.relief.max()))))
                    aggregate[split].update(classes)
                    for side,p in [('S',s),('G',g)]:
                        for j,(xyz,zone,yaw,relief,lift) in enumerate(zip(p.xyz,p.region,p.yaw,p.relief,p.lift)):
                            allpoints.append([name,seed,level,split,side,j,int(zone),*xyz.tolist(),yaw,relief,lift])
                    if split:routes.extend(fixed_pairs(name,seed,level,s,g,grid))
        summaries.append(dict(terrain=name,train_classes=dict(aggregate[0]),holdout_classes=dict(aggregate[1]),local_skill=DOMAINS[name]))
    # Check all coordinates across corresponding and noncorresponding indices within each layout.
    for p,q in zip(banks[0].reshape(260,400,7),banks[1].reshape(260,400,7)):
        assert not set(map(tuple,p[:,:6])).intersection(map(tuple,q[:,:6]))
    for i,label in enumerate(('train','holdout')):
        for suffix,data in (('_yaw',ys[i]),('_classes',cs[i]),('_route_types',chains[i])):
            np.save(root/(label+suffix+'.npy'),data,allow_pickle=False)
        np.save(root/(('geometry' if i==0 else 'holdout')+'.npy'),banks[i],allow_pickle=False)
    result=dict(version=GEOMETRY_VERSION,layouts=260,n_per_bank=TASK_COUNT,classes=CLASS_NAMES,route_types=names,
            terrains=summaries,geometry_sha256=[hashlib.sha256(a.tobytes()).hexdigest() for a in banks],
            distinct_mesh_height_hashes=len(set(geom_hash)),collision_sha256=hashlib.sha256(''.join(geom_hash).encode()).hexdigest(),
            yaw_sha256=[hashlib.sha256(a.tobytes()).hexdigest() for a in ys],
            pool_seed=[17221,'kind','terrain_seed','level','split','role'],exact_train_holdout_rows_in_common=0)
    import csv
    with (root/'coordinates.csv').open('w',newline='') as out:
        writer=csv.writer(out);writer.writerow('terrain seed level split role index support x_m y_m z_m yaw_rad paw_relief_m birth_clearance_offset_m'.split());writer.writerows(allpoints)
    for key,data in (('catalog',result),('fixed_routes',routes),('layout_statistics',per_layout)):
        (root/(key+'.json')).write_text(json.dumps(data,indent=2)+'\n')
    return result

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path)
    print(build(parser.parse_args().output)['geometry_sha256'])
