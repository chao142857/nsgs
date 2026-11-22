"""Independent 13-family terrain atlas for the SGS ANYmal-D experiment."""
from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path

import numpy as np

from terrain_compat import TERRAIN_NAMES, decode, encode, mesh, GRID, RESOLUTION, HALF_WIDTH

TERRAIN_SEEDS, LEVELS, PATCHES = 2, 10, 20
TILE_COUNT, TASK_COUNT = 260, 104000
GEOMETRY_VERSION = 'sgs-paper32k-skill-courses-v1'
DROP = -1.8
BIRTH_REPLACEMENTS = {}


def birth_lift(task_id):
    return 0.


def height_at(grid, x, y):
    i, j = (int(np.floor((v + HALF_WIDTH) / RESOLUTION)) for v in (x, y))
    return grid[i][j] if 0 <= i < GRID and 0 <= j < GRID else DROP


def surface(kind, seed, level):
    name = TERRAIN_NAMES[kind]
    d = level / 9
    rng = np.random.default_rng([739, kind, seed, level])
    axis = (np.arange(GRID) + .5) * RESOLUTION - HALF_WIDTH
    x, y = np.meshgrid(axis, axis, indexing='ij')
    width = 2.55 + .15*seed
    mid = np.abs(x) < 1.9
    high = 0.
    a = np.zeros_like(x)
    if name == 'gap':
        high = .08 + .26*d + .025*seed
        a = np.where(x > 0., high, 0.)
        a[np.abs(x-(seed-.5)*.18) < .12 + .22*d] = DROP
    elif name == 'pit':
        a[np.abs(x) < .85 + .28*d] = -(.17 + .48*d)
    elif name == 'radial_beam':
        r = np.hypot(x, y)
        theta = np.arctan2(y, x)
        delta = theta - np.rint(theta/(np.pi/4))*(np.pi/4)
        keep = (r < .72) | (r*np.abs(np.sin(delta)) < .37-.12*d)
        a[mid & ~keep] = DROP
    elif name == 'stairs':
        rise = .085+.095*d
        high = 8*rise
        a = np.clip(np.floor((x+1.65)/.42)+1, 0, 8)*rise
    elif name == 'balance_beam':
        a[mid & (np.abs(y-(seed-.5)*.36) > .43-.18*d)] = DROP
    elif name in ('jump_box','stepping_stones','parallel_box','climb_box'):
        a[mid] = DROP
        pitch_x = .76 if name == 'stepping_stones' else .95
        pitch_y = .81 if name == 'stepping_stones' else 1.08
        high = .4+.44*d if name == 'climb_box' else 0.
        for i in range(-3, 4):
            for j in range(-4, 5):
                cx = pitch_x*i + rng.uniform(-.06,.06)
                cy = pitch_y*j + rng.uniform(-.06,.06)
                if name == 'parallel_box':
                    cx=pitch_x*i
                    cy=pitch_y*j
                gap = .065+.12*d
                hx = pitch_x*.5-gap*.5
                hy = pitch_y*.5-gap*.5
                patch=(np.abs(x-cx)<hx)&(np.abs(y-cy)<hy)&mid
                if name=='climb_box':
                    z=high*np.clip((i+2)/4,0.,1.) + rng.uniform(-.03,.03)
                elif name=='jump_box':
                    z=rng.uniform(0,.16+.42*d)
                elif name=='parallel_box':
                    z=(i%2)*(.09+.27*d)
                else:
                    z=rng.uniform(-.035,.07+.12*d)
                a[patch]=z
    elif name == 'maze':
        # Gates alternate across the corridor, so a straight centre-line cut intersects walls.
        for i,cx in enumerate((-.98,0.,.98)):
            gap_center = (-1.)**(seed+i)*(.56+.53*d)
            wall = (np.abs(x-cx)<.13)&(np.abs(y-gap_center)>.64)
            a[wall]=.8+.4*d
    elif name == 'contour':
        # Tessellated rock relief, with smooth transitions at the two landing platforms.
        fade=np.clip((1.9-np.abs(x))/.38,0.,1.)
        v=.5*np.sin(2.1*x+.42*seed)+.34*np.cos(1.8*y+.7*x)+.22*np.sin(3.2*x-2.25*y)
        a=np.round(v*(.25+.46*d)*fade/.045)*.045
    elif name == 'islands':
        a[mid] = DROP
        for j in (-1,0,1):
            lane=j*1.65
            for i in (-1,0,1):
                cx=i*1.5
                cy=lane+(.19 if (i+seed)%2 else -.19)
                island=(np.hypot(x-cx,y-cy)<(.61-.07*d))
                bridge=(np.abs(y-lane)<(.25-.075*d))&(np.abs(x-cx)<.90)
                mask=mid&(island|bridge)
                a[mask]=.08+.1*((i+j)%2)*(1+d)
    elif name=='inverted_slope':
        a=-np.clip(1.9-np.abs(x),0,1.6)*(.16+.27*d)
    else:
        raise KeyError(name)
    if name in ('radial_beam','balance_beam','islands'):
        # Height of the central landing changes continuously even when a width
        # increment is smaller than one collision cell.
        crest=(np.abs(x)<.53)&(a>DROP+.1)
        a[crest]+=.015+.045*d
    a[x <= -1.9]=0.
    a[x >= 1.9]=high
    a[(np.abs(x)>5.55)|(np.abs(y)>width)]=DROP
    # A seeded lateral reflection keeps starts/goals on supported patches.
    if seed:
        a=a[:,::-1]
    return np.round(a,5)


def endpoints(grid,kind,seed,level,split):
    # Two independent sets of twenty. Crossing and same-side cases both occur.
    rng=np.random.default_rng([6217,split,kind,seed,level])
    def sample():
        points=[]
        for side in (-1,1):
            for k in range(10):
                x=side*(2.72+(k%2)*.70+rng.uniform(-.12,.12))
                y=-1.75+(k//2)*.87+rng.uniform(-.18,.18)
                points.append((x,y,float(height_at(grid,x,y))))
        return points
    return sample(),sample()


@dataclass(frozen=True)
class Tile:
    grid: tuple
    starts: tuple
    goals: tuple


@lru_cache(maxsize=260)
def tile(kind,seed,level):
    grid=surface(kind,seed,level)
    starts,goals=endpoints(grid,kind,seed,level,0)
    return Tile(tuple(map(tuple,grid.tolist())),starts,goals)


def build(root):
    import hashlib
    root=Path(root)
    bank=[np.empty((TASK_COUNT,7),np.float32) for _ in range(2)]
    different_geometry=set()
    for kind in range(13):
        for seed in range(2):
            for level in range(10):
                grid=surface(kind,seed,level)
                different_geometry.add(hashlib.sha256(grid.tobytes()).hexdigest())
                for split,arr in enumerate(bank):
                    sp,gp=endpoints(grid,kind,seed,level,split)
                    for point in sp+gp:
                        x,y,z=point
                        assert all(abs(height_at(grid,x+dx,y+dy)-z)<1e-7 for dx in (-.58,0,.58) for dy in (-.37,0,.37))
                    start=np.repeat(sp,20,axis=0); goal=np.tile(gp,(20,1))
                    ix=encode(kind,seed,level,0,0)
                    group=arr[ix:ix+400]
                    group[:,:3]=start;group[:,3:6]=goal;group[:,6]=0.
                    crossing=start[:,0]*goal[:,0]<0
                    assert crossing.sum()==200
                    assert np.all(np.abs(goal[crossing,0]-start[crossing,0])>4.)
                    if TERRAIN_NAMES[kind] in ('stairs','climb_box'):
                        assert np.all(np.abs(start[crossing,2]-goal[crossing,2])>.39)
                    # Every crossing has to pass through the obstacle strip; course margins have no surface.
                    assert (grid[[0,-1]]==DROP).all() and (grid[:,[0,-1]]==DROP).all()
    assert not np.any(np.all(bank[0][:,:6]==bank[1][:,:6],axis=1))
    for name,arr in zip(('geometry','holdout'),bank):
        np.save(root/(name+'.npy'),arr,allow_pickle=False)
    rec=dict(families=list(TERRAIN_NAMES),layouts=TILE_COUNT,distinct_raster_hashes=len(different_geometry),
             configs=TASK_COUNT,crossing=52000,same_side=52000,uncollapsed_train_holdout=True,
             geometry_sha=[hashlib.sha256(a.tobytes()).hexdigest() for a in bank])
    (root/'catalog.json').write_text(json.dumps(rec,indent=2)+'\n')
    return rec
