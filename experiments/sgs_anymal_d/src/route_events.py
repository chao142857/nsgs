"""Frozen local behavioral labels. This module never feeds PPO or SGS."""
from dataclasses import dataclass, field
from functools import lru_cache
import numpy as np

DT = .02
DEADLINE = 500
HOLD = 150
LOAD_N = 15.0
FOOT_MESH_TOL_M = .14
ROLLOVER_DEG = 75.0
ROLLOVER_TICKS = 3

@dataclass
class Clock:
    first_arrival: int | None = None
    hold_start: int | None = None
    hold: int = 0
    best_hold: int = 0
    done: str | None = None
    end: int | None = None
    hold_interrupted: int = 0

    def observe(self,t,arrived,stable,failure=None):
        if self.done is not None:
            return
        if failure:
            self.done=failure
        else:
            if arrived and self.first_arrival is None and t<=DEADLINE:
                self.first_arrival=t
            if arrived and stable:
                if not self.hold and t<=DEADLINE:
                    self.hold_start=t
                if self.hold or t<=DEADLINE:
                    self.hold+=1
                    self.best_hold=max(self.best_hold,self.hold)
            else:
                if self.hold:
                    self.hold_interrupted+=1
                self.hold=0
                self.hold_start=None
            if self.hold>=HOLD:
                self.done='success'
            elif t>=DEADLINE and self.hold==0:
                self.done='hold_interrupted' if self.first_arrival is not None else 'navigation_timeout'
        if self.done is not None:
            self.end=t

def clock_checks():
    def run(rows):
        c=Clock()
        for i,(a,s,f) in enumerate(rows,1): c.observe(i,a,s,f)
        return c
    good=(True,True,None); no=(False,False,None)
    assert run([good]*149).done is None
    assert run([good]*150).done=='success'
    assert run([good]*149+[(True,True,'base_contact')]).done=='base_contact'
    assert run([good]*145+[(False,False,'base_contact')]).done=='base_contact'
    assert run([good]*80+[no]+[good]*149).done is None
    assert run([no]*474+[good]*150).end==624
    assert run([no]*499+[good]*150).end==649
    assert run([no]*500+[good]*150).done=='navigation_timeout'
    assert run([no]*474+[good]*40+[no]).done=='hold_interrupted'
    return True

@lru_cache(maxsize=260)
def rasters(name,seed,level):
    from course import TERRAIN_NAMES,region_grid,surface
    k=TERRAIN_NAMES.index(name)
    return np.asarray(region_grid(k,seed,level)),np.asarray(surface(k,seed,level))

def sample_mesh(m,xy,default):
    from terrain_compat import HALF_WIDTH,RESOLUTION
    xy=np.asarray(xy)
    ij=np.floor((xy+HALF_WIDTH)/RESOLUTION).astype(int)
    valid=np.all((ij>=0)&(ij<np.asarray(m.shape)),axis=-1)
    safe=np.clip(ij,0,np.asarray(m.shape)-1)
    return np.where(valid,m[safe[...,0],safe[...,1]],default)

@dataclass
class Event:
    route: dict
    visits: list = field(default_factory=list)
    seen: set = field(default_factory=set)
    start_loaded: bool = False
    airborne: bool = False
    flight_from: set = field(default_factory=set)
    previous_support: set = field(default_factory=set)
    air_to_land: list = field(default_factory=list)
    start_tick: int | None = None
    first_transfer: dict | None = None
    event_tick: int | None = None
    rollover: int = 0
    furthest: float = 0.
    support_lo: float = float('inf')
    support_hi: float = -float('inf')
    beam_distance: list = field(default_factory=list)
    center_seen: bool = False
    floor_seen: bool = False
    staircase: int = 0
    last_tread: int | None = None
    maze_passages: set = field(default_factory=set)
    prev_root: np.ndarray | None = None

    def observe(self,t,root,feet,foot_forces,tilt,linear,angular,base_force,labels,ray_heights=None,nonfoot_force=0.):
        name=self.route['terrain']; seed=self.route['terrain_seed']; level=self.route['level']
        regions,ground=rasters(name,seed,level)
        start=np.asarray(self.route['start']); goal=np.asarray(self.route['goal'])
        a=self.route['start_support']; b=self.route['goal_support']
        fz=foot_forces[:,2]
        z=sample_mesh(ground,feet[:,:2],-99.) if ray_heights is None else ray_heights
        foot_region=sample_mesh(regions,feet[:,:2],-1)
        foot_region=np.where((fz>LOAD_N)&(np.abs(feet[:,2]-z)<=FOOT_MESH_TOL_M),foot_region,-1)
        foot_region=np.where(foot_region>=0,foot_region,-1)
        support=set(int(i) for i in foot_region if i>=0)
        root_region=int(sample_mesh(regions,root[:2],-1))
        delta=goal[:2]-start[:2]
        length=max(float(np.linalg.norm(delta)),1e-8)
        d=float(np.dot(root[:2]-start[:2],delta)/length)
        self.furthest=max(self.furthest,d)
        if a in support:
            self.start_loaded=True
            if self.start_tick is None: self.start_tick=t
        if self.start_loaded and not np.any(fz>LOAD_N):
            self.airborne=True
            self.flight_from.update(self.previous_support)
        if self.flight_from and support:
            if support-self.flight_from:
                self.air_to_land.append([t,sorted(self.flight_from),sorted(support)])
            self.flight_from.clear()
        self.previous_support=support.copy()
        if self.start_loaded:
            if self.first_transfer is None and t>self.start_tick and (support-{a}):
                self.first_transfer=dict(tick=t,regions=sorted(support-{a}))
            for key in sorted(support-self.seen):
                self.visits.append([t,key])
            self.seen.update(support)
        if support:
            zs=np.asarray(z)[foot_region>=0]
            self.support_lo=min(self.support_lo,float(min(zs)))
            self.support_hi=max(self.support_hi,float(max(zs)))
        if name=='balance_beam' and 1 in support:
            self.beam_distance.append(d)
        if name=='radial_beam' and self.start_loaded and 0 in support:
            self.center_seen=True
        if name in ('pit','inverted_slope') and self.start_loaded and (1 if name=='pit' else 2) in support:
            self.floor_seen=True
        if name=='stairs' and support and self.start_loaded:
            direction=1 if b>a else -1
            next_tread=max(support) if direction>0 else min(support)
            if self.last_tread is None:self.last_tread=a
            if (next_tread-self.last_tread)*direction>0:
                self.staircase+=1; self.last_tread=next_tread
        if name=='maze' and self.prev_root is not None and root_region>=0 and self.start_loaded:
            # Accept either direction and any open passage in each intervening wall.
            from terrain_compat import RESOLUTION,HALF_WIDTH
            wall_indices=np.flatnonzero((regions==-2).any(axis=1))
            groups=np.split(wall_indices,np.flatnonzero(np.diff(wall_indices)!=1)+1)
            for j,cols in enumerate(g for g in groups if len(g)):
                wx=(np.mean(cols)+.5)*RESOLUTION-HALF_WIDTH
                x0,x1=self.prev_root[0],root[0]
                if (x0-wx)*(x1-wx)<=0 and abs(x1-x0)>1e-6:
                    f=(wx-x0)/(x1-x0)
                    pos=self.prev_root[:2]+f*(root[:2]-self.prev_root[:2])
                    if int(sample_mesh(regions,pos,-1))>=0:self.maze_passages.add(j)
        self.prev_root=np.array(root)
        dest=b in support
        crossed=self.start_loaded and dest and a!=b
        if name in ('islands','jump_box'):
            crossed=crossed and any(b in landed and b not in launch for _,launch,landed in self.air_to_land)
        elif name=='stepping_stones':
            crossed=crossed and len([i for i in self.seen if i>=10])>=2
        elif name=='parallel_box':
            crossed=crossed and len([i for i in self.seen if i>=10])>=2
        elif name=='climb_box':
            crossed=crossed and self.support_hi-self.support_lo>=.18 and any(i>=10 for i in self.seen)
        elif name=='stairs': crossed=crossed and self.staircase>=min(3,abs(b-a))
        elif name=='balance_beam':
            crossed=crossed and len(self.beam_distance)>1 and np.ptp(self.beam_distance)>=.60
        elif name=='radial_beam': crossed=crossed and self.center_seen
        elif name in ('pit','inverted_slope'): crossed=crossed and self.floor_seen
        elif name=='contour':
            crossed=crossed and len(self.seen)>=3 and self.support_hi-self.support_lo>=.10
        elif name=='maze': crossed=crossed and len(self.maze_passages)>=abs(b-a)
        if crossed and self.event_tick is None:self.event_tick=t
        dist=float(np.linalg.norm(goal[:2]-root[:2]))
        err=abs(float(root[2]-goal[2]-.60))
        correct=root_region==b and np.count_nonzero(foot_region==b)>=2
        arrived=self.event_tick is not None and dist<=.40 and err<=.36 and correct
        stable=arrived and tilt<30. and linear<.45 and angular<1.5
        ground_under_root=float(sample_mesh(ground,root[:2],-99.))
        fallen=(tilt>=ROLLOVER_DEG and nonfoot_force>LOAD_N and root[2]-ground_under_root<.45)
        self.rollover=self.rollover+1 if fallen else 0
        failure=None
        if labels==2 or root[2]<-1.05: failure='drop'
        elif labels==3 or base_force>30: failure='base_contact'
        elif self.rollover>=ROLLOVER_TICKS: failure='rollover'
        return bool(arrived),bool(stable),failure,foot_region,z

    def row(self):
        return dict(event_tick=self.event_tick,first_support_transfer=self.first_transfer,
                    first_region_visits=self.visits,max_progress_m=self.furthest,airborne=self.airborne,flight_landings=self.air_to_land,
                    min_loaded_z_m=self.support_lo if np.isfinite(self.support_lo) else None,
                    max_loaded_z_m=self.support_hi if np.isfinite(self.support_hi) else None,
                    maze_passages=sorted(self.maze_passages),stair_transitions=self.staircase)
