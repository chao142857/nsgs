"""Geometry-bin H100 estimator with the frozen per-configuration sharp sampler."""
from hashlib import sha256
import random
import time

import numpy as np

from .p3b_sampler import LargePoolSampler, sharp_logit, sharp_weight
from .sgs_kernels import KERNELS


GEOMETRY = dict(bins=[16,32],axes=['horizontal_distance','azimuth','height'],
    bounds=[[0.,3.],[-np.pi,np.pi],[.1,.85]],
    endpoint_rule='r/z final bin includes upper endpoint; +pi wraps to -pi; 1 float32 ULP of endpoint representation tolerated',
    index_order='(radius_bin * b + azimuth_bin) * b + height_bin',
    history='Last 100 naturally ended labels per occupied cell, in training event order.',
    empty_prior=.5,per_config_mass='K(p_cell) / sum_i K(p_cell(i))',
    per_cell_mass='n_cell * K(p_cell) / sum_cells(n_cell * K(p_cell))')


def _bins(values, lower, upper, count, *, periodic=False):
    values = np.asarray(values,dtype=np.float64)
    lower_tol = abs(float(np.spacing(np.float32(lower))))
    upper_tol = abs(float(np.spacing(np.float32(upper))))
    if (not np.isfinite(values).all() or np.any(values<lower-lower_tol) or np.any(values>upper+upper_tol)):
        raise ValueError(f'Grid coordinates outside [{lower}, {upper}]')
    values = np.clip(values,lower,upper)
    if periodic:
        values = np.where(values==upper,lower,values)
    return np.minimum(np.floor((values-lower)*(count/(upper-lower))).astype(np.int32),count-1)


def cell_ids(pool, count):
    if count not in GEOMETRY['bins']:
        raise ValueError('Only 16 and 32 bins per axis are registered')
    radius, angle, height = (_bins(pool[key],lower,upper,count,periodic=key=='azimuth')
                             for key,(lower,upper) in zip(GEOMETRY['axes'],GEOMETRY['bounds']))
    if radius.ndim!=1 or radius.shape!=angle.shape or radius.shape!=height.shape:
        raise ValueError('Grid coordinate arrays must have the same one-dimensional shape')
    if not len(radius):
        raise ValueError('Grid needs at least one configuration')
    return (radius*count+angle)*count+height


class GridSampler(LargePoolSampler):
    """Use the original cumulative draw, with mass expanded to canonical pool order."""

    def __init__(self, pool, seed, bins):
        self.bins = bins
        self.regions = pool['region']
        self.ids = cell_ids(pool,bins)
        self.n = len(self.ids)
        self.mode = self.interface_mode = 'sgs'
        self.kernel_id = 'sharp'
        self.rng = random.Random(seed)
        self.history = KERNELS['sharp']['history']
        self.occupancy = np.bincount(self.ids,minlength=bins**3)
        self.sample_counts = np.zeros(self.n,np.uint32)
        self.completed_counts = np.zeros_like(self.sample_counts)
        self.success_counts = np.zeros_like(self.sample_counts)
        self.cell_counts = np.zeros(bins**3,np.uint32)
        self.ring = np.zeros((bins**3,self.history),np.uint8)
        self.ring_updates = np.zeros_like(self.ring,dtype=np.uint16)
        self.ring_success = np.zeros(bins**3,np.uint16)
        self.cell_rates = np.full(bins**3,.5)
        self._cached_rates = self._probability = self._cumulative = None
        self.sample_wall_s = 0.
        self.sample_calls = 0
        self.feedback_wall_s = self.probability_wall_s = 0.
        self.cell_ids_sha256 = sha256(self.ids.astype('<i4',copy=False).tobytes()).hexdigest()

    @property
    def cached_rates(self):
        if self._cached_rates is None:
            self._cached_rates = self.cell_rates[self.ids]
        return self._cached_rates

    def probabilities(self):
        if self._probability is None:
            # Normalize over configurations. Equivalently, cell weights are
            # multiplied by cell occupancy before normalizing.
            start=time.monotonic()
            active = self.occupancy>0
            logits = sharp_logit(self.cell_rates)
            weight = np.exp(logits-logits[active].max())
            values = weight[self.ids]
            self._probability = values/values.sum()
            self.probability_wall_s += time.monotonic()-start
        return self._probability

    def set_probabilities(self, values):
        raise ValueError('The grid control accepts only its own H100 estimates')

    def record(self, ids, successes, end_updates):
        if len(ids)!=len(successes) or len(ids)!=len(end_updates):
            raise ValueError('Configuration, outcome and event timestamp counts differ')
        if not len(ids):
            return
        start=time.monotonic()
        task = np.asarray(ids,dtype=np.intp)
        wins = np.asarray(successes)
        updates = np.asarray(end_updates)
        if (np.any(task<0) or np.any(task>=self.n) or np.any((wins!=0)&(wins!=1))
                or np.any(updates<1) or np.any(updates>8000)):
            raise ValueError('Invalid grid feedback record')
        changed = self.ids[task]
        np.add.at(self.completed_counts,task,1)
        np.add.at(self.success_counts,task,wins.astype(np.uint32,copy=False))
        for cell,success,update in zip(changed.tolist(),wins.tolist(),updates.tolist()):
            slot = int(self.cell_counts[cell])%self.history
            self.ring_success[cell] = int(self.ring_success[cell])+success-int(self.ring[cell,slot])
            self.ring[cell,slot] = success
            self.ring_updates[cell,slot] = update
            self.cell_counts[cell] += 1
        cells = np.unique(changed)
        self.cell_rates[cells] = self.ring_success[cells]/np.minimum(self.cell_counts[cells],self.history)
        self._cached_rates = self._probability = self._cumulative = None
        self.feedback_wall_s += time.monotonic()-start

    def state_dict(self):
        return dict(version=1,mode='grid',kernel=self.kernel_id,history=self.history,n=self.n,bins=self.bins,
            cell_ids_sha256=self.cell_ids_sha256,sample_counts=self.sample_counts.copy(),
            completed_counts=self.completed_counts.copy(),success_counts=self.success_counts.copy(),
            cell_counts=self.cell_counts.copy(),history_bits=np.packbits(self.ring,axis=1,bitorder='little'),
            ring_updates=self.ring_updates.copy(),rng_state=self.rng.getstate())

    def load_state_dict(self, state):
        expected = (1,'grid','sharp',self.history,self.n,self.bins,self.cell_ids_sha256)
        keys = ('version','mode','kernel','history','n','bins','cell_ids_sha256')
        if tuple(state[k] for k in keys)!=expected:
            raise ValueError('Incompatible grid sampler state')
        for key in ('sample_counts','completed_counts','success_counts','cell_counts','ring_updates'):
            saved, current = state[key], getattr(self,key)
            if saved.shape!=current.shape or saved.dtype!=current.dtype:
                raise ValueError(f'Grid state array differs: {key}')
            setattr(self,key,saved.copy())
        self.ring[:] = np.unpackbits(state['history_bits'],axis=1,count=self.history,bitorder='little')
        self.ring_success[:] = self.ring.sum(axis=1)
        tracked = np.minimum(self.cell_counts,self.history)
        self.cell_rates.fill(.5)
        np.divide(self.ring_success,tracked,out=self.cell_rates,where=tracked>0)
        self.rng.setstate(state['rng_state'])
        self._cached_rates = self._probability = self._cumulative = None

    def diagnostics(self):
        active = self.occupancy>0
        has_labels = self.cell_counts>0
        indices = np.flatnonzero(has_labels)
        counts = self.cell_counts[has_labels].astype(np.int64)
        first = (counts-np.minimum(counts,self.history))%self.history
        last = (counts-1)%self.history
        ages = self.ring_updates[indices,last].astype(np.int64)-self.ring_updates[indices,first]
        mass = np.bincount(self.ids,weights=self.probabilities(),minlength=self.bins**3)
        floor = sharp_weight(self.cell_rates)<=KERNELS['sharp']['epsilon']
        return dict(bins=self.bins,cell_ids_sha256=self.cell_ids_sha256,
            occupied_cells=int(active.sum()),empty_cells=int((~active).sum()),
            occupancy_quantiles=np.quantile(self.occupancy[active],[0,.25,.5,.75,1]).tolist(),
            cells_without_labels=int((active&~has_labels).sum()),
            labeled_cells=int(has_labels.sum()),cells_with_full_history=int((self.cell_counts>=self.history).sum()),
            history_span_update_quantiles=np.quantile(ages,[0,.25,.5,.75,1]).tolist() if len(ages) else None,
            floor_cell_count=int((active&floor).sum()),floor_probability_mass=float(mass[floor].sum()),
            cell_mass_quantiles=np.quantile(mass[active],[0,.25,.5,.75,1]).tolist())


def verify_grid_restore(path, pool, replay, update):
    import torch
    from .neural_sgs import kernel_probabilities

    ckpt=torch.load(path,map_location='cpu',weights_only=False)
    assert ckpt['completed_updates']==update
    assert all(k in ckpt for k in ('actor','critic','optimizer','ema','sampler','forecast','torch_rng',
                                   'cuda_rng','numpy_rng','python_rng'))
    assert all(torch.isfinite(v).all().item() for k in ('actor','critic','ema') for v in ckpt[k].values())
    state=ckpt['sampler']
    recorded=replay.state_dict()
    for key in ('completed_counts','success_counts','cell_counts','history_bits','ring_updates'):
        if not np.array_equal(state[key],recorded[key]):
            raise ValueError(f'Grid checkpoint differs from ordered label replay: {key}')
    a=GridSampler(pool,0,replay.bins)
    b=GridSampler(pool,1,replay.bins)
    a.load_state_dict(state)
    b.load_state_dict(state)
    p=a.probabilities()
    assert np.array_equal(p,b.probabilities())
    assert np.array_equal(p,kernel_probabilities(a.cached_rates,'sharp'))
    assert np.array_equal(a.draw(4096)[0],b.draw(4096)[0])
    return dict(sampler_round_trip=True,formula_exact=True,ordered_label_replay=True,
                grid=replay.diagnostics(),isolated_restore_ppo_updates=0)
