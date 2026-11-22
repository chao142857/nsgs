"""Pool-sized arrays and ordered reset sampling with K2 sharp history semantics."""
import random
import time

import numpy as np

from .sgs_kernels import KERNELS


def sharp_weight(rates):
    settings = KERNELS['sharp']
    e, t, k = (settings[key] for key in ('epsilon','target','concentration'))
    p = np.asarray(rates, dtype=np.float64)
    return (p+e)**(k*t) * (1-p+e)**(k*(1-t))


def sharp_logit(rates):
    settings = KERNELS['sharp']
    e = settings['epsilon']
    return np.log(np.maximum(sharp_weight(rates), e)+e)/settings['temperature']


class LargePoolSampler:
    def __init__(self, regions, seed, mode):
        if mode not in ('sgs', 'external'):
            raise ValueError('Expected the P3B1 table or neural sampler')
        self.n = len(regions)
        self.regions = regions
        self.mode = self.interface_mode = mode
        self.kernel_id = 'sharp'
        self.history = KERNELS['sharp']['history']
        self.rng = random.Random(seed)
        self.sample_counts = np.zeros(self.n, np.uint32)
        self.completed_counts = np.zeros_like(self.sample_counts)
        self.success_counts = np.zeros_like(self.sample_counts)
        self.ring_success = np.zeros(self.n, np.uint8)
        self.ring = np.zeros((self.n, self.history), np.uint8)
        self.cached_rates = np.full(self.n, .5)
        self.logits = sharp_logit(self.cached_rates) if mode == 'sgs' else None
        self.external = None
        self._probability = self._cumulative = None
        self.sample_wall_s = 0.
        self.sample_calls = 0

    def kernel_settings(self):
        return dict(KERNELS['sharp'])

    def set_probabilities(self, values):
        p = np.asarray(values,dtype=np.float64)
        mass = p.sum()
        if p.shape != (self.n,) or not np.isfinite(p).all() or np.any(p<=0) or not np.isfinite(mass) or mass<=0:
            raise ValueError('Expected nonzero finite probability mass on every P3B1 task')
        self.external = p/mass
        self._probability = self._cumulative = None

    def probabilities(self):
        if self._probability is None:
            if self.mode == 'sgs':
                w = np.exp(self.logits-self.logits.max())
            else:
                if self.external is None:
                    raise ValueError('No neural probability vector')
                w = self.external
            self._probability = w/w.sum()
        return self._probability

    def draw(self, count):
        start = time.monotonic()
        p = self.probabilities()
        if self._cumulative is None:
            self._cumulative = np.cumsum(p)
        total = self._cumulative[-1]
        selected = np.searchsorted(self._cumulative,
            np.fromiter((self.rng.random()*total for _ in range(count)),np.float64,count=count),side='right')
        # random.choices excludes the last entry from its bisect bounds.
        np.minimum(selected,self.n-1,out=selected)
        previous = []
        for i in selected:
            previous.append(int(self.sample_counts[i]))
            self.sample_counts[i] += 1
        self.sample_wall_s += time.monotonic()-start
        self.sample_calls += 1
        return selected, p[selected], previous

    def record(self, ids, successes):
        if len(ids) != len(successes):
            raise ValueError('Task/outcome lengths differ')
        for task, success in zip(ids, successes):
            if not 0<=task<self.n or success not in (0,1):
                raise ValueError('Invalid task or binary outcome')
            slot = int(self.completed_counts[task]) % self.history
            self.ring_success[task] = int(self.ring_success[task]) + int(success) - int(self.ring[task,slot])
            self.ring[task,slot] = success
            self.completed_counts[task] += 1
            self.success_counts[task] += int(success)
        if ids:
            changed = np.unique(ids)
            self.cached_rates[changed] = self.ring_success[changed]/np.minimum(self.completed_counts[changed],self.history)
            if self.mode == 'sgs':
                self.logits[changed] = sharp_logit(self.cached_rates[changed])
                self._probability = self._cumulative = None

    def state_dict(self):
        return dict(version=1,n=self.n,mode=self.mode,history=self.history,kernel='sharp',
            sample_counts=self.sample_counts.copy(),completed_counts=self.completed_counts.copy(),
            success_counts=self.success_counts.copy(),history_bits=np.packbits(self.ring,axis=1,bitorder='little'),
            external=None if self.external is None else self.external.copy(),rng_state=self.rng.getstate())

    def load_state_dict(self, state):
        if (state['version'],state['n'],state['mode'],state['history'],state['kernel'])!=(1,self.n,self.mode,self.history,'sharp'):
            raise ValueError('Incompatible P3B1 sampler state')
        for key in ('sample_counts','completed_counts','success_counts'):
            counts = state[key]
            if counts.shape!=(self.n,) or counts.dtype!=np.uint32:
                raise ValueError('Invalid sampler count array')
            setattr(self,key,counts.copy())
        self.ring[:] = np.unpackbits(state['history_bits'],axis=1,count=self.history,bitorder='little')
        self.ring_success[:] = self.ring.sum(axis=1)
        self.cached_rates.fill(.5)
        completed = np.minimum(self.completed_counts,self.history)
        np.divide(self.ring_success,completed,out=self.cached_rates,where=completed>0)
        if self.mode == 'sgs':
            self.logits[:] = sharp_logit(self.cached_rates)
        else:
            self.external = state['external'].copy()
        self.rng.setstate(state['rng_state'])
        self._probability = self._cumulative = None
