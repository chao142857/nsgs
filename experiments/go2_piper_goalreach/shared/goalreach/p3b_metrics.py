"""Causal 50-update forecast windows and full-pool sampling diagnostics."""
from collections import deque

import numpy as np

from .core import append_rows
from .p3b_sampler import sharp_weight
from .sgs_kernels import KERNELS
from tools.build_p2a_pools import cells


def geometry_cells(pool):
    region, sector, height, radius, near = (pool[k] for k in (
        'region','azimuth_sector','height','horizontal_distance','near_distance_band'))
    strata = np.searchsorted([.35,.60],height,side='right')
    ids = np.full(len(region),-1,np.int8)
    for cell in cells():
        r, b, a, h = (cell[k] for k in ('region','distance_band','azimuth_sector','height_stratum'))
        mask = region==r
        if b>=0:
            mask &= (near==b if r==0 else (radius>=2.25)==b)
        if a>=0:
            mask &= sector==a
        if h>=0:
            mask &= strata==h
        ids[mask] = cell['cell_id']
    if np.any(ids<0):
        raise ValueError('Incomplete forecast geometry')
    return ids


def measure(pred, pos, neg):
    count = pos+neg
    points = len(pred)
    positives = int(pos.sum())
    negatives = int(neg.sum())
    n = positives+negatives
    if n:
        order=np.argsort(pred,kind='stable')
        _, first = np.unique(pred[order],return_index=True)
        pos_sorted=np.add.reduceat(pos[order],first)
        neg_sorted=np.add.reduceat(neg[order],first)
        auc_sum=float(np.dot(pos_sorted,np.cumsum(neg_sorted)-.5*neg_sorted))
        loss=float(-(np.dot(pos,np.log(pred))+np.dot(neg,np.log1p(-pred))))
        brier=float(np.dot(pos,(1-pred)**2)+np.dot(neg,pred**2))
    else:
        auc_sum=loss=brier=0.
    bins=np.minimum((15*pred).astype(int),14)
    total=np.bincount(bins,weights=count,minlength=15)
    actual=np.bincount(bins,weights=pos,minlength=15)
    expected=np.bincount(bins,weights=pred*count,minlength=15)
    pairs=positives*negatives
    return dict(episodes=n,positives=positives,negatives=negatives,tasks_in_anchor=points,
        log_loss_sum=loss,log_loss=loss/n if n else None,brier_sum=brier,brier=brier/n if n else None,
        auc_pairs=pairs,auc_concordant=auc_sum,auc=auc_sum/pairs if pairs else None,
        ece=float(abs(actual-expected).sum())/n if n else None,
        calibration=dict(episodes=total.astype(int).tolist(),positives=actual.astype(int).tolist(),
                         prediction_sum=expected.tolist()))


class Forecast:
    """Anchor u uses endings strictly before u, scoring only after [u,u+50)."""
    def __init__(self, pool, out, rates):
        self.out = out/'forecast50.jsonl'
        self.region = pool['region']
        self.cell = geometry_cells(pool)
        self.pos = np.zeros(len(rates),np.uint32)
        self.neg = np.zeros_like(self.pos)
        self.prior = deque(maxlen=4)
        self.anchor = None
        self.begin(0,rates,np.zeros_like(self.pos))

    def begin(self, anchor, rates, completed):
        if self.anchor is not None:
            self.flush()
        self.anchor = anchor
        self.rates = rates.copy()
        self.completed = completed.copy()
        pos, neg = sum(self.prior,np.zeros((2,42),np.int64))
        n=pos+neg
        self.boundary = (n>=50)&(10*(pos+1)>=n+2)&(10*(pos+1)<=9*(n+2))

    def record(self, rows):
        for row in rows:
            if not self.anchor <= row['end_update'] < self.anchor+50:
                raise ValueError('Forecast outcome outside the open causal window')
            values = self.pos if row['success'] else self.neg
            values[row['task_id']] += 1

    def flush(self):
        ids=np.flatnonzero(self.pos+self.neg)
        p=np.clip(self.rates[ids],1e-6,1-1e-6)
        pos,neg=self.pos[ids],self.neg[ids]
        n=self.completed[ids]
        groups=dict(full=np.ones(len(ids),bool),boundary=self.boundary[self.cell[ids]],low_visit=n<=2)
        groups.update({name:self.region[ids]==i for i,name in enumerate(('near','inplace','step','walk'))})
        groups.update({'0':n==0,'1-2':(n>=1)&(n<=2),'3-9':(n>=3)&(n<=9),'10+':n>=10})
        metrics={key:measure(p[m],pos[m],neg[m]) for key,m in groups.items()}
        append_rows(self.out,[dict(anchor=self.anchor,window_stop_exclusive=self.anchor+50,
            boundary_cells=np.flatnonzero(self.boundary).tolist(),groups=metrics)])
        self.prior.append(np.array([np.bincount(self.cell[ids],weights=y,minlength=42) for y in (pos,neg)],np.int64))
        self.pos.fill(0)
        self.neg.fill(0)

    def state_dict(self):
        return dict(anchor=self.anchor,rates=self.rates.copy(),completed=self.completed.copy(),
            boundary=self.boundary.copy(),positive=self.pos.copy(),negative=self.neg.copy(),prior=list(self.prior))


def diagnostics(runtime, update):
    sampler = runtime.sampler
    scorer = runtime.neural.estimator if runtime.neural else None
    rates = scorer.neural_rate if scorer else sampler.cached_rates
    probabilities = sampler.probabilities()
    completed = sampler.completed_counts
    floor = sharp_weight(rates)<=KERNELS['sharp']['epsilon']
    sector = runtime.pool['azimuth_sector']
    groups = {'no_completed_results':completed==0,'completed_1_to_2':(completed>=1)&(completed<=2),
              'unassigned':sampler.sample_counts==0,'right':sector==2,'back':sector==3}
    return dict(update=update,positive_labels=int(sampler.success_counts.sum()),
        negative_labels=int(completed.sum()-sampler.success_counts.sum()),
        prediction_quantiles=dict(zip(('min','p01','p25','median','p75','p99','max'),
            np.quantile(rates,[0,.01,.25,.5,.75,.99,1]).tolist())),
        sharp_floor_fraction=float(floor.mean()),sharp_floor_probability_mass=float(probabilities[floor].sum()),
        probability_sum=float(probabilities.sum()),minimum_probability=float(probabilities.min()),
        entropy=float(-(probabilities*np.log(probabilities)).sum()),effective_configurations=float(1/(probabilities**2).sum()),
        region_probability_mass=np.bincount(runtime.pool['region'],weights=probabilities,minlength=4).tolist(),
        configuration_groups={name:dict(count=int(mask.sum()),probability_mass=float(probabilities[mask].sum()),
            assignments=int(sampler.sample_counts[mask].sum())) for name,mask in groups.items()},
        ensemble_std_mean=float(scorer.member_disagreement.mean()) if scorer else None,
        full_pool_size=len(rates))
