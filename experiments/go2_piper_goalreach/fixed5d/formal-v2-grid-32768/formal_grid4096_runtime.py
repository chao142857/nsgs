"""Verify the selected sampler against its registered scientific semantics."""
from datetime import datetime
from hashlib import sha256
import random

import numpy as np

from grid_formal_protocol import ARM, BINS, CELL_MAP, OCCUPIED


def normalized(sampler):
    rates = sampler.cell_rates
    kernel = sampler.kernel_settings()
    prior = kernel['epsilon']
    power = kernel['concentration']
    target = kernel['target']
    values = (rates+prior)**(power*target)*(1-rates+prior)**(power*(1-target))
    logits = np.log(np.maximum(values,prior)+prior)/kernel['temperature']
    weights = np.exp(logits-logits.max())
    mass = weights[sampler.ids]
    expected = mass/mass.sum()
    actual = sampler.probabilities()
    assert np.allclose(actual,expected,rtol=5e-15,atol=0.)
    assert np.isfinite(actual).all() and np.min(actual)>0 and abs(float(actual.sum())-1)<2e-14
    cell_mass = np.bincount(sampler.ids,weights=actual,minlength=len(weights))
    expected_cells = weights*sampler.occupancy
    expected_cells /= expected_cells.sum()
    assert np.allclose(cell_mass,expected_cells,rtol=5e-14,atol=0.)
    return actual


def entry(pool, sampler, seed):
    from goalreach.sgs_kernels import EMPTY_HISTORY_PRIOR, KERNELS
    from samplers import SparseGridSampler, grid_ids

    assert type(sampler) is SparseGridSampler and sampler.bins==BINS
    assert sampler.kernel_id=='sharp' and sampler.history==100
    assert sampler.kernel_settings()==dict(KERNELS['sharp'])
    assert sampler.cell_ids_sha256==CELL_MAP and len(sampler.occupancy)==OCCUPIED
    assert all(not np.any(array) for array in
               (sampler.cell_counts,sampler.completed_counts,sampler.success_counts,sampler.ring))
    assert EMPTY_HISTORY_PRIOR==.5 and np.all(sampler.cell_rates==EMPTY_HISTORY_PRIOR)
    assert np.array_equal(normalized(sampler),np.full(sampler.n,1/sampler.n))
    runtime_before = sampler.state_dict()

    # Exercise actual native binning, feedback and sampling on detached state.
    coordinates = grid_ids(pool,5,BINS)
    assert sha256(coordinates.astype('<i4',copy=False).tobytes()).hexdigest()==CELL_MAP
    probe = SparseGridSampler(pool,seed,5,BINS)
    dense_cell = int(np.argmax(probe.occupancy))
    candidates = np.flatnonzero(probe.ids==dense_cell)[:2]
    first, second = map(int,candidates)
    ids = [first,second]*65
    labels = [1]*30+[0]*25+[1]*75
    updates = list(range(1,len(labels)+1))
    probe.record(ids,labels,updates)
    assert probe.cell_counts[dense_cell]==130
    assert probe.cell_rates[dense_cell]==sum(labels[-100:])/100
    assert probe.cached_rates[first]==probe.cached_rates[second]
    assert np.all(probe.cell_rates[np.arange(len(probe.cell_rates))!=dense_cell]==.5)
    assert int(probe.completed_counts.sum())==130 and int(probe.success_counts.sum())==sum(labels)
    probabilities = normalized(probe)
    cdf=np.cumsum(probabilities)
    seeded=random.Random(seed)
    units=np.fromiter((seeded.random()*cdf[-1] for _ in range(512)),np.float64,count=512)
    intended=np.minimum(np.searchsorted(cdf,units,side='right'),probe.n-1)
    selected,chosen_mass,_=probe.draw(512)
    assert np.array_equal(selected,intended) and np.array_equal(chosen_mass,probabilities[selected])
    assert len(np.unique(pool['config_id'][selected]))>500
    from protocol import AXES, BOUNDS

    def has_continuous_values(axis):
        position=AXES.index(axis)
        lo,hi=BOUNDS[position]
        width=(hi-lo)/BINS[position]
        values=pool[axis][selected]
        centers=lo+(np.floor((values-lo)/width)+.5)*width
        return np.count_nonzero(np.abs(values-centers)>1e-6)>500

    assert all(has_continuous_values(axis) for axis in ('mu','alpha_deg'))
    runtime_after = sampler.state_dict()
    assert all(np.array_equal(value,runtime_after[key]) if isinstance(value,np.ndarray)
               else value==runtime_after[key] for key,value in runtime_before.items())
    return dict(at=datetime.now().astimezone().isoformat(), method=ARM, grid=list(BINS),
        total_cells=int(np.prod(BINS)), occupied_cells=OCCUPIED, cell_map_sha256=CELL_MAP,
        kernel=sampler.kernel_settings(), empty_prior=EMPTY_HISTORY_PRIOR,
        actual_labels_at_entry=0, engineering_replay_labels=130, engineering_draws=512,
        engineering_ppo=0, recent_100_rate=probe.cell_rates[dense_cell],
        exact_pool_order_draw=True, no_state_transfer=True, statistics_device='cpu',
        continuous_pool_values=True, training_slots=4096, training_pool_rows=sampler.n)


def committed(rt, update):
    from goalreach.core import append_rows

    sampler=rt.sampler
    actual=normalized(sampler)
    ends = int(sampler.completed_counts.sum())
    assert ends==int(sampler.cell_counts.sum())==rt.episode_log.record_count
    drawn = int(sampler.sample_counts.sum())
    assert drawn>=ends+rt.discarded
    append_rows(rt.out/'spatial_grid.jsonl',[dict(update=update,grid=list(BINS),
        history=sampler.history, kernel=sampler.kernel_id, cell_map_sha256=sampler.cell_ids_sha256,
        natural_ending_labels=ends, positives=int(sampler.success_counts.sum()),
        draws=drawn, discarded=rt.discarded, ongoing_or_unlogged=drawn-ends-rt.discarded,
        probability_sum=float(actual.sum()), probability_range=[float(actual.min()),float(actual.max())],
        probability_formula_verified=True, diagnostics=sampler.diagnostics())])
