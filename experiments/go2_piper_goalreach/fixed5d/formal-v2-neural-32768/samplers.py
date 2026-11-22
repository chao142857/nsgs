"""Shared terrain descriptors and six registered sampling arms."""
from hashlib import sha256
import random

import numpy as np
from goalreach.p3b_sampler import LargePoolSampler
from goalreach.p3s1_grid import GridSampler, _bins
from goalreach.sgs_kernels import KERNELS


from grid_development import DEVELOPMENT_GRIDS, development_sampler
from protocol import AXES, BOUNDS, METHODS

GRID = dict(physical_grid16=(5,16), physical_grid32=(5,32),
            target_grid16=(3,16), target_grid32=(3,32))
GRID.update({name:(5,counts) for name,counts in DEVELOPMENT_GRIDS.items()})


def grid_ids(pool, dimensions, bins):
    keys = AXES[:dimensions]
    shape = np.shape(pool[keys[0]])
    if len(shape) != 1 or not shape[0]:
        raise ValueError('Expected a nonempty one-dimensional task pool')
    ids = np.zeros(shape, dtype=np.int32)
    counts = (bins,)*dimensions if isinstance(bins,int) else tuple(bins)
    assert len(counts)==dimensions
    for axis, (lo, hi), count in zip(keys, BOUNDS, counts):
        if np.shape(pool[axis]) != shape:
            raise ValueError(f'Wrong coordinate shape for {axis}')
        ids = ids*count + _bins(pool[axis], lo, hi, count, periodic=(axis == 'azimuth'))
    return ids


class SparseGridSampler(GridSampler):
    """Keep native GridSampler feedback and draw logic with compact active histories."""

    def __init__(self, pool, seed, dimensions, bins):
        self.dimensions = dimensions
        self.bins = bins
        self.regions = pool['region']
        cell = grid_ids(pool, dimensions, bins)
        _, dense, self.occupancy = np.unique(cell, return_inverse=True, return_counts=True)
        self.ids = dense.astype(np.int32, copy=False)
        n_active = len(self.occupancy)
        self.n = len(cell)
        self.mode = self.interface_mode = 'sgs'
        self.kernel_id = 'sharp'
        self.rng = random.Random(seed)
        self.history = KERNELS['sharp']['history']
        self.sample_counts = np.zeros(self.n, np.uint32)
        self.completed_counts = np.zeros_like(self.sample_counts)
        self.success_counts = np.zeros_like(self.sample_counts)
        self.cell_counts = np.zeros(n_active, np.uint32)
        self.ring = np.zeros((n_active, self.history), np.uint8)
        self.ring_updates = np.zeros_like(self.ring, dtype=np.uint16)
        self.ring_success = np.zeros(n_active, np.uint16)
        self.cell_rates = np.full(n_active, .5)
        self._cached_rates = self._probability = self._cumulative = None
        self.sample_wall_s = self.feedback_wall_s = self.probability_wall_s = 0.
        self.sample_calls = 0
        self.cell_ids_sha256 = sha256(cell.astype('<i4', copy=False).tobytes()).hexdigest()

    def diagnostics(self):
        cells = np.flatnonzero(self.cell_counts)
        counts = self.cell_counts[cells].astype(np.int64)
        start = (counts-np.minimum(counts, self.history)) % self.history
        end = (counts-1) % self.history
        spans = self.ring_updates[cells, end]-self.ring_updates[cells, start]
        sampled = np.zeros(len(self.occupancy), dtype=np.uint64)
        np.add.at(sampled, self.ids, self.sample_counts)
        quantiles = [0, .25, .5, .75, 1]
        return dict(dimensions=self.dimensions, bins=self.bins, map_hash=self.cell_ids_sha256,
                    occupied_cells=len(self.occupancy), empty_cells=(self.bins**self.dimensions if isinstance(self.bins,int) else int(np.prod(self.bins)))-len(self.occupancy),
                    occupancy_quantiles=np.quantile(self.occupancy, quantiles).tolist(),
                    labeled_cells=len(cells), cells_without_labels=len(self.occupancy)-len(cells),
                    cells_with_full_history=int(np.count_nonzero(self.cell_counts >= self.history)),
                    ended_episodes_quantiles=np.quantile(self.cell_counts, quantiles).tolist(),
                    sampled_episodes_quantiles=np.quantile(sampled, quantiles).tolist(),
                    same_configuration_revisits=int(np.maximum(self.sample_counts, 1).sum()-self.n),
                    history_span_update_quantiles=np.quantile(spans, quantiles).tolist() if len(spans) else None)


def sampler_for(method, pool, seed):
    if method in DEVELOPMENT_GRIDS:
        return development_sampler(pool,seed,DEVELOPMENT_GRIDS[method])
    if method not in METHODS:
        raise ValueError(f'Unregistered sampling method: {method}')
    if method in GRID:
        return SparseGridSampler(pool, seed, *GRID[method])
    sampler = LargePoolSampler(pool['region'], seed, 'external' if method == METHODS[0] else 'sgs')
    if method == METHODS[0]:
        sampler.set_probabilities(np.ones(len(pool['region']), dtype=np.float64))
    return sampler
