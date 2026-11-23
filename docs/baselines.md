# Unified Task-Sampling Baselines Framework (`nsgs.baselines`)

This module provides a single, domain-agnostic `TaskSampler` interface for fair,
matched-budget comparisons across curriculum and task-sampling methods in low-,
medium-, and high-dimensional robot control pools (3D, 5D, 10D+, and heterogeneous
13-terrain navigation).

## Registered Samplers

| Method key | Class | Mechanism | Notes / Expected Regime |
| :--- | :--- | :--- | :--- |
| `uniform` | `UniformSampler` | Uniform $1/N$ | Lower-bound reference |
| `tabular_sgs` | `TabularSGSSampler` | Per-config circular buffer ($H=100$, prior $0.5$) + SGS Beta kernel | Collapses when $N \gg$ completed episodes per config |
| `grid_sgs` | `GridSGSSampler` | Sparse active-cell spatial grid ($H=100$) + SGS kernel | Strong in 3D/5D; suffers curse of dimensionality in high-D / non-Euclidean spaces |
| `knn_sgs` | `KNNSGSSampler` | Chunked PyTorch time- + distance-weighted $k$-NN + SGS kernel | Strong non-parametric baseline in low/mid-D; degrades under distance concentration and distractor dims |
| `plr` | `PLRSampler` | Tabular Prioritized Level Replay (Jiang et al., ICML 2021): $(1-\rho)P_S + \rho P_C$ + unseen gate | Standard UED baseline; per-ID storage cannot generalize to unseen tasks in 1M pools |
| `neural_plr` | `NeuralPLRSampler` | 5-MLP ensemble predicting PLR score + staleness | Ablation decoupling neural generalization from SGS frontier vs. PLR value-loss |
| `nsgs` | `NSGSSampler` (v1) | 5-MLP ensemble mean, `history=1M`, `tau=100`, `sharp` kernel | Exact v1 baseline from the 3D/5D studies |
| `nsgs_v2` | `NSGSSampler` (v2) | 5-MLP ensemble + **one-sided epistemic optimism** ($\mu + \beta\sigma$ when $\mu < t$), `history=200K`, `tau=30`, `sharp_safe` kernel | Fixes pessimistic frontier lock-out and non-stationary tracking lag without spatial grids |

## Core Usage

```python
import numpy as np
from nsgs.baselines import TaskPool, build_sampler

# Build a pool once from raw task descriptors (e.g. 3D/5D/12D or terrain features)
pool = TaskPool.from_raw(raw_features, bounds=bounds, periodic_dims=(1,))

# Swap any baseline with one string key
sampler = build_sampler("nsgs_v2", pool, seed=73205, device="cuda:0")

for update in range(1, num_updates + 1):
    task_ids = sampler.sample(num_resets)
    # ... run PPO rollout, collect natural completions ...
    # `scores` (e.g., positive value loss / GAE L1) is used by PLR / Neural-PLR and ignored by SGS methods
    sampler.record(completed_task_ids, episode_successes, update=update, scores=episode_scores)
    diag = sampler.step_update(update)
```

All samplers isolate their RNG (`np.random.default_rng(seed)` / `torch.random.fork_rng`),
report unified diagnostics (`entropy`, `ess_fraction`, `top_1pct_mass`, `unvisited_prob_mass`,
`last_update_wall_s`), and support bit-exact checkpointing via `state_dict()` / `load_state_dict()`.

## Verification

Run the unit test suite (CPU-only, single-threaded BLAS):

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 python -m unittest tests/test_baselines.py -v
```

Run the synthetic multi-dimensional scaling benchmark (e.g. 12D with 4 active coupled
dimensions + 8 distractors):

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 python -m nsgs.baselines.benchmark \
  --num-tasks 8192 --total-dims 12 --active-dims 4 --updates 80 --seed 43
```
