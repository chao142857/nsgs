# ANYmal-D 13-Terrain 3D Trimesh Goal Navigation Benchmark

This directory contains the 1:1 3D Trimesh ANYmal-D goal-navigation benchmark across **13 procedural obstacle terrains** (`gap`, `pit`, `radial_beam`, `stairs`, `parallel_box`, `balance_beam`, `stepping_stones`, `jump_box`, `maze`, `contour`, `islands`, `climb_box`, `inverted_slope`) comparing **8 task-curriculum samplers** over **4 paired random seeds** (`s73205`, `s10001`, `s20261`, `s30307`) with identical `u0` initial policy weights per seed.

## Benchmark Setup

- **Task pool (`catalog_v4`, `v6` continuous terrains)**: 260 3D Trimesh tiles (`13 terrains x 2 seeds x 10 difficulty levels`), **104,000** training routes (`10%` Tier-0 Easy, `35%` Tier-1 Cross-Edge Bridge, `55%` Tier-2 Full Cross) and **2,080** disjoint pure full-cross holdout routes (`160 / terrain`, zero endpoint overlap with training).
- **Policy and recipe (`recipe-13t-v6-agile`)**: Bounded heteroscedastic Gaussian PPO (`4,096` parallel envs, `5,000` updates, `24` steps/update, `50 Hz` control), asymmetric arrival/drop terminals, and forward height scan `[-0.75, +1.45]m x [-0.70, +0.70]m`.
- **Samplers (`src/nsgs/baselines/`)**: `uniform`, `plr`, `tabular_sgs` (`eps=1e-4, t=0.5, k=1.0` and `sharp: eps=1e-8, t=0.66, k=5.0`), `grid_sgs`, `knn_sgs`, `nsgs` (v1 ensemble mean), and `nsgs_v2` (recency-weighted ensemble + epistemic optimism `beta=0.30` + adaptive target).

## Four-Seed Results (`u5000`, 2,080 Holdout Routes)

Full per-seed and per-terrain records are saved in [four_seed_u5000_summary.json](../../results/sgs_13terrain_trimesh/four_seed_u5000_summary.json).

| Rank | Method | 4-Seed Holdout Success (mean ± std) | Mean Routes (/2080) | Clean Success (mean ± std) | Mean Wall Time |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **1** | **`nsgs` (v1)** | **`75.20% ± 1.66%`** | **`1564.2`** | **`72.42% ± 3.19%`** | **`3.26 h`** |
| **2** | **`nsgs_v2`** | **`73.58% ± 1.63%`** | **`1530.5`** | `63.59% ± 13.31%` | `3.45 h` |
| 3 | `tabular_sgs_sharp` | `63.31% ± 2.87%` | `1316.8` | `62.01% ± 2.82%` | `3.44 h` |
| 4 | `tabular_sgs_eps1e4` | `62.76% ± 6.21%` | `1305.5` | `59.98% ± 5.62%` | `3.41 h` |
| 5 | `grid_sgs` | `58.35% ± 4.35%` | `1213.8` | `56.92% ± 4.21%` | `3.48 h` |
| 6 | `knn_sgs` | `58.28% ± 3.62%` | `1212.2` | `56.85% ± 3.25%` | `3.47 h` |
| 7 | `uniform` | `57.61% ± 3.36%` | `1198.2` | `56.47% ± 3.12%` | `3.38 h` |
| 8 | `plr` | `57.28% ± 4.02%` | `1191.5` | `55.83% ± 3.89%` | `3.37 h` |

Both neural curriculum variants (`nsgs` and `nsgs_v2`) outperform all six non-neural baselines on **4 out of 4 seeds** (`+11.89%` over `tabular_sgs_sharp`, `+17.59%` over `uniform`, and `+17.92%` over `plr`), while `nsgs` (v1) achieves the fastest wall-clock training time on every seed (`3.26 h` vs `3.37-3.48 h`).

## Source Files

| Component | Files |
| :--- | :--- |
| 13 procedural 3D Trimesh terrains (`v3` and `v6`) | [trimesh_terrains.py](src/trimesh_terrains.py) |
| 3-tier continuous task catalog generator (`catalog_v4`) | [build_catalog_v4.py](src/build_catalog_v4.py), [build_catalog.py](src/build_catalog.py) |
| Isaac Lab batched physics scene, observations, and rewards | [trimesh_scene.py](src/trimesh_scene.py), [navigation.py](src/navigation.py) |
| Bounded heteroscedastic PPO and multi-baseline trainer | [ppo_smoke.py](src/ppo_smoke.py), [trimesh_train.py](src/trimesh_train.py) |
| Deterministic 2,080-route evaluation and gait diagnostics | [eval_haa_full.py](src/eval_haa_full.py), [diagnostics.py](src/diagnostics.py), [gait_observables_v1.py](src/gait_observables_v1.py) |
| Shared 8-method sampler library | [src/nsgs/baselines](../../src/nsgs/baselines/) |

## Reproduction

Requires Isaac Sim 5.1, Isaac Lab, and RSL-RL.

```bash
# 1. Build the frozen 260-tile v6 catalog (104,000 train + 2,080 holdout routes)
python experiments/sgs_13terrain_trimesh/src/build_catalog_v4.py

# 2. Train nSGS (swap --method for uniform, plr, tabular_sgs, grid_sgs, knn_sgs, nsgs_v2)
python experiments/sgs_13terrain_trimesh/src/trimesh_train.py \
  --catalog experiments/sgs_13terrain_trimesh/catalog_v4 \
  --method nsgs \
  --seed 73205 \
  --updates 5000 \
  --num-envs 4096 \
  --recipe recipe-13t-v6-agile \
  --out runs/s73205/nsgs

# 3. Run deterministic 2,080-route heldout evaluation
python experiments/sgs_13terrain_trimesh/src/eval_haa_full.py \
  --catalog experiments/sgs_13terrain_trimesh/catalog_v4 \
  --checkpoint runs/s73205/nsgs/checkpoint-5000.pt \
  --recipe recipe-13t-v6-agile \
  --out runs/s73205/nsgs/eval_haa_full-u5000
```
