# nSGS

**Neural Success-Guided Sampling for Whole-Body Control**

English · [简体中文](README.zh-CN.md)

A Go2 quadruped with a Piper arm learns to reach a target with its end effector using PPO. Each naturally completed training episode supplies a success or failure label. The sampler estimates the current policy’s probability of success for each goal and uses those estimates to select subsequent goals. We compare three estimators under a common sampling kernel: a table indexed by goal, a table pooling observations within spatial cells, and a neural predictor trained across goals.

## Goal reaching on a million-task pool

On a fixed pool of 1,048,576 three-dimensional goal configurations, a neural success estimator reached **98.51% walking success** and **98.50% equal-weight coverage across 42 cells** on seed 43. A per-configuration table scored 0% on both metrics. Both runs used the same sharp sampling kernel, 8,000 PPO updates and 2,016 fixed evaluation tasks. Endpoint values average the last three deterministic EMA evaluations.

A spatial table with 16 bins per dimension also learned the task. On seed 43 it reached the walking threshold at update 2,400, ahead of the neural estimator at 5,600 (three consecutive evaluations at or above 50%). This comparison motivates testing which forms of sharing work across goals and seeds. The seed 43 scientific audit passed. Host swap activity exceeded the original resource threshold, so we omit wall-clock comparisons here.

Earlier implementations and results are documented in the [experiment history](docs/experiments.md).

## Acknowledgements

This research builds on:

- [A Balanced Data Diet (SGS)](https://arxiv.org/abs/2610.12465v1)
- [LeggedManip_Lab](https://github.com/zzzJie-Robot/LeggedManip_Lab)
- [Isaac Lab](https://github.com/isaac-sim/IsaacLab)
- [RSL-RL](https://github.com/leggedrobotics/rsl_rl)

The experiments reported here are conducted in simulation.

## License

Original code is released under [Apache-2.0](LICENSE). Upstream implementations and external robot assets remain subject to their respective licenses. See [THIRD_PARTY.md](THIRD_PARTY.md), [NOTICE](NOTICE), and [LICENSES](LICENSES).
