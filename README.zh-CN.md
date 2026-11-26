# nSGS

**面向全身控制的神经成功引导采样**

[English](README.md) · 简体中文

机器人用 PPO 学习走到目标附近并让机械臂末端到达指定位置。每个自然结束的训练回合提供一个成功或失败标签。采样器用这些标签估计当前策略在各目标上的成功率，再用同一采样核选择后续目标。我们比较三种使用标签的方式：逐目标单独计数、相邻目标在格子内共享计数，以及用神经网络跨目标预测。

## 实验代码

- [统一 8 方法课程采样器库（`nsgs.baselines`）](docs/baselines.md)
- [ANYmal-D 十三地形 3D Trimesh 8 方法 × 4 种子正式基准](experiments/sgs_13terrain_trimesh/)
- [ANYmal-D 十三地形 SGS 导航（早期快照）](experiments/sgs_anymal_d/)
- [Go2＋Piper 末端到达，原三维与固定五维](experiments/go2_piper_goalreach/)

## ANYmal-D 十三地形 3D Trimesh 四种子正式基准

在 260 块三维三角网格地形（13 类障碍、**104,000** 条训练路线、**2,080** 条完全不重叠的跨障留出路线、5,000 次 PPO 更新、4 组配对随机种子）上，**`nSGS` (v1)** 取得 **`75.20% ± 1.66%`**（`1,564.2 / 2,080`）留出成功率，且四轮训练耗时均最快（**`3.26 h`**）；**`nSGS-v2`** 取得 **`73.58% ± 1.63%`**（`1,530.5 / 2,080`）。两种神经采样器在 **4 / 4 种子上全部领先全部 6 个非神经基线**（`tabular_sgs_sharp`: `63.31%`、`tabular_sgs_eps1e4`: `62.76%`、`grid_sgs`: `58.35%`、`knn_sgs`: `58.28%`、`uniform`: `57.61%`、`plr`: `57.28%`）。详见 [experiments/sgs_13terrain_trimesh](experiments/sgs_13terrain_trimesh/) 与 [four_seed_u5000_summary.json](results/sgs_13terrain_trimesh/four_seed_u5000_summary.json)。

## 百万任务池上的目标到达实验

在由距离、方位和高度定义的 1,048,576 个目标配置上，seed 43 的神经成功率估计器取得 **98.51% 的行走成功率**和 **98.50% 的 42 格等权覆盖**，逐配置查表的两项结果均为 0%。两臂使用同一尖核、8,000 次 PPO 更新和 2,016 项固定评估任务。终点取末三次确定性 EMA 评估的均值。

按空间位置共享数据的 16³ 格子表同样学会了任务。seed 43 上，格子表在第 2,400 次更新达到行走门槛，神经估计器在第 5,600 次达到门槛，判据均为连续三次评估成功率至少 50%。后续配对实验考察这两种共享方式的差距能否跨种子复现。seed 43 科学审计通过，宿主机换页超过原资源门槛，此处仅比较更新次数和任务表现。

此前实验的实现与结果见[实验历史](docs/experiments.md)。

研究基于 [SGS](https://arxiv.org/abs/2610.12465v1)、[LeggedManip_Lab](https://github.com/zzzJie-Robot/LeggedManip_Lab)、[Isaac Lab](https://github.com/isaac-sim/IsaacLab) 与 [RSL-RL](https://github.com/leggedrobotics/rsl_rl)。全部结果来自仿真。原创代码采用 [Apache-2.0](LICENSE)，上游代码与资产遵循各自许可，见 [第三方声明](THIRD_PARTY.md)。
