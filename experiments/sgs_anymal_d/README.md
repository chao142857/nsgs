# ANYmal-D 十三地形 SGS

这是按 SGS 论文独立实现的导航实验源码。机器人用机身到达目标。训练版有 13 类地形，每类 2 个布局种子、10 个难度，每块地形有 20 个出生点和 20 个目标点。104,000 个组合中，52,000 个起终点在障碍两侧，52,000 个在同侧。稳定到达要求连续 20 个控制步，每步 0.02 秒。作者演示见 [SGS 项目页](https://sgs-rl.github.io/)。

| 内容 | 文件 |
| --- | --- |
| 地形几何、起终点生成、留出集 | [course.py](src/course.py)、[terrain_compat.py](src/terrain_compat.py) |
| ANYmal-D 物理场景、动作和奖励、终止条件 | [scene.py](src/scene.py)、[native.py](src/native.py)、[navigation.py](src/navigation.py) |
| 逐配置 SGS 采样和最近 100 回合历史 | [sampler.py](src/sampler.py)、[table.py](src/table.py) |
| PPO、评估和恢复 | [train.py](src/train.py)、[diagnostics.py](src/diagnostics.py)、[ppo_smoke.py](src/ppo_smoke.py) |

[protocol.json](protocol.json) 记录了运行配置和 15 个运行模块的 SHA-256。[catalog.json](catalog.json) 记录了训练与留出几何的校验值。在本目录可以用 NumPy 重新生成两套任务，再比较输出的两个哈希。

```sh
PYTHONPATH=src python -c 'from pathlib import Path; from course import build; print(build(Path("."))["geometry_sha"])'
```

仿真依赖外部提供的 Isaac Sim、Isaac Lab、RSL-RL 和 ANYmal-D 资产。历史启动入口沿用实验部署目录中的 Kit 配置和资产布局。几何生成可在 CPU 上单独运行。
