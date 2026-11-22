# Go2＋Piper 末端到达

目标是让机械臂末端到达固定的世界坐标。三维版沿用平地环境，五维版为每个回合另外指定地面摩擦系数和坡角。原三维配置的成功半径为 5 cm，连续保持 25 个控制步，共 0.5 秒。

| 版本 | 源码入口 |
| --- | --- |
| 原三维任务与 PPO | [环境与奖励](original_3d/goalreach/config.py)、[回合与成功判定](original_3d/goalreach/mdp.py)、[训练与评估](original_3d/tools/run.py)、[各臂配置](original_3d/configs/) |
| 后续三维采样对照 | [共享训练入口](shared/tools/run.py)、[百万池生成](shared/tools/build_p3b_pool.py)、[逐配置表、格子和神经估计器](shared/goalreach/) |
| 固定五维扩展 | [32,768 并行入口](fixed5d/run_v2.py)、[固定目标与物理条件生成](fixed5d/formal-v2-neural-32768/build_domain.py)、[坡面与摩擦](fixed5d/formal-v2-neural-32768/planes.py)、[神经臂训练](fixed5d/formal-v2-neural-32768/spatial_train.py) |

五维范围为摩擦系数 0.4–1.0、坡角 −21° 至 +21°。[四个 32,768 并行分支](fixed5d/)分别保留神经、格子、均匀和手工课程入口用到的源文件。默认神经分支使用 R0。[R1 差异](fixed5d/neural_r1.patch)记录实际运行过的奖励切换，可在 fixed5d 目录用 `patch -p1 < neural_r1.patch` 应用到神经分支。

历史代码依赖 Isaac Sim、Isaac Lab、RSL-RL、LeggedManip_Lab 及相应机器人资产。运行原三维版时将 `original_3d` 加入 Python 搜索路径。运行后续三维或五维版使用 `shared` 下的同名包。训练池、模型派生的运动学文件、部署回执和 Kit 路径由运行环境提供。历史硬件绑定检查里的设备标识改由 `NSGS_GPU_UUID_*` 和 `NSGS_HOSTNAME_*` 环境变量提供。[依赖与许可](../../THIRD_PARTY.md)
