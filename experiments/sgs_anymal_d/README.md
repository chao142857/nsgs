# ANYmal-D 十三地形 SGS 单卡先导

这个目录复建论文的四足导航实验。一个目标条件策略从随机权重训练，动作是 12 个关节的位置偏移。训练用附录 A.3.1 的七项共享奖励。任务库有 104,000 个固定组合。三秒站稳和各地形的足端事件独立评估。[论文 PDF，第 8–9 页](https://arxiv.org/pdf/2610.12465#page=8) · [作者项目页及 ANYmal-D 视频](https://sgs-rl.github.io/)

## 从干净检出重跑

在 Linux 的 Isaac Lab 环境中运行。外部依赖为 Isaac Sim 5.1、Isaac Lab 的 ANYmal-D 机器人配置及可访问的 USD/执行器资产、RSL-RL、PyTorch、NumPy。此实现使用公开配置中源自 ANYmal-C 的学习型执行器。将 `SGS_PYTHON` 设为已安装这些依赖的解释器，`SGS_KIT_EXPERIENCE` 指向该版本 headless Kit 文件，`SGS_ASSET_ROOT` 指向包含 Isaac 资产的根目录。图形设备索引与 CUDA_VISIBLE_DEVICES 中绑定的物理卡必须一致。

```sh
git clone --branch reproduction/sgs-anymal-fig7 --single-branch https://github.com/chao142857/nsgs.git
cd nsgs/experiments/sgs_anymal_d
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PXR_WORK_THREAD_LIMIT=2
bash run_stage.sh build
bash run_stage.sh cpu --report runs/cpu.json
bash run_stage.sh physics --device-index 0 --out runs/physics
bash run_stage.sh pilot --device-index 0 --out runs/seed73205 --seed 73205 --num-envs 4096 --updates 500 --eval-at 250 500 --physics-receipt runs/physics/physics_acceptance.json
for step in 0250 0500; do
  bash run_stage.sh evaluate --device-index 0 --checkpoint "runs/seed73205/checkpoint-${step}.pt" --out "runs/eval-${step}"
done
```

录像工具在 macOS 下使用 Xcode 命令行工具、SceneKit、Metal 和 AVFoundation。在 Mac 上把对应的评估目录作为输入，运行以下命令。渲染器读取真实每步刚体姿态和实验使用的本地机器人网格，按模拟时间戳编码，保留首次终局帧。原始接触、足端、地表高度和动作位于每段对应的 NPZ。

```sh
bash run_stage.sh render runs/eval-0500 runs/movies-0500
```

每次独立实验须使用新的工作检出及输出目录。本轮共用一份 500 更账本，恢复必须提供同一版本完整检查点。所有 PPO 标签来自自然结束的训练回合。两次离线推理重用同一固定评估面板。池、几何、俯视图、证据与实验结论均标注其适用版本。
