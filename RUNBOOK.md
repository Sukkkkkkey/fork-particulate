# Particulate 运行说明

> **这是一份活文档。** `当前代码版本`必须随工作进展持续更新

## 固定路径

- 代码：`/data1/LiuShuqi/code/articulation/baselines/particulate`
- Conda：`/home/LiuShuqi/miniconda3/bin/conda`
- 环境：`/home/LiuShuqi/.conda/envs/particulate`
- 环境版本：Python 3.9.12、PyTorch 2.6.0+cu124、PyTorch3D 0.7.9（使用服务器现有 CUDA 12.4）
- 原始数据：`/data2/LiuShuqi/data/raw/<dataset>`（只读）
  - PartNetMobility
  - Lightwheel-simready-asset/unzipped/Lightwheel_OpenSource
- 处理数据：`/data2/LiuShuqi/data/processed/<dataset>`
  - PartNetMobility_test_particulate
  - Lightwheel_test_particulate
- 权重、缓存、日志和结果：`/data2/LiuShuqi/output/particulate`

## 进入环境

```bash
source /home/LiuShuqi/miniconda3/etc/profile.d/conda.sh
conda activate /home/LiuShuqi/.conda/envs/particulate
cd /data1/LiuShuqi/code/articulation/baselines/particulate
```

## 必要环境变量

```bash
export PARTICULATE_OUTPUT_DIR=/data2/LiuShuqi/output/particulate/inference
export PARTFIELD_MODEL_DIR=/data2/LiuShuqi/output/particulate/checkpoints/partfield
export HF_HOME=/data2/LiuShuqi/output/particulate/huggingface
export XDG_CACHE_HOME=/data2/LiuShuqi/output/particulate/cache
export PIP_CACHE_DIR=/data2/LiuShuqi/output/particulate/cache/pip
export TORCH_HOME=/data2/LiuShuqi/output/particulate/cache/torch
export WANDB_DIR=/data2/LiuShuqi/output/particulate/logs/wandb
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HTTP_PROXY=http://127.0.0.1:8888
export HTTPS_PROXY=http://127.0.0.1:8888
```

主模型固定为 `/data2/LiuShuqi/output/particulate/checkpoints/particulate/model.pt`；PartField 权重固定为 `$PARTFIELD_MODEL_DIR/model_objaverse.ckpt`。

### PyTorch3D 离线配置

当前 wheel 基于官方源码提交 `e73a7e7bfc580d1f9225ad9ff7d2c753c320aabd` 构建，只包含 RTX 4090 所需的 CUDA 架构 8.9。源码、wheel 和日志分别位于：

- `/data2/LiuShuqi/output/particulate/cache/pytorch3d-src`
- `/data2/LiuShuqi/output/particulate/cache/wheels`
- `/data2/LiuShuqi/output/particulate/logs/pytorch3d`

在已有 wheel 的服务器上使用以下命令离线安装，不需要代理：

```bash
python -m pip install -r requirements-pytorch3d-local.txt
```

若 PyTorch 或 CUDA ABI 发生变化，从本地源码重新构建：

```bash
CUDA_HOME=/usr/local/cuda-12.4 FORCE_CUDA=1 \
TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=4 PIP_NO_INDEX=1 \
python -m pip wheel --no-deps --no-build-isolation \
  --wheel-dir /data2/LiuShuqi/output/particulate/cache/wheels \
  /data2/LiuShuqi/output/particulate/cache/pytorch3d-src
```

## 示例推理

仓库原始 `foldingchair.glb` 有 50,000 个面；默认 102,400 点在 48 GB RTX 4090 上会 OOM。已验证的 smoke test 使用 output 中的 10,000 面简化副本：

```bash
CUDA_VISIBLE_DEVICES=5 python infer.py \
  --input_mesh /data2/LiuShuqi/output/particulate/inference/inputs/foldingchair_10000.obj \
  --ckpt_path /data2/LiuShuqi/output/particulate/checkpoints/particulate/model.pt \
  --output_dir /data2/LiuShuqi/output/particulate/inference/foldingchair-smoke \
  --up_dir=-Z \
  --num_points 22000
```

运行前用 `nvidia-smi` 替换为当时空闲的 GPU。`num_points=22000` 中有 11,000 个均匀采样点，仍大于简化网格的 10,000 个面。成功后应生成 `mesh_parts_with_axes_*.glb` 和 `animated_textured_*.glb`。

- 作者在 https://github.com/RuiningLi/particulate/issues/4 提到将 `num_points` 设为 51,200 即可满足 48GB 显存推理。

## Evaluation 最小验证

当前已在 `/data2/LiuShuqi/output/particulate/evaluation-smoke` 跑通单样本评测。
推理增加 `--eval` 命令行参数使 `$output_dir/eval`下保存 `pred.obj` 与 `pred.npz` ：

```bash
CUDA_VISIBLE_DEVICES=2 python infer.py \
  --input_mesh /data2/LiuShuqi/output/particulate/inference/inputs/foldingchair_10000.obj \
  --ckpt_path /data2/LiuShuqi/output/particulate/checkpoints/particulate/model.pt \
  --output_dir /data2/LiuShuqi/output/particulate/evaluation-smoke/predictions/foldingchair-smoke \
  --up_dir=-Z --num_points 22000 --animation_frames 2 --eval

CUDA_VISIBLE_DEVICES=2 python evaluate.py \
  --gt_dir /data2/LiuShuqi/output/particulate/evaluation-smoke/gt \
  --result_dir /data2/LiuShuqi/output/particulate/evaluation-smoke/predictions \
  --output_dir /data2/LiuShuqi/output/particulate/evaluation-smoke/metrics-pytorch3d \
  --num_points 2000
```

该 smoke GT 从同一次预测中采样，只验证字段、匹配、关节变换和指标输出，不代表模型真实性能。
README 说对每个预处理资产执行 `python -m particulate.data.cache_points --format eval` 生成同名 GT，但 `particulate.data.cache_gt` 实际已并入`particulate.data.cache_points`，执行时指定 `--format eval` 即可。

## 数据集处理约定

- 从 `/data2/LiuShuqi/data/raw/<dataset>` 读取原始 URDF/USD，不得原地修改。
- 将 `process_urdf`、`process_usd` 和 `cache_points` 的结果写入 `/data2/LiuShuqi/data/processed/<dataset>`。
- 训练时将 `configs/train-particulate-B.yaml` 的 `output_dir` 改为 `/data2/LiuShuqi/output/particulate/<run>`，并把数据集 root 指向 processed 目录。

## 当前代码版本

- 远程：`https://github.com/Sukkkkkkey/fork-particulate.git`
- 分支：`main`
- 初始提交：`dee37a75c449f324d9989993461ee09eaccc1686`
- 仓库采用稀疏检出：包含完整代码及 `hunyuan3d-examples`；`assets` 可按需用 `git sparse-checkout` 补取。

代码修改
- inference：`PartField` 的演示数据加载器改为按需导入；推理不要求安装仅用于其数据处理的 `mesh2sdf`、`tetgen`、`vtk` 和 `pymeshlab`。

## 意外发现

- evaluation 生成 GT：原作者 README.md 提到的 `particulate.data.cache_gt` 已并入`particulate.data.cache_points`，执行时调用 `cache_points --format eval`
