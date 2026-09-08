# Hunyuan3D Examples 定性评估脚本

本目录的五个脚本用于比较 `hunyuan3d-examples/` 中 8 个 GLB 在原始面数和约 10k 面两种输入下的 Particulate 结果。流程覆盖网格简化、几何分析、预测分析、关节运动环视渲染，以及最终产物校验和报告生成。

固定样本为：

```text
cabinet eyeglasses foldingchair laptop scissors toilet trashcan washingmachine
```

这组样本没有 articulation ground truth。报告衡量两分支的一致性和视觉合理性，不应解释为模型准确率。

## 环境

从仓库根目录执行：

```bash
source /home/LiuShuqi/miniconda3/etc/profile.d/conda.sh
conda activate /home/LiuShuqi/.conda/envs/particulate
cd /data1/LiuShuqi/code/articulation/baselines/particulate

export ROOT=/data2/LiuShuqi/output/particulate/hunyuan3d-examples-qualitative
export PARTFIELD_MODEL_DIR=/data2/LiuShuqi/output/particulate/checkpoints/partfield
export HF_HOME=/data2/LiuShuqi/output/particulate/huggingface
export XDG_CACHE_HOME=/data2/LiuShuqi/output/particulate/cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

所需工具和依赖：

- Conda 环境：NumPy、SciPy、trimesh、PyTorch、Particulate 和 PartField。
- Blender 2.90：当前 `blender` 位于 `/home/LiuShuqi/Workspace/software/blender-2.90.0-linux64/blender`。
- `xvfb-run`：在无桌面会话时为 Blender 提供显示上下文。
- `ffmpeg`、`ffprobe`：合成视频和校验视频规格。
- ImageMagick `montage`：生成关键帧总览图。

模型路径：

```text
/data2/LiuShuqi/output/particulate/checkpoints/particulate/model.pt
/data2/LiuShuqi/output/particulate/checkpoints/partfield/model_objaverse.ckpt
```

## 脚本

### `blender_simplify_glb.py`

通过 Blender 合并 mesh object、三角化并执行 Collapse Decimate，然后导出单个 GLB。输出先写临时文件，再原子移动到目标路径。

```bash
xvfb-run -a blender -b --python scripts/blender_simplify_glb.py -- \
  --input hunyuan3d-examples/foldingchair.glb \
  --output "$ROOT/preprocessed-10k/foldingchair.glb" \
  --target-faces 10000
```

标准输出最后一行为 JSON，包含原始面数、最终面数、顶点数和材质槽数量。脚本保留材质和 UV，但会合并对象，因此不保留输入 GLB 的对象级语义边界。

### `analyze_hunyuan_simplification.py`

对每一对原始/简化网格采样表面点，输出：

- 顶点数、面数、缩减比例和 SHA-256。
- 基于 face adjacency 的 connected-component 数及 watertight 状态。
- 表面积、包围盒中心与尺寸变化。
- 对称 mean、RMS、p95 和 sampled Hausdorff 距离，均除以原始包围盒对角线。
- 最近邻表面法向的绝对余弦一致率。

```bash
python scripts/analyze_hunyuan_simplification.py \
  --original-dir hunyuan3d-examples \
  --simplified-dir "$ROOT/preprocessed-10k" \
  --output-dir "$ROOT/analysis" \
  --num-points 100000
```

输出为 `geometry_comparison.json` 和 `geometry_comparison.csv`。采样种子固定，重复分析可复现。

### `analyze_hunyuan_predictions.py`

读取两分支的 `<sample>/eval/pred.obj` 和 `pred.npz`，对预测表面各采样 100k 点，再通过双向最近邻和 Hungarian matching 建立 part 对应。

主要指标：

- `matched_part_mIoU`：只对已匹配 part 求宏平均 IoU。
- `part_mIoU_unmatched_zero`：未匹配 part 计零的宏平均 IoU，推荐用于跨分支比较。
- `symmetric_label_agreement`：双向最近邻后的面积加权标签一致率。
- `motion_class_agreement_unmatched_zero`：比较 fixed、revolute、prismatic，未匹配 part 计零。
- `axis_abs_cosine`：匹配运动 part 的轴方向绝对余弦。
- `revolute_axis_line_distance_normalized`：旋转轴线距离，坐标按 mesh 最大边长归一化。
- `revolute_range_endpoint_mae_radians`：旋转范围端点 MAE，单位为弧度。
- `prismatic_range_endpoint_mae_normalized`：平移范围端点 MAE，单位为最大边长归一化距离。
- `hierarchy_edge_jaccard`：映射后的层级边 Jaccard；未匹配 part 的边仍参与惩罚。

```bash
python scripts/analyze_hunyuan_predictions.py \
  --original-root "$ROOT/predictions/original" \
  --simplified-root "$ROOT/predictions/10k" \
  --output-dir "$ROOT/analysis" \
  --num-points 100000
```

输出为 `prediction_comparison.json` 和 `prediction_comparison.csv`。两分支使用固定但不同的分析采样种子，结果可重复且不会复用同一随机序列。

该脚本按本次 `--up_dir=-Z` 的坐标约定对 `pred.obj` 做对齐。若推理使用其他 up direction，必须同步修改 `load_prediction()` 中的旋转矩阵。

### `blender_render_orbit.py`

导入 `animated_textured_*.glb`，将关节动画线性重定时到目标帧数，并让相机绕 Z 轴旋转 360 度。脚本按 mesh object 名称分配 part 颜色，根据动画起点、中点和终点自动取景，并用 EEVEE 输出 H.264 MP4。

```bash
xvfb-run -a blender -b --python scripts/blender_render_orbit.py -- \
  --input "$ROOT/predictions/original/foldingchair/animated_textured_<timestamp>.glb" \
  --output "$ROOT/renders/original/foldingchair.mp4" \
  --frames 120 \
  --fps 30 \
  --resolution 720 \
  --samples 8
```

输入必须含动画 keyframe。颜色仅用于辨识；若 part 数或对象排序改变，同一颜色不保证代表跨分支的同一语义 part。

### `summarize_hunyuan_qualitative.py`

执行最终强校验并生成 manifest 和中文报告。它会检查：

- 每个分支恰好有一个 animated GLB、一个 axes GLB、`pred.obj` 和 `pred.npz`。
- NPZ 字段齐全、数组有限、part 与 hierarchy 索引合法。
- 16 个单分支视频均为 H.264、yuv420p、720x720、30 fps、120 帧。
- 8 个并排视频均为 H.264、yuv420p、1440x720、30 fps、120 帧。
- 8 张逐样本 contact sheet 和总览图存在且非空。
- 成功推理与渲染日志不存在 traceback、OOM、显示错误或样本级异常。

```bash
python scripts/summarize_hunyuan_qualitative.py --root "$ROOT"
```

输出为 `$ROOT/manifest.json` 和 `$ROOT/analysis/REPORT.md`。脚本内的 `QUALITATIVE_NOTES` 是针对固定 8 个样本的人工观察；更换样本集时必须同步修改。

## 完整流程

### 1. 生成 10k GLB

```bash
mkdir -p \
  "$ROOT/preprocessed-10k" \
  "$ROOT/analysis/contact-sheets" \
  "$ROOT/logs/preprocess" \
  "$ROOT/logs/infer" \
  "$ROOT/logs/render" \
  "$ROOT/predictions/original" \
  "$ROOT/predictions/10k" \
  "$ROOT/renders/original" \
  "$ROOT/renders/10k" \
  "$ROOT/renders/comparison"

for name in cabinet eyeglasses foldingchair laptop scissors toilet trashcan washingmachine; do
  xvfb-run -a blender -b --python scripts/blender_simplify_glb.py -- \
    --input "hunyuan3d-examples/$name.glb" \
    --output "$ROOT/preprocessed-10k/$name.glb" \
    --target-faces 10000 \
    > "$ROOT/logs/preprocess/$name.log" 2>&1
done
```

先运行几何分析并确认面数、有限顶点、材质类型和包围盒变化，再启动推理。

### 2. GPU 分片

| GPU | 样本 |
|---|---|
| 4 | cabinet、foldingchair、scissors、trashcan |
| 5 | eyeglasses、laptop、toilet、washingmachine |

分片目录只放指向仓库 GLB 或 `$ROOT/preprocessed-10k` 的软链接：

```text
$ROOT/inputs/original/gpu4/*.glb
$ROOT/inputs/original/gpu5/*.glb
$ROOT/inputs/10k/gpu4/*.glb
$ROOT/inputs/10k/gpu5/*.glb
```

创建命令：

```bash
for variant in original 10k; do
  for gpu in 4 5; do
    mkdir -p "$ROOT/inputs/$variant/gpu$gpu"
  done
done

for name in cabinet foldingchair scissors trashcan; do
  test -e "$ROOT/inputs/original/gpu4/$name.glb" ||
    ln -s "$PWD/hunyuan3d-examples/$name.glb" \
      "$ROOT/inputs/original/gpu4/$name.glb"
  test -e "$ROOT/inputs/10k/gpu4/$name.glb" ||
    ln -s "$ROOT/preprocessed-10k/$name.glb" \
      "$ROOT/inputs/10k/gpu4/$name.glb"
done

for name in eyeglasses laptop toilet washingmachine; do
  test -e "$ROOT/inputs/original/gpu5/$name.glb" ||
    ln -s "$PWD/hunyuan3d-examples/$name.glb" \
      "$ROOT/inputs/original/gpu5/$name.glb"
  test -e "$ROOT/inputs/10k/gpu5/$name.glb" ||
    ln -s "$ROOT/preprocessed-10k/$name.glb" \
      "$ROOT/inputs/10k/gpu5/$name.glb"
done
```

### 3. 双分支推理

每张卡依次运行原始和 10k 分支。GPU 5 命令只需改为 `GPU=5`：

```bash
GPU=4
for variant in original 10k; do
  CUDA_VISIBLE_DEVICES="$GPU" python infer.py \
    --input_mesh "$ROOT/inputs/$variant/gpu$GPU/*.glb" \
    --ckpt_path /data2/LiuShuqi/output/particulate/checkpoints/particulate/model.pt \
    --output_dir "$ROOT/predictions/$variant" \
    --up_dir=-Z \
    --num_points 51200 \
    --animation_frames 120 \
    --eval \
    > "$ROOT/logs/infer/$variant-gpu$GPU.log" 2>&1
done
```

不要传 `--no_strict`。`infer.py` 捕获单样本异常后会继续并可能返回退出码 0，因此必须逐样本检查 GLB、OBJ 和 NPZ。

### 4. 渲染和并排视频

对两分支的 16 个 animated GLB 分别调用 `blender_render_orbit.py`：

```bash
for variant in original 10k; do
  for name in cabinet eyeglasses foldingchair laptop scissors toilet trashcan washingmachine; do
    input=$(find "$ROOT/predictions/$variant/$name" -maxdepth 1 \
      -type f -name 'animated_textured_*.glb' -print -quit)
    test -n "$input"
    xvfb-run -a blender -b --python scripts/blender_render_orbit.py -- \
      --input "$input" \
      --output "$ROOT/renders/$variant/$name.mp4" \
      --frames 120 --fps 30 --resolution 720 --samples 8 \
      > "$ROOT/logs/render/$variant-$name.log" 2>&1
  done
done
```

生成左原始、右 10k 的带标签并排视频：

```bash
FONT=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf
for name in cabinet eyeglasses foldingchair laptop scissors toilet trashcan washingmachine; do
  ffmpeg -y -v warning \
    -i "$ROOT/renders/original/$name.mp4" \
    -i "$ROOT/renders/10k/$name.mp4" \
    -filter_complex \
      "[0:v]drawtext=fontfile=$FONT:text=Original:x=24:y=24:fontsize=30:fontcolor=white:box=1:boxcolor=black@0.55[left];[1:v]drawtext=fontfile=$FONT:text='10k faces':x=24:y=24:fontsize=30:fontcolor=white:box=1:boxcolor=black@0.55[right];[left][right]hstack=inputs=2[v]" \
    -map '[v]' -c:v libx264 -preset medium -crf 18 \
    -pix_fmt yuv420p -movflags +faststart \
    "$ROOT/renders/comparison/$name.mp4" \
    > "$ROOT/logs/render/comparison-$name.log" 2>&1
done
```

抽取第 1、40、80、120 帧并生成总览：

```bash
for name in cabinet eyeglasses foldingchair laptop scissors toilet trashcan washingmachine; do
  ffmpeg -y -v error -i "$ROOT/renders/comparison/$name.mp4" \
    -vf 'select=eq(n\,0)+eq(n\,39)+eq(n\,79)+eq(n\,119),scale=720:360,tile=2x2' \
    -frames:v 1 "$ROOT/analysis/contact-sheets/$name.png"
done

montage -label '%t' "$ROOT"/analysis/contact-sheets/*.png \
  -thumbnail 720x360 -tile 2x4 -geometry +8+24 \
  -background '#10141b' -fill white -pointsize 18 \
  "$ROOT/analysis/contact-sheet-overview.png"
```

最后通过 `ffprobe -count_frames` 或 `summarize_hunyuan_qualitative.py` 验证编码、尺寸、帧率和实际帧数。

### 5. 分析和汇总

```bash
python scripts/analyze_hunyuan_simplification.py \
  --original-dir hunyuan3d-examples \
  --simplified-dir "$ROOT/preprocessed-10k" \
  --output-dir "$ROOT/analysis" --num-points 100000

python scripts/analyze_hunyuan_predictions.py \
  --original-root "$ROOT/predictions/original" \
  --simplified-root "$ROOT/predictions/10k" \
  --output-dir "$ROOT/analysis" --num-points 100000

python scripts/summarize_hunyuan_qualitative.py --root "$ROOT"
```

## 输出结构

```text
$ROOT/
├── analysis/
│   ├── REPORT.md
│   ├── contact-sheet-overview.png
│   ├── contact-sheets/<sample>.png
│   ├── geometry_comparison.{json,csv}
│   └── prediction_comparison.{json,csv}
├── inputs/{original,10k}/gpu{4,5}/
├── logs/{preprocess,infer,render}/
├── manifest.json
├── predictions/{original,10k}/<sample>/
├── preprocessed-10k/<sample>.glb
└── renders/
    ├── original/<sample>.mp4
    ├── 10k/<sample>.mp4
    └── comparison/<sample>.mp4
```

本次输出根目录为 `/data2/LiuShuqi/output/particulate/hunyuan3d-examples-qualitative`。

## 指标解释限制

- 原始网格有 37,892-50,000 面；`num_points=51200` 只包含 25,600 个均匀表面点，其余为 sharp-edge 点，因此原始分支不能保证逐面覆盖。10k 分支的均匀采样点数高于面数。
- `infer.py` 固定 PyTorch 种子，但未固定 NumPy 表面采样种子。小幅边界、range 或 part 变化同时包含推理采样噪声，不能全部归因于 decimation。
- 两个分析脚本使用固定采样种子，因此对既有产物重复分析是确定的。
- strict connectivity 对 connected-component topology 敏感。即使表面距离很小，component 合并或拆分仍可能改变最终 part。
- 宏平均 part mIoU 对微小 part 很敏感，应与面积加权 `symmetric_label_agreement`、part 数变化和未匹配面积占比一起阅读。
- 颜色只服务于视频辨识，不是跨分支 part 对应依据。
- 无 GT 条件下，“两分支一致”只能证明简化稳定，不能证明预测语义正确。

## 兼容性事项

- Blender 2.90 的 `export_scene.gltf(export_materials=...)` 要求布尔值；当前脚本使用 `True`，不要改为新版 Blender 示例中的字符串枚举。
- 无 `DISPLAY` 时直接运行 Blender 会报 `Unable to open a display`，统一通过 `xvfb-run -a` 启动。
- Blender 可能打印 `id_us_min` 和 `/run/user/.../gvfs` 信息；只要最终 JSON、GLB/MP4 和强校验通过，这些信息不表示样本失败。
- 现有输出保留首次 cabinet 参数不兼容日志和四个首次无 Xvfb 渲染日志，均列于 `manifest.json.validation.recovered_failures`。
