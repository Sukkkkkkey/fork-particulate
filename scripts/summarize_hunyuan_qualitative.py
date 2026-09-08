#!/usr/bin/env python3
"""Validate and summarize the Hunyuan original-vs-10k qualitative run."""

import argparse
import json
import math
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


REQUIRED_PREDICTION_FIELDS = {
    "face_part_ids",
    "motion_hierarchy",
    "is_part_revolute",
    "is_part_prismatic",
    "revolute_plucker",
    "revolute_range",
    "prismatic_axis",
    "prismatic_range",
}

QUALITATIVE_NOTES = {
    "cabinet": (
        "两分支都预测出固定柜体和两个旋转面板，配对分割与面板运动轨迹在视觉上几乎一致。"
    ),
    "eyeglasses": (
        "两分支都恢复出固定镜框和两个旋转镜腿；简化后镜腿运动仍与原面数结果对齐。"
    ),
    "foldingchair": (
        "两分支高度一致，但都只预测两个 part 和一个旋转关节，把真实的多连杆折叠机构合并了；"
        "这是与简化无关的欠关节化错误。"
    ),
    "laptop": (
        "两分支都恢复出预期的两部件铰链结构，分割和轴稳定，但旋转范围端点差异大于其他样本。"
    ),
    "scissors": (
        "两分支都恢复出两个刚性半部和一个相对旋转关节，运动合理，剩余差异集中在局部边界。"
    ),
    "toilet": (
        "原面数分支预测底座、座圈和盖板；10k 分支在保留两个主要旋转 part 的同时新增一个很小的"
        "平移 part，座圈和盖板的主体轨迹仍然对齐。"
    ),
    "trashcan": (
        "两分支都预测三个 part 和两个旋转运动；盖板运动和主体分割一致，差异局限于小部件或边界。"
    ),
    "washingmachine": (
        "两分支都预测四个 part，其中三个运动。主体标签一致，但小 part 在宏平均 IoU 下差异很大；"
        "对于常见滚筒洗衣机，这种多关节动画本身也存在合理性疑问。"
    ),
}


def probe_video(path: Path):
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=codec_name,width,height,r_frame_rate,nb_read_frames,pix_fmt",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    streams = json.loads(result.stdout).get("streams", [])
    if len(streams) != 1:
        raise ValueError(f"Expected one video stream in {path}")
    stream = streams[0]
    stream["nb_read_frames"] = int(stream["nb_read_frames"])
    return stream


def validate_prediction(prediction_dir: Path):
    eval_dir = prediction_dir / "eval"
    npz_path = eval_dir / "pred.npz"
    obj_path = eval_dir / "pred.obj"
    animated = sorted(prediction_dir.glob("animated_textured_*.glb"))
    axes = sorted(prediction_dir.glob("mesh_parts_with_axes_*.glb"))
    if len(animated) != 1 or len(axes) != 1:
        raise ValueError(f"Expected one animated and one axes GLB in {prediction_dir}")
    required_files = [npz_path, obj_path, animated[0], axes[0]]
    if any(not path.is_file() or path.stat().st_size == 0 for path in required_files):
        raise ValueError(f"Missing or empty prediction artifact in {prediction_dir}")

    with np.load(npz_path, allow_pickle=False) as values:
        missing = REQUIRED_PREDICTION_FIELDS - set(values.files)
        if missing:
            raise ValueError(f"Missing fields in {npz_path}: {sorted(missing)}")
        if not all(np.isfinite(values[key]).all() for key in values.files):
            raise ValueError(f"Non-finite prediction values in {npz_path}")
        num_parts = len(values["is_part_revolute"])
        expected_first_dimension = (
            "is_part_prismatic",
            "revolute_plucker",
            "revolute_range",
            "prismatic_axis",
            "prismatic_range",
        )
        if any(len(values[key]) != num_parts for key in expected_first_dimension):
            raise ValueError(f"Inconsistent part dimensions in {npz_path}")
        labels = values["face_part_ids"]
        if labels.size == 0 or labels.min() < 0 or labels.max() >= num_parts:
            raise ValueError(f"Invalid face labels in {npz_path}")
        hierarchy = np.asarray(values["motion_hierarchy"])
        if hierarchy.size:
            hierarchy = hierarchy.reshape(-1, 2)
            if hierarchy.min() < 0 or hierarchy.max() >= num_parts:
                raise ValueError(f"Invalid hierarchy indices in {npz_path}")

    return {
        "num_parts": num_parts,
        "num_faces": int(labels.size),
        "pred_npz": str(npz_path),
        "pred_obj": str(obj_path),
        "animated_glb": str(animated[0]),
        "axes_glb": str(axes[0]),
    }


def load_rows(path: Path):
    rows = json.loads(path.read_text())
    return {row["name"]: row for row in rows}


def scan_success_logs(root: Path):
    pattern = re.compile(
        r"Error processing|Traceback|CUDA out of memory|Unable to open a display|Killed"
    )
    logs = list((root / "logs" / "infer").glob("*.log"))
    logs += [
        path
        for path in (root / "logs" / "render").glob("*.log")
        if "failure" not in path.name and not path.name.startswith("render-smoke")
    ]
    failures = []
    for path in logs:
        matches = pattern.findall(path.read_text(errors="replace"))
        if matches:
            failures.append({"path": str(path), "matches": matches})
    return failures


def artifact_size(root: Path):
    return sum(
        path.stat().st_size
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink()
    )


def format_optional(value, digits=3):
    return "n/a" if value is None else f"{value:.{digits}f}"


def build_report(geometry, predictions):
    names = sorted(geometry)
    mean_geometry_distance = np.mean(
        [geometry[name]["symmetric_mean_distance_normalized"] for name in names]
    )
    mean_label_agreement = np.mean(
        [predictions[name]["symmetric_label_agreement"] for name in names]
    )
    median_part_iou = np.median(
        [predictions[name]["part_mIoU_unmatched_zero"] for name in names]
    )
    lines = [
        "# Hunyuan3D Examples：原面数与 10k 面对比",
        "",
        "## 实验设置",
        "",
        "- 对仓库内 8 个 GLB 做无 GT 评估，因此结论属于配对稳定性和定性分析。",
        "- 原始网格含 37,892-50,000 面，简化网格含 9,999-10,000 面。",
        "- 两分支均使用 `--num_points 51200 --up_dir=-Z --animation_frames 120 --eval` 和 strict connectivity。",
        "- 单分支视频为 720x720、120 帧、30 fps；对比视频左侧为 Original，右侧为 10k faces。",
        "- 推理固定了 PyTorch 种子但未固定 NumPy 种子，表面采样噪声是混杂因素，故本实验并非严格确定性的因果消融。",
        "",
        "## 总结",
        "",
        f"简化移除了 73.6%-80.0% 的面，同时将表面积变化控制在 0.30% 内。对称采样表面距离均值为原始包围盒对角线的 {mean_geometry_distance:.4f}；面积加权标签一致率均值为 {mean_label_agreement:.4f}；将未匹配 part 计为零后，宏平均 part mIoU 中位数为 {median_part_iou:.4f}。",
        "",
        "8 个样本中有 7 个保持了 part 数、运动 part 数、运动类别和层级；轴方向几乎不变。主要的 topology 敏感例外是 toilet，其 10k 网格产生了一个额外的微小平移 part。washingmachine 的面积加权一致率很高，但小 part 在宏平均中具有同等权重，导致 part mIoU 偏低。",
        "",
        "对这组样本而言，10k 预处理在视觉上可接受，并降低了 25,600 个均匀 decode 采样点无法覆盖每个面的风险。但它不会修正模型语义错误：foldingchair 在两分支中都欠关节化，washingmachine 则都呈现过度关节化。",
        "",
        "## 配对指标",
        "",
        "| 样本 | 面数（原始->10k） | Part 数 | Part mIoU* | 标签一致率 | 运动一致率* | 轴角差 | 轴线偏移 | 旋转范围 MAE | 层级 Jaccard |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in names:
        g = geometry[name]
        p = predictions[name]
        axis_angle = math.degrees(math.acos(np.clip(p["axis_abs_cosine"], -1.0, 1.0)))
        range_degrees = (
            None if p["revolute_range_endpoint_mae_radians"] is None
            else math.degrees(p["revolute_range_endpoint_mae_radians"])
        )
        lines.append(
            f"| {name} | {g['original_faces']:,}->{g['simplified_faces']:,} | "
            f"{p['original_parts']}->{p['simplified_parts']} | "
            f"{p['part_mIoU_unmatched_zero']:.3f} | "
            f"{p['symmetric_label_agreement']:.3f} | "
            f"{p['motion_class_agreement_unmatched_zero']:.3f} | "
            f"{axis_angle:.3f} deg | "
            f"{p['revolute_axis_line_distance_normalized']:.4f} | "
            f"{format_optional(range_degrees, 2)} deg | "
            f"{p['hierarchy_edge_jaccard']:.3f} |"
        )
    lines += [
        "",
        "`*` 未匹配 part 计为零。标签一致率按面积加权，part mIoU 按 part 等权；轴线偏移位于模型按最大边长归一化后的坐标系。",
        "",
        "## 逐样本定性观察",
        "",
    ]
    for name in names:
        lines.append(f"- **{name}:** {QUALITATIVE_NOTES[name]}")
    lines += [
        "",
        "## 结论与建议",
        "",
        "- 从欧氏几何和关节轴方向看，简化误差较小，因此 10k 网格是这组样本上合理的推理表示。",
        "- connected component 数仍可能明显变化（washingmachine：1,053->690；laptop：1,476->1,372）。strict connectivity 使 topology 变化的影响大于表面距离所显示的程度。",
        "- 原始网格面数高于 25,600 个均匀 decode 采样点；10k 网格则相反，因此更利于 strict 面标签传播时的逐面覆盖。",
        "- 若需严格消融，应显式固定 NumPy 种子并对两分支各重复多次；先将简化差异与同网格重复推理方差比较，再把小幅范围或边界变化归因于 decimation。",
        "- 这些样本没有 articulation GT，视觉合理性和分支一致性衡量的是稳定性而非正确性。foldingchair 与 washingmachine 表明，预测即使稳定也可能在语义上错误。",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()

    geometry_path = args.root / "analysis" / "geometry_comparison.json"
    prediction_path = args.root / "analysis" / "prediction_comparison.json"
    geometry = load_rows(geometry_path)
    predictions = load_rows(prediction_path)
    if set(geometry) != set(predictions) or set(geometry) != set(QUALITATIVE_NOTES):
        raise ValueError("Geometry, prediction, and qualitative sample sets differ")

    samples = {}
    validation_failures = []
    for name in sorted(geometry):
        branches = {}
        for variant in ("original", "10k"):
            prediction_dir = args.root / "predictions" / variant / name
            prediction = validate_prediction(prediction_dir)
            video_path = args.root / "renders" / variant / f"{name}.mp4"
            video = probe_video(video_path)
            branch_spec = (
                video["width"],
                video["height"],
                video["nb_read_frames"],
                video["r_frame_rate"],
                video["codec_name"],
                video["pix_fmt"],
            )
            if branch_spec != (720, 720, 120, "30/1", "h264", "yuv420p"):
                validation_failures.append(f"Unexpected {variant} video spec for {name}: {video}")
            branches[variant] = {"prediction": prediction, "video": str(video_path), "video_probe": video}

        comparison_path = args.root / "renders" / "comparison" / f"{name}.mp4"
        comparison = probe_video(comparison_path)
        comparison_spec = (
            comparison["width"],
            comparison["height"],
            comparison["nb_read_frames"],
            comparison["r_frame_rate"],
            comparison["codec_name"],
            comparison["pix_fmt"],
        )
        if comparison_spec != (1440, 720, 120, "30/1", "h264", "yuv420p"):
            validation_failures.append(f"Unexpected comparison video spec for {name}: {comparison}")
        contact_sheet = args.root / "analysis" / "contact-sheets" / f"{name}.png"
        if not contact_sheet.is_file() or contact_sheet.stat().st_size == 0:
            validation_failures.append(f"Missing contact sheet for {name}: {contact_sheet}")
        samples[name] = {
            "geometry": geometry[name],
            "prediction_comparison": predictions[name],
            "branches": branches,
            "comparison_video": str(comparison_path),
            "comparison_video_probe": comparison,
            "contact_sheet": str(contact_sheet),
            "qualitative_note": QUALITATIVE_NOTES[name],
        }

    log_failures = scan_success_logs(args.root)
    validation_failures.extend(f"Log anomaly: {item}" for item in log_failures)
    overview = args.root / "analysis" / "contact-sheet-overview.png"
    if not overview.is_file() or overview.stat().st_size == 0:
        validation_failures.append(f"Missing contact-sheet overview: {overview}")
    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment": {
            "samples": sorted(samples),
            "num_points": 51200,
            "up_dir": "-Z",
            "strict_connectivity": True,
            "animation_frames": 120,
            "render": {"width": 720, "height": 720, "fps": 30, "samples": 8},
            "comparison_render": {"width": 1440, "height": 720, "fps": 30},
            "checkpoint": "/data2/LiuShuqi/output/particulate/checkpoints/particulate/model.pt",
            "partfield_checkpoint": "/data2/LiuShuqi/output/particulate/checkpoints/partfield/model_objaverse.ckpt",
        },
        "validation": {
            "passed": not validation_failures,
            "failures": validation_failures,
            "expected_branch_videos": 16,
            "expected_comparison_videos": 8,
            "recovered_failures": [
                str(args.root / "logs" / "preprocess" / "cabinet.log"),
                *[
                    str(path)
                    for path in sorted(
                        (args.root / "logs" / "render").glob("*-display-failure.log")
                    )
                ],
            ],
        },
        "total_output_bytes_excluding_symlinks": artifact_size(args.root),
        "contact_sheet_overview": str(overview),
        "samples": samples,
    }
    manifest_path = args.root / "manifest.json"
    report_path = args.root / "analysis" / "REPORT.md"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    report_path.write_text(build_report(geometry, predictions))
    print(json.dumps({
        "manifest": str(manifest_path.resolve()),
        "report": str(report_path.resolve()),
        "samples": len(samples),
        "validation_passed": not validation_failures,
    }, sort_keys=True))
    if validation_failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
