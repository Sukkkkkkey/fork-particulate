#!/usr/bin/env python3
"""Compare paired Particulate evaluation runs and the published tables."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np


METRICS = {
    "rest_gIoU": ("rest_per_part_avg_giou", True),
    "rest_PC": ("rest_per_part_avg_chamfer", False),
    "rest_mIoU": ("rest_per_part_avg_mIoU", True),
    "full_gIoU": ("fully_per_part_articulated_avg_giou", True),
    "full_PC": ("fully_per_part_articulated_avg_chamfer", False),
    "full_OC": ("fully_articulated_overall_chamfer_distances", False),
}

# Table 2 and Table 3 in the paper. The dagger rows use mesh connectivity.
PAPER_RESULTS = {
    "Lightwheel": {
        "strict": {
            "rest_gIoU": 0.332,
            "rest_PC": 0.168,
            "rest_mIoU": 0.576,
            "full_gIoU": 0.305,
            "full_PC": 0.208,
            "full_OC": 0.009,
        },
        "no_strict": {
            "rest_gIoU": 0.183,
            "rest_PC": 0.163,
            "rest_mIoU": 0.430,
            "full_gIoU": 0.165,
            "full_PC": 0.200,
            "full_OC": 0.008,
        },
    },
    "PartNet-Mobility": {
        "strict": {
            "rest_gIoU": 0.880,
            "rest_PC": 0.003,
            "rest_mIoU": 0.884,
            "full_gIoU": 0.843,
            "full_PC": 0.022,
            "full_OC": 0.003,
        },
        "no_strict": {
            "rest_gIoU": 0.879,
            "rest_PC": 0.003,
            "rest_mIoU": 0.883,
            "full_gIoU": 0.842,
            "full_PC": 0.024,
            "full_OC": 0.003,
        },
    },
}


def parse_dataset(value: Sequence[str]) -> Tuple[str, Path, Path, Path]:
    name, strict_root, no_strict_root, gt_dir = value
    if name not in PAPER_RESULTS:
        raise ValueError(f"No paper values configured for dataset {name!r}")
    return name, Path(strict_root), Path(no_strict_root), Path(gt_dir)


def load_split(root: Path) -> List[Tuple[str, str]]:
    split = json.loads((root / "data_split.json").read_text())
    entries = []
    seen = set()
    for value in split.get("test", []):
        if not isinstance(value, str) or value.count("/") != 1:
            raise ValueError(f"Invalid test split entry: {value!r}")
        category, sample_id = value.split("/", 1)
        if sample_id in seen:
            raise ValueError(f"Duplicate sample ID: {sample_id}")
        seen.add(sample_id)
        entries.append((category, sample_id))
    if not entries:
        raise ValueError(f"Empty test split in {root}")
    return entries


def load_metric_file(path: Path) -> Dict[str, float]:
    values = json.loads(path.read_text())
    result = {}
    for label, (key, _) in METRICS.items():
        value = float(values[key])
        if not math.isfinite(value):
            raise ValueError(f"Non-finite {key} in {path}")
        result[label] = value
    return result


def prediction_part_count(root: Path, sample_id: str) -> int:
    path = root / "predictions" / sample_id / "eval" / "pred.npz"
    with np.load(path, allow_pickle=False) as values:
        flags = values["is_part_revolute"]
        if flags.ndim != 1 or len(flags) == 0:
            raise ValueError(f"Invalid part flags in {path}")
        return int(len(flags))


def prediction_face_count(root: Path, sample_id: str) -> int:
    path = root / "predictions" / sample_id / "eval" / "pred.obj"
    with path.open("rb") as handle:
        count = sum(line.startswith(b"f ") for line in handle)
    if count == 0:
        raise ValueError(f"No faces in {path}")
    return count


def gt_part_count(gt_dir: Path, sample_id: str) -> int:
    path = gt_dir / f"{sample_id}.npz"
    with np.load(path, allow_pickle=False) as values:
        flags = values["is_part_revolute"]
        if flags.ndim != 1 or len(flags) == 0:
            raise ValueError(f"Invalid part flags in {path}")
        return int(len(flags))


def load_num_points(root: Path) -> int:
    manifest = json.loads((root / "run_manifest.json").read_text())
    return int(manifest["parameters"]["num_points"])


def mean(values: Sequence[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def bootstrap_ci(values: Sequence[float], rng: np.random.Generator) -> List[float]:
    data = np.asarray(values, dtype=np.float64)
    if len(data) == 1:
        return [float(data[0]), float(data[0])]
    sample_indices = rng.integers(0, len(data), size=(10000, len(data)))
    sample_means = data[sample_indices].mean(axis=1)
    low, high = np.percentile(sample_means, [2.5, 97.5])
    return [float(low), float(high)]


def part_count_summary(rows: Sequence[Mapping[str, Any]], key: str) -> Dict[str, float]:
    predicted = np.asarray([row[key] for row in rows], dtype=np.int64)
    target = np.asarray([row["gt_parts"] for row in rows], dtype=np.int64)
    error = predicted - target
    return {
        "mean": float(predicted.mean()),
        "gt_mean": float(target.mean()),
        "mae": float(np.abs(error).mean()),
        "exact_fraction": float(np.mean(error == 0)),
        "under_fraction": float(np.mean(error < 0)),
        "over_fraction": float(np.mean(error > 0)),
    }


def summarize_rows(
    rows: Sequence[Mapping[str, Any]], rng: np.random.Generator
) -> Dict[str, Any]:
    summary = {}
    for metric, (_, higher_is_better) in METRICS.items():
        strict_values = [float(row[f"strict_{metric}"]) for row in rows]
        no_strict_values = [float(row[f"no_strict_{metric}"]) for row in rows]
        raw_deltas = [s - n for s, n in zip(strict_values, no_strict_values)]
        improvements = raw_deltas if higher_is_better else [-value for value in raw_deltas]
        tolerance = 1e-12
        sorted_by_magnitude = sorted(improvements, key=abs, reverse=True)
        trimmed = sorted_by_magnitude[5:] if len(sorted_by_magnitude) > 5 else improvements
        summary[metric] = {
            "higher_is_better": higher_is_better,
            "strict_mean": mean(strict_values),
            "no_strict_mean": mean(no_strict_values),
            "strict_minus_no_strict": mean(raw_deltas),
            "strict_improvement": mean(improvements),
            "strict_improvement_median": float(np.median(improvements)),
            "strict_improvement_without_top5_abs": mean(trimmed),
            "strict_improvement_ci95": bootstrap_ci(improvements, rng),
            "strict_better": sum(value > tolerance for value in improvements),
            "tie": sum(abs(value) <= tolerance for value in improvements),
            "no_strict_better": sum(value < -tolerance for value in improvements),
        }
    return summary


def analyze_dataset(
    name: str,
    strict_root: Path,
    no_strict_root: Path,
    gt_dir: Path,
    rng: np.random.Generator,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    strict_entries = load_split(strict_root)
    no_strict_entries = load_split(no_strict_root)
    if strict_entries != no_strict_entries:
        raise ValueError(f"Split mismatch for {name}")

    rows = []
    for category, sample_id in strict_entries:
        strict = load_metric_file(strict_root / "metrics" / f"{sample_id}.json")
        no_strict = load_metric_file(no_strict_root / "metrics" / f"{sample_id}.json")
        strict_faces = prediction_face_count(strict_root, sample_id)
        no_strict_faces = prediction_face_count(no_strict_root, sample_id)
        if strict_faces != no_strict_faces:
            raise ValueError(f"Face-count mismatch for {name}/{sample_id}")
        row: Dict[str, Any] = {
            "dataset": name,
            "category": category,
            "sample_id": sample_id,
            "faces": strict_faces,
            "strict_parts": prediction_part_count(strict_root, sample_id),
            "no_strict_parts": prediction_part_count(no_strict_root, sample_id),
            "gt_parts": gt_part_count(gt_dir, sample_id),
        }
        for metric, (_, higher_is_better) in METRICS.items():
            row[f"strict_{metric}"] = strict[metric]
            row[f"no_strict_{metric}"] = no_strict[metric]
            row[f"strict_minus_no_strict_{metric}"] = strict[metric] - no_strict[metric]
            row[f"strict_improvement_{metric}"] = (
                strict[metric] - no_strict[metric]
                if higher_is_better
                else no_strict[metric] - strict[metric]
            )
        rows.append(row)

    input_points = {
        "strict": load_num_points(strict_root),
        "no_strict": load_num_points(no_strict_root),
    }
    if input_points["strict"] != input_points["no_strict"]:
        raise ValueError(f"Input-point mismatch for paired runs of {name}")

    paired = summarize_rows(rows, rng)
    face_counts = np.asarray([row["faces"] for row in rows], dtype=np.int64)
    local_uniform_points = input_points["strict"] // 2
    overall = {}
    for metric, (_, higher_is_better) in METRICS.items():
        paper_strict = PAPER_RESULTS[name]["strict"][metric]
        paper_no_strict = PAPER_RESULTS[name]["no_strict"][metric]
        paper_raw_delta = paper_strict - paper_no_strict
        paper_improvement = paper_raw_delta if higher_is_better else -paper_raw_delta
        overall[metric] = {
            **paired[metric],
            "paper_strict": paper_strict,
            "paper_no_strict": paper_no_strict,
            "strict_minus_paper_strict": paired[metric]["strict_mean"] - paper_strict,
            "no_strict_minus_paper_no_strict": (
                paired[metric]["no_strict_mean"] - paper_no_strict
            ),
            "paper_strict_improvement": paper_improvement,
        }
        improvements = np.asarray(
            [row[f"strict_improvement_{metric}"] for row in rows], dtype=np.float64
        )
        overall[metric]["log_face_count_correlation"] = float(
            np.corrcoef(np.log1p(face_counts), improvements)[0, 1]
        )
        overall[metric]["largest_absolute_differences"] = [
            {
                "sample_id": str(rows[index]["sample_id"]),
                "category": str(rows[index]["category"]),
                "strict_improvement": float(improvements[index]),
            }
            for index in np.argsort(np.abs(improvements))[::-1][:5]
        ]

    categories = []
    by_category: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_category[str(row["category"])].append(row)
    for category, category_rows in sorted(by_category.items()):
        category_summary = summarize_rows(category_rows, rng)
        record: Dict[str, Any] = {
            "dataset": name,
            "category": category,
            "count": len(category_rows),
        }
        for metric in METRICS:
            values = category_summary[metric]
            record[f"strict_{metric}"] = values["strict_mean"]
            record[f"no_strict_{metric}"] = values["no_strict_mean"]
            record[f"strict_improvement_{metric}"] = values["strict_improvement"]
        categories.append(record)

    summary = {
        "dataset": name,
        "count": len(rows),
        "paths": {
            "strict_root": str(strict_root.resolve()),
            "no_strict_root": str(no_strict_root.resolve()),
            "gt_dir": str(gt_dir.resolve()),
        },
        "input_points": input_points,
        "face_counts": {
            "min": int(face_counts.min()),
            "median": float(np.median(face_counts)),
            "mean": float(face_counts.mean()),
            "p90": float(np.percentile(face_counts, 90)),
            "max": int(face_counts.max()),
            "local_uniform_points": local_uniform_points,
            "above_local_uniform_points": int(np.sum(face_counts > local_uniform_points)),
            "above_local_uniform_fraction": float(np.mean(face_counts > local_uniform_points)),
        },
        "overall": overall,
        "part_counts": {
            "strict": part_count_summary(rows, "strict_parts"),
            "no_strict": part_count_summary(rows, "no_strict_parts"),
        },
    }
    return summary, rows, categories


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_value(value: float) -> str:
    return f"{value:.6f}"


def write_report(path: Path, summaries: Sequence[Mapping[str, Any]], paper_points: int) -> None:
    lines = [
        "# Particulate connectivity comparison",
        "",
        "Positive `strict improvement` always means that mesh connectivity helped; "
        "the sign is reversed for lower-is-better metrics.",
        "",
        f"Paper rows are labeled as the default {paper_points:,}-point inference. "
        "Local-versus-paper gaps are not a controlled point-count ablation.",
    ]
    for summary in summaries:
        name = str(summary["dataset"])
        points = summary["input_points"]["strict"]
        faces = summary["face_counts"]
        lines.extend([
            "",
            f"## {name}",
            "",
            f"Paired local runs: {summary['count']} samples, {points:,} inference points.",
            f"The 50/50 sampler provides {faces['local_uniform_points']:,} uniform points; "
            f"{faces['above_local_uniform_points']}/{summary['count']} meshes "
            f"({faces['above_local_uniform_fraction']:.1%}) have more faces than that. "
            f"Face-count median/p90/max: {faces['median']:,.0f}/{faces['p90']:,.0f}/{faces['max']:,}.",
            "",
            "| Metric | Local strict | Paper strict | Local no-strict | Paper no-strict | Local strict improvement | 95% bootstrap CI | Paper strict improvement |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for metric in METRICS:
            values = summary["overall"][metric]
            ci = values["strict_improvement_ci95"]
            lines.append(
                f"| {metric} | {format_value(values['strict_mean'])} | "
                f"{format_value(values['paper_strict'])} | "
                f"{format_value(values['no_strict_mean'])} | "
                f"{format_value(values['paper_no_strict'])} | "
                f"{format_value(values['strict_improvement'])} | "
                f"[{format_value(ci[0])}, {format_value(ci[1])}] | "
                f"{format_value(values['paper_strict_improvement'])} |"
            )

        lines.extend([
            "",
            "| Metric | Mean strict improvement | Median | Mean without top-5 absolute differences | log(face count) correlation |",
            "|---|---:|---:|---:|---:|",
        ])
        for metric in METRICS:
            values = summary["overall"][metric]
            lines.append(
                f"| {metric} | {format_value(values['strict_improvement'])} | "
                f"{format_value(values['strict_improvement_median'])} | "
                f"{format_value(values['strict_improvement_without_top5_abs'])} | "
                f"{values['log_face_count_correlation']:.3f} |"
            )

        strict_counts = summary["part_counts"]["strict"]
        no_strict_counts = summary["part_counts"]["no_strict"]
        lines.extend([
            "",
            "| Postprocess | Predicted parts | GT parts | Part-count MAE | Exact | Under | Over |",
            "|---|---:|---:|---:|---:|---:|---:|",
            f"| strict | {strict_counts['mean']:.3f} | {strict_counts['gt_mean']:.3f} | "
            f"{strict_counts['mae']:.3f} | {strict_counts['exact_fraction']:.1%} | "
            f"{strict_counts['under_fraction']:.1%} | {strict_counts['over_fraction']:.1%} |",
            f"| no-strict | {no_strict_counts['mean']:.3f} | {no_strict_counts['gt_mean']:.3f} | "
            f"{no_strict_counts['mae']:.3f} | {no_strict_counts['exact_fraction']:.1%} | "
            f"{no_strict_counts['under_fraction']:.1%} | {no_strict_counts['over_fraction']:.1%} |",
            "",
            "| Metric | Strict better | Tie | No-strict better |",
            "|---|---:|---:|---:|",
        ])
        for metric in METRICS:
            values = summary["overall"][metric]
            lines.append(
                f"| {metric} | {values['strict_better']} | {values['tie']} | "
                f"{values['no_strict_better']} |"
            )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare strict and no-strict Particulate evaluations"
    )
    parser.add_argument(
        "--dataset",
        action="append",
        nargs=4,
        metavar=("NAME", "STRICT_ROOT", "NO_STRICT_ROOT", "GT_DIR"),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--paper-input-points", type=int, default=102400)
    args = parser.parse_args()

    rng = np.random.default_rng(20260907)
    summaries = []
    per_asset_rows = []
    category_rows = []
    for raw_dataset in args.dataset:
        name, strict_root, no_strict_root, gt_dir = parse_dataset(raw_dataset)
        summary, rows, categories = analyze_dataset(
            name, strict_root, no_strict_root, gt_dir, rng
        )
        summaries.append(summary)
        per_asset_rows.extend(rows)
        category_rows.extend(categories)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "paper_input_points_assumption": args.paper_input_points,
        "datasets": summaries,
    }
    (args.output_dir / "comparison.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    write_csv(args.output_dir / "per_asset_comparison.csv", per_asset_rows)
    write_csv(args.output_dir / "category_comparison.csv", category_rows)
    write_report(args.output_dir / "report.md", summaries, args.paper_input_points)
    print(json.dumps({summary["dataset"]: summary["count"] for summary in summaries}, indent=2))


if __name__ == "__main__":
    main()
