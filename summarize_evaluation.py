#!/usr/bin/env python3
import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


OVERALL_FILENAMES = {
    "OVERALL_EVAL_RESULTS.json",
    "OVERALL_EVAL_RESULTS_NOPUNISH.json",
}


def load_split(split_file: Path, split_name: str) -> List[Tuple[str, str]]:
    split_data = json.loads(split_file.read_text())
    if split_name not in split_data or not isinstance(split_data[split_name], list):
        raise ValueError(f"Split '{split_name}' is missing or is not a list")

    entries = []
    seen = set()
    for item in split_data[split_name]:
        if not isinstance(item, str) or "/" not in item:
            raise ValueError(f"Invalid split entry: {item!r}")
        category, sample_id = item.split("/", 1)
        if not category or not sample_id or "/" in sample_id:
            raise ValueError(f"Invalid split entry: {item!r}")
        if sample_id in seen:
            raise ValueError(f"Duplicate sample id in split: {sample_id}")
        seen.add(sample_id)
        entries.append((category, sample_id))
    return entries


def flatten_metrics(metrics: Mapping[str, Any]) -> Dict[str, float]:
    flattened: Dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, bool):
            flattened[key] = float(value)
        elif isinstance(value, (int, float)):
            flattened[key] = float(value)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                if isinstance(item, bool):
                    flattened[f"{key}_state_{index}"] = float(item)
                elif isinstance(item, (int, float)):
                    flattened[f"{key}_state_{index}"] = float(item)
                else:
                    raise ValueError(f"Metric '{key}' contains a non-numeric list item")
        else:
            raise ValueError(f"Metric '{key}' has unsupported type {type(value).__name__}")

    non_finite = [key for key, value in flattened.items() if not math.isfinite(value)]
    if non_finite:
        raise ValueError(f"Non-finite metrics: {non_finite}")
    return flattened


def metric_statistics(values: Sequence[float]) -> Dict[str, float]:
    return {
        "mean": mean(values),
        "std": pstdev(values),
        "median": median(values),
        "min": min(values),
        "max": max(values),
    }


def aggregate_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, float]]:
    metric_values: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        for key, value in row.items():
            if key not in {"category", "sample_id"}:
                metric_values[key].append(float(value))
    return {
        key: metric_statistics(values)
        for key, values in sorted(metric_values.items())
        if values
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def load_author_overall(metrics_dir: Path) -> Dict[str, Any]:
    overall = {}
    for filename in sorted(OVERALL_FILENAMES):
        path = metrics_dir / filename
        if path.is_file():
            overall[path.stem] = json.loads(path.read_text())
    return overall


def summarize(
    metrics_dir: Path,
    split_file: Path,
    output_dir: Path,
    split_name: str,
    allow_incomplete: bool,
) -> Dict[str, Any]:
    entries = load_split(split_file, split_name)
    expected_ids = {sample_id for _, sample_id in entries}
    discovered_ids = {
        path.stem
        for path in metrics_dir.glob("*.json")
        if path.name not in OVERALL_FILENAMES
    }

    rows = []
    missing = []
    invalid = {}
    for category, sample_id in entries:
        metric_path = metrics_dir / f"{sample_id}.json"
        if not metric_path.is_file():
            missing.append(sample_id)
            continue
        try:
            metrics = json.loads(metric_path.read_text())
            flattened = flatten_metrics(metrics)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            invalid[sample_id] = str(exc)
            continue
        rows.append({"category": category, "sample_id": sample_id, **flattened})

    unexpected = sorted(discovered_ids - expected_ids)
    metric_names = sorted(
        {key for row in rows for key in row if key not in {"category", "sample_id"}}
    )
    per_asset_rows = [
        {"category": row["category"], "sample_id": row["sample_id"], **{key: row.get(key, "") for key in metric_names}}
        for row in rows
    ]
    write_csv(
        output_dir / "per_asset_metrics.csv",
        per_asset_rows,
        ["category", "sample_id", *metric_names],
    )

    category_rows = []
    expected_by_category: Dict[str, int] = defaultdict(int)
    evaluated_by_category: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for category, _ in entries:
        expected_by_category[category] += 1
    for row in rows:
        evaluated_by_category[str(row["category"])].append(row)

    category_summary = {}
    for category in sorted(expected_by_category):
        category_data = evaluated_by_category[category]
        aggregates = aggregate_rows(category_data) if category_data else {}
        means = {key: value["mean"] for key, value in aggregates.items()}
        category_rows.append(
            {
                "category": category,
                "expected": expected_by_category[category],
                "evaluated": len(category_data),
                **{key: means.get(key, "") for key in metric_names},
            }
        )
        category_summary[category] = {
            "expected": expected_by_category[category],
            "evaluated": len(category_data),
            "metrics": aggregates,
        }

    write_csv(
        output_dir / "category_metrics.csv",
        category_rows,
        ["category", "expected", "evaluated", *metric_names],
    )

    summary = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "metrics_dir": str(metrics_dir.resolve()),
        "split_file": str(split_file.resolve()),
        "split": split_name,
        "counts": {
            "expected": len(entries),
            "evaluated": len(rows),
            "missing": len(missing),
            "invalid": len(invalid),
            "unexpected": len(unexpected),
        },
        "missing_samples": missing,
        "invalid_samples": invalid,
        "unexpected_samples": unexpected,
        "overall": aggregate_rows(rows) if rows else {},
        "by_category": category_summary,
        "evaluate_py_overall": load_author_overall(metrics_dir),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )

    if not allow_incomplete and (missing or invalid or unexpected):
        raise RuntimeError(
            f"Incomplete evaluation: missing={len(missing)}, invalid={len(invalid)}, "
            f"unexpected={len(unexpected)}"
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize per-asset Particulate evaluation JSON files")
    parser.add_argument("--metrics_dir", type=Path, required=True)
    parser.add_argument("--split_file", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--allow_incomplete", action="store_true")
    args = parser.parse_args()

    summary = summarize(
        metrics_dir=args.metrics_dir,
        split_file=args.split_file,
        output_dir=args.output_dir,
        split_name=args.split,
        allow_incomplete=args.allow_incomplete,
    )
    print(json.dumps(summary["counts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
