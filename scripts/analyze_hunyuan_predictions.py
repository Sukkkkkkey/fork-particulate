#!/usr/bin/env python3
"""Compare articulation predictions made from original and simplified meshes."""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from particulate.data_utils import load_obj_raw_preserve


def load_prediction(root: Path, name: str):
    pred_dir = root / name / "eval"
    values = dict(np.load(pred_dir / "pred.npz", allow_pickle=False))
    vertices, faces = load_obj_raw_preserve(pred_dir / "pred.obj")
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    if len(values["face_part_ids"]) != len(mesh.faces):
        raise ValueError(f"Face labels do not match mesh for {root}/{name}")

    rotation = np.diag([1.0, -1.0, -1.0])
    transformed = mesh.vertices @ rotation.T
    lower, upper = transformed.min(axis=0), transformed.max(axis=0)
    scale = float((upper - lower).max())
    if scale <= 0:
        raise ValueError(f"Degenerate prediction mesh for {root}/{name}")
    mesh.vertices = (transformed - (lower + upper) * 0.5) / scale
    return mesh, values


def sample_labels(mesh, values, count, seed):
    np.random.seed(seed)
    points, face_indices = mesh.sample(count, return_index=True)
    return points, values["face_part_ids"][face_indices].astype(np.int64)


def motion_class(values, part_id):
    revolute = bool(values["is_part_revolute"][part_id])
    prismatic = bool(values["is_part_prismatic"][part_id])
    if revolute and prismatic:
        return "both"
    if revolute:
        return "revolute"
    if prismatic:
        return "prismatic"
    return "fixed"


def axis(values, part_id, kind):
    if kind == "revolute":
        vector = values["revolute_plucker"][part_id, :3]
    elif kind == "prismatic":
        vector = values["prismatic_axis"][part_id]
    else:
        return None
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 1e-12 else None


def range_values(values, part_id, kind):
    if kind == "revolute":
        return values["revolute_range"][part_id]
    if kind == "prismatic":
        return values["prismatic_range"][part_id]
    return None


def revolute_axis_line_distance(original, simplified):
    original_direction = axis(original, 0, "revolute")
    simplified_direction = axis(simplified, 0, "revolute")
    if original_direction is None or simplified_direction is None:
        return None

    original_moment = original["revolute_plucker"][0, 3:]
    simplified_moment = simplified["revolute_plucker"][0, 3:]
    original_point = np.cross(original_moment, original_direction)
    simplified_point = np.cross(simplified_moment, simplified_direction)
    offset = simplified_point - original_point
    cross = np.cross(original_direction, simplified_direction)
    cross_norm = np.linalg.norm(cross)
    if cross_norm > 1e-8:
        return float(abs(np.dot(offset, cross)) / cross_norm)
    return float(np.linalg.norm(np.cross(offset, original_direction)))


def compare(name, original_root, simplified_root, count):
    original_mesh, original = load_prediction(original_root, name)
    simplified_mesh, simplified = load_prediction(simplified_root, name)
    original_points, original_labels = sample_labels(original_mesh, original, count, 20260908)
    simplified_points, simplified_labels = sample_labels(simplified_mesh, simplified, count, 20260909)

    simplified_tree = cKDTree(simplified_points)
    original_tree = cKDTree(original_points)
    nearest_simplified = simplified_labels[simplified_tree.query(original_points)[1]]
    nearest_original = original_labels[original_tree.query(simplified_points)[1]]

    original_count = len(original["is_part_revolute"])
    simplified_count = len(simplified["is_part_revolute"])
    confusion = np.zeros((original_count, simplified_count), dtype=np.int64)
    np.add.at(confusion, (original_labels, nearest_simplified), 1)
    np.add.at(confusion, (nearest_original, simplified_labels), 1)
    original_indices, simplified_indices = linear_sum_assignment(-confusion)
    pairs = list(zip(original_indices.tolist(), simplified_indices.tolist()))

    ious = []
    motion_matches = []
    axis_alignments = []
    axis_line_distances = []
    revolute_range_errors = []
    prismatic_range_errors = []
    for original_id, simplified_id in pairs:
        intersection = confusion[original_id, simplified_id]
        union = confusion[original_id].sum() + confusion[:, simplified_id].sum() - intersection
        ious.append(float(intersection / union) if union else 0.0)
        original_kind = motion_class(original, original_id)
        simplified_kind = motion_class(simplified, simplified_id)
        motion_matches.append(original_kind == simplified_kind)
        if original_kind == simplified_kind and original_kind in {"revolute", "prismatic"}:
            original_axis = axis(original, original_id, original_kind)
            simplified_axis = axis(simplified, simplified_id, simplified_kind)
            if original_axis is not None and simplified_axis is not None:
                axis_alignments.append(float(abs(np.dot(original_axis, simplified_axis))))
            if original_kind == "revolute":
                original_part = {
                    "revolute_plucker": original["revolute_plucker"][[original_id]],
                }
                simplified_part = {
                    "revolute_plucker": simplified["revolute_plucker"][[simplified_id]],
                }
                distance = revolute_axis_line_distance(original_part, simplified_part)
                if distance is not None:
                    axis_line_distances.append(distance)
            original_range = range_values(original, original_id, original_kind)
            simplified_range = range_values(simplified, simplified_id, simplified_kind)
            range_error = float(np.mean(np.abs(original_range - simplified_range)))
            if original_kind == "revolute":
                revolute_range_errors.append(range_error)
            else:
                prismatic_range_errors.append(range_error)

    simplified_to_original = {simplified_id: original_id for original_id, simplified_id in pairs}
    original_edges = {
        (f"o{int(parent)}", f"o{int(child)}")
        for parent, child in np.asarray(original["motion_hierarchy"]).reshape(-1, 2)
    }
    simplified_edges = {
        (
            f"o{simplified_to_original[int(parent)]}"
            if int(parent) in simplified_to_original else f"s{int(parent)}",
            f"o{simplified_to_original[int(child)]}"
            if int(child) in simplified_to_original else f"s{int(child)}",
        )
        for parent, child in np.asarray(simplified["motion_hierarchy"]).reshape(-1, 2)
    }
    edge_union = original_edges | simplified_edges
    denominator = max(original_count, simplified_count)
    unmatched_original = set(range(original_count)) - set(original_indices)
    unmatched_simplified = set(range(simplified_count)) - set(simplified_indices)
    unmatched_original_fraction = float(
        np.isin(original_labels, list(unmatched_original)).mean()
    ) if unmatched_original else 0.0
    unmatched_simplified_fraction = float(
        np.isin(simplified_labels, list(unmatched_simplified)).mean()
    ) if unmatched_simplified else 0.0

    return {
        "name": name,
        "original_parts": original_count,
        "simplified_parts": simplified_count,
        "part_count_delta": simplified_count - original_count,
        "original_moving_parts": int(
            np.count_nonzero(original["is_part_revolute"] | original["is_part_prismatic"])
        ),
        "simplified_moving_parts": int(
            np.count_nonzero(simplified["is_part_revolute"] | simplified["is_part_prismatic"])
        ),
        "matched_parts": len(pairs),
        "matched_part_mIoU": float(np.mean(ious)) if ious else 0.0,
        "part_mIoU_unmatched_zero": float(sum(ious) / denominator),
        "symmetric_label_agreement": float(
            sum(confusion[original_id, simplified_id] for original_id, simplified_id in pairs)
            / confusion.sum()
        ),
        "unmatched_original_area_fraction": unmatched_original_fraction,
        "unmatched_simplified_area_fraction": unmatched_simplified_fraction,
        "motion_class_agreement": float(np.mean(motion_matches)) if motion_matches else 0.0,
        "motion_class_agreement_unmatched_zero": float(sum(motion_matches) / denominator),
        "axis_abs_cosine": float(np.mean(axis_alignments)) if axis_alignments else None,
        "revolute_axis_line_distance_normalized": float(np.mean(axis_line_distances))
        if axis_line_distances else None,
        "revolute_range_endpoint_mae_radians": float(np.mean(revolute_range_errors))
        if revolute_range_errors else None,
        "prismatic_range_endpoint_mae_normalized": float(np.mean(prismatic_range_errors))
        if prismatic_range_errors else None,
        "hierarchy_edge_jaccard": float(len(original_edges & simplified_edges) / len(edge_union))
        if edge_union else 1.0,
    }


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--simplified-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-points", type=int, default=100000)
    args = parser.parse_args()

    names = sorted(path.name for path in args.original_root.iterdir() if path.is_dir())
    if not names:
        raise ValueError("No original prediction directories found")
    rows = [
        compare(name, args.original_root, args.simplified_root, args.num_points)
        for name in names
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "prediction_comparison.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n"
    )
    write_csv(args.output_dir / "prediction_comparison.csv", rows)
    print(json.dumps({"meshes": len(rows), "output_dir": str(args.output_dir.resolve())}))


if __name__ == "__main__":
    main()
