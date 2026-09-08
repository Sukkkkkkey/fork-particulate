#!/usr/bin/env python3
"""Measure geometric changes introduced by simplifying Hunyuan GLBs."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree


def load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, process=False)
    if isinstance(loaded, trimesh.Scene):
        meshes = [
            geometry.copy()
            for geometry in loaded.dump(concatenate=False)
            if isinstance(geometry, trimesh.Trimesh)
        ]
        if not meshes:
            raise ValueError(f"No mesh geometry in {path}")
        loaded = trimesh.util.concatenate(meshes)
    if len(loaded.faces) == 0 or not np.isfinite(loaded.vertices).all():
        raise ValueError(f"Invalid mesh in {path}")
    return loaded


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def component_count(mesh: trimesh.Trimesh) -> int:
    components = trimesh.graph.connected_components(
        mesh.face_adjacency,
        nodes=np.arange(len(mesh.faces)),
        min_len=1,
    )
    return len(components)


def sample(mesh: trimesh.Trimesh, count: int, seed: int):
    np.random.seed(seed)
    points, faces = mesh.sample(count, return_index=True)
    normals = mesh.face_normals[faces]
    return points, normals


def texture_kind(mesh: trimesh.Trimesh) -> str:
    return type(mesh.visual).__name__


def compare(name: str, original_path: Path, simplified_path: Path, count: int):
    original = load_mesh(original_path)
    simplified = load_mesh(simplified_path)
    original_points, original_normals = sample(original, count, 20260908)
    simplified_points, simplified_normals = sample(simplified, count, 20260909)

    original_tree = cKDTree(original_points)
    simplified_tree = cKDTree(simplified_points)
    simplified_to_original, simplified_indices = original_tree.query(simplified_points)
    original_to_simplified, original_indices = simplified_tree.query(original_points)

    diagonal = float(np.linalg.norm(original.extents))
    if diagonal <= 0:
        raise ValueError(f"Degenerate original mesh: {original_path}")
    combined = np.concatenate([original_to_simplified, simplified_to_original])
    normal_consistency = 0.5 * (
        np.abs(np.sum(original_normals * simplified_normals[original_indices], axis=1)).mean()
        + np.abs(np.sum(simplified_normals * original_normals[simplified_indices], axis=1)).mean()
    )

    return {
        "name": name,
        "original_path": str(original_path.resolve()),
        "simplified_path": str(simplified_path.resolve()),
        "original_sha256": sha256(original_path),
        "simplified_sha256": sha256(simplified_path),
        "original_vertices": int(len(original.vertices)),
        "simplified_vertices": int(len(simplified.vertices)),
        "original_faces": int(len(original.faces)),
        "simplified_faces": int(len(simplified.faces)),
        "face_reduction_fraction": float(1.0 - len(simplified.faces) / len(original.faces)),
        "original_components": component_count(original),
        "simplified_components": component_count(simplified),
        "original_watertight": bool(original.is_watertight),
        "simplified_watertight": bool(simplified.is_watertight),
        "original_visual": texture_kind(original),
        "simplified_visual": texture_kind(simplified),
        "surface_area_ratio": float(simplified.area / original.area),
        "bbox_diagonal": diagonal,
        "bbox_center_shift_normalized": float(
            np.linalg.norm(simplified.bounding_box.centroid - original.bounding_box.centroid) / diagonal
        ),
        "bbox_extent_relative_l2": float(
            np.linalg.norm(simplified.extents - original.extents) / diagonal
        ),
        "symmetric_mean_distance_normalized": float(combined.mean() / diagonal),
        "symmetric_rms_distance_normalized": float(np.sqrt(np.mean(combined ** 2)) / diagonal),
        "symmetric_p95_distance_normalized": float(np.percentile(combined, 95) / diagonal),
        "sampled_hausdorff_distance_normalized": float(combined.max() / diagonal),
        "normal_consistency": float(normal_consistency),
    }


def write_csv(path: Path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-dir", type=Path, required=True)
    parser.add_argument("--simplified-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-points", type=int, default=100000)
    args = parser.parse_args()

    originals = sorted(args.original_dir.glob("*.glb"))
    if not originals:
        raise ValueError("No original GLBs found")
    rows = []
    for original in originals:
        simplified = args.simplified_dir / original.name
        if not simplified.is_file():
            raise FileNotFoundError(simplified)
        rows.append(compare(original.stem, original, simplified, args.num_points))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "geometry_comparison.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n"
    )
    write_csv(args.output_dir / "geometry_comparison.csv", rows)
    print(json.dumps({"meshes": len(rows), "output_dir": str(args.output_dir.resolve())}))


if __name__ == "__main__":
    main()
