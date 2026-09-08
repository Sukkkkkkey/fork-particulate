#!/usr/bin/env python3
"""Simplify a GLB with Blender while preserving its materials and UVs."""

import argparse
import json
import os
import sys
from pathlib import Path

import bpy


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-faces", type=int, default=10000)
    return parser.parse_args(argv)


def mesh_objects():
    return [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]


def select_only(objects):
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]


def main():
    args = parse_args()
    if args.target_faces <= 0:
        raise ValueError("target-faces must be positive")

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(args.input.resolve()))
    objects = mesh_objects()
    if not objects:
        raise ValueError("Input GLB contains no mesh objects")

    select_only(objects)
    if len(objects) > 1:
        bpy.ops.object.join()
    obj = bpy.context.view_layer.objects.active
    obj.name = args.input.stem

    triangulate = obj.modifiers.new(name="Triangulate", type="TRIANGULATE")
    bpy.ops.object.modifier_apply(modifier=triangulate.name)
    original_faces = len(obj.data.polygons)

    if original_faces > args.target_faces:
        decimate = obj.modifiers.new(name="Decimate10k", type="DECIMATE")
        decimate.decimate_type = "COLLAPSE"
        decimate.ratio = args.target_faces / float(original_faces)
        if hasattr(decimate, "use_collapse_triangulate"):
            decimate.use_collapse_triangulate = True
        bpy.ops.object.modifier_apply(modifier=decimate.name)

    triangulate = obj.modifiers.new(name="TriangulateFinal", type="TRIANGULATE")
    bpy.ops.object.modifier_apply(modifier=triangulate.name)
    final_faces = len(obj.data.polygons)
    final_vertices = len(obj.data.vertices)

    for other in list(bpy.context.scene.objects):
        if other != obj:
            bpy.data.objects.remove(other, do_unlink=True)
    select_only([obj])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.stem + ".tmp.glb")
    bpy.ops.export_scene.gltf(
        filepath=str(temporary),
        export_format="GLB",
        export_materials=True,
    )
    os.replace(str(temporary), str(args.output))
    print(json.dumps({
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "original_faces": original_faces,
        "final_faces": final_faces,
        "final_vertices": final_vertices,
        "materials": len(obj.data.materials),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
