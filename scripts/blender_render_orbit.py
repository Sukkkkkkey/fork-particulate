#!/usr/bin/env python3
"""Render an animated GLB while orbiting the camera around the object."""

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector


PALETTE = (
    (0.19, 0.55, 0.91, 1.0),
    (0.93, 0.35, 0.29, 1.0),
    (0.20, 0.72, 0.48, 1.0),
    (0.95, 0.68, 0.20, 1.0),
    (0.58, 0.39, 0.82, 1.0),
    (0.16, 0.70, 0.75, 1.0),
    (0.90, 0.42, 0.65, 1.0),
    (0.48, 0.65, 0.20, 1.0),
    (0.35, 0.40, 0.50, 1.0),
    (0.95, 0.52, 0.18, 1.0),
    (0.28, 0.76, 0.69, 1.0),
    (0.67, 0.48, 0.34, 1.0),
    (0.45, 0.60, 0.88, 1.0),
    (0.86, 0.61, 0.78, 1.0),
    (0.60, 0.60, 0.60, 1.0),
    (0.74, 0.75, 0.22, 1.0),
)


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--resolution", type=int, default=720)
    parser.add_argument("--samples", type=int, default=32)
    return parser.parse_args(argv)


def retime_actions(frame_start, frame_end):
    points = [
        point
        for action in bpy.data.actions
        for curve in action.fcurves
        for point in curve.keyframe_points
    ]
    if not points:
        raise ValueError("Imported GLB has no animation keyframes")
    source_start = min(point.co.x for point in points)
    source_end = max(point.co.x for point in points)
    if source_end <= source_start:
        raise ValueError("Imported animation has a zero-length time range")
    scale = (frame_end - frame_start) / (source_end - source_start)
    for point in points:
        point.co.x = frame_start + (point.co.x - source_start) * scale
        point.handle_left.x = frame_start + (point.handle_left.x - source_start) * scale
        point.handle_right.x = frame_start + (point.handle_right.x - source_start) * scale
        point.interpolation = "LINEAR"
    return source_start, source_end


def assign_part_materials(objects):
    for index, obj in enumerate(sorted(objects, key=lambda value: value.name)):
        material = bpy.data.materials.new(name="part_color_%02d" % index)
        material.use_nodes = True
        material.diffuse_color = PALETTE[index % len(PALETTE)]
        principled = material.node_tree.nodes.get("Principled BSDF")
        principled.inputs["Base Color"].default_value = material.diffuse_color
        principled.inputs["Roughness"].default_value = 0.62
        principled.inputs["Specular"].default_value = 0.28
        obj.data.materials.clear()
        obj.data.materials.append(material)


def animated_bounds(scene, objects, frames):
    lower = Vector((float("inf"),) * 3)
    upper = Vector((float("-inf"),) * 3)
    for frame in frames:
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        for obj in objects:
            for corner in obj.bound_box:
                point = obj.matrix_world @ Vector(corner)
                lower = Vector(min(lower[i], point[i]) for i in range(3))
                upper = Vector(max(upper[i], point[i]) for i in range(3))
    return lower, upper


def add_material(name, color, roughness):
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    material.diffuse_color = color
    principled = material.node_tree.nodes.get("Principled BSDF")
    principled.inputs["Base Color"].default_value = color
    principled.inputs["Roughness"].default_value = roughness
    return material


def add_area_light(name, location, energy, size):
    data = bpy.data.lights.new(name=name, type="AREA")
    data.energy = energy
    data.size = size
    obj = bpy.data.objects.new(name=name, object_data=data)
    bpy.context.collection.objects.link(obj)
    obj.location = location
    return obj


def point_at(obj, target):
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()


def main():
    args = parse_args()
    if args.frames < 2 or args.resolution <= 0 or args.fps <= 0:
        raise ValueError("frames, fps, and resolution must be positive")

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.fps = args.fps
    bpy.ops.import_scene.gltf(filepath=str(args.input.resolve()))

    for obj in list(scene.objects):
        if obj.type in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)
    objects = [obj for obj in scene.objects if obj.type == "MESH"]
    if not objects:
        raise ValueError("Animated GLB contains no mesh objects")
    assign_part_materials(objects)

    frame_start, frame_end = 1, args.frames
    source_range = retime_actions(frame_start, frame_end)
    scene.frame_start = frame_start
    scene.frame_end = frame_end
    sample_frames = sorted({frame_start, (frame_start + frame_end) // 2, frame_end})
    lower, upper = animated_bounds(scene, objects, sample_frames)
    center = (lower + upper) * 0.5
    extents = upper - lower
    radius = max(extents.length * 0.5, 0.25)

    floor_size = max(radius * 7.0, 3.0)
    bpy.ops.mesh.primitive_plane_add(size=floor_size, location=(center.x, center.y, lower.z - radius * 0.04))
    floor = bpy.context.object
    floor.data.materials.append(add_material("floor", (0.055, 0.065, 0.08, 1.0), 0.78))

    target = bpy.data.objects.new("orbit_target", None)
    bpy.context.collection.objects.link(target)
    target.location = center

    camera_data = bpy.data.cameras.new("orbit_camera")
    camera = bpy.data.objects.new("orbit_camera", camera_data)
    bpy.context.collection.objects.link(camera)
    scene.camera = camera
    camera_data.lens = 52
    camera_data.sensor_width = 36

    pivot = bpy.data.objects.new("orbit_pivot", None)
    bpy.context.collection.objects.link(pivot)
    pivot.location = center
    camera.parent = pivot
    camera.location = (radius * 2.55, 0.0, radius * 0.78)
    track = camera.constraints.new(type="TRACK_TO")
    track.target = target
    track.track_axis = "TRACK_NEGATIVE_Z"
    track.up_axis = "UP_Y"

    pivot.rotation_euler[2] = 0.0
    pivot.keyframe_insert(data_path="rotation_euler", index=2, frame=frame_start)
    pivot.rotation_euler[2] = 2.0 * math.pi
    pivot.keyframe_insert(data_path="rotation_euler", index=2, frame=frame_end)
    for curve in pivot.animation_data.action.fcurves:
        for point in curve.keyframe_points:
            point.interpolation = "LINEAR"

    key = add_area_light(
        "key_light",
        (center.x + radius * 2.8, center.y - radius * 2.4, center.z + radius * 3.5),
        950,
        radius * 3.0,
    )
    fill = add_area_light(
        "fill_light",
        (center.x - radius * 2.5, center.y - radius * 1.2, center.z + radius * 1.4),
        650,
        radius * 3.5,
    )
    rim = add_area_light(
        "rim_light",
        (center.x, center.y + radius * 3.0, center.z + radius * 2.5),
        850,
        radius * 2.5,
    )
    for light in (key, fill, rim):
        point_at(light, center)

    world = bpy.data.worlds.new("world") if not bpy.data.worlds else bpy.data.worlds[0]
    scene.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.018, 0.023, 0.032, 1.0)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.32

    scene.render.engine = "BLENDER_EEVEE"
    scene.eevee.use_gtao = True
    scene.eevee.gtao_distance = radius * 2.0
    scene.eevee.gtao_factor = 1.25
    scene.eevee.taa_render_samples = args.samples
    scene.render.resolution_x = args.resolution
    scene.render.resolution_y = args.resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "FFMPEG"
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    scene.render.ffmpeg.constant_rate_factor = "MEDIUM"
    scene.render.ffmpeg.ffmpeg_preset = "GOOD"
    scene.render.ffmpeg.gopsize = args.fps
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "Filmic"
    scene.view_settings.look = "Medium High Contrast"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(args.output.resolve())
    scene.frame_set(frame_start)
    bpy.ops.render.render(animation=True)
    if not args.output.is_file() or args.output.stat().st_size == 0:
        raise RuntimeError("Blender did not create the requested video")
    print(json.dumps({
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "frames": args.frames,
        "fps": args.fps,
        "resolution": args.resolution,
        "mesh_objects": len(objects),
        "source_animation_range": list(source_range),
        "bounds": [list(lower), list(upper)],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
