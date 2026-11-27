"""Render the Serrano geometries with our own materials and Serrano's environment maps (run inside Blender).

  blender -b --factory-startup -P scripts/serrano_shape_render.py -- --out data/serrano_render

Purpose: find why the stereo-trained highlight networks over-predict on matte, fine-detailed Serrano shapes. If the
network map is also high on OUR matte renders of the same geometry, the geometry itself (out of the training
distribution) is the cause; if it stays low, the difference lies in Serrano's rendering (Mitsuba, measured BRDFs, tone
mapping, JPEG). Same cameras, passes and file format as stereo_dataset.py (left and right view), so stereo_targets.py
gives the true residual and specular share for these scenes as well.

Geometry and environment maps come from Serrano's `materials_illuminations_geometries.zip` (data/serrano/geometry,
not redistributed).
"""
import json
import os
import sys
import time

import bpy
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stereo_dataset as sd  # noqa: E402  (render_view)
import stereo_render as sr  # noqa: E402

GEOM = "data/serrano/geometry"
SHAPES = ["sphere", "bunny_vt", "happy_vrip_vt", "xyzrgb_dragon_vt"]
LIGHTS = ["1_1_cambridge_2k", "2_2_art_studio_2k", "3_2_chinese_garden_2k_shift_500"]
MATERIALS = {  # name: (kind, colour, roughness)
    "matte_grey": ("diffuse", (0.6, 0.6, 0.6), None),
    "matte_dark": ("diffuse", (0.08, 0.1, 0.15), None),
    "semi_blue": ("dielectric", (0.08, 0.1, 0.25), 0.25),
    "glossy_blue": ("dielectric", (0.08, 0.1, 0.25), 0.05),
}


def parse():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/serrano_render")
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--spp", type=int, default=256)
    p.add_argument("--radius", type=float, default=0.19, help="bounding-sphere radius (m); training used 0.15")
    p.add_argument("--device", default="GPU")
    p.add_argument("--shapes", default=",".join(SHAPES))
    p.add_argument("--materials", default=",".join(MATERIALS))
    p.add_argument("--lights", default=",".join(LIGHTS))
    return p.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])


def load_shape(name):
    sr.clear_objects()
    bpy.ops.wm.obj_import(filepath=os.path.abspath(os.path.join(GEOM, name + ".obj")))
    ob = bpy.context.selected_objects[0]
    bpy.context.view_layer.objects.active = ob
    bpy.ops.object.shade_smooth()
    sr.normalise(ob)
    ob.pass_index = 1
    return ob


def material(kind, col, rough):
    mat = bpy.data.materials.new("m")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    if kind == "diffuse":
        bsdf = nt.nodes.new("ShaderNodeBsdfDiffuse")
        bsdf.inputs["Color"].default_value = (*col, 1)
    else:
        bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
        bsdf.inputs["Base Color"].default_value = (*col, 1)
        bsdf.inputs["Roughness"].default_value = rough
        bsdf.inputs["IOR"].default_value = 1.5
    nt.links.new(bsdf.outputs[0], out.inputs["Surface"])
    return mat


def set_world_file(path):
    w = bpy.data.worlds.get("serrano") or bpy.data.worlds.new("serrano")
    bpy.context.scene.world = w
    w.use_nodes = True
    nt = w.node_tree
    nt.nodes.clear()
    env = nt.nodes.new("ShaderNodeTexEnvironment")
    env.image = bpy.data.images.load(os.path.abspath(path), check_existing=True)
    bg = nt.nodes.new("ShaderNodeBackground")
    out = nt.nodes.new("ShaderNodeOutputWorld")
    nt.links.new(env.outputs["Color"], bg.inputs["Color"])
    nt.links.new(bg.outputs[0], out.inputs["Surface"])


def main():
    a = parse()
    os.makedirs(a.out, exist_ok=True)
    tmp = os.path.abspath(os.path.join(a.out, "_tmp"))
    bpy.ops.wm.read_factory_settings(use_empty=True)
    a.texture = False
    sr.RADIUS = a.radius
    sr.setup_render(a, tmp)
    cams = sr.make_cameras()
    for shape in a.shapes.split(","):
        todo = [(m, l) for m in a.materials.split(",") for l in a.lights.split(",")
                if not os.path.exists(os.path.join(a.out, f"{shape}__{m}__{l}.npz"))]
        if not todo:
            continue
        t = time.time()
        ob = load_shape(shape)
        print(f"{shape}: imported in {time.time() - t:.0f}s, {len(ob.data.polygons)} faces", flush=True)
        for m, light in todo:
            t = time.time()
            for old in list(bpy.data.materials):
                bpy.data.materials.remove(old)
            ob.data.materials.clear()
            ob.data.materials.append(material(*MATERIALS[m]))
            set_world_file(os.path.join(GEOM, light + ".hdr"))
            L, R = sd.render_view(tmp, cams["L"]), sd.render_view(tmp, cams["R"])
            h = np.float16
            np.savez_compressed(os.path.join(a.out, f"{shape}__{m}__{light}.npz"),
                                L_rgb=L["rgb"].astype(h), R_rgb=R["rgb"].astype(h), L_glossy=L["glossy"].astype(h),
                                L_position=L["position"].astype(h), R_position=R["position"].astype(h),
                                L_mask=L["mask"], R_mask=R["mask"], L_view=L["view"], L_proj=L["proj"],
                                R_view=R["view"], R_proj=R["proj"],
                                params=json.dumps(dict(shape=shape, material=m, light=light)))
            print(f"{shape} {m} {light}: {time.time() - t:.1f}s", flush=True)


if __name__ == "__main__":
    main()
