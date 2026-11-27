"""Random stereo scenes for training a monocular highlight network (run inside Blender).

  blender -b --factory-startup -P scripts/stereo_dataset.py -- --out data/stereo_ds --start 0 --count 100

Each scene: one random object (noise-displaced sphere with up to three detail scales, sometimes a plain sphere or the
Erato scan), one random Principled / diffuse material (roughness log-uniform 0.02-1, about a quarter pure diffuse,
some metals, some clear coats, half with an albedo pattern), one of Blender's eight bundled environment maps at a random
rotation, seen by a left and a right camera (6.5 cm baseline, 0.6 m). Saved per scene as scene_XXXXX.npz (float16):
L_rgb, R_rgb (linear), L_glossy (luminance of the glossy pass), L_position, R_position, L_mask, R_mask, camera
matrices, and the scene parameters as JSON. Targets (left-right residual with exact correspondence, specular fraction)
are computed afterwards by stereo_targets.py.
"""
import json
import math
import os
import sys
import time

import bpy
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stereo_render as sr  # noqa: E402  (helpers: render setup, cameras, worlds, shapes, EXR reading)

WORLD_NAMES = ["city", "courtyard", "forest", "interior", "night", "studio", "sunrise", "sunset"]


def parse():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/stereo_ds")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--count", type=int, default=10)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--spp", type=int, default=256)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--device", default="GPU")
    return p.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])


def random_shape(rng):
    sr.clear_objects()
    if rng.random() < 0.05:
        sr.import_erato()
        ob = bpy.context.selected_objects[0]
        bpy.ops.object.shade_smooth()
        spec = {"kind": "erato"}
    else:
        layers = []
        if rng.random() > 0.1:  # 10 % plain spheres
            for name, scale, strength, p in (("low", (0.06, 0.3), (0.0, 0.07), 0.9), ("mid", (0.025, 0.06), (0.0, 0.02), 0.5),
                                             ("fine", (0.006, 0.025), (0.0, 0.012), 0.5)):
                if rng.random() < p:
                    layers.append([name, round(float(rng.uniform(*scale)), 4), round(float(rng.uniform(*strength)), 4),
                                   int(rng.integers(0, 3))])
        fine = any(l[0] == "fine" for l in layers)
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=8 if fine else 7, radius=sr.RADIUS)
        ob = bpy.context.active_object
        for name, sc, st, depth in layers:
            sr.displace(ob, name, sc, st, depth)
        bpy.ops.object.shade_smooth()
        for m in list(ob.modifiers):
            try:
                bpy.ops.object.modifier_apply(modifier=m.name)
            except RuntimeError:  # strength rounded to 0: Blender disables the modifier, it has no effect anyway
                ob.modifiers.remove(m)
        spec = {"kind": "blob", "layers": layers}
    sr.normalise(ob)
    if spec["kind"] == "erato":  # keep the statue upright, turn it about the vertical axis
        ob.rotation_euler = (0.0, 0.0, float(rng.uniform(0, 2 * math.pi)))
    else:
        ob.rotation_euler = tuple(rng.uniform(0, 2 * math.pi, 3))
    s = rng.uniform(0.8, 1.13)
    ob.scale = (s, s, s)
    spec.update(rotation=[round(float(a), 3) for a in ob.rotation_euler], scale=round(s, 3))
    ob.pass_index = 1
    return ob, spec


def random_colour(rng, bright_bias=False):
    import colorsys
    v = rng.uniform(0.5, 0.95) if bright_bias else float(np.exp(rng.uniform(np.log(0.03), np.log(0.9))))
    s = rng.choice([rng.uniform(0, 0.15), rng.uniform(0.15, 0.8)], p=[0.6, 0.4])
    return colorsys.hsv_to_rgb(rng.random(), s, v)


def random_material(rng):
    mat = bpy.data.materials.new("m")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    u = rng.random()
    spec = {}
    if u < 0.25:  # pure diffuse, often bright: the case where the gloss map fires on shading
        bsdf = nt.nodes.new("ShaderNodeBsdfDiffuse")
        col = random_colour(rng, bright_bias=rng.random() < 0.6)
        bsdf.inputs["Color"].default_value = (*col, 1)
        spec.update(kind="diffuse", color=col)
        colour_input = bsdf.inputs["Color"]
    else:
        bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
        metal = rng.random() < 0.2
        col = random_colour(rng, bright_bias=metal or rng.random() < 0.3)
        rough = float(np.exp(rng.uniform(np.log(0.02), np.log(1.0))))
        level = float(rng.uniform(0.0, 1.0))
        bsdf.inputs["Base Color"].default_value = (*col, 1)
        bsdf.inputs["Metallic"].default_value = 1.0 if metal else 0.0
        bsdf.inputs["Roughness"].default_value = rough
        bsdf.inputs["IOR"].default_value = 1.5
        bsdf.inputs["Specular IOR Level"].default_value = level
        spec.update(kind="metal" if metal else "dielectric", color=col, roughness=round(rough, 4),
                    specular_ior_level=round(level, 3))
        if rng.random() < 0.2:
            cw, cr = float(rng.uniform(0.3, 1.0)), float(np.exp(rng.uniform(np.log(0.01), np.log(0.3))))
            bsdf.inputs["Coat Weight"].default_value = cw
            bsdf.inputs["Coat Roughness"].default_value = cr
            spec.update(coat_weight=round(cw, 3), coat_roughness=round(cr, 4))
        colour_input = bsdf.inputs["Base Color"]
    nt.links.new(bsdf.outputs[0], out.inputs["Surface"])
    if rng.random() < 0.5:  # albedo pattern fixed to the surface
        coord = nt.nodes.new("ShaderNodeTexCoord")
        noise = nt.nodes.new("ShaderNodeTexNoise")
        sc, contrast = float(rng.uniform(5, 80)), float(rng.uniform(0.1, 0.6))
        noise.inputs["Scale"].default_value, noise.inputs["Detail"].default_value = sc, float(rng.uniform(1, 8))
        mix = nt.nodes.new("ShaderNodeMix")
        mix.data_type = "RGBA"
        mix.blend_type = "MULTIPLY"
        mix.inputs["Factor"].default_value = 1.0
        mix.inputs[6].default_value = (*spec["color"], 1)  # A
        ramp = nt.nodes.new("ShaderNodeMapRange")
        ramp.inputs["To Min"].default_value, ramp.inputs["To Max"].default_value = 1 - contrast, 1 + contrast * 0.5
        nt.links.new(coord.outputs["Object"], noise.inputs["Vector"])
        nt.links.new(noise.outputs["Fac"], ramp.inputs["Value"])
        nt.links.new(ramp.outputs["Result"], mix.inputs[7])  # B
        nt.links.new(mix.outputs[2], colour_input)
        spec.update(texture_scale=round(sc, 2), texture_contrast=round(contrast, 3))
    spec["color"] = [round(c, 4) for c in spec["color"]]
    return mat, spec


def render_view(tmp, cam):
    bpy.context.scene.camera = cam
    bpy.ops.render.render(write_still=False)
    sc = bpy.context.scene
    f = {p: sr.read_exr(os.path.join(tmp, f"{p}_{sc.frame_current:04d}.exr")) for p in sr.PASSES}
    rgb = lambda k: f[k][..., :3]
    glossy = (rgb("GlossDir") + rgb("GlossInd")) * rgb("GlossCol")
    w = sc.render.resolution_x
    proj = cam.calc_matrix_camera(bpy.context.evaluated_depsgraph_get(), x=w, y=w)
    return dict(rgb=rgb("Image"), position=rgb("Position"), mask=f["IndexOB"][..., 0] > 0.5,
                glossy=glossy @ np.array([0.2126, 0.7152, 0.0722], np.float32),
                view=np.array(cam.matrix_world.inverted(), np.float32), proj=np.array(proj, np.float32))


def main():
    a = parse()
    os.makedirs(a.out, exist_ok=True)
    tmp = os.path.abspath(os.path.join(a.out, f"_tmp_{a.start}"))
    bpy.ops.wm.read_factory_settings(use_empty=True)
    a.texture = False
    sr.setup_render(a, tmp)
    cams = sr.make_cameras()
    print(f"device: {bpy.context.scene.cycles.device}", flush=True)
    for i in range(a.start, a.start + a.count):
        path = os.path.join(a.out, f"scene_{i:05d}.npz")
        if os.path.exists(path):
            continue
        t = time.time()
        bpy.ops.outliner.orphans_purge(do_recursive=True)  # meshes, textures and materials of earlier scenes
        rng = np.random.default_rng(a.seed + i)
        ob, shape = random_shape(rng)
        for m in list(bpy.data.materials):
            bpy.data.materials.remove(m)
        mat, material = random_material(rng)
        ob.data.materials.append(mat)
        world = dict(name=str(rng.choice(WORLD_NAMES)), rotation=round(float(rng.uniform(0, 2 * math.pi)), 3))
        sr.set_world(world["name"], world["rotation"])
        bpy.context.scene.cycles.seed = i
        L, R = render_view(tmp, cams["L"]), render_view(tmp, cams["R"])
        if L["mask"].mean() < 0.01:
            print(f"scene {i}: object not visible, skipped", flush=True)
            continue
        h = np.float16
        np.savez_compressed(path, L_rgb=L["rgb"].astype(h), R_rgb=R["rgb"].astype(h), L_glossy=L["glossy"].astype(h),
                            L_position=L["position"].astype(h), R_position=R["position"].astype(h),
                            L_mask=L["mask"], R_mask=R["mask"], L_view=L["view"], L_proj=L["proj"],
                            R_view=R["view"], R_proj=R["proj"],
                            params=json.dumps(dict(shape=shape, material=material, world=world)))
        print(f"scene {i}: {time.time() - t:.1f}s {shape['kind']} {material['kind']} {world['name']}", flush=True)


if __name__ == "__main__":
    main()
