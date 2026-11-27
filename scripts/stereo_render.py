"""Stereo renders for the binocular highlight pilot (run inside Blender).

  blender -b --factory-startup -P scripts/stereo_render.py -- --out runs/stereo_pilot [--test]

For every shape x material x illumination it renders three toe-in cameras aimed at the object: L, R (6.5 cm right of L)
and R2 (13 cm right of L), at 0.6 m from an object of radius 0.15 m. Each view is saved as an .npz of float32 arrays
(rows top to bottom): rgb (linear combined), diffuse and glossy (direct + indirect, times colour), position (world),
normal, mask (object index), plus the camera matrices, so that views can be warped onto each other exactly.
"""
import argparse
import os
import sys
import time

import bpy
import numpy as np
from mathutils import Matrix, Vector

ERATO = os.environ.get("ERATO_OBJ", "")  # path to an Erato scan .obj, needed only for the "erato" shape
WORLDS = os.path.join(bpy.utils.system_resource("DATAFILES"), "studiolights", "world")  # Blender's studio HDRIs
BASE = 0.6  # body albedo of every material
CAMS = {"L": 0.0, "R": 0.065, "R2": 0.13}  # x offset from the left eye, metres
DIST, RADIUS = 0.6, 0.15


def args_after_dashes():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="runs/stereo_pilot")
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--spp", type=int, default=512)
    p.add_argument("--shapes", default="sphere,blob,blob_fine,erato")
    p.add_argument("--materials", default="matte,semi,glossy")
    p.add_argument("--worlds", default="courtyard,interior")
    p.add_argument("--cams", default="L,R,R2")
    p.add_argument("--device", default="GPU")
    p.add_argument("--texture", action="store_true", help="albedo noise pattern on the surface (helps stereo matching)")
    return p.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])


def clear_objects():
    for ob in [o for o in bpy.data.objects if o.type == "MESH"]:
        bpy.data.objects.remove(ob, do_unlink=True)


def normalise(ob):
    """Centre the object at the origin and scale it to a bounding sphere of RADIUS."""
    bpy.context.view_layer.objects.active = ob
    ob.select_set(True)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    pts = np.array([v.co for v in ob.data.vertices])
    c = (pts.max(0) + pts.min(0)) / 2
    r = np.linalg.norm(pts - c, axis=1).max()
    ob.data.transform(Matrix.Scale(RADIUS / r, 4) @ Matrix.Translation(-Vector(c)))


def displace(ob, name, scale, strength, depth):
    tex = bpy.data.textures.new(name, "CLOUDS")
    tex.noise_scale, tex.noise_depth, tex.noise_basis = scale, depth, "BLENDER_ORIGINAL"
    m = ob.modifiers.new(name, "DISPLACE")
    m.texture, m.texture_coords, m.strength, m.mid_level = tex, "LOCAL", strength, 0.5


def import_erato():
    if not ERATO or not os.path.exists(ERATO):
        raise SystemExit("the erato shape needs ERATO_OBJ set to the path of the Erato scan .obj")
    bpy.ops.wm.obj_import(filepath=ERATO)


def make_shape(shape):
    clear_objects()
    if shape == "erato":
        import_erato()
        ob = bpy.context.selected_objects[0]
        bpy.ops.object.shade_smooth()
    else:
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=8, radius=RADIUS)
        ob = bpy.context.active_object
        if shape in ("blob", "blob_fine"):
            displace(ob, "low", 0.12, 0.05, 0)
        if shape == "blob_fine":
            displace(ob, "fine", 0.012, 0.012, 2)
        bpy.ops.object.shade_smooth()
        for m in list(ob.modifiers):
            bpy.ops.object.modifier_apply(modifier=m.name)
    normalise(ob)
    ob.pass_index = 1
    return ob


def make_material(kind, texture=False):
    mat = bpy.data.materials.new(kind)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    if kind == "matte":
        bsdf = nt.nodes.new("ShaderNodeBsdfDiffuse")
        bsdf.inputs["Color"].default_value = (BASE, BASE, BASE, 1)
    else:
        bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
        bsdf.inputs["Base Color"].default_value = (BASE, BASE, BASE, 1)
        bsdf.inputs["Roughness"].default_value = {"semi": 0.25, "glossy": 0.05}[kind]
        bsdf.inputs["IOR"].default_value = 1.5
    nt.links.new(bsdf.outputs[0], out.inputs["Surface"])
    if texture:  # albedo pattern fixed to the surface (object coordinates): noise between 0.4 and 0.8 x BASE
        coord = nt.nodes.new("ShaderNodeTexCoord")
        noise = nt.nodes.new("ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value, noise.inputs["Detail"].default_value = 40.0, 4.0
        ramp = nt.nodes.new("ShaderNodeMapRange")
        ramp.inputs["From Min"].default_value, ramp.inputs["From Max"].default_value = 0.3, 0.7
        ramp.inputs["To Min"].default_value, ramp.inputs["To Max"].default_value = 0.4 * BASE / 0.6, 0.8 * BASE / 0.6
        nt.links.new(coord.outputs["Object"], noise.inputs["Vector"])
        nt.links.new(noise.outputs["Fac"], ramp.inputs["Value"])
        nt.links.new(ramp.outputs["Result"], bsdf.inputs[0])
    return mat


def set_world(name, rotation=0.0):
    """Environment map `name` (Blender's bundled studio-light worlds), rotated about the vertical axis (radians)."""
    w = bpy.data.worlds.get("pilot") or bpy.data.worlds.new("pilot")
    bpy.context.scene.world = w
    w.use_nodes = True
    nt = w.node_tree
    nt.nodes.clear()
    coord = nt.nodes.new("ShaderNodeTexCoord")
    mapping = nt.nodes.new("ShaderNodeMapping")
    mapping.inputs["Rotation"].default_value = (0, 0, rotation)
    nt.links.new(coord.outputs["Generated"], mapping.inputs["Vector"])
    env = nt.nodes.new("ShaderNodeTexEnvironment")
    nt.links.new(mapping.outputs["Vector"], env.inputs["Vector"])
    env.image = bpy.data.images.load(os.path.join(WORLDS, name + ".exr"), check_existing=True)
    bg = nt.nodes.new("ShaderNodeBackground")
    out = nt.nodes.new("ShaderNodeOutputWorld")
    nt.links.new(env.outputs["Color"], bg.inputs["Color"])
    nt.links.new(bg.outputs[0], out.inputs["Surface"])


def make_cameras():
    cams = {}
    for name, dx in CAMS.items():
        data = bpy.data.cameras.new(name)
        data.lens, data.sensor_width, data.clip_start = 50, 36, 0.05
        cam = bpy.data.objects.new(name, data)
        bpy.context.scene.collection.objects.link(cam)
        cam.location = (-0.0325 + dx, -DIST, 0.0)
        cam.rotation_euler = (Vector((0, 0, 0)) - cam.location).to_track_quat("-Z", "Y").to_euler()
        cams[name] = cam
    return cams


PASSES = ["Image", "Position", "Normal", "IndexOB", "DiffDir", "DiffInd", "DiffCol", "GlossDir", "GlossInd", "GlossCol"]


def setup_render(a, tmp):
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    if a.device == "GPU":
        prefs.compute_device_type = "CUDA"
        prefs.get_devices()
        for d in prefs.devices:
            d.use = d.type == "CUDA"
        sc.cycles.device = "GPU" if any(d.use for d in prefs.devices) else "CPU"
    sc.cycles.samples, sc.cycles.use_denoising, sc.cycles.use_adaptive_sampling = a.spp, False, False
    sc.cycles.seed = 0
    sc.render.resolution_x = sc.render.resolution_y = a.size
    sc.render.resolution_percentage = 100
    sc.render.film_transparent = False
    sc.view_settings.view_transform = "Standard"
    vl = sc.view_layers[0]
    vl.use_pass_position = vl.use_pass_normal = vl.use_pass_object_index = True
    vl.use_pass_diffuse_direct = vl.use_pass_diffuse_indirect = vl.use_pass_diffuse_color = True
    vl.use_pass_glossy_direct = vl.use_pass_glossy_indirect = vl.use_pass_glossy_color = True
    sc.use_nodes = True
    nt = sc.node_tree
    nt.nodes.clear()
    rl = nt.nodes.new("CompositorNodeRLayers")
    fo = nt.nodes.new("CompositorNodeOutputFile")
    fo.base_path = tmp
    fo.format.file_format, fo.format.color_depth, fo.format.color_management = "OPEN_EXR", "32", "OVERRIDE"
    fo.format.view_settings.view_transform = "Standard"
    fo.file_slots.clear()
    for p in PASSES:
        fo.file_slots.new(p + "_")
        nt.links.new(rl.outputs[p], fo.inputs[p + "_"])
    nt.nodes.new("CompositorNodeComposite")
    nt.links.new(rl.outputs["Image"], nt.nodes["Composite"].inputs["Image"])


def read_exr(path):
    img = bpy.data.images.load(path)
    img.colorspace_settings.name = "Non-Color"
    w, h = img.size
    px = np.array(img.pixels[:], dtype=np.float32).reshape(h, w, 4)[::-1]
    bpy.data.images.remove(img)
    return px


def save_view(tmp, out_path, cam):
    sc = bpy.context.scene
    f = {p: read_exr(os.path.join(tmp, f"{p}_{sc.frame_current:04d}.exr")) for p in PASSES}
    rgb = lambda k: f[k][..., :3]
    w, h = sc.render.resolution_x, sc.render.resolution_y
    depsgraph = bpy.context.evaluated_depsgraph_get()
    proj = cam.calc_matrix_camera(depsgraph, x=w, y=h)
    np.savez_compressed(out_path, rgb=rgb("Image"), position=rgb("Position"), normal=rgb("Normal"),
                        mask=f["IndexOB"][..., 0] > 0.5,
                        diffuse=(rgb("DiffDir") + rgb("DiffInd")) * rgb("DiffCol"),
                        glossy=(rgb("GlossDir") + rgb("GlossInd")) * rgb("GlossCol"),
                        view=np.array(cam.matrix_world.inverted(), np.float32), proj=np.array(proj, np.float32),
                        cam_pos=np.array(cam.location, np.float32))


def main():
    a = args_after_dashes()
    os.makedirs(a.out, exist_ok=True)
    tmp = os.path.abspath(os.path.join(a.out, "_tmp"))
    bpy.ops.wm.read_factory_settings(use_empty=True)
    setup_render(a, tmp)
    cams = make_cameras()
    mats = {m: make_material(m, a.texture) for m in a.materials.split(",")}
    print(f"device: {bpy.context.scene.cycles.device}", flush=True)
    for shape in a.shapes.split(","):
        t = time.time()
        ob = make_shape(shape)
        print(f"{shape}: {len(ob.data.polygons)} faces, built in {time.time() - t:.0f}s", flush=True)
        for m, mat in mats.items():
            ob.data.materials.clear()
            ob.data.materials.append(mat)
            for wname in a.worlds.split(","):
                set_world(wname)
                d = os.path.join(a.out, f"{shape}_{m}_{wname}")
                os.makedirs(d, exist_ok=True)
                for cname in a.cams.split(","):
                    t = time.time()
                    bpy.context.scene.camera = cams[cname]
                    bpy.ops.render.render(write_still=False)
                    save_view(tmp, os.path.join(d, cname + ".npz"), cams[cname])
                    print(f"  {shape} {m} {wname} {cname}: {time.time() - t:.1f}s", flush=True)


if __name__ == "__main__":
    main()
