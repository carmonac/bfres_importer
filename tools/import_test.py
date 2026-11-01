"""Headless import test harness for the BFRES importer.

Run inside Blender:

    blender -b --factory-startup --python tools/import_test.py -- \
        PATH [PATH ...] [--out report.json] [--render DIR]
        [--baseline old.json] [--opt name=value ...] [--all]

PATH may be a model file or a folder. In a folder, every ``*.sbfres`` /
``*.bfres`` is imported except texture-only (``*.Tex.*``) and
animation-only (``*_Animation.*``) files, unless ``--all`` is given.
Texture files next to a model are picked up by the importer itself.

For each model the scene is reset, the importer runs, and a JSON entry
records: status, error traceback, wall time, log warnings/errors, meshes
(vertices, faces, UV layers, vertex groups, materials, bounding box, UVs
out of range / NaN), armatures (bones), images (size, whether pixel data
was decoded) and materials.

--render DIR    writes DIR/<model>.png: a Workbench render with texture
                colors, camera framing every mesh.
--engine eevee  renders with EEVEE instead (sun + ambient light), to judge
                materials rather than geometry.
--baseline F    compares against a previous report and prints differences
                (status, vertex/face/image counts, time).
--opt k=v       forwards an option to the import operator, e.g.
                --opt first_lod=False. Values are parsed as Python literals.

The exit code is 1 when any model fails to import.

The add-on is loaded from this repository, so the harness also works
with --factory-startup (no installed add-on needed).
"""
import ast
import importlib.util
import io
import json
import logging
import math
import os
import sys
import time
import traceback

import bpy
import mathutils

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = "bfres_importer_dev"


# ---------------------------------------------------------------- add-on

def ensure_addon():
    """Make sure bpy.ops.import_scene.nxbfres exists, loading this repo."""
    if hasattr(bpy.types, "IMPORT_SCENE_OT_nxbfres"):
        return "installed"
    spec = importlib.util.spec_from_file_location(
        PKG, os.path.join(REPO, "__init__.py"),
        submodule_search_locations=[REPO])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[PKG] = mod
    spec.loader.exec_module(mod)
    mod.register()
    return "repo"


# ---------------------------------------------------------------- inputs

# Tiled surfaces legitimately use UVs in the hundreds (BotW walls go past
# 170), so only flag values no texture mapping would produce, such as raw
# integer UVs that were never normalized.
UV_LIMIT = 4096.0

MODEL_EXTS = (".sbfres", ".bfres", ".fres", ".szs", ".zs")


def is_aux(name):
    base = name.lower()
    return ".tex." in base or "_animation." in base


def collect(paths, include_all):
    out = []
    for p in paths:
        p = os.path.abspath(os.path.expanduser(p))
        if os.path.isdir(p):
            for fn in sorted(os.listdir(p)):
                if fn.lower().endswith(MODEL_EXTS) and (include_all or not is_aux(fn)):
                    out.append(os.path.join(p, fn))
        else:
            out.append(p)
    return out


# ---------------------------------------------------------------- logging

class Capture(logging.Handler):
    def __init__(self):
        super().__init__(logging.WARNING)
        self.records = []

    def emit(self, record):
        try:
            msg = record.getMessage()
        except Exception:
            msg = str(record.msg)
        self.records.append("%s %s: %s" % (record.levelname, record.name, msg))


# ---------------------------------------------------------------- scene stats

def reset_scene():
    bpy.ops.wm.read_homefile(use_empty=True)
    ensure_addon()


def normal_agreement(me):
    """How well the custom (split) normals agree with the face normals.

    Returns (mean dot product, fraction of corners pointing away from
    their face). Correct normals give a mean close to 1 and ~0 flipped;
    normals in the wrong space or axis give a low mean or many flips.
    """
    import numpy as np
    n_loops = len(me.loops)
    if not n_loops or not len(me.polygons):
        return None, None
    corner = np.empty(n_loops * 3, np.float32)
    me.corner_normals.foreach_get("vector", corner)
    face = np.empty(len(me.polygons) * 3, np.float32)
    me.polygon_normals.foreach_get("vector", face)
    start = np.empty(len(me.polygons), np.int64)
    total = np.empty(len(me.polygons), np.int64)
    me.polygons.foreach_get("loop_start", start)
    me.polygons.foreach_get("loop_total", total)
    polys = np.repeat(np.arange(len(me.polygons)), total)
    within = np.arange(total.sum()) - np.repeat(np.cumsum(total) - total, total)
    poly_of_loop = np.empty(n_loops, np.int64)
    poly_of_loop[np.repeat(start, total) + within] = polys
    dots = np.einsum("ij,ij->i", corner.reshape(-1, 3), face.reshape(-1, 3)[poly_of_loop])
    return round(float(dots.mean()), 4), round(float((dots < 0).mean()), 4)


def mesh_stats(ob):
    me = ob.data
    nrm_mean, nrm_flipped = normal_agreement(me)
    uv_out = uv_nan = 0
    uv_total = 0
    for layer in me.uv_layers:
        n = len(layer.data)
        if not n:
            continue
        buf = [0.0] * (n * 2)
        layer.data.foreach_get("uv", buf)
        uv_total += n
        for i in range(0, len(buf), 2):
            u, v = buf[i], buf[i + 1]
            if not (math.isfinite(u) and math.isfinite(v)):
                uv_nan += 1
            elif not (-UV_LIMIT <= u <= UV_LIMIT and -UV_LIMIT <= v <= UV_LIMIT):
                uv_out += 1
    corners = [ob.matrix_world @ mathutils.Vector(c) for c in ob.bound_box]
    lo = [min(c[i] for c in corners) for i in range(3)]
    hi = [max(c[i] for c in corners) for i in range(3)]
    return {
        "name": ob.name,
        "verts": len(me.vertices),
        "faces": len(me.polygons),
        "uv_layers": [l.name for l in me.uv_layers],
        "uv_out_of_range": uv_out,
        "normal_agreement": nrm_mean,
        "normals_flipped": nrm_flipped,
        "uv_nan": uv_nan,
        "uv_loops": uv_total,
        "color_attrs": [a.name for a in me.color_attributes],
        "vertex_groups": len(ob.vertex_groups),
        "materials": [m.name for m in me.materials if m],
        "bbox_min": [round(x, 4) for x in lo],
        "bbox_max": [round(x, 4) for x in hi],
    }


def image_stats(img):
    w, h = img.size
    return {
        "name": img.name,
        "size": [w, h],
        "has_data": bool(img.has_data and w and h),
        "colorspace": img.colorspace_settings.name,
    }


def _upstream_images(socket, seen=None):
    """Image names feeding a node input, following links backwards."""
    seen = seen if seen is not None else set()
    names = []
    for link in socket.links:
        node = link.from_node
        if node.name in seen:
            continue
        seen.add(node.name)
        if node.type == "TEX_IMAGE":
            names.append(node.image.name if node.image else None)
        for inp in node.inputs:
            names += _upstream_images(inp, seen)
    return names


def material_stats(mat):
    tex = []
    inputs = {}
    if mat.node_tree:
        for n in mat.node_tree.nodes:
            if n.type == "TEX_IMAGE":
                tex.append(n.image.name if n.image else None)
            if n.type == "BSDF_PRINCIPLED":
                for inp in n.inputs:
                    if inp.is_linked:
                        inputs[inp.name] = _upstream_images(inp)
    return {"name": mat.name, "textures": tex, "inputs": inputs,
            "backface_culling": mat.use_backface_culling}


def scene_report():
    meshes = [mesh_stats(o) for o in bpy.data.objects if o.type == "MESH"]
    arms = [{"name": o.name, "bones": len(o.data.bones)}
            for o in bpy.data.objects if o.type == "ARMATURE"]
    imgs = [image_stats(i) for i in bpy.data.images
            if i.type == "IMAGE" and i.source in {"GENERATED", "FILE", "PACKED"}]
    mats = [material_stats(m) for m in bpy.data.materials]
    return {
        "totals": {
            "objects": len(bpy.data.objects),
            "meshes": len(meshes),
            "verts": sum(m["verts"] for m in meshes),
            "faces": sum(m["faces"] for m in meshes),
            "bones": sum(a["bones"] for a in arms),
            "images": len(imgs),
            "images_without_data": sum(1 for i in imgs if not i["has_data"]),
            "materials": len(mats),
            "uv_out_of_range": sum(m["uv_out_of_range"] for m in meshes),
            "uv_nan": sum(m["uv_nan"] for m in meshes),
            # meshes whose normals mostly disagree with their faces
            "bad_normals": sum(1 for m in meshes if m["normal_agreement"] is not None
                               and m["normal_agreement"] < 0.5),
        },
        "meshes": meshes,
        "armatures": arms,
        "images": imgs,
        "materials": mats,
    }


# ---------------------------------------------------------------- render

def render(path, size=768, engine="workbench"):
    meshes = [o for o in bpy.data.objects if o.type == "MESH" and o.visible_get()]
    if not meshes:
        return False
    pts = [o.matrix_world @ mathutils.Vector(c) for o in meshes for c in o.bound_box]
    lo = mathutils.Vector([min(p[i] for p in pts) for i in range(3)])
    hi = mathutils.Vector([max(p[i] for p in pts) for i in range(3)])
    center = (lo + hi) / 2
    radius = max((hi - lo).length / 2, 1e-3)

    scene = bpy.context.scene
    cam_data = bpy.data.cameras.new("harness_cam")
    cam_data.lens = 50
    cam_data.clip_start = radius / 1000
    cam_data.clip_end = radius * 100
    cam = bpy.data.objects.new("harness_cam", cam_data)
    scene.collection.objects.link(cam)
    direction = mathutils.Vector((1.0, -1.4, 0.8)).normalized()
    fov = 2 * math.atan(cam_data.sensor_width / (2 * cam_data.lens))
    dist = radius / math.sin(fov / 2) * 1.05
    cam.location = center + direction * dist
    cam.rotation_euler = (center - cam.location).to_track_quat("-Z", "Y").to_euler()
    scene.camera = cam

    if scene.world is None:
        scene.world = bpy.data.worlds.new("harness_world")
    scene.world.color = (0.18, 0.18, 0.2)

    if engine == "eevee":
        # judge materials: real shading with a key light and soft fill
        for name in ("BLENDER_EEVEE", "BLENDER_EEVEE_NEXT"):
            try:
                scene.render.engine = name
                break
            except TypeError:
                continue
        scene.world.use_nodes = True
        bg = scene.world.node_tree.nodes.get("Background")
        if bg:
            bg.inputs["Color"].default_value = (0.35, 0.37, 0.42, 1)
            bg.inputs["Strength"].default_value = 0.8
        sun_data = bpy.data.lights.new("harness_sun", "SUN")
        sun_data.energy = 3.0
        sun = bpy.data.objects.new("harness_sun", sun_data)
        sun.rotation_euler = (0.9, 0.2, 0.6)
        scene.collection.objects.link(sun)
    else:
        scene.render.engine = "BLENDER_WORKBENCH"
        shading = scene.display.shading
        shading.light = "STUDIO"
        shading.color_type = "TEXTURE"
    scene.render.resolution_x = size
    scene.render.resolution_y = size
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    return True


# ---------------------------------------------------------------- run

def import_one(path, opts, render_dir, engine="workbench"):
    reset_scene()
    cap = Capture()
    root = logging.getLogger()
    root.addHandler(cap)
    entry = {"file": os.path.basename(path), "path": path}
    t0 = time.time()
    stdout = sys.stdout
    sys.stdout = io.StringIO()  # importer prints progress spam
    try:
        res = bpy.ops.import_scene.nxbfres(filepath=path, **opts)
        entry["status"] = "ok" if "FINISHED" in res else "cancelled"
        entry["error"] = None
    except Exception:
        entry["status"] = "error"
        entry["error"] = traceback.format_exc()[-4000:]
    finally:
        sys.stdout = stdout
        root.removeHandler(cap)
    entry["seconds"] = round(time.time() - t0, 2)
    entry["log"] = cap.records[-200:]
    entry["log_count"] = len(cap.records)
    entry.update(scene_report())
    if render_dir and entry["status"] == "ok":
        os.makedirs(render_dir, exist_ok=True)
        png = os.path.join(render_dir, os.path.splitext(entry["file"])[0] + ".png")
        try:
            entry["render"] = png if render(png, engine=engine) else None
        except Exception:
            entry["render"] = None
            entry["log"].append("HARNESS render failed: " + traceback.format_exc()[-500:])
    return entry


def summary_line(e):
    t = e["totals"]
    err = ""
    if e["error"]:
        lines = [l for l in e["error"].strip().splitlines()
                 if l.strip() and not l.startswith(("Location:", "  "))]
        err = (lines[-1] if lines else "")[:70]
    return "%-36s %-6s %6.1fs  mesh %3d  vtx %7d  img %3d/%-3d  uvbad %4d  nrmbad %3d  log %3d  %s" % (
        e["file"][:36], e["status"], e["seconds"], t["meshes"], t["verts"],
        t["images"] - t["images_without_data"], t["images"],
        t["uv_out_of_range"] + t["uv_nan"], t.get("bad_normals", 0), e["log_count"], err)


def compare(report, baseline_path):
    with open(baseline_path) as f:
        old = {e["file"]: e for e in json.load(f)["results"]}
    lines = []
    for e in report["results"]:
        o = old.get(e["file"])
        if not o:
            lines.append("NEW      %s" % e["file"])
            continue
        diffs = []
        if o["status"] != e["status"]:
            diffs.append("status %s -> %s" % (o["status"], e["status"]))
        for k in ("meshes", "verts", "faces", "bones", "images", "images_without_data",
                  "uv_out_of_range", "uv_nan", "bad_normals"):
            if o["totals"].get(k) != e["totals"].get(k):
                diffs.append("%s %s -> %s" % (k, o["totals"].get(k), e["totals"].get(k)))
        if o["seconds"] and abs(e["seconds"] - o["seconds"]) / o["seconds"] > 0.25:
            diffs.append("time %.1fs -> %.1fs" % (o["seconds"], e["seconds"]))
        if diffs:
            lines.append("CHANGED  %s: %s" % (e["file"], "; ".join(diffs)))
    return lines


def parse_args(argv):
    import argparse
    ap = argparse.ArgumentParser(prog="import_test.py")
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--out")
    ap.add_argument("--render")
    ap.add_argument("--engine", choices=("workbench", "eevee"), default="workbench",
                    help="renderer for --render (eevee shows materials)")
    ap.add_argument("--baseline")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--opt", action="append", default=[])
    return ap.parse_args(argv)


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    args = parse_args(argv)
    opts = {}
    for kv in args.opt:
        k, v = kv.split("=", 1)
        try:
            opts[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            opts[k] = v
    loaded = ensure_addon()
    files = collect(args.paths, args.all)
    report = {
        "blender": bpy.app.version_string,
        "addon": loaded,
        "options": opts,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "results": [],
    }
    print("BFRES harness: %d file(s), add-on from %s" % (len(files), loaded))
    for path in files:
        e = import_one(path, opts, args.render, args.engine)
        report["results"].append(e)
        print(summary_line(e), flush=True)
    ok = sum(1 for e in report["results"] if e["status"] == "ok")
    print("== %d/%d imported, %.1fs total" % (
        ok, len(files), sum(e["seconds"] for e in report["results"])))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=1)
        print("report: %s" % args.out)
    if args.baseline:
        for line in compare(report, args.baseline) or ["no differences"]:
            print(line)
    return 0 if ok == len(files) else 1


if __name__ == "__main__":
    code = 1
    try:
        code = main()
    except SystemExit as e:
        code = e.code or 0
    except Exception:
        traceback.print_exc()
    sys.stdout.flush()
    os._exit(code)
