"""Fingerprint everything the importer produces, to prove a refactor
changes nothing.

Run inside Blender, once per version of the code:

    blender -b --factory-startup --python tools/snapshot.py -- \
        PATH [PATH ...] --out snap.json [--compare other.json]

PATH is a model file or a folder (texture-only and animation-only files
are skipped, as in import_test.py). For every mesh it hashes vertex
positions, face corners, loop starts, smooth flags, every UV layer,
corner normals and the full list of (vertex, group, weight); for every
image, its pixels. Floats are rounded (positions 1e-4, UVs 1e-5,
normals 1e-3, pixels 1e-4) so the hashes ignore float noise.

--compare prints every mesh or image whose fingerprint differs from a
previous snapshot, and which fields differ. Exit code 1 if any do.
"""
import hashlib
import io
import json
import os
import sys

import bpy
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import import_test as harness  # noqa: E402


def _h(arr):
    return hashlib.sha1(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]


def _get(collection, attr, n, dtype, width=1):
    buf = np.empty(n * width, dtype)
    collection.foreach_get(attr, buf)
    return buf


def mesh_fingerprint(ob):
    me = ob.data
    nv, nl, np_ = len(me.vertices), len(me.loops), len(me.polygons)
    names = [g.name for g in ob.vertex_groups]
    weights = sorted((v.index, names[g.group], round(g.weight, 4))
                     for v in me.vertices for g in v.groups if g.weight > 0)
    return {
        "verts": nv,
        "faces": np_,
        "co": _h(np.round(_get(me.vertices, "co", nv, np.float32, 3), 4)),
        "corners": _h(_get(me.loops, "vertex_index", nl, np.int32)),
        "loop_start": _h(_get(me.polygons, "loop_start", np_, np.int32)),
        "smooth": _h(_get(me.polygons, "use_smooth", np_, bool)),
        "uv": {l.name: _h(np.round(_get(l.data, "uv", nl, np.float32, 2), 5))
               for l in me.uv_layers},
        "normals": _h(np.round(_get(me.corner_normals, "vector", nl, np.float32, 3), 3)),
        "groups": len(names),
        "weights": hashlib.sha1(repr(weights).encode()).hexdigest()[:16],
        "materials": [m.name for m in me.materials if m],
    }


def image_fingerprint(img):
    w, h = img.size
    px = np.empty(w * h * 4, np.float32)
    img.pixels.foreach_get(px)
    return {"size": [w, h], "pixels": _h(np.round(px, 4))}


def snapshot(path):
    harness.reset_scene()
    stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        bpy.ops.import_scene.nxbfres(filepath=path)
    finally:
        sys.stdout = stdout
    name = os.path.basename(path)
    out = {}
    for ob in bpy.data.objects:
        if ob.type == "MESH":
            out["%s/mesh/%s" % (name, ob.name)] = mesh_fingerprint(ob)
    for img in bpy.data.images:
        if img.size[0] and img.size[1]:
            out["%s/image/%s" % (name, img.name)] = image_fingerprint(img)
    return out


def compare(new, old):
    lines = []
    for key in sorted(set(new) | set(old)):
        a, b = old.get(key), new.get(key)
        if a == b:
            continue
        if a is None or b is None:
            lines.append("%s  %s" % ("NEW    " if a is None else "MISSING", key))
        else:
            fields = [f for f in sorted(set(a) | set(b)) if a.get(f) != b.get(f)]
            lines.append("CHANGED  %s: %s" % (key, ", ".join(fields)))
    return lines


def main():
    import argparse
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser(prog="snapshot.py")
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--compare")
    args = ap.parse_args(argv)

    harness.ensure_addon()
    snap = {}
    for path in harness.collect(args.paths, False):
        snap.update(snapshot(path))
        print("snapshot: %s" % os.path.basename(path), flush=True)
    with open(args.out, "w") as f:
        json.dump(snap, f, indent=1, sort_keys=True)
    meshes = sum(1 for k in snap if "/mesh/" in k)
    print("%d meshes, %d images -> %s" % (meshes, len(snap) - meshes, args.out))

    if args.compare:
        with open(args.compare) as f:
            diffs = compare(snap, json.load(f))
        for line in diffs or ["identical to %s" % args.compare]:
            print(line)
        return 1 if diffs else 0
    return 0


if __name__ == "__main__":
    code = 1
    try:
        code = main()
    except Exception:
        import traceback
        traceback.print_exc()
    sys.stdout.flush()
    os._exit(code)
