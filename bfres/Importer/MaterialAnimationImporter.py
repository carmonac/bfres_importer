"""Import FMAA material animations as Blender actions.

One action per FMAA. It animates, for each affected material:
- tex_srt<N> parameters -> the material's 'tex_srt<N>:<uv>' Mapping
  nodes (location / rotation / scale, converted to Blender's UV space);
- texture pattern animations -> a texture switch built on the sampler's
  image node: one image node per texture the sampler ever shows, chained
  with Mix nodes keyed on a 'pattern:<sampler>' Value node (constant
  interpolation);
- every other shader parameter (const_color*, const_value*, uk_*...) ->
  a custom property 'param_<name>' on the material, initialised with the
  material's static value. uking_mat's color combiner is not emulated
  (#25), so these are exposed for users to wire up rather than connected.
- material visibility ("*_fvm") -> hide_viewport / hide_render of every
  mesh object that uses the material (constant keys).

Parameter curves are sampled at every frame (linear keys), as for
skeletal animations. Node animation lives on the material's node tree,
properties on the material and visibility on the objects; the action gets
one slot for each.
"""
import logging; log = logging.getLogger(__name__)
import struct

import bpy
import numpy as np

from . import animmath
from .TextureImporter import find_image

LINEAR, CONSTANT = 1, 0  # Keyframe.interpolation enum values
SRT_OFFSETS = {'sx': 4, 'sy': 8, 'rot': 12, 'tx': 16, 'ty': 20}


def materials_named(name, candidates):
    return [m for m in candidates if m.get('bfres_name') == name]


def _keys(fc, frames, values, interpolation):
    n = len(frames)
    kps = fc.keyframe_points
    kps.add(n)
    co = np.empty(2 * n, dtype=np.float32)
    co[0::2] = frames
    co[1::2] = values
    kps.foreach_set('co', co)
    kps.foreach_set('interpolation', np.full(n, interpolation, dtype=np.int32))
    fc.update()


def _base_param(mat, name):
    """Static value of a shader parameter as a list of floats, from the
    custom property MaterialImporter stores (matParam_<name>)."""
    p = mat.get('matParam_' + name)
    if p is None:
        return []
    data = p.get('data') if hasattr(p, 'get') else None
    if data is None:
        return []
    if isinstance(data, (bytes, bytearray)):
        return list(struct.unpack('<%df' % (len(data) // 4), data[:len(data) // 4 * 4]))
    try:
        return [float(x) for x in data]
    except TypeError:
        try:
            return [float(data)]
        except (TypeError, ValueError):
            return []


def objects_using(mat):
    """Mesh objects with `mat` in one of their material slots."""
    return [ob for ob in bpy.data.objects if ob.type == 'MESH'
            and any(sl.material == mat for sl in ob.material_slots)]


class MaterialAnimationImporter:
    def __init__(self, context):
        self.context = context

    def import_anims(self, sets, materials):
        materials = [m for m in materials if m.get('bfres_name') and m.node_tree]
        if not sets or not materials:
            return []
        switches = self._build_switches(sets, materials)
        actions = []
        for s in sets:
            try:
                act = self._import_one(s, materials, switches)
            except Exception:
                log.exception("Material animation %s could not be imported", s.name)
                continue
            if act is not None:
                actions.append(act)
        log.info("Imported %d material animation(s)", len(actions))
        return actions

    # ------------------------------------------------------------ patterns

    def _build_switches(self, sets, materials):
        """(material, sampler) -> {texture name: index} for every sampler a
        pattern animation touches, after building its texture switch."""
        wanted = {}
        for s in sets:
            frames = s.frames()
            for ma in s.materials:
                for pt in ma.patterns:
                    names = [s.textures[i] for i in np.unique(pt.sample(frames))
                             if 0 <= i < len(s.textures)]
                    for mat in materials_named(ma.name, materials):
                        lst = wanted.setdefault((mat.name, pt.name), [])
                        for n in names:
                            if n not in lst and find_image(n, mat.get('bfres_family')) is not None:
                                lst.append(n)
        switches = {}
        for (mat_name, sampler), names in wanted.items():
            mat = bpy.data.materials[mat_name]
            node = mat.node_tree.nodes.get('sampler:' + sampler)
            if node is None or not names:
                continue
            switches[(mat_name, sampler)] = self._switch(mat, node, names)
        return switches

    def _switch(self, mat, node, names):
        """Turn `node` into a texture switch over `names`. Returns
        {texture name: index of the Value node}."""
        nt = mat.node_tree
        # keep the material's own texture as index 0 (the look when the
        # animation is not playing)
        # game texture names (images may have been renamed, see TextureImporter)
        cur = node.image.get('bfres_texture') or node.image.name if node.image is not None else None
        if cur is not None and cur in names:
            names = [cur] + [n for n in names if n != cur]
        elif cur is not None:
            names = [cur] + names
        index = {n: i for i, n in enumerate(names)}
        if len(names) == 1:
            return index
        nodes, links = nt.nodes, nt.links
        value = nodes.new('ShaderNodeValue')
        value.name = value.label = 'pattern:' + node.name.split(':', 1)[1]
        value.location = (node.location.x - 250, node.location.y + 200)
        value.outputs[0].default_value = 0.0
        vec_src = node.inputs['Vector'].links[0].from_socket if node.inputs['Vector'].links else None
        # consumers of the original outputs, re-pointed to the chain's end
        consumers = {o.name: [l.to_socket for l in o.links] for o in node.outputs}
        color, alpha = node.outputs['Color'], node.outputs['Alpha']
        for i, name in enumerate(names[1:], start=1):
            img = nodes.new('ShaderNodeTexImage')
            img.image = find_image(name, mat.get('bfres_family'))
            img.label = '%s [%d]' % (name, i)
            img.location = (node.location.x, node.location.y - 260 * i)
            if vec_src is not None:
                links.new(vec_src, img.inputs['Vector'])
            cmp = nodes.new('ShaderNodeMath')
            cmp.operation = 'COMPARE'
            cmp.inputs[1].default_value = float(i)
            cmp.inputs[2].default_value = 0.5
            cmp.location = (node.location.x + 200, node.location.y - 260 * i + 120)
            links.new(value.outputs[0], cmp.inputs[0])
            mix_c = nodes.new('ShaderNodeMix')
            mix_c.data_type = 'RGBA'
            mix_c.location = (node.location.x + 400, node.location.y - 260 * i)
            links.new(cmp.outputs[0], mix_c.inputs['Factor'])
            links.new(color, mix_c.inputs['A'])
            links.new(img.outputs['Color'], mix_c.inputs['B'])
            mix_a = nodes.new('ShaderNodeMix')
            mix_a.data_type = 'FLOAT'
            mix_a.location = (node.location.x + 400, node.location.y - 260 * i - 120)
            links.new(cmp.outputs[0], mix_a.inputs['Factor'])
            links.new(alpha, mix_a.inputs['A'])
            links.new(img.outputs['Alpha'], mix_a.inputs['B'])
            color, alpha = mix_c.outputs['Result'], mix_a.outputs['Result']
        for sock in consumers.get('Color', []):
            links.new(color, sock)
        for sock in consumers.get('Alpha', []):
            links.new(alpha, sock)
        return index

    # ------------------------------------------------------------ actions

    def _import_one(self, s, materials, switches):
        frames = s.frames()
        act = bpy.data.actions.new(name=s.name or 'MaterialAnim')
        act.use_fake_user = True
        act.use_frame_range = True
        act.frame_start, act.frame_end = 0, max(s.frame_count, 1)
        act.use_cyclic = s.looping
        act['bfres_looping'] = s.looping
        wrote = False
        for ma in s.materials:
            for mat in materials_named(ma.name, materials):
                targets = [(mat, 'MATERIAL', self._write_params),
                           (mat.node_tree, 'NODETREE', self._write_nodes)]
                if ma.visibility is not None:
                    targets += [(ob, 'OBJECT', self._write_visibility)
                                for ob in objects_using(mat)]
                for idb, id_type, writer in targets:
                    if idb.animation_data is None:
                        idb.animation_data_create()
                    previous = idb.animation_data.action
                    previous_slot = idb.animation_data.action_slot
                    idb.animation_data.action = act
                    # Every embedded node tree is called "Shader Nodetree",
                    # so the default slot (named after the ID) would be
                    # shared by all materials: give each its own slot.
                    slot = act.slots.new(id_type, idb.name if id_type == 'OBJECT' else mat.name)
                    idb.animation_data.action_slot = slot
                    n = writer(act, idb, mat, ma, s, frames, switches)
                    wrote = wrote or n > 0
                    if n == 0:
                        act.slots.remove(slot)
                    # Keep looping animations (wing blur, idle glow) playing;
                    # one-shot ones and variant selectors (e.g. Lynel colors,
                    # one frame per variant) stay available as actions
                    # without changing the material's default look. The
                    # first animation that qualifies wins.
                    if previous is not None or n == 0 or not s.looping:
                        idb.animation_data.action = previous
                        if previous is not None:
                            idb.animation_data.action_slot = previous_slot
        if not wrote:
            bpy.data.actions.remove(act)
            return None
        return act

    def _write_visibility(self, act, ob, mat, ma, s, frames, switches):
        hidden = (~ma.sample_visibility(frames)).astype(np.float64)
        for path in ('hide_viewport', 'hide_render'):
            fc = act.fcurve_ensure_for_datablock(ob, path, index=0,
                                                 group_name='visibility:' + ma.name)
            _keys(fc, frames, hidden, CONSTANT)
        return 2

    def _write_params(self, act, mat, _mat, ma, s, frames, switches):
        n = 0
        for pa in ma.params:
            if pa.name.startswith('tex_srt'):
                continue
            offsets = sorted(set(pa.curves) | set(pa.constants))
            if not offsets:
                continue
            base = _base_param(mat, pa.name)
            size = max(len(base), offsets[-1] // 4 + 1)
            base = base + [0.0] * (size - len(base))
            prop = 'param_' + pa.name
            mat[prop] = base
            for off in offsets:
                comp = off // 4
                values = pa.sample(frames, off, base[comp])
                fc = act.fcurve_ensure_for_datablock(mat, '["%s"]' % prop, index=comp,
                                                     group_name=pa.name)
                _keys(fc, frames, values, LINEAR)
                n += 1
        return n

    def _write_nodes(self, act, nt, mat, ma, s, frames, switches):
        n = 0
        for pa in ma.params:
            if not pa.name.startswith('tex_srt'):
                continue
            try:
                idx = int(pa.name[len('tex_srt'):])
            except ValueError:
                continue
            nodes = [nd for nd in nt.nodes if nd.name.startswith('tex_srt%d:' % idx)]
            if not nodes:
                continue
            base = nodes[0].get('bfres_srt') or (0, 1.0, 1.0, 0.0, 0.0, 0.0)
            vals = {k: pa.sample(frames, off, base[1 + i])
                    for i, (k, off) in enumerate(SRT_OFFSETS.items())}
            (lx, ly), r, (sx, sy) = animmath.srt_to_mapping(
                vals['sx'], vals['sy'], vals['rot'], vals['tx'], vals['ty'])
            for nd in nodes:
                base_path = 'nodes["%s"].inputs[%%d].default_value' % nd.name
                for inp, comp, values in ((1, 0, lx), (1, 1, ly), (2, 2, r), (3, 0, sx), (3, 1, sy)):
                    fc = act.fcurve_ensure_for_datablock(nt, base_path % inp, index=comp,
                                                         group_name=pa.name)
                    _keys(fc, frames, np.broadcast_to(values, frames.shape), LINEAR)
                    n += 1
        for pt in ma.patterns:
            index = switches.get((mat.name, pt.name))
            if index is None or len(index) < 2:
                continue
            local = pt.sample(frames)
            values = np.array([index.get(s.textures[i], 0) if 0 <= i < len(s.textures) else 0
                               for i in local], dtype=np.float64)
            path = 'nodes["pattern:%s"].outputs[0].default_value' % pt.name
            fc = act.fcurve_ensure_for_datablock(nt, path, index=0, group_name=pt.name)
            _keys(fc, frames, values, CONSTANT)
            n += 1
        return n
