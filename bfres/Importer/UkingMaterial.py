"""Blender materials for Breath of the Wild's `uking_mat` shader.

uking_mat is a programmable shader: its texture slots (`_a0`, `_s0`,
`_n0`, ... in the shader's sampler dictionary) are generic, and shader
options decide what each slot is used for and how colors are combined
(`uking_colorN_*`). The same slot can hold a normal map in one material
and a second albedo layer in another. What *is* reliable is the name of
the **material** sampler that feeds a slot -- `_a0` albedo, `_n0` normal,
`_s0` specular, `_e0` emission... -- which always matches the texture's
own suffix (`_Alb`, `_Nrm`, `_Spm`, `_Emm`...). Roles are taken from it.

The UV set of a texture comes from the slot that consumes it:
`uking_textureN_texcoord` is the UV layer of the N-th shader slot
(dictionary order). Tangent data in the game files confirms both this and
that normal maps need no green-channel flip after the importer's V flip.

What is built (a faithful translation of the common case, not an
emulation of the color combiner):
  albedo (+ AO multiplied in) -> Base Color, albedo alpha (or the only
  channel of a BC4 albedo) -> Alpha, clipped at gsys_alpha_test_value for
  alpha-tested materials, blended for translucent / color-blended ones,
  normal map,
  specular mask (red channel, scaled) -> Specular IOR Level, metalness,
  emission color (x emission mask), backface culling from
  gsys_render_state_display_face. Other textures (second layers, terrain
  arrays, indirect maps) are added as unconnected, labeled image nodes.
"""
import logging; log = logging.getLogger(__name__)
import bpy

from . import animmath

SHADER = 'uking_mat'

# BotW has no roughness maps; this matte default (plus the specular mask
# scaled to at most Blender's default specular) avoids the plastic look.
DEFAULT_ROUGHNESS = 0.7
SPECULAR_SCALE = 0.5

# material sampler name prefix -> role; longer prefixes first
SAMPLER_ROLES = (
    ('_em', 'emission_mask'),
    ('_mt', 'metalness'),
    ('_ao', 'ao'),
    ('_op', 'opacity'),
    ('_a', 'albedo'),
    ('_n', 'normal'),
    ('_s', 'specular'),
    ('_e', 'emission'),
)
# texture name suffix -> role, for samplers with generic names (sampler0...)
SUFFIX_ROLES = {
    'Alb': 'albedo', 'Nrm': 'normal', 'Spm': 'specular', 'Emm': 'emission',
    'EmmMsk': 'emission_mask', 'Mtl': 'metalness', 'AO': 'ao',
}
COLOR_ROLES = {'albedo', 'emission'}
SINGLE_CHANNEL_FORMATS = {'BC4'}
DATA_ROLES = {'normal', 'specular', 'metalness', 'ao', 'emission_mask', 'opacity'}


def texture_suffix(name):
    base = name.rsplit('.', 1)[0] if name[-1:].isdigit() and '.' in name else name
    return base.rsplit('_', 1)[-1]


def sampler_role(sampler, texture):
    """Role of a material sampler: from its name, else the texture suffix."""
    for prefix, role in SAMPLER_ROLES:
        if sampler.startswith(prefix) and sampler[len(prefix):].isdigit():
            return role
    return SUFFIX_ROLES.get(texture_suffix(texture))


def sampler_index(sampler):
    digits = ''.join(c for c in sampler if c.isdigit())
    return int(digits) if digits else 0


def render_info(fmat, name, default=None):
    info = fmat.renderInfo.get(name)
    if not info or not info['vals']:
        return default
    return info['vals'][0]


def _int_option(fmat, name, default):
    try:
        return int(fmat.shaderOptions.get(name, default))
    except (TypeError, ValueError):
        return default


def coords_for_samplers(fmat):
    """Material sampler name -> (UV layer name, tex_srt index or None).

    shader slot N -> texcoord t = uking_textureN_texcoord;
    texcoord t -> UV layer from uking_texcoord{t}_mapping (0 = _u0,
    1 = _u1, ...; 10+ are projections we don't reproduce -> _u0, no SRT)
    and texture transform tex_srt{uking_texcoord{t}_srt} (-1 = none).
    """
    assign = fmat.shader_assign
    slots = [n.name for n in fmat.tex_attribute_dict.nodes][1:]
    values = [fmat.fres.readStr(fmat.fres.read('Q', assign['tex_attr_names'] + i * 8))
              for i in range(assign['num_tex_attrs'])]
    result = {}
    for slot_idx, (slot, sampler) in enumerate(zip(slots, values)):
        if sampler in result:
            continue
        tc = _int_option(fmat, 'uking_texture%d_texcoord' % slot_idx, 0)
        if not 0 <= tc <= 4:
            result[sampler] = ('_u0', None)   # procedural / indirect coordinates
            continue
        mapping = _int_option(fmat, 'uking_texcoord%d_mapping' % tc, 0)
        srt = _int_option(fmat, 'uking_texcoord%d_srt' % tc, -1)
        if 0 <= mapping <= 3:
            result[sampler] = ('_u%d' % mapping, srt if srt >= 0 else None)
        else:
            result[sampler] = ('_u0', None)
    return result


def uv_layers_for_samplers(fmat):
    """Material sampler name -> UV layer name (see coords_for_samplers)."""
    return {k: v[0] for k, v in coords_for_samplers(fmat).items()}


def tex_srt_values(fmat, index):
    """(mode, scale x, scale y, rotation, translate x, translate y) of the
    material's tex_srt<index> parameter, or None."""
    import struct
    p = fmat.materialParams.get('tex_srt%d' % index)
    if not p:
        return None
    data = p['data']
    if isinstance(data, (bytes, bytearray)) and len(data) >= 24:
        return struct.unpack_from('<I5f', data)
    if isinstance(data, (list, tuple)) and len(data) >= 6:
        return tuple(data[:6])
    return None


class UkingMaterialBuilder:
    """Builds the node tree of one uking_mat material."""

    def __init__(self, fmat, mat, operator=None):
        self.fmat = fmat
        self.mat = mat
        self.operator = operator
        self.nodes = mat.node_tree.nodes
        self.links = mat.node_tree.links
        self._uv_nodes = {}
        self._mapping_nodes = []
        self._loose_y = 0

    # ---------------------------------------------------------- helpers

    def _node(self, typ, x, y, label=None):
        n = self.nodes.new(typ)
        n.location = (x, y)
        if label:
            n.label = label
        return n

    def _uv(self, layer):
        if layer not in self._uv_nodes:
            n = self._node('ShaderNodeUVMap', -1300, 300 - 160 * len(self._uv_nodes), layer)
            n.uv_map = layer
            self._uv_nodes[layer] = n
        return self._uv_nodes[layer]

    def _mapping(self, layer, srt):
        """Mapping node applying tex_srt<srt> to UV layer `layer` (shared by
        every texture on that coordinate). Named 'tex_srt<srt>:<layer>' so
        material animations can find and keyframe it."""
        name = 'tex_srt%d:%s' % (srt, layer)
        n = self.nodes.get(name)
        if n is None:
            n = self._node('ShaderNodeMapping', -1050, 300 - 220 * len(self._mapping_nodes),
                           'tex_srt%d' % srt)
            n.name = name
            n.vector_type = 'POINT'
            self.links.new(self._uv(layer).outputs['UV'], n.inputs['Vector'])
            vals = tex_srt_values(self.fmat, srt)
            if vals:
                n['bfres_srt'] = [float(v) for v in vals]  # base for animations
                _mode, sx, sy, rot, tx, ty = vals
                (lx, ly), r, (msx, msy) = animmath.srt_to_mapping(sx, sy, rot, tx, ty)
                n.inputs['Location'].default_value = (float(lx), float(ly), 0.0)
                n.inputs['Rotation'].default_value = (0.0, 0.0, float(r))
                n.inputs['Scale'].default_value = (float(msx), float(msy), 1.0)
            self._mapping_nodes.append(n)
        return n

    def _image(self, image, role, sampler, coord, x, y, wired=True):
        layer, srt = coord
        n = self._node('ShaderNodeTexImage', x, y, '%s (%s)' % (role, sampler))
        n.name = 'sampler:%s' % sampler  # found by texture pattern animations
        n.image = image
        # Images are shared between materials: only switch to Non-Color for
        # data maps that are actually wired, never for loose nodes, so an
        # albedo that also appears unconnected elsewhere keeps its colors.
        if wired and role in DATA_ROLES:
            image.colorspace_settings.name = 'Non-Color'
        if srt is not None:
            self.links.new(self._mapping(layer, srt).outputs['Vector'], n.inputs['Vector'])
        elif layer != '_u0':
            self.links.new(self._uv(layer).outputs['UV'], n.inputs['Vector'])
        return n

    # ---------------------------------------------------------- build

    def build(self, textures):
        """`textures`: list of (material sampler, texture name, image)."""
        nodes, links = self.nodes, self.links
        nodes.clear()
        out = self._node('ShaderNodeOutputMaterial', 400, 300)
        bsdf = self._node('ShaderNodeBsdfPrincipled', 0, 300)
        links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])
        bsdf.inputs['Roughness'].default_value = DEFAULT_ROUGHNESS
        self.bsdf = bsdf

        coord_of = coords_for_samplers(self.fmat)
        by_role = {}
        loose = []
        for sampler, texname, image in textures:
            role = sampler_role(sampler, texname)
            entry = (sampler, texname, image, coord_of.get(sampler, ('_u0', None)))
            # Wire one texture per role. Albedo and normal can come in
            # layers (_a0/_a1, _n0/_n1) blended by the combiner; only the
            # base layer (index 0) is wired, the others stay loose.
            layered = role in ('albedo', 'normal') and sampler_index(sampler) != 0
            if role and role not in by_role and not layered:
                by_role[role] = entry
            else:
                loose.append((role, entry))
        # a material whose only albedo is _a1 (rare) still gets one
        for role in ('albedo', 'normal'):
            if role not in by_role:
                for i, (r, e) in enumerate(loose):
                    if r == role:
                        by_role[role] = e
                        loose.pop(i)
                        break

        self._albedo(by_role.get('albedo'), by_role.get('ao'))
        self._normal(by_role.get('normal'))
        self._specular(by_role.get('specular'))
        self._metalness(by_role.get('metalness'))
        self._emission(by_role.get('emission'), by_role.get('emission_mask'))
        for role, (sampler, texname, image, uv) in loose:
            n = self._image(image, role or 'unused', sampler, uv, -900, -900 - self._loose_y,
                            wired=False)
            n.label = '%s (%s, not connected)' % (role or 'unknown', sampler)
            self._loose_y += 280
            log.debug("%s: texture %s (%s) left unconnected",
                self.fmat.name, texname, sampler)
        self._render_state()

    def _albedo(self, entry, ao):
        if not entry:
            return
        sampler, _, image, uv = entry
        tex = self._image(image, 'albedo', sampler, uv, -700, 500)
        color = tex.outputs['Color']
        if ao:
            a_sampler, _, a_image, a_uv = ao
            ao_tex = self._image(a_image, 'ao', a_sampler, a_uv, -700, 220)
            mix = self._node('ShaderNodeMix', -300, 450, 'albedo x AO')
            mix.data_type = 'RGBA'
            mix.blend_type = 'MULTIPLY'
            mix.inputs['Factor'].default_value = 1.0
            self.links.new(color, mix.inputs['A'])
            self.links.new(ao_tex.outputs['Color'], mix.inputs['B'])
            color = mix.outputs['Result']
        self.links.new(color, self.bsdf.inputs['Base Color'])
        if image.get('bfres_format') in SINGLE_CHANNEL_FORMATS:
            # a one-channel "albedo" (e.g. the bee's wing blur, BC4) has no
            # alpha of its own: its channel is the mask
            self._alpha_source = self._red(tex, -450, 700)
        else:
            self._alpha_source = tex.outputs['Alpha']

    def _normal(self, entry):
        if not entry:
            return
        sampler, _, image, uv = entry
        tex = self._image(image, 'normal', sampler, uv, -700, -350)
        nm = self._node('ShaderNodeNormalMap', -300, -350)
        nm.uv_map = uv[0]
        self.links.new(tex.outputs['Color'], nm.inputs['Color'])
        self.links.new(nm.outputs['Normal'], self.bsdf.inputs['Normal'])

    def _red(self, tex, x, y):
        sep = self._node('ShaderNodeSeparateColor', x, y)
        self.links.new(tex.outputs['Color'], sep.inputs['Color'])
        return sep.outputs['Red']

    def _specular(self, entry):
        if not entry:
            return
        sampler, _, image, uv = entry
        tex = self._image(image, 'specular', sampler, uv, -700, -60)
        # Spm is a specular *mask*: red channel (BC4 gray, or R of BC5)
        scale = self._node('ShaderNodeMath', -250, -60, 'specular mask')
        scale.operation = 'MULTIPLY'
        scale.inputs[1].default_value = SPECULAR_SCALE
        self.links.new(self._red(tex, -450, -60), scale.inputs[0])
        self.links.new(scale.outputs['Value'], self.bsdf.inputs['Specular IOR Level'])

    def _metalness(self, entry):
        if not entry:
            return
        sampler, _, image, uv = entry
        tex = self._image(image, 'metalness', sampler, uv, -700, 100)
        self.links.new(self._red(tex, -450, 100), self.bsdf.inputs['Metallic'])

    def _emission(self, entry, mask):
        if not entry:
            return
        sampler, _, image, uv = entry
        tex = self._image(image, 'emission', sampler, uv, -700, -650)
        color = tex.outputs['Color']
        if mask:
            m_sampler, _, m_image, m_uv = mask
            m_tex = self._image(m_image, 'emission_mask', m_sampler, m_uv, -700, -930)
            mix = self._node('ShaderNodeMix', -300, -700, 'emission x mask')
            mix.data_type = 'RGBA'
            mix.blend_type = 'MULTIPLY'
            mix.inputs['Factor'].default_value = 1.0
            self.links.new(color, mix.inputs['A'])
            self.links.new(m_tex.outputs['Color'], mix.inputs['B'])
            color = mix.outputs['Result']
        self.links.new(color, self.bsdf.inputs['Emission Color'])
        self.bsdf.inputs['Emission Strength'].default_value = 1.0

    _alpha_source = None

    def _render_state(self):
        fmat, mat = self.fmat, self.mat
        mode = render_info(fmat, 'gsys_render_state_mode', 'opaque')
        alpha_test = render_info(fmat, 'gsys_alpha_test_enable', 'false') == 'true'
        threshold = render_info(fmat, 'gsys_alpha_test_value', 0.5)
        blend = render_info(fmat, 'gsys_render_state_blend_mode', 'none')
        try:
            threshold = float(threshold)
        except (TypeError, ValueError):
            threshold = 0.5

        if self._alpha_source is not None and (mode == 'mask' or alpha_test):
            # alpha test 'gequal': keep texels with alpha >= threshold
            cmp = self._node('ShaderNodeMath', -250, 650, 'alpha test')
            cmp.operation = 'GREATER_THAN'
            cmp.inputs[1].default_value = threshold - 1e-4
            self.links.new(self._alpha_source, cmp.inputs[0])
            self.links.new(cmp.outputs['Value'], self.bsdf.inputs['Alpha'])
        elif self._alpha_source is not None and (mode == 'translucent' or blend == 'color'):
            self.links.new(self._alpha_source, self.bsdf.inputs['Alpha'])
            if hasattr(mat, 'surface_render_method'):
                mat.surface_render_method = 'BLENDED'

        face = render_info(fmat, 'gsys_render_state_display_face', 'front')
        mat.use_backface_culling = (face == 'front')
