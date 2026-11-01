"""uking_mat's color combiner, decoded from shader options (no Blender).

uking_mat computes up to 14 intermediate values, "colors" 0..13. Stage N
is enabled by `uking_enable_calc_colorN` and computes
`uking_colorN_calc_type` from four arguments `uking_colorN_A..D`, each with
a `_channel` selector, optionally clamped to 0..1 (`_clamp01`). The
material's outputs (`uking_albedo_color`, `uking_emission_color`,
`uking_transparency_color` + `_channel`, `uking_ao_color` + `_channel`,
`uking_specmask_scaler` + `uking_specular_channel`, `uking_metal_color` +
`_channel`, `uking_normalmap_blend_ratio` + `_channel`) name a source the
same way.

Source codes, worked out from the Breath of the Wild corpus:
    0..15    texture slot N (order of the material's shader sampler
             dictionary, the same N as uking_textureN_*)
    100+N    material parameter const_colorN (horse coat colors, animated
             by the *_fsp shader parameter animations, come in this way)
    200+N    result of stage N (only ever earlier stages)
    300      vertex color _c0 (every material using it has _c0; most use
             its red channel as ambient occlusion)
    400+N    material parameter const_valueN
    500+N    unknown (not supported)
Channel selectors: tens = component (0 all, 1 R, 2 G, 3 B, 4 A), units 1 =
one minus. Other values (2, 33, 1001...) are not understood.

Calc types understood (others are not supported):
    0   A
    2   A * B
    4   mix(A, B, C) = A * (1 - C) + B * C   (layer blends: rock/moss by
        the moss mask, wall A/B by vertex alpha, horse coat colors by the
        pattern mask, mouth texture over the coat by its alpha)

parse() returns an expression tree for a source, or raises Unsupported when
anything in it is outside this subset, so callers can fall back to the
plain texture-role translation.
"""

SUPPORTED_TYPES = {0: 1, 2: 2, 4: 3}  # calc type -> number of arguments used
N_STAGES = 14


class Unsupported(Exception):
    pass


class Tex:
    def __init__(self, slot):
        self.slot = slot

    def __repr__(self):
        return 'Tex(%d)' % self.slot


class VertexColor:
    def __repr__(self):
        return 'VertexColor()'


class ConstColor:
    def __init__(self, index):
        self.index = index

    def __repr__(self):
        return 'ConstColor(%d)' % self.index


class ConstValue:
    def __init__(self, index):
        self.index = index

    def __repr__(self):
        return 'ConstValue(%d)' % self.index


class Channel:
    """Component `comp` (1 R .. 4 A) of `expr`, or all of it (0); `invert`
    for one minus."""

    def __init__(self, expr, comp, invert):
        self.expr, self.comp, self.invert = expr, comp, invert

    @property
    def scalar(self):
        return self.comp != 0

    def __repr__(self):
        return 'Channel(%r, %d%s)' % (self.expr, self.comp, ', inv' if self.invert else '')


class Stage:
    def __init__(self, index, calc_type, args, clamp):
        self.index, self.calc_type, self.args, self.clamp = index, calc_type, args, clamp

    def __repr__(self):
        return 'Stage(%d, type %d, %r%s)' % (self.index, self.calc_type, self.args,
                                             ', clamp' if self.clamp else '')


def _int(opts, name, default=0):
    try:
        return int(opts.get(name, default))
    except (TypeError, ValueError):
        return default


class Combiner:
    """Parses sources of one material.

    opts: shader options (strings); n_slots: number of texture slots, or
    the set of slots that have a texture;
    has_vertex_color: the meshes using the material have _c0;
    const_colors / const_values: indices of the const_colorN /
    const_valueN parameters the material has.
    """

    def __init__(self, opts, n_slots, has_vertex_color, const_colors, const_values):
        self.opts = opts
        # n_slots: a count, or the set of slots that have a texture
        self.bound = set(range(n_slots)) if isinstance(n_slots, int) else set(n_slots)
        self.has_vertex_color = has_vertex_color
        self.const_colors = set(const_colors)
        self.const_values = set(const_values)
        self._stages = {}

    def parse(self, code, channel=0):
        """Expression for source `code` with channel selector `channel`."""
        code, channel = int(code), int(channel)
        comp, unit = divmod(channel, 10)
        if comp > 4 or unit not in (0, 1):
            raise Unsupported('channel %d' % channel)
        expr = self._source(code)
        if comp == 0 and unit == 0:
            return expr
        return Channel(expr, comp, bool(unit))

    def parse_option(self, source_option, channel_option=None):
        return self.parse(_int(self.opts, source_option),
                          _int(self.opts, channel_option) if channel_option else 0)

    def _source(self, code):
        kind, n = divmod(code, 100)
        if kind == 0:
            if n not in self.bound:
                raise Unsupported('texture slot %d not bound' % n)
            return Tex(n)
        if kind == 1:
            if n not in self.const_colors:
                raise Unsupported('const_color%d' % n)
            return ConstColor(n)
        if kind == 2:
            return self.stage(n)
        if kind == 3 and n == 0:
            if not self.has_vertex_color:
                raise Unsupported('no vertex color')
            return VertexColor()
        if kind == 4:
            if n not in self.const_values:
                raise Unsupported('const_value%d' % n)
            return ConstValue(n)
        raise Unsupported('source %d' % code)

    def stage(self, n):
        if n in self._stages:
            return self._stages[n]
        if not 0 <= n < N_STAGES or self.opts.get('uking_enable_calc_color%d' % n) != '1':
            raise Unsupported('stage %d disabled' % n)
        calc = _int(self.opts, 'uking_color%d_calc_type' % n)
        if calc not in SUPPORTED_TYPES:
            raise Unsupported('calc type %d' % calc)
        self._stages[n] = None  # guard against cycles
        try:
            args = [self.parse(_int(self.opts, 'uking_color%d_%s' % (n, a)),
                               _int(self.opts, 'uking_color%d_%s_channel' % (n, a)))
                    for a in 'ABCD'[:SUPPORTED_TYPES[calc]]]
        except Unsupported:
            del self._stages[n]
            raise
        st = Stage(n, calc, args, self.opts.get('uking_color%d_clamp01' % n) == '1')
        self._stages[n] = st
        return st

    def try_parse(self, source_option, channel_option=None):
        """parse_option, or None when unsupported."""
        try:
            return self.parse_option(source_option, channel_option)
        except Unsupported:
            return None


def textures_used(expr, out=None):
    """Texture slots an expression reads."""
    out = set() if out is None else out
    if isinstance(expr, Tex):
        out.add(expr.slot)
    elif isinstance(expr, Channel):
        textures_used(expr.expr, out)
    elif isinstance(expr, Stage):
        for a in expr.args:
            textures_used(a, out)
    return out


def evaluate(expr, textures, vertex_color, const_colors, const_values):
    """Numeric evaluation of an expression (RGBA tuples), for tests and
    for checking the node translation. textures: slot -> RGBA."""
    def ev(e):
        if isinstance(e, Tex):
            return tuple(textures[e.slot])
        if isinstance(e, VertexColor):
            return tuple(vertex_color)
        if isinstance(e, ConstColor):
            return tuple(const_colors[e.index])
        if isinstance(e, ConstValue):
            v = const_values[e.index]
            return (v, v, v, v)
        if isinstance(e, Channel):
            v = ev(e.expr)
            if e.comp:
                v = (v[e.comp - 1],) * 4
            return tuple(1 - x for x in v) if e.invert else v
        if isinstance(e, Stage):
            a = [ev(x) for x in e.args]
            if e.calc_type == 0:
                r = a[0]
            elif e.calc_type == 2:
                r = tuple(x * y for x, y in zip(a[0], a[1]))
            elif e.calc_type == 4:
                r = tuple(x * (1 - c) + y * c for x, y, c in zip(a[0], a[1], a[2]))
            else:
                raise Unsupported('calc type %d' % e.calc_type)
            if e.clamp:
                r = tuple(min(max(x, 0.0), 1.0) for x in r)
            return r
        raise TypeError(e)
    return ev(expr)
