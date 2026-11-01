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

The meaning of the codes was read from the game's own compiled shaders:
Shader/uking_mat.product.sbfsha has one program per combination of these
options; tools/uking_shader.py finds a material's program and the Metal
translation the NX2Apple port made of it, whose LLVM IR shows the
arithmetic (see DEVELOPMENT.md). Uniforms of the material parameter block
(gsys_material) appear there as loads at their byte offsets, textures as
samples of the material's samplers.

Sources:
    0..7     texture of shader sampler slot N, in the shader's fixed order
             _a0 _s0 _n0 _e0 _t0 _a1 array0 array1 (TEXTURE_SLOTS), whatever
             material sampler the material assigns to it
    8, 9     engine buffers (8 is the deferred lighting result): unsupported
    100-103  const_color0..3        112-115  const_color4..7
    104-111  const_value0..7
    200+N    result of stage N (only ever earlier stages)
    300      vertex color _c0
    400-403  the constants 0, 0.5, 1, 2
    502      rim term from the vertex normal: clamp(1 + N.V), 0 facing the
             camera, 1 at grazing angles (Blender: Layer Weight > Facing)
    503      the same with the normal-mapped normal
    504      view depth (Blender: Camera Data > View Z Depth)
    500 (depth difference with the scene behind), 501, 505..518, 404..406:
             unsupported
Channel selectors: tens = component (0 all, 1 R, 2 G, 3 B, 4 A); units 0 =
as is, 1 = one minus, 2 = negated. Other values (3, 1000s) are unsupported.

Calc types (CALC): see the table below; others are unsupported. Arguments a
type doesn't use are often set to the first texture or 0 by the tools.
"""

TEXTURE_SLOTS = ('_a0', '_s0', '_n0', '_e0', '_t0', '_a1', 'array0', 'array1')
LITERALS = {400: 0.0, 401: 0.5, 402: 1.0, 403: 2.0}
RIM_VERTEX, RIM_NORMAL, VIEW_DEPTH = 502, 503, 504
N_STAGES = 14

# calc type -> (arguments used, formula); formulas are per component
CALC = {
    0: ('A', 'A'),
    1: ('AB', 'A + B'),
    2: ('AB', 'A * B'),
    4: ('ABC', 'mix(A, B, C)'),
    5: ('ABC', '(A + B) * C'),
    6: ('ABCD', 'C * D - A * B'),
    7: ('AB', 'pow(A, B)'),
    8: ('ABC', 'A * B + C'),
    9: ('ABC', 'A * B * C'),
    11: ('ABCD', 'A * B * C * D'),
    12: ('ABCD', 'mix(A, B, C) * D'),
    15: ('ABCD', 'mix(A, B, C) + D'),
    21: ('ABCD', '(A * B + C) * D'),
    28: ('ABCD', 'pow(A, B) * C + D'),
}


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


class Literal:
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return 'Literal(%g)' % self.value


class Builtin:
    """An engine-computed value: RIM_VERTEX, RIM_NORMAL or VIEW_DEPTH."""

    def __init__(self, code):
        self.code = code

    def __repr__(self):
        return 'Builtin(%d)' % self.code


class Channel:
    """Component `comp` (1 R .. 4 A) of `expr`, or all of it (0), then
    `mod`: 0 as is, 1 one minus, 2 negated."""

    def __init__(self, expr, comp, mod):
        self.expr, self.comp, self.mod = expr, comp, mod

    @property
    def invert(self):
        return self.mod == 1

    def __repr__(self):
        return 'Channel(%r, %d, %d)' % (self.expr, self.comp, self.mod)


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

    opts: shader options (strings); bound: the texture slots (indices into
    TEXTURE_SLOTS) that have a texture, or their count;
    has_vertex_color: the meshes using the material have _c0;
    const_colors / const_values: indices of the const_colorN /
    const_valueN parameters the material has.
    """

    def __init__(self, opts, bound, has_vertex_color, const_colors, const_values):
        self.opts = opts
        self.bound = set(range(bound)) if isinstance(bound, int) else set(bound)
        self.has_vertex_color = has_vertex_color
        self.const_colors = set(const_colors)
        self.const_values = set(const_values)
        self._stages = {}

    def parse(self, code, channel=0):
        """Expression for source `code` with channel selector `channel`."""
        code, channel = int(code), int(channel)
        comp, mod = divmod(channel, 10)
        if comp > 4 or mod not in (0, 1, 2):
            raise Unsupported('channel %d' % channel)
        expr = self._source(code)
        if comp == 0 and mod == 0:
            return expr
        return Channel(expr, comp, mod)

    def parse_option(self, source_option, channel_option=None):
        return self.parse(_int(self.opts, source_option),
                          _int(self.opts, channel_option) if channel_option else 0)

    def _source(self, code):
        if 0 <= code < len(TEXTURE_SLOTS):
            if code not in self.bound:
                raise Unsupported('texture slot %d (%s) not bound' % (code, TEXTURE_SLOTS[code]))
            return Tex(code)
        if 100 <= code <= 115:
            n = code - 100
            if 4 <= n <= 11:
                if n - 4 not in self.const_values:
                    raise Unsupported('const_value%d' % (n - 4))
                return ConstValue(n - 4)
            index = n if n < 4 else n - 8
            if index not in self.const_colors:
                raise Unsupported('const_color%d' % index)
            return ConstColor(index)
        if 200 <= code < 200 + N_STAGES:
            return self.stage(code - 200)
        if code == 300:
            if not self.has_vertex_color:
                raise Unsupported('no vertex color')
            return VertexColor()
        if code in LITERALS:
            return Literal(LITERALS[code])
        if code in (RIM_VERTEX, RIM_NORMAL, VIEW_DEPTH):
            return Builtin(code)
        raise Unsupported('source %d' % code)

    def stage(self, n):
        if n in self._stages:
            if self._stages[n] is None:
                raise Unsupported('stage %d refers to itself' % n)
            return self._stages[n]
        if not 0 <= n < N_STAGES or self.opts.get('uking_enable_calc_color%d' % n) != '1':
            raise Unsupported('stage %d disabled' % n)
        calc = _int(self.opts, 'uking_color%d_calc_type' % n)
        if calc not in CALC:
            raise Unsupported('calc type %d' % calc)
        self._stages[n] = None  # guard against cycles
        try:
            args = [self.parse(_int(self.opts, 'uking_color%d_%s' % (n, a)),
                               _int(self.opts, 'uking_color%d_%s_channel' % (n, a)))
                    for a in CALC[calc][0]]
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


def _mix(a, b, c):
    return a * (1 - c) + b * c


OPS = {
    0: lambda A: A,
    1: lambda A, B: A + B,
    2: lambda A, B: A * B,
    4: lambda A, B, C: _mix(A, B, C),
    5: lambda A, B, C: (A + B) * C,
    6: lambda A, B, C, D: C * D - A * B,
    7: lambda A, B: A ** B if A > 0 else 0.0,
    8: lambda A, B, C: A * B + C,
    9: lambda A, B, C: A * B * C,
    11: lambda A, B, C, D: A * B * C * D,
    12: lambda A, B, C, D: _mix(A, B, C) * D,
    15: lambda A, B, C, D: _mix(A, B, C) + D,
    21: lambda A, B, C, D: (A * B + C) * D,
    28: lambda A, B, C, D: (A ** B if A > 0 else 0.0) * C + D,
}


def evaluate(expr, textures, vertex_color, const_colors, const_values, builtins=None):
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
        if isinstance(e, Literal):
            return (e.value,) * 4
        if isinstance(e, Builtin):
            return ((builtins or {})[e.code],) * 4
        if isinstance(e, Channel):
            v = ev(e.expr)
            if e.comp:
                v = (v[e.comp - 1],) * 4
            if e.mod == 1:
                v = tuple(1 - x for x in v)
            elif e.mod == 2:
                v = tuple(-x for x in v)
            return v
        if isinstance(e, Stage):
            a = [ev(x) for x in e.args]
            op = OPS[e.calc_type]
            r = tuple(op(*comps) for comps in zip(*a))
            if e.clamp:
                r = tuple(min(max(x, 0.0), 1.0) for x in r)
            return r
        raise TypeError(e)
    return ev(expr)


class _Fill(dict):
    """Any texture slot -> the same RGBA."""

    def __init__(self, value):
        super().__init__()
        self.value = (value,) * 4

    def __missing__(self, key):
        return self.value


def peak(expr, const_colors, const_values, texture_max=None):
    """Brightest RGB component an expression reaches with every texture at
    its brightest (texture_max: slot -> RGBA maxima, default 1) and vertex
    color and engine values at 1 (and everything at 0): an estimate of how
    far above 1 an HDR value (emission of the moon, glows) goes.
    const_colors / const_values: index -> RGBA / float."""
    best = 0.0
    for fill in (1.0, 0.0):
        tex = _Fill(fill)
        if fill and texture_max:
            tex.update({k: tuple(v) for k, v in texture_max.items()})
        try:
            v = evaluate(expr, tex, (fill,) * 4, const_colors, const_values,
                         {code: fill for code in range(500, 520)})
        except (KeyError, ValueError, ZeroDivisionError, OverflowError, TypeError):
            continue
        best = max(best, max(abs(x) for x in v[:3]))
    return best
