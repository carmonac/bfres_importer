#!/usr/bin/env python3
"""Find the shader program a BotW material uses, and its translated code.

Breath of the Wild's `uking_mat` material shader is an uber-shader: every
combination of static options (the material's `uking_*` shader options) is
a separate compiled program in `Shader/uking_mat.product.sbfsha` (inside
`Pack/Bootup_Graphics.pack`). This tool:

  * parses the BFSHA (Switch, version 3-6 layout as read by
    KillzXGaming/BfshaLibrary): shading models, static/dynamic options
    with their key bit fields, programs and the key table;
  * finds the program matching a set of options (like
    BfshaLibrary's ShaderOptionSearcher) and its BNSH variation index;
  * maps the variation to the nxnative / NX2Apple Metal translation of its
    pixel shader (`Contents/Resources/dm/shader_map.tsv` ->
    `msl/*/<key>_b.metallib`), whose LLVM IR `xcrun metal-objdump
    --disassemble` prints (macOS with the Metal toolchain only).

The uniform layout of the model (e.g. material parameters const_color0 at
offset N of the material block) is printed by `info`.

Usage:
  uking_shader.py extract <romfs.img> <out.bfsha>
  uking_shader.py info <bfsha> [--model uking_mat] [--options]
  uking_shader.py find <bfsha> <options.json> [--app <Game.app>]
      options.json: {"uking_color0_calc_type": "12", ...} (a material's
      shader options; missing options take their default)

Plain Python 3 (no third-party modules), read-only on the inputs.
"""
import argparse
import json
import os
import struct
import sys


# ---------------------------------------------------------------- archives

def yaz0(b):
    if b[:4] != b'Yaz0':
        return b
    size = struct.unpack('>I', b[4:8])[0]
    out = bytearray(size)
    i, o = 16, 0
    while o < size:
        code = b[i]
        i += 1
        for _ in range(8):
            if o >= size:
                break
            if code & 0x80:
                out[o] = b[i]
                i += 1
                o += 1
            else:
                b1, b2 = b[i], b[i + 1]
                i += 2
                dist = ((b1 & 0xF) << 8 | b2) + 1
                n = b1 >> 4
                if n == 0:
                    n = b[i] + 0x12
                    i += 1
                else:
                    n += 2
                for _ in range(n):
                    out[o] = out[o - dist]
                    o += 1
            code <<= 1
    return bytes(out)


def sarc(data):
    """{name: bytes} of a SARC archive."""
    if data[:4] != b'SARC':
        raise ValueError('not a SARC')
    hl, bom = struct.unpack_from('<HH', data, 4)
    E = '<' if bom == 0xFEFF else '>'
    _size, doff = struct.unpack_from(E + 'II', data, 8)
    _shl, n, _hm = struct.unpack_from(E + 'HHI', data, hl + 4)
    nodes = [struct.unpack_from(E + 'IIII', data, hl + 12 + i * 16) for i in range(n)]
    names = hl + 12 + n * 16 + 8
    out = {}
    for _h, attr, s, e in nodes:
        name = None
        if attr & 0x01000000:
            no = names + (attr & 0xFFFF) * 4
            name = data[no:data.index(b'\0', no)].decode()
        out[name] = data[doff + s:doff + e]
    return out


def extract(romfs, out_path):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from extract_corpus import open_romfs
    import tempfile
    fs = open_romfs(romfs)
    with tempfile.TemporaryDirectory() as tmp:
        pack = os.path.join(tmp, 'pack')
        fs.copy('Pack/Bootup_Graphics.pack', pack)
        with open(pack, 'rb') as f:
            files = sarc(yaz0(f.read()))
    data = yaz0(files['Shader/uking_mat.product.sbfsha'])
    with open(out_path, 'wb') as f:
        f.write(data)
    return len(data)


# ---------------------------------------------------------------- BFSHA

class Reader:
    def __init__(self, data):
        self.d = data
        self.major = struct.unpack_from('<H', data, 0xA)[0]

    def u8(self, o): return self.d[o]
    def u16(self, o): return struct.unpack_from('<H', self.d, o)[0]
    def i16(self, o): return struct.unpack_from('<h', self.d, o)[0]
    def u32(self, o): return struct.unpack_from('<I', self.d, o)[0]
    def i32(self, o): return struct.unpack_from('<i', self.d, o)[0]
    def u64(self, o): return struct.unpack_from('<Q', self.d, o)[0]

    def string(self, off):
        if not off:
            return ''
        e = self.d.index(b'\0', off + 2)
        return self.d[off + 2:e].decode('utf-8', 'replace')

    def dict_keys(self, off):
        """Keys of a _DIC in value order."""
        if not off:
            return []
        n = self.i32(off + 4)
        return [self.string(self.u64(off + 8 + i * 16 + 8)) for i in range(1, n + 1)]


class Option:
    def __init__(self, r, o):
        self.name = r.string(r.u64(o))
        self.choices = r.dict_keys(r.u64(o + 8))
        if r.major >= 9:
            raise NotImplementedError('BFSHA v9 options')
        self.count = r.u8(o + 24)
        self.default = r.u8(o + 25)
        self.block_offset = r.u16(o + 27)
        self.key_offset = r.u8(o + 29)
        self.index = r.u8(o + 30)
        self.shift = r.u8(o + 31)
        self.mask = r.u32(o + 32)
    SIZE = 40

    def choice_of(self, key):
        return (key & self.mask) >> self.shift

    def set(self, keys, base, choice):
        k = keys[base] & ~self.mask & 0xFFFFFFFF
        keys[base] = k | ((choice << self.shift) & self.mask)


class Uniform:
    def __init__(self, r, o, name):
        self.name = name
        self.index = r.i32(o + 8)
        self.offset = r.u16(o + 12)
        self.block = r.u8(o + 14)


class Block:
    def __init__(self, r, o, name):
        self.name = name
        arr, dic, default = r.u64(o), r.u64(o + 8), r.u64(o + 16)
        self.index, self.type, self.size, n = r.u8(o + 24), r.u8(o + 25), r.u16(o + 26), r.u16(o + 28)
        names = r.dict_keys(dic)
        self.uniforms = [Uniform(r, arr + i * 16, names[i] if i < len(names) else '?')
                         for i in range(n)]
        self.default = r.d[default:default + self.size] if default else b''
    SIZE = 32


class Program:
    def __init__(self, r, o, shader_file):
        if r.major >= 7:
            raise NotImplementedError('BFSHA v7+ programs')
        self.sampler_table = r.u64(o)
        self.block_table = r.u64(o + 8)
        self.variation_offset = r.u64(o + 16)
        self.num_samplers = r.u16(o + 38)
        self.num_blocks = r.u16(o + 40)
        self.variation = (self.variation_offset - (shader_file + 192)) // 64
        loc = 16  # vertex, geometry, fragment, compute (int32 each)
        self.sampler_locations = [struct.unpack_from('<4i', r.d, self.sampler_table + i * loc)
                                  for i in range(self.num_samplers)]
        self.block_locations = [struct.unpack_from('<4i', r.d, self.block_table + i * loc)
                                for i in range(self.num_blocks)]
    SIZE_V5 = 0x30  # v4/v5 header size (v5 adds nothing that is read here)


class ShaderModel:
    def __init__(self, r, o):
        self.r = r
        self.name = r.string(r.u64(o))
        def dict_at(p):
            return r.u64(p), r.u64(p + 8)  # values, dict
        st_v, st_d = dict_at(o + 8)
        dy_v, dy_d = dict_at(o + 24)
        at_v, at_d = dict_at(o + 40)
        sa_v, sa_d = dict_at(o + 56)
        p = o + 72
        if r.major >= 8:
            p += 16
        ub_v, ub_d = dict_at(p)
        p += 16
        _uniform_array = r.u64(p); p += 8
        if r.major >= 7:
            p += 24
        prog_arr, key_table, _archive, _symbols, shader_file = (r.u64(p + i * 8) for i in range(5))
        p += 5 * 8 + 3 * 8
        if r.major >= 7:
            p += 16
        _uniform_count = r.u32(p); p += 4
        if r.major >= 7:
            p += 4
        self.default_program = r.i32(p); p += 4
        n_static, n_dynamic, n_programs = r.u16(p), r.u16(p + 2), r.u16(p + 4); p += 6
        if r.major < 7:
            p += 2
        self.static_key_len, self.dynamic_key_len = r.u8(p), r.u8(p + 1); p += 2
        _n_attr, n_samplers = r.u8(p), r.u8(p + 1); p += 2
        if r.major >= 8:
            p += 1
        n_blocks = r.u8(p)
        self.shader_file = shader_file

        self.static = [Option(r, st_v + i * Option.SIZE) for i in range(n_static)]
        self.dynamic = [Option(r, dy_v + i * Option.SIZE) for i in range(n_dynamic)]
        self.samplers = r.dict_keys(sa_d)
        self.attributes = r.dict_keys(at_d)
        names = r.dict_keys(ub_d)
        self.blocks = [Block(r, ub_v + i * Block.SIZE, names[i]) for i in range(n_blocks)]
        self.programs = [Program(r, prog_arr + i * Program.SIZE_V5, shader_file)
                         for i in range(n_programs)]
        nk = self.static_key_len + self.dynamic_key_len
        self.keys = list(struct.unpack_from('<%di' % (nk * n_programs), r.d, key_table))

    # ------------------------------------------------------------ lookup

    def program_static(self, i):
        """{option: choice} of program i (static options)."""
        nk = self.static_key_len + self.dynamic_key_len
        base = nk * i
        out = {}
        for opt in self.static:
            c = opt.choice_of(self.keys[base + opt.index])
            out[opt.name] = opt.choices[c] if c < len(opt.choices) else '?%d' % c
        return out

    def program_dynamic(self, i):
        nk = self.static_key_len + self.dynamic_key_len
        base = nk * i + self.static_key_len
        out = {}
        for opt in self.dynamic:
            c = opt.choice_of(self.keys[base + opt.index - opt.key_offset])
            out[opt.name] = opt.choices[c] if c < len(opt.choices) else '?%d' % c
        return out

    def find(self, options, dynamic=None):
        """Indices of the programs whose static options equal `options`
        (missing ones take the option's default), and dynamic ones equal
        `dynamic` where given."""
        nk = self.static_key_len + self.dynamic_key_len
        want = [0] * self.static_key_len
        for opt in self.static:
            choice = opt.default
            v = options.get(opt.name)
            if v is not None:
                v = str(v)
                if v not in opt.choices:
                    raise KeyError('%s: no choice %r (%s)' % (opt.name, v, opt.choices[:8]))
                choice = opt.choices.index(v)
            opt.set(want, opt.index, choice)
        found = []
        for i in range(len(self.programs)):
            if self.keys[nk * i:nk * i + self.static_key_len] != want:
                continue
            if dynamic:
                d = self.program_dynamic(i)
                if any(d.get(k) != str(v) for k, v in dynamic.items()):
                    continue
            found.append(i)
        return found


def load(path):
    with open(path, 'rb') as f:
        data = f.read()
    r = Reader(data)
    if data[:4] != b'FSHA':
        raise ValueError('not a BFSHA')
    models_values, models_dict = r.u64(0x48), r.u64(0x50)
    names = r.dict_keys(models_dict)
    models = {}
    # shading models are stored back to back; BotW's file has one
    for i, name in enumerate(names):
        if i:
            raise NotImplementedError('several shading models')
        models[name] = ShaderModel(r, models_values)
    return r, models


# ---------------------------------------------------------------- nxnative

def metallib_for(app, bnsh_offset, variation, stage='pixel'):
    """Path of the Metal library nxnative made for one BNSH variation."""
    res = os.path.join(app, 'Contents', 'Resources')
    tag = '#BNSH@0x%x#var%d' % (bnsh_offset, variation)
    key = None
    with open(os.path.join(res, 'dm', 'shader_map.tsv')) as f:
        for line in f:
            cols = line.rstrip('\n').split('\t')
            if len(cols) >= 5 and cols[2] == stage and cols[4].endswith(tag) \
                    and 'uking_mat.product' in cols[4]:
                key = cols[3]
                break
    if key is None:
        return None
    msl = os.path.join(res, 'msl')
    for d in sorted(os.listdir(msl)):
        p = os.path.join(msl, d, key + '_b.metallib')
        if os.path.exists(p):
            return p
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    e = sub.add_parser('extract')
    e.add_argument('romfs')
    e.add_argument('out')
    i = sub.add_parser('info')
    i.add_argument('bfsha')
    i.add_argument('--options', action='store_true')
    f = sub.add_parser('find')
    f.add_argument('bfsha')
    f.add_argument('options')
    f.add_argument('--app')
    args = ap.parse_args(argv)

    if args.cmd == 'extract':
        print(extract(args.romfs, args.out), 'bytes')
        return 0
    r, models = load(args.bfsha)
    sm = models['uking_mat']
    if args.cmd == 'info':
        print('version', r.major, 'programs', len(sm.programs), 'static', len(sm.static),
              'dynamic', len(sm.dynamic), 'keys', sm.static_key_len, sm.dynamic_key_len,
              'shader file 0x%x' % sm.shader_file)
        print('samplers', sm.samplers)
        for b in sm.blocks:
            print('block', b.index, b.name, 'size', b.size,
                  ', '.join('%s@%d' % (u.name, u.offset) for u in b.uniforms))
        if args.options:
            for o in sm.static:
                print('static', o.name, o.choices, 'default', o.default)
            for o in sm.dynamic:
                print('dynamic', o.name, o.choices, 'default', o.default)
        return 0
    with open(args.options) as fh:
        opts = json.load(fh)
    progs = sm.find(opts)
    for p in progs:
        prog = sm.programs[p]
        lib = metallib_for(args.app, sm.shader_file, prog.variation) if args.app else None
        print('program', p, 'variation', prog.variation, 'dynamic', sm.program_dynamic(p), lib)
    return 0 if progs else 1


if __name__ == '__main__':
    sys.exit(main())
