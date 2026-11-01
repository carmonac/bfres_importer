#!/usr/bin/env python3
"""Survey the BRTI texture headers of a whole RomFS (or a corpus folder).

Reads only the headers (Yaz0 files are decompressed just far enough to
reach the embedded BNTX's texture headers), so a full Breath of the Wild
RomFS takes a few minutes. For every texture it records format, size,
mip/array counts, image dimension (2D, cube...), block height, channel
selectors and the mip offsets, and checks the layout the importer computes
(bfres/BNTX/pixelfmt/swizzle.py ``surface_layout``) against the offsets and
data size stored in the file:

    python3 tools/scan_textures.py romfs.img --out /tmp/tex_headers.json
    python3 tools/scan_textures.py ~/Documents/bfres_corpus/botw

Needs numpy (swizzle.py imports it); Blender's own Python has it. The
source is a DMROMFS1 image, an extracted RomFS folder or any folder with
.sbfres/.bfres/.bntx files. ``--grep REGEX`` picks the RomFS paths
(default: every .sbfres, .bntx and Yaz0-compressed texture container).
"""
import argparse
import collections
import json
import os
import re
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

import extract_corpus  # noqa: E402  (RomFS readers)
from bfres.YAZ0.Decoder import _run as _yaz0_run  # noqa: E402
from bfres.BNTX.pixelfmt import swizzle  # noqa: E402
from bfres.BNTX.pixelfmt.formatinfo import formats, bpps, blk_dims  # noqa: E402

DEFAULT_GREP = r'\.(s?bfres|s?bntx|sbmaptex|sbstftex|sbreviewtex|sbitemico)$'
IMG_DIMS = ["1D", "2D", "3D", "Cube", "1D Array", "2D Array", "2D MS",
            "2D MS Array", "Cube Array"]
SEL = "01RGBA"   # channel selector: Zero, One, Red, Green, Blue, Alpha


class Lazy:
    """Bytes of a (possibly Yaz0-compressed) file, decompressed on demand."""

    def __init__(self, raw):
        self.raw = raw
        self.yaz0 = raw[:4] in (b'Yaz0', b'Yaz1')
        self.size = struct.unpack_from('>I', raw, 4)[0] if self.yaz0 else len(raw)
        self.buf = b'' if self.yaz0 else raw

    def need(self, n):
        n = min(n, self.size)
        if len(self.buf) < n:
            want = min(self.size, max(n, 2 * len(self.buf), 1 << 16))
            out = bytearray(want)
            _yaz0_run(memoryview(self.raw), out, want, None)
            self.buf = bytes(out)
        return self.buf

    def read(self, fmt, at):
        b = self.need(at + struct.calcsize(fmt))
        return struct.unpack_from(fmt, b, at)


def bntx_offsets(f):
    """Offsets of the BNTX files in an FRES (or 0 for a bare BNTX)."""
    magic = f.need(8)[:8]
    if magic[:4] == b'BNTX':
        return [0]
    if magic != b'FRES    ':
        return []
    major, minor = f.read('<2H', 8)
    if (major, minor) == (0, 10):
        embed_at, cnt_at = 0xB8, 0xEE
    else:
        embed_at, cnt_at = 0x98, 0xC8
    (embed,), (cnt,) = f.read('<Q', embed_at), f.read('<H', cnt_at)
    out = []
    for i in range(cnt if embed else 0):
        off, size = f.read('<QI', embed + 16 * i)
        if f.read('4s', off)[0] == b'BNTX':
            out.append(off)
    return out


def read_name(f, at):
    (n,) = f.read('<H', at)
    return f.need(at + 2 + n)[at + 2:at + 2 + n].decode('utf-8', 'replace')


def brti(f, base, at):
    h = f.read('<4sIQBBHHHHHBB2xIiiiIII20xII4BiI4xQQ', base + at)
    (magic, _l, _l2, flags, dim, tile, swz, mips, _ms, _r, dtype, ftype,
     _acc, w, hgt, depth, arr, layout, layout2, data_len, align,
     c0, c1, c2, c3, img_dim, name_off, _parent, ptrs) = h
    if magic != b'BRTI':
        raise ValueError("no BRTI at 0x%X" % (base + at))
    mip_ptrs = [f.read('<Q', base + ptrs + 8 * i)[0] for i in range(max(1, mips))]
    fid = (ftype << 8) | dtype
    bw, bh = blk_dims.get(ftype, (1, 1))
    bpp = bpps.get(ftype)
    rec = {
        'name': read_name(f, base + name_off),
        'format': formats.get(fid, '0x%04X' % fid), 'fmt_type': ftype, 'dtype': dtype,
        'width': w, 'height': hgt, 'depth': depth, 'array': arr, 'mips': mips,
        'tile': tile, 'dim': dim, 'img_dim': IMG_DIMS[img_dim] if img_dim < len(IMG_DIMS) else img_dim,
        'flags': flags, 'swizzle': swz, 'layout': layout, 'layout2': layout2,
        'bh_log2': layout & 7, 'data_len': data_len, 'alignment': align,
        'sel': ''.join(SEL[c] if c < 6 else '?' for c in (c0, c1, c2, c3)),
        'mip_offsets': [p - mip_ptrs[0] for p in mip_ptrs],
    }
    if bpp is None:
        rec['check'] = 'unknown format'
        return rec
    ew, eh = swizzle.DIV_ROUND_UP(w, bw), swizzle.DIV_ROUND_UP(hgt, bh)
    rec['bh0_driver'] = swizzle.block_height_mip0(eh).bit_length() - 1
    # level-0 block height: new rule vs. the one used before (BNTX-Editor)
    levels, layer = swizzle.surface_layout(w, hgt, bw, bh, bpp, tile,
                                           layout & 7, mips, arr)
    rec['bh_levels'] = [lv['blockHeightLog2'] for lv in levels]
    old = layout & 7
    if swizzle.pow2_round_up(eh) < (1 << old) * 8:
        old = max(0, old - 1)
    rec['bh0_old'] = old
    rec['layer_size'] = layer
    exp_offs = [lv['offset'] for lv in levels]
    total = layer * arr if arr > 1 else levels[-1]['offset'] + levels[-1]['size']
    rec['expected_len'] = total
    problems = []
    if exp_offs != rec['mip_offsets']:
        problems.append('mip offsets')
    if total != data_len:
        problems.append('data_len %d != %d' % (data_len, total))
    if arr > 1 and data_len % arr:
        problems.append('data_len not a multiple of array')
    rec['check'] = ', '.join(problems) or 'ok'
    return rec


def scan_blob(f, base):
    hdr = f.read('<4s4xIHBBI2xHII', base)
    if hdr[0] != b'BNTX':
        return []
    count, table = f.read('<4sIQ', base + 0x20)[1:]
    out = []
    for i in range(count):
        (at,) = f.read('<Q', base + table + 8 * i)
        out.append(brti(f, base, at))
    return out


def summarize(recs, out=sys.stdout):
    def table(title, counter, n=40):
        print("\n%s" % title, file=out)
        for k, v in counter.most_common(n):
            print("  %6d  %s" % (v, k), file=out)

    print("%d textures in %d files" % (len(recs), len({r['file'] for r in recs})), file=out)
    table("formats", collections.Counter(r['format'] for r in recs))
    table("image dimension", collections.Counter(r['img_dim'] for r in recs))
    table("array layers / depth", collections.Counter(
        'array %d depth %d' % (r['array'], r['depth']) for r in recs))
    table("tile mode", collections.Counter(r['tile'] for r in recs))
    table("mip count", collections.Counter(r['mips'] for r in recs), 20)
    table("channel selectors (format: RGBA source)", collections.Counter(
        '%-18s %s' % (r['format'].rsplit('_', 1)[0], r['sel']) for r in recs))
    table("level-0 block height log2 (header == driver rule)", collections.Counter(
        '%d %s' % (r['bh_log2'], 'same' if r.get('bh0_driver') == r['bh_log2'] else
                   'driver %s' % r.get('bh0_driver')) for r in recs))
    table("old level-0 rule differs from header", collections.Counter(
        'header %d old %d' % (r['bh_log2'], r['bh0_old'])
        for r in recs if r.get('bh0_old', r['bh_log2']) != r['bh_log2']))
    npot = sum(1 for r in recs if (r['width'] & (r['width'] - 1)) or (r['height'] & (r['height'] - 1)))
    print("\nnon-power-of-two: %d" % npot, file=out)
    table("layout check", collections.Counter(r['check'] for r in recs))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("source")
    ap.add_argument("--grep", default=DEFAULT_GREP)
    ap.add_argument("--out", help="write every record as JSON")
    ap.add_argument("--limit", type=int, default=0, help="stop after N files")
    args = ap.parse_args(argv)

    src = os.path.expanduser(args.source)
    fs = extract_corpus.open_romfs(src)
    rx = re.compile(args.grep, re.I)
    names = sorted(n for n in fs.names() if rx.search(n))
    if args.limit:
        names = names[:args.limit]
    recs, errors = [], []
    for k, n in enumerate(names):
        if isinstance(fs, extract_corpus.ImageRomFS):
            off, size = fs.files[n]
            with open(fs.path, 'rb') as fh:
                fh.seek(off)
                raw = fh.read(size)
        else:
            with open(fs.files[n], 'rb') as fh:
                raw = fh.read()
        try:
            f = Lazy(raw)
            for base in bntx_offsets(f):
                for r in scan_blob(f, base):
                    r['file'] = n
                    recs.append(r)
        except Exception as ex:  # keep going: report at the end
            errors.append((n, repr(ex)))
        if k % 500 == 0:
            print("%d/%d files, %d textures" % (k, len(names), len(recs)), file=sys.stderr)
    summarize(recs)
    for n, e in errors:
        print("ERROR %s: %s" % (n, e))
    if args.out:
        with open(args.out, 'w') as fh:
            json.dump({'textures': recs, 'errors': errors}, fh)
    return 1 if errors else 0


if __name__ == '__main__':
    sys.exit(main())
