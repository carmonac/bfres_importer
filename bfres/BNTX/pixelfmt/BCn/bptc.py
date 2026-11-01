"""Vectorized BC6H and BC7 (BPTC) block decoders (numpy).

Pure-Python counterparts of the native decoder (texture2ddecoder), used:
  * for BC6H always, because the native decoder only gives 8-bit LDR output
    while BC6H is an HDR format (half floats);
  * for BC7 when the native module is not available.

Both take raw, already deswizzled block data and the image size in pixels,
like the functions in decode.py:
  bc6h -> float32 array (height, width, 4), RGB as decoded, alpha 1
  bc7  -> bytes, RGBA8, width*height*4

Blocks are grouped by mode and every group is decoded at once. The bits of
a block are unpacked into a (blocks, 128) array, so any field, wherever a
mode puts it, is a gather.

Bit layouts and arithmetic follow the D3D11 / Khronos BPTC specification;
the partition tables are those of the specification (as also used by
bcdec, https://github.com/iOrange/bcdec, MIT / Unlicense).
"""
import numpy as np

from .decode import _blocks, _assemble

_P2 = (
    '0011001100110011', '0001000100010001', '0111011101110111', '0001001100110111',
    '0000000100010011', '0011011101111111', '0001001101111111', '0000000100110111',
    '0000000000010011', '0011011111111111', '0000000101111111', '0000000000010111',
    '0001011111111111', '0000000011111111', '0000111111111111', '0000000000001111',
    '0000100011101111', '0111000100000000', '0000000010001110', '0111001100010000',
    '0011000100000000', '0000100011001110', '0000000010001100', '0111001100110001',
    '0011000100010000', '0000100010001100', '0110011001100110', '0011011001101100',
    '0001011111101000', '0000111111110000', '0111000110001110', '0011100110011100',
    '0101010101010101', '0000111100001111', '0101101001011010', '0011001111001100',
    '0011110000111100', '0101010110101010', '0110100101101001', '0101101010100101',
    '0111001111001110', '0001001111001000', '0011001001001100', '0011101111011100',
    '0110100110010110', '0011110011000011', '0110011010011001', '0000011001100000',
    '0100111001000000', '0010011100100000', '0000001001110010', '0000010011100100',
    '0110110010010011', '0011011011001001', '0110001110011100', '0011100111000110',
    '0110110011001001', '0110001100111001', '0111111010000001', '0001100011100111',
    '0000111100110011', '0011001111110000', '0010001011101110', '0100010001110111',
)
_P3 = (
    '0011001102212222', '0001001122112221', '0000200122112211', '0222002200110111',
    '0000000011221122', '0011001100220022', '0022002211111111', '0011001122112211',
    '0000000011112222', '0000111111112222', '0000111122222222', '0012001200120012',
    '0112011201120112', '0122012201220122', '0011011211221222', '0011200122002220',
    '0001001101121122', '0111001120012200', '0000112211221122', '0022002200221111',
    '0111011102220222', '0001000122212221', '0000001101220122', '0000110022102210',
    '0122012200110000', '0012001211222222', '0110122112210110', '0000011012211221',
    '0022110211020022', '0110011020022222', '0011012201220011', '0000200022112221',
    '0000000211221222', '0222002200120011', '0011001200220222', '0120012001200120',
    '0000111122220000', '0120120120120120', '0120201212010120', '0011220011220011',
    '0011112222000011', '0101010122222222', '0000000021212121', '0022112200221122',
    '0022001100220011', '0220122102201221', '0101222222220101', '0000212121212121',
    '0101010101012222', '0222011102220111', '0002111200021112', '0000211221122112',
    '0222011101110222', '0002111211120002', '0110011001102222', '0000000021122112',
    '0110011022222222', '0022001100110022', '0022112211220022', '0000000000002112',
    '0002000100020001', '0222122202221222', '0101222222222222', '0111201122012220',
)
# second-subset anchor (fix-up) pixel of each 2-subset partition
_A2 = (
    15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
    15,  2,  8,  2,  2,  8,  8, 15,  2,  8,  2,  2,  8,  8,  2,  2,
    15, 15,  6,  8,  2,  8, 15, 15,  2,  8,  2,  2,  2, 15, 15,  6,
     6,  2,  6,  8, 15, 15,  2,  2, 15, 15, 15, 15, 15,  2,  2, 15,
)
# anchor pixels of subsets 1 and 2 of each 3-subset partition
_A3 = (
    (3, 15), (3, 8), (8, 15), (3, 15), (8, 15), (3, 15), (3, 15), (8, 15),
    (8, 15), (8, 15), (6, 15), (6, 15), (6, 15), (5, 15), (3, 15), (3, 8),
    (3, 15), (3, 8), (8, 15), (3, 15), (3, 15), (3, 8), (6, 15), (8, 10),
    (3, 5), (8, 15), (6, 8), (6, 10), (8, 15), (5, 15), (10, 15), (8, 15),
    (8, 15), (3, 15), (3, 15), (5, 10), (6, 10), (8, 10), (8, 9), (10, 15),
    (6, 15), (3, 15), (8, 15), (5, 15), (3, 15), (6, 15), (6, 15), (8, 15),
    (3, 15), (3, 15), (5, 15), (5, 15), (5, 15), (8, 15), (5, 15), (10, 15),
    (5, 15), (10, 15), (8, 15), (13, 15), (3, 15), (12, 15), (3, 15), (3, 8),
)


def _maps(rows):
    return np.array([[int(c) for c in r] for r in rows], np.intp)

P2 = _maps(_P2)
P3 = _maps(_P3)


def _index_layout(anchors, bits):
    """Per-pixel (offset, width) of an index field, for each partition.

    `anchors[p]` lists the fix-up pixels of partition p (pixel 0 is always
    one), which are stored with one bit less.
    """
    n = len(anchors)
    width = np.full((n, 16), bits, np.intp)
    width[:, 0] -= 1
    for p, a in enumerate(anchors):
        for px in a:
            width[p, px] -= 1
    off = np.zeros((n, 16), np.intp)
    off[:, 1:] = np.cumsum(width, 1)[:, :-1]
    return off, width


def _bits(blk):
    return np.unpackbits(blk.reshape(-1, 16), axis=1, bitorder='little')


def _field(bits, start, n):
    """n-bit unsigned field at a fixed bit position -> int32 per block."""
    v = np.zeros(bits.shape[0], np.int32)
    for k in range(n):
        v |= bits[:, start + k].astype(np.int32) << k
    return v


def _gather(bits, off, width):
    """Variable fields: off/width (blocks, 16) relative bit positions."""
    v = np.zeros(off.shape, np.int32)
    for k in range(int(width.max())):
        b = np.take_along_axis(bits, np.minimum(off + k, 127), 1).astype(np.int32)
        v |= np.where(k < width, b, 0) << k
    return v


def _interp(a, b, w):
    return (a * (64 - w) + b * w + 32) >> 6


W2 = np.array((0, 21, 43, 64), np.int32)
W3 = np.array((0, 9, 18, 27, 37, 46, 55, 64), np.int32)
W4 = np.array((0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64), np.int32)


# ------------------------------------------------------------------ BC7

# mode: subsets, partition bits, rotation bits, index selection bit,
#       color bits, alpha bits, p-bits (0 none, 1 per endpoint, 2 shared
#       per subset), index bits, secondary index bits
_BC7 = (
    (3, 4, 0, 0, 4, 0, 1, 3, 0),
    (2, 6, 0, 0, 6, 0, 2, 3, 0),
    (3, 6, 0, 0, 5, 0, 0, 2, 0),
    (2, 6, 0, 0, 7, 0, 1, 2, 0),
    (1, 0, 2, 1, 5, 6, 0, 2, 3),
    (1, 0, 2, 0, 7, 8, 0, 2, 2),
    (1, 0, 0, 0, 7, 7, 1, 4, 0),
    (2, 6, 0, 0, 5, 5, 1, 2, 0),
)

_ANCHORS = {1: [()], 2: [(a,) for a in _A2], 3: [tuple(a) for a in _A3]}
_SUBSETS = {1: np.zeros((1, 16), np.intp), 2: P2, 3: P3}


def _bc7_mode(bits, mode):
    ns, pb, rb, isb, cb, ab, pbits, ib, ib2 = _BC7[mode]
    n = bits.shape[0]
    pos = mode + 1
    part = _field(bits, pos, pb) if pb else np.zeros(n, np.int32)
    pos += pb
    rot = _field(bits, pos, rb) if rb else np.zeros(n, np.int32)
    pos += rb
    isel = _field(bits, pos, isb) if isb else np.zeros(n, np.int32)
    pos += isb

    ne = 2 * ns
    ep = np.zeros((n, ne, 4), np.int32)
    for c in range(3):
        for e in range(ne):
            ep[:, e, c] = _field(bits, pos, cb)
            pos += cb
    if ab:
        for e in range(ne):
            ep[:, e, 3] = _field(bits, pos, ab)
            pos += ab

    cprec, aprec = cb, ab
    if pbits:
        ep <<= 1
        if pbits == 2:      # one p-bit per subset, shared by both endpoints
            for s in range(ns):
                p = bits[:, pos + s].astype(np.int32)
                ep[:, 2 * s:2 * s + 2, :] |= p[:, None, None]
            pos += ns
        else:
            for e in range(ne):
                ep[:, e, :] |= bits[:, pos + e].astype(np.int32)[:, None]
            pos += ne
        cprec += 1
        aprec += 1
    ep[..., :3] = (ep[..., :3] << (8 - cprec)) | (ep[..., :3] >> (2 * cprec - 8))
    if ab:
        ep[..., 3] = (ep[..., 3] << (8 - aprec)) | (ep[..., 3] >> (2 * aprec - 8))
    else:
        ep[..., 3] = 255

    off, width = _index_layout(_ANCHORS[ns], ib)
    part_i = part.astype(np.intp)
    idx = _gather(bits, pos + off[part_i], width[part_i])
    pos += ib * 16 - ns

    subset = _SUBSETS[ns][part_i]                         # (n, 16)
    lo = np.take_along_axis(ep, (2 * subset)[..., None], 1)        # (n, 16, 4)
    hi = np.take_along_axis(ep, (2 * subset + 1)[..., None], 1)

    wtab = {2: W2, 3: W3, 4: W4}
    w1 = wtab[ib][idx]
    if ib2:
        off2, width2 = _index_layout([()], ib2)
        idx2 = _gather(bits, pos + np.broadcast_to(off2[0], (n, 16)),
                       np.broadcast_to(width2[0], (n, 16)))
        w2 = wtab[ib2][idx2]
        sel = (isel == 1)[:, None]
        wc = np.where(sel, w2, w1)
        wa = np.where(sel, w1, w2)
    else:
        wc = wa = w1
    out = np.empty((n, 16, 4), np.int32)
    out[..., :3] = _interp(lo[..., :3], hi[..., :3], wc[..., None])
    out[..., 3] = _interp(lo[..., 3], hi[..., 3], wa)

    for r in (1, 2, 3):     # swap alpha with R, G or B
        m = rot == r
        if m.any():
            c = r - 1
            t = out[m, :, c].copy()
            out[m, :, c] = out[m, :, 3]
            out[m, :, 3] = t
    return out


def bc7(data, width, height):
    blk = _blocks(data, width, height, 16)
    bh, bw = blk.shape[:2]
    flat = blk.reshape(-1, 16)
    b0 = flat[:, 0]
    # mode = number of trailing zero bits of the first byte (0 = reserved)
    mode = np.full(b0.shape, 8, np.intp)
    for m in range(7, -1, -1):
        mode[(b0 >> m) & 1 == 1] = m
    out = np.zeros((flat.shape[0], 16, 4), np.int32)   # reserved -> 0
    bits = _bits(flat)
    for m in range(8):
        sel = mode == m
        if sel.any():
            out[sel] = _bc7_mode(bits[sel], m)
    tex = out.astype(np.uint8).reshape(bh, bw, 16, 4)
    return _assemble(tex, width, height).tobytes()


# ------------------------------------------------------------------ BC6H

# Header bit layouts, in stored order after the mode bits. "rw:0-9" is
# bits 0..9 of rw stored LSB first; "rw:15-10" is stored MSB first.
# w/x/y/z are the endpoints A0, B0, A1, B1; d is the partition index.
_BC6_LAYOUT = {
    0x00: "gy:4 by:4 bz:4 rw:0-9 gw:0-9 bw:0-9 rx:0-4 gz:4 gy:0-3 gx:0-4 bz:0 "
          "gz:0-3 bx:0-4 bz:1 by:0-3 ry:0-4 bz:2 rz:0-4 bz:3 d:0-4",
    0x01: "gy:5 gz:4 gz:5 rw:0-6 bz:0 bz:1 by:4 gw:0-6 by:5 bz:2 gy:4 bw:0-6 "
          "bz:3 bz:5 bz:4 rx:0-5 gy:0-3 gx:0-5 gz:0-3 bx:0-5 by:0-3 ry:0-5 "
          "rz:0-5 d:0-4",
    0x02: "rw:0-9 gw:0-9 bw:0-9 rx:0-4 rw:10 gy:0-3 gx:0-3 gw:10 bz:0 gz:0-3 "
          "bx:0-3 bw:10 bz:1 by:0-3 ry:0-4 bz:2 rz:0-4 bz:3 d:0-4",
    0x06: "rw:0-9 gw:0-9 bw:0-9 rx:0-3 rw:10 gz:4 gy:0-3 gx:0-4 gw:10 gz:0-3 "
          "bx:0-3 bw:10 bz:1 by:0-3 ry:0-3 bz:0 bz:2 rz:0-3 gy:4 bz:3 d:0-4",
    0x0A: "rw:0-9 gw:0-9 bw:0-9 rx:0-3 rw:10 by:4 gy:0-3 gx:0-3 gw:10 bz:0 "
          "gz:0-3 bx:0-4 bw:10 by:0-3 ry:0-3 bz:1 bz:2 rz:0-3 bz:4 bz:3 d:0-4",
    0x0E: "rw:0-8 by:4 gw:0-8 gy:4 bw:0-8 bz:4 rx:0-4 gz:4 gy:0-3 gx:0-4 bz:0 "
          "gz:0-3 bx:0-4 bz:1 by:0-3 ry:0-4 bz:2 rz:0-4 bz:3 d:0-4",
    0x12: "rw:0-7 gz:4 by:4 gw:0-7 bz:2 gy:4 bw:0-7 bz:3 bz:4 rx:0-5 gy:0-3 "
          "gx:0-4 bz:0 gz:0-3 bx:0-4 bz:1 by:0-3 ry:0-5 rz:0-5 d:0-4",
    0x16: "rw:0-7 bz:0 by:4 gw:0-7 gy:5 gy:4 bw:0-7 gz:5 bz:4 rx:0-4 gz:4 "
          "gy:0-3 gx:0-5 gz:0-3 bx:0-4 bz:1 by:0-3 ry:0-4 bz:2 rz:0-4 bz:3 d:0-4",
    0x1A: "rw:0-7 bz:1 by:4 gw:0-7 by:5 gy:4 bw:0-7 bz:5 bz:4 rx:0-4 gz:4 "
          "gy:0-3 gx:0-4 bz:0 gz:0-3 bx:0-5 by:0-3 ry:0-4 bz:2 rz:0-4 bz:3 d:0-4",
    0x1E: "rw:0-5 gz:4 bz:0 bz:1 by:4 gw:0-5 gy:5 by:5 bz:2 gy:4 bw:0-5 gz:5 "
          "bz:3 bz:5 bz:4 rx:0-5 gy:0-3 gx:0-5 gz:0-3 bx:0-5 by:0-3 ry:0-5 "
          "rz:0-5 d:0-4",
    0x03: "rw:0-9 gw:0-9 bw:0-9 rx:0-9 gx:0-9 bx:0-9",
    0x07: "rw:0-9 gw:0-9 bw:0-9 rx:0-8 rw:10 gx:0-8 gw:10 bx:0-8 bw:10",
    0x0B: "rw:0-9 gw:0-9 bw:0-9 rx:0-7 rw:11-10 gx:0-7 gw:11-10 bx:0-7 bw:11-10",
    0x0F: "rw:0-9 gw:0-9 bw:0-9 rx:0-3 rw:15-10 gx:0-3 gw:15-10 bx:0-3 bw:15-10",
}
# mode -> (endpoint precision, delta bits R, G, B, transformed)
_BC6_PREC = {
    0x00: (10, 5, 5, 5, True), 0x01: (7, 6, 6, 6, True),
    0x02: (11, 5, 4, 4, True), 0x06: (11, 4, 5, 4, True),
    0x0A: (11, 4, 4, 5, True), 0x0E: (9, 5, 5, 5, True),
    0x12: (8, 6, 5, 5, True), 0x16: (8, 5, 6, 5, True),
    0x1A: (8, 5, 5, 6, True), 0x1E: (6, 6, 6, 6, False),
    0x03: (10, 10, 10, 10, False), 0x07: (11, 9, 9, 9, True),
    0x0B: (12, 8, 8, 8, True), 0x0F: (16, 4, 4, 4, True),
}


def _parse_layout(spec):
    """-> list of (endpoint, channel, bit) in stored order, or 'd' entries."""
    out = []
    for tok in spec.split():
        name, rng = tok.split(':')
        if '-' in rng:
            a, b = (int(v) for v in rng.split('-'))
            seq = range(a, b + 1) if a <= b else range(a, b - 1, -1)
        else:
            seq = (int(rng),)
        for bit in seq:
            if name == 'd':
                out.append((-1, 0, bit))
            else:
                out.append(('wxyz'.index(name[1]), 'rgb'.index(name[0]), bit))
    return out

_BC6_FIELDS = {m: _parse_layout(s) for m, s in _BC6_LAYOUT.items()}


def _sext(v, bits):
    s = 32 - bits
    return (v << s) >> s


def _bc6_unquantize(v, prec, signed):
    if not signed:
        if prec >= 15:
            return v
        u = ((v << 16) + 0x8000) >> prec
        u = np.where(v == 0, 0, u)
        return np.where(v == (1 << prec) - 1, 0xFFFF, u)
    if prec >= 16:
        return v
    neg = v < 0
    a = np.abs(v)
    u = ((a << 15) + 0x4000) >> (prec - 1)
    u = np.where(a == 0, 0, u)
    u = np.where(a >= (1 << (prec - 1)) - 1, 0x7FFF, u)
    return np.where(neg, -u, u)


def _bc6_finish(v, signed):
    """Interpolated value -> half float bit pattern (uint16)."""
    if not signed:
        return ((v * 31) >> 6).astype(np.uint16)
    neg = v < 0
    m = (np.abs(v) * 31) >> 5
    return (np.where(neg, 0x8000, 0) | m).astype(np.uint16)


def _bc6_mode(bits, mode, signed):
    prec, dr, dg, db, transformed = _BC6_PREC[mode]
    two = mode not in (0x03, 0x07, 0x0B, 0x0F)
    n = bits.shape[0]
    ep = np.zeros((n, 4, 3), np.int32)
    part = np.zeros(n, np.int32)
    pos = 2 if mode < 2 else 5
    for e, c, bit in _BC6_FIELDS[mode]:
        b = bits[:, pos].astype(np.int32) << bit
        if e < 0:
            part |= b
        else:
            ep[:, e, c] |= b
        pos += 1
    ne = 4 if two else 2
    ep = ep[:, :ne]
    dbits = (dr, dg, db)
    if signed:
        ep[:, 0] = _sext(ep[:, 0], prec)
    if transformed or signed:
        for c in range(3):
            ep[:, 1:, c] = _sext(ep[:, 1:, c], dbits[c])
    if transformed:
        mask = (1 << prec) - 1
        ep[:, 1:] = (ep[:, 1:] + ep[:, :1]) & mask
        if signed:
            ep[:, 1:] = _sext(ep[:, 1:], prec)
    ep = _bc6_unquantize(ep, prec, signed)

    if two:
        part_i = part.astype(np.intp)
        off, width = _index_layout(_ANCHORS[2][:32], 3)
        idx = _gather(bits, pos + off[part_i], width[part_i])
        subset = P2[:32][part_i]
        w = W3[idx]
    else:
        off, width = _index_layout([()], 4)
        idx = _gather(bits, pos + np.broadcast_to(off[0], (n, 16)),
                      np.broadcast_to(width[0], (n, 16)))
        subset = np.zeros((n, 16), np.intp)
        w = W4[idx]
    lo = np.take_along_axis(ep, (2 * subset)[..., None], 1)
    hi = np.take_along_axis(ep, (2 * subset + 1)[..., None], 1)
    return _bc6_finish(_interp(lo, hi, w[..., None]), signed)


def bc6h(data, width, height, signed=False):
    blk = _blocks(data, width, height, 16)
    bh, bw = blk.shape[:2]
    flat = blk.reshape(-1, 16)
    b0 = flat[:, 0].astype(np.intp)
    mode = np.where((b0 & 3) < 2, b0 & 3, b0 & 0x1F)
    half = np.zeros((flat.shape[0], 16, 3), np.uint16)   # reserved -> 0
    bits = _bits(flat)
    for m in _BC6_PREC:
        sel = mode == m
        if sel.any():
            half[sel] = _bc6_mode(bits[sel], m, signed)
    rgb = half.view(np.float16).astype(np.float32)
    rgba = np.concatenate((rgb, np.ones(rgb.shape[:-1] + (1,), np.float32)), -1)
    return _assemble(rgba.reshape(bh, bw, 16, 4), width, height)
