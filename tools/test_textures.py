#!/usr/bin/env python3
"""Synthetic BNTX with every block-compressed format, and a check of it.

Breath of the Wild only uses BC1-BC5 (and a few uncompressed formats), so
BC6H, BC7 and ASTC are tested with textures made here, from generated
images (no game data):

1. ``make`` (plain Python; needs ``pip install numpy imagecodecs etcpak
   astc-encoder-py``) encodes test images with real encoders (etcpak: BC1,
   BC3, BC4, BC5, BC7; astcenc: ASTC; a small BC6H encoder below), writes
   them as block-linear (swizzled) textures into OUT.bntx, and the pixels a
   reference decoder gives (bcdec through imagecodecs, astcenc) into
   OUT.npz::

       python3 tools/test_textures.py make /tmp/tex_synth/synth

2. ``check`` (inside Blender) imports OUT.bntx with the add-on and compares
   every image with the reference, plus its float buffer / colorspace::

       blender -b --factory-startup --python tools/test_textures.py -- \\
           check /tmp/tex_synth/synth.bntx

   The exit code is non-zero when an image is off by more than the
   tolerance: 0 for BC4/BC5/BC6H/BC7, 1/255 for BC1-BC3 (bcdec rounds the
   interpolated colors differently) and ASTC (against astcenc).

3. ``corpus`` (any Python with numpy, e.g. Blender's) decodes every texture
   of the models in a folder with the native and with the Python decoders,
   and reports the largest pixel difference and the decode times::

       python3 tools/test_textures.py corpus ~/Documents/bfres_corpus/botw

4. ``levels`` (any Python with numpy) decodes every mip level and array
   layer of every texture in a folder and compares each level with a 2x2
   box reduction of the one above it: a level deswizzled with the wrong
   block height comes out scrambled, far from the reduction (mean
   difference > 0.1)::

       python3 tools/test_textures.py levels ~/Documents/bfres_corpus/botw
"""
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

UNORM, SNORM, SRGB, SFLOAT, UFLOAT = 1, 2, 6, 5, 10
ASTC_SIZES = ((4, 4), (5, 4), (5, 5), (6, 5), (6, 6), (8, 5), (8, 6), (8, 8),
              (10, 5), (10, 6), (10, 8), (10, 10), (12, 10), (12, 12))


# ---------------------------------------------------------------- make

def test_image(w, h):
    import numpy as np
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    r = x / max(w - 1, 1)
    g = y / max(h - 1, 1)
    b = 0.5 + 0.5 * np.sin(x * 0.4) * np.cos(y * 0.3)
    a = 1 - 0.7 * ((x // 8 + y // 8) % 2)
    return (np.stack((r, g, b, a), -1) * 255 + 0.5).astype(np.uint8)


def pad4(img):
    import numpy as np
    h, w = img.shape[:2]
    ph, pw = -h % 4, -w % 4
    return np.pad(img, ((0, ph), (0, pw), (0, 0)), mode='edge')


def bc6h_encode(rgb, signed):
    """Tiny BC6H encoder: mode 11 (one region, 10-bit endpoints) only.

    Good enough to make valid HDR test blocks; the reference decoder, not
    this, defines the expected pixels."""
    import numpy as np
    h, w = rgb.shape[:2]
    half = rgb.astype(np.float16).view(np.uint16).astype(np.int64)
    if signed:   # sign-magnitude half -> signed 10-bit endpoint value
        mag = half & 0x7FFF
        val = np.where(half & 0x8000, -1, 1) * np.minimum(mag // 62, 511)
    else:
        val = np.minimum(half // 31, 1023)
    out = bytearray()
    for by in range(0, h, 4):
        for bx in range(0, w, 4):
            p = val[by:by + 4, bx:bx + 4].reshape(16, 3)
            lo, hi = p.min(0), p.max(0)
            d = hi - lo
            t = ((p - lo) * d).sum(1) / max(int((d * d).sum()), 1)
            idx = np.clip(np.rint(t * 15), 0, 15).astype(int)
            if idx[0] > 7:          # anchor index has 3 bits: swap endpoints
                lo, hi, idx = hi, lo, 15 - idx
            bits, n = 0x03, 5
            for v in (*lo, *hi):
                bits |= (int(v) & 0x3FF) << n
                n += 10
            bits |= int(idx[0]) << n
            n += 3
            for i in idx[1:]:
                bits |= int(i) << n
                n += 4
            out += bits.to_bytes(16, 'little')
    return bytes(out)


def swizzle(data, width, height, bw, bh, bpp, block_height_log2):
    """Linear blocks of one level -> Tegra block-linear, the way BRTI reads it."""
    import numpy as np
    sys.path.insert(0, ROOT)
    from bfres.BNTX.pixelfmt.swizzle import getAddrBlockLinear, DIV_ROUND_UP, mip_size
    ew, eh = DIV_ROUND_UP(width, bw), DIV_ROUND_UP(height, bh)
    block_height = 1 << block_height_log2
    size = mip_size(ew, eh, bpp, 0, block_height)
    y, x = np.meshgrid(np.arange(eh, dtype=np.int64), np.arange(ew, dtype=np.int64), indexing='ij')
    pos = getAddrBlockLinear(x, y, ew, bpp, 0, block_height)
    src = np.frombuffer(data, np.uint8)[:ew * eh * bpp].reshape(eh, ew, bpp)
    surf = np.zeros(size, np.uint8)
    surf[pos[..., None] + np.arange(bpp)] = src
    return surf.tobytes()


def swizzle_surface(layers, width, height, bw, bh, bpp, layout):
    """layers[layer][level] linear data -> (surface, level offsets), laid
    out like the console's driver does (bfres/BNTX/pixelfmt/swizzle.py
    surface_layout: per-level block height, aligned array layers)."""
    sys.path.insert(0, ROOT)
    from bfres.BNTX.pixelfmt.swizzle import surface_layout
    levels, layer_size = surface_layout(width, height, bw, bh, bpp, 0, layout & 7,
                                        len(layers[0]), len(layers))
    out = bytearray(layer_size * len(layers) if len(layers) > 1 else
                    levels[-1]['offset'] + levels[-1]['size'])
    for k, layer in enumerate(layers):
        for lv, data in zip(levels, layer):
            surf = swizzle(data, lv['width'], lv['height'], bw, bh, bpp, lv['blockHeightLog2'])
            at = k * layer_size + lv['offset']
            out[at:at + len(surf)] = surf
    return bytes(out), [lv['offset'] for lv in levels]


def build_textures():
    """-> list of (name, fmt_type, dtype, w, h, block dims, data, ref, compare)."""
    import numpy as np
    import etcpak
    import imagecodecs
    import astc_encoder as ae
    texs = []

    def bcn(name, ftype, dtype, w, h, enc, ref_fmt, ref_ch, compare):
        img = pad4(test_image(w, h))
        ph, pw = img.shape[:2]
        data = enc(img, pw, ph)
        ref = imagecodecs.bcn_decode(data, ref_fmt, shape=(ph, pw, ref_ch) if ref_ch > 1 else (ph, pw))
        ref = ref.reshape(ph, pw, -1)[:h, :w]
        texs.append((name, ftype, dtype, w, h, (4, 4), data, ref, compare))

    def c(fn):
        return lambda img, w, h: getattr(etcpak, fn)(np.ascontiguousarray(img).tobytes(), w, h)

    def bc2(img, w, h):
        col = etcpak.compress_bc1(np.ascontiguousarray(img).tobytes(), w, h)
        out = bytearray()
        a = img[..., 3] >> 4
        k = 0
        for by in range(0, h, 4):
            for bx in range(0, w, 4):
                nib = a[by:by + 4, bx:bx + 4].ravel()
                out += bytes((int(nib[i]) | (int(nib[i + 1]) << 4)) for i in range(0, 16, 2))
                out += col[k * 8:k * 8 + 8]
                k += 1
        return bytes(out)

    # BC1-BC3: bcdec rounds the 1/3 and 2/3 colors differently, off by 1
    bcn('T_BC1_UNORM', 0x1a, UNORM, 64, 64, c('compress_bc1'), 1, 4, 'rgba~1')
    bcn('T_BC1_SRGB', 0x1a, SRGB, 60, 44, c('compress_bc1'), 1, 4, 'rgba~1')
    bcn('T_BC2_UNORM', 0x1b, UNORM, 64, 32, bc2, 2, 4, 'rgba~1')
    bcn('T_BC3_SRGB', 0x1c, SRGB, 64, 64, c('compress_bc3'), 3, 4, 'rgba~1')
    bcn('T_BC3_UNORM', 0x1c, UNORM, 50, 30, c('compress_bc3'), 3, 4, 'rgba~1')
    bcn('T_BC4_UNORM', 0x1d, UNORM, 64, 64, c('compress_bc4'), 4, 1, 'r')
    bcn('T_BC5_UNORM_Nrm', 0x1e, UNORM, 64, 64, c('compress_bc5'), 5, 2, 'rg')
    bcn('T_BC7_UNORM', 0x20, UNORM, 64, 64, c('compress_bc7'), 7, 4, 'rgba')
    bcn('T_BC7_SRGB', 0x20, SRGB, 50, 30, c('compress_bc7'), 7, 4, 'rgba')

    # signed BC4/BC5: imagecodecs has no reference; covered by unit tests
    rng = np.random.default_rng(7)
    for name, ftype, bs in (('T_BC4_SNORM', 0x1d, 8), ('T_BC5_SNORM', 0x1e, 16)):
        texs.append((name, ftype, SNORM, 32, 32, (4, 4),
                     rng.integers(0, 256, 64 * bs, np.uint8).tobytes(), None, None))

    # BC6H: HDR gradient up to 16 (unsigned) / -8..8 (signed)
    w = h = 64
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    hdr = np.stack((x / 4, y / 8, 0.25 + np.sin(x * 0.2) ** 2 * 3), -1)
    for name, dtype, signed, img in (('T_BC6H_UF16', UFLOAT, False, hdr),
                                     ('T_BC6H_SF16', SFLOAT, True, hdr - 8)):
        data = bc6h_encode(img, signed)
        ref = imagecodecs.bcn_decode(data, -6 if signed else 6, shape=(h, w, 3)).astype(np.float32)
        texs.append((name, 0x1f, dtype, w, h, (4, 4), data, ref, 'rgb'))

    # ASTC, every block size; 40x30 is not a multiple of most of them
    w, h = 40, 30
    img = test_image(w, h)
    for i, (bw, bh) in enumerate(ASTC_SIZES):
        dtype = SRGB if (bw, bh) == (4, 4) else UNORM
        prof = ae.ASTCProfile.LDR_SRGB if dtype == SRGB else ae.ASTCProfile.LDR
        ctx = ae.ASTCContext(ae.ASTCConfig(prof, bw, bh, 1, 60))
        sw = ae.ASTCSwizzle.from_str('RGBA')
        data = ctx.compress(ae.ASTCImage(ae.ASTCType.U8, w, h, 1, img.tobytes()), sw)
        # astcenc decodes sRGB blocks to linear; the importer keeps the
        # stored (sRGB-encoded) values, so the reference is the LDR decode
        rctx = ae.ASTCContext(ae.ASTCConfig(ae.ASTCProfile.LDR, bw, bh, 1, 60))
        ref = np.frombuffer(rctx.decompress(data, ae.ASTCImage(ae.ASTCType.U8, w, h, 1), sw).data,
                            np.uint8).reshape(h, w, 4)
        texs.append(('T_ASTC%dx%d' % (bw, bh), 0x2d + i, dtype, w, h, (bw, bh), data, ref, 'rgba~1'))
    return texs


def write_bntx(path, texs):
    """Minimal BNTX: what bfres/BNTX reads, nothing more.

    Each texture is (name, fmt_type, dtype, w, h, (bw, bh), data, ref,
    compare[, opts]); `opts` may give 'levels' (list of array layers, each a
    list of linear mip levels, replacing `data`), 'bpp', 'layout' (block
    height log2, default: the driver's choice), 'sel' (four channel
    selectors, default RGBA) and 'img_dim' (default 1 = 2D; 3 = cube).
    """
    sys.path.insert(0, ROOT)
    from bfres.BNTX.pixelfmt.swizzle import block_height_mip0, DIV_ROUND_UP
    names = [b'synth'] + [t[0].encode() for t in texs]
    strtab = bytearray()
    str_offs = []
    STR_AT = 0x50
    pos = STR_AT + 0x14
    for n in names:
        if pos & 1:
            strtab += b'\0'
            pos += 1
        str_offs.append(pos)
        strtab += struct.pack('<H', len(n)) + n + b'\0'
        pos += 2 + len(n) + 1
    strhdr = b'_STR' + bytes(4) + struct.pack('<I', 0x14 + len(strtab)) + bytes(4) + struct.pack('<I', len(names))
    head_end = STR_AT + len(strhdr) + len(strtab)
    table_at = (head_end + 7) & ~7
    brti_at = table_at + 8 * len(texs)

    surfs, sizes = [], []
    for t in texs:
        name, ftype, dtype, w, h, (bw, bh), data, _ref, _cmp = t[:9]
        opts = t[9] if len(t) > 9 else {}
        bpp = opts.get('bpp', 16 if ftype not in (0x1a, 0x1d) else 8)
        layers = opts.get('levels', [[data]])
        layout = opts.get('layout', block_height_mip0(DIV_ROUND_UP(h, bh)).bit_length() - 1)
        surf, offs = swizzle_surface(layers, w, h, bw, bh, bpp, layout)
        surfs.append((surf, offs, layout, len(layers), opts))
        sizes.append(0x78 + 8 * len(offs))
    brti_offs = []
    at = brti_at
    for size in sizes:
        brti_offs.append(at)
        at += size
    data_at = (at + 0x1FF) & ~0x1FF

    blobs, brtis, d = [], [], data_at
    for k, t in enumerate(texs):
        name, ftype, dtype, w, h, (bw, bh), data, _ref, _cmp = t[:9]
        surf, offs, layout, nlayers, opts = surfs[k]
        sel = opts.get('sel', (2, 3, 4, 5))
        at = brti_offs[k]
        hdr = struct.pack('<4sIQBBHHHHHBB2xIiiiIII20xII4BiI4xQQ',
            b'BRTI', sizes[k], sizes[k], 1, 2, 0, 0, len(offs), 1, 0, dtype, ftype,
            0x20, w, h, 1, nlayers, layout, 0x10007, len(surf), 0x200,
            *sel, opts.get('img_dim', 1), str_offs[k + 1], 0x20, at + 0x78)
        assert len(hdr) == 0x78
        brtis.append(hdr + b''.join(struct.pack('<Q', d + o) for o in offs))
        blobs.append((d, surf))
        d = (d + len(surf) + 0x1FF) & ~0x1FF
    size = d
    out = bytearray(size)
    out[0:0x20] = struct.pack('<4s4xIHBBI2xHII', b'BNTX', 0x00040000, 0xFEFF, 0x0C, 0x40,
                              str_offs[0] + 2, STR_AT, size, size)
    out[0x20:0x44] = struct.pack('<4sIQQQI', b'NX  ', len(texs), table_at, data_at, 0, 0)
    out[STR_AT:head_end] = strhdr + strtab
    for k in range(len(texs)):
        out[table_at + 8 * k:table_at + 8 * k + 8] = struct.pack('<Q', brti_offs[k])
        out[brti_offs[k]:brti_offs[k] + sizes[k]] = brtis[k]
    for at, surf in blobs:
        out[at:at + len(surf)] = surf
    if path is None:
        return bytes(out)
    with open(path, 'wb') as f:
        f.write(out)


def make(out):
    import numpy as np
    texs = build_textures()
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    write_bntx(out + '.bntx', texs)
    refs = {}
    for name, ftype, dtype, w, h, blk, data, ref, cmp in texs:
        if ref is not None:
            refs[name] = ref
            refs[name + '.compare'] = np.array(cmp)
    np.savez_compressed(out + '.npz', **refs)
    print("%d textures -> %s.bntx, reference %s.npz" % (len(texs), out, out))


# ---------------------------------------------------------------- check

def check(path):
    import bpy
    import numpy as np
    sys.path.insert(0, HERE)
    import import_test as harness
    harness.reset_scene()
    bpy.ops.import_scene.nxbfres(filepath=path)
    refs = np.load(os.path.splitext(path)[0] + '.npz')
    bad = 0
    print("%-18s %-7s %-6s %-5s %-16s %s" % ("image", "format", "decod", "float", "colorspace", "max diff"))
    for img in sorted(bpy.data.images, key=lambda i: i.name):
        if not img.name.startswith('T_'):
            continue
        w, h = img.size
        px = np.empty(w * h * 4, np.float32)
        img.pixels.foreach_get(px)
        px = np.flipud(px.reshape(h, w, 4))
        err = img.get('bfres_decode_error')
        diff = '-'
        if err:
            bad += 1
            diff = 'ERROR ' + err
        elif img.name in refs:
            ref = refs[img.name].astype(np.float32)
            cmp = str(refs[img.name + '.compare'])
            tol = 1 if cmp.endswith('~1') else 0
            ch = {'rgba': 4, 'rgba~1': 4, 'rgb': 3, 'rg': 2, 'r': 1}[cmp]
            got = px[..., :ch]
            if not img.is_float:
                got = np.rint(got * 255)
                ref = ref[..., :ch]
            else:
                ref = ref[..., :ch]
            d = float(np.abs(got - ref).max())
            diff = '%g' % d + ('' if img.is_float else '/255')
            if d > tol:
                bad += 1
                diff += '  FAIL'
        print("%-18s %-7s %-6s %-5s %-16s %s" % (img.name, img.get('bfres_format'),
              img.get('bfres_decoder'), img.is_float, img.colorspace_settings.name, diff))
    print("%d problems" % bad)
    return bad


# ---------------------------------------------------------------- corpus

def _bntx_blobs(data):
    """Every BNTX inside a (decompressed) file."""
    at = data.find(b'BNTX')
    while at >= 0:
        if data[at + 0xC:at + 0xE] in (b'\xff\xfe', b'\xfe\xff'):
            size = struct.unpack_from('<I', data, at + 0x1C)[0]
            yield data[at:at + size]
        at = data.find(b'BNTX', at + 4)


def corpus(folder):
    import io
    import logging
    import time
    import numpy as np
    sys.path.insert(0, ROOT)
    logging.basicConfig(level=logging.ERROR)
    from bfres import YAZ0
    from bfres.BinaryFile import BinaryFile
    from bfres.BNTX import BNTX
    from bfres.BNTX.pixelfmt import native
    if native.get() is None:
        print("native decoder not available")
        return 1
    print("native decoder: %s" % native.source)
    total = {'native': 0.0, 'python': 0.0}
    worst, ntex, fails, native_only = 0.0, 0, 0, 0
    per_fmt = {}
    for fn in sorted(os.listdir(folder)):
        if not fn.endswith(('.sbfres', '.bfres', '.bntx')):
            continue
        with open(os.path.join(folder, fn), 'rb') as f:
            data = f.read()
        if data[:4] in (b'Yaz0', b'Yaz1'):
            data = YAZ0.decompress(data)
        for blob in _bntx_blobs(data):
            bio = io.BytesIO(blob)
            bio.name = fn
            bntx = BNTX(BinaryFile(bio))
            bntx.decode()     # decodes once already (native)
            for tex in bntx.textures:
                ntex += 1
                fmt = tex.format_
                name = type(fmt).__name__
                out = {}
                for mode in ('native', 'python'):
                    native.set_enabled(mode == 'native')
                    try:
                        t0 = time.perf_counter()
                        raw = fmt.decompress(tex)
                        dt = time.perf_counter() - t0
                        out[mode] = (np.asarray(fmt.decodePixels(raw), np.float32),
                                     getattr(fmt, 'decoder', None) or 'python')
                    except Exception as ex:
                        out[mode] = (None, 'ERROR %s' % ex)
                        dt = 0.0
                    total[mode] += dt
                    pf = per_fmt.setdefault(name, {'n': 0, 'native': 0.0, 'python': 0.0, 'diff': 0.0})
                    pf[mode] += dt
                native.set_enabled(True)
                per_fmt[name]['n'] += 1
                a, b = out['native'][0], out['python'][0]
                if a is not None and b is None and name.startswith('ASTC'):
                    native_only += 1      # no Python ASTC decoder
                    continue
                if a is None or b is None:
                    fails += 1
                    print("FAIL %s %s %s: %s / %s" % (fn, tex.name, name, out['native'][1], out['python'][1]))
                    continue
                d = float(np.abs(a - b).max()) if a.shape == b.shape else float('inf')
                per_fmt[name]['diff'] = max(per_fmt[name]['diff'], d)
                worst = max(worst, d)
    print("%-10s %6s %10s %10s %14s" % ("format", "count", "native s", "python s", "max diff"))
    for name, pf in sorted(per_fmt.items()):
        print("%-10s %6d %10.3f %10.3f %14s" % (name, pf['n'], pf['native'], pf['python'],
              "%.6f (%g/255)" % (pf['diff'], round(pf['diff'] * 255, 3))))
    print("%d textures (%d ASTC: native only), %d failures, max diff %g/255, "
          "decode time native %.2fs, python %.2fs" % (ntex, native_only, fails,
          round(worst * 255, 3), total['native'], total['python']))
    return 1 if fails else 0


# ---------------------------------------------------------------- levels

def _reduce(px, w, h):
    """2x2 box reduction of (H, W, 4) pixels to (h, w, 4)."""
    import numpy as np
    H, W = px.shape[:2]
    if H < 2 * h:
        px = np.repeat(px, 2, 0)
    if W < 2 * w:
        px = np.repeat(px, 2, 1)
    px = px[:2 * h, :2 * w]
    return px.reshape(h, 2, w, 2, 4).mean((1, 3))


def levels(folder, limit=0.1):
    import io
    import logging
    import numpy as np
    sys.path.insert(0, ROOT)
    logging.basicConfig(level=logging.ERROR)
    from bfres import YAZ0
    from bfres.BinaryFile import BinaryFile
    from bfres.BNTX import BNTX
    ntex = nlev = fails = bad = 0
    worst = (0.0, None)
    for fn in sorted(os.listdir(folder)):
        if not fn.endswith(('.sbfres', '.bfres', '.bntx')):
            continue
        with open(os.path.join(folder, fn), 'rb') as f:
            data = f.read()
        if data[:4] in (b'Yaz0', b'Yaz1'):
            data = YAZ0.decompress(data)
        for blob in _bntx_blobs(data):
            bio = io.BytesIO(blob)
            bio.name = fn
            bntx = BNTX(BinaryFile(bio))
            bntx.decode()
            for tex in bntx.textures:
                ntex += 1
                hdr = getattr(tex.format_, 'hdr', False)
                for layer in range(tex.layerCount):
                    prev = None
                    for m in range(tex.mipCount):
                        nlev += 1
                        try:
                            px = tex.rgba(m, layer)
                        except Exception as ex:
                            fails += 1
                            print("FAIL %s %s level %d layer %d: %s" % (fn, tex.name, m, layer, ex))
                            break
                        if prev is not None and not hdr:
                            d = float(np.abs(_reduce(prev, px.shape[1], px.shape[0]) - px).mean())
                            if d > worst[0]:
                                worst = (d, "%s %s level %d layer %d" % (fn, tex.name, m, layer))
                            if d > limit:
                                bad += 1
                                print("SUSPECT %s %s level %d layer %d (%dx%d): mean diff %.3f"
                                      % (fn, tex.name, m, layer, px.shape[1], px.shape[0], d))
                        prev = px
    print("%d textures, %d levels, %d failures, %d suspect levels; largest mean "
          "difference to the reduced level above: %.4f (%s)" % (ntex, nlev, fails, bad,
          worst[0], worst[1]))
    return 1 if fails or bad else 0


def main():
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:]
    if len(argv) != 2 or argv[0] not in ('make', 'check', 'corpus', 'levels'):
        print(__doc__)
        return 2
    if argv[0] == 'make':
        make(argv[1])
        return 0
    if argv[0] == 'corpus':
        return corpus(os.path.expanduser(argv[1]))
    if argv[0] == 'levels':
        return levels(os.path.expanduser(argv[1]))
    return 1 if check(argv[1]) else 0


if __name__ == '__main__':
    code = main()
    if 'bpy' in sys.modules:
        import bpy
        if bpy.app.background:
            sys.exit(code)
    else:
        sys.exit(code)
