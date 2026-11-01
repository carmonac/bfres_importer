"""Vectorized BC1-BC5 block decoders (numpy).

Every function takes the raw, already deswizzled block data and the image
size in pixels. Blocks are decoded all at once into a padded image whose
size is a multiple of 4, which is then cropped to `width` x `height`, so
textures whose size is not a multiple of 4 come out right.

Outputs match what the TextureFormat classes in this package expect:
  bc1 / bc2 / bc3 -> bytes, RGBA8, width*height*4
  bc4             -> bytes, one channel, width*height
  bc5             -> uint8 array of shape (2, width*height)
Signed (SNORM) BC4/BC5 values are returned offset by +128.
"""
import numpy as np


def _blocks(data, width, height, block_size):
    bw = (width + 3) // 4
    bh = (height + 3) // 4
    need = bw * bh * block_size
    buf = np.frombuffer(bytes(data[:need]), dtype=np.uint8)
    if buf.size < need:
        raise ValueError("BC data too short: %d bytes for %dx%d (need %d)"
            % (buf.size, width, height, need))
    return buf.reshape(bh, bw, block_size)


def _assemble(texels, width, height):
    """(bh, bw, 16, C) block texels -> (height, width, C) image."""
    bh, bw, _, c = texels.shape
    img = texels.reshape(bh, bw, 4, 4, c).transpose(0, 2, 1, 3, 4)
    return img.reshape(bh * 4, bw * 4, c)[:height, :width]


def _u16(blk, i):
    return blk[..., i].astype(np.uint32) | (blk[..., i + 1].astype(np.uint32) << 8)


def _color_block(blk, four_color_only):
    """Decode the 8-byte color part of BC1/2/3 blocks -> (bh, bw, 16, 4)."""
    c0 = _u16(blk, 0)
    c1 = _u16(blk, 2)
    bits = (blk[..., 4].astype(np.uint32)
        | (blk[..., 5].astype(np.uint32) << 8)
        | (blk[..., 6].astype(np.uint32) << 16)
        | (blk[..., 7].astype(np.uint32) << 24))

    def rgb565(c):
        r = (c >> 11) & 0x1F
        g = (c >> 5) & 0x3F
        b = c & 0x1F
        return np.stack(((r << 3) | (r >> 2), (g << 2) | (g >> 4), (b << 3) | (b >> 2)), -1)

    p0 = rgb565(c0).astype(np.int32)
    p1 = rgb565(c1).astype(np.int32)
    four = (c0 > c1) if not four_color_only else np.ones(c0.shape, bool)
    f = four[..., None]
    p2 = np.where(f, (2 * p0 + p1) // 3, (p0 + p1) // 2)
    p3 = np.where(f, (p0 + 2 * p1) // 3, 0)

    alpha = np.full(c0.shape + (4,), 255, np.int32)
    alpha[..., 3] = np.where(four, 255, 0)  # BC1 3-color mode: index 3 is transparent black
    pal = np.stack((p0, p1, p2, p3), -2)                 # (bh, bw, 4, 3)
    pal = np.concatenate((pal, alpha[..., None]), -1)    # (bh, bw, 4, 4)

    shifts = np.arange(16, dtype=np.uint32) * 2
    idx = ((bits[..., None] >> shifts) & 3).astype(np.intp)    # (bh, bw, 16)
    return np.take_along_axis(pal, idx[..., None], axis=-2).astype(np.uint8)


def _alpha_block(blk, signed=False):
    """Decode an 8-byte BC3/BC4/BC5 alpha block -> (bh, bw, 16) values.

    Unsigned blocks give 0..255; signed blocks give -128..127.
    """
    if signed:
        a0 = blk[..., 0].astype(np.int8).astype(np.int32)
        a1 = blk[..., 1].astype(np.int8).astype(np.int32)
        lo, hi = -128, 127
    else:
        a0 = blk[..., 0].astype(np.int32)
        a1 = blk[..., 1].astype(np.int32)
        lo, hi = 0, 255
    i = np.arange(8, dtype=np.int32)
    eight = (8 - i) * a0[..., None] + (i - 1) * a1[..., None]
    eight = eight // 7
    six = (6 - i) * a0[..., None] + (i - 1) * a1[..., None]
    six = six // 5
    six[..., 6] = lo
    six[..., 7] = hi
    table = np.where((a0 > a1)[..., None], eight, six)
    table[..., 0] = a0
    table[..., 1] = a1

    bits = np.zeros(blk.shape[:-1], np.uint64)
    for k in range(6):
        bits |= blk[..., 2 + k].astype(np.uint64) << np.uint64(8 * k)
    shifts = np.arange(16, dtype=np.uint64) * np.uint64(3)
    idx = ((bits[..., None] >> shifts) & np.uint64(7)).astype(np.intp)
    return np.take_along_axis(table, idx, axis=-1)


def bc1(data, width, height):
    blk = _blocks(data, width, height, 8)
    return _assemble(_color_block(blk, False), width, height).tobytes()


def bc2(data, width, height):
    blk = _blocks(data, width, height, 16)
    rgba = _color_block(blk[..., 8:], True)
    nib = blk[..., :8]
    a = np.stack((nib & 0x0F, nib >> 4), -1).reshape(blk.shape[:-1] + (16,))
    rgba[..., 3] = a * 17
    return _assemble(rgba, width, height).tobytes()


def bc3(data, width, height):
    blk = _blocks(data, width, height, 16)
    rgba = _color_block(blk[..., 8:], True)
    rgba[..., 3] = _alpha_block(blk[..., :8]).astype(np.uint8)
    return _assemble(rgba, width, height).tobytes()


def _channel(blk, signed):
    v = _alpha_block(blk, signed)
    if signed:
        v = v + 128
    return v.astype(np.uint8)


def bc4(data, width, height, signed=False):
    blk = _blocks(data, width, height, 8)
    r = _channel(blk, signed)[..., None]
    return _assemble(r, width, height).tobytes()


def bc5(data, width, height, signed=False):
    blk = _blocks(data, width, height, 16)
    rg = np.stack((_channel(blk[..., :8], signed), _channel(blk[..., 8:], signed)), -1)
    img = _assemble(rg, width, height)
    return np.ascontiguousarray(img.reshape(-1, 2).T)


def fix_color_blocks(rgba, data, width, height, block_size, four_color_only):
    """Patch a natively decoded BC1/BC2/BC3 image where the native decoder
    differs from the specification, i.e. blocks with color0 <= color1:

    * BC1: index 3 is transparent black (the native decoder makes it opaque);
    * BC2/BC3: such blocks still use 4-color interpolation (the native
      decoder uses BC1's 3-color mode, so indices 2 and 3 differ).

    `rgba` is a writable (height, width, 4) uint8 image. Only the affected
    pixels are decoded again, with the Python decoder, and written back in
    place. Returns the image.
    """
    blk = _blocks(data, width, height, block_size)
    col = blk[..., block_size - 8:]
    by, bx = np.nonzero(_u16(col, 0) <= _u16(col, 2))
    if not by.size:
        return rgba
    sub = col[by, bx]                                         # (k, 8)
    bits = (sub[:, 4].astype(np.uint32) | (sub[:, 5].astype(np.uint32) << 8)
        | (sub[:, 6].astype(np.uint32) << 16) | (sub[:, 7].astype(np.uint32) << 24))
    idx = (bits[:, None] >> (np.arange(16, dtype=np.uint32) * 2)) & 3   # (k, 16)
    hit = idx >= 2 if four_color_only else idx == 3
    k, p = np.nonzero(hit)
    y = by[k] * 4 + p // 4
    x = bx[k] * 4 + p % 4
    keep = (y < height) & (x < width)
    k, p, y, x = k[keep], p[keep], y[keep], x[keep]
    if not four_color_only:
        rgba[y, x] = 0          # BC1 3-color mode: index 3 is transparent black
        return rgba
    # BC2/BC3: decode the 4-color palette of the affected blocks only
    blocks, k = np.unique(k, return_inverse=True)
    tex = _color_block(sub[blocks][None], True)[0]            # (n, 16, 4)
    rgba[y, x, :3] = tex[k, p, :3]  # alpha is coded separately: keep it
    return rgba
