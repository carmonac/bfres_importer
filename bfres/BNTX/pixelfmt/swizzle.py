# This file is a modified file from
# https://github.com/aboood40091/BNTX-Editor/
# Which is licensed under GPL-3

import logging; log = logging.getLogger(__name__)
import struct
import math
import numpy as np

def DIV_ROUND_UP(n, d):
    return (n + d - 1) // d

def round_up(x, y):
    return ((x - 1) | (y - 1)) + 1


def pow2_round_up(x):
    x -= 1
    x |= x >> 1
    x |= x >> 2
    x |= x >> 4
    x |= x >> 8
    x |= x >> 16

    return x + 1
    
GOB_WIDTH  = 64    # bytes
GOB_HEIGHT = 8     # rows (of elements)
GOB_SIZE   = 512   # bytes


def block_height_mip0(height):
    """Block height (in GOBs: 1..16) the driver picks for a surface whose
    first mip level is `height` elements (pixels, or BCn/ASTC blocks) tall.

    BRTI stores its choice in textureLayout & 7 (log2); this is only used to
    check it. Rule from Ryujinx, as in tegra_swizzle `block_height_mip0`.
    """
    height_and_half = height + height // 2
    for bh, lim in ((16, 128), (8, 64), (4, 32), (2, 16)):
        if height_and_half >= lim:
            return bh
    return 1


def mip_block_height(mip_height, block_height_mip0):
    """Block height (in GOBs) of a mip level `mip_height` elements tall.

    The level-0 block height is halved while the level fits in half of a
    block (tegra_swizzle `mip_block_height`, from the Ryujinx driver code).
    Level 0 itself uses it too, which is a no-op when the BRTI block height
    is the driver's own choice.
    """
    bh = block_height_mip0
    while bh > 1 and mip_height <= (bh // 2) * GOB_HEIGHT:
        bh //= 2
    return bh


def mip_size(width, height, bpp, tileMode, blockHeight):
    """Bytes one swizzled level takes; width/height in elements."""
    if tileMode == 1:
        return round_up(width * bpp, 32) * height
    return round_up(width * bpp, GOB_WIDTH) * round_up(height, blockHeight * GOB_HEIGHT)


def align_layer_size(size, height, block_height_mip0):
    """Array layers start at a multiple of a block of GOBs (Ryujinx
    `AlignLayerSize`, gobBlocksInTileX = 1): the level-0 block height,
    reduced like a mip level's for short surfaces; `height` in elements."""
    bh = mip_block_height(height, block_height_mip0)
    return round_up(size, bh * GOB_SIZE) if size else 0


def surface_layout(width, height, blkWidth, blkHeight, bpp, tileMode,
                   blockHeightLog2, mipCount=1, arrayCount=1):
    """Where every mip level of a (2D / array / cube) surface lies.

    Returns (levels, layerSize): `levels` has one dict per mip level
    (`level`, `width`/`height` in pixels, `offset` from the start of the
    layer, `size` in bytes, `blockHeightLog2` used to deswizzle it) and
    `layerSize` is the stride between array layers (cube faces are layers).
    """
    bh0 = 1 << blockHeightLog2
    levels = []
    offset = 0
    for level in range(max(1, mipCount)):
        w = max(1, width >> level)
        h = max(1, height >> level)
        ew, eh = DIV_ROUND_UP(w, blkWidth), DIV_ROUND_UP(h, blkHeight)
        bh = 1 if tileMode == 1 else mip_block_height(eh, bh0)
        size = mip_size(ew, eh, bpp, tileMode, bh)
        levels.append({'level': level, 'width': w, 'height': h, 'offset': offset,
                       'size': size, 'blockHeightLog2': bh.bit_length() - 1})
        offset += size
    layerSize = offset
    if arrayCount > 1 and tileMode != 1:
        layerSize = align_layer_size(offset, DIV_ROUND_UP(height, blkHeight), bh0)
    return levels, layerSize


def deswizzle(width, height, blkWidth, blkHeight, bpp, tileMode, blockHeightLog2, data):
    """Return the image data in linear order (row by row of elements).

    `width`/`height` are in pixels; for block-compressed formats an element
    is one blkWidth x blkHeight block of `bpp` bytes. tileMode 1 is linear
    (rows padded to 32 bytes), anything else is the Tegra block-linear
    layout. Vectorized with numpy: addresses for every element are computed
    at once and the bytes gathered in one step.
    """
    blockHeight = 1 << blockHeightLog2

    width = DIV_ROUND_UP(width, blkWidth)
    height = DIV_ROUND_UP(height, blkHeight)

    if tileMode == 1:
        # If wii u support is added this depends on if the header of the texture is "NX" or not
        pitch = round_up(width * bpp, 32)
        surfSize = pitch * height
    else:
        pitch = round_up(width * bpp, 64)
        surfSize = pitch * round_up(height, blockHeight * 8)

    y, x = np.meshgrid(np.arange(height, dtype=np.int64),
                       np.arange(width, dtype=np.int64), indexing='ij')
    if tileMode == 1:
        pos = y * pitch + x * bpp
    else:
        pos = getAddrBlockLinear(x, y, width, bpp, 0, blockHeight)

    src = np.zeros(surfSize, dtype=np.uint8)
    raw = np.frombuffer(bytes(data[:surfSize]), dtype=np.uint8)
    src[:raw.size] = raw

    # Elements whose bytes fall outside the surface stay zero, as before.
    idx = pos[..., None] + np.arange(bpp, dtype=np.int64)
    valid = (pos + bpp <= surfSize)[..., None]
    out = np.where(valid, src[np.minimum(idx, surfSize - 1)], 0).astype(np.uint8)
    return bytearray(out.tobytes())


def getAddrBlockLinear(x, y, image_width, bytes_per_pixel, base_address, blockHeight):
    image_width_in_gobs = DIV_ROUND_UP(image_width * bytes_per_pixel, 64)
    GOB_address = (base_address
                   + (y // (8 * blockHeight)) * 512 * blockHeight * image_width_in_gobs
                   + (x * bytes_per_pixel // 64) * 512 * blockHeight
                   + (y % (8 * blockHeight) // 8) * 512)

    x = x * bytes_per_pixel

    Address = (GOB_address + ((x % 64) // 32) * 256 + ((y % 8) // 2) * 64
               + ((x % 32) // 16) * 32 + (y % 2) * 16 + (x % 16))

    return Address