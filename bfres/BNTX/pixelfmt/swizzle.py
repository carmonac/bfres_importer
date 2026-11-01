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