"""Block-compressed texture formats: BC1-BC7 and ASTC.

Originally based on the DXT1/3/5 decompressor by MasterVermilli0n / AboodXD
(2018).

Every format decodes with the native decoder (texture2ddecoder, see
../native.py) when it is available and falls back to the numpy decoders in
decode.py (BC1-BC5) and bptc.py (BC6H, BC7) otherwise. The two paths give
the same pixels: where texture2ddecoder deviates from the specification
(BC1/BC3 blocks with color0 <= color1) its output is patched, and formats
or variants it lacks (BC2, signed BC4/BC5, BC6H as HDR) always use numpy.
ASTC is only available natively.

`decompress()` output, as expected by each class's `decodePixels()`:
  BC1/2/3/7, ASTC -> RGBA8 bytes
  BC4             -> one byte per pixel
  BC5             -> uint8 array (2, pixels)
  BC6H            -> float32 array (height, width, 4), linear HDR
Signed BC4/BC5 values are offset by +128 (-1..1 -> 0..1 after decoding).
"""
from . import decode, bptc
from .. import native
from ..base import TextureFormat
import numpy as np

# BRTI.TextureDataType values
UNORM, SNORM, SRGB, SFLOAT, UFLOAT = 1, 2, 6, 5, 10


def _data(tex, block_bytes, bw=4, bh=4):
    data = tex.mipData
    if not isinstance(data, (bytes, bytearray)):
        data = bytes(data)
    need = ((tex.width + bw - 1) // bw) * ((tex.height + bh - 1) // bh) * block_bytes
    if len(data) < need:
        raise ValueError("compressed data is incomplete: %d bytes for %dx%d, need %d"
            % (len(data), tex.width, tex.height, need))
    return bytes(data[:need])


def _dtype(tex):
    return int(tex.header['fmt_dtype'])


class _Block(TextureFormat):
    bytesPerPixel = 4
    depth = 8
    blockBytes = 16
    hdr = False            # decodes to floats beyond 0..1 (BC6H)
    decoder = None         # 'native' or 'python' after decompress()

    def _native(self, value):
        self.decoder = 'python' if value is None else 'native'
        return value


class BC1(_Block):
    id = 0x1a
    blockBytes = 8

    def decompress(self, tex):
        data = _data(tex, 8)
        px = self._native(native.rgba('decode_bc1', data, tex.width, tex.height))
        if px is None:
            return decode.bc1(data, tex.width, tex.height)
        return decode.fix_color_blocks(px, data, tex.width, tex.height, 8, False).tobytes()


class BC2(_Block):
    # texture2ddecoder has no BC2
    id = 0x1b

    def decompress(self, tex):
        self.decoder = 'python'
        return decode.bc2(_data(tex, 16), tex.width, tex.height)


class BC3(_Block):
    id = 0x1c

    def decompress(self, tex):
        data = _data(tex, 16)
        px = self._native(native.rgba('decode_bc3', data, tex.width, tex.height))
        if px is None:
            return decode.bc3(data, tex.width, tex.height)
        return decode.fix_color_blocks(px, data, tex.width, tex.height, 16, True).tobytes()


class BC4(_Block):
    id = 0x1d
    blockBytes = 8
    channels = 1

    def decompress(self, tex):
        data = _data(tex, 8)
        signed = _dtype(tex) == SNORM
        px = None if signed else native.channels('decode_bc4', data, tex.width, tex.height, (0,))
        if self._native(px) is None:
            return decode.bc4(data, tex.width, tex.height, signed)
        return px.tobytes()

    def decodePixels(self, data):
        rgba = np.empty(len(data)*4)
        rgba[0::4] = rgba[1::4] = rgba[2::4] = np.frombuffer(data, dtype='B') / 255
        rgba[3::4] = 1
        return rgba


class BC5(_Block):
    id = 0x1e
    channels = 2

    def decompress(self, tex):
        data = _data(tex, 16)
        signed = _dtype(tex) == SNORM
        px = None if signed else native.channels('decode_bc5', data, tex.width, tex.height, (0, 1))
        if self._native(px) is None:
            return decode.bc5(data, tex.width, tex.height, signed)
        return np.ascontiguousarray(px.reshape(-1, 2).T)

    def decodePixels(self, data):
        r = data[0].astype(np.float32) / 255
        g = data[1].astype(np.float32) / 255
        x = r * 2 - 1
        y = g * 2 - 1
        z = abs(1 - x**2 - y**2)**0.5
        rgba = np.empty(data.size*2, dtype=np.float32)
        rgba[0::4] = r
        rgba[1::4] = g
        rgba[2::4] = np.real((z + 1) * 0.5)
        rgba[3::4] = 1
        return rgba


class BC6(_Block):
    """BC6H: HDR, half float RGB. Always decoded with numpy: the native
    decoder clamps it to 8 bits."""
    id = 0x1f
    hdr = True
    channels = 3
    depth = 16

    def decompress(self, tex):
        self.decoder = 'python'
        signed = _dtype(tex) == SFLOAT    # 0x1f05 BC6H_SF16, 0x1f0a BC6H_UF16
        return bptc.bc6h(_data(tex, 16), tex.width, tex.height, signed)

    def decodePixels(self, data):
        return np.asarray(data, np.float32).ravel()


class BC7(_Block):
    id = 0x20

    def decompress(self, tex):
        data = _data(tex, 16)
        px = self._native(native.rgba('decode_bc7', data, tex.width, tex.height))
        if px is None:
            return bptc.bc7(data, tex.width, tex.height)
        return px.tobytes()


class _ASTC(_Block):
    blkWidth = blkHeight = 4

    def decompress(self, tex):
        bw, bh = self.blkWidth, self.blkHeight
        data = _data(tex, 16, bw, bh)
        px = self._native(native.rgba('decode_astc', data, tex.width, tex.height, bw, bh))
        if px is None:
            raise RuntimeError("ASTC %dx%d needs the native texture decoder "
                "(texture2ddecoder), which is not available" % (bw, bh))
        return px.tobytes()


# 0x2d ASTC_4x4 ... 0x3a ASTC_12x12 (UNORM / SRGB)
ASTC_SIZES = ((4, 4), (5, 4), (5, 5), (6, 5), (6, 6), (8, 5), (8, 6), (8, 8),
              (10, 5), (10, 6), (10, 8), (10, 10), (12, 10), (12, 12))
for _i, (_w, _h) in enumerate(ASTC_SIZES):
    _name = 'ASTC%dx%d' % (_w, _h)
    globals()[_name] = type(_name, (_ASTC,), {
        'id': 0x2d + _i, 'blkWidth': _w, 'blkHeight': _h,
        '__doc__': 'ASTC with %dx%d blocks.' % (_w, _h)})
del _i, _w, _h, _name
