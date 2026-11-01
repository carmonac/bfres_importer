# This file is part of botwtools.
#
# botwtools is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# botwtools is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with botwtools.  If not, see <https://www.gnu.org/licenses/>.
"""Uncompressed texture formats.

Bit layouts follow nn::gfx / NVN: the first component of the format name
is in the lowest bits of the little-endian word (R5_G6_B5: red in bits
0-4, blue in 11-15), as BNTX-Editor's formConv and Ryujinx (GL
UnsignedShort565Reversed, Vulkan B5G6R5UnormPack16) read them. Components
a format does not have read as 0 (green, blue) and 1 (alpha), like the
GPU, except that one-channel formats are shown as gray (R, R, R, 1), the
way BC4 is. SNORM 8-bit data is offset by +128 (-1..1 -> 0..1), like
signed BC4/BC5. Breath of the Wild only uses R8 and R8_G8_B8_A8; the other
classes are checked with synthetic data (tests/test_rgb_formats.py).

`decodePixels()` returns a flat float32 RGBA array, row by row.
"""
import logging; log = logging.getLogger(__name__)
from .base import TextureFormat
import numpy as np

# BRTI.TextureDataType
UNORM, SNORM, UINT, SINT, FLOAT, SRGB, UFLOAT = 1, 2, 3, 4, 5, 6, 10


def _rgba(n, r=None, g=None, b=None, a=None):
    out = np.zeros((n, 4), np.float32)
    out[:, 3] = 1
    for i, c in enumerate((r, g, b, a)):
        if c is not None:
            out[:, i] = c
    return out.ravel()


def _gray(v):
    v = np.asarray(v, np.float32)
    return _rgba(v.size, v, v, v)


def _bits(word, shift, n):
    return ((word >> shift) & ((1 << n) - 1)).astype(np.float32) / ((1 << n) - 1)


class _Uncompressed(TextureFormat):
    channels = 4   # stored components (for the channel selectors)

    def decompress(self, tex):
        self.dtype = int(tex.header['fmt_dtype'])
        need = tex.width * tex.height * self.bytesPerPixel
        data = bytes(tex.mipData[:need])
        if len(data) < need:
            raise ValueError("texture data is incomplete: %d bytes for %dx%d, need %d"
                % (len(data), tex.width, tex.height, need))
        return data

    def _bytes(self, data):
        v = np.frombuffer(bytes(data), np.uint8)
        if getattr(self, 'dtype', UNORM) == SNORM:
            v = v ^ 0x80
        return v

    def _words(self, data, dt):
        return np.frombuffer(bytes(data), dt).astype(np.uint32)


class R4G4(_Uncompressed):
    id = 0x01
    bytesPerPixel = 1
    channels = 2

    def decodePixels(self, data):
        v = self._words(data, '<u1')
        return _rgba(v.size, _bits(v, 0, 4), _bits(v, 4, 4))


class R8(_Uncompressed):
    """One channel, shown as gray (like BC4)."""
    id = 0x02
    bytesPerPixel = 1
    channels = 1

    def decodePixels(self, data):
        return _gray(self._bytes(data) / 255)


class R4G4B4A4(_Uncompressed):
    id = 0x03
    bytesPerPixel = 2

    def decodePixels(self, data):
        v = self._words(data, '<u2')
        return _rgba(v.size, _bits(v, 0, 4), _bits(v, 4, 4), _bits(v, 8, 4), _bits(v, 12, 4))


class A4B4G4R4(_Uncompressed):
    id = 0x04
    bytesPerPixel = 2

    def decodePixels(self, data):
        v = self._words(data, '<u2')
        return _rgba(v.size, _bits(v, 12, 4), _bits(v, 8, 4), _bits(v, 4, 4), _bits(v, 0, 4))


class R5G5B5A1(_Uncompressed):
    id = 0x05
    bytesPerPixel = 2

    def decodePixels(self, data):
        v = self._words(data, '<u2')
        return _rgba(v.size, _bits(v, 0, 5), _bits(v, 5, 5), _bits(v, 10, 5), _bits(v, 15, 1))


class A1B5G5R5(_Uncompressed):
    id = 0x06
    bytesPerPixel = 2

    def decodePixels(self, data):
        v = self._words(data, '<u2')
        return _rgba(v.size, _bits(v, 11, 5), _bits(v, 6, 5), _bits(v, 1, 5), _bits(v, 0, 1))


class B5G5R5A1(_Uncompressed):
    id = 0x3B
    bytesPerPixel = 2

    def decodePixels(self, data):
        v = self._words(data, '<u2')
        return _rgba(v.size, _bits(v, 10, 5), _bits(v, 5, 5), _bits(v, 0, 5), _bits(v, 15, 1))


class R5G6B5(_Uncompressed):
    id = 0x07
    bytesPerPixel = 2
    channels = 3

    def decodePixels(self, data):
        v = self._words(data, '<u2')
        return _rgba(v.size, _bits(v, 0, 5), _bits(v, 5, 6), _bits(v, 11, 5))


class B5G6R5(_Uncompressed):
    id = 0x08
    bytesPerPixel = 2
    channels = 3

    def decodePixels(self, data):
        v = self._words(data, '<u2')
        return _rgba(v.size, _bits(v, 11, 5), _bits(v, 5, 6), _bits(v, 0, 5))


class R8G8(_Uncompressed):
    id = 0x09
    bytesPerPixel = 2
    channels = 2

    def decodePixels(self, data):
        v = self._bytes(data).reshape(-1, 2) / 255
        return _rgba(len(v), v[:, 0], v[:, 1])


class R16(_Uncompressed):
    """One 16-bit channel (UNORM, SNORM or half float), shown as gray."""
    id = 0x0A
    bytesPerPixel = 2
    channels = 1

    def decompress(self, tex):
        data = super().decompress(tex)
        self.hdr = self.dtype == FLOAT
        return data

    def decodePixels(self, data):
        dtype = getattr(self, 'dtype', UNORM)
        if dtype == FLOAT:
            v = np.frombuffer(bytes(data), '<f2').astype(np.float32)
        elif dtype == SNORM:
            v = ((np.frombuffer(bytes(data), '<u2') ^ 0x8000) / 65535)
        else:
            v = np.frombuffer(bytes(data), '<u2') / 65535
        return _gray(v)


class R8G8B8A8(_Uncompressed):
    id = 0x0B
    bytesPerPixel = 4

    def decodePixels(self, data):
        return (self._bytes(data) / 255).astype(np.float32)


class B8G8R8A8(_Uncompressed):
    id = 0x0C
    bytesPerPixel = 4

    def decodePixels(self, data):
        v = self._bytes(data).reshape(-1, 4)[:, (2, 1, 0, 3)] / 255
        return v.astype(np.float32).ravel()


class R10G10B10A2(_Uncompressed):
    id = 0x0E
    bytesPerPixel = 4

    def decodePixels(self, data):
        v = self._words(data, '<u4')
        return _rgba(v.size, _bits(v, 0, 10), _bits(v, 10, 10), _bits(v, 20, 10), _bits(v, 30, 2))


def _ufloat(v, mbits):
    """Unsigned small float (5-bit exponent, `mbits` mantissa) -> float32."""
    e = (v >> mbits) & 0x1F
    m = (v & ((1 << mbits) - 1)).astype(np.float32) / (1 << mbits)
    out = np.where(e == 0, m * 2.0 ** -14, (1 + m) * np.exp2(e.astype(np.float32) - 15))
    out = np.where(e == 31, np.where(m == 0, np.inf, np.nan), out)
    return out.astype(np.float32)


class R11G11B10(_Uncompressed):
    """Packed unsigned floats (HDR): 11, 11 and 10 bits."""
    id = 0x0F
    bytesPerPixel = 4
    channels = 3
    hdr = True

    def decodePixels(self, data):
        v = self._words(data, '<u4')
        return _rgba(v.size, _ufloat(v & 0x7FF, 6), _ufloat((v >> 11) & 0x7FF, 6),
                     _ufloat(v >> 22, 5))


class R32(_Uncompressed):
    """One 32-bit channel (float or UNORM), shown as gray."""
    id = 0x14
    bytesPerPixel = 4
    channels = 1

    def decompress(self, tex):
        data = super().decompress(tex)
        self.hdr = self.dtype == FLOAT
        return data

    def decodePixels(self, data):
        if getattr(self, 'dtype', UNORM) == FLOAT:
            v = np.frombuffer(bytes(data), '<f4')
        else:
            v = np.frombuffer(bytes(data), '<u4') / 4294967295.0
        return _gray(v)
