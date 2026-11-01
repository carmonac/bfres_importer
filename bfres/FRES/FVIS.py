"""FVIS: bone and material visibility animations in Switch BFRES files
(before v9). Layout from BfresLibrary (Switch/VisibilityAnimParser).

FVIS header (0x68 bytes), magic "FBVS"
  0x00 "FBVS"  0x04 block header (u32 offset, u64 size)
  0x10 name  0x18 path  0x20 bind model  0x28 bind indices (u16 x anims)
  0x30 curves  0x38 base values (bits)  0x40 name array
  0x48 user data  0x50 user data dict   (u64 each)
  0x58 u16 flags  0x5A u16 user data count  0x5C s32 frame count
  0x60 u16 anim count  0x62 u16 curve count  0x64 u32 baked size

Flags: 0x4 looping, 0x100 material visibility (else bone visibility).
Each entry is a bone (or material) name with a base visibility (one bit
per entry, LSB first in 32-bit words); a boolean step curve whose target
is the entry index overrides it.
"""
import struct

import numpy as np

from .FSKA import AnimCurve, CURVE_SIZE, FLAG_LOOP, _str

FVIS_SIZE = 0x68
FLAG_MATERIAL = 0x100


class FVISError(ValueError):
    pass


def unpack_bits(data, offset, count):
    """`count` booleans packed one bit each, LSB first, in 32-bit words."""
    if not count:
        return np.zeros(0, dtype=bool)
    words = np.frombuffer(data, '<u4', (count + 31) // 32, offset)
    bits = (words[:, None] >> np.arange(32, dtype=np.uint32)) & 1
    return bits.reshape(-1)[:count].astype(bool)


class VisibilityAnim:
    """One FVIS: per-bone (or per-material) visibility over time."""

    def __init__(self, data, offset):
        if bytes(data[offset:offset + 4]) != b'FBVS':
            raise FVISError("no FBVS at 0x%X" % offset)
        (name_off, path_off, _model, _bind, curves_off, base_off, names_off,
         _ud, _udd) = struct.unpack_from('<9Q', data, offset + 0x10)
        (self.flags, _n_ud, self.frame_count, n_anims, n_curves,
         self.baked_size) = struct.unpack_from('<HHiHHI', data, offset + 0x58)
        self.name = _str(data, name_off)
        self.path = _str(data, path_off)
        self.looping = bool(self.flags & FLAG_LOOP)
        self.is_material = bool(self.flags & FLAG_MATERIAL)
        self.names = [_str(data, struct.unpack_from('<Q', data, names_off + 8 * i)[0])
                      for i in range(n_anims)] if names_off else []
        self.base = unpack_bits(data, base_off, n_anims) if base_off \
            else np.ones(n_anims, dtype=bool)
        self.curves = {}
        for i in range(n_curves):
            c = AnimCurve(data, curves_off + i * CURVE_SIZE)
            self.curves[c.target] = c

    def frames(self):
        return np.arange(0, max(self.frame_count, 0) + 1, dtype=np.float64)

    def sample(self, index, frames):
        """Visibility of entry `index` at each frame (bool array)."""
        c = self.curves.get(index)
        if c is not None:
            return c.evaluate(frames) > 0.5
        return np.full(len(frames), bool(self.base[index]))


def read_all(data, offset, count):
    return [VisibilityAnim(data, offset + i * FVIS_SIZE) for i in range(count)]
