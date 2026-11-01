"""FSKA: skeletal animations in Switch BFRES files.

Layouts follow BfresLibrary (SkeletalAnim / BoneAnim / AnimCurve) for
Switch files before version 9 (Breath of the Wild is 5.3). Everything is
parsed from the raw, decompressed file bytes with `struct`, so this module
does not depend on Blender and can be tested on its own.

FSKA header (0x60 bytes)
  0x00 "FSKA"  0x04 block header (u32 offset, u64 size)
  0x10 name  0x18 path  0x20 bind skeleton  0x28 bind index array
  0x30 bone anim array  0x38 user data  0x40 user data dict   (u64 each)
  0x48 u32 flags  0x4C s32 frame count  0x50 s32 curve count
  0x54 u32 baked size  0x58 u16 bone anim count  0x5A u16 user data count
BoneAnim (0x28 bytes)
  0x00 name  0x08 curve array  0x10 base data   (u64 each)
  0x18 u32 flags  0x1C u8 begin rotate, begin translate, curve count,
  begin base translate  0x20 s32 begin curve  0x24 pad
  base data: scale (3f) if flag, rotate (4f) if flag, translate (3f) if flag
AnimCurve (0x30 bytes)
  0x00 frame array  0x08 key array   (u64 each)
  0x10 u16 flags  0x12 u16 key count  0x14 u32 target offset
  0x18 f32 start frame, end frame, scale, offset, delta  0x2C pad

Strings are offsets to a u16 length followed by the characters.
"""
import struct

import numpy as np

try:
    from ..BinaryStruct.text import decode_str
except ImportError:  # loaded on its own by the unit tests
    import importlib.util as _ilu, os as _os
    _spec = _ilu.spec_from_file_location('_bfres_text', _os.path.join(
        _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
        'BinaryStruct', 'text.py'))
    _text = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_text)
    decode_str = _text.decode_str

FSKA_SIZE = 0x60
BONE_ANIM_SIZE = 0x28
CURVE_SIZE = 0x30

# FSKA flags
FLAG_BAKED = 1 << 0
FLAG_LOOP = 1 << 2
ROTATE_MASK = 0x7000
ROTATE_EULER_XYZ = 1 << 12

# BoneAnim flags: which base values are stored
BASE_SCALE = 1 << 3
BASE_ROTATE = 1 << 4
BASE_TRANSLATE = 1 << 5

# AnimCurve target offsets into the bone's (S, T, R) data
TARGETS = {
    0x04: ('scale', 0), 0x08: ('scale', 1), 0x0C: ('scale', 2),
    0x10: ('translate', 0), 0x14: ('translate', 1), 0x18: ('translate', 2),
    0x20: ('rotate', 0), 0x24: ('rotate', 1), 0x28: ('rotate', 2), 0x2C: ('rotate', 3),
}

# AnimCurve flags
FRAME_SINGLE, FRAME_DECIMAL10X5, FRAME_BYTE = 0, 1, 2
KEY_SINGLE, KEY_INT16, KEY_SBYTE = 0, 1, 2
CURVE_CUBIC, CURVE_LINEAR, CURVE_BAKED_FLOAT = 0, 1, 2
CURVE_STEP_INT, CURVE_BAKED_INT, CURVE_STEP_BOOL, CURVE_BAKED_BOOL = 4, 5, 6, 7


class FSKAError(ValueError):
    pass


def _str(data, offset):
    if not offset:
        return None
    n, = struct.unpack_from('<H', data, offset)
    return decode_str(data[offset + 2:offset + 2 + n], offset)


class AnimCurve:
    """One animated component, evaluated as a cubic, linear or step curve."""

    def __init__(self, data, offset):
        (frames_off, keys_off, self.flags, n, self.target,
         self.start, self.end, self.scale, self.offset, self.delta) = \
            struct.unpack_from('<QQHHI5f', data, offset)
        if ((self.flags >> 4) & 0x7) in (CURVE_STEP_INT, CURVE_BAKED_INT,
                                         CURVE_STEP_BOOL, CURVE_BAKED_BOOL):
            # integer curves store their offset as an int32, not a float
            self.offset, = struct.unpack_from('<i', data, offset + 0x24)
        self.frame_type = self.flags & 0x3
        self.key_type = (self.flags >> 2) & 0x3
        self.curve_type = (self.flags >> 4) & 0x7

        if self.frame_type == FRAME_SINGLE:
            frames = np.frombuffer(data, '<f4', n, frames_off)
        elif self.frame_type == FRAME_DECIMAL10X5:
            # unsigned 10.5 fixed point
            frames = np.frombuffer(data, '<u2', n, frames_off) / 32.0
        elif self.frame_type == FRAME_BYTE:
            frames = np.frombuffer(data, 'u1', n, frames_off)
        else:
            raise FSKAError("unknown frame type %d" % self.frame_type)
        self.frames = frames.astype(np.float64)

        per_key = {CURVE_CUBIC: 4, CURVE_LINEAR: 2}.get(self.curve_type, 1)
        if self.curve_type in (CURVE_STEP_BOOL, CURVE_BAKED_BOOL):
            # one bit per key, packed LSB first in 32-bit words
            words = np.frombuffer(data, '<u4', (n + 31) // 32, keys_off)
            bits = (words[:, None] >> np.arange(32, dtype=np.uint32)) & 1
            self.keys = bits.reshape(-1)[:n].astype(np.float64).reshape(n, 1)
            return
        if self.key_type == KEY_SINGLE:
            dt = '<i4' if self.curve_type == CURVE_STEP_INT else '<f4'
        elif self.key_type == KEY_INT16:
            dt = '<i2'
        elif self.key_type == KEY_SBYTE:
            dt = 'i1'
        else:
            raise FSKAError("unknown key type %d" % self.key_type)
        self.keys = np.frombuffer(data, dt, n * per_key, keys_off) \
            .astype(np.float64).reshape(n, per_key)

    def evaluate(self, frames):
        """Values of the curve at the given frame numbers (numpy array)."""
        f = np.asarray(frames, dtype=np.float64)
        n = len(self.frames)
        k = self.keys
        if n == 0:
            return np.zeros_like(f)
        i = np.clip(np.searchsorted(self.frames, f, side='right') - 1, 0, n - 1)
        nxt = np.minimum(i + 1, n - 1)
        span = self.frames[nxt] - self.frames[i]
        with np.errstate(divide='ignore', invalid='ignore'):
            t = np.where(span > 0, (f - self.frames[i]) / span, 0.0)
        t = np.clip(t, 0.0, 1.0)

        scale = self.scale if self.scale > 0 else 1.0
        if self.curve_type == CURVE_CUBIC:
            v = k[i, 0] + t * (k[i, 1] + t * (k[i, 2] + t * k[i, 3]))
            return v * scale + self.offset
        if self.curve_type == CURVE_LINEAR:
            return (k[i, 0] + t * k[i, 1]) * scale + self.offset
        if self.curve_type in (CURVE_STEP_BOOL, CURVE_BAKED_BOOL):
            return k[i, 0]
        if self.curve_type in (CURVE_STEP_INT, CURVE_BAKED_INT):
            return k[i, 0] + self.offset
        return k[i, 0] * scale + self.offset


class BoneAnim:
    """Animation of one bone: base S/R/T values plus per-component curves."""

    def __init__(self, data, offset):
        (name_off, curves_off, base_off, self.flags, self.begin_rotate,
         self.begin_translate, n_curves, self.begin_base_translate,
         self.begin_curve) = struct.unpack_from('<QQQIBBBBi', data, offset)
        self.name = _str(data, name_off)
        self.scale = self.rotate = self.translate = None
        pos = base_off
        if base_off:
            if self.flags & BASE_SCALE:
                self.scale = struct.unpack_from('<3f', data, pos); pos += 12
            if self.flags & BASE_ROTATE:
                self.rotate = struct.unpack_from('<4f', data, pos); pos += 16
            if self.flags & BASE_TRANSLATE:
                self.translate = struct.unpack_from('<3f', data, pos); pos += 12
        self.curves = [AnimCurve(data, curves_off + j * CURVE_SIZE) for j in range(n_curves)]

    def sample(self, frames, rest_scale, rest_rotate, rest_translate):
        """Scale (n,3), rotate (n,4) and translate (n,3) at each frame.

        Components with neither a base value nor a curve keep the bone's
        rest value.
        """
        n = len(frames)
        s = np.tile(np.asarray(self.scale if self.scale is not None else rest_scale, np.float64)[:3], (n, 1))
        r = np.tile(np.asarray(self.rotate if self.rotate is not None else rest_rotate, np.float64)[:4], (n, 1))
        t = np.tile(np.asarray(self.translate if self.translate is not None else rest_translate, np.float64)[:3], (n, 1))
        out = {'scale': s, 'rotate': r, 'translate': t}
        for c in self.curves:
            target = TARGETS.get(c.target)
            if target is None:
                continue
            name, comp = target
            out[name][:, comp] = c.evaluate(frames)
        return s, r, t


class SkeletalAnim:
    """One FSKA: a named animation of a skeleton's bones."""

    def __init__(self, data, offset):
        if bytes(data[offset:offset + 4]) != b'FSKA':
            raise FSKAError("no FSKA at 0x%X" % offset)
        (name_off, path_off, skel_off, bind_off, bones_off, ud_off, ud_dict,
         self.flags, self.frame_count, self.curve_count, self.baked_size,
         n_bones, n_userdata) = struct.unpack_from('<7QIiiIHH', data, offset + 0x10)
        self.name = _str(data, name_off)
        self.path = _str(data, path_off)
        self.looping = bool(self.flags & FLAG_LOOP)
        self.euler = (self.flags & ROTATE_MASK) == ROTATE_EULER_XYZ
        self.bone_anims = [BoneAnim(data, bones_off + i * BONE_ANIM_SIZE) for i in range(n_bones)]
        self.bind_indices = list(np.frombuffer(data, '<u2', n_bones, bind_off)) if bind_off else []

    def frames(self):
        """Integer frames 0..frame_count (inclusive) to sample."""
        return np.arange(0, max(self.frame_count, 0) + 1, dtype=np.float64)


def read_all(data, offset, count):
    """Read `count` consecutive FSKA headers starting at `offset`."""
    return [SkeletalAnim(data, offset + i * FSKA_SIZE) for i in range(count)]
