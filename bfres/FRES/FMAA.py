"""FMAA: material animations in Switch BFRES files (before v9).

Layouts from BfresLibrary (MaterialAnim / MaterialAnimData /
ParamAnimInfo / PatternAnimInfo). Parsed from the raw decompressed bytes,
independent of Blender. Curves use the same AnimCurve layout as FSKA.

FMAA header (0x78 bytes)
  0x00 "FMAA"  0x04 block header (u32 offset, u64 size)
  0x10 name  0x18 path  0x20 bind model  0x28 bind indices
  0x30 material anim array  0x38 (unused)  0x40 texture name array
  0x48 user data  0x50 user data dict  0x58 texture bind array  (u64 each)
  0x60 u16 flags  0x62 u16 user data count  0x64 u16 material count
  0x66 u16 curve count  0x68 s32 frame count  0x6C u32 baked size
  0x70 u16 shader param anim count, texture pattern anim count,
       visibility anim count, texture count
MaterialAnimData (0x40 bytes)
  0x00 name  0x08 param anim infos  0x10 pattern anim infos
  0x18 curves  0x20 constants   (u64 each)
  0x28 u16 shader param curve index, texture pattern curve index,
       visual constant index, visual curve index, begin visual constant,
       param anim count, pattern anim count, constant count, curve count
ParamAnimInfo (0x18): name, u16 begin curve, float curves, int curves,
  begin constant, constant count, sub-bind index
PatternAnimInfo (0x10): name, s16 curve index, u16 begin constant,
  s8 sub-bind index
AnimConstant (8): u32 target offset, 32-bit value (float or int)

Material visibility ("*_fvm" animations): the material has a boolean step
curve (one bit per key) that no parameter or pattern uses. The curve
indices stored in MaterialAnimData are begin indices into the curves of
the whole FMAA, so the visibility curve is identified as the material's
leftover boolean curve instead.
"""
import struct

import numpy as np

from .FSKA import (AnimCurve, CURVE_SIZE, FLAG_LOOP, _str,
                   CURVE_STEP_BOOL, CURVE_BAKED_BOOL)

FMAA_SIZE = 0x78
MAT_ANIM_SIZE = 0x40
PARAM_INFO_SIZE = 0x18
PATTERN_INFO_SIZE = 0x10
CONSTANT_SIZE = 8
NONE16 = 0xFFFF


class FMAAError(ValueError):
    pass


class ParamAnim:
    """Animation of one shader parameter (e.g. tex_srt0) of a material.

    `components` maps the byte offset inside the parameter's data to either
    an AnimCurve or a constant value. For tex_srt params the offsets are
    0 mode, 4 scale X, 8 scale Y, 12 rotation, 16 translate X, 20 translate Y.
    """

    def __init__(self, name, curves, constants):
        self.name = name
        self.curves = curves          # {offset: AnimCurve}
        self.constants = constants    # {offset: (raw int, float)}

    def sample(self, frames, offset, default):
        if offset in self.curves:
            return self.curves[offset].evaluate(frames)
        if offset in self.constants:
            return np.full(len(frames), self.constants[offset][1])
        return np.full(len(frames), default, dtype=np.float64)


class PatternAnim:
    """Texture pattern animation: which texture a sampler shows per frame."""

    def __init__(self, name, curve, constant):
        self.name = name          # material sampler name, e.g. "_a0"
        self.curve = curve        # AnimCurve of texture indices, or None
        self.constant = constant  # texture index when not animated

    def sample(self, frames):
        if self.curve is not None:
            return np.rint(self.curve.evaluate(frames)).astype(np.int64)
        return np.full(len(frames), self.constant or 0, dtype=np.int64)


class MaterialAnim:
    """The animated parameters and texture patterns of one material."""

    def __init__(self, data, offset):
        (name_off, params_off, patterns_off, curves_off, consts_off,
         self.param_curve_index, self.pattern_curve_index, self.vis_constant_index,
         self.vis_curve_index, self.begin_vis_constant, n_params, n_patterns,
         n_consts, n_curves) = struct.unpack_from('<5Q9H', data, offset)
        self.name = _str(data, name_off)
        curves = [AnimCurve(data, curves_off + i * CURVE_SIZE) for i in range(n_curves)]
        consts = [struct.unpack_from('<Ii', data, consts_off + i * CONSTANT_SIZE)
                  for i in range(n_consts)]
        consts = [(off, raw, struct.unpack('<f', struct.pack('<i', raw))[0])
                  for off, raw in consts]

        self.params = []
        for i in range(n_params):
            (pname, begin_curve, n_float, n_int, begin_const, n_const,
             _sub) = struct.unpack_from('<Q6H', data, params_off + i * PARAM_INFO_SIZE)
            pc = {}
            for c in curves[begin_curve:begin_curve + n_float + n_int]:
                pc[c.target] = c
            pk = {}
            if begin_const != NONE16:
                for off, raw, val in consts[begin_const:begin_const + n_const]:
                    pk[off] = (raw, val)
            self.params.append(ParamAnim(_str(data, pname), pc, pk))

        self.patterns = []
        for i in range(n_patterns):
            pname, curve_idx, begin_const, _sub = struct.unpack_from(
                '<QhHb', data, patterns_off + i * PATTERN_INFO_SIZE)
            curve = curves[curve_idx] if 0 <= curve_idx < len(curves) else None
            const = consts[begin_const][1] if begin_const != NONE16 and begin_const < len(consts) else None
            self.patterns.append(PatternAnim(_str(data, pname), curve, const))

        used = {id(c) for p in self.params for c in p.curves.values()}
        used |= {id(p.curve) for p in self.patterns if p.curve is not None}
        self.visibility = next((c for c in curves if id(c) not in used
                                and c.curve_type in (CURVE_STEP_BOOL, CURVE_BAKED_BOOL)), None)

    def sample_visibility(self, frames):
        """Visibility of the material per frame (bool array), or None."""
        if self.visibility is None:
            return None
        return self.visibility.evaluate(frames) > 0.5


class MaterialAnimSet:
    """One FMAA: a named animation over several materials."""

    def __init__(self, data, offset):
        if bytes(data[offset:offset + 4]) != b'FMAA':
            raise FMAAError("no FMAA at 0x%X" % offset)
        (name_off, path_off, _model, _bind, mats_off, _unk, texnames_off,
         _ud, _udd, _texbind) = struct.unpack_from('<10Q', data, offset + 0x10)
        (self.flags, _n_ud, n_mats, self.curve_count, self.frame_count,
         self.baked_size, self.n_param_anims, self.n_pattern_anims,
         self.n_vis_anims, n_textures) = struct.unpack_from('<4HiI4H', data, offset + 0x60)
        self.name = _str(data, name_off)
        self.path = _str(data, path_off)
        self.looping = bool(self.flags & FLAG_LOOP)
        self.textures = [_str(data, struct.unpack_from('<Q', data, texnames_off + 8 * i)[0])
                         for i in range(n_textures)] if texnames_off else []
        self.materials = [MaterialAnim(data, mats_off + i * MAT_ANIM_SIZE) for i in range(n_mats)]

    def frames(self):
        return np.arange(0, max(self.frame_count, 0) + 1, dtype=np.float64)


def read_all(data, offset, count):
    return [MaterialAnimSet(data, offset + i * FMAA_SIZE) for i in range(count)]
