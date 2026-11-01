"""FSKA parsing and curve evaluation tests on hand-built binary data.
Needs numpy: python3 -m unittest discover tests"""
import importlib.util
import os
import struct
import unittest

try:
    import numpy as np
except ImportError:
    np = None

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load():
    spec = importlib.util.spec_from_file_location(
        "fska", os.path.join(ROOT, "bfres", "FRES", "FSKA.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Builder:
    """Lays out structures in a bytearray and patches offsets."""

    def __init__(self):
        self.buf = bytearray(16)  # keep offset 0 unused (0 = null)

    def put(self, data, align=8):
        while len(self.buf) % align:
            self.buf.append(0)
        off = len(self.buf)
        self.buf += data
        return off

    def string(self, s):
        b = s.encode()
        return self.put(struct.pack('<H', len(b)) + b + b'\0', 2)


def curve_bytes(b, frames, keys, flags, target, scale=1.0, offset=0.0,
                frame_fmt='<f', key_fmt='<f'):
    f_off = b.put(b''.join(struct.pack(frame_fmt, f) for f in frames))
    k_off = b.put(b''.join(struct.pack(key_fmt, k) for row in keys for k in row))
    return struct.pack('<QQHHI5fi', f_off, k_off, flags, len(frames), target,
                       frames[0], frames[-1], scale, offset, 0.0, 0)


@unittest.skipIf(np is None, "numpy not available")
class FSKATest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.F = load()

    def build_anim(self):
        b = Builder()
        F = self.F
        # cubic curve on rotate X: segment 0..10 from 1 to 3 (linear-ish
        # cubic: c0=1, c1=2), last key holds 3
        c_rot = curve_bytes(b, [0, 10], [[1, 2, 0, 0], [3, 0, 0, 0]],
                            F.CURVE_CUBIC << 4, 0x20)
        # linear curve on translate Y with int16 keys scaled by 0.5, offset 1
        c_tr = curve_bytes(b, [0, 4], [[0, 8], [8, 0]],
                           (F.CURVE_LINEAR << 4) | (F.KEY_INT16 << 2), 0x14,
                           scale=0.5, offset=1.0, key_fmt='<h')
        # decimal10x5 frames: 2.5 and 6.0 -> raw 80 and 192
        c_sc = curve_bytes(b, [80, 192], [[1], [2]],
                           (F.CURVE_STEP_INT << 4) | F.FRAME_DECIMAL10X5, 0x04,
                           frame_fmt='<H', key_fmt='<I')
        curves = b.put(c_rot + c_tr + c_sc)
        base = b.put(struct.pack('<3f4f3f', 1, 1, 1, 0.5, 0, 0, 1, 0, 0, 0))
        name = b.string('Arm_L')
        flags = F.BASE_SCALE | F.BASE_ROTATE | F.BASE_TRANSLATE
        bone = b.put(struct.pack('<QQQIBBBBii', name, curves, base, flags, 0, 0, 3, 0, 0, 0))
        anim_name = b.string('Wait')
        fska = b.put(b'FSKA' + b'\0' * 12 + struct.pack(
            '<7QIiiIHHI', anim_name, 0, 0, 0, bone, 0, 0,
            F.ROTATE_EULER_XYZ | F.FLAG_LOOP, 10, 3, 0, 1, 0, 0))
        return F.SkeletalAnim(bytes(b.buf), fska)

    def test_header(self):
        a = self.build_anim()
        self.assertEqual(a.name, 'Wait')
        self.assertEqual(a.frame_count, 10)
        self.assertTrue(a.looping and a.euler)
        self.assertEqual(len(a.bone_anims), 1)
        self.assertEqual(a.bone_anims[0].name, 'Arm_L')
        self.assertEqual(len(a.frames()), 11)

    def test_cubic(self):
        c = self.build_anim().bone_anims[0].curves[0]
        np.testing.assert_allclose(c.evaluate([0, 5, 10, 20]), [1, 2, 3, 3])

    def test_linear_int16_scaled(self):
        c = self.build_anim().bone_anims[0].curves[1]
        # (k0 + t*k1) * 0.5 + 1 with k0=0, k1=8 over frames 0..4
        np.testing.assert_allclose(c.evaluate([0, 2, 4]), [1, 3, 5])

    def test_step_with_fixed_point_frames(self):
        c = self.build_anim().bone_anims[0].curves[2]
        np.testing.assert_allclose(c.frames, [2.5, 6.0])
        np.testing.assert_allclose(c.evaluate([0, 3, 6, 9]), [1, 1, 2, 2])

    def test_sample_merges_base_and_curves(self):
        b = self.build_anim().bone_anims[0]
        s, r, t = b.sample(np.array([0.0, 5.0]), (1, 1, 1), (0, 0, 0, 1), (9, 9, 9))
        np.testing.assert_allclose(r[:, 0], [1, 2])      # curve
        np.testing.assert_allclose(r[:, 1:3], 0)          # base
        np.testing.assert_allclose(t[:, 1], [1, 5])       # linear curve, clamped after frame 4
        np.testing.assert_allclose(t[:, 0], 0)            # base translate, not rest
        np.testing.assert_allclose(s[:, 0], [1, 1])

    def test_rest_values_when_no_base(self):
        F = self.F
        b = Builder()
        name = b.string('Leg')
        bone = b.put(struct.pack('<QQQIBBBBii', name, 0, 0, 0, 0, 0, 0, 0, 0, 0))
        ba = F.BoneAnim(bytes(b.buf), bone)
        s, r, t = ba.sample(np.array([0.0]), (2, 2, 2), (0.1, 0.2, 0.3, 1), (4, 5, 6))
        np.testing.assert_allclose(s[0], [2, 2, 2])
        np.testing.assert_allclose(t[0], [4, 5, 6])

    def test_bad_magic(self):
        with self.assertRaises(self.F.FSKAError):
            self.F.SkeletalAnim(b'\0' * 0x80, 0)


if __name__ == '__main__':
    unittest.main()
