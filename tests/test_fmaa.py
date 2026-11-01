"""FMAA parsing tests on hand-built binary data. Needs numpy."""
import importlib.util
import os
import struct
import sys
import types
import unittest

try:
    import numpy as np
except ImportError:
    np = None

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tests"))
from test_fska import Builder, curve_bytes  # noqa: E402


def load():
    # FMAA does `from .FSKA import ...`: load both into a stub package
    pkg = types.ModuleType("fmaatest")
    pkg.__path__ = [os.path.join(ROOT, "bfres", "FRES")]
    sys.modules["fmaatest"] = pkg
    mods = {}
    for name in ("FSKA", "FMAA"):
        spec = importlib.util.spec_from_file_location(
            "fmaatest." + name, os.path.join(ROOT, "bfres", "FRES", name + ".py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules["fmaatest." + name] = mod
        spec.loader.exec_module(mod)
        mods[name] = mod
    return mods["FSKA"], mods["FMAA"]


@unittest.skipIf(np is None, "numpy not available")
class FMAATest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.F, cls.M = load()

    def build(self):
        F, M = self.F, self.M
        b = Builder()
        # tex_srt0: translate X scrolls 0 -> 10 over 10 frames (linear curve)
        c_tx = curve_bytes(b, [0, 10], [[0, 10], [10, 0]], F.CURVE_LINEAR << 4, 16)
        # pattern: texture index 0,1,0 (step int, sbyte keys, int offset 1)
        c_pat = curve_bytes(b, [0, 1, 2], [[-1], [0], [-1]],
                            (F.CURVE_STEP_INT << 4) | (F.KEY_SBYTE << 2) | F.FRAME_BYTE, 0,
                            scale=0.0, frame_fmt='<B', key_fmt='<b', int_offset=1)
        curves = b.put(c_tx + c_pat)
        consts = b.put(struct.pack('<If', 4, 0.25))  # scale X constant 0.25
        param = b.put(struct.pack('<Q6Hi', b.string('tex_srt0'), 0, 1, 0, 0, 1, 0, 0))
        pattern = b.put(struct.pack('<QhHb3x', b.string('_a0'), 1, 0xFFFF, 0))
        mat_name = b.string('Mt_Body')
        mat = b.put(struct.pack('<5Q9H6x', mat_name, param, pattern, curves, consts,
                                0, 1, 0xFFFF, 0xFFFF, 0xFFFF, 1, 1, 1, 2))
        t0, t1 = b.string('Body_Alb.0'), b.string('Body_Alb.1')
        texnames = b.put(struct.pack('<QQ', t0, t1))
        name = b.string('Body_ftp')
        hdr = b'FMAA' + b'\0' * 12 + struct.pack(
            '<10Q4HiI4H', name, 0, 0, 0, mat, 0, texnames, 0, 0, 0,
            F.FLAG_LOOP, 0, 1, 2, 10, 0, 1, 1, 0, 2)
        off = b.put(hdr)
        return M.MaterialAnimSet(bytes(b.buf), off)

    def test_header(self):
        s = self.build()
        self.assertEqual(s.name, 'Body_ftp')
        self.assertEqual(s.frame_count, 10)
        self.assertTrue(s.looping)
        self.assertEqual(s.textures, ['Body_Alb.0', 'Body_Alb.1'])
        self.assertEqual([m.name for m in s.materials], ['Mt_Body'])

    def test_param_curve_and_constant(self):
        pa = self.build().materials[0].params[0]
        self.assertEqual(pa.name, 'tex_srt0')
        frames = np.array([0.0, 5.0, 10.0])
        np.testing.assert_allclose(pa.sample(frames, 16, 0.0), [0, 5, 10])   # curve
        np.testing.assert_allclose(pa.sample(frames, 4, 1.0), [0.25] * 3)   # constant
        np.testing.assert_allclose(pa.sample(frames, 8, 1.0), [1.0] * 3)    # default

    def test_pattern(self):
        pt = self.build().materials[0].patterns[0]
        self.assertEqual(pt.name, '_a0')
        self.assertEqual(pt.sample(np.array([0.0, 1.0, 2.0])).tolist(), [0, 1, 0])

    def build_fvm(self, frames, word_list):
        """A visibility animation like Horse_Link_Foot_fvm: one material, one
        boolean step curve (byte frames, keys packed one bit per key)."""
        F, M = self.F, self.M
        b = Builder()
        flags = (F.CURVE_STEP_BOOL << 4) | F.FRAME_BYTE
        c = curve_bytes(b, frames, [[w] for w in word_list], flags, 0,
                        frame_fmt='<B', key_fmt='<I', int_offset=0)
        curves = b.put(c)
        mat = b.put(struct.pack('<5Q9H6x', b.string('Mt_Foot'), 0, 0, curves, 0,
                                0xFFFF, 0xFFFF, 0, 0, 0xFFFF, 0, 0, 0, 1))
        hdr = b'FMAA' + b'\0' * 12 + struct.pack(
            '<10Q4HiI4H', b.string('Foot_fvm'), 0, 0, 0, mat, 0, 0, 0, 0, 0,
            0, 0, 1, 1, frames[-1], 0, 0, 0, 0, 0)
        off = b.put(hdr)
        return M.MaterialAnimSet(bytes(b.buf), off)

    def test_visibility(self):
        s = self.build_fvm([0, 1, 2], [0b101])
        ma = s.materials[0]
        self.assertIsNotNone(ma.visibility)
        self.assertEqual(ma.params, [])
        self.assertEqual(ma.sample_visibility(s.frames()).tolist(), [True, False, True])

    def test_visibility_many_keys(self):
        # 40 keys span two 32-bit words: key 33 is bit 1 of the second word
        frames = list(range(40))
        s = self.build_fvm(frames, [0xFFFFFFFF, 0b10])
        vis = s.materials[0].sample_visibility(np.arange(40.0))
        self.assertEqual(vis.tolist(), [True] * 32 + [False, True] + [False] * 6)

    def test_no_visibility(self):
        self.assertIsNone(self.build().materials[0].visibility)
        self.assertIsNone(self.build().materials[0].sample_visibility(np.arange(3.0)))

    def test_bad_magic(self):
        with self.assertRaises(self.M.FMAAError):
            self.M.MaterialAnimSet(b'\0' * 0x100, 0)


if __name__ == '__main__':
    unittest.main()
