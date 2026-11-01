"""Vertex attribute decoding tests. Needs numpy: python3 -m unittest discover tests"""
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
        "attrdecode", os.path.join(ROOT, "bfres", "FRES", "FMDL", "Attribute", "decode.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def interleave(records, stride):
    """Pack one record per vertex into a buffer with the given stride."""
    out = bytearray()
    for r in records:
        out += r + b"\xAA" * (stride - len(r))
    return bytes(out)


@unittest.skipIf(np is None, "numpy not available")
class AttrDecodeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = load()

    def test_unorm16_uv(self):
        data = interleave([struct.pack("<2H", 0, 65535), struct.pack("<2H", 32768, 16384)], 12)
        out = self.d.decode(0x1201, data, 0, 12, 2)
        np.testing.assert_allclose(out, [[0, 1], [32768 / 65535, 16384 / 65535]], rtol=1e-6)
        self.assertEqual(out.dtype, np.float32)

    def test_snorm16_clamps(self):
        data = struct.pack("<2h", -32768, 32767)
        np.testing.assert_allclose(self.d.decode(0x1202, data, 0, 4, 1), [[-1, 1]])

    def test_unorm8_weights_with_offset(self):
        rec = b"\x00\x00\x00\x00" + bytes((255, 128, 0, 0))
        out = self.d.decode(0x0B01, interleave([rec, rec], 16), 4, 16, 2)
        np.testing.assert_allclose(out[1], [1, 128 / 255, 0, 0], rtol=1e-6)

    def test_uint8_indices_stay_integers(self):
        out = self.d.decode(0x0B03, bytes((1, 2, 3, 250)), 0, 4, 1)
        self.assertEqual(out.tolist(), [[1, 2, 3, 250]])
        self.assertTrue(np.issubdtype(out.dtype, np.integer))

    def test_float32_positions(self):
        data = interleave([struct.pack("<3f", 1.5, -2, 3), struct.pack("<3f", 4, 5, 6.25)], 20)
        out = self.d.decode(0x1805, data, 0, 20, 2)
        np.testing.assert_allclose(out, [[1.5, -2, 3], [4, 5, 6.25]])

    def test_half_and_arm_alternative_half(self):
        # 0x3C00 = 1.0, 0xC000 = -2.0; 0x7C00 is 65536 in ARM AHP, not inf
        data = struct.pack("<4H", 0x3C00, 0xC000, 0x7C00, 0x3555)
        out = self.d.decode(0x1505, data, 0, 8, 1)[0]
        self.assertEqual(out[0], 1.0)
        self.assertEqual(out[1], -2.0)
        self.assertEqual(out[2], 65536.0)
        self.assertAlmostEqual(out[3], 0.333252, places=5)

    def test_10_10_10_2_snorm_normal(self):
        def pack(x, y, z):
            q = [int(round(v * 511)) & 0x3FF for v in (x, y, z)]
            return struct.pack("<I", q[0] | q[1] << 10 | q[2] << 20)
        data = pack(1, 0, -1) + pack(0, -0.5, 0.5)
        out = self.d.decode(0x0E02, data, 0, 4, 2)
        np.testing.assert_allclose(out[:, :3], [[1, 0, -1], [0, -0.5, 0.5]], atol=2e-3)

    def test_reading_past_buffer_raises(self):
        with self.assertRaises(self.d.AttributeFormatError):
            self.d.decode(0x1805, b"\0" * 20, 0, 12, 2)

    def test_unknown_layout_raises(self):
        with self.assertRaises(self.d.AttributeFormatError):
            self.d.decode(0x7701, b"\0" * 16, 0, 4, 1)


if __name__ == "__main__":
    unittest.main()
