"""BC1-BC5 decoder tests. Needs numpy (bundled with Blender):
    python3 -m unittest discover tests
"""
import importlib.util
import os
import struct
import unittest

try:
    import numpy as np
except ImportError:  # e.g. macOS system python
    np = None

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load():
    spec = importlib.util.spec_from_file_location(
        "bcdecode", os.path.join(ROOT, "bfres", "BNTX", "pixelfmt", "BCn", "decode.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def bc1_block(c0, c1, indices):
    bits = 0
    for p, i in enumerate(indices):
        bits |= i << (2 * p)
    return struct.pack("<HHI", c0, c1, bits)


def alpha_block(a0, a1, indices):
    bits = 0
    for p, i in enumerate(indices):
        bits |= i << (3 * p)
    return bytes((a0, a1)) + bits.to_bytes(6, "little")


@unittest.skipIf(np is None, "numpy not available")
class BCDecodeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = load()

    def rgba(self, data, w, h):
        return np.frombuffer(data, np.uint8).reshape(h, w, 4)

    def test_bc1_four_color(self):
        # c0 white > c1 black: palette white, black, 2/3 white, 1/3 white
        blk = bc1_block(0xFFFF, 0x0000, list(range(4)) * 4)
        img = self.rgba(self.d.bc1(blk, 4, 4), 4, 4)
        self.assertEqual(img[0, :, 0].tolist(), [255, 0, 170, 85])
        self.assertTrue((img[..., 3] == 255).all())

    def test_bc1_three_color_transparent(self):
        # c0 <= c1: index 2 is the average, index 3 transparent black
        blk = bc1_block(0x0000, 0xFFFF, [2, 3] * 8)
        img = self.rgba(self.d.bc1(blk, 4, 4), 4, 4)
        self.assertEqual(img[0, 0].tolist(), [127, 127, 127, 255])
        self.assertEqual(img[0, 1].tolist(), [0, 0, 0, 0])

    def test_rgb565_expansion(self):
        # pure green 6-bit value 32 expands to 130 (standard bit replication)
        blk = bc1_block(32 << 5, 0, [0] * 16)
        self.assertEqual(self.rgba(self.d.bc1(blk, 4, 4), 4, 4)[0, 0, 1], 130)

    def test_non_multiple_of_four(self):
        # 6x5 image = 2x2 blocks; each block a solid, distinct color
        cols = [0xF800, 0x07E0, 0x001F, 0xFFFF]
        data = b"".join(bc1_block(c, 0, [0] * 16) for c in cols)
        out = self.d.bc1(data, 6, 5)
        self.assertEqual(len(out), 6 * 5 * 4)
        img = self.rgba(out, 6, 5)
        self.assertEqual(img[0, 3].tolist(), [255, 0, 0, 255])   # block (0,0)
        self.assertEqual(img[0, 4].tolist(), [0, 255, 0, 255])   # block (0,1)
        self.assertEqual(img[4, 0].tolist(), [0, 0, 255, 255])   # block (1,0)
        self.assertEqual(img[4, 5].tolist(), [255, 255, 255, 255])

    def test_bc3_alpha(self):
        a = alpha_block(255, 0, [0, 1, 2, 7] * 4)
        c = bc1_block(0xFFFF, 0, [0] * 16)
        img = self.rgba(self.d.bc3(a + c, 4, 4), 4, 4)
        # 8-value mode: index 2 = (6*255 + 0)/7, index 7 = (1*255 + 6*0)/7
        self.assertEqual(img[0, :, 3].tolist(), [255, 0, 218, 36])

    def test_bc2_alpha(self):
        nibbles = bytes([0xF0] * 8)  # low nibble 0, high nibble 15
        c = bc1_block(0xFFFF, 0, [0] * 16)
        img = self.rgba(self.d.bc2(nibbles + c, 4, 4), 4, 4)
        self.assertEqual(img[0, :, 3].tolist(), [0, 255, 0, 255])

    def test_bc4_six_value_mode(self):
        # a0 <= a1: indices 6 and 7 are 0 and 255
        blk = alpha_block(10, 200, [0, 1, 6, 7] * 4)
        out = np.frombuffer(self.d.bc4(blk, 4, 4), np.uint8).reshape(4, 4)
        self.assertEqual(out[0].tolist(), [10, 200, 0, 255])

    def test_bc4_signed(self):
        blk = alpha_block(0x7F, 0x81, [0, 1] * 8)   # 127, -127
        out = np.frombuffer(self.d.bc4(blk, 4, 4, True), np.uint8).reshape(4, 4)
        self.assertEqual(out[0, :2].tolist(), [255, 1])

    def test_bc5_shape(self):
        blk = alpha_block(255, 0, [0] * 16) + alpha_block(0, 255, [0] * 16)
        out = self.d.bc5(blk * 2, 5, 3)
        self.assertEqual(out.shape, (2, 15))
        self.assertTrue((out[0] == 255).all() and (out[1] == 0).all())

    def test_short_data(self):
        with self.assertRaises(ValueError):
            self.d.bc1(b"\0" * 8, 8, 8)


if __name__ == "__main__":
    unittest.main()
