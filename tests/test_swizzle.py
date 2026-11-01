"""Deswizzle tests: the vectorized version must match a straightforward
per-element reference. Needs numpy: python3 -m unittest discover tests"""
import importlib.util
import os
import random
import unittest

try:
    import numpy as np
except ImportError:
    np = None

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load():
    spec = importlib.util.spec_from_file_location(
        "swz", os.path.join(ROOT, "bfres", "BNTX", "pixelfmt", "swizzle.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def reference(sw, width, height, blkWidth, blkHeight, bpp, tileMode, blockHeightLog2, data):
    """The original element-by-element loop from BNTX-Editor."""
    blockHeight = 1 << blockHeightLog2
    width = sw.DIV_ROUND_UP(width, blkWidth)
    height = sw.DIV_ROUND_UP(height, blkHeight)
    if tileMode == 1:
        pitch = sw.round_up(width * bpp, 32)
        surfSize = pitch * height
    else:
        pitch = sw.round_up(width * bpp, 64)
        surfSize = pitch * sw.round_up(height, blockHeight * 8)
    result = bytearray(width * height * bpp)
    for y in range(height):
        for x in range(width):
            if tileMode == 1:
                pos = y * pitch + x * bpp
            else:
                pos = sw.getAddrBlockLinear(x, y, width, bpp, 0, blockHeight)
            pos_ = (y * width + x) * bpp
            if pos + bpp <= surfSize:
                chunk = data[pos:pos + bpp]
                result[pos_:pos_ + len(chunk)] = chunk
    return bytes(result)


@unittest.skipIf(np is None, "numpy not available")
class SwizzleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sw = load()

    def check(self, w, h, bw, bh, bpp, tile, bhl):
        rng = random.Random(w * 31 + h * 7 + bpp + tile + bhl)
        blocks_h = self.sw.DIV_ROUND_UP(h, bh)
        size = self.sw.round_up(self.sw.DIV_ROUND_UP(w, bw) * bpp, 64) * \
            self.sw.round_up(blocks_h, (1 << bhl) * 8)
        data = bytes(rng.getrandbits(8) for _ in range(size))
        got = bytes(self.sw.deswizzle(w, h, bw, bh, bpp, tile, bhl, data))
        exp = reference(self.sw, w, h, bw, bh, bpp, tile, bhl, data)
        self.assertEqual(got, exp, (w, h, bw, bh, bpp, tile, bhl))

    def test_block_linear_bc1(self):
        for w, h, bhl in [(64, 64, 2), (454, 234, 3), (1024, 128, 2), (16, 8, 0)]:
            self.check(w, h, 4, 4, 8, 0, bhl)

    def test_block_linear_bc3(self):
        for w, h, bhl in [(128, 128, 3), (114, 59, 1)]:
            self.check(w, h, 4, 4, 16, 0, bhl)

    def test_block_linear_rgba8(self):
        self.check(100, 37, 1, 1, 4, 0, 2)

    def test_linear(self):
        self.check(60, 20, 1, 1, 4, 1, 0)


if __name__ == "__main__":
    unittest.main()
