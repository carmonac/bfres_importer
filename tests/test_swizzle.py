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


# (expected bytes for 6 cube faces, width, height, block w, block h, bpp, mips):
# nutexb cubemaps from Smash Ultimate, as in tegra_swizzle's tests
CUBE_SIZES = [
    (6144, 16, 16, 1, 1, 4, 1), (3072, 16, 16, 4, 4, 8, 1),
    (25165824, 2048, 2048, 4, 4, 16, 1), (1572864, 256, 256, 1, 1, 4, 1),
    (393216, 64, 64, 1, 1, 16, 1), (147456, 128, 128, 4, 4, 16, 8),
    (15360, 16, 16, 4, 4, 16, 5), (540672, 256, 256, 4, 4, 16, 9),
    (1204224, 288, 288, 4, 4, 16, 9), (2113536, 512, 512, 4, 4, 16, 10),
    (49152, 64, 64, 4, 4, 16, 7),
]

# Breath of the Wild textures (RomFS scan, tools/scan_textures.py): size,
# block size, bytes per block, mips, BRTI block height log2, mip offsets
# and data length as stored in the file.
BOTW = [
    # Obj_TreeConiferous_B: Tree_TreeConiferousLeaf_B_Alb, BC1. The old
    # rule (one block-height step per level) got level 4 wrong.
    (450, 526, 4, 8, 10, 4, [0, 245760, 311296, 327680, 331776, 332288,
                             332800, 333312, 333824, 334336], 334848),
    # TwnObj_City_GerudoFurniture_A: CmnTex_Cloth_..._S_Nrm, BC1
    (512, 261, 4, 8, 10, 3, [0, 131072, 163840, 172032, 173056, 173568,
                             174080, 174592, 175104, 175616], 176128),
    # Armor_017: Armor_017_Lower_Emm, BC4
    (384, 384, 4, 8, 9, 4, [0, 98304, 122880, 129024, 131072, 131584,
                            132096, 132608, 133120], 133632),
    # DgnObj_RemainsTerminalBody_A: ..._Emm, BC5
    (896, 896, 4, 16, 10, 4, [0, 917504, 1146880, 1204224, 1218560, 1222656,
                              1223680, 1224192, 1224704, 1225216], 1225728),
    # DgnObj_HingeBoard_A: Etc_DgnObjHingeBoardAText_A_Alb, BC3 207x30
    (207, 30, 4, 16, 8, 0, [0, 6656, 10240, 12288, 13312, 13824, 14336,
                            14848], 15360),
]


@unittest.skipIf(np is None, "numpy not available")
class LayoutTest(unittest.TestCase):
    """Block heights and surface layout (mips, array layers)."""

    @classmethod
    def setUpClass(cls):
        cls.sw = load()

    def test_block_height_mip0(self):
        # tegra_swizzle / Smash Ultimate: BCn height in pixels -> GOBs
        for px, bh in [(36, 1), (40, 1), (48, 2), (84, 2), (96, 4), (168, 4),
                       (176, 8), (340, 8), (360, 16), (1408, 16)]:
            self.assertEqual(self.sw.block_height_mip0(px // 4), bh, px)

    def test_mip_block_height(self):
        sw = self.sw
        self.assertEqual(sw.mip_block_height(64, 16), 8)
        self.assertEqual(sw.mip_block_height(65, 16), 16)
        self.assertEqual(sw.mip_block_height(32, 16), 4)
        self.assertEqual(sw.mip_block_height(1, 16), 1)
        # the driver's own level-0 choice is never reduced
        for h in range(1, 2000):
            bh = sw.block_height_mip0(h)
            self.assertEqual(sw.mip_block_height(h, bh), bh, h)

    def test_cube_sizes(self):
        sw = self.sw
        for exp, w, h, bw, bh, bpp, mips in CUBE_SIZES:
            bh0 = sw.block_height_mip0(sw.DIV_ROUND_UP(h, bh))
            _levels, layer = sw.surface_layout(w, h, bw, bh, bpp, 0,
                                               bh0.bit_length() - 1, mips, 6)
            self.assertEqual(layer * 6, exp, (w, h, bpp, mips))

    def test_botw_offsets(self):
        for w, h, blk, bpp, mips, bhl, offsets, size in BOTW:
            levels, layer = self.sw.surface_layout(w, h, blk, blk, bpp, 0, bhl, mips)
            self.assertEqual([lv['offset'] for lv in levels], offsets, (w, h))
            self.assertEqual(layer, size, (w, h))

    def test_old_rule_differs(self):
        # level 4 of the 450x526 BC1: 8 blocks tall, one GOB is enough
        levels, _ = self.sw.surface_layout(450, 526, 4, 4, 8, 0, 4, 10)
        self.assertEqual([lv['blockHeightLog2'] for lv in levels],
                         [4, 4, 3, 2, 0, 0, 0, 0, 0, 0])


if __name__ == "__main__":
    unittest.main()
