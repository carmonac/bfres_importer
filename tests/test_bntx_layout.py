"""BRTI reading: mip levels, array layers / cube faces and channel
selectors, on a synthetic BNTX made by tools/test_textures.py write_bntx
(swizzled with the driver's layout rules). Needs numpy:
    python3 -m unittest discover tests
"""
import importlib.util
import io
import os
import random
import sys
import unittest

try:
    import numpy as np
except ImportError:
    np = None

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UNORM = 1


def tools():
    spec = importlib.util.spec_from_file_location(
        "test_textures_tool", os.path.join(ROOT, "tools", "test_textures.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def rand(n, seed):
    rng = random.Random(seed)
    return bytes(rng.getrandbits(8) for _ in range(n))


def levels(w, h, mips, bw, bh, bpp, layers, seed):
    out = []
    for k in range(layers):
        lay = []
        for m in range(mips):
            lw, lh = max(1, w >> m), max(1, h >> m)
            n = -(-lw // bw) * -(-lh // bh) * bpp
            lay.append(rand(n, seed * 1000 + k * 50 + m))
        out.append(lay)
    return out


# name, fmt_type, w, h, mips, layers, block, bytes per block/pixel, opts
CASES = [
    ('RGBA8_mips', 0x0b, 37, 23, 6, 1, 1, 4, {}),
    ('BC1_tree', 0x1a, 450, 526, 10, 1, 4, 8, {}),           # BotW-like
    ('RGBA8_array', 0x0b, 32, 16, 4, 3, 1, 4, {'img_dim': 5}),
    ('BC4_cube', 0x1d, 64, 64, 7, 6, 4, 8, {'img_dim': 3, 'sel': (2, 2, 2, 2)}),
    ('R8_R000', 0x02, 20, 12, 1, 1, 1, 1, {'sel': (2, 0, 0, 0)}),
    ('RG8_RG01', 0x09, 16, 8, 2, 1, 1, 2, {'sel': (2, 3, 0, 1)}),
    ('RGBA8_BGRA', 0x0b, 8, 8, 1, 1, 1, 4, {'sel': (4, 3, 2, 5)}),
    ('BC1_layout_hi', 0x1a, 64, 64, 4, 1, 4, 8, {'layout': 4}),  # header > driver
]


@unittest.skipIf(np is None, "numpy not available")
class BNTXLayoutTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if ROOT not in sys.path:
            sys.path.insert(0, ROOT)
        from bfres.BinaryFile import BinaryFile
        from bfres.BNTX import BNTX
        from bfres.BNTX.pixelfmt import channels
        cls.channels = channels
        t = tools()
        texs, cls.src = [], {}
        for i, (name, ft, w, h, mips, layers, blk, bpp, opts) in enumerate(CASES):
            lv = levels(w, h, mips, blk, blk, bpp, layers, i + 1)
            cls.src[name] = lv
            o = dict(opts, levels=lv, bpp=bpp)
            texs.append((name, ft, UNORM, w, h, (blk, blk), lv[0][0], None, None, o))
        blob = t.write_bntx(None, texs)
        bio = io.BytesIO(blob)
        bio.name = 'synth.bntx'
        bntx = BNTX(BinaryFile(bio))
        bntx.decode()
        cls.tex = {tx.name: tx for tx in bntx.textures}

    def test_all_decode(self):
        for name, tx in self.tex.items():
            self.assertIsNone(tx.decodeError, name)

    def test_levels_and_layers(self):
        for name, ft, w, h, mips, layers, blk, bpp, opts in CASES:
            tx = self.tex[name]
            self.assertEqual((tx.mipCount, tx.layerCount), (mips, layers), name)
            for k in range(layers):
                for m in range(mips):
                    self.assertEqual(bytes(tx.levelData(m, k)), self.src[name][k][m],
                                     (name, k, m))

    def test_level0_is_mipdata(self):
        for name, tx in self.tex.items():
            self.assertEqual(bytes(tx.mipData), self.src[name][0][0], name)

    def test_decode_other_levels(self):
        tx = self.tex['RGBA8_array']
        px = tx.rgba(2, 1)
        exp = np.frombuffer(self.src['RGBA8_array'][1][2], np.uint8).reshape(4, 8, 4) / 255
        np.testing.assert_allclose(px, exp, atol=1e-6)
        tx = self.tex['BC4_cube']
        for face in range(6):
            for m in range(tx.mipCount):
                px = tx.rgba(m, face)
                self.assertEqual(px.shape, (max(1, 64 >> m), max(1, 64 >> m), 4))

    def test_selectors_default_off(self):
        self.assertEqual(self.channels.mode, 'off')
        tx = self.tex['R8_R000']
        self.assertEqual(tx.channelSelectors, 'R000')
        px = tx.rgba()
        np.testing.assert_allclose(px[..., 0], px[..., 1])     # gray
        np.testing.assert_allclose(px[..., 3], 1)

    def test_selectors_on(self):
        r8 = np.frombuffer(self.src['R8_R000'][0][0], np.uint8).reshape(12, 20) / 255
        px = self.tex['R8_R000'].rgba(selectors=True)
        np.testing.assert_allclose(px[..., 0], r8, atol=1e-6)
        np.testing.assert_allclose(px[..., 1:], 0)
        rg = np.frombuffer(self.src['RG8_RG01'][0][1], np.uint8).reshape(4, 8, 2) / 255
        px = self.tex['RG8_RG01'].rgba(1, selectors=True)
        np.testing.assert_allclose(px[..., :2], rg, atol=1e-6)
        np.testing.assert_allclose(px[..., 2], 0)
        np.testing.assert_allclose(px[..., 3], 1)
        c = np.frombuffer(self.src['RGBA8_BGRA'][0][0], np.uint8).reshape(8, 8, 4) / 255
        px = self.tex['RGBA8_BGRA'].rgba(selectors=True)
        np.testing.assert_allclose(px, c[..., (2, 1, 0, 3)], atol=1e-6)
        px = self.tex['BC4_cube'].rgba(selectors=True)
        np.testing.assert_allclose(px[..., 3], px[..., 0])      # RRRR

    def test_layer_stride(self):
        tx = self.tex['BC4_cube']
        self.assertEqual(tx.layerSize * 6, tx.header['data_len'])


if __name__ == "__main__":
    unittest.main()
