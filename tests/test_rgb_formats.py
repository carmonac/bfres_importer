"""Uncompressed texture formats (bfres/BNTX/pixelfmt/rgb.py) and channel
selectors (channels.py) on synthetic pixels. Needs numpy:
    python3 -m unittest discover tests
"""
import os
import struct
import sys
import unittest

try:
    import numpy as np
except ImportError:
    np = None

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

UNORM, SNORM, FLOAT, SRGB = 1, 2, 5, 6


class FakeTex:
    def __init__(self, data, w, h, dtype=UNORM):
        self.mipData = data
        self.width = w
        self.height = h
        self.header = {'fmt_dtype': dtype}


def pf():
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    import bfres.BNTX.pixelfmt as mod
    import bfres.BNTX.pixelfmt.channels
    return mod


@unittest.skipIf(np is None, "numpy not available")
class RGBFormatTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pf = pf()

    def decode(self, fid, data, w=1, h=None, dtype=UNORM):
        fmt = self.pf.TextureFormat.get(fid)()
        h = h or len(data) // (fmt.bytesPerPixel * w)
        raw = fmt.decompress(FakeTex(data, w, h, dtype))
        px = np.asarray(fmt.decodePixels(raw), np.float64).reshape(-1, 4)
        self.assertEqual(px.shape[0], w * h)
        return fmt, px

    def close(self, px, exp, places=6):
        np.testing.assert_allclose(px, np.asarray(exp, np.float64).reshape(-1, 4),
                                   atol=10 ** -places)

    def test_registered(self):
        for fid in (0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A,
                    0x0B, 0x0C, 0x0E, 0x0F, 0x14, 0x3B):
            self.assertEqual(self.pf.TextureFormat.get(fid).id, fid)

    def test_unknown_format(self):
        cls = self.pf.TextureFormat.get(0x7F)
        with self.assertRaises(TypeError):
            cls().decompress(FakeTex(b'\0' * 4, 1, 1))

    def test_r8(self):
        _, px = self.decode(0x02, bytes((0, 51, 255)), 3, 1)
        self.close(px, [[0, 0, 0, 1], [.2, .2, .2, 1], [1, 1, 1, 1]])

    def test_r8_snorm(self):
        # -128 (0x80) -> 0, 0 -> 128/255, 127 -> 1
        _, px = self.decode(0x02, bytes((0x80, 0x00, 0x7F)), 3, 1, SNORM)
        np.testing.assert_allclose(px[:, 0], [0, 128 / 255, 1], atol=1e-6)

    def test_r4g4(self):
        _, px = self.decode(0x01, bytes((0xF3,)))
        self.close(px, [[3 / 15, 1, 0, 1]])

    def test_r8g8(self):
        _, px = self.decode(0x09, bytes((10, 200)))
        self.close(px, [[10 / 255, 200 / 255, 0, 1]])

    def test_rgba8_bgra8(self):
        _, px = self.decode(0x0B, bytes((1, 2, 3, 4)))
        self.close(px, [[1 / 255, 2 / 255, 3 / 255, 4 / 255]])
        _, px = self.decode(0x0C, bytes((1, 2, 3, 4)))
        self.close(px, [[3 / 255, 2 / 255, 1 / 255, 4 / 255]])

    def test_565(self):
        # red in the low bits: r=31, g=0, b=0 / r=0, g=63, b=0 / blue
        words = struct.pack('<3H', 0x001F, 0x07E0, 0xF800)
        _, px = self.decode(0x07, words, 3, 1)
        self.close(px, [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]])
        _, px = self.decode(0x08, words, 3, 1)
        self.close(px, [[0, 0, 1, 1], [0, 1, 0, 1], [1, 0, 0, 1]])
        _, px = self.decode(0x07, struct.pack('<H', (16 << 11) | (32 << 5) | 8))
        self.close(px, [[8 / 31, 32 / 63, 16 / 31, 1]])

    def test_rgba4(self):
        w = struct.pack('<H', 0x4321)
        _, px = self.decode(0x03, w)
        self.close(px, [[1 / 15, 2 / 15, 3 / 15, 4 / 15]])
        _, px = self.decode(0x04, w)
        self.close(px, [[4 / 15, 3 / 15, 2 / 15, 1 / 15]])

    def test_5551(self):
        v = 3 | (7 << 5) | (11 << 10) | (1 << 15)
        _, px = self.decode(0x05, struct.pack('<H', v))
        self.close(px, [[3 / 31, 7 / 31, 11 / 31, 1]])
        _, px = self.decode(0x3B, struct.pack('<H', v))
        self.close(px, [[11 / 31, 7 / 31, 3 / 31, 1]])
        v = 1 | (3 << 1) | (7 << 6) | (11 << 11)    # A1 B5 G5 R5
        _, px = self.decode(0x06, struct.pack('<H', v))
        self.close(px, [[11 / 31, 7 / 31, 3 / 31, 1]])

    def test_rgb10a2(self):
        v = 1023 | (512 << 10) | (0 << 20) | (2 << 30)
        _, px = self.decode(0x0E, struct.pack('<I', v))
        self.close(px, [[1, 512 / 1023, 0, 2 / 3]])

    def test_r16(self):
        _, px = self.decode(0x0A, struct.pack('<2H', 0, 65535), 2, 1)
        self.close(px, [[0, 0, 0, 1], [1, 1, 1, 1]])
        fmt, px = self.decode(0x0A, np.array([0.5, 3.0], '<f2').tobytes(), 2, 1, FLOAT)
        self.close(px, [[.5, .5, .5, 1], [3, 3, 3, 1]])
        self.assertTrue(fmt.hdr)

    def test_r32(self):
        fmt, px = self.decode(0x14, np.array([0.25, 8.0], '<f4').tobytes(), 2, 1, FLOAT)
        self.close(px, [[.25, .25, .25, 1], [8, 8, 8, 1]])
        self.assertTrue(fmt.hdr)
        fmt, px = self.decode(0x14, struct.pack('<I', 0xFFFFFFFF))
        self.close(px, [[1, 1, 1, 1]])
        self.assertFalse(getattr(fmt, 'hdr', False))

    def test_r11g11b10(self):
        def f11(e, m):
            return (e << 6) | m
        # 1.0 = exponent 15, mantissa 0; 2.5 = 1.25 * 2^1 -> e 16, m 16/64;
        # blue (10 bits: 5 mantissa): 0.5 = e 14
        v = f11(15, 0) | (f11(16, 16) << 11) | (((14 << 5) | 0) << 22)
        fmt, px = self.decode(0x0F, struct.pack('<I', v))
        self.close(px, [[1, 2.5, .5, 1]])
        self.assertTrue(fmt.hdr)
        # denormal: e 0, m 32 -> 0.5 * 2^-14
        _, px = self.decode(0x0F, struct.pack('<I', 32))
        self.assertAlmostEqual(px[0, 0], 0.5 * 2 ** -14, places=12)

    def test_short_data(self):
        fmt = self.pf.TextureFormat.get(0x0B)()
        with self.assertRaises(ValueError):
            fmt.decompress(FakeTex(b'\0' * 7, 2, 1))


@unittest.skipIf(np is None, "numpy not available")
class ChannelSelectorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ch = pf().channels

    def test_strings(self):
        self.assertEqual(self.ch.to_string((2, 2, 2, 1)), 'RRR1')
        self.assertTrue(self.ch.is_identity((2, 3, 4, 5)))

    def test_apply(self):
        px = np.array([[.1, .2, .3, .4]], np.float32)
        out = self.ch.apply(px, (4, 3, 2, 1))        # BGR1
        np.testing.assert_allclose(out, [[.3, .2, .1, 1]], atol=1e-7)
        out = self.ch.apply(px, (5, 0, 1, 2))        # A01R
        np.testing.assert_allclose(out, [[.4, 0, 1, .1]], atol=1e-7)

    def test_stored_components(self):
        # BC4 / R8 decode as gray: RRRR must give R in alpha, R000 red only
        gray = np.array([[.5, .5, .5, 1]], np.float32)
        np.testing.assert_allclose(self.ch.apply(gray, (2, 2, 2, 2), 1), [[.5] * 4])
        np.testing.assert_allclose(self.ch.apply(gray, (2, 0, 0, 0), 1), [[.5, 0, 0, 0]])
        # BC5 with a rebuilt Z in blue: RG11 gives blue 1, RGGG blue = green
        bc5 = np.array([[.2, .7, .9, 1]], np.float32)
        np.testing.assert_allclose(self.ch.apply(bc5, (2, 3, 1, 1), 2), [[.2, .7, 1, 1]])
        np.testing.assert_allclose(self.ch.apply(bc5, (2, 3, 3, 3), 2), [[.2, .7, .7, .7]])
        np.testing.assert_allclose(self.ch.apply(bc5, (2, 2, 2, 3), 2), [[.2, .2, .2, .7]])
        # selecting the stored blue of a 2-channel format reads 0
        np.testing.assert_allclose(self.ch.apply(bc5, (4, 4, 4, 5), 2), [[0, 0, 0, 1]])


if __name__ == "__main__":
    unittest.main()
