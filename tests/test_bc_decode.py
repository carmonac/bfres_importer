"""Texture block decoder tests. Needs numpy (bundled with Blender):
    python3 -m unittest discover tests

The native-decoder tests run only when texture2ddecoder can be loaded
(installed, or from a matching wheel in ./wheels); the comparisons against
reference implementations only when these optional packages are installed:
imagecodecs (bcdec: BC1-BC7), etcpak (BC7 encoder), astc_encoder (ARM's
astcenc: ASTC encoder/decoder).
"""
import importlib.util
import os
import struct
import sys
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


def pixelfmt():
    """The bfres.BNTX.pixelfmt package (needs no bpy)."""
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    import bfres.BNTX.pixelfmt as pf
    import bfres.BNTX.pixelfmt.native
    import bfres.BNTX.pixelfmt.BCn.bptc
    return pf


def optional(name):
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


class BitWriter:
    """Builds a 128-bit block field by field, LSB first."""
    def __init__(self):
        self.v = 0
        self.n = 0

    def put(self, value, bits):
        self.v |= (value & ((1 << bits) - 1)) << self.n
        self.n += bits
        return self

    def block(self):
        assert self.n == 128, self.n
        return self.v.to_bytes(16, "little")


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


class FakeTex:
    """The parts of a BRTI the format classes read."""
    def __init__(self, data, w, h, dtype=1):
        self.mipData = data
        self.width = w
        self.height = h
        self.header = {"fmt_dtype": dtype}


@unittest.skipIf(np is None, "numpy not available")
class BPTCTest(unittest.TestCase):
    """BC6H / BC7 numpy decoders on hand-built blocks."""

    @classmethod
    def setUpClass(cls):
        cls.b = pixelfmt().BCn.bptc

    def test_bc7_mode6(self):
        # mode 6: 7-bit RGBA endpoints + one p-bit each, 4-bit indices
        e0, e1 = (5, 10, 15, 127), (100, 110, 120, 127)
        w = BitWriter().put(1 << 6, 7)
        for c in range(4):
            w.put(e0[c], 7).put(e1[c], 7)
        w.put(0, 1).put(1, 1)                        # p-bits
        idx = [p % 8 if p == 0 else p for p in range(16)]
        w.put(idx[0], 3)
        for i in idx[1:]:
            w.put(i, 4)
        img = np.frombuffer(self.b.bc7(w.block(), 4, 4), np.uint8).reshape(16, 4)
        self.assertEqual(img[0].tolist(), [10, 20, 30, 254])      # (v << 1) | p
        self.assertEqual(img[15].tolist(), [201, 221, 241, 255])
        self.assertEqual(img[8, 0], (10 * 30 + 201 * 34 + 32) >> 6)  # weight 34

    def test_bc7_reserved(self):
        img = self.b.bc7(bytes(16), 4, 4)
        self.assertEqual(img, bytes(64))

    def test_bc6h_mode11_unsigned(self):
        # mode 0x03: one region, 10-bit endpoints, no delta, 4-bit indices
        w = BitWriter().put(0x03, 5)
        for v in (0, 0, 0, 1023, 1023, 1023):  # rw gw bw rx gx bx
            w.put(v, 10)
        w.put(0, 3)
        for i in range(1, 16):
            w.put(i, 4)
        img = self.b.bc6h(w.block(), 4, 4, False).reshape(16, 4)
        self.assertEqual(img.dtype, np.float32)
        self.assertEqual(img[0, :3].tolist(), [0, 0, 0])
        self.assertEqual(img[15, :3].tolist(), [65504.0] * 3)    # max half
        self.assertAlmostEqual(float(img[8, 0]), 2.935546875)     # weight 34
        self.assertTrue((img[:, 3] == 1).all())

    def test_bc6h_signed_negative(self):
        w = BitWriter().put(0x03, 5)
        for v in (-512, -512, -512, 511, 511, 511):  # 10-bit two's complement
            w.put(v, 10)
        w.put(0, 3)
        for i in range(1, 16):
            w.put(i, 4)
        img = self.b.bc6h(w.block(), 4, 4, True).reshape(16, 4)
        self.assertEqual(img[0, 0], -img[15, 0])
        self.assertLess(img[0, 0], -60000)

    def test_bc6h_reserved_is_black(self):
        img = self.b.bc6h(bytes([0x13]) + bytes(15), 4, 4, False)
        self.assertTrue((img[..., :3] == 0).all())

    def test_sizes(self):
        data = np.random.default_rng(0).integers(0, 256, 4 * 2 * 16, np.uint8).tobytes()
        self.assertEqual(len(self.b.bc7(data, 13, 7)), 13 * 7 * 4)
        self.assertEqual(self.b.bc6h(data, 13, 7).shape, (7, 13, 4))


@unittest.skipIf(np is None or optional("imagecodecs") is None, "imagecodecs not installed")
class ReferenceTest(unittest.TestCase):
    """Random blocks reach every mode and partition: compare with bcdec."""

    @classmethod
    def setUpClass(cls):
        cls.b = pixelfmt().BCn.bptc
        cls.ic = optional("imagecodecs")
        cls.data = np.random.default_rng(1).integers(0, 256, 64 * 64 // 16 * 16, np.uint8).tobytes()

    def test_bc7(self):
        ref = self.ic.bcn_decode(self.data, 7, shape=(64, 64, 4))
        out = np.frombuffer(self.b.bc7(self.data, 64, 64), np.uint8).reshape(64, 64, 4)
        self.assertTrue((out == ref).all())

    def test_bc6h(self):
        for signed in (False, True):
            ref = self.ic.bcn_decode(self.data, -6 if signed else 6, shape=(64, 64, 3))
            out = self.b.bc6h(self.data, 64, 64, signed)[..., :3]
            ref = ref.astype(np.float32)
            same = (out == ref) | (np.isnan(out) & np.isnan(ref))
            self.assertTrue(same.all(), "signed=%s" % signed)


def native_module():
    if np is None:
        return None
    return pixelfmt().native.get()


@unittest.skipIf(native_module() is None, "native decoder (texture2ddecoder) not available")
class NativeTest(unittest.TestCase):
    """The native path must give the same pixels as the numpy decoders."""

    @classmethod
    def setUpClass(cls):
        cls.pf = pixelfmt()
        cls.native = cls.pf.native
        cls.rng = np.random.default_rng(2)

    def both(self, cls, data, w, h, dtype=1):
        fmt = cls()
        self.native.set_enabled(True)
        try:
            a = fmt.decompress(FakeTex(data, w, h, dtype))
            self.assertEqual(fmt.decoder, "native")
            self.native.set_enabled(False)
            b = fmt.decompress(FakeTex(data, w, h, dtype))
            self.assertEqual(fmt.decoder, "python")
        finally:
            self.native.set_enabled(True)
        return np.asarray(fmt.decodePixels(a)), np.asarray(fmt.decodePixels(b))

    def test_same_as_python(self):
        BCn = self.pf.BCn
        for cls, bs in ((BCn.BC1, 8), (BCn.BC3, 16), (BCn.BC4, 8), (BCn.BC5, 16), (BCn.BC7, 16)):
            for w, h in ((64, 64), (13, 7)):
                nb = ((w + 3) // 4) * ((h + 3) // 4)
                data = self.rng.integers(0, 256, nb * bs, np.uint8).tobytes()
                a, b = self.both(cls, data, w, h)
                self.assertEqual(a.shape, b.shape)
                self.assertEqual(float(np.abs(a - b).max()), 0.0, "%s %dx%d" % (cls.__name__, w, h))

    def test_bc1_three_color_alpha(self):
        # native decoders make index 3 opaque; the spec says transparent
        blk = bc1_block(0x0000, 0xFFFF, [3] * 16)
        a, _ = self.both(self.pf.BCn.BC1, blk, 4, 4)
        self.assertTrue((a.reshape(16, 4)[:, 3] == 0).all())

    def test_bc3_always_four_color(self):
        blk = alpha_block(255, 255, [0] * 16) + bc1_block(0x0000, 0xFFFF, [3] * 16)
        a, _ = self.both(self.pf.BCn.BC3, blk, 4, 4)
        # 4-color index 3 = (c0 + 2*c1) / 3 = 170, not BC1's black
        self.assertTrue((a.reshape(16, 4)[:, 0] * 255 == 170).all())

    def test_astc_void_extent(self):
        # constant-color block: 0x...FDFC header, then RGBA as UNORM16
        blk = (0xFFFFFFFFFFFFFDFC).to_bytes(8, "little") + struct.pack("<4H", 0xFFFF, 0, 0x8080, 0xFFFF)
        cls = self.pf.fmts[0x2d]
        out = np.frombuffer(cls().decompress(FakeTex(blk, 4, 4)), np.uint8).reshape(16, 4)
        self.assertTrue((out == [255, 0, 128, 255]).all())
        # 12x12 blocks, size not a multiple of the block
        cls = self.pf.fmts[0x3a]
        out = cls().decompress(FakeTex(blk * 4, 20, 13))
        self.assertEqual(len(out), 20 * 13 * 4)

    def test_astc_all_sizes_registered(self):
        for i, (bw, bh) in enumerate(self.pf.BCn.ASTC_SIZES):
            cls = self.pf.fmts[0x2d + i]
            self.assertEqual((cls.blkWidth, cls.blkHeight), (bw, bh))

    @unittest.skipIf(optional("astc_encoder") is None, "astc_encoder not installed")
    def test_astc_vs_astcenc(self):
        import astc_encoder as ae
        w, h = 40, 30
        y, x = np.mgrid[0:h, 0:w]
        img = np.stack((x * 6, y * 8, (x + y) * 3, 255 - x * 2), -1).astype(np.uint8)
        for i, (bw, bh) in enumerate(self.pf.BCn.ASTC_SIZES):
            cfg = ae.ASTCConfig(ae.ASTCProfile.LDR, bw, bh, 1, 60)
            ctx = ae.ASTCContext(cfg)
            src = ae.ASTCImage(ae.ASTCType.U8, w, h, 1, img.tobytes())
            comp = ctx.compress(src, ae.ASTCSwizzle.from_str("RGBA"))
            dst = ae.ASTCImage(ae.ASTCType.U8, w, h, 1)
            ref = np.frombuffer(ctx.decompress(comp, dst, ae.ASTCSwizzle.from_str("RGBA")).data,
                                np.uint8).reshape(h, w, 4).astype(int)
            out = self.pf.fmts[0x2d + i]().decompress(FakeTex(comp, w, h))
            out = np.frombuffer(out, np.uint8).reshape(h, w, 4).astype(int)
            self.assertLessEqual(np.abs(out - ref).max(), 1, "ASTC %dx%d" % (bw, bh))

    @unittest.skipIf(optional("etcpak") is None, "etcpak not installed")
    def test_bc7_encoded_image(self):
        import etcpak
        y, x = np.mgrid[0:32, 0:32]
        img = np.stack((x * 8, y * 8, 255 - x * 4, 128 + y * 2), -1).astype(np.uint8)
        comp = etcpak.compress_bc7(img.tobytes(), 32, 32)
        a, b = self.both(self.pf.BCn.BC7, comp, 32, 32)
        self.assertEqual(float(np.abs(a - b).max()), 0.0)
        err = np.abs(a.reshape(32, 32, 4) * 255 - img).mean()
        self.assertLess(err, 4)   # a sane encode/decode round trip


if __name__ == "__main__":
    unittest.main()
