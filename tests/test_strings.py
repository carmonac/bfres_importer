"""String decoding tests (issue #34). Run with: python3 -m unittest discover tests

BotW files store names as UTF-8, and some contain Japanese text written by
the artists (e.g. Enemy_SiteBoss_Spear). They used to be decoded as
Shift-JIS, which aborted the import.
"""
import importlib.util
import io
import logging
import os
import struct
import sys
import types
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_text():
    spec = importlib.util.spec_from_file_location(
        "bfres_text", os.path.join(ROOT, "bfres", "BinaryStruct", "text.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_bfres_subpackage(name):
    """Import bfres.<name> without running bfres/__init__.py (needs bpy)."""
    if "strtest" not in sys.modules:
        pkg = types.ModuleType("strtest")
        pkg.__path__ = [os.path.join(ROOT, "bfres")]
        sys.modules["strtest"] = pkg
    return importlib.import_module("strtest." + name)


text = load_text()

# 'ハイトマップ' (height map), the string that broke Enemy_SiteBoss_Spear
HEIGHTMAP = "ハイトマップ"


class DecodeStrTest(unittest.TestCase):
    def test_ascii(self):
        self.assertEqual(text.decode_str(b"Mt_Body"), "Mt_Body")
        self.assertEqual(text.decode_str(b""), "")

    def test_bytes_like(self):
        self.assertEqual(text.decode_str(bytearray(b"abc")), "abc")
        self.assertEqual(text.decode_str(memoryview(b"xabcx")[1:4]), "abc")

    def test_utf8(self):
        raw = HEIGHTMAP.encode("utf-8")
        self.assertEqual(raw[:6], b"\xe3\x83\x8f\xe3\x82\xa4")  # bytes from #34
        self.assertEqual(text.decode_str(raw), HEIGHTMAP)
        self.assertEqual(text.decode_str("Mt_été".encode()), "Mt_été")

    def test_shift_jis_fallback(self):
        raw = HEIGHTMAP.encode("shift-jis")
        with self.assertRaises(UnicodeDecodeError):
            raw.decode("utf-8")
        self.assertEqual(text.decode_str(raw), HEIGHTMAP)

    def test_invalid_does_not_raise(self):
        raw = b"ab\xff\xfecd\x80"  # neither UTF-8 nor Shift-JIS
        with self.assertLogs(text.log, logging.WARNING) as cm:
            s = text.decode_str(raw, 0x1234)
        self.assertEqual(s, raw.decode("utf-8", "replace"))
        self.assertTrue(s.startswith("ab") and "�" in s)
        self.assertIn("0x1234", cm.output[0])

    def test_custom_encodings(self):
        self.assertEqual(text.decode_str(b"\xe9", encodings=("latin-1",)), "é")


def string_table(strings, start=0x10):
    """Build a _STR block like the one at the end of an FRES."""
    buf = bytearray(start)
    buf += b"_STR" + b"\0" * 4 + struct.pack("<I", 0) + b"\0" * 4
    buf += struct.pack("<I", len(strings))
    offsets = []
    for s in strings:
        if len(buf) & 1:
            buf.append(0)
        raw = s.encode("utf-8") if isinstance(s, str) else s
        offsets.append(len(buf))
        buf += struct.pack("<H", len(raw)) + raw + b"\0"
    return bytes(buf), offsets


def mem_file(data):
    BinaryFile = load_bfres_subpackage("BinaryFile").BinaryFile
    f = io.BytesIO(data)
    f.name = "<test>"
    return BinaryFile(f)


class StringTableTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.ST = load_bfres_subpackage("Common.StringTable")
            cls.SO = load_bfres_subpackage("BinaryStruct.StringOffset")
        except ImportError as ex:  # e.g. numpy missing
            raise unittest.SkipTest("bfres package not importable: %s" % ex)

    def test_mixed_table(self):
        names = ["Mt_Body", HEIGHTMAP, "Spear", HEIGHTMAP.encode("shift-jis"),
                 b"bad\xff"]
        data, offsets = string_table(names)
        with self.assertLogs("strtest.BinaryStruct.text", logging.WARNING) as cm:
            st = self.ST.StringTable().readFromFile(mem_file(data), 0x10)
        self.assertEqual(len(cm.output), 1)  # only the invalid one warns
        got = [st.strings[o] for o in offsets]
        self.assertEqual(got[:4], ["Mt_Body", HEIGHTMAP, "Spear", HEIGHTMAP])
        self.assertEqual(got[4], "bad�")

    def test_string_offset_auto(self):
        raw = HEIGHTMAP.encode("utf-8")
        data = struct.pack("<I", 8) + b"\0" * 4 + struct.pack("<H", len(raw)) + raw
        field = self.SO.StringOffset("name", "<I", 1024, "auto", "<H")
        self.assertEqual(field.readFromFile(mem_file(data), 0), HEIGHTMAP)

    def test_switch_string_default(self):
        Switch = load_bfres_subpackage("BinaryStruct.Switch")
        self.assertEqual(Switch.String("name").encoding, "auto")


class AnimStrTest(unittest.TestCase):
    def test_fska_str(self):
        try:
            spec = importlib.util.spec_from_file_location(
                "fska_strtest", os.path.join(ROOT, "bfres", "FRES", "FSKA.py"))
            fska = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(fska)
        except ImportError as ex:
            self.skipTest("FSKA not importable: %s" % ex)
        raw = HEIGHTMAP.encode("utf-8")
        data = b"\0\0" + struct.pack("<H", len(raw)) + raw
        self.assertEqual(fska._str(data, 2), HEIGHTMAP)
        self.assertIsNone(fska._str(data, 0))


if __name__ == "__main__":
    unittest.main()
