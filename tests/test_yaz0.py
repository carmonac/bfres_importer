"""Yaz0 decoder tests. Run with: python3 -m unittest discover tests

They load bfres/YAZ0/Decoder.py on its own, so neither Blender nor game
files are needed.
"""
import importlib.util
import os
import random
import struct
import sys
import types
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_decoder():
    # Decoder.py does `from ..BinaryFile import BinaryFile`; give it a stub
    # package so it can be imported without the rest of the add-on (and bpy).
    pkg = types.ModuleType("yaz0test")
    pkg.__path__ = []
    sys.modules["yaz0test"] = pkg
    bf = types.ModuleType("yaz0test.BinaryFile")
    bf.BinaryFile = object
    sys.modules["yaz0test.BinaryFile"] = bf
    sub = types.ModuleType("yaz0test.YAZ0")
    sub.__path__ = []
    sys.modules["yaz0test.YAZ0"] = sub
    spec = importlib.util.spec_from_file_location(
        "yaz0test.YAZ0.Decoder", os.path.join(ROOT, "bfres", "YAZ0", "Decoder.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


Decoder = load_decoder()


def compress(data):
    """Small greedy Yaz0 compressor, good enough to exercise every code path."""
    out = bytearray(b"Yaz0" + struct.pack(">I", len(data)) + b"\0" * 8)
    i = 0
    while i < len(data):
        head = len(out)
        out.append(0)
        code = 0
        for bit in range(8):
            if i >= len(data):
                break
            best_len, best_dist = 0, 0
            for dist in range(1, min(i, 0x1000) + 1):
                n = 0
                while n < 0x111 and i + n < len(data) and data[i + n - dist] == data[i + n]:
                    n += 1
                if n > best_len:
                    best_len, best_dist = n, dist
                    if n == 0x111:
                        break
            if best_len >= 3:
                d = best_dist - 1
                if best_len <= 0x11:
                    out += bytes((((best_len - 2) << 4) | (d >> 8), d & 0xFF))
                else:
                    out += bytes((d >> 8, d & 0xFF, best_len - 0x12))
                i += best_len
            else:
                code |= 0x80 >> bit
                out.append(data[i])
                i += 1
        out[head] = code
    return bytes(out)


class Yaz0Test(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(Decoder.decompress(compress(b"")), b"")

    def test_literals_only(self):
        data = bytes(range(256)) * 3
        self.assertEqual(Decoder.decompress(compress(data)), data)

    def test_overlapping_runs(self):
        # long runs force 3-byte references with dist < length
        data = b"a" * 1000 + b"ab" * 500 + b"abc" * 333
        self.assertEqual(Decoder.decompress(compress(data)), data)

    def test_random(self):
        rng = random.Random(1234)
        for _ in range(30):
            alpha = rng.choice([b"a", b"ab", b"abcdef", bytes(range(256))])
            n = rng.randint(0, 2500)
            data = bytes(rng.choice(alpha) for _ in range(n))
            data += data[: n // 2]
            self.assertEqual(Decoder.decompress(compress(data)), data)

    def test_progress_reports_total(self):
        data = b"xyz" * 4000
        calls = []
        Decoder.decompress(compress(data), lambda cur, total: calls.append((cur, total)))
        self.assertEqual(calls[0], (0, len(data)))
        self.assertEqual(calls[-1], (len(data), len(data)))

    def test_bad_magic(self):
        with self.assertRaises(ValueError):
            Decoder.decompress(b"FRES" + b"\0" * 12)

    def test_truncated(self):
        c = compress(bytes(range(200)))
        with self.assertRaises(ValueError):
            Decoder.decompress(c[:-20])


if __name__ == "__main__":
    unittest.main()
