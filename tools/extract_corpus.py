#!/usr/bin/env python3
"""Extract a set of test models from a game's RomFS.

The source can be:
  * a plain extracted RomFS folder, or
  * a single-file RomFS image in the ``DMROMFS1`` format used by
    nxnative / NX2Apple (``<Game>.app/Contents/Resources/romfs.img``).

DMROMFS1 layout (little endian):
  header  32 bytes: magic b"DMROMFS1", u64 index offset, u64 index size,
          u64 entry count
  data    file contents back to back
  index   entry count x 24 bytes: u64 a, u64 b, u32 name offset,
          u16 name length, u8 type (0 = dir, 1 = file), u8 pad
          dir:  a = first child entry, b = number of children
          file: a = data offset in the image, b = size
          followed by the name pool. Entry 0 is the root.

Usage:
  extract_corpus.py SOURCE OUT [--list FILE] [--grep REGEX]

The corpus list has one entry per line: a model base name (``Animal_Fox``)
pulls ``Model/<name>.sbfres`` and, when present, ``Model/<name>.Tex.sbfres``.
An entry ending in ``.sbfres`` (or any other extension) is copied as-is.
Blank lines and ``#`` comments are ignored.

Written to run on any Python >= 3.8 (macOS system python included); it has
no third-party dependencies and never modifies the source.
"""
import argparse
import os
import re
import shutil
import struct
import sys

HEAD = struct.Struct("<8sQQQ")
ENT = struct.Struct("<QQIHBB")
MAGIC = b"DMROMFS1"
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LIST = os.path.join(HERE, "corpus", "botw.txt")


class ImageRomFS:
    """Read-only view of a DMROMFS1 image."""

    def __init__(self, path):
        self.path = path
        with open(path, "rb") as f:
            magic, ioff, isz, n = HEAD.unpack(f.read(HEAD.size))
            if magic != MAGIC:
                raise ValueError("%s is not a DMROMFS1 image" % path)
            f.seek(ioff)
            raw = f.read(isz)
        if len(raw) != isz or isz < n * ENT.size:
            raise ValueError("%s: truncated index" % path)
        pool = raw[n * ENT.size:]
        ents = [ENT.unpack_from(raw, k * ENT.size) for k in range(n)]
        self.files = {}  # "Model/X.sbfres" -> (offset, size)
        todo = [(0, "")]
        while todo:
            k, prefix = todo.pop()
            a, b, _no, _nl, typ, _ = ents[k]
            if typ != 0:
                continue
            for c in range(a, a + b):
                ca, cb, no, nl, ctyp, _ = ents[c]
                name = pool[no:no + nl].decode("utf-8")
                path = prefix + "/" + name if prefix else name
                if ctyp == 0:
                    todo.append((c, path))
                else:
                    self.files[path] = (ca, cb)

    def names(self):
        return self.files.keys()

    def copy(self, name, dst):
        off, size = self.files[name]
        with open(self.path, "rb") as src, open(dst, "wb") as out:
            src.seek(off)
            left = size
            while left:
                chunk = src.read(min(left, 8 << 20))
                if not chunk:
                    raise IOError("%s: short read for %s" % (self.path, name))
                out.write(chunk)
                left -= len(chunk)


class FolderRomFS:
    """An extracted RomFS directory."""

    def __init__(self, path):
        self.path = path
        self.files = {}
        for root, _dirs, files in os.walk(path):
            for fn in files:
                full = os.path.join(root, fn)
                rel = os.path.relpath(full, path).replace(os.sep, "/")
                self.files[rel] = full

    def names(self):
        return self.files.keys()

    def copy(self, name, dst):
        shutil.copyfile(self.files[name], dst)


def open_romfs(path):
    if os.path.isdir(path):
        return FolderRomFS(path)
    return ImageRomFS(path)


def read_list(path):
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if line:
                items.append(line)
    return items


def resolve(fs, items):
    """Map corpus entries to RomFS paths. Returns (paths, missing)."""
    by_lower = {}
    for n in fs.names():
        by_lower.setdefault(n.lower(), n)
    wanted, missing = [], []

    def find(rel):
        for cand in (rel, "Model/" + rel):
            hit = by_lower.get(cand.lower())
            if hit:
                return hit
        return None

    for item in items:
        if "." in os.path.basename(item):
            hit = find(item)
            (wanted.append(hit) if hit else missing.append(item))
            continue
        hit = find(item + ".sbfres")
        if not hit:
            missing.append(item)
            continue
        wanted.append(hit)
        tex = find(item + ".Tex.sbfres")
        if tex:
            wanted.append(tex)
    return wanted, missing


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("source", help="romfs.img (DMROMFS1) or an extracted RomFS folder")
    ap.add_argument("out", help="destination folder (keep it outside version control)")
    ap.add_argument("--list", default=DEFAULT_LIST,
                    help="corpus list file (default: tools/corpus/botw.txt)")
    ap.add_argument("--grep", metavar="REGEX",
                    help="only print RomFS paths matching REGEX and exit")
    args = ap.parse_args(argv)

    fs = open_romfs(args.source)

    if args.grep:
        rx = re.compile(args.grep, re.I)
        for n in sorted(fs.names()):
            if rx.search(n):
                print(n)
        return 0

    paths, missing = resolve(fs, read_list(args.list))
    os.makedirs(args.out, exist_ok=True)
    total = 0
    for p in paths:
        dst = os.path.join(args.out, os.path.basename(p))
        fs.copy(p, dst)
        size = os.path.getsize(dst)
        total += size
        print("%-48s %10d" % (os.path.basename(p), size))
    print("%d files, %.1f MB -> %s" % (len(paths), total / 1e6, args.out))
    for m in missing:
        print("MISSING: %s" % m, file=sys.stderr)
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
