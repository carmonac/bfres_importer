#!/usr/bin/env python3
"""Find FRES files whose string table has non-ASCII strings (issue #34).

For every FRES (Yaz0-compressed or not) it reads the `_STR` string table,
decompressing only as far as the table's end, and reports each non-ASCII
string with the encodings it decodes as (strict UTF-8, strict Shift-JIS):

    python3 tools/scan_strings.py romfs.img --grep '^Model/.*\\.sbfres$' -j 8
    python3 tools/scan_strings.py ~/Documents/bfres_corpus/botw --out /tmp/s.json

The source is a DMROMFS1 image, an extracted RomFS folder or any folder with
.sbfres/.bfres files (see extract_corpus.py). Pure Python, no numpy needed;
the Yaz0 decoder is the add-on's own, so a whole Model/ folder takes a while
(use -j to spread it over several processes).
"""
import argparse
import collections
import json
import multiprocessing
import os
import re
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

import extract_corpus  # noqa: E402  (RomFS readers)
from bfres.YAZ0.Decoder import _run as _yaz0_run  # noqa: E402

DEFAULT_GREP = r'\.s?bfres$'


class Lazy:
    """Bytes of a (possibly Yaz0-compressed) file, decompressed on demand."""

    def __init__(self, raw):
        self.raw = raw
        self.yaz0 = raw[:4] in (b'Yaz0', b'Yaz1')
        self.size = struct.unpack_from('>I', raw, 4)[0] if self.yaz0 else len(raw)
        self.buf = b'' if self.yaz0 else raw

    def need(self, n):
        n = min(n, self.size)
        if len(self.buf) < n:
            want = min(self.size, max(n, 2 * len(self.buf), 1 << 16))
            out = bytearray(want)
            _yaz0_run(memoryview(self.raw), out, want, None)
            self.buf = bytes(out)
        return self.buf


def encodings_of(raw):
    out = []
    for enc in ('utf-8', 'shift-jis'):
        try:
            raw.decode(enc)
            out.append(enc)
        except UnicodeDecodeError:
            pass
    return out


def scan_fres(f):
    """Non-ASCII strings of the FRES string table: [(offset, bytes)]."""
    head = f.need(0xD0)
    if head[:8] != b'FRES    ':
        return None
    major, minor = struct.unpack_from('<2H', head, 8)
    tab_at = 0xB0 if (major, minor) != (0, 10) else 0xD0
    if (major, minor) == (0, 10):
        head = f.need(0xE0)
    tab, tab_size = struct.unpack_from('<QI', head, tab_at)
    if not tab:
        return []
    start = tab - 0x14  # str_tab_offset points just past the _STR header
    buf = f.need(start + tab_size + 0x14)
    if buf[start:start + 4] != b'_STR':
        raise ValueError("no _STR at 0x%X" % start)
    (count,) = struct.unpack_from('<I', buf, start + 0x10)
    at, out = tab, []
    for _ in range(count):
        at += at & 1
        (n,) = struct.unpack_from('<H', buf, at)
        raw = bytes(buf[at + 2:at + 2 + n])
        if not raw.isascii():
            out.append((at, raw))
        at += n + 3
    return out


def read_raw(fs, name):
    if isinstance(fs, extract_corpus.ImageRomFS):
        off, size = fs.files[name]
        with open(fs.path, 'rb') as fh:
            fh.seek(off)
            return fh.read(size)
    with open(fs.files[name], 'rb') as fh:
        return fh.read()


_FS = None


def _init(src):
    global _FS
    _FS = extract_corpus.open_romfs(src)


def scan_one(name):
    try:
        f = Lazy(read_raw(_FS, name))
        strs = scan_fres(f)
        if strs is None:
            return name, None, None, len(f.buf)
        return name, [{'offset': o, 'hex': r.hex(), 'encodings': encodings_of(r),
                       'text': r.decode('utf-8', 'replace')} for o, r in strs], None, len(f.buf)
    except Exception as ex:  # keep going: report at the end
        return name, None, repr(ex), 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("source")
    ap.add_argument("--grep", default=DEFAULT_GREP, help="RomFS paths to scan (regex)")
    ap.add_argument("--sample", type=int, default=0,
                    help="scan only every Nth matching file")
    ap.add_argument("--limit", type=int, default=0, help="stop after N files")
    ap.add_argument("-j", "--jobs", type=int, default=1)
    ap.add_argument("--out", help="write the results as JSON")
    args = ap.parse_args(argv)

    src = os.path.expanduser(args.source)
    _init(src)
    rx = re.compile(args.grep, re.I)
    names = sorted(n for n in _FS.names() if rx.search(n))
    if args.sample > 1:
        names = names[::args.sample]
    if args.limit:
        names = names[:args.limit]

    found, errors, nfres, nbytes = {}, [], 0, 0
    if args.jobs > 1:
        pool = multiprocessing.Pool(args.jobs, _init, (src,))
        it = pool.imap_unordered(scan_one, names, chunksize=4)
    else:
        it = map(scan_one, names)
    for k, (name, strs, err, size) in enumerate(it):
        nbytes += size
        if err:
            errors.append((name, err))
        elif strs is not None:
            nfres += 1
            if strs:
                found[name] = strs
        if k % 200 == 0:
            print("%d/%d files, %d with non-ASCII strings, %.0f MB decompressed"
                  % (k, len(names), len(found), nbytes / 1e6), file=sys.stderr, flush=True)

    encs = collections.Counter('+'.join(s['encodings']) or 'none'
                               for v in found.values() for s in v)
    texts = collections.Counter(s['text'] for v in found.values() for s in v)
    print("%d files scanned, %d FRES, %d with non-ASCII strings (%d strings)"
          % (len(names), nfres, len(found), sum(len(v) for v in found.values())))
    print("\nstrict encodings that decode them:")
    for k, v in encs.most_common():
        print("  %6d  %s" % (v, k))
    print("\nmost common strings:")
    for k, v in texts.most_common(40):
        print("  %6d  %s" % (v, k))
    print("\nfiles:")
    for n in sorted(found):
        print("  %s: %s" % (n, ', '.join(sorted({s['text'] for s in found[n]}))))
    for n, e in errors:
        print("ERROR %s: %s" % (n, e))
    if args.out:
        with open(args.out, 'w') as fh:
            json.dump({'scanned': len(names), 'fres': nfres, 'files': found,
                       'errors': errors}, fh, ensure_ascii=False, indent=1)
    return 1 if errors else 0


if __name__ == '__main__':
    sys.exit(main())
