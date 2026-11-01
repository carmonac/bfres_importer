import logging; log = logging.getLogger(__name__)
import struct
from ..BinaryFile import BinaryFile

HEADER_SIZE = 16
PROGRESS_STEP = 1 << 18  # report progress every 256 KiB of output


def decompress(data, progressCallback=None) -> bytes:
    """Decompress a whole Yaz0/Yaz1 stream held in memory.

    `data` is the complete file, header included. Returns the
    decompressed bytes. `progressCallback(cur, total)` is called
    periodically with the number of bytes produced so far.
    """
    data = memoryview(data)
    if bytes(data[:4]) not in (b'Yaz0', b'Yaz1'):
        raise ValueError("Not a Yaz0 stream: %r" % bytes(data[:4]))
    size = struct.unpack_from('>I', data, 4)[0]
    out = bytearray(size)
    if progressCallback:
        progressCallback(0, size)

    try:
        dst = _run(data, out, size, progressCallback)
    except IndexError:
        raise ValueError("Yaz0 stream truncated (%d bytes in)" % len(data)) from None

    if progressCallback:
        progressCallback(size, size)
    return bytes(out)


def _run(data, out, size, progressCallback):
    src = HEADER_SIZE
    dst = 0
    end = len(data)
    next_report = PROGRESS_STEP
    while dst < size:
        if src >= end:
            raise ValueError("Yaz0 stream truncated at 0x%X (%d/%d bytes out)"
                % (src, dst, size))
        code = data[src]; src += 1
        for bit in range(8):
            if dst >= size:
                break
            if code & (0x80 >> bit):
                out[dst] = data[src]
                src += 1; dst += 1
                continue
            b1 = data[src]; b2 = data[src + 1]; src += 2
            dist = (((b1 & 0x0F) << 8) | b2) + 1
            n = b1 >> 4
            if n:
                n += 2
            else:
                n = data[src] + 0x12; src += 1
            copy = dst - dist
            if copy < 0:
                raise ValueError("Yaz0 back reference before start at 0x%X" % src)
            if n > size - dst:
                n = size - dst
            if dist >= n:
                # Source and destination don't overlap: one slice copy.
                out[dst:dst + n] = out[copy:copy + n]
            else:
                # Overlapping copy repeats the last `dist` bytes.
                pattern = out[copy:dst]
                reps, rem = divmod(n, dist)
                out[dst:dst + n] = pattern * reps + pattern[:rem]
            dst += n
        if progressCallback and dst >= next_report:
            progressCallback(dst, size)
            next_report = dst + PROGRESS_STEP
    return dst


class Decoder:
    """YAZ0 decoder.

    Kept for compatibility with existing callers: decompresses the whole
    stream on first use and serves it from memory.
    """
    def __init__(self, file:BinaryFile, progressCallback=None):
        self.file = file
        self.progressCallback = progressCallback
        self._data = None
        self._pos = 0

    def _decode(self) -> bytes:
        if self._data is None:
            raw = self.file.read(self.file.size, 0)
            self._data = decompress(raw, self.progressCallback)
        return self._data

    @property
    def size(self) -> int:
        return len(self._decode())

    def bytes(self):
        """Yield the decompressed stream (as a single chunk)."""
        yield self._decode()

    def read(self, size:int=-1) -> bytes:
        """File-like interface for reading the decompressed stream."""
        data = self._decode()
        if size < 0:
            size = len(data) - self._pos
        res = data[self._pos:self._pos + size]
        self._pos += len(res)
        return res

    def __str__(self):
        return "<Yaz0 stream at 0x%x>" % id(self)
