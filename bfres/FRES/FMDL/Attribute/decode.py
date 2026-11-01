"""Vectorized decoding of FVTX vertex attributes (numpy).

Switch BFRES attribute format IDs (SwitchAttribFormat in BfresLibrary) are
`0xLLTT`: LL is the component layout, TT the component type.

  layouts: 0x02 8, 0x09 8_8, 0x0B 8_8_8_8, 0x0E 10_10_10_2,
           0x12 16_16, 0x15 16_16_16_16, 0x17 32_32, 0x18 32_32_32,
           0x19 32_32_32_32
  types:   0x01 UNorm, 0x02 SNorm, 0x03 UInt, 0x04 SInt, 0x05 Float

`decode()` returns an (n, components) array: float32 for normalized and
float formats (UNorm -> 0..1, SNorm -> -1..1), int32 for integer formats.
"""
import numpy as np

UNORM, SNORM, UINT, SINT, FLOAT = 1, 2, 3, 4, 5

# layout -> (components, bits per component)
LAYOUTS = {
    0x02: (1, 8), 0x09: (2, 8), 0x0B: (4, 8),
    0x12: (2, 16), 0x15: (4, 16),
    0x14: (1, 32), 0x17: (2, 32), 0x18: (3, 32), 0x19: (4, 32),
}
PACKED_10_10_10_2 = 0x0E

_INT_DTYPES = {
    (8, False): np.uint8, (8, True): np.int8,
    (16, False): np.uint16, (16, True): np.int16,
    (32, False): np.uint32, (32, True): np.int32,
}


class AttributeFormatError(ValueError):
    pass


def components(fmt_id):
    layout = fmt_id >> 8
    if layout == PACKED_10_10_10_2:
        return 4
    try:
        return LAYOUTS[layout][0]
    except KeyError:
        raise AttributeFormatError("Unknown attribute layout in format 0x%04X" % fmt_id)


def element_size(fmt_id):
    layout = fmt_id >> 8
    if layout == PACKED_10_10_10_2:
        return 4
    n, bits = LAYOUTS[layout]
    return n * bits // 8


def _strided(data, offset, stride, count, itemsize):
    """(count, itemsize) uint8 view of `data` starting at `offset`."""
    buf = np.frombuffer(data, dtype=np.uint8)
    end = offset + (count - 1) * stride + itemsize if count else offset
    if offset < 0 or end > buf.size:
        raise AttributeFormatError(
            "Attribute reads past its buffer (offset 0x%X, stride %d, %d vertices, buffer 0x%X bytes)"
            % (offset, stride, count, buf.size))
    return np.lib.stride_tricks.as_strided(
        buf[offset:], shape=(count, itemsize), strides=(stride, 1), writeable=False)


def _arm_half(raw):
    """ARM alternative half precision (no Inf/NaN; exponent 31 is a normal
    exponent) -> float32."""
    raw = raw.astype(np.uint16)
    out = raw.view(np.float16).astype(np.float32)
    big = (raw & 0x7C00) == 0x7C00
    if big.any():
        r = raw[big].astype(np.int64)
        mant = (r & 0x3FF).astype(np.float32) / 1024.0
        sign = np.where(r & 0x8000, -1.0, 1.0).astype(np.float32)
        out[big] = sign * np.float32(65536.0) * (1.0 + mant)
    return out


def decode(fmt_id, data, offset, stride, count):
    layout, typ = fmt_id >> 8, fmt_id & 0xFF

    if layout == PACKED_10_10_10_2:
        raw = _strided(data, offset, stride, count, 4).copy().view('<u4').reshape(count)
        out = np.empty((count, 4), np.float32)
        for i in range(3):
            v = ((raw >> (10 * i)) & 0x3FF).astype(np.int32)
            if typ == SNORM:
                v = np.where(v & 0x200, v - 1024, v)
                out[:, i] = np.maximum(v / 511.0, -1.0)
            else:
                out[:, i] = v / 1023.0
        w = (raw >> 30).astype(np.int32)
        if typ == SNORM:
            w = np.where(w & 2, w - 4, w)
            out[:, 3] = np.maximum(w, -1)
        else:
            out[:, 3] = w / 3.0
        return out

    if layout not in LAYOUTS:
        raise AttributeFormatError("Unknown attribute layout in format 0x%04X" % fmt_id)
    n, bits = LAYOUTS[layout]
    size = n * bits // 8
    raw = _strided(data, offset, stride, count, size).copy()

    if typ == FLOAT:
        if bits == 16:
            return _arm_half(raw.view('<u2')).reshape(count, n)
        if bits == 32:
            return raw.view('<f4').astype(np.float32).reshape(count, n)
        raise AttributeFormatError("No %d-bit float attributes (0x%04X)" % (bits, fmt_id))

    signed = typ in (SNORM, SINT)
    dt = np.dtype(_INT_DTYPES[(bits, signed)]).newbyteorder('<')
    vals = raw.view(dt).reshape(count, n)
    if typ == UNORM:
        return (vals / float((1 << bits) - 1)).astype(np.float32)
    if typ == SNORM:
        return np.maximum(vals / float((1 << (bits - 1)) - 1), -1.0).astype(np.float32)
    if typ in (UINT, SINT):
        return vals.astype(np.int64 if bits == 32 else np.int32)
    raise AttributeFormatError("Unknown attribute component type in format 0x%04X" % fmt_id)
