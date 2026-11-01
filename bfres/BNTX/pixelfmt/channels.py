"""BRTI channel selectors (nn::gfx ChannelMapping / "component selectors").

Every BRTI says, for each of the R, G, B and A the shader samples, which
stored component it reads: Zero, One, Red, Green, Blue or Alpha (BRTI
`channel_types`, values 0..5, in R, G, B, A order). Breath of the Wild uses
(survey of every texture header in the RomFS, tools/scan_textures.py):

    BC1/BC3/RGBA8  RGBA (identity), except 6 BC1 normal maps: RGB1
    BC4            RRRR (2868), R000 (162, *_f00 bake maps), RRR1 (9)
    BC5            RG11 (896, normal maps and some _Spm), RGGG (764, _Spm,
                   _Emm...), RRRG (13: Obj_Moon_A albedo, ghost _Fx)
    R8             RRR1

The importer historically ignores them (BC4/R8 as gray with alpha 1, BC5
with the normal's Z rebuilt in blue). `apply()` gives exactly what the GPU
samples instead. Which one BRTI.rgba() returns is set by `mode`:

    'off'  (default) the decoders' own output, unchanged;
    'on'   apply the selectors.

The environment variable BFRES_CHANNEL_SELECTORS (off/on) sets the
default. 'on' changes what materials see (BC4 alpha = red, BC5 normal maps
with blue = 1 instead of the rebuilt Z, RGGG maps with blue = alpha =
green, RRRG albedos gray with alpha = green), so it is only the default
once the material code expects it.
"""
import os
import numpy as np

MODES = ('off', 'on')
mode = os.environ.get('BFRES_CHANNEL_SELECTORS', 'off').strip().lower()
if mode not in MODES:
    mode = 'off'

NAMES = '01RGBA'   # Zero, One, Red, Green, Blue, Alpha


def to_string(selectors):
    """(2, 2, 2, 5) -> 'RRRA'."""
    return ''.join(NAMES[s] if 0 <= s < 6 else '?' for s in selectors)


def is_identity(selectors):
    return tuple(selectors) == (2, 3, 4, 5)


def stored(rgba, channels):
    """The components a format really stores, the rest as the GPU reads
    them (green/blue 0, alpha 1): undoes the decoders' display choices
    (one-channel formats as gray, BC5's rebuilt Z)."""
    rgba = np.array(rgba, np.float32, copy=True).reshape(-1, 4)
    if channels < 4:
        rgba[:, 3] = 1
    if channels < 3:
        rgba[:, 2] = 0
    if channels < 2:
        rgba[:, 1] = 0
    return rgba


def apply(rgba, selectors, channels=4):
    """RGBA pixels (any shape ending in 4 or flat) -> the selected ones,
    as an (N, 4) float32 array."""
    src = stored(rgba, channels)
    out = np.empty_like(src)
    for i, s in enumerate(selectors):
        if s == 0:
            out[:, i] = 0
        elif s == 1:
            out[:, i] = 1
        elif 2 <= s <= 5:
            out[:, i] = src[:, s - 2]
        else:
            raise ValueError("invalid channel selector %r" % (s,))
    return out
