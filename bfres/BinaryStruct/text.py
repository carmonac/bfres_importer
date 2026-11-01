"""Decoding of the strings stored in Switch BFRES/BNTX files.

Nintendo's tools write UTF-8. Breath of the Wild models contain a few
Japanese names (bone/material user data written by the artists, e.g.
'ハイトマップ'), so the old hard-coded Shift-JIS failed on them and aborted
the whole import. Order tried here:

1. UTF-8 (strict): what the files actually use. Plain ASCII is identical
   in every candidate encoding, so most strings never get past this step.
2. Shift-JIS (strict): kept as a fallback in case some file really uses it
   (none found when scanning the BotW RomFS, see tools/scan_strings.py).
3. UTF-8 with errors='replace' and a warning, so one broken name never
   stops an import.

This module has no dependencies on the rest of the add-on (nor on bpy), so
it can be unit-tested on its own.
"""
import logging; log = logging.getLogger(__name__)

FALLBACK_ENCODINGS = ('utf-8', 'shift-jis')


def decode_str(data, where=None, encodings=FALLBACK_ENCODINGS):
    """Decode raw string bytes from a BFRES/BNTX file into a str.

    data:      bytes-like object (without length prefix or terminator).
    where:     optional description of the location (e.g. an offset),
        used only in the warning when nothing decodes cleanly.
    encodings: encodings tried strictly, in order.

    Never raises UnicodeDecodeError.
    """
    data = bytes(data)
    if data.isascii():
        return data.decode('ascii')
    for enc in encodings:
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            pass
    text = data.decode(encodings[0], 'replace')
    if isinstance(where, int): where = '0x%X' % where
    log.warning("String%s is not valid %s; decoded as %r",
        '' if where is None else ' at ' + str(where),
        '/'.join(encodings), text)
    return text
