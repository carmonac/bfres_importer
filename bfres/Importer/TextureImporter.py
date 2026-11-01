import logging; log = logging.getLogger(__name__)
import bmesh
import bpy
import bpy_extras
import struct
import os
import os.path
import numpy as np
import time

def _isNormalMap(name):
    base = name.rsplit('.', 1)[0] if '.' in name else name
    return base.endswith('_Nrm') or base.endswith('Nrm')


# BRTI.TextureDataType
_SNORM, _SRGB = 2, 6


def _setColorspace(image, dtype, hdr):
    """Initial colorspace from the texture's component type.

    SRGB textures hold sRGB-encoded color, SNORM ones signed data (Non-Color)
    and BC6H linear HDR (the float default, scene linear). UNORM is
    ambiguous: Breath of the Wild stores many albedos as UNORM and applies
    the gamma in its shaders, so it keeps the 8-bit default (sRGB); the
    material code switches the images it wires as data maps (normal,
    roughness, masks...) to Non-Color.
    """
    try:
        if hdr:
            return  # float_buffer images default to linear
        if dtype == _SNORM:
            image.colorspace_settings.is_data = True
        elif dtype == _SRGB:
            image.colorspace_settings.name = 'sRGB'
    except (TypeError, AttributeError):  # unknown colorspace name in this build
        pass


class TextureImporter:
    """Imports texture images from BNTX archive."""

    def __init__(self, parent):
        self.parent   = parent
        self.operator = parent.operator
        self.context  = parent.context


    def importTextures(self, bntx):
        """Import textures from BNTX."""
        images = {}
        for i, tex in enumerate(bntx.textures):
            log.info("Importing texture %3d/%3d '%s' (%s)...",
                i+1, len(bntx.textures), tex.name,
                type(tex.format_).__name__)

            fmt = tex.format_
            hdr = getattr(fmt, 'hdr', False)
            image = bpy.data.images.new(name=tex.name,
                width=tex.width, height=tex.height, float_buffer=hdr)
            # remember the source format: material building needs to know
            # e.g. that a BC4 texture only has one meaningful channel
            image['bfres_format'] = type(fmt).__name__
            dtype = tex.header['fmt_dtype']
            image['bfres_component'] = getattr(dtype, 'name', str(dtype))
            _setColorspace(image, int(dtype), hdr)
            # image.use_alpha = True

            if tex.pixels is None:
                # decoding failed (already logged): magenta placeholder
                image['bfres_decode_error'] = tex.decodeError or 'unknown'
                pixels = np.tile(np.array((1, 0, 1, 1), np.float32),
                    (tex.height, tex.width, 1))
            else:
                decoder = getattr(fmt, 'decoder', None)
                if decoder:
                    image['bfres_decoder'] = decoder
                pixels = fmt.decodePixels(tex.pixels)
                if type(fmt).__name__ == 'BC5' and not _isNormalMap(tex.name):
                    # BC5 stores two channels. decodePixels rebuilds a normal
                    # map's Z in blue, which is meaningless for other data
                    # (e.g. BotW _Spm specular masks): leave blue empty there.
                    pixels = np.array(pixels, dtype=np.float32).reshape(-1, 4)
                    pixels[:, 2] = 0.0

            # flip image from dx to gl
            pixels = np.flipud(np.asarray(pixels).reshape((tex.height, tex.width, 4)))

            # foreach_set copies the whole buffer at once; assigning to
            # image.pixels converts every float through Python.
            image.pixels.foreach_set(
                np.ascontiguousarray(pixels, dtype=np.float32).ravel())

            # save to file
            if self.operator.dump_textures:
                base, ext = os.path.splitext(self.parent.path)
                dirPath = "%s/%s" % (base, bntx.name)
                os.makedirs(dirPath, exist_ok=True)
                image.filepath_raw = "%s/%s.png" % (
                    dirPath, tex.name)
                image.file_format = 'PNG'
                log.info("Saving image to %s", image.filepath_raw)
                image.save()

            image.update()
            image.pack()
            images[tex.name] = image
        return images
