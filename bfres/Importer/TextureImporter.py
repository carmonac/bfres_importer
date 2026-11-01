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

            image = bpy.data.images.new(name=tex.name,
                width=tex.width, height=tex.height)
            # image.use_alpha = True

            pixels = tex.format_.decodePixels(tex.pixels)
            if type(tex.format_).__name__ == 'BC5' and not _isNormalMap(tex.name):
                # BC5 stores two channels. decodePixels rebuilds a normal
                # map's Z in blue, which is meaningless for other data
                # (e.g. BotW _Spm specular masks): leave blue empty there.
                pixels = np.array(pixels, dtype=np.float32).reshape(-1, 4)
                pixels[:, 2] = 0.0

            # flip image from dx to gl
            pixels = np.flipud(pixels.reshape((tex.height, tex.width, 4)))

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
