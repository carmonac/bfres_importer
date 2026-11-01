import logging; log = logging.getLogger(__name__)
import bmesh
import bpy
import bpy_extras
import struct
import os
import os.path
import numpy as np
import time
import hashlib

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


# Image custom properties
TEXTURE_PROP = 'bfres_texture'   # game texture name (the image may be renamed)
HASH_PROP = 'bfres_hash'         # identity of the texture data, for reuse
FAMILIES_PROP = 'bfres_families'  # models whose files contain it, one per line
ALIASES_PROP = 'bfres_aliases'   # other texture names with the same data


def family(path):
    """The model a file belongs to: its name up to the first dot, so
    Animal_Fox.sbfres, Animal_Fox.Tex.sbfres and Animal_Fox.Tex1.sbfres all
    give 'Animal_Fox'."""
    return os.path.basename(path or '').split('.', 1)[0]


def texture_hash(tex):
    """Identity of a texture: format, size and the raw (deswizzled) data.
    None when the data could not be read."""
    if tex.pixels is None:
        return None
    h = hashlib.sha1()
    h.update(('%s %dx%d ' % (type(tex.format_).__name__, tex.width, tex.height)).encode())
    h.update(memoryview(np.ascontiguousarray(np.asarray(tex.pixels))).cast('B'))
    return h.hexdigest()


def _families(image):
    return [f for f in (image.get(FAMILIES_PROP) or '').split('\n') if f]


def _add_family(image, fam):
    fams = _families(image)
    if fam and fam not in fams:
        image[FAMILIES_PROP] = '\n'.join(fams + [fam])


def _names(image):
    names = [image.get(TEXTURE_PROP)]
    names += [n for n in (image.get(ALIASES_PROP) or '').split('\n') if n]
    return names


def find_image(name, fam=None):
    """The image of game texture `name`, preferring one imported from the
    files of model `fam` (texture names repeat across models: CmnTex_*,
    or the same model imported twice after an edit). Images imported by
    older versions, without the properties, are found by name."""
    candidates = [im for im in bpy.data.images if name in _names(im)]
    if fam:
        for im in candidates:
            if fam in _families(im):
                return im
    if candidates:
        return candidates[0]
    return bpy.data.images.get(name)


def _safe_filename(name):
    return ''.join(c if c.isalnum() or c in '-_.' else '_' for c in name) or 'texture'


class TextureImporter:
    """Imports texture images from BNTX archive.

    Images keep the game texture name when it is free; a different texture
    with a name already taken gets the model as prefix
    ("Enemy_Lynel:CmnTex_Noise"), and a texture identical to an image
    already in the file (same format, size and data) reuses that image.
    find_image() resolves a texture name for a model.

    With texture_storage == 'FILES', images are saved as PNG (EXR for HDR)
    in a folder next to the imported file and referenced from there
    instead of being packed in the .blend.
    """

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
            key = texture_hash(tex)
            fam = family(self.parent.path)
            if key is not None:
                existing = next((im for im in bpy.data.images
                                 if im.get(HASH_PROP) == key and im.has_data), None)
                if existing is not None:
                    log.info("Reusing image %s for texture %s", existing.name, tex.name)
                    _add_family(existing, fam)
                    if tex.name not in _names(existing):
                        aliases = _names(existing)[1:] + [tex.name]
                        existing[ALIASES_PROP] = '\n'.join(aliases)
                    images[tex.name] = existing
                    continue
            name = tex.name
            if bpy.data.images.get(name) is not None:
                name = '%s:%s' % (fam, tex.name)
            image = bpy.data.images.new(name=name,
                width=tex.width, height=tex.height, float_buffer=hdr)
            image[TEXTURE_PROP] = tex.name
            if key is not None:
                image[HASH_PROP] = key
            _add_family(image, fam)
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

            image.update()
            if getattr(self.operator, 'texture_storage', 'PACK') == 'FILES' \
                    and tex.pixels is not None:
                self._saveToFolder(image, tex.name, hdr)
            else:
                image.pack()
            images[tex.name] = image
        return images


    def textureFolder(self):
        """Where texture files go: the operator's texture_dir, or
        <model>_textures next to the imported file."""
        folder = getattr(self.operator, 'texture_dir', '') or ''
        if folder:
            return bpy.path.abspath(folder)
        path = os.path.abspath(self.parent.path)
        return os.path.join(os.path.dirname(path), family(path) + '_textures')


    def _saveToFolder(self, image, texName, hdr):
        """Write `image` to the texture folder and make it reference the
        file (relative to the .blend when it has been saved)."""
        folder = self.textureFolder()
        os.makedirs(folder, exist_ok=True)
        ext, fmt = ('.exr', 'OPEN_EXR') if hdr else ('.png', 'PNG')
        path = os.path.join(folder, _safe_filename(texName) + ext)
        image.filepath_raw = path
        image.file_format = fmt
        log.info("Saving image to %s", path)
        image.save()
        if bpy.data.filepath:
            try:
                image.filepath = bpy.path.relpath(path)
            except ValueError:  # other drive (Windows)
                pass
