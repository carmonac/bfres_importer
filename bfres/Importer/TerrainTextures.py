"""Breath of the Wild's terrain material texture arrays.

Field objects that blend into the terrain (grass fields, cliffs, mounds,
sand...) use the samplers `tma` / `tmc`, bound to the textures
`MaterialAlb` and `MaterialCmb`. These are not in the model's own .Tex:
they are 2D array textures of the terrain renderer, in
`Model/Terrain.Tex.sbfres` inside `Pack/TitleBG.pack` (83 layers, one per
terrain material: MaterialAlb 1024x1024 BC1 sRGB albedo, MaterialCmb
512x512 BC1 with channel selectors RGBB, a normal map in RG whose blue is
also read as the specular mask; plus the grass/flower/tree arrays).

uking_mat reads them through texture slots: shader option
`uking_texture_array_textureN` (0 = array0, 1 = array1, -1 = none) makes
slot N sample that array, at the layer given by the material parameter
`texture_array_indexN` (rounded; read from gsys_material at
texture_array_indexN's offset in the shader's IR). So one material can show
several layers of the same array, one per slot.

This module finds the terrain texture file, decodes the layers materials
ask for into images named `<texture>[<layer>]`, and caches the parsed file
(and, for the Yaz0-compressed original, the decompressed copy on disk in
the add-on's cache folder, since decompressing ~150 MB in Python takes a
while).
"""
import logging; log = logging.getLogger(__name__)
import os

import numpy as np

TERRAIN_FILES = ('Terrain.Tex.sbfres', 'Terrain.Tex.bfres')
LAYER_PROP = 'bfres_layer'

_loaded = {}  # path -> TerrainTextures (or None when it failed)


def image_name(texture, layer):
    return '%s[%d]' % (texture, layer)


def find_terrain_file(model_path, explicit=''):
    """The terrain texture file for a model: `explicit` when given, else a
    Terrain.Tex(.s)bfres next to the model, in its parent folder or in a
    Model/ folder there (a RomFS dump with TitleBG.pack extracted), else
    the BFRES_TERRAIN_TEX environment variable."""
    cands = []
    if explicit:
        cands.append(os.path.expanduser(explicit))
    if model_path:
        here = os.path.dirname(os.path.abspath(model_path))
        for d in (here, os.path.dirname(here), os.path.join(os.path.dirname(here), 'Model')):
            cands += [os.path.join(d, n) for n in TERRAIN_FILES]
    env = os.environ.get('BFRES_TERRAIN_TEX')
    if env:
        cands.append(os.path.expanduser(env))
    for c in cands:
        if os.path.isfile(c):
            return c
    return None


def get(path):
    """Parsed terrain textures of `path` (cached), or None."""
    if path is None:
        return None
    key = (os.path.abspath(path), os.path.getmtime(path))
    if key not in _loaded:
        try:
            _loaded[key] = TerrainTextures(path)
        except Exception:
            log.exception("Could not read terrain textures from %s", path)
            _loaded[key] = None
    return _loaded[key]


def _decompressed(path):
    """Path of the uncompressed file: `path` itself, or a cached copy."""
    with open(path, 'rb') as f:
        magic = f.read(4)
    if magic not in (b'Yaz0', b'Yaz1'):
        return path
    from ..BNTX.pixelfmt.native import cache_dir
    st = os.stat(path)
    cached = os.path.join(cache_dir(), 'terrain', '%s.%d.%d.bfres' % (
        os.path.basename(path).split('.')[0], st.st_size, int(st.st_mtime)))
    if os.path.isfile(cached):
        return cached
    from ..YAZ0 import decompress
    log.info("Decompressing %s (once; cached in %s)...", path, cached)
    with open(path, 'rb') as f:
        data = decompress(f.read())
    os.makedirs(os.path.dirname(cached), exist_ok=True)
    tmp = cached + '.tmp'
    with open(tmp, 'wb') as f:
        f.write(data)
    os.replace(tmp, cached)
    return cached


class TerrainTextures:
    """The textures of a terrain .Tex file, by name (BRTI objects)."""

    def __init__(self, path):
        from ..BinaryFile import BinaryFile
        from ..FRES import FRES
        from ..BNTX import BNTX
        self.path = path
        self.textures = {}
        fres = FRES(BinaryFile(_decompressed(path)))
        fres.decode()
        for emb in fres.embeds:
            f = emb.toTempFile()
            f.seek(0)
            if f.read(4) != b'BNTX':
                continue
            f.seek(0)
            bntx = BNTX(f)
            bntx.decode()
            for tex in bntx.textures:
                self.textures[tex.name] = tex
        log.info("Terrain textures from %s: %s", path, ', '.join(
            '%s x%d' % (n, t.layerCount) for n, t in self.textures.items()))

    def layer_count(self, name):
        tex = self.textures.get(name)
        return tex.layerCount if tex is not None else 0

    def layer_image(self, name, layer):
        """Blender image of one layer of array texture `name` (created once,
        then reused), or None."""
        import bpy
        tex = self.textures.get(name)
        if tex is None:
            return None
        layer = min(max(int(layer), 0), tex.layerCount - 1)
        iname = image_name(name, layer)
        for im in bpy.data.images:
            if im.get('bfres_texture') == name and im.get(LAYER_PROP) == layer:
                return im
        # with the BRTI channel selectors: MaterialCmb (RGBB) has its blue in alpha
        px = np.flipud(tex.rgba(0, layer, selectors=True)).astype(np.float32)
        image = bpy.data.images.new(iname, width=tex.width, height=tex.height)
        image['bfres_texture'] = name
        image[LAYER_PROP] = layer
        image['bfres_format'] = type(tex.format_).__name__
        image['bfres_terrain'] = os.path.basename(self.path)
        dtype = int(tex.header['fmt_dtype'])
        try:
            # BRTI TextureDataType: 6 = SRGB (albedo); the rest is data
            image.colorspace_settings.name = 'sRGB' if dtype == 6 else 'Non-Color'
        except TypeError:
            pass
        image.pixels.foreach_set(np.ascontiguousarray(px).ravel())
        image.update()
        image.pack()
        return image
