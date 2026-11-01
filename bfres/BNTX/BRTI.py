import logging; log = logging.getLogger(__name__)
from ..BinaryStruct import BinaryStruct, BinaryObject
from ..BinaryStruct.Padding import Padding
from ..BinaryStruct.StringOffset import StringOffset
from ..BinaryStruct.Switch import Offset32, Offset64, String
from ..BinaryFile import BinaryFile
from ..Common import StringTable
from enum import IntEnum
from .pixelfmt import TextureFormat
from .pixelfmt.swizzle import deswizzle, surface_layout, DIV_ROUND_UP
from .pixelfmt import channels as _channels
import numpy as np
from .pixelfmt.formatinfo import formats, bpps, blk_dims

class Header(BinaryStruct):
    """BRTI object header."""
    magic  = b'BRTI'
    fields = (
        ('4s',   'magic'), # Header
        ('I',    'length'),
        ('Q',    'length2'),
        ('B',    'flags'),  # Texture Info
        ('B',    'dimensions'),
        ('H',    'tile_mode'),
        ('H',    'swizzle_size'),
        ('H',    'mipmap_cnt'),
        ('H',    'multisample_cnt'),
        ('H',    'reserved1A'),
        ('B',    'fmt_dtype', lambda v: BRTI.TextureDataType(v)),
        ('B',    'fmt_type',  lambda v: TextureFormat.get(v)()),
        Padding(2),   #end of format
        ('I',    'access_flags'),
        ('i',    'width'),
        ('i',    'height'),
        ('i',    'depth'),
        ('I',    'array_cnt'),
        ('I',    'texture_layout'),
        ('I',    'texture_layout2'),
        Padding(20),
        ('I',    'data_len'),
        ('I',    'alignment'),
        ('4B',   'channel_types',lambda v:tuple(v)),
        ('i',    'tex_type'),
        String(  'name'),
        Padding(4),
        Offset64('parent_offset'),
        Offset64('ptrs_offset'),
    )


class BRTI:
    """A BRTI in a BNTX."""
    Header = Header

    class ChannelType(IntEnum):
        Zero  = 0
        One   = 1
        Red   = 2
        Green = 3
        Blue  = 4
        Alpha = 5

    class TextureType(IntEnum):  # nn::gfx ImageDimension
        Image1D        = 0
        Image2D        = 1
        Image3D        = 2
        Cube           = 3
        Image1DArray   = 4
        Image2DArray   = 5
        Image2DMS      = 6
        Image2DMSArray = 7
        CubeArray      = 8

    class TextureDataType(IntEnum):
        UNorm  = 1
        SNorm  = 2
        UInt   = 3
        SInt   = 4
        Single = 5
        SRGB   = 6
        UHalf  = 10

    def __init__(self):
        self.file       = None
        self.mipOffsets = []   # from the first level, as stored in the file


    def dump(self):
        """Dump to string for debug."""
        res = []
        res.append("BRTI Name:     '%s'" % self.name)
        res.append("Length:          0x%06X / 0x%06X" % (
            self.header['length'], self.header['length2']))
        res.append("Flags:           0x%02X" % self.header['flags'])
        res.append("Dimensions:      0x%02X" % self.header['dimensions'])
        res.append("Tile Mode:       0x%04X" % self.header['tile_mode'])
        res.append("Swiz Size:       0x%04X" % self.header['swizzle_size'])
        res.append("Mipmap Cnt:      0x%04X" % self.header['mipmap_cnt'])
        res.append("Multisample Cnt: 0x%04X" % self.header['multisample_cnt'])
        res.append("Reserved 1A:     0x%04X" % self.header['reserved1A'])
        res.append("Fmt Data Type:   %2d %s" % (
            int(self.header['fmt_dtype']),
            self.header['fmt_dtype'].name))
        res.append("Fmt Type:        %2d %s" % (
            self.header['fmt_type'].fmt_id,
            type(self.header['fmt_type']).__name__))
        res.append("Access Flags:    0x%08X" % self.header['access_flags'])
        res.append("Width x Height:  %5d/%5d" % (self.width, self.height))
        res.append("Depth:           %3d" % self.header['depth'])
        res.append("Array Cnt:       %3d" % self.header['array_cnt'])
        res.append("Block Height:    %8d" % self.header['texture_layout'])
        res.append("Unk38:           %04X %04X" % (
            self.header['unk38'], self.header['unk3A']))
        res.append("Unk3C:           %d, %d, %d, %d, %d" % (
            self.header['image_size'], self.header['unk40'],
            self.header['unk44'], self.header['unk48'],
            self.header['unk4C']))
        res.append("Data Len:        0x%08X" % self.header['data_len'])
        res.append("Alignment:       0x%08X" % self.header['alignment'])
        res.append("Channel Types:   %s, %s, %s, %s" % (
            self.header['channel_types'][0].name,
            self.header['channel_types'][1].name,
            self.header['channel_types'][2].name,
            self.header['channel_types'][3].name))
        res.append("Texture Type:    0x%08X" % self.header['tex_type'])
        res.append("Parent Offs:     0x%08X" % self.header['parent_offset'])
        res.append("Ptrs Offs:       0x%08X" % self.header['ptrs_offset'])
        return '\n'.join(res).replace('\n', '\n  ')


    def readFromFile(self, file:BinaryFile, offset=0):
        """Decode objects from the file."""
        self.file            = file
        self.header          = self.Header().readFromFile(file, offset)
        self.name            = self.header['name']
        self.format_         = self.header['fmt_type']
        self.width           = self.header['width']
        self.height          = self.header['height']
        self.tile_mode       = self.header['tile_mode']
        self.channel_types   = self.header['channel_types']
        # e.g. 'RRRR' for a BC4 texture the shader reads as R in all four
        self.channelSelectors = _channels.to_string(self.channel_types)

        self.fmt_id          = self.format_.id
        self.bpp             = bpps.get(self.fmt_id, self.format_.bytesPerPixel)

        if (self.fmt_id) in blk_dims:
            self.blkWidth, self.blkHeight = blk_dims[self.fmt_id]
        else:
            self.blkWidth, self.blkHeight = 1, 1
        self.blockHeightLog2 = self.header['texture_layout'] & 7
        self.mipCount   = max(1, self.header['mipmap_cnt'])
        self.layerCount = max(1, self.header['array_cnt'])  # cube: 6 faces
        self.depth      = max(1, self.header['depth'])
        self.textureType = self.header['tex_type']

        log.debug("Reading texture %s (%s)", self.name, type(self.format_).__name__)

        self._readMipmaps()
        # a texture that fails to decode must not take the whole BNTX
        # (and the model's other textures) down with it
        self.decodeError = None
        try:
            self._readData()
            self.pixels = self.format_.decompress(self)
        except Exception as ex:
            log.error("Could not decode texture '%s' (%s %dx%d): %s",
                self.name, type(self.format_).__name__, self.width,
                self.height, ex)
            self.pixels = None
            self.decodeError = str(ex)
        if self.layerCount > 1 or self.depth > 1:
            log.info("Texture '%s' has %d layers, depth %d (type %d): "
                "the images are made from layer 0", self.name,
                self.layerCount, self.depth, self.textureType)
        return self


    def _readMipmaps(self):
        """Read the mipmap offsets (relative to the first level)."""
        ptrs = [self.file.read('Q', self.header['ptrs_offset'] + i * 8)
                for i in range(self.mipCount)]
        self.mipOffsets = [p - ptrs[0] for p in ptrs]


    def _readData(self):
        """Read the raw image data and deswizzle level 0 of layer 0."""
        base = self.file.read('Q', self.header['ptrs_offset'])
        self.data = self.file.read(self.header['data_len'], base)
        if self.depth > 1:
            raise NotImplementedError("3D textures (depth %d) are not supported" % self.depth)

        self.levels, self.layerSize = surface_layout(self.width, self.height,
            self.blkWidth, self.blkHeight, self.bpp, self.tile_mode,
            self.blockHeightLog2, self.mipCount, self.layerCount)
        computed = [lv['offset'] for lv in self.levels]
        if computed != self.mipOffsets:
            # never seen in Breath of the Wild: trust the file's offsets
            log.warning("Texture '%s': mip offsets %s, expected %s",
                self.name, self.mipOffsets, computed)
            for lv, offs in zip(self.levels, self.mipOffsets):
                lv['offset'] = offs
        if self.layerCount > 1:
            stride = self.header['data_len'] // self.layerCount
            if stride != self.layerSize:
                log.warning("Texture '%s': layer size %d, expected %d",
                    self.name, stride, self.layerSize)
                self.layerSize = stride

        self.mipData = self.levelData(0, 0)


    def levelData(self, level=0, layer=0):
        """Deswizzled (linear) data of one mip level of one array layer."""
        lv = self.levels[level]
        start = layer * self.layerSize + lv['offset']
        size = (DIV_ROUND_UP(lv['width'], self.blkWidth) *
                DIV_ROUND_UP(lv['height'], self.blkHeight) * self.bpp)
        result = deswizzle(lv['width'], lv['height'], self.blkWidth,
            self.blkHeight, self.bpp, self.tile_mode, lv['blockHeightLog2'],
            self.data[start:start + lv['size']])
        return result[:size]


    def decodeLevel(self, level=0, layer=0):
        """Decoded data of one level/layer, in the format's own
        decompress() output (what `pixels` holds for level 0, layer 0)."""
        if level == 0 and layer == 0:
            if self.pixels is None:
                raise ValueError(self.decodeError or "texture did not decode")
            return self.pixels
        return self.format_.decompress(_Level(self, level, layer))


    def rgba(self, level=0, layer=0, selectors=None):
        """Float32 RGBA pixels of a level/layer, shape (height, width, 4),
        top row first (as stored; Blender wants them flipped).

        `selectors`: apply the BRTI channel selectors (see
        pixelfmt/channels.py); None follows `channels.mode` ('off' by
        default: the decoders' own output, as the importer always had).
        """
        lv = self.levels[level]
        px = np.asarray(self.format_.decodePixels(self.decodeLevel(level, layer)),
                        np.float32)
        if selectors is None:
            selectors = _channels.mode == 'on'
        if selectors and not _channels.is_identity(self.channel_types):
            px = _channels.apply(px, self.channel_types,
                                 getattr(self.format_, 'channels', 4))
        return px.reshape(lv['height'], lv['width'], 4)


class _Level:
    """What a format's decompress() reads, for another level or layer."""

    def __init__(self, tex, level, layer):
        lv = tex.levels[level]
        self.name    = '%s[level %d, layer %d]' % (tex.name, level, layer)
        self.header  = tex.header
        self.width   = lv['width']
        self.height  = lv['height']
        self.mipData = tex.levelData(level, layer)
