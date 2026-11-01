# Development

Notes for working on the add-on with Blender 5.2 (tested on macOS, Apple silicon).

## Load the add-on from the working copy

Link the repository into Blender's user extension repository, so edits are
picked up without reinstalling:

```bash
EXT="$HOME/Library/Application Support/Blender/5.2/extensions/user_default"   # macOS
# Linux:   ~/.config/blender/5.2/extensions/user_default
# Windows: %APPDATA%\Blender Foundation\Blender\5.2\extensions\user_default
mkdir -p "$EXT"
ln -s "$PWD" "$EXT/bfres_importer"
```

Then in Blender: *Edit → Preferences → Get Extensions → ⌄ → Refresh Local*,
and enable **Nintendo BFRES format** (module `bl_ext.user_default.bfres_importer`).
The importer appears under *File → Import → Nintendo Switch BFRES*.

After changing code, reload with *Edit → Preferences → Add-ons*, disable and
re-enable the add-on, or restart Blender. The headless harness below always
loads the code fresh.

## Test corpus

Game files are not part of this repository. `tools/extract_corpus.py` copies a
small set of test models out of a RomFS you dumped from your own console:

```bash
python3 tools/extract_corpus.py <romfs> ~/Documents/bfres_corpus/botw
```

`<romfs>` is an extracted RomFS folder, or a single-file `DMROMFS1` image such
as the `Contents/Resources/romfs.img` of an nxnative / NX2Apple Mac app.
The list of models is `tools/corpus/botw.txt`; `--list` picks another one and
`--grep REGEX` just searches the RomFS:

```bash
python3 tools/extract_corpus.py <romfs> /tmp/x --grep '^Model/Enemy_Lynel'
```

`tools/corpus/botw_vis.txt` lists a few extra models with bone visibility
animations (FVIS), which none of the main corpus has; extract them to their
own folder (`~/Documents/bfres_corpus/botw_vis`) with `--list`.

Keep the corpus outside the repository.

## Headless import harness

`tools/import_test.py` imports every model in a folder with a background
Blender and reports what came out. It loads the add-on from the repository, so
it works with `--factory-startup`:

```bash
BLENDER=/Applications/Blender.app/Contents/MacOS/Blender
$BLENDER -b --factory-startup --python tools/import_test.py -- \
    ~/Documents/bfres_corpus/botw \
    --out report.json --render renders/
```

Each line of output is one model: status, time, meshes, vertices, decoded
images / total images, UVs that are NaN/infinite or beyond ±4096 (tiled
surfaces can legitimately go into the hundreds), and the number of
log warnings and errors. `report.json` has the details per mesh, image and
material; `renders/` has one Workbench render per model.

Before and after a change, compare against a saved report:

```bash
$BLENDER -b --factory-startup --python tools/import_test.py -- \
    ~/Documents/bfres_corpus/botw --out after.json --baseline report.json
```

Other options:

- `--opt first_lod=False` forwards an option to the import operator
  (any property of `ImportBFRES`; values are Python literals).
- `--all` also imports `*.Tex.*` and `*_Animation.*` files on their own.
- Single files can be given instead of a folder.

The exit code is non-zero when any model fails, so the harness can gate a
commit.

## Texture decoders

Block-compressed textures are decoded by the native module
[texture2ddecoder](https://github.com/K0lb3/texture2ddecoder) (MIT) when
it can be loaded, and by the numpy decoders in `bfres/BNTX/pixelfmt/BCn/`
otherwise (BC1-BC7; ASTC needs the native module). Both give the same
pixels; BC6H always uses numpy, because the native decoder clamps it to
8 bits. `bfres/BNTX/pixelfmt/native.py` looks for the module:

1. `import texture2ddecoder`: when the add-on is installed as an extension,
   Blender installs the matching wheel listed in `blender_manifest.toml`;
2. otherwise (add-on loaded from a checkout, as the harness does) the
   matching wheel in `wheels/` is unzipped once into a cache folder
   (`~/Library/Caches/bfres_importer`, `%LOCALAPPDATA%\bfres_importer`,
   `~/.cache/bfres_importer`; `BFRES_CACHE_DIR` overrides) and imported
   from there.

The wheels are `abi3` builds for Python 3.11+ (macOS arm64 and x86_64,
Windows x64, Linux x64). To update them:

```bash
for p in macosx_11_0_arm64 macosx_10_9_x86_64 win_amd64 manylinux2014_x86_64; do
  pip download texture2ddecoder --no-deps --only-binary=:all: \
      --platform $p --python-version 3.13 -d wheels
done
```

and update the `wheels` list in `blender_manifest.toml`. Set
`BFRES_NATIVE_DECODER=0` to force the Python decoders (testing, timing).
Images the importer creates carry `bfres_format`, `bfres_component`
(UNorm/SRGB/SNorm...), `bfres_decoder` (native/python) and, if decoding
failed, `bfres_decode_error`; the harness counts the latter.

BotW has no BC6H, BC7 or ASTC textures. `tools/test_textures.py` makes a
synthetic BNTX with every format from generated images and checks the
import against reference decoders (bcdec, astcenc). Its `make` step needs
`pip install numpy imagecodecs etcpak astc-encoder-py` in some Python
(not Blender's own):

```bash
python3 tools/test_textures.py make /tmp/synth/synth
$BLENDER -b --factory-startup --python tools/test_textures.py -- check /tmp/synth/synth.bntx
```

### Texture layout: block heights, mips, layers, channel selectors

`bfres/BNTX/pixelfmt/swizzle.py` follows the console driver's rules, as
ported by Ryujinx / [tegra_swizzle](https://github.com/ScanMountGoat/tegra_swizzle):
GOBs of 64 bytes x 8 rows, each level deswizzled with the BRTI block height
(`textureLayout & 7`) halved while the level fits in half a block
(`mip_block_height`), levels back to back, array layers / cube faces
aligned to a block of GOBs (`surface_layout`). `BRTI` checks the computed
offsets against the file's own (`mipOffsets`, `data_len`) and exposes every
level and layer: `levelData(level, layer)`, `decodeLevel()`, `rgba()`.
The importer only makes images of level 0, layer 0.

BRTI channel selectors (`channel_types`, e.g. `RRRR` for most BC4) are
applied by `rgba(selectors=True)`, or by default with
`BFRES_CHANNEL_SELECTORS=on`; they are off by default because the
material code expects the decoders' own output (`pixelfmt/channels.py`
lists what BotW uses and what would change).

```bash
# every texture header of a RomFS: formats, sizes, block heights,
# selectors, and a check of the computed layout against the file
$BLENDER_PY tools/scan_textures.py romfs.img --out /tmp/tex_headers.json
# decode every level of every texture, compare with the reduced level above
$BLENDER_PY tools/test_textures.py levels ~/Documents/bfres_corpus/botw
```

(`$BLENDER_PY`: any Python with numpy, e.g. Blender's own
`Contents/Resources/5.2/python/bin/python3.13`.)

## Proving a refactor changes nothing

`tools/snapshot.py` fingerprints every mesh (positions, faces, smooth flags,
UVs, corner normals, color / tangent attributes, vertex group weights) and every image (pixels) the
importer produces. Take one snapshot before the change and compare after:

```bash
git stash   # or check out the base commit
$BLENDER -b --factory-startup --python tools/snapshot.py -- \
    ~/Documents/bfres_corpus/botw --out before.json
git stash pop
$BLENDER -b --factory-startup --python tools/snapshot.py -- \
    ~/Documents/bfres_corpus/botw --out after.json --compare before.json
```

It prints `identical to before.json`, or each mesh/image that changed and
which fields. Use it for performance work and clean-ups; for fixes, the
listed differences should be exactly the ones you meant.


## Reading BotW's material shader (uking_mat)

`bfres/Importer/uking_combiner.py` decodes uking_mat's color combiner
options. The formulas were read from the game's compiled shaders, and
`tools/uking_shader.py` repeats that for any material:

```bash
# once: Shader/uking_mat.product.sbfsha out of Pack/Bootup_Graphics.pack
python3 tools/uking_shader.py extract <romfs.img> /tmp/uking_mat.bfsha
# uniform blocks (gsys_material offsets: const_color0 @ 368, const_value0 @ 496...)
python3 tools/uking_shader.py info /tmp/uking_mat.bfsha
# programs for a material's shader options (a JSON dict of fmat.shaderOptions)
python3 tools/uking_shader.py find /tmp/uking_mat.bfsha opts.json --app <Game>.app
xcrun metal-objdump --disassemble <printed .metallib>   # LLVM IR (macOS)
```

`find` lists the programs whose static options match (render-state options
like `gsys_alpha_test_func` are set by the engine and ignored) with their
BNSH variation and, with `--app`, the Metal library NX2Apple/nxnative made
for the pixel shader (`gsys_assign_gbuffer` is the deferred pass that
writes albedo/normal/emission). In the IR, the material block is the
`fp_c7`-like buffer bound to the program's `gsys_material` location (index
= byte offset / 16), textures are `fp_t_tcb_<hex>` with register
`(hex - 8) / 2` = the program's sampler location. Material parameters
that have the same value in every material using a program are folded into
the code (Chuchu's `const_value1 = 0`, the 1.0 defaults of the moss
materials), so an argument without a load is a constant, not an unused one.
