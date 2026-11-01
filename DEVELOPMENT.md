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
images / total images, UVs that are NaN or far outside 0–1, and the number of
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
