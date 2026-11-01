A fork of an blender addon to import BFRES models on newer versions of blender and with different versions of BFRES

The original version only worked with BFRES version 3.5, or BOTW

This addon is currently in heavy development and large portions of it do not work yet.

# Progress
- Model Versions:
    - [x] BFRES 5.3 (BOTW, MK8DX, ARMS; Breaks often, give me a heads up)
    - [ ] BFRES 8/9 (Nearly every other game; Planned)
    - [X] BFRES 10 (Splatoon 3, Mario Wonder, TOTK)
- Importing into Blender
    - [x] Models
        - Custom split normals from the game (`_n0`); left out with "Smooth Faces" off
        - Vertex colors `_c0`, `_c1`... as color attributes (float, per vertex, values as stored)
        - Tangents / bitangents as `_t0` / `_b0` vector attributes, their sign in `_t0_w` / `_b0_w`
    - [x] Textures
        - BC1-BC7 (BC6H as a float/HDR image), all ASTC block sizes, and the uncompressed formats BotW uses
        - Decoded with the bundled native decoder [texture2ddecoder](https://github.com/K0lb3/texture2ddecoder) (MIT license, wheels in `wheels/`); without it, built-in Python decoders are used (everything but ASTC)
    - [ ] Materials
        - Basic PBR samplers (albedo, normal, roughness, etc) apply correctly
        - Does not yet read material attributes or shader settings
        - Textures use default mapping to the first UV map only
    - [x] Bones 
        - Bones import with proper parents, not connected tip to tail by default
        - "Auto Connect Bones" points each bone at its chain child (heads, skinning and animations stay exact)
    - [x] LODs
        - Only the most detailed LOD by default; with "First LOD" off the others go into hidden `<model>.LOD1`, `<model>.LOD2`... collections
        - "Import Far Model" also imports `<model>_Far` into a hidden `<model>.Far` collection
    - [x] Skeletal animations (FSKA, BFRES 5.x)
        - Each animation becomes an action on the armature (sampled per frame, linear keys)
        - A model's `<model>_Animation` file next to it is imported automatically
    - [x] Material animations (FMAA, BFRES 5.x)
        - Texture SRT (scrolling water...) keyframed on Mapping nodes
        - Texture patterns (BotW color variants, dyes, wing blur) as a texture switch
        - Other shader parameters as animated `param_<name>` material properties
        - Material visibility (`*_fvm`) as `hide_viewport`/`hide_render` keys on the objects using the material
    - [x] Visibility animations (FVIS, BFRES 5.x)
        - Bone visibility hides the shapes bound to the bone (constant keys on the objects)
        - Shares the action of the skeletal animation with the same name, if any

# Resources
- https://wiki.wexosmk.xyz/index.php/BFRES_(File_Format)/Switch
- https://github.com/KillzXGaming/Switch-Toolbox
- https://github.com/KillzXGaming/BfresLibrary
- https://github.com/aboood40091/BNTX-Editor
- https://gitlab.com/Syroot/NintenTools/Bfres
- https://github.com/shibbo/Fushigi/tree/main

# Original readme below this header
Nintendo BFRES importer (and eventually exporter) for Blender.

Imports models from games such as Breath of the Wild.

Currently only supports Switch formats (and probably not very well).

Based on https://github.com/Syroot/io_scene_bfres

Very much in development and may be full of bugs and rough corners.

# What works:
- Importing models, with their skeletons and textures, from `.bfres` and `.sbfres` files (at least `Animal_Fox` works)
    - Includes cases where textures are in a separate `.Tex.sbfres` file in the same directory
    - Textures are embedded in the `.blend` file
    - Each LOD (level of detail) model is imported as a separate object; LODs other than the first are kept in hidden collections.
    - Materials' render/shader/material parameters are stored as Blender custom properties on the material objects.
    - Text files in the FRES are embedded into the blend file.

# What's broken:
- Specular intensity is way too high (models are shinier than they should be)
- `npc_zelda_miko` is weird, needs investigation
- Decompressing is slow
    - Progress indicator sucks, but I don't think Blender provides any way to do a better one
- `Animal_Bee`'s UV map is all wrong
- Some texture formats might not decode correctly
- Extra files in the FRES, which are neither text nor a supported format, are discarded
    - The game probably never does this, but it's technically possible

# What's not even started yet:
- Animations
- Exporting
    - Exports as a `.dat` file containing Python code, which might be useful for some scripts
    - Eventually will export a `bfres` file
- WiiU files

# Why another importer?
I wanted to convert to a common format such as Collada, which any 3D editor could then use, but Blender seems to have issues with every suitable format. ¯\\\_(ツ)\_/¯

Syroot's importer didn't work for me, and as far as I know, didn't support skeletons. Since I'd already made a convertor, it was easier to build an importer from that.
