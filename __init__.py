#!/usr/bin/env python3
"""BFRES importer/decoder for Blender.

This script can also run from the command line without Blender,
in which case it just prints useful information about the BFRES.
"""

bl_info = {
    "name": "Nintendo BFRES format",
    "description": "Import BFRES models",
    "author": "RenaKunisaki, Ranidspace",
    "version": (0, 2, 0),
    "blender": (4, 2, 0),
    "location": "File > Import-Export",
    "warning": "This add-on is under heavy development, please ensure you're using the latest version",
    "doc_url": "https://github.com/RenaKunisaki/bfres_importer/wiki",
    "tracker_url": "https://github.com/carmonac/bfres_importer/issues",
    "support": 'COMMUNITY',
    "category": "Import-Export"
}

# Reload the package modules when reloading add-ons in Blender with F8.
print("BFRES MAIN")
if "bpy" in locals():
    import importlib
    names = ('BinaryStruct', 'FRES', 'FMDL', 'Importer', 'Exporter',
    'YAZ0')
    for name in names:
        ls = locals()
        if name in ls:
            importlib.reload(ls[name])

import bpy
from bpy.props import (
        StringProperty,
        BoolProperty,
        FloatProperty,
        EnumProperty,
        CollectionProperty,
        )
from bpy_extras.io_utils import (
        ImportHelper,
        ExportHelper,
        orientation_helper,
        axis_conversion,
        )


# set up debug log
from .bfres import logger
bfres.logger.setup('bfres')
log = bfres.logger.logging.getLogger()

class ImportBFRES(bpy.types.Operator, ImportHelper):
    """Load a BFRES model file"""
    bl_idname    = "import_scene.nxbfres"
    bl_label     = "Import NX BFRES"
    bl_options   = {'UNDO'}

    directory: StringProperty()  # type: ignore

    filename_ext = ".bfres"

    filter_glob: StringProperty(
        default="*.sbfres;*.bfres;*.fres;*.szs;*.zs",
        options={'HIDDEN'},
    )  # type: ignore

    files: CollectionProperty(name="File Path",
            type=bpy.types.OperatorFileListElement,
            ) # type: ignore
    
    ui_tab: EnumProperty(
            items=(('MAIN', "Main", "Main basic settings"),
                  ),
            name="ui_tab",
            description="Import options categories",
            ) # type: ignore

    import_tex_file: BoolProperty(
        name="Import .Tex File",
        description="Import textures from .Tex file with same name.",
        default=True)
    
    component_selector: BoolProperty(
        name="Use Component Selector",
        description="[Currently Broken] turn on if colours look wrong",
        default=False)

    texture_storage: EnumProperty(name="Textures",
        description="Where the imported textures are kept",
        items=(('PACK', "Pack in .blend", "Embed the images in the .blend file"),
               ('FILES', "Save to folder",
                "Save the images as PNG (EXR for HDR) in a folder and reference "
                "the files instead of packing them")),
        default='PACK')

    terrain_tex_file: StringProperty(name="Terrain Textures",
        description="Breath of the Wild's Terrain.Tex.sbfres (in Pack/TitleBG.pack/Model/), "
                    "for materials that blend with the terrain (MaterialAlb/MaterialCmb). "
                    "Empty: look next to the imported file and in its parent folder",
        subtype='FILE_PATH', default="")

    texture_dir: StringProperty(name="Texture Folder",
        description="Folder for the saved textures. Empty: <model>_textures "
                    "next to the imported file",
        subtype='DIR_PATH', default="")

    dump_debug: BoolProperty(name="Dump Debug Info",
        description="Create `fres-SomeFile-dump.txt` files for debugging.",
        default=False)
    
    enable_sss: BoolProperty(name="Use Subsurface",
        description="Apply Subsurface Scattering textures",
        default=False)

    smooth_faces: BoolProperty(name="Smooth Faces",
        description="Set smooth=True on generated faces.",
        default=True)
    
    import_animations: BoolProperty(name="Import Animations",
        description="Import skeletal animations as actions, including those "
                    "in a <model>_Animation file next to the model",
        default=True)

    first_lod: BoolProperty(name="First LOD",
        description="Only import the first, most detailed LOD. Otherwise "
                    "the other LODs go into hidden collections <model>.LOD1, "
                    "<model>.LOD2...",
        default=True)

    import_far: BoolProperty(name="Import Far Model",
        description="Also import the distant model (<model>_Far file next to "
                    "this one) into a hidden collection <model>.Far",
        default=False)

    connect_bones: BoolProperty(name="Auto Connect Bones",
        description="Attempt to connect bones together",
        default=False)

    save_decompressed: BoolProperty(name="Save Decompressed Files",
        description="Keep decompressed FRES files.",
        default=False)

    parent_ob_name: StringProperty(name="Name of a parent object to which FSHP mesh objects will be added.")

    mat_name_prefix: StringProperty(name="Text prepended to material names to keep them unique.")

    def draw(self, context):
        pass

    def invoke(self, context, event):
        # the add-on preferences give the default of dump_debug
        from .bfres.Importer.Preferences import get_preferences
        prefs = get_preferences(context)
        if prefs is not None and not self.properties.is_property_set("dump_debug"):
            self.dump_debug = prefs.debugDumpFiles
        return ImportHelper.invoke(self, context, event)

    def execute(self, context):
        keywords = self.as_keywords(ignore=("filter_glob", "directory", "ui_tab", "filepath", "files"))

        import os
        from .bfres.Importer import Importer

        log.info("Attempting To Import Linked files")
        if self.import_tex_file:
            texpath, ext = os.path.splitext(self.filepath)
            texpath = texpath + '.Tex' + ext
            if os.path.exists(texpath):
                log.info("Importing linked file: %s", texpath)
                importer = Importer(self, context)
                importer.run(texpath, **keywords)     
    
        log.info("importing: %s", self.properties.filepath)
        importer = Importer(self, context)
        result = importer.run(self.properties.filepath, **keywords)

        # BotW keeps a model's skeletal animations in <model>_Animation.sbfres
        if self.import_animations and getattr(importer, 'armature', None) is not None:
            stem, ext = os.path.splitext(self.filepath)
            if stem.endswith('.Tex'):
                stem = stem[:-4]
            # Enemy_Lynel -> Enemy_Lynel_Animation; Enemy_Guardian_A ->
            # Enemy_Guardian_Animation (variants share their family's file)
            candidates = [stem + '_Animation' + ext]
            head, _, tail = os.path.basename(stem).rpartition('_')
            if head:
                candidates.append(os.path.join(os.path.dirname(stem), head + '_Animation' + ext))
            animpath = next((c for c in candidates if os.path.exists(c)), None)
            if animpath and animpath != self.filepath:
                log.info("Importing animations from: %s", animpath)
                animImporter = Importer(self, context)
                animImporter.animationsOnly = True
                animImporter.targetArmature = importer.armature
                animImporter.targetMaterials = importer.__dict__.get('createdMaterials')
                animImporter.run(animpath, **keywords)

        # the distant (low detail) model is kept in <model>_Far.sbfres
        if self.import_far:
            self._importFar(context, keywords)
        return result

    def _importFar(self, context, keywords):
        import os
        from .bfres.Importer import Importer
        from .bfres.Importer.ModelImporter import hide_collection
        stem, ext = os.path.splitext(self.filepath)
        if stem.endswith('.Tex'):
            stem = stem[:-4]
        if stem.endswith('_Far'):
            return
        farpath = stem + '_Far' + ext
        if not os.path.exists(farpath):
            log.info("No far model: %s", farpath)
            return
        coll = bpy.data.collections.new(os.path.basename(stem) + '.Far')
        context.scene.collection.children.link(coll)
        if self.import_tex_file:
            texpath = stem + '_Far.Tex' + ext
            if os.path.exists(texpath):
                log.info("Importing linked file: %s", texpath)
                Importer(self, context).run(texpath, **keywords)
        log.info("Importing far model: %s", farpath)
        farImporter = Importer(self, context)
        farImporter.collection = coll
        farImporter.run(farpath, **keywords)
        # hide only now: the armature must be in the view layer to edit it
        hide_collection(context, coll)

class BFRES_PT_import_textures(bpy.types.Panel):
    bl_space_type = 'FILE_BROWSER'
    bl_region_type = 'TOOL_PROPS'
    bl_label = "Textures"
    bl_parent_id = "FILE_PT_operator"

    @classmethod
    def poll(cls, context):
        sfile = context.space_data
        operator = sfile.active_operator

        return operator.bl_idname == "IMPORT_SCENE_OT_nxbfres"
    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False  # No animation.

        sfile = context.space_data
        operator = sfile.active_operator

        layout.prop(operator, "import_tex_file")
        layout.prop(operator, "terrain_tex_file")
        layout.prop(operator, "texture_storage")
        if operator.texture_storage == 'FILES':
            layout.prop(operator, "texture_dir")
        layout.prop(operator, "component_selector")

class BFRES_PT_import_mesh(bpy.types.Panel):
    bl_space_type = 'FILE_BROWSER'
    bl_region_type = 'TOOL_PROPS'
    bl_label = "Model"
    bl_parent_id = "FILE_PT_operator"

    @classmethod
    def poll(cls, context):
        sfile = context.space_data
        operator = sfile.active_operator

        return operator.bl_idname == "IMPORT_SCENE_OT_nxbfres"
    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False  # No animation.

        sfile = context.space_data
        operator = sfile.active_operator

        layout.prop(operator, "smooth_faces")
        layout.prop(operator, "first_lod")
        layout.prop(operator, "import_far")
        layout.prop(operator, "connect_bones")

class BFRES_PT_import_mat(bpy.types.Panel):
    bl_space_type = 'FILE_BROWSER'
    bl_region_type = 'TOOL_PROPS'
    bl_label = "Materials"
    bl_parent_id = "FILE_PT_operator"

    @classmethod
    def poll(cls, context):
        sfile = context.space_data
        operator = sfile.active_operator

        return operator.bl_idname == "IMPORT_SCENE_OT_nxbfres"
    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False  # No animation.

        sfile = context.space_data
        operator = sfile.active_operator

        layout.prop(operator, "enable_sss")
        layout.prop(operator, "team colour")

class BFRES_PT_import_misc(bpy.types.Panel):
    bl_space_type = 'FILE_BROWSER'
    bl_region_type = 'TOOL_PROPS'
    bl_label = "Misc"
    bl_parent_id = "FILE_PT_operator"

    @classmethod
    def poll(cls, context):
        sfile = context.space_data
        operator = sfile.active_operator

        return operator.bl_idname == "IMPORT_SCENE_OT_nxbfres"
    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False  # No animation.

        sfile = context.space_data
        operator = sfile.active_operator

        layout.prop(operator, "import_animations")
        layout.prop(operator, "save_decompressed")
        layout.prop(operator, "dump_debug")



def menu_func_import(self, context):
    self.layout.operator(ImportBFRES.bl_idname, text="Nintendo Switch BFRES (.bfres/.szs)")

#def menu_func_export(self, context):
#    self.layout.operator(ExportBFRES.bl_idname, text="Nintendo Switch BFRES (.bfres)")


from .bfres.Importer.Preferences import BfresPreferences

classes = (
    BfresPreferences,
    ImportBFRES,
    BFRES_PT_import_textures,
    BFRES_PT_import_mesh,
    BFRES_PT_import_mat,
    BFRES_PT_import_misc,
    #ExportBFRES,
)

# define Blender functions
def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    bpy.types.TOPBAR_MT_file_import.append(menu_func_import)
    #bpy.types.TOPBAR_MT_file_export.append(menu_func_export)


def unregister():
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)
    #bpy.types.TOPBAR_MT_file_export.remove(menu_func_export)

    for cls in classes:
        bpy.utils.unregister_class(cls)


# define main function, for running script outside of Blender.
# XXX Has not been tested in new versions, needs a rewrite
'''def main():
    if len(sys.argv) < 2:
        print("Usage: %s file" % sys.argv[0])
        return

    InPath = sys.argv[1]
    InFile = None

    # try to decompress the input to a temporary file.
    file  = BinaryFile(InPath, 'rb')
    magic = file.read(4)
    file.seek(0) # rewind
    if magic in (b'Yaz0', b'Yaz1'):
        print("Decompressing YAZ0...")

        # create temp file and write it
        InFile = tempfile.TemporaryFile()
        YAZ0.decompressFile(file, InFile)
        InFile.seek(0)
        InFile = BinaryFile(InFile)
        file.close()
        file = None

    elif magic == b'FRES':
        print("Input already decompressed")
        InFile = BinaryFile(file)

    else:
        file.close()
        file = None
        raise TypeError("Unsupported file type: "+str(magic))

    # decode decompressed file
    print("Decoding FRES...")
    fres = FRES.FRES(InFile)
    fres.decode()
    print("FRES contents:\n" + fres.dump())'''


if __name__ == '__main__':
    # main() # see above function
    register()