import bpy
from bpy.props import BoolProperty

# Add-on preferences are keyed by the add-on's module name, which for an
# extension is its package, e.g. bl_ext.user_default.bfres_importer.
ADDON_ID = __package__.rpartition('.bfres.Importer')[0] or __package__


def get_preferences(context):
    """The add-on preferences, or None when the add-on isn't enabled
    through Blender (e.g. loaded by a script)."""
    addon = context.preferences.addons.get(ADDON_ID)
    return addon.preferences if addon is not None else None


class BfresPreferences(bpy.types.AddonPreferences):
    bl_idname = ADDON_ID

    debugDumpFiles: BoolProperty(
        name="Dump debug info to files",
        description="Default for the importer's Dump Debug Info option: "
                    "create `fres-SomeFile-dump.txt` files for debugging",
        default=False,
    )  # type: ignore

    def draw(self, context):
        self.layout.prop(self, "debugDumpFiles")
