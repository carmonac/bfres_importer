import logging; log = logging.getLogger(__name__)
import bmesh
import bpy
import bpy_extras
import os
import os.path
import tempfile
import shutil
import struct
import math
import zstandard
from ..Exceptions import UnsupportedFileTypeError
from ..BinaryFile import BinaryFile
from .. import YAZ0, FRES, BNTX
from .ModelImporter import ModelImporter
from .TextureImporter import TextureImporter


class Importer(ModelImporter):
    def __init__(self, operator, context):
        self.operator = operator
        self.context  = context


    @staticmethod
    def _add_object_to_group(ob, group_name):
        # Get or create the required group.
        group = bpy.data.objects[group_name]

        # Link the provided object to it.
        ob.parent = group
        return group


    def run(self, path, **keywords):
        """Perform the import."""
        self.wm   = bpy.context.window_manager
        self.path = path
        return self.unpackFile(path)


    def unpackFile(self, file):
        """Try to unpack the given file.

        file: A file object, or a path to a file.

        If the file format is recognized, will try to unpack it.
        If the file is compressed, will first decompress it and
        then try to unpack the result.
        Raises UnsupportedFileTypeError if the file format isn't
        recognized.
        """
        if type(file) is str: # a path
            file = BinaryFile(file)
        self.file = file

        # read magic from header
        file.seek(0) # rewind
        magic = file.read(4)
        file.seek(0) # rewind
        match magic:
            case b'Yaz0' | b'Yaz1': # Compressed YAZ file
                r = self.decompressYaz(file)
                return self.unpackFile(r)
            
            case b'(\xb5/\xfd': # Compressed ZSTD file
                r = self.decompressZS(file)
                return self.unpackFile(r)
            
            case b'FRES':
                return self._importFres(file)

            case b'BNTX':
                return self._importBntx(file)
            case _:
                raise UnsupportedFileTypeError(magic)

    def decompressZS(self, file):
        filecontents = file.read(file.size, 0)
        decompressed = zstandard.decompress(filecontents)
        result = tempfile.TemporaryFile()
        result.write(decompressed)
        result.seek(0)
        return BinaryFile(result)

    def decompressYaz(self, file):
        """Decompress given file.

        Returns a temporary file.
        """
        log.debug("Decompressing input file...")
        result = tempfile.TemporaryFile()

        # make progress callback to update UI
        progress = 0
        def progressCallback(cur, total):
            nonlocal progress
            pct = math.floor((cur / total) * 100)
            if pct - progress >= 1:
                self.wm.progress_update(pct)
                progress = pct
            print("\rDecompressing... %3d%%" % pct, end='')
        self.wm.progress_begin(0, 100)

        # decompress the file
        decoder = YAZ0.Decoder(file, progressCallback)
        for data in decoder.bytes():
            result.write(data)
        self.wm.progress_end()
        print("") # end status line
        result.seek(0)

        if self.operator.save_decompressed: # write back to file
            path, ext = os.path.splitext(file.name)
            # 's' prefix indicates compressed;
            # eg '.sbfres' is compressed '.bfres'
            if ext.startswith('.s'): ext = '.'+ext[2:]
            else: ext = '.out'
            log.info("Saving decompressed file to: %s", path+ext)
            with open(path+ext, 'wb') as save:
                shutil.copyfileobj(result, save)
            result.seek(0)

        return BinaryFile(result)


    def _importFres(self, file):
        """Import FRES file."""
        self.fres = FRES.FRES(file)
        self.fres.decode()

        if self.operator.dump_debug:
            with open('./fres-%s-dump.txt' % self.fres.name, 'w') as f:
                f.write(self.fres.dump())
            #print("FRES contents:\n" + self.fres.dump())

        # decode embedded files
        for file in self.fres.embeds:
            self._importFresEmbed(file)

        # import the models.
        existing = set(bpy.data.objects)
        if not self.animationsOnly:
            for i, model in enumerate(self.fres.models):
                log.info("Importing model    %3d / %3d...",
                    i+1, len(self.fres.models))
                self._importModel(model)

        if getattr(self.operator, 'import_animations', True):
            self._importSkeletalAnims()
            self._importMaterialAnims()
            self._importVisibilityAnims(
                [ob for ob in bpy.data.objects if ob not in existing and ob.type == 'ARMATURE'])

        return {'FINISHED'}


    # Set by the caller to import only the animations of a file onto an
    # existing armature (e.g. a model's *_Animation.sbfres).
    animationsOnly = False
    targetArmature = None
    targetMaterials = None

    def _importMaterialAnims(self):
        from .MaterialAnimationImporter import MaterialAnimationImporter
        try:
            sets = self.fres.readMaterialAnims()
        except Exception:
            log.exception("Could not read material animations")
            return
        if not sets:
            return
        # materials of the model imported with this file (or its parent
        # import); otherwise every material imported by this add-on
        materials = self.targetMaterials or self.__dict__.get('createdMaterials') \
            or [m for m in bpy.data.materials if m.get('bfres_name')]
        self.materialActions = MaterialAnimationImporter(self.context).import_anims(sets, materials)

    def _importVisibilityAnims(self, armatures):
        from .VisibilityAnimationImporter import VisibilityAnimationImporter
        try:
            anims = self.fres.readVisibilityAnims()
        except Exception:
            log.exception("Could not read visibility animations")
            return
        if not anims:
            return
        materials = self.targetMaterials or self.__dict__.get('createdMaterials') \
            or [m for m in bpy.data.materials if m.get('bfres_name')]
        # the meshes of the model each animation is for: the armature (of
        # this import, or the one its animations go on) with the most bones
        # named in it, as bone names like "Root" repeat across models
        if self.targetArmature is not None:
            armatures = [self.targetArmature]
        def meshes_of(arm):
            return [ob for ob in bpy.data.objects if ob.type == 'MESH'
                    and any(m.type == 'ARMATURE' and m.object == arm for m in ob.modifiers)]
        groups = {}
        for anim in anims:
            if anim.is_material or not armatures:
                groups.setdefault(None, []).append(anim)
                continue
            names = set(anim.names)
            arm = max(armatures, key=lambda a: len(names & set(a.data.bones.keys())))
            groups.setdefault(arm, []).append(anim)
        # the game plays a model's skeletal and visibility animations of the
        # same name together (e.g. FldObj_Mound_A_01_Break): share the action
        shared = {a['bfres_name']: a for a in self.__dict__.get('actions') or []
                  if a.get('bfres_name')}
        imp = VisibilityAnimationImporter(self.context)
        self.visibilityActions = []
        for arm, group in groups.items():
            objects = meshes_of(arm) if arm is not None else \
                [ob for a in armatures for ob in meshes_of(a)] if armatures else \
                [ob for ob in bpy.data.objects if ob.type == 'MESH']
            self.visibilityActions += imp.import_anims(group, objects, materials, shared)

    def _importSkeletalAnims(self):
        from .AnimationImporter import AnimationImporter, find_target_armature
        try:
            anims = self.fres.readSkeletalAnims()
        except Exception:
            log.exception("Could not read skeletal animations")
            return
        if not anims:
            return
        target = self.targetArmature or getattr(self, 'armature', None)
        if target is None:
            names = [b.name for a in anims for b in a.bone_anims]
            target = find_target_armature(self.context, names)
        self.actions = AnimationImporter(self.context).import_anims(anims, target)


    def _importFresEmbed(self, file):
        """Import embedded file from FRES."""
        if file.name.endswith('.txt'): # embed into blend file
            obj = bpy.data.texts.new(name=file.name)
            obj.write(file.data.decode('utf-8'))
        else: # try to decode, may be BNTX
            try:
                self.unpackFile(file.toTempFile())
            except UnsupportedFileTypeError as ex:
                log.debug("Embedded file '%s' is of unsupported type '%s'",
                    file.name, ex.magic)


    def _importBntx(self, file):
        """Import BNTX file."""
        self.bntx = BNTX.BNTX(file)
        self.bntx.decode()
        if self.operator.dump_debug:
            with open('./fres-%s-bntx-dump.txt' % self.bntx.name, 'w') as f:
                f.write(self.bntx.dump())

        imp = TextureImporter(self)
        imp.importTextures(self.bntx)

        return {'FINISHED'}
