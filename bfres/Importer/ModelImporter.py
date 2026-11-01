import logging; log = logging.getLogger(__name__)
import bmesh
import bpy
import bpy_extras
import struct
from .MaterialImporter import MaterialImporter
from .SkeletonImporter import SkeletonImporter
from .LodImporter      import LodImporter


def hide_collection(context, coll):
    """Exclude a collection from the view layer and from renders, so the
    scene looks as if it wasn't there until the user enables it."""
    coll.hide_render = True

    def find(layer):
        if layer.collection == coll:
            return layer
        for child in layer.children:
            found = find(child)
            if found is not None:
                return found
        return None

    layer = find(context.view_layer.layer_collection)
    if layer is not None:
        layer.exclude = True


class ModelImporter:
    """Mixin class for importing the geometric models."""

    # Collection new objects are linked to; None is the scene collection.
    collection = None

    def targetCollection(self):
        return self.collection or bpy.context.scene.collection

    def lodCollection(self, idx):
        """Collection for the objects of LOD `idx`: LOD 0 goes with the
        rest of the model, each further LOD into a hidden child collection
        `<model>.LOD<idx>`."""
        base = self.targetCollection()
        if idx == 0:
            return base
        cache = self.__dict__.setdefault('_lodCollections', {})
        key = (self.fmdl.name, idx)
        coll = cache.get(key)
        if coll is None:
            coll = bpy.data.collections.new("%s.LOD%d" % (self.fmdl.name, idx))
            base.children.link(coll)
            hide_collection(bpy.context, coll)
            cache[key] = coll
        return coll

    def _importModel(self, fmdl):
        """Import specified model."""
        # import the skeleton
        self.fmdl = fmdl
        self.skelImp  = SkeletonImporter(self, fmdl)
        self.armature = self.skelImp.importSkeleton(fmdl.skeleton)

        # import the materials.
        self.matImp = MaterialImporter(self, fmdl)
        for i, fmat in enumerate(fmdl.fmats):
            log.info("Importing material %3d / %3d...",
                i+1, len(fmdl.fmats))
            self.matImp.importMaterial(fmat)

        # create the shapes.
        for i, fshp in enumerate(fmdl.fshps):
            log.info("Importing shape %3d / %3d '%s'...",
                i+1, len(fmdl.fshps), fshp.name)
            self._importShape(fmdl, fshp, self.armature)


    def _importShape(self, fmdl, fshp, parent):
        """Import FSHP.

        fmdl: FMDL to import from.
        fshp: FSHP to import.
        parent: Object to parent the LOD models to.
        """
        fvtx = fmdl.fvtxs[fshp.header['fvtx_idx']]
        if self.operator.first_lod == True:
            lod = fshp.lods[0]
            lodImp  = LodImporter(self)
            meshObj = lodImp._importLod(fvtx, fmdl, fshp, lod, 0)
            meshObj.parent = parent
        else:
            for ilod, lod in enumerate(fshp.lods):
                log.info("Importing LOD %3d / %3d...",
                    ilod+1, len(fshp.lods))

                lodImp  = LodImporter(self)
                meshObj = lodImp._importLod(fvtx, fmdl, fshp, lod, ilod)
                meshObj.parent = parent
