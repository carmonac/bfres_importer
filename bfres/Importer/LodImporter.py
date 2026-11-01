import logging; log = logging.getLogger(__name__)
import bmesh
import bpy
import bpy_extras
import struct
import numpy as np
from .MaterialImporter import MaterialImporter
from .SkeletonImporter import SkeletonImporter
from ..Exceptions import UnsupportedFormatError, MalformedFileError
from ..FRES.FMDL.Attribute import decode as attrdecode
import mathutils


class LodImporter:
    """Handles importing LOD models."""
    def __init__(self, parent):
        self.parent = parent


    def _importLod(self, fvtx, fmdl, fshp, lod, idx):
        """Import given LOD model."""
        self.fvtx   = fvtx
        self.fmdl   = fmdl
        self.fshp   = fshp
        self.lod    = lod
        self.lodIdx = idx
        self.attrBuffers = self._getAttrBuffers()

        # Create an object for this LOD
        if self.parent.operator.first_lod == True:
            self.lodName = self.fshp.name
        else:
            self.lodName = "%s.%d" % (self.fshp.name, self.lodIdx)
        self.meshObj = self._createMesh()

        self._addUvMap()
        self._addVertexWeights()
        self._addArmature()
        
        return self.meshObj


    def _getAttrBuffers(self):
        """Get attribute data for this LOD.

        Returns a dict of attribute name => numpy array of shape
        (vertices, components). Normalized formats come out as floats
        (UNorm 0..1, SNorm -1..1) and integer formats as ints.
        """
        # Faces index into one vertex array shared by every submesh, so
        # read each vertex once, up to the highest index any submesh uses.
        nVtxs = 0
        for i, submesh in enumerate(self.lod.submeshes):
            idxs = submesh['idxs']
            if len(idxs) == 0:
                raise MalformedFileError("Submesh %d is empty" % i)
            nVtxs = max(nVtxs, max(idxs) + 1)
        if len(self.lod.idx_buf):
            nVtxs = max(nVtxs, max(self.lod.idx_buf) + 1)

        attrBuffers = {}
        for attr in self.fvtx.attrs:
            buf = self.fvtx.buffers[attr.buf_idx]
            try:
                attrBuffers[attr.name] = attrdecode.decode(attr.formatID,
                    buf.data, attr.buf_offs, buf.stride, nVtxs)
            except attrdecode.AttributeFormatError as ex:
                raise MalformedFileError("Attribute '%s': %s" % (attr.name, ex))
        return attrBuffers


    def _boneRotation(self, bone):
        """Rotation part of a bone's world matrix (includes Y-up -> Z-up)."""
        return bone.matrix.to_3x3().normalized()


    def _vertexNormals(self, n0):
        """Per-vertex normals in Blender space, consistent with how
        `_addVerticesToMesh` places the positions."""
        skin = self.fshp.header['vtx_skin_cnt']
        n = np.asarray(n0, dtype=np.float64)[:, :3]
        if skin == 0:
            # rigid mesh bound to one bone: rotate like the positions
            R = np.array(self._boneRotation(
                self.fmdl.skeleton.bones[self.fshp.header['single_bind']]))
            out = n @ R.T
        elif skin == 1:
            # each vertex follows its own rigid matrix
            i0 = self.attrBuffers['_i0'][:, 0]
            out = np.empty_like(n)
            for midx in np.unique(i0):
                R = np.array(self._boneRotation(
                    self.fmdl.skeleton.boneRigidMtxGroups[int(midx)][0]))
                sel = i0 == midx
                out[sel] = n[sel] @ R.T
        else:
            # skinned mesh: bind pose is in model space, Y-up -> Z-up
            out = np.stack((n[:, 0], -n[:, 2], n[:, 1]), axis=1)
        length = np.linalg.norm(out, axis=1, keepdims=True)
        length[length == 0] = 1
        return out / length


    def _createMesh(self):
        p0   = self.attrBuffers['_p0']
        n0   = self.attrBuffers['_n0']
        idxs = self.lod.idx_buf
        # the game doesn't tell how many vertices each LOD has,
        # but we can usually rely on this.
        nVtxs = int(self.lod.header['idx_cnt'] / 3)
        log.debug("LOD has %d vtxs, %d idxs", nVtxs, len(idxs))

        # create a mesh and add faces to it
        mesh = bmesh.new()
        self._addVerticesToMesh(mesh, p0, n0)
        self._createFaces(idxs, mesh)

        # Write the bmesh data back to a new mesh.
        fshpMesh = bpy.data.meshes.new(name=self.lodName)
        mesh.to_mesh(fshpMesh)
        mesh.free()
        meshObj = bpy.data.objects.new(name=fshpMesh.name, object_data=fshpMesh)
        mdata   = meshObj.data

        # make split normals
        normals = self._vertexNormals(n0)[:len(mdata.vertices)].tolist()

        mdata.normals_split_custom_set_from_vertices(normals)

        bpy.context.scene.collection.objects.link(meshObj)
        # self.parent._add_object_to_group(meshObj, self.fmdl.name)

        # Add material
        mat = self.fmdl.fmats[self.fshp.header['fmat_idx']]
        mdata.materials.append(bpy.data.materials[mat.name])

        return meshObj

    def _createFaces(self, idxs, mesh):
        """Create the faces."""
        fmt = self.lod.prim_fmt
        meth = getattr(self, '_createFaces_'+fmt, None)
        if meth is None:
            log.error("Unsupported prim format: %s", fmt)
            raise UnsupportedFormatError(
                "Unsupported prim format: " + fmt)
        try:
            return meth(idxs, mesh)
        except (struct.error, IndexError):
            raise MalformedFileError("LOD submesh faces are out of bounds")

    def _createFacesBasic(self, idxs, mesh, step, nVtxs):
        smooth = self.parent.operator.smooth_faces
        skipped = 0
        for i in range(0, len(idxs) - nVtxs + 1, step):
            try:
                vs = [mesh.verts[j] for j in idxs[i:i+nVtxs]]
            except IndexError:
                log.error("LOD submesh face %d is out of bounds (max %d)",
                    i, len(idxs))
                raise
            try:
                face = mesh.faces.new(vs)
            except ValueError:
                # The game data can repeat a triangle over the same vertices
                # (e.g. the back side of a double-sided surface) or contain
                # degenerate triangles; Blender rejects both. Blender draws
                # both sides of a face anyway, so these are safe to drop.
                skipped += 1
                continue
            face.smooth = smooth
        if skipped:
            log.warning("%s: skipped %d duplicate or degenerate face(s)",
                self.lodName, skipped)

    def _createFaces_point_list(self, idxs, mesh):
        return self._createFacesBasic(idxs, mesh, 1, 1)

    def _createFaces_line_list(self, idxs, mesh):
        return self._createFacesBasic(idxs, mesh, 2, 2)

    def _createFaces_line_strip(self, idxs, mesh):
        return self._createFacesBasic(idxs, mesh, 1, 2)

    def _createFaces_triangle_list(self, idxs, mesh):
        return self._createFacesBasic(idxs, mesh, 3, 3)


    def _addVerticesToMesh(self, mesh, vtxs, nrms):
        """Add vertices (from `_p0` attribute) to a `bmesh`."""
        vtxs = np.asarray(vtxs, dtype=np.float64)
        if vtxs.shape[1] == 4:
            bad = np.count_nonzero(vtxs[:, 3] != 1)
            if bad:
                # Blender has no W coordinate; it is normally always 1.
                log.warning("%s: %d vertex position(s) have W != 1",
                    self.fshp.name, bad)
        pos = vtxs[:, :3]

        # Rigid meshes (skin count 0 or 1) are stored in bone space and
        # need the bone's world transform; skinned meshes are in model space.
        match self.fshp.header['vtx_skin_cnt']:
            case 0:
                M = self.fmdl.skeleton.bones[self.fshp.header['single_bind']].matrix
                pos = self._transform(pos, M)
            case 1:
                i0 = self.attrBuffers['_i0'][:, 0]
                out = np.empty_like(pos)
                for midx in np.unique(i0):
                    M = self.fmdl.skeleton.boneRigidMtxGroups[int(midx)][0].matrix
                    sel = i0 == midx
                    out[sel] = self._transform(pos[sel], M)
                pos = out
            case _:
                pos = np.stack((pos[:, 0], -pos[:, 2], pos[:, 1]), axis=1)

        for i, co in enumerate(pos.tolist()):
            vert = mesh.verts.new(co)
            vert.index = i
        mesh.verts.ensure_lookup_table()


    @staticmethod
    def _transform(points, M):
        M = np.array(M, dtype=np.float64)
        return points @ M[:3, :3].T + M[:3, 3]


    def _addUvMap(self):
        """Add UV maps from `_u0`, `_u1`... attributes."""
        mdata = self.meshObj.data
        nLoops = len(mdata.loops)
        loopVtx = np.empty(nLoops, dtype=np.int64)
        mdata.loops.foreach_get('vertex_index', loopVtx)
        idx = 0
        while True:
            attr = '_u%d' % idx
            data = self.attrBuffers.get(attr)
            if data is None:
                break
            # values are already normalized; flip V (DirectX -> OpenGL)
            uv = np.asarray(data, dtype=np.float32)[loopVtx, :2].copy()
            uv[:, 1] = 1.0 - uv[:, 1]
            uv_layer = mdata.uv_layers.new(name=attr)
            uv_layer.data.foreach_set('uv', uv.ravel())
            idx += 1


    def _addVertexWeights(self):
        """Add vertex weights (`_w0` attribute) to mesh."""
        try:
            self._makeVertexGroup()
        except KeyError:
            log.info("mesh '%s' has no weights", self.lodName)


    def _addArmature(self):
        """Add armature to mesh."""
        mod = self.meshObj.modifiers.new(name=self.lodName, type='ARMATURE')
        mod.object = self.parent.armature
        mod.use_bone_envelopes = False
        mod.use_vertex_groups  = True
        return mod


    def _makeVertexGroup(self):
        """Make vertex group for mesh object from attributes."""
        # XXX move to SkeletonImporter?
        groups = {}

        i0 = self.attrBuffers.get('_i0')
        w0 = self.attrBuffers.get('_w0')

        # create a vertex group for each bone
        # each bone affects the vertex group with the same
        # name as that bone, and these weights define how much.
        if not self.fshp.header['vtx_skin_cnt'] == 0:
            for bone in self.fmdl.skeleton.bones:
                grp = self.meshObj.vertex_groups.new(name=bone.name)
                groups[bone.smooth_mtx_idx] = grp
                groups[bone.rigid_mtx_idx] = grp
            
            skin = self.fshp.header['vtx_skin_cnt']
            if skin == 1:
                # i0 specifies the bone rigid matrix group.
                for i, idx in enumerate(i0[:, 0].tolist()):
                    groups[idx].add([i], 1.0, 'REPLACE')
            else:
                # i0/i1 give up to 8 smooth matrix indices and w0/w1 how
                # much each one weighs (already normalized to 0..1).
                idxs = np.concatenate([a for a in (i0, self.attrBuffers.get('_i1')) if a is not None], axis=1)[:, :skin]
                wgts = np.concatenate([a for a in (w0, self.attrBuffers.get('_w1')) if a is not None], axis=1)
                if wgts.shape[1] < skin:
                    # the last weight is implied when it isn't stored
                    rest = np.clip(1.0 - wgts.sum(axis=1, keepdims=True), 0, 1)
                    wgts = np.concatenate((wgts, rest), axis=1)
                wgts = wgts[:, :skin]
                missing = set()
                for i, (vidx, vw) in enumerate(zip(idxs.tolist(), wgts.tolist())):
                    for idx, w in zip(vidx, vw):
                        if w <= 0:
                            continue
                        grp = groups.get(idx)
                        if grp is None:
                            missing.add(idx)
                            continue
                        # ADD: the same bone may appear twice for a vertex
                        grp.add([i], w, 'ADD')
                if missing:
                    log.warning("%s: weights reference unknown bone matrices %s",
                        self.lodName, sorted(missing))
        else: 
            # no i0 or w0, mesh is parented to the bone it's on
            idx = self.fshp.header['single_bind']
            grp = self.meshObj.vertex_groups.new(name=self.fmdl.skeleton.bones[idx].name)
            for i in range(0, len(self.attrBuffers['_p0'])):
                grp.add([i], 1, 'REPLACE')
