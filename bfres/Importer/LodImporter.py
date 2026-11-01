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
            vals = attrBuffers[attr.name]
            if vals.dtype.kind == 'f':
                bad = np.count_nonzero(~np.isfinite(vals))
                if bad:
                    log.warning("%s: attribute %s has %d non-finite value(s)",
                        self.fshp.name, attr.name, bad)
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

        if self.lod.prim_fmt == 'triangle_list':
            fshpMesh = self._buildTriangleMesh(p0, idxs)
        else:
            # rare primitive types: build through bmesh
            mesh = bmesh.new()
            self._addVerticesToMesh(mesh, p0, n0)
            self._createFaces(idxs, mesh)
            fshpMesh = bpy.data.meshes.new(name=self.lodName)
            mesh.to_mesh(fshpMesh)
            mesh.free()
        meshObj = bpy.data.objects.new(name=fshpMesh.name, object_data=fshpMesh)
        mdata   = meshObj.data

        # make split normals
        normals = self._vertexNormals(n0)[:len(mdata.vertices)].tolist()

        mdata.normals_split_custom_set_from_vertices(normals)

        self.parent.lodCollection(self.lodIdx).objects.link(meshObj)

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


    def _buildTriangleMesh(self, p0, idxs):
        """Create the mesh for a triangle list directly with foreach_set.

        Produces the same result as adding the faces one by one through
        bmesh: all vertices are kept (in order), and triangles bmesh would
        reject -- degenerate ones, or a repeat of an earlier triangle's
        three vertices in any order -- are dropped, keeping the first.
        """
        pos = self._vertexPositions(p0)
        nV = len(pos)
        idx = np.asarray(idxs, dtype=np.int64)
        tris = idx[:len(idx) // 3 * 3].reshape(-1, 3)
        if tris.size and (tris.min() < 0 or tris.max() >= nV):
            log.error("LOD face index out of bounds (max %d, have %d vertices)",
                tris.max(), nV)
            raise MalformedFileError("LOD submesh faces are out of bounds")

        degenerate = ((tris[:, 0] == tris[:, 1]) | (tris[:, 1] == tris[:, 2])
                      | (tris[:, 0] == tris[:, 2]))
        valid = np.flatnonzero(~degenerate)
        _, first = np.unique(np.sort(tris[valid], axis=1), axis=0, return_index=True)
        keep = valid[np.sort(first)]
        skipped = len(tris) - len(keep)
        if skipped:
            log.warning("%s: skipped %d duplicate or degenerate face(s)",
                self.lodName, skipped)
        tris = tris[keep]

        me = bpy.data.meshes.new(name=self.lodName)
        me.vertices.add(nV)
        me.vertices.foreach_set('co', pos.astype(np.float32).ravel())
        me.loops.add(tris.size)
        me.loops.foreach_set('vertex_index', tris.astype(np.int32).ravel())
        me.polygons.add(len(tris))
        me.polygons.foreach_set('loop_start',
            np.arange(0, tris.size, 3, dtype=np.int32))
        me.polygons.foreach_set('use_smooth',
            np.full(len(tris), bool(self.parent.operator.smooth_faces)))
        me.update(calc_edges=True)
        return me


    def _vertexPositions(self, vtxs):
        """Vertex positions (from `_p0`) in Blender space, (n, 3)."""
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
                return self._transform(pos, M)
            case 1:
                i0 = self.attrBuffers['_i0'][:, 0]
                out = np.empty_like(pos)
                for midx in np.unique(i0):
                    M = self.fmdl.skeleton.boneRigidMtxGroups[int(midx)][0].matrix
                    sel = i0 == midx
                    out[sel] = self._transform(pos[sel], M)
                return out
            case _:
                return np.stack((pos[:, 0], -pos[:, 2], pos[:, 1]), axis=1)


    def _addVerticesToMesh(self, mesh, vtxs, nrms):
        """Add vertices (from `_p0` attribute) to a `bmesh`."""
        for i, co in enumerate(self._vertexPositions(vtxs).tolist()):
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
                bones = i0[:, 0].astype(np.int64)
                vids = np.arange(len(bones))
                wgts = np.ones(len(bones), dtype=np.float32)
            else:
                # i0/i1 give up to 8 smooth matrix indices and w0/w1 how
                # much each one weighs (already normalized to 0..1).
                idxs = np.concatenate([a for a in (i0, self.attrBuffers.get('_i1')) if a is not None], axis=1)[:, :skin]
                ws = np.concatenate([a for a in (w0, self.attrBuffers.get('_w1')) if a is not None], axis=1)
                if ws.shape[1] < skin:
                    # the last weight is implied when it isn't stored
                    rest = np.clip(1.0 - ws.sum(axis=1, keepdims=True), 0, 1)
                    ws = np.concatenate((ws, rest), axis=1)
                ws = ws[:, :skin]
                vids = np.repeat(np.arange(len(idxs)), skin)
                bones = idxs.astype(np.int64).ravel()
                wgts = ws.astype(np.float32).ravel()
                used = wgts > 0
                vids, bones, wgts = vids[used], bones[used], wgts[used]
                # a vertex may list the same bone twice: sum those weights
                if len(bones):
                    nb = int(bones.max()) + 1
                    keys, inv = np.unique(vids * nb + bones, return_inverse=True)
                    wgts = np.bincount(inv, weights=wgts).astype(np.float32)
                    vids, bones = keys // nb, keys % nb
            self._assignWeights(groups, vids, bones, wgts)
        else: 
            # no i0 or w0, mesh is parented to the bone it's on
            idx = self.fshp.header['single_bind']
            grp = self.meshObj.vertex_groups.new(name=self.fmdl.skeleton.bones[idx].name)
            grp.add(list(range(len(self.attrBuffers['_p0']))), 1.0, 'REPLACE')


    def _assignWeights(self, groups, vids, bones, wgts):
        """Add (vertex, bone matrix index, weight) triples to the vertex
        groups, one `add()` call per bone and distinct weight value."""
        missing = []
        for bone in np.unique(bones).tolist():
            grp = groups.get(bone)
            if grp is None:
                missing.append(bone)
                continue
            sel = bones == bone
            bv, bw = vids[sel], wgts[sel]
            for w in np.unique(bw).tolist():
                grp.add(bv[bw == w].tolist(), w, 'REPLACE')
        if missing:
            log.warning("%s: weights reference unknown bone matrices %s",
                self.lodName, missing)
