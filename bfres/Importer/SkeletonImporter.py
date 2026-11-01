import logging; log = logging.getLogger(__name__)
import bmesh
import bpy
import bpy_extras
import struct
import math
import re
from mathutils import Matrix, Vector

class SkeletonImporter:
    """Imports skeleton from FMDL."""

    def __init__(self, parent, fmdl):
        self.fmdl     = fmdl
        self.operator = parent.operator
        self.context  = parent.context
        self.collection = parent.targetCollection()


    def importSkeleton(self, fskl):
        """Import specified skeleton."""
        name = self.fmdl.name

        # This is completely stupid. We basically automate the UI
        # to create the bones manually.
        # There is a better way (which still involves the UI, but
        # less so), but it of course doesn't work. (The created
        # bones fail to actually exist.)
        bpy.context.scene.cursor.location = (0.0, 0.0, 0.0)

        # Create armature and object
        amt = bpy.data.armatures.new(name=name+'.Armature')
        armObj = bpy.data.objects.new(name=name, object_data=amt)
        
        self.collection.objects.link(armObj)
        bpy.context.view_layer.objects.active = armObj

        bpy.ops.object.mode_set(mode='EDIT', toggle=False)
        boneObjs = {}
        for i, bone in enumerate(fskl.bones):
            boneObj = amt.edit_bones.new(name=bone.name)
            boneObjs[i] = boneObj
            boneObj.use_relative_parent = True
            boneObj.use_local_location = True

            boneObj.length = 0.1
            if bone.parent:
                boneObj.parent = boneObjs[bone.parent_idx]
                boneObj.matrix = bone.parent.matrix @ bone.computeTransform()
            else:
                boneObj.matrix = bone.computeTransform()
            # the game rest pose; meshes and animations are placed with it
            bone.matrix = boneObj.matrix

        if self.operator.connect_bones:
            connect_bones([boneObjs[i] for i in range(len(fskl.bones))])

        bpy.ops.object.mode_set(mode='OBJECT')
        from .AnimationImporter import store_rest
        store_rest(armObj, fskl)
        return armObj


def _is_chain_child(bone, child):
    """Whether `child` continues a numbered chain of `bone`, e.g.
    Tail_1 -> Tail_2 or Spine1 -> Spine2."""
    a = re.findall(r'\d+|\D+', bone.name)
    b = re.findall(r'\d+|\D+', child.name)
    if len(a) != len(b):
        return False
    diff = [(x, y) for x, y in zip(a, b) if x != y]
    if len(diff) != 1:
        return False
    x, y = diff[0]
    return x.isdigit() and y.isdigit() and int(y) == int(x) + 1


def _chain_target(bone, eps):
    """The child whose head this bone's tail should point at, or None."""
    children = [c for c in bone.children if (c.head - bone.head).length > eps]
    if len(children) == 1:
        return children[0]
    chain = [c for c in children if _is_chain_child(bone, c)]
    if len(chain) == 1:
        return chain[0]
    return None


def connect_bones(bones):
    """Point each bone's tail at the head of the child continuing it.

    `bones` are edit bones whose matrices are the game rest pose, parents
    before children. Heads never move; only the direction (the bone's Y
    axis, turned by the smallest rotation so roll changes as little as
    possible) and length change. The bones are not marked use_connect, so
    animated translations still apply. Bones without such a child keep
    their orientation and get a length from their parent or the model.
    """
    heads = [b.head.copy() for b in bones]
    if heads:
        lo = Vector([min(h[i] for h in heads) for i in range(3)])
        hi = Vector([max(h[i] for h in heads) for i in range(3)])
        size = (hi - lo).length
    else:
        size = 0.0
    if size <= 0:
        size = 1.0
    eps = size * 1e-4
    minLength = size * 1e-3

    lengths = {}
    for bone in bones:
        target = _chain_target(bone, eps)
        if target is None:
            continue
        d = target.head - bone.head
        M = bone.matrix.copy()
        y = Vector(M.col[1][:3]).normalized()
        q = y.rotation_difference(d.normalized())
        R = q.to_matrix().to_4x4() @ Matrix(M.to_3x3()).to_4x4()
        R.translation = bone.head
        bone.matrix = R
        bone.length = d.length
        lengths[bone.name] = d.length

    for bone in bones:  # parents come first
        if bone.name in lengths:
            continue
        if bone.parent is not None and bone.parent.name in lengths:
            length = lengths[bone.parent.name] * 0.5
        else:
            near = [(c.head - bone.head).length for c in bone.children]
            near = [x for x in near if x > eps]
            length = min(near) if near else size * 0.05
        length = max(length, minLength)
        M = bone.matrix.copy()
        bone.length = length
        bone.matrix = M  # keep the orientation exactly
        lengths[bone.name] = length
