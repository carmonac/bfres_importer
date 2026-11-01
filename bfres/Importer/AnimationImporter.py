"""Import FSKA skeletal animations as Blender actions.

Each FSKA becomes one action on the armature. Every bone animation is
sampled at each integer frame (curves are cubic in the game; sampling keeps
their exact shape without converting tangents) and written as linear
keyframes on the pose bone's location, rotation_quaternion and scale.

Pose bone values are the basis relative to the rest pose:
    basis = rest_local^-1 @ anim_local
where rest_local is the bone's parent-relative rest matrix in the armature
and anim_local is built from the animated game values in the same
convention SkeletonImporter used for the rest pose (animmath.bone_local).
"""
import logging; log = logging.getLogger(__name__)
import bpy
import numpy as np

from . import animmath

REST_PROP = 'bfres_rest'  # bone custom property: scale(3) + rot(4) + pos(3)
CHANNELS = (('location', 3), ('rotation_quaternion', 4), ('scale', 3))
LINEAR = 1  # Keyframe.interpolation enum value of 'LINEAR'


def store_rest(arm_obj, fskl):
    """Remember each bone's game rest values on the armature, so animations
    imported later (from a separate file) can fill unanimated components."""
    bones = arm_obj.data.bones
    for b in fskl.bones:
        if b.name in bones:
            bones[b.name][REST_PROP] = list(b.scale[:3]) + list(b.rot[:4]) + list(b.pos[:3])


def _rest_values(bone):
    v = bone.get(REST_PROP)
    if v is not None and len(v) == 10:
        v = list(v)
        return v[0:3], v[3:7], v[7:10]
    # armature imported before rest values were stored: identity values;
    # only affects components that the animation neither stores nor curves
    return (1, 1, 1), (0, 0, 0, 1), (0, 0, 0)


def _rest_local(bone):
    M = np.array(bone.matrix_local, dtype=np.float64)
    if bone.parent:
        M = np.linalg.inv(np.array(bone.parent.matrix_local, dtype=np.float64)) @ M
    return M


def find_target_armature(context, bone_names):
    """The armature an animation file should go on: the active object if it
    is an armature, else the scene armature sharing the most bone names."""
    obj = context.active_object
    if obj is not None and obj.type == 'ARMATURE':
        return obj
    names = set(bone_names)
    best, best_n = None, 0
    for ob in context.scene.objects:
        if ob.type == 'ARMATURE':
            n = len(names & set(ob.data.bones.keys()))
            if n > best_n:
                best, best_n = ob, n
    return best


class AnimationImporter:
    def __init__(self, context):
        self.context = context

    def import_anims(self, anims, arm_obj):
        """Create one action per FSKA on `arm_obj`. Returns the actions."""
        if arm_obj is None:
            log.warning("No armature to put %d animation(s) on", len(anims))
            return []
        arm_obj.animation_data_create()
        previous = arm_obj.animation_data.action
        actions = []
        for anim in anims:
            try:
                act = self._import_one(anim, arm_obj)
            except Exception:
                log.exception("Animation %s could not be imported", anim.name)
                continue
            actions.append(act)
        # leave the first animation playing (or what was there before)
        arm_obj.animation_data.action = previous or (actions[0] if actions else None)
        if actions and arm_obj.animation_data.action is actions[0]:
            scene = self.context.scene
            scene.frame_start = int(actions[0].frame_range[0])
            scene.frame_end = int(actions[0].frame_range[1])
        log.info("Imported %d animation(s) onto %s", len(actions), arm_obj.name)
        return actions

    def _import_one(self, anim, arm_obj):
        frames = anim.frames()
        n = len(frames)
        act = bpy.data.actions.new(name=anim.name or 'Anim')
        act.use_fake_user = True  # keep every animation, not only the active one
        act.use_frame_range = True
        act.frame_start, act.frame_end = 0, max(anim.frame_count, 1)
        act.use_cyclic = anim.looping
        act['bfres_looping'] = anim.looping
        arm_obj.animation_data.action = act

        bones = arm_obj.data.bones
        pose = arm_obj.pose.bones
        missing = []
        for ba in anim.bone_anims:
            bone = bones.get(ba.name)
            if bone is None:
                missing.append(ba.name)
                continue
            rs, rr, rt = _rest_values(bone)
            s, r, t = ba.sample(frames, rs, rr, rt)
            anim_local = animmath.bone_local(s, r, t, anim.euler, bone.parent is None)
            basis = animmath.pose_basis(_rest_local(bone), anim_local)
            loc, quat, scale = animmath.decompose(basis)
            quat = animmath.make_continuous(quat)
            pose[ba.name].rotation_mode = 'QUATERNION'
            values = {'location': loc, 'rotation_quaternion': quat, 'scale': scale}
            for path, size in CHANNELS:
                data_path = 'pose.bones["%s"].%s' % (ba.name.replace('"', '\\"'), path)
                for i in range(size):
                    fc = act.fcurve_ensure_for_datablock(arm_obj, data_path, index=i,
                                                         group_name=ba.name)
                    kps = fc.keyframe_points
                    kps.add(n)
                    co = np.empty(2 * n, dtype=np.float32)
                    co[0::2] = frames
                    co[1::2] = values[path][:, i]
                    kps.foreach_set('co', co)
                    kps.foreach_set('interpolation', np.full(n, LINEAR, dtype=np.int32))
                    fc.update()
        if missing:
            log.warning("%s: %d animated bone(s) not in %s, skipped (%s...)",
                anim.name, len(missing), arm_obj.name, ', '.join(missing[:5]))
        return act
