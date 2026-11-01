"""Import FVIS visibility animations as Blender actions.

One action per FVIS. Bone visibility hides the shapes bound to the bone
(the FSHP bone index, kept as the object's 'bfres_bone' property by
LodImporter), as the game does; material visibility hides the objects
using the material. Both become constant hide_viewport / hide_render keys
on the objects, one action slot per object.

As for material animations, only looping animations are left playing
(e.g. FldObj_Mound_A_01_Wait, which hides the fragments of the intact
mound); the others stay available as actions. A looping animation never
shows its last frame in the game (frame_count wraps to 0), so that frame
gets the values of frame 0.

When the same file has a skeletal animation of the same name, the game
plays both together, so the visibility keys are added to that action
(new object slots) instead of a separate one.
"""
import logging; log = logging.getLogger(__name__)

import bpy
import numpy as np

from .MaterialAnimationImporter import CONSTANT, _keys, objects_using, materials_named


class VisibilityAnimationImporter:
    def __init__(self, context):
        self.context = context

    def import_anims(self, anims, objects, materials, shared=None):
        objects = [ob for ob in objects if ob.type == 'MESH']
        if not anims or not objects:
            return []
        actions = []
        for anim in anims:
            try:
                act = self._import_one(anim, objects, materials, (shared or {}).get(anim.name))
            except Exception:
                log.exception("Visibility animation %s could not be imported", anim.name)
                continue
            if act is not None:
                actions.append(act)
        log.info("Imported %d visibility animation(s)", len(actions))
        return actions

    def _targets(self, anim, name, objects, materials):
        if anim.is_material:
            obs = set()
            for mat in materials_named(name, materials):
                obs.update(objects_using(mat))
            return [ob for ob in objects if ob in obs]
        return [ob for ob in objects if ob.get('bfres_bone') == name]

    def _import_one(self, anim, objects, materials, act=None):
        frames = anim.frames()
        sample_at = frames % anim.frame_count if anim.looping and anim.frame_count > 0 else frames
        created = act is None
        if created:
            act = bpy.data.actions.new(name=anim.name or 'VisibilityAnim')
            act.use_fake_user = True
            act.use_frame_range = True
            act.frame_start, act.frame_end = 0, max(anim.frame_count, 1)
            act.use_cyclic = anim.looping
            act['bfres_looping'] = anim.looping
        wrote = False
        for index, name in enumerate(anim.names):
            hidden = (~anim.sample(index, sample_at)).astype(np.float64)
            for ob in self._targets(anim, name, objects, materials):
                if ob.animation_data is None:
                    ob.animation_data_create()
                previous = ob.animation_data.action
                previous_slot = ob.animation_data.action_slot
                ob.animation_data.action = act
                ob.animation_data.action_slot = act.slots.new('OBJECT', ob.name)
                for path in ('hide_viewport', 'hide_render'):
                    fc = act.fcurve_ensure_for_datablock(ob, path, index=0,
                                                         group_name='visibility:' + name)
                    _keys(fc, frames, hidden, CONSTANT)
                wrote = True
                if previous is not None or not anim.looping:
                    ob.animation_data.action = previous
                    if previous is not None:
                        ob.animation_data.action_slot = previous_slot
        if not wrote:
            if created:
                bpy.data.actions.remove(act)
            return None
        return act
