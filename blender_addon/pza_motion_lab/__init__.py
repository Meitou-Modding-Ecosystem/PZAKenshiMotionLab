# PZA's Kenshi Motion Lab - Blender add-on. Author: PZA. Licence: GPLv3 or later.
#
# 3D View > Sidebar (N) > "Motion Lab":
#   1. select your model's mesh, press "Auto-rig & Animate": the model is rigged (named skeleton, 3 weights per
#      vertex) and gets its clips (idle, walk, run, hover, takeoff, land, attack, sting) as actions;
#   2. play any clip from the list;
#   3. if a limb was found wrong: "Show limb markers", move / delete markers, "Rebuild from markers";
#   4. export to Kenshi with the Kenshi IO add-on (the report checks bones and weights against Kenshi's limits).
# The rigging itself is the Motion Lab's Python package (bundled in this add-on as `motionlab`), the same code
# as the command line.
import json, os, tempfile
import bpy
from bpy.props import BoolProperty, EnumProperty, IntProperty, StringProperty, PointerProperty

bl_info = {"name": "PZA's Kenshi Motion Lab", "author": "PZA", "version": (0, 1, 0), "blender": (4, 2, 0),
           "location": "3D View > Sidebar > Motion Lab", "category": "Animation",
           "description": "Auto-rig a model and animate it from code, for Kenshi"}

CLIPS = ('idle', 'walk', 'run', 'hover', 'takeoff', 'land', 'attack', 'sting')
MARKERS = 'Motion Lab markers'


def _ml():
    from . import motionlab  # bundled package
    from .motionlab import pipeline
    return pipeline


# glTF (Y up) <-> Blender (Z up), the conversion Blender's glTF importer uses
def g2b(p): return (p[0], -p[2], p[1])
def b2g(p): return [p[0], p[2], -p[1]]


class MotionLabSettings(bpy.types.PropertyGroup):
    out_dir: StringProperty(name='Output folder', subtype='DIR_PATH', default='//motionlab/',
                            description='Where the rigged .glb, report and landmarks go ("//" = next to the .blend; a temp folder if unsaved)')
    ground_wings: EnumProperty(name='Wings on the ground', items=[
        ('hidden', 'Hidden', 'Wings shrink away on landing and grow back on take-off (0.25 s)'),
        ('folded', 'Folded', 'Wings stay folded on the back on the ground')], default='hidden')
    leg_bones: IntProperty(name='Leg bones', default=4, min=2, max=6)
    wing_bones: IntProperty(name='Wing bones', default=2, min=1, max=4)
    antenna_bones: IntProperty(name='Antenna bones', default=2, min=1, max=4)
    abdomen_bones: IntProperty(name='Abdomen bones', default=3, min=1, max=5)
    fps: IntProperty(name='Frames per second', default=30, min=10, max=60)
    own_rig: EnumProperty(name='Rig', items=[
        ('auto', 'Keep its own rig if known', "Kenshi's Bip01 rigs (humans, animals) and Motion Lab rigs keep their skeleton and skin; others are auto-rigged"),
        ('no', 'Always auto-rig', 'Build a new skeleton from the mesh shape even if it has one')], default='auto')
    show_options: BoolProperty(name='Options', default=False)
    last_report: StringProperty(default='')


for _c in CLIPS:
    MotionLabSettings.__annotations__['clip_' + _c] = BoolProperty(name=_c, default=True)


def _out_dir(s):
    d = bpy.path.abspath(s.out_dir) if s.out_dir else ''
    if not d or (s.out_dir.startswith('//') and not bpy.data.filepath):
        d = os.path.join(tempfile.gettempdir(), 'pza_motion_lab')
    os.makedirs(d, exist_ok=True)
    return d


def _rig_of(obj):
    if obj.parent is not None and obj.parent.type == 'ARMATURE': return obj.parent
    for md in obj.modifiers:
        if md.type == 'ARMATURE' and md.object is not None: return md.object
    return None


def _export_selected(obj, path, with_rig=True):
    """Write the chosen mesh (as it stands, materials included) to a .glb the rig pipeline reads. With its armature
    when it has one, so a rig the Motion Lab knows (Kenshi's Bip01 bones from Kenshi IO, a Motion Lab rig) is kept."""
    for o in bpy.context.selected_objects: o.select_set(False)
    obj.select_set(True); bpy.context.view_layer.objects.active = obj
    arm = _rig_of(obj) if with_rig else None
    if arm is not None: arm.select_set(True)
    bpy.ops.export_scene.gltf(filepath=path, export_format='GLB', use_selection=True, export_animations=False,
                              export_yup=True, export_apply=False, export_skins=arm is not None)


def _import_result(glb, name):
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=glb)
    new = [o for o in bpy.data.objects if o not in before]
    col = bpy.data.collections.new('Motion Lab: ' + name)
    bpy.context.scene.collection.children.link(col)
    for o in new:
        for c in list(o.users_collection): c.objects.unlink(o)
        col.objects.link(o)
    arm = next((o for o in new if o.type == 'ARMATURE'), None)
    for o in new:      # the glTF importer's bone-display helper (an Icosphere) is a visible object: hide it
        if o.type == 'MESH' and o.parent is None and len(o.data.vertices) <= 200:
            o.hide_set(True); o.hide_render = True
    return arm, new, col


def _clip_actions(arm):
    """The clips that belong to this armature: actions named in its stored clip list."""
    names = json.loads(arm.get('motionlab_clips', '[]'))
    out = []
    for nm in names:
        act = bpy.data.actions.get(nm) or next((a for a in bpy.data.actions if a.name.startswith(nm + '_') or a.name.startswith(nm + '.')), None)
        if act: out.append((nm, act))
    return out


def _active_armature(context):
    o = context.active_object
    if o is None: return None
    if o.type == 'ARMATURE' and 'motionlab_landmarks' in o: return o
    if o.type == 'MESH' and o.parent and o.parent.type == 'ARMATURE' and 'motionlab_landmarks' in o.parent: return o.parent
    if o.type == 'EMPTY' and o.get('motionlab_limb'):
        arm = bpy.data.objects.get(o.get('motionlab_armature', ''))
        return arm
    return next((a for a in context.scene.objects if a.type == 'ARMATURE' and 'motionlab_landmarks' in a), None)


def _run(context, src_glb, name, landmarks=None):
    s = context.scene.motion_lab
    clips = [c for c in CLIPS if getattr(s, 'clip_' + c)]
    log = []
    r = _ml().rig_file(src_glb, _out_dir(s), log=log.append, name=name, landmarks=landmarks, leg_bones=s.leg_bones,
                       wing_bones=s.wing_bones, antenna_bones=s.antenna_bones, abdomen_bones=s.abdomen_bones,
                       ground_wings=s.ground_wings, clips=','.join(clips), fps=s.fps, gifs=False, png=False,
                       own_rig=s.own_rig)
    context.scene.render.fps = s.fps; context.scene.render.fps_base = 1.0   # keys land on the clip's own frames
    arm, new, col = _import_result(r['glb'], r['name'])
    if arm is None: raise RuntimeError('the rigged file has no armature')
    arm['motionlab_landmarks'] = r['landmarks'] or ''; arm['motionlab_source'] = src_glb
    arm['motionlab_plan'] = r.get('plan') or ''
    arm['motionlab_clips'] = json.dumps(r['clips']); arm['motionlab_report'] = r['report']
    txt = bpy.data.texts.get(r['name'] + ' report') or bpy.data.texts.new(r['name'] + ' report')
    txt.clear(); txt.write(r['report_text'] + '\n\nLog:\n' + '\n'.join(log) + '\n')
    s.last_report = txt.name
    if arm.animation_data and arm.animation_data.nla_tracks:
        for tr in arm.animation_data.nla_tracks: tr.mute = True
    acts = _clip_actions(arm)
    if acts: _play(context, arm, acts[0][1], start=False)
    return r, arm, new


def _play(context, arm, act, start=True):
    ad = arm.animation_data or arm.animation_data_create()
    for tr in ad.nla_tracks: tr.mute = True
    ad.action = act
    try:
        if getattr(act, 'slots', None) and len(act.slots) and getattr(ad, 'action_slot', None) is None:
            ad.action_slot = act.slots[0]
    except Exception:
        pass
    sc = context.scene
    f0, f1 = act.frame_range
    sc.frame_start = int(f0); sc.frame_end = max(int(round(f1)) - 1, int(f0) + 1)
    sc.render.fps = sc.motion_lab.fps
    sc.frame_set(int(f0))
    if start and context.screen and not context.screen.is_animation_playing:
        bpy.ops.screen.animation_play()


class MOTIONLAB_OT_rig(bpy.types.Operator):
    bl_idname = 'motionlab.rig'
    bl_label = 'Auto-rig & Animate'
    bl_description = 'Find legs, wings and antennae in the selected mesh, rig it and add the clips as actions'
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.active_object is not None and context.active_object.type == 'MESH'

    def execute(self, context):
        obj = context.active_object
        name = bpy.path.clean_name(obj.name)
        src = os.path.join(_out_dir(context.scene.motion_lab), name + '_source.glb')
        try:
            _export_selected(obj, src, with_rig=context.scene.motion_lab.own_rig != 'no')
            r, arm, new = _run(context, src, name)
        except Exception as ex:
            self.report({'ERROR'}, 'Motion Lab: %s' % ex); return {'CANCELLED'}
        obj.hide_set(True); obj.hide_render = True
        if obj.parent is not None and obj.parent.type == 'ARMATURE':      # the model's old rig (e.g. Meshy's)
            obj.parent.hide_set(True); obj.parent.hide_render = True
            for o in context.scene.objects:                                   # and its bone-display helper
                if o.type == 'MESH' and o.parent is None and o.name.startswith('Icosphere') and len(o.data.vertices) <= 200:
                    o.hide_set(True); o.hide_render = True
        for o in context.selected_objects: o.select_set(False)
        arm.select_set(True); context.view_layer.objects.active = arm
        fr = r['frame']
        self.report({'INFO'}, 'Rigged %s: %d bones, %d clips (model turned %.1f deg to face forward; original hidden)' % (
            r['name'], r['bones'], len(r['clips']), fr.get('turn_about_y_deg', 0.0)))
        return {'FINISHED'}


class MOTIONLAB_OT_play(bpy.types.Operator):
    bl_idname = 'motionlab.play'
    bl_label = 'Play clip'
    bl_options = {'REGISTER'}
    clip: StringProperty()

    def execute(self, context):
        arm = _active_armature(context)
        if arm is None: self.report({'ERROR'}, 'no Motion Lab rig in the scene'); return {'CANCELLED'}
        act = dict(_clip_actions(arm)).get(self.clip)
        if act is None: self.report({'ERROR'}, 'no clip %s' % self.clip); return {'CANCELLED'}
        if context.screen and context.screen.is_animation_playing: bpy.ops.screen.animation_cancel(restore_frame=False)
        _play(context, arm, act)
        return {'FINISHED'}


class MOTIONLAB_OT_markers(bpy.types.Operator):
    bl_idname = 'motionlab.markers'
    bl_label = 'Show limb markers'
    bl_description = 'One small sphere per point of every limb found (hip to tip). Move or delete them, then Rebuild'
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        arm = _active_armature(context)
        if arm is None: self.report({'ERROR'}, 'rig a model first'); return {'CANCELLED'}
        if not arm.get('motionlab_landmarks'):
            self.report({'INFO'}, 'This model kept its own rig (%s): limb markers are for auto-rigged models' % arm.get('motionlab_plan', ''))
            return {'CANCELLED'}
        lm = json.load(open(arm['motionlab_landmarks'], encoding='utf-8'))
        col = bpy.data.collections.get(MARKERS) or bpy.data.collections.new(MARKERS)
        if col.name not in context.scene.collection.children: context.scene.collection.children.link(col)
        for o in list(col.objects): bpy.data.objects.remove(o, do_unlink=True)
        size = 0.012 * float(lm.get('size', 2.0))
        for kind in ('legs', 'wings', 'antennae'):
            for e in lm.get(kind, []):
                pts = e['points']
                keep = sorted(set([0, len(pts) - 1] + [int(round(i * (len(pts) - 1) / 5)) for i in range(6)]))
                for k, i in enumerate(keep):
                    em = bpy.data.objects.new('%s.%02d' % (e['name'], k), None)
                    em.empty_display_type = 'SPHERE'; em.empty_display_size = size * (1.6 if k == 0 or k == len(keep) - 1 else 1.0)
                    em.location = g2b(pts[i])
                    em['motionlab_limb'] = e['name']; em['motionlab_kind'] = kind; em['motionlab_index'] = k
                    em['motionlab_armature'] = arm.name
                    col.objects.link(em)
        self.report({'INFO'}, 'Markers: big spheres = hip / tip. Move them, delete a whole limb to drop it, then Rebuild')
        return {'FINISHED'}


class MOTIONLAB_OT_rebuild(bpy.types.Operator):
    bl_idname = 'motionlab.rebuild'
    bl_label = 'Rebuild from markers'
    bl_description = 'Rig again with the limbs where the markers are now'
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        arm = _active_armature(context)
        col = bpy.data.collections.get(MARKERS)
        if arm is None or col is None: self.report({'ERROR'}, 'show the limb markers first'); return {'CANCELLED'}
        lm = json.load(open(arm['motionlab_landmarks'], encoding='utf-8'))
        limbs = {}
        for o in col.objects:
            if o.get('motionlab_armature') != arm.name: continue
            limbs.setdefault((o['motionlab_kind'], o['motionlab_limb']), []).append((o['motionlab_index'], b2g(o.location)))
        for kind in ('legs', 'wings', 'antennae'):
            out = []
            for e in lm.get(kind, []):
                pts = limbs.get((kind, e['name']))
                if not pts or len(pts) < 2: continue          # all its markers deleted: the limb is dropped
                e = dict(e); e['points'] = [p for _, p in sorted(pts)]
                out.append(e)
            lm[kind] = out
        src = arm['motionlab_source']
        old_col = arm.users_collection[0] if arm.users_collection else None
        try:
            r, new_arm, new = _run(context, src, _base_name(arm), landmarks=lm)
        except Exception as ex:
            self.report({'ERROR'}, 'Motion Lab: %s' % ex); return {'CANCELLED'}
        if old_col is not None and old_col.name.startswith('Motion Lab: '):
            for o in list(old_col.objects): bpy.data.objects.remove(o, do_unlink=True)
            bpy.data.collections.remove(old_col)
        for o in list(col.objects):
            o['motionlab_armature'] = new_arm.name
        context.view_layer.objects.active = new_arm
        self.report({'INFO'}, 'Rebuilt: %d bones, %d clips' % (r['bones'], len(r['clips'])))
        return {'FINISHED'}


def _base_name(arm):
    src = arm.get('motionlab_source', '')
    b = os.path.basename(src)
    return b[:-len('_source.glb')] if b.endswith('_source.glb') else (arm.name or 'creature')


class MOTIONLAB_OT_report(bpy.types.Operator):
    bl_idname = 'motionlab.report'
    bl_label = 'Show report'
    bl_description = 'Open the rig report (bones and weights against Kenshi limits, limbs, clip speeds, foot checks)'

    def execute(self, context):
        name = context.scene.motion_lab.last_report
        txt = bpy.data.texts.get(name)
        if txt is None: self.report({'ERROR'}, 'no report yet'); return {'CANCELLED'}
        for area in context.screen.areas:
            if area.type == 'TEXT_EDITOR':
                area.spaces.active.text = txt; return {'FINISHED'}
        self.report({'INFO'}, 'Report is in the Text Editor as "%s"' % txt.name)
        return {'FINISHED'}


class MOTIONLAB_PT_panel(bpy.types.Panel):
    bl_space_type = 'VIEW_3D'; bl_region_type = 'UI'; bl_category = 'Motion Lab'
    bl_label = "PZA's Kenshi Motion Lab"

    def draw(self, context):
        s = context.scene.motion_lab; L = self.layout
        box = L.box(); box.label(text='1. Rig', icon='ARMATURE_DATA')
        o = context.active_object
        box.label(text='Model: %s' % (o.name if o and o.type == 'MESH' else 'select a mesh'))
        box.operator('motionlab.rig', icon='PLAY')
        box.prop(s, 'show_options', icon='TRIA_DOWN' if s.show_options else 'TRIA_RIGHT')
        if s.show_options:
            box.prop(s, 'own_rig'); box.prop(s, 'ground_wings'); box.prop(s, 'out_dir')
            r = box.row(align=True); r.prop(s, 'leg_bones'); r.prop(s, 'wing_bones')
            r = box.row(align=True); r.prop(s, 'antenna_bones'); r.prop(s, 'abdomen_bones')
            box.prop(s, 'fps')
            g = box.grid_flow(columns=4, align=True)
            for c in CLIPS: g.prop(s, 'clip_' + c, toggle=True)
        arm = _active_armature(context)
        box = L.box(); box.label(text='2. Clips', icon='ACTION')
        if arm is None: box.label(text='rig a model first')
        else:
            col = box.column(align=True)
            for nm, act in _clip_actions(arm):
                col.operator('motionlab.play', text=nm, icon='PLAY').clip = nm
            box.operator('motionlab.report', icon='TEXT')
        box = L.box(); box.label(text='3. Fix a limb', icon='EMPTY_DATA')
        box.operator('motionlab.markers', icon='SPHERE')
        box.operator('motionlab.rebuild', icon='FILE_REFRESH')
        box = L.box(); box.label(text='4. Kenshi', icon='EXPORT')
        box.label(text='Export with the Kenshi IO add-on')
        box.label(text='(the report checks bones / weights)')


classes = (MotionLabSettings, MOTIONLAB_OT_rig, MOTIONLAB_OT_play, MOTIONLAB_OT_markers, MOTIONLAB_OT_rebuild,
           MOTIONLAB_OT_report, MOTIONLAB_PT_panel)


def register():
    for c in classes: bpy.utils.register_class(c)
    bpy.types.Scene.motion_lab = PointerProperty(type=MotionLabSettings)


def unregister():
    del bpy.types.Scene.motion_lab
    for c in reversed(classes): bpy.utils.unregister_class(c)
