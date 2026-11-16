# PZA's Kenshi Motion Lab - Blender add-on test (headless). Author: PZA. GPLv3.
#   set BLENDER_USER_RESOURCES to an empty temp folder first (so the install does not touch your own Blender), then
#   blender -b --factory-startup --python tests/blender/test_addon.py -- <addon zip> <unrigged test .glb> <work dir>
import bpy, sys, json, os, addon_utils

argv = sys.argv[sys.argv.index('--') + 1:]
zip_path, glb, work = argv[0], argv[1], argv[2]
res = {'steps': [], 'fail': []}
def ok(cond, msg):
    res['steps'].append(('PASS ' if cond else 'FAIL ') + msg)
    if not cond: res['fail'].append(msg)

# 1. install the zip the way a modder does (Preferences > Get Extensions > Install from Disk)
bpy.ops.extensions.package_install_files(filepath=zip_path, repo='user_default', enable_on_install=True)
ok(hasattr(bpy.types.Scene, 'motion_lab') and hasattr(bpy.ops.motionlab, 'rig'), 'extension installs and registers')

bpy.ops.wm.read_factory_settings(use_empty=True)
if not hasattr(bpy.types.Scene, 'motion_lab'):
    addon_utils.enable('bl_ext.user_default.pza_motion_lab', default_set=True)
s = bpy.context.scene.motion_lab
s.out_dir = work
# 2. an unrigged model, as Meshy / Blender would give it
bpy.ops.import_scene.gltf(filepath=glb)
mesh = max((o for o in bpy.context.scene.objects if o.type == 'MESH'), key=lambda o: len(o.data.vertices))
bpy.context.view_layer.objects.active = mesh; mesh.select_set(True)
r = bpy.ops.motionlab.rig()
arm = next((o for o in bpy.context.scene.objects if o.type == 'ARMATURE' and 'motionlab_clips' in o), None)
ok(r == {'FINISHED'} and arm is not None, 'Auto-rig & Animate made an armature')
nb = len(arm.data.bones) if arm else 0
ok(nb == 38, 'bones: %d' % nb)
clips = json.loads(arm['motionlab_clips']) if arm else []
from bl_ext.user_default.pza_motion_lab import _clip_actions
acts = _clip_actions(arm) if arm else []
ok(len(acts) == 8, 'clips as actions: %s' % [n for n, _ in acts])
ok(mesh.hide_get(), 'original mesh hidden')
skinned = [o for o in bpy.context.scene.objects if o.type == 'MESH' and o.parent == arm and len(o.data.vertices) > 100]
ok(len(skinned) == 1 and len(skinned[0].vertex_groups) >= 36, 'skinned mesh with vertex groups')
# 3. play a clip: the walk moves the feet
bpy.context.view_layer.objects.active = arm
bpy.ops.motionlab.play(clip='walk')
ok(arm.animation_data.action is not None and arm.animation_data.action.name.startswith('walk'), 'play sets the walk action')
f0, f1 = arm.animation_data.action.frame_range
ok(abs((f1 - f0) - 30) < 0.5 and bpy.context.scene.render.fps == 30, 'walk is 1.0 s = 30 frames at 30 fps (frames %.1f..%.1f)' % (f0, f1))
pb = arm.pose.bones['leg_L1_04']
sc = bpy.context.scene
sc.frame_set(sc.frame_start); a0 = (arm.matrix_world @ pb.tail).copy()
sc.frame_set(sc.frame_start + 10); a1 = (arm.matrix_world @ pb.tail).copy()
ok((a1 - a0).length > 0.01, 'walk moves a foot (%.3f m in 10 frames)' % (a1 - a0).length)
# 4. markers: move one, drop a whole limb, rebuild
bpy.ops.motionlab.markers()
col = bpy.data.collections.get('Motion Lab markers')
n_mk = len(col.objects) if col else 0
ok(n_mk > 20, 'limb markers shown: %d' % n_mk)
for o in [o for o in col.objects if o['motionlab_limb'] == 'antenna_R']: bpy.data.objects.remove(o, do_unlink=True)
tip = next(o for o in col.objects if o['motionlab_limb'] == 'leg_L1' and o['motionlab_index'] == max(
    x['motionlab_index'] for x in col.objects if x['motionlab_limb'] == 'leg_L1'))
tip.location.x += 0.05
bpy.context.view_layer.objects.active = next(o for o in bpy.context.scene.objects if o.type == 'ARMATURE' and 'motionlab_clips' in o)
r = bpy.ops.motionlab.rebuild()
arms = [o for o in bpy.context.scene.objects if o.type == 'ARMATURE' and 'motionlab_clips' in o]
ok(r == {'FINISHED'} and len(arms) == 1, 'rebuild replaces the rig (%d armature)' % len(arms))
nb2 = len(arms[0].data.bones) if arms else 0
ok(nb2 == 36 and 'antenna_R_01' not in arms[0].data.bones, 'antenna_R dropped by deleting its markers: %d bones' % nb2)
ok(bpy.data.texts.get(bpy.context.scene.motion_lab.last_report) is not None, 'report text in Blender')
# 5. materials survive (base colour texture) and a picture of the rig mid-walk
arm = arms[0]
sk = [o for o in bpy.context.scene.objects if o.type == 'MESH' and o.parent == arm and len(o.data.vertices) > 100][0]
imgs = [n.image for m in sk.data.materials if m and m.use_nodes for n in m.node_tree.nodes if n.type == 'TEX_IMAGE' and n.image]
ok(len(imgs) >= 1 or not any(m for m in sk.data.materials), 'textures kept: %d image(s)' % len(imgs))
if len(argv) > 3:
    from mathutils import Vector
    bpy.context.view_layer.objects.active = arm
    bpy.ops.motionlab.play(clip='walk'); sc.frame_set(sc.frame_start + 8)
    cam = bpy.data.objects.new('cam', bpy.data.cameras.new('cam')); sc.collection.objects.link(cam)
    lo = Vector([min(v[i] for v in (sk.matrix_world @ Vector(c) for c in sk.bound_box)) for i in range(3)])
    hi = Vector([max(v[i] for v in (sk.matrix_world @ Vector(c) for c in sk.bound_box)) for i in range(3)])
    ctr = (lo + hi) / 2; d = (hi - lo).length
    cam.location = ctr + Vector((0.9 * d, -1.3 * d, 0.55 * d))
    cam.rotation_euler = (ctr - cam.location).to_track_quat('-Z', 'Y').to_euler()
    sc.camera = cam
    sun = bpy.data.objects.new('sun', bpy.data.lights.new('sun', 'SUN')); sun.rotation_euler = (0.7, 0.2, 0.6); sc.collection.objects.link(sun)
    sc.render.engine = 'BLENDER_WORKBENCH'; sc.display.shading.color_type = 'TEXTURE'; sc.display.shading.light = 'STUDIO'
    sc.render.resolution_x, sc.render.resolution_y = 640, 480; sc.render.filepath = argv[3]
    bpy.ops.render.render(write_still=True)
    ok(os.path.exists(argv[3]), 'render written')
print('ADDON_TEST ' + json.dumps(res))
