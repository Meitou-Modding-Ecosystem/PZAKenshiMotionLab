# PZA's Kenshi Motion Lab - Blender add-on test: a model with a Kenshi-style Bip01 rig keeps its rig. Author: PZA. GPLv3.
#   BLENDER_USER_RESOURCES=<empty folder> blender -b --factory-startup --python tests/blender/test_addon_ownrig.py -- <addon zip> <biped .glb> <work dir>
import bpy, sys, json

argv = sys.argv[sys.argv.index('--') + 1:]
zip_path, glb, work = argv[0], argv[1], argv[2]
res = {'steps': [], 'fail': []}
def ok(cond, msg):
    res['steps'].append(('PASS ' if cond else 'FAIL ') + msg)
    if not cond: res['fail'].append(msg)

bpy.ops.extensions.package_install_files(filepath=zip_path, repo='user_default', enable_on_install=True)
ok(hasattr(bpy.ops.motionlab, 'rig'), 'extension installs')
bpy.ops.wm.read_factory_settings(use_empty=True)
if not hasattr(bpy.types.Scene, 'motion_lab'):
    import addon_utils; addon_utils.enable('bl_ext.user_default.pza_motion_lab', default_set=True)
bpy.context.scene.motion_lab.out_dir = work
bpy.ops.import_scene.gltf(filepath=glb)
src_arm = next(o for o in bpy.context.scene.objects if o.type == 'ARMATURE')
mesh = max((o for o in bpy.context.scene.objects if o.type == 'MESH'), key=lambda o: len(o.data.vertices))
n_src = len(src_arm.data.bones)
bpy.context.view_layer.objects.active = mesh; mesh.select_set(True)
r = bpy.ops.motionlab.rig()
arm = next((o for o in bpy.context.scene.objects if o.type == 'ARMATURE' and 'motionlab_clips' in o), None)
ok(r == {'FINISHED'} and arm is not None, 'rigged')
ok(arm is not None and len(arm.data.bones) == n_src and 'Bip01 L Thigh' in arm.data.bones, 'own Bip01 rig kept (%d bones)' % (len(arm.data.bones) if arm else 0))
clips = json.loads(arm['motionlab_clips']) if arm else []
ok(clips == ['idle', 'walk', 'run'], 'clips for a person: %s' % clips)
ok(arm.get('motionlab_plan') == 'biped', 'plan: %s' % arm.get('motionlab_plan'))
ok(src_arm.hide_get() and mesh.hide_get(), 'original model and its old armature hidden')
bpy.context.view_layer.objects.active = arm
bpy.ops.motionlab.play(clip='walk')
sc = bpy.context.scene; pb = arm.pose.bones['Bip01 L Calf']
sc.frame_set(sc.frame_start); a0 = (arm.matrix_world @ pb.head).copy()
sc.frame_set(sc.frame_start + 10); a1 = (arm.matrix_world @ pb.head).copy()
ok((a1 - a0).length > 0.02, 'walk moves the knee (%.3f m in 10 frames)' % (a1 - a0).length)
r = bpy.ops.motionlab.markers()
ok(r == {'CANCELLED'}, 'markers explain themselves on an own rig (no crash)')
print('ADDON_TEST ' + json.dumps(res))
