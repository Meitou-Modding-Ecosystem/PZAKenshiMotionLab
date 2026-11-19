# PZA's Kenshi Motion Lab - convert reference animation files (.fbx, e.g. Mixamo "Without Skin" downloads) to .glb with
# headless Blender, and report what each holds. Author: PZA. GPLv3.
#   blender -b --factory-startup --python tools/fbx_to_glb.py -- <input folder> <output folder> [report.json]
# Reference clips are for measuring only: they stay outside the repo (out/ is git-ignored) and are never shipped.
import bpy, sys, os, json, glob

argv = sys.argv[sys.argv.index('--') + 1:]
src, dst = argv[0], argv[1]
os.makedirs(dst, exist_ok=True)
report = []
for path in sorted(glob.glob(os.path.join(src, '*.fbx'))):
    name = os.path.splitext(os.path.basename(path))[0]
    bpy.ops.wm.read_factory_settings(use_empty=True)
    r = {'name': name, 'ok': False}
    try:
        bpy.ops.import_scene.fbx(filepath=path, automatic_bone_orientation=False)
        arms = [o for o in bpy.data.objects if o.type == 'ARMATURE']
        acts = list(bpy.data.actions)
        r['bones'] = len(arms[0].data.bones) if arms else 0
        r['actions'] = len(acts)
        if acts:
            f0, f1 = acts[0].frame_range
            r['frames'] = [float(f0), float(f1)]
            r['fps'] = bpy.context.scene.render.fps / bpy.context.scene.render.fps_base
        out = os.path.join(dst, name + '.glb')
        bpy.ops.export_scene.gltf(filepath=out, export_format='GLB', export_animations=True, export_skins=True)
        r['glb_bytes'] = os.path.getsize(out)
        r['ok'] = bool(arms) and bool(acts) and r['bones'] > 20
    except Exception as ex:
        r['error'] = '%s: %s' % (type(ex).__name__, ex)
    report.append(r)
print('FBX2GLB ' + json.dumps(report))
if len(argv) > 2:
    with open(argv[2], 'w') as f: json.dump(report, f, indent=1)
