# PZA's Kenshi Motion Lab - open a rigged .glb in Blender (headless) and list what Blender sees.
# Author: PZA. GPLv3.
#   blender -b --factory-startup --python tools/blender_check.py -- <file.glb> [out.json]
# Prints bones, skinned vertices, the actions and their frame ranges: proof that Blender (and so the Kenshi IO
# exporter in Blender, for .mesh / .skeleton) reads the file the way the Motion Lab wrote it.
import bpy, sys, json

argv = sys.argv[sys.argv.index('--') + 1:]
path = argv[0]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=path)
arm = [o for o in bpy.data.objects if o.type == 'ARMATURE']
meshes = [o for o in bpy.data.objects if o.type == 'MESH']
out = {'file': path, 'armatures': len(arm), 'bones': len(arm[0].data.bones) if arm else 0,
       'bone_names': [b.name for b in arm[0].data.bones] if arm else [],
       'meshes': [{'name': o.name, 'vertices': len(o.data.vertices), 'groups': len(o.vertex_groups),
                   'armature_modifier': any(md.type == 'ARMATURE' for md in o.modifiers)} for o in meshes],
       'actions': sorted([{'name': a.name, 'frames': [float(a.frame_range[0]), float(a.frame_range[1])]} for a in bpy.data.actions],
                         key=lambda x: x['name'])}
print('BLENDER_CHECK ' + json.dumps(out))
if len(argv) > 1:
    with open(argv[1], 'w') as f: json.dump(out, f, indent=1)
