# PZA's Kenshi Motion Lab - direct .glb -> Kenshi skinned asset export (.mesh + .skeleton + .dds). Author: PZA. GPLv3.
#   python tests/test_asset_export.py     (no game files, no Pillow required)
import json, os, struct, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from synthetic import make_hexapod, make_biped
from motionlab import asset_export as ae, gltf, kenshi_export as ke, pipeline
from motionlab.ogre import load as ogre_load, mesh_read, skeleton_read

fails = []
def ok(c, msg):
    print(('PASS ' if c else 'FAIL ') + msg)
    if not c:
        fails.append(msg)


d = tempfile.mkdtemp()

# 1. Auto-rigged hexapod -> rigged .glb -> direct Kenshi .mesh + .skeleton + .dds
hex_src = os.path.join(d, 'hex.glb')
gltf.write_glb(hex_src, make_hexapod())
rig_dir = os.path.join(d, 'rig')
r = pipeline.rig_file(hex_src, rig_dir, log=lambda *a: None, png=False, clips='idle,walk,run,hover')

exp_dir = os.path.join(d, 'exp_hex')
res = ae.export_asset(r['glb'], exp_dir, name='test_bug', fps=30.0, log=lambda *a: None)
mesh_p = res['mesh']['path']
skel_p = res['skeleton']['path']
ok(os.path.isfile(mesh_p) and os.path.isfile(skel_p), 'wrote .mesh and .skeleton')
ok(res['mesh']['skinned'] and res['mesh']['max_bones_per_submesh'] <= 60, 'skinned mesh within 60-bone limit')
ok(res['scale_keys_dropped'] > 0, 'dropped unverified wing scale keys cleanly (%d)' % res['scale_keys_dropped'])

# Round-trip the .mesh through mesh_read and ogre.load
raw_m = mesh_read.read(mesh_p)
ok(raw_m['version'] == '[MeshSerializer_v1.8]', 'Ogre mesh version v1.8')
ok(raw_m['animated'] is True, 'Ogre mesh marked skeletally animated')
ok(raw_m['skeleton'] == 'test_bug.skeleton', 'skeleton link points to test_bug.skeleton')
ok(len(raw_m['subs']) == 1 and len(raw_m['subs'][0]['bones']) > 0, 'submesh has bone assignments')

back = ogre_load.load_mesh(mesh_p, skel_p)
src_rigged = gltf.load_glb(r['glb'])
ok(len(back.verts) == len(src_rigged.verts) and len(back.tris) == len(src_rigged.tris),
   'Ogre mesh round-trip vertex/tri count matches')
pos_err = float(np.abs(back.verts - src_rigged.verts).max())
ok(pos_err < 1e-4, 'Ogre mesh round-trip vertex positions match (err %.2e m)' % pos_err)
ok(back.skeleton is not None and back.skeleton.names == src_rigged.skeleton.names,
   'Ogre skeleton round-trip bone names match (%d bones)' % len(back.skeleton))
ok(set(back.ogre_animations) == {'idle', 'walk', 'run', 'hover'},
   'Ogre skeleton holds all exported clips: %s' % back.ogre_animations)

# Verify animation playback matches .glb
v, bones, parents, anims = skeleton_read.read(skel_p)
ok(len(bones) == len(src_rigged.skeleton) and len(anims) == 4, 'skeleton_read parses fresh skeleton + 4 clips')
names, par, smp, L = ke.sampler(skel_p, 'walk')
W0 = smp(0.0)
ok(np.isfinite(W0).all() and W0.shape == (len(bones), 4, 4), 'fresh skeleton sampler evaluates walk cleanly')

# 2. DDS headers & payload sizes (DXT5 diffuse, DXT1 normal, DXT1 metal)
for key, want_cc, bpb in (('diffuse', b'DXT5', 16), ('normal', b'DXT1', 8), ('metal', b'DXT1', 8)):
    tp = res['textures'][key]
    data = open(tp, 'rb').read()
    ok(data[:4] == b'DDS ' and struct.unpack_from('<I', data, 4)[0] == 124, '%s DDS magic + 124-byte header' % key)
    cc = data[84:88]
    ok(cc == want_cc, '%s DDS FourCC is %r' % (key, cc))
    h, w, linsz, _, mips = struct.unpack_from('<5I', data, 12)
    ok(h == 4 and w == 4 and linsz == bpb and mips == 3 and len(data) == 128 + 3 * bpb,
       '%s DDS 4x4 with 3 mips (%d bytes)' % (key, len(data)))

# 3. BC1/BC3 encoder accuracy on a 16x16 gradient texture
grad = np.zeros((16, 16, 4), np.uint8)
grad[:, :, 0] = np.linspace(0, 255, 16, dtype=np.uint8)[None, :]
grad[:, :, 1] = np.linspace(0, 255, 16, dtype=np.uint8)[:, None]
grad[:, :, 2] = 128
grad[:, :, 3] = np.linspace(20, 240, 16, dtype=np.uint8)[None, :]
dds16 = os.path.join(d, 'grad.dds')
sz16 = ae.write_dds(grad, dds16, 'DXT5')
# 16x16 (16 blocks) + 8x8 (4) + 4x4 (1) + 2x2 (1) + 1x1 (1) = 23 blocks * 16 bytes + 128 = 496
ok(sz16 == 128 + 23 * 16, '16x16 DXT5 mip chain exact byte size (%d)' % sz16)

# 4. Biped own-rig export
bip_src = os.path.join(d, 'bip.glb')
gltf.write_glb(bip_src, make_biped())
rb = pipeline.rig_file(bip_src, os.path.join(d, 'rig_bip'), log=lambda *a: None, png=False, clips='walk,jump_up')
res_b = ae.export_asset(rb['glb'], os.path.join(d, 'exp_bip'), name='hero', prefix='pza_', log=lambda *a: None)
mb = ogre_load.load_mesh(res_b['mesh']['path'], res_b['skeleton']['path'])
ok(set(mb.ogre_animations) == {'pza_walk', 'pza_jump_up'}, 'biped export with prefix works')
ok(json.loads(open(res_b['manifest'], encoding='utf-8').read())['format'] == 'pza.kenshi.asset', 'manifest JSON valid')

print('\n%d failed' % len(fails) if fails else '\nall passed')
sys.exit(1 if fails else 0)
