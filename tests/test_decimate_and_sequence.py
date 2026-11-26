# PZA's Kenshi Motion Lab - auto-decimation and clip sequencing tests. Author: PZA. GPLv3.
#   python tests/test_decimate_and_sequence.py     (no game files, no Pillow required)
import os, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from synthetic import make_hexapod, make_biped
from motionlab import decimate, existing, gltf, motion, pipeline, sequence

fails = []
def ok(c, msg):
    print(('PASS ' if c else 'FAIL ') + msg)
    if not c:
        fails.append(msg)


d = tempfile.mkdtemp()

# 1. Decimation: reduce synthetic hexapod (~3,500 tris) to 1,200 tris ('weapon' budget)
hex_m = make_hexapod()
orig_tris = len(hex_m.tris)
ok(orig_tris > 2000, 'hexapod starts with %d tris' % orig_tris)
target = decimate.resolve_target(budget='weapon')
ok(target == 1200, 'weapon budget resolves to 1200')
dec_m = decimate.decimate_model(hex_m, target, log=lambda *a: None)
ok(len(dec_m.tris) <= target and len(dec_m.tris) >= target // 2,
   'decimated to %d tris (target %d)' % (len(dec_m.tris), target))
ok(np.isfinite(dec_m.verts).all() and np.isfinite(dec_m.normals).all(), 'decimated verts and normals finite')
# Bounding box stays close to original
lo0, hi0 = hex_m.bounds()
lo1, hi1 = dec_m.bounds()
box_err = float(max(np.abs(lo1 - lo0).max(), np.abs(hi1 - hi0).max()))
ok(box_err < 0.08, 'bounding box preserved after decimation (max diff %.4f m)' % box_err)

# Decimate integrated into pipeline.rig_file via target_tris
src = os.path.join(d, 'hex.glb')
gltf.write_glb(src, hex_m)
r = pipeline.rig_file(src, os.path.join(d, 'rig_dec'), target_tris=1800, log=lambda *a: None, png=False, clips='idle,walk')
back = gltf.load_glb(r['glb'])
ok(len(back.tris) <= 1800, 'pipeline rig_file with target_tris=1800 wrote %d tris' % len(back.tris))
ok(back.skeleton is not None and len(back.skeleton) <= 60, 'decimated model still auto-rigs cleanly (%d bones)' % len(back.skeleton))

# 2. Clip sequencing with inertialized joins on a biped
bip = make_biped()
rm = existing.rigmap_from_skeleton(bip)
drv = motion.Driver(motion.Rig(bip.skeleton, rm))
steps = sequence.parse_steps('walk:1.4,run:0.8:0.25,jump_run::0.1,roll_dive::0.2')
ok(len(steps) == 4 and steps[0]['duration'] == 1.4 and steps[1]['blend'] == 0.25, 'parse_steps works')

clip, meta = sequence.build_sequence(drv, steps, fps=30, name='combo', travel=True)
ok(meta['frames'] == len(clip.times) and len(meta['joins']) == 3,
   'sequence built %d frames (%.2f s) with %d inertialized joins' % (meta['frames'], meta['length_s'], len(meta['joins'])))
# Root travels continuously forward without a backward snap at joins
root_i = drv.rig.b(rm['root'])
rt = clip.tracks[root_i]['t']
rz = rt[:, 2]
ok(float(rz[-1] - rz[0]) > 2.0, 'sequence root travels forward (%.2f m)' % float(rz[-1] - rz[0]))
# Check smoothness AT each join: inertialization removes any vertical snap across cuts
ry = rt[:, 1]
join_dy = max(abs(float(ry[j] - ry[j - 1])) for j in meta['joins'])
ok(join_dy < 0.05, 'no vertical snap at joins (max join dy %.4f m)' % join_dy)

# Round-trip the sequence clip through .glb
seq_glb = os.path.join(d, 'combo.glb')
gltf.write_glb(seq_glb, bip, [clip])
rc = gltf.read_clips(seq_glb)
ok('combo' in rc, 'sequence clip round-trips through .glb')

print('\n%d failed' % len(fails) if fails else '\nall passed')
sys.exit(1 if fails else 0)
