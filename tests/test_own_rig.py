# PZA's Kenshi Motion Lab - a model that already has a Kenshi-style (Bip01) rig keeps it and gets motion.
# Author: PZA. GPLv3.   python tests/test_own_rig.py
import os, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..')); sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from synthetic import make_biped
from motionlab import gltf, pipeline, existing, motion, check

fails = []
def ok(c, msg):
    print(('PASS ' if c else 'FAIL ') + msg)
    if not c: fails.append(msg)

m = make_biped()
rm = existing.rigmap_from_skeleton(m)
ok(rm is not None and rm['plan'] == 'biped', 'Bip01 rig recognised as a biped')
ok(len(rm['legs']) == 2 and len(rm['arms']) == 2, 'legs %d, arms %d' % (len(rm['legs']), len(rm['arms'])))
ok(all(e.get('foot') for e in rm['legs']), 'feet kept level: %s' % [e.get('foot') for e in rm['legs']])
drv = motion.Driver(motion.Rig(m.skeleton, rm))
names = [c.name for c in motion.standard_clips(drv)]
ok(names == ['idle', 'walk', 'run'], 'clips for a person: %s' % names)
for cd in motion.standard_clips(drv):
    ck = check.foot_check(drv, cd)
    miss = max(ck['ik_miss_m'].values(), default=0); slip = max(ck['slip_m_per_s'].values(), default=0)
    ok(miss < 0.01 and slip < 0.05, '%s: foot miss %.4f m, slip %.4f m/s' % (cd.name, miss, slip))
# through the file pipeline: the rig and skin come out unchanged
d = tempfile.mkdtemp(); src = os.path.join(d, 'biped.glb')
gltf.write_glb(src, m)
r = pipeline.rig_file(src, d, log=lambda *a: None, png=False)
back = gltf.load_glb(r['glb'])
ok(r['own_rig'] and r['plan'] == 'biped', 'pipeline kept the own rig')
ok(back.skeleton.names == m.skeleton.names, 'bones unchanged (%d)' % len(back.skeleton))
ok(float(np.abs(back.verts - m.verts).max()) < 1e-5, 'geometry unchanged')
ok(sorted(gltf.read_clips(r['glb'])) == ['idle', 'run', 'walk'], 'clips in the file')
# the walk moves the legs and swings the arms
rc = gltf.read_clips(r['glb'])['walk']
ok('Bip01 L UpperArm' in rc and np.ptp(rc['Bip01 L UpperArm']['rotation'], axis=0).max() > 0.05, 'arms swing in the walk')
# a model the Motion Lab rigged earlier can be re-animated: its bone ends travel in the .glb, its rig is kept
from synthetic import make_hexapod
d2 = tempfile.mkdtemp(); src2 = os.path.join(d2, 'hex.glb')
gltf.write_glb(src2, make_hexapod(10.0), skeleton=None)
r1 = pipeline.rig_file(src2, d2, log=lambda *a: None, png=False, clips='walk')
r2 = pipeline.rig_file(r1['glb'], os.path.join(d2, 'again'), log=lambda *a: None, png=False, clips='walk,idle', name='again')
ok(r2['own_rig'] and r2['bones'] == r1['bones'], 're-animating a Motion Lab rig keeps it (%d bones)' % r2['bones'])
first = gltf.load_glb(r1['glb']); rm2 = existing.rigmap_from_skeleton(first)
ok(all(np.allclose(e['tip'], first.skeleton.tails[e['bones'][-1]]) for e in rm2['legs']), 'leg tips come from the stored bone ends')
drv2 = motion.Driver(motion.Rig(first.skeleton, rm2))
ck = check.foot_check(drv2, [c for c in motion.standard_clips(drv2) if c.name == 'walk'][0])
miss2 = max(ck['ik_miss_m'].values(), default=0)
ok(miss2 < 0.01, 're-animated walk keeps its feet (miss %.4f m)' % miss2)
print('\n%d failed' % len(fails) if fails else '\nall passed')
sys.exit(1 if fails else 0)
