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
print('\n%d failed' % len(fails) if fails else '\nall passed')
sys.exit(1 if fails else 0)
