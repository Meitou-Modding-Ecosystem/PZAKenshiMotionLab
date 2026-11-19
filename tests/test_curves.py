# PZA's Kenshi Motion Lab - retargeting from a reference clip's curves, with no outside files: one of our own moves
# is baked onto a test body whose bones carry Mixamo-style names (the "reference"), then retargeted back onto the
# same body with its Kenshi (Bip01) names. The round trip must land where it started.
# Author: PZA. GPLv3.   python tests/test_curves.py
import copy, os, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..')); sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from synthetic import make_biped
from motionlab import gltf, existing, motion, bake, curves, check

fails = []
def ok(c, msg):
    print(('PASS ' if c else 'FAIL ') + msg)
    if not c: fails.append(msg)

TO_MIXAMO = {'Bip01 Pelvis': 'Hips', 'Bip01 Spine': 'Spine', 'Bip01 Spine1': 'Spine1', 'Bip01 Spine2': 'Spine2',
             'Bip01 Neck': 'Neck', 'Bip01 Head': 'Head'}
for s, S in (('L', 'Left'), ('R', 'Right')):
    TO_MIXAMO.update({'Bip01 %s Thigh' % s: S + 'UpLeg', 'Bip01 %s Calf' % s: S + 'Leg', 'Bip01 %s Foot' % s: S + 'Foot',
                      'Bip01 %s Toe0' % s: S + 'ToeBase', 'Bip01 %s Clavicle' % s: S + 'Shoulder',
                      'Bip01 %s UpperArm' % s: S + 'Arm', 'Bip01 %s Forearm' % s: S + 'ForeArm', 'Bip01 %s Hand' % s: S + 'Hand'})

m = make_biped()
drv = motion.Driver(motion.Rig(m.skeleton, existing.rigmap_from_skeleton(m)))
d = tempfile.mkdtemp()
src = {c.name: c for c in motion.standard_clips(drv)}
for name in ('getup_front', 'roll_forward', 'knockdown_back'):
    cd = src[name]
    clip = bake.bake(drv, cd, fps=30)
    ref = copy.deepcopy(m); ref.skeleton = copy.deepcopy(m.skeleton)
    ref.skeleton.names = ['mixamorig:' + TO_MIXAMO.get(n, n.replace(' ', '_')) for n in m.skeleton.names]
    path = os.path.join(d, name + '.glb')
    gltf.write_glb(path, ref, [clip])
    cur = curves.extract(path, fps=30)
    rd = curves.curve_clip(drv, cur, 'ref_' + name)
    ok(abs(rd.length - cd.length) < 0.05, '%s: length %.2f s (made %.2f s)' % (name, rd.length, cd.length))
    # compare the bodies frame by frame: pelvis, feet, hands, head
    r = drv.rig
    def pts(cdef):
        drv.reset(); out = []
        for k in range(int(cdef.length * 30)):
            q, t, s, W = drv.pose(cdef.fn(k / 30, cdef.length), dt=1 / 30)
            out.append(np.array([W[r.b(b)][:3, 3] for b in ('Bip01', 'Bip01 L Foot', 'Bip01 R Foot', 'Bip01 L Hand', 'Bip01 R Hand', 'Bip01 Head')]))
        return np.array(out)
    a, b = pts(cd), pts(rd); n = min(len(a), len(b))
    err = np.linalg.norm(a[:n] - b[:n], axis=2).max(axis=0)
    ok(err[0] < 0.04, '%s: pelvis follows the reference (worst %.3f m)' % (name, err[0]))
    ok(max(err[1:3]) < 0.04, '%s: feet follow (worst %.3f m)' % (name, max(err[1:3])))
    ok(max(err[3:5]) < 0.08, '%s: hands follow (worst %.3f m)' % (name, max(err[3:5])))
    ok(err[5] < 0.06, '%s: head follows (worst %.3f m)' % (name, err[5]))
    mc = check.move_check(drv, rd)
    ok(mc['lowest_joint'] > -0.03, '%s: retargeted, lowest joint %.3f of standing height' % (name, mc['lowest_joint']))
# a reference that lacks bones is refused with a clear message
bad = copy.deepcopy(m); bad.skeleton = copy.deepcopy(m.skeleton); bad.skeleton.names = ['x%d' % i for i in range(len(m.skeleton))]
p = os.path.join(d, 'bad.glb'); gltf.write_glb(p, bad, [bake.bake(drv, src['idle'], fps=30)])
try:
    curves.extract(p); ok(False, 'a reference with unknown bones is refused')
except ValueError as e:
    ok('no bone for' in str(e), 'a reference with unknown bones is refused (%s...)' % str(e)[:40])
print('\n%d failed' % len(fails) if fails else '\nall passed')
sys.exit(1 if fails else 0)
