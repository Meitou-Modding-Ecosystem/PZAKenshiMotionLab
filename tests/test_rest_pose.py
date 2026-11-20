# PZA's Kenshi Motion Lab - bodies whose rest pose is not Kenshi's: arms in a T-pose, feet a little lopsided.
# Author: PZA. GPLv3.   python tests/test_rest_pose.py
#   1. Arm motion is a turn away from Kenshi's rest, where the arms hang. A T-pose body (Meshy / Mixamo exports) got
#      the same turn from arms out and threw its arms over its head in the run: its arms are lowered first.
#   2. Whether feet roll is decided per body: a lopsided model rolled one foot and dragged the other (1.2 m/s slip).
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..')); sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from synthetic import make_biped
from motionlab import existing, motion, check

fails = []
def ok(c, msg):
    print(('PASS ' if c else 'FAIL ') + msg)
    if not c: fails.append(msg)

def driver(m):
    rm = existing.rigmap_from_skeleton(m); rig = motion.Rig(m.skeleton, rm)
    return rig, motion.Driver(rig), {d.name: d for d in motion.standard_clips(motion.Driver(rig))}

def hands(rig, drv, cd, n=40):
    """Per frame: each hand's height above its shoulder, and each hand's place relative to the root."""
    drv.reset(); out = []
    root = rig.b(drv.map['root'])
    for k in range(n):
        q, t, s, W = drv.pose(cd.fn(k / n * cd.length, cd.length))
        row = []
        for e, ch in drv.arms:
            sh = W[ch[0]][:3, 3]; hd = rig.chain_points(ch, W)[-1]
            row.append((hd[1] - sh[1], hd - W[root][:3, 3]))
        out.append(row)
    return out

# 1. arms
rig0, drv0, defs0 = driver(make_biped())
ok(rig0.arms_lowered == [], 'hanging arms are left alone: %s' % rig0.arms_lowered)
rigT, drvT, defsT = driver(make_biped(tpose=True))
ok(len(rigT.arms_lowered) == 2 and all(d > 80 for b, d in rigT.arms_lowered), 'T-pose arms lowered: %s' % rigT.arms_lowered)
for c in ('idle', 'walk', 'run', 'jump_run'):
    hT = hands(rigT, drvT, defsT[c]); h0 = hands(rig0, drv0, defs0[c])
    top = max(r[0] for row in hT for r in row)
    # the T-pose body's hands follow the hanging body's (T-pose arms are the same length: 0.3 + 0.28 + 0.12)
    diff = max(float(np.linalg.norm(a[1] - b[1])) for ra, rb in zip(hT, h0) for a, b in zip(ra, rb))
    ok(top < 0.15, '%s: T-pose hands never go over the head (highest %+.2f m above the shoulder)' % (c, top))
    ok(diff < 0.08, '%s: T-pose hands follow the hanging-arm body within %.3f m' % (c, diff))
# the file's bind pose is not touched: the skeleton the model is skinned to still holds the T-pose
mT = make_biped(tpose=True); W = mT.skeleton.world_rest(); ix = {n: i for i, n in enumerate(mT.skeleton.names)}
ok(abs(W[ix['Bip01 L Forearm']][1, 3] - 1.5) < 1e-6, 'bind pose unchanged (forearm still at shoulder height)')

# 2. lopsided feet like the spike-footed body that showed it: left toe 4.0 cm ahead of its ankle, right toe 2.8 cm,
#    either side of the cut-off (2% of the 1.65 m height = 3.3 cm). (Very unequal feet, 15 cm against 2 cm, still slide
#    in the run: the stride is corrected for one foot's roll. Not handled yet.)
rigL, drvL, defsL = driver(make_biped(toe_l=0.04, toe_r=0.028))
ok(set(drvL.pivots) == {e['name'] for e, c in drvL.legs}, 'both feet roll: %s' % sorted(drvL.pivots))
ok(drvL.feet_unrolled == 1, 'rule counted the foot that rolls because its pair does: %d' % drvL.feet_unrolled)
for c in ('walk', 'run'):
    ck = check.foot_check(drvL, defsL[c]); slip = max(ck['slip_m_per_s'].values(), default=0)
    ok(slip < 0.1, '%s on lopsided feet: worst planted slip %.3f m/s' % (c, slip))
# a body with no forward toe at all (stumps) rolls neither foot
rigS, drvS, _ = driver(make_biped(toe_r=-0.01))
ok(len(drvS.pivots) == 1, 'a backward toe never rolls (only the normal left foot does): %s' % sorted(drvS.pivots))

print('rest-pose tests: %s' % ('PASS' if not fails else 'FAIL (%d)' % len(fails)))
sys.exit(1 if fails else 0)
