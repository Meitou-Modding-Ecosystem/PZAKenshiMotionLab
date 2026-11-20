# PZA's Kenshi Motion Lab - animals and creatures, the faults the 2026-11-19 benchmark found (no game files: synthetic
# bodies built like Kenshi's): a four-legged body whose front legs hang off the NECK and are shorter than its hind legs
# (like Kenshi's dog), and a six-legged flyer with a leg that rides along with the body.
# Author: PZA. GPLv3.   python tests/test_animals.py
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..')); sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from synthetic import make_quadruped, make_hexapod
from motionlab import existing, motion, check, autorig, rigbuild

fails = []
def ok(c, msg):
    print(('PASS ' if c else 'FAIL ') + msg)
    if not c: fails.append(msg)


def worst(drv, cd, fps=60):
    """Worst foot speed change and worst hip speed change from one 1/60 s frame to the next."""
    r = drv.rig; root = r.b(drv.map['root']); drv.reset(); P, Y = [], []
    for k in range(int(cd.length * fps) + 1):
        t = (k / fps) % cd.length if cd.loop else min(k / fps, cd.length)
        q, tt, s, W = drv.pose(cd.fn(t, cd.length), dt=1 / fps)
        P.append([r.chain_points(ch, W)[-1] for e, ch in drv.legs]); Y.append(W[root][1, 3])
    P = np.array(P); Y = np.array(Y)
    return (float(np.linalg.norm(np.diff(np.diff(P, axis=0) * fps, axis=0), axis=2).max()),
            float(np.abs(np.diff(np.diff(Y) * fps)).max()))


# --- four legs, front legs under the neck
m = make_quadruped(); rm = existing.rigmap_from_skeleton(m)
drv = motion.Driver(motion.Rig(m.skeleton, rm))
ok(rm['plan'] == 'quadruped' and len(drv.legs) == 4, 'four-legged body recognised (%s, %d legs)' % (rm['plan'], len(drv.legs)))
ok(drv.neck_carries_legs, 'its neck carries the front legs (the head turn then goes on the head bone only)')
fl = sorted(drv.leg_lens.values())
ok(fl[0] < 0.85 * fl[-1], 'front legs shorter than hind legs (%.2f vs %.2f m)' % (fl[0], fl[-1]))
for cd in motion.standard_clips(drv):
    ck = check.foot_check(drv, cd, fps=60)
    slip = max(ck['slip_m_per_s'].values(), default=0.0); miss = max(ck['ik_miss_m'].values(), default=0.0)
    ok(slip < 0.05 and miss < 0.01, 'quadruped %s: planted feet hold (slip %.3f m/s, miss %.3f m)' % (cd.name, slip, miss))
    foot, hip = worst(drv, cd)
    ok(foot < 8.5, 'quadruped %s: no foot snaps (worst %.1f m/s in one frame)' % (cd.name, foot))
    # a bounding run turns over sharply at the top of each leap: Kenshi's own dog run does 3.0 m/s per frame
    lim = 3.5 if cd.name == 'run' else 1.0
    ok(hip < lim, 'quadruped %s: hips change speed by at most %.1f m/s per frame (worst %.2f)' % (cd.name, lim, hip))

# each leg swings at most 0.45 of ITS OWN length above its planted height (one lift for all legs aimed the dog's short
# front paws at its shoulders, out of reach: 24-30 m/s snaps in its run)
run = [c for c in motion.standard_clips(drv) if c.name == 'run'][0]
hi = {e['name']: 0.0 for e, c in drv.legs}
for k in range(120):
    f = run.fn(k / 120 * run.length, run.length)['feet']
    for e, c in drv.legs:
        if e['name'] in f and f[e['name']] is not None: hi[e['name']] = max(hi[e['name']], float(f[e['name']][1] - drv.rest_tip[e['name']][1]))
for e, c in drv.legs:
    nm = e['name']; ok(hi[nm] <= 0.45 * drv.leg_lens[nm] + 0.005,
                       'quadruped run: %s swings at most 0.45 of its own length (%.3f of %.3f m)' % (nm, hi[nm], drv.leg_lens[nm]))

# --- six legs and wings, one leg riding along with the body (lifted): it must tuck as gradually as the others
src = make_hexapod()
mh, lm = autorig.analyse(src, log=lambda *a: None)
sk, rmh = rigbuild.build_skeleton(lm, log=lambda *a: None)
lifted = sorted(e['name'] for e in rmh['legs'])[-1]
for e in rmh['legs']:
    if e['name'] == lifted: e['lifted'] = True
dh = motion.Driver(motion.Rig(sk, rmh))
ok(bool(dh.wings), 'the test flyer has wings (%d)' % len(dh.wings))
for cd in motion.standard_clips(dh):
    if cd.name not in ('takeoff', 'land', 'hover'): continue
    foot, hip = worst(dh, cd)
    ok(foot < 8.5, 'flyer %s with a lifted leg (%s): no leg snaps as the legs tuck (worst %.1f m/s in one frame)' % (cd.name, lifted, foot))
print('\n%d failed' % len(fails) if fails else '\nall passed')
sys.exit(1 if fails else 0)
