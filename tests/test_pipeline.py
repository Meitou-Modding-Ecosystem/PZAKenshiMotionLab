# PZA's Kenshi Motion Lab - end-to-end test on the procedural hexapod. Author: PZA. GPLv3.
#   python tests/test_pipeline.py        (from the repo root; exits 1 on a failure)
import os, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from motionlab import autorig, rigbuild, shape, motion, bake, gltf, check
from synthetic import make_hexapod

fails = []
def ok(cond, msg):
    print(('PASS ' if cond else 'FAIL ') + msg)
    if not cond: fails.append(msg)

src = make_hexapod(yaw_deg=25.0)            # turned on purpose: the frame step must turn it back
m, lm = autorig.analyse(src, log=lambda *a: None)
ok(abs(abs(lm['frame']['turn_about_y_deg']) - 25.0) < 4.0, 'frame: turned back %.1f deg (want 25)' % lm['frame']['turn_about_y_deg'])
sides = sorted(e['side'] for e in lm['legs'])
ok(len(lm['legs']) == 6 and sides.count('L') == 3, 'legs: %d found, sides %s' % (len(lm['legs']), ''.join(sides)))
ok(len(lm['wings']) == 2, 'wings: %d found' % len(lm['wings']))
ok(len(lm['antennae']) == 2, 'antennae: %d found' % len(lm['antennae']))
sk, rm = rigbuild.build_skeleton(lm, log=lambda *a: None)
ok(len(sk) <= rigbuild.KENSHI_MAX_BONES_PER_SUBMESH, 'bones %d <= 60' % len(sk))
J, Wk, st = rigbuild.bind(m, sk, rm, lm, shape.Surface(m), log=lambda *a: None)
ok(st['unweighted'] == 0 and J.shape[1] == 3, 'bind: 3 weights, %d unweighted' % st['unweighted'])
ok(np.allclose(Wk.sum(1), 1, atol=1e-6), 'bind: weights sum to 1')
m.skeleton, m.joints, m.weights = sk, J, Wk
P, _ = m.skin(sk.world_rest())
ok(float(np.abs(P - m.verts).max()) < 1e-6, 'rest pose skins back to the mesh')
drv = motion.Driver(motion.Rig(sk, rm))
defs = {d.name: d for d in motion.standard_clips(drv)}
for nm in ('walk', 'run', 'idle'):
    ck = check.foot_check(drv, defs[nm])
    miss = max(ck['ik_miss_m'].values(), default=0); slip = max(ck['slip_m_per_s'].values(), default=0)
    ok(miss < 0.01 and slip < 0.06 * max(defs[nm].speed, 1.0), '%s: foot IK miss %.4f m, planted slip %.4f m/s' % (nm, miss, slip))
clips = [bake.bake(drv, d) for d in defs.values()]
with tempfile.TemporaryDirectory() as td:
    p = os.path.join(td, 'hex.glb')
    gltf.write_glb(p, m, clips)
    back = gltf.load_glb(p); rc = gltf.read_clips(p)
    ok(float(np.abs(back.verts - m.verts).max()) < 1e-5, 'glb round trip: rest geometry')
    ok(sorted(rc) == sorted(defs), 'glb round trip: clips %s' % ','.join(sorted(rc)))
    ok(back.skeleton.names == sk.names, 'glb round trip: bone names')
    # a baked frame read back from the file poses like the driver did
    for tt in (0.0, 0.5):
        q, t, s, Wd = drv.pose(defs['walk'].fn(tt, defs['walk'].length))
        Wf = back.skeleton.world(bake.sample_read_clip(back.skeleton, rc['walk'], tt))
        legs = [sk.index(b) for e in rm['legs'] if e['parent'] == 'body' for b in e['bones']]
        err = float(np.abs(Wf[legs, :3, 3] - Wd[legs, :3, 3]).max())
        ok(err < 1e-3, 'walk t=%.1f: legs read back from the file match the driver (%.2e m)' % (tt, err))
    wing = sk.index(rm['wings'][0]['bones'][0])
    sc_ground = rc['walk'][sk.names[wing]]['scale'][:, 0].max(); sc_air = rc['hover'][sk.names[wing]]['scale'][:, 0].min()
    ok(sc_ground < 0.01 and sc_air > 0.99, 'wings: hidden on the ground (%.3f), full in the air (%.3f)' % (sc_ground, sc_air))
    to = rc['takeoff'][sk.names[wing]]; k = int(np.searchsorted(to['times'], 0.25))
    ok(to['scale'][0, 0] < 0.01 and to['scale'][k, 0] > 0.99, 'takeoff: wings grow 0 -> 1 in 0.25 s')
print('\n%d failed' % len(fails) if fails else '\nall passed')
sys.exit(1 if fails else 0)
