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
MOVES = ['knockdown_back', 'knockdown_front', 'getup_back', 'getup_front', 'roll_forward', 'roll_dive',
         'jump_up', 'jump_forward', 'jump_down', 'land_soft', 'land_hard']
ok(names == ['idle', 'walk', 'run'] + MOVES, 'clips for a person: %s' % names)
for cd in motion.standard_clips(drv):
    if cd.name in MOVES: continue
    ck = check.foot_check(drv, cd)
    miss = max(ck['ik_miss_m'].values(), default=0); slip = max(ck['slip_m_per_s'].values(), default=0)
    ok(miss < 0.01 and slip < 0.05, '%s: foot miss %.4f m, slip %.4f m/s' % (cd.name, miss, slip))
# knockdowns and get-ups: start standing / lying, end lying / standing, nothing sinks under the floor, hands reach
for cd in motion.standard_clips(drv):
    if cd.name not in MOVES: continue
    mc = check.move_check(drv, cd)
    if cd.name.startswith(('jump', 'land')):           # end standing; a jump starts standing, a ledge / fall starts high
        start_ok = abs(mc['pelvis_start'] - 1) < 0.03 if cd.name in ('jump_up', 'jump_forward') else mc['pelvis_start'] > 1.5
        ok(start_ok and abs(mc['pelvis_end'] - 1) < 0.03 and mc['pelvis_min'] < 0.8,
           '%s: pelvis %.2f -> lowest %.2f -> %.2f of standing height' % (cd.name, mc['pelvis_start'], mc['pelvis_min'], mc['pelvis_end']))
    elif cd.name.startswith('roll'):                     # stand -> roll over the back (pelvis low) -> stand again
        ok(abs(mc['pelvis_start'] - 1) < 0.03 and abs(mc['pelvis_end'] - 1) < 0.03 and mc['pelvis_min'] < 0.3,
           '%s: pelvis %.2f -> lowest %.2f -> %.2f of standing height' % (cd.name, mc['pelvis_start'], mc['pelvis_min'], mc['pelvis_end']))
    else:
        down = cd.name.startswith('knockdown')
        a, b = (mc['pelvis_start'], mc['pelvis_end']) if down else (mc['pelvis_end'], mc['pelvis_start'])
        ok(abs(a - 1) < 0.03 and b < 0.3, '%s: pelvis %.2f -> %.2f of standing height' % (cd.name, mc['pelvis_start'], mc['pelvis_end']))
    ok(mc['lowest_joint'] > -0.03, '%s: lowest joint %.3f of standing height' % (cd.name, mc['lowest_joint']))
    ok(mc['hand_miss_m'] < 0.05, '%s: planted hand miss %.3f m (%d let go out of reach, %d knees set on the floor, %d body lifts)'
       % (cd.name, mc['hand_miss_m'], mc['hand_releases'], mc['knee_floor_hits'], mc['body_floor_lifts']))
    ok(not cd.loop and 1.0 < cd.length < 5.0, '%s: one-shot, %.2f s' % (cd.name, cd.length))
# jumps really leave the ground (both feet up at the top of the flight) and the long jump lands where it says
rig = drv.rig; feet_b = [rig.b('Bip01 L Foot'), rig.b('Bip01 R Foot')]; root_b = rig.b('Bip01')
for name in ('jump_up', 'jump_forward'):
    cd = [c for c in motion.standard_clips(drv) if c.name == name][0]; drv.reset()
    lift = []
    for k in range(int(cd.length * 30) + 1):
        q, t, s, W = drv.pose(cd.fn(k / 30, cd.length)); lift.append(min(W[b][1, 3] for b in feet_b))
    rest_ankle = min(rig.Wrest[b][1, 3] for b in feet_b)
    ok(max(lift) - rest_ankle > 0.2, '%s: both feet leave the ground (%.2f m at the top)' % (name, max(lift) - rest_ankle))
    q, t, s, W = drv.pose(cd.fn(cd.length, cd.length)); z = W[root_b][2, 3] - rig.Wrest[root_b][2, 3]
    ok(abs(z - (1.8 if name == 'jump_forward' else 0.0) * rig.Wrest[root_b][1, 3]) < 0.05, '%s: lands %.2f m forward' % (name, z))
# through the file pipeline: the rig and skin come out unchanged
d = tempfile.mkdtemp(); src = os.path.join(d, 'biped.glb')
gltf.write_glb(src, m)
r = pipeline.rig_file(src, d, log=lambda *a: None, png=False)
back = gltf.load_glb(r['glb'])
ok(r['own_rig'] and r['plan'] == 'biped', 'pipeline kept the own rig')
ok(back.skeleton.names == m.skeleton.names, 'bones unchanged (%d)' % len(back.skeleton))
ok(float(np.abs(back.verts - m.verts).max()) < 1e-5, 'geometry unchanged')
ok(sorted(gltf.read_clips(r['glb'])) == sorted(['idle', 'run', 'walk'] + MOVES), 'clips in the file')
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
