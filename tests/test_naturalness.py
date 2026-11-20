# PZA's Kenshi Motion Lab - naturalness gates (2026-11-19): the "machine" look as numbers. A person's walk / run must
# turn the pelvis, counter-turn the chest, roll the feet heel -> toe and keep the hip height free of corners, measured
# the same way the Studio does (motionlab/metrics.py) against the numbers measured from Kenshi's own clips
# (kenshi_gaits.json 'scorecard'). Runs on the procedural test body, so no game files are needed; it covers the
# summary path (no curves file) and, with made-up curves, the measured-curve path. Author: PZA. GPLv3.
#   python tests/test_naturalness.py
import copy, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..')); sys.path.insert(0, os.path.dirname(__file__))
os.environ['MOTIONLAB_NO_CURVES'] = '1'                 # the summary path first (no curves file)
import numpy as np
from synthetic import make_biped
from motionlab import existing, motion, metrics, studio, check, profiles

fails = []
def ok(c, msg):
    print(('PASS ' if c else 'FAIL ') + msg)
    if not c: fails.append(msg)

GATED = ('pelvis yaw range (deg)', 'chest counter-turn range (deg)', 'foot roll range (deg)', 'pelvis bob (cm)',
         'hip-height corners (%)', 'pelvis jerk (m/s3)')


def card(rm):
    drv = motion.Driver(motion.Rig(m.skeleton, rm)); R = metrics.roles(drv.rig.names, rm)
    out = {}
    for cd in motion.standard_clips(drv):
        if cd.name not in ('walk', 'run'): continue
        F = studio.our_frames(drv, cd)
        out[cd.name], _ = metrics.cycle(F, drv.rig.Wrest, R, studio.FPS)
        ck = check.foot_check(drv, cd, fps=60)
        out[cd.name]['_slip'] = max(ck['slip_m_per_s'].values(), default=0.0)
        out[cd.name]['_miss'] = max(ck['ik_miss_m'].values(), default=0.0)
    return out


m = make_biped()
rm = existing.rigmap_from_skeleton(m)
prof = rm.get('gait_profile') or {}
ok('scorecard' in prof.get('walk', {}), 'the test body has Kenshi reference numbers (%s)' % rm.get('gait_profile_from'))
pub = card(rm)
for g in ('walk', 'run'):
    ref = prof[g]['scorecard']
    for k in GATED:
        gr = metrics.grade(k, pub[g][k], ref.get(k))
        ok(gr != 'red', 'public %s: %s %.1f against Kenshi %.1f (%s)' % (g, k, pub[g][k], ref.get(k, float('nan')), gr))
    ok(pub[g]['_slip'] < 0.05 and pub[g]['_miss'] < 0.01, 'public %s: planted feet hold (slip %.3f m/s, miss %.3f m)' % (g, pub[g]['_slip'], pub[g]['_miss']))
# the old machine look must stay gone: never a pelvis that does not turn, or feet that do not roll
for g in ('walk', 'run'):
    ok(pub[g]['pelvis yaw range (deg)'] > 5 and pub[g]['foot roll range (deg)'] > 40,
       'public %s: the pelvis turns (%.0f deg) and the feet roll (%.0f deg)' % (g, pub[g]['pelvis yaw range (deg)'], pub[g]['foot roll range (deg)']))

# the measured-curve path, with made-up curves of the right shape (the curves file adds these to the profile)
src = open(os.path.join(os.path.dirname(__file__), 'fuzz', 'fuzz_curves.py'), encoding='utf-8').read()
ns = {'np': np}; exec(src[src.index('def made_up'):src.index('good = {')], ns)
rm2 = copy.deepcopy(rm)
for g in ('walk', 'run'):
    for k, v in ns['made_up'](g).items():
        ok(profiles.valid_curve(k, v), 'made-up %s %s passes the curves check' % (g, k))
        rm2['gait_profile'][g][k] = v
cur = card(rm2)
for g in ('walk', 'run'):
    c = cur[g]
    ok(c['_slip'] < 0.05 and c['_miss'] < 0.01, 'curves %s: planted feet hold on the measured path (slip %.3f m/s, miss %.3f m)' % (g, c['_slip'], c['_miss']))
    ok(c['pelvis yaw range (deg)'] > 5 and c['foot roll range (deg)'] > 40 and c['hip-height corners (%)'] < 8,
       'curves %s: pelvis turns %.0f deg, feet roll %.0f deg, hip corners %.1f%%' % (g, c['pelvis yaw range (deg)'], c['foot roll range (deg)'], c['hip-height corners (%)']))
print('\n%d failed' % len(fails) if fails else '\nall passed')
sys.exit(1 if fails else 0)
