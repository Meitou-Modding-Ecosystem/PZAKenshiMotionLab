# PZA's Kenshi Motion Lab - measure the walk / run of every Kenshi skeleton that has them (humans of every race share
# male_skeleton / female_skeleton; the animals have their own) into motionlab/data/kenshi_gaits.json.
# Only measured numbers are written, never animation data. Author: PZA. GPLv3.
#   python tools/build_gait_profiles.py "<Kenshi>\data"
import glob, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from motionlab import profiles

data = sys.argv[1]
files = sorted(set(glob.glob(os.path.join(data, 'character', '**', '*.skeleton'), recursive=True) +
                   glob.glob(os.path.join(data, 'animal', '**', '*.skeleton'), recursive=True)))
out = {'note': "Walk / run numbers measured from Kenshi's own skeleton animations (tools/build_gait_profiles.py). "
               "Lengths are relative to leg length; angles in degrees; no animation data.", 'skeletons': {}, 'median': {}}
for f in files:
    key = os.path.splitext(os.path.basename(f))[0].lower()
    if key in out['skeletons']: continue
    try:
        p = profiles.profile_for_skeleton(f)
    except Exception as ex:
        print('skip %-26s %s' % (key, ex)); continue
    if p is None: continue
    out['skeletons'][key] = p
    w = p.get('walk', {}); r = p.get('run', {})
    print('%-26s %-9s walk %4.2fs duty %.2f stride %.2fL lift %.2fL knee %3.0f-%3.0f | run %4.2fs duty %.2f stride %.2fL' % (
        key, p['plan'], w.get('cycle_s', 0), w.get('duty', 0), w.get('stride', 0), w.get('lift', 0), *w.get('knee_deg', [0, 0]),
        r.get('cycle_s', 0), r.get('duty', 0), r.get('stride', 0)))
for plan in ('biped', 'quadruped'):
    ps = [p for p in out['skeletons'].values() if p['plan'] == plan]
    if not ps: continue
    med = {'plan': plan, 'source': 'median of %d skeletons' % len(ps)}
    for kind in ('walk', 'run'):
        ks = [p[kind] for p in ps if kind in p]
        if not ks: continue
        m = {}
        for k in ('cycle_s', 'duty', 'stride', 'lift', 'speed_m_s', 'leg_len_m', 'foot_centre'):
            m[k] = float(np.median([x[k] for x in ks]))
        for k in ('pelvis_drop', 'knee_deg', 'arm_deg', 'elbow_deg'):
            v = [x[k] for x in ks if k in x]
            if v: m[k] = [float(np.median([a[0] for a in v])), float(np.median([a[1] for a in v]))]
        med[kind] = m
    out['median'][plan] = med
os.makedirs(os.path.dirname(profiles.DATA), exist_ok=True)
with open(profiles.DATA, 'w', encoding='utf-8') as f: json.dump(out, f, indent=1)
print('wrote', profiles.DATA, '(%d skeletons)' % len(out['skeletons']))
