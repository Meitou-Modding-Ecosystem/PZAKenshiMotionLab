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
        # curves: element-wise medians over PEOPLE only (other two-legged skeletons move differently)
        kp = [p[kind] for p in ps if kind in p and p['source'].lower().endswith('_skeleton.skeleton')] or ks
        tw = [x['turns'] for x in kp if 'turns' in x]
        if tw: m['turns'] = {pt: {a: [float(v) for v in np.median([x[pt][a] for x in tw], axis=0)] for a in tw[0][pt]} for pt in tw[0]}
        am = [x['arms'] for x in kp if 'arms' in x]
        if am: m['arms'] = {pt: {a: [float(v) for v in np.median([x[pt][a] for x in am if pt in x], axis=0)] for a in am[0][pt]} for pt in am[0]}
        v = [x['ankle_sink'] for x in kp if 'ankle_sink' in x]
        if v: m['ankle_sink'] = float(np.median(v))
        ap = [x['ankle_path'] for x in kp if 'ankle_path' in x]
        if ap: m['ankle_path'] = {k: [float(v) for v in np.median([x[k] for x in ap], axis=0)] for k in ap[0]}
        v = [x['ankle_z'] for x in kp if 'ankle_z' in x]
        if v: m['ankle_z'] = [float(np.median([a[0] for a in v])), float(np.median([a[1] for a in v]))]
        py = [x['pelvis_y'] for x in kp if 'pelvis_y' in x]
        if py: m['pelvis_y'] = [float(v) for v in np.median(py, axis=0)]
        sc = [x['scorecard'] for x in kp if 'scorecard' in x]
        if sc: m['scorecard'] = {k: float(np.median([x[k] for x in sc if k in x])) for k in sc[0]}
        fr = [x['foot_roll'] for x in kp if 'foot_roll' in x]
        if fr: m['foot_roll'] = {k: (float(np.median([x[k] for x in fr])) if k == 'duty' else
                                     [float(v) for v in np.median([x[k] for x in fr], axis=0)]) for k in fr[0]}
        med[kind] = m
    out['median'][plan] = med
# curves (how the pelvis / chest / head / arms turn, the feet roll, the ankle's path) go to their own file,
# motionlab/data/kenshi_curves.json; kenshi_gaits.json keeps the summary numbers (and lossy summaries of the curves,
# which the gait falls back on when the curves file is missing)
curves = {'note': "Curves measured from Kenshi's walk / run clips (tools/build_gait_profiles.py).", 'skeletons': {}, 'median': {}}
for grp in ('skeletons', 'median'):
    for key, p in out[grp].items():
        for kind in ('walk', 'run'):
            if kind not in p: continue
            profiles.add_summaries(p[kind])            # the public file keeps lossy summaries the fallback gait uses
            c = {k: p[kind].pop(k) for k in profiles.CURVE_KEYS if k in p[kind]}
            if c: curves[grp].setdefault(key, {})[kind] = c
os.makedirs(os.path.dirname(profiles.DATA), exist_ok=True)
with open(profiles.DATA, 'w', encoding='utf-8') as f: json.dump(out, f, indent=1)
with open(profiles.CURVES, 'w', encoding='utf-8') as f: json.dump(curves, f, indent=1)
print('wrote', profiles.DATA, '(%d skeletons, summary numbers) and the curves' % len(out['skeletons']), profiles.CURVES)
