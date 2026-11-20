# PZA's Kenshi Motion Lab - curves-file fuzzer: a damaged motionlab/data/kenshi_curves.json must never crash the driver or put a broken number into a pose: bad entries are
# dropped and that gait falls back to the summaries. Uses made-up curves (no game data). Author: PZA. GPLv3.
#   python tests/fuzz/fuzz_curves.py [iterations]
import copy, json, os, random, sys, tempfile, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from motionlab import profiles, existing, motion
from synthetic import make_biped

N = int(sys.argv[1]) if len(sys.argv) > 1 else 150
rng = random.Random(7151)


def made_up(kind):
    """Plausible curves with the exact shapes the driver reads (not measured from anything)."""
    f = lambda a, ph: [0.0, a * np.cos(ph), a * np.sin(ph)] + [0.0] * 6
    run = kind == 'run'
    turns = {pt: {'pitch': f(1.0, 0.3), 'yaw': f(20.0 if run else 6.0, 1.0 + i), 'roll': f(4.0, 0.5)}
             for i, pt in enumerate(('pel', 'chest', 'head'))}
    arms = {sd + '_' + k: {'pitch': f(15.0, 0.2 if sd == 'L' else 3.3), 'yaw': f(3.0, 0.0), 'roll': f(2.0, 0.0)}
            for sd in 'LR' for k in ('clav', 'upper', 'fore', 'hand')}
    u = np.linspace(0, 1, 9); v = np.linspace(0, 1, 17)
    roll = {'duty': 0.25 if run else 0.55, 'stance': list(np.interp(u, [0, .3, 1], [-15, 0, 30])),
            'swing': list(np.interp(u, [0, .4, 1], [30, 60, -15])), 'toe_stance': [0.0] * 9, 'toe_swing': [0.0] * 9}
    path = {'stance_z': list(0.35 - 0.8 * v), 'stance_y': [0.0] * 17,
            'swing_z': list(-0.45 + 0.8 * v), 'swing_y': list(0.25 * np.sin(np.pi * v))}
    return {'turns': turns, 'arms': arms, 'foot_roll': roll, 'pelvis_y': [-0.10, 0, 0, 0.015, 0, 0, 0, 0, 0], 'ankle_path': path}


good = {'note': 'fuzz', 'skeletons': {}, 'median': {'biped': {k: made_up(k) for k in ('walk', 'run')}}}
for k in ('walk', 'run'):                                   # the made-up curves themselves must pass the validator
    for key, val in good['median']['biped'][k].items():
        assert profiles.valid_curve(key, val), 'made-up %s %s rejected' % (k, key)
td = tempfile.mkdtemp(); path = os.path.join(td, 'kenshi_curves.json')
m = make_biped()
BAD = [float('nan'), float('inf'), -float('inf'), 1e308, -1e9, 'x', None, True, [], {}, [1.0] * 3, 0]


def mutate(d):
    """A random structural / value change somewhere inside the curves."""
    d = copy.deepcopy(d); node = d['median']['biped'][rng.choice(['walk', 'run'])]
    trail = []
    for _ in range(rng.randint(1, 4)):
        if isinstance(node, dict) and node:
            k = rng.choice(list(node)); trail.append((node, k)); node = node[k]
        elif isinstance(node, list) and node:
            k = rng.randrange(len(node)); trail.append((node, k)); node = node[k]
        else: break
    if not trail: return d
    par, k = trail[-1]
    op = rng.randrange(4)
    if op == 0: par[k] = rng.choice(BAD)
    elif op == 1 and isinstance(par, dict): del par[k]
    elif op == 2 and isinstance(par[k], list) and par[k]: par[k] = par[k][:rng.randrange(len(par[k]))]
    else: par[k] = rng.uniform(-1e4, 1e4) if not isinstance(par[k], (dict, list)) else rng.choice(BAD)
    return d


def run_once(text):
    with open(path, 'w', encoding='utf-8') as f: f.write(text)
    profiles.CURVES = path; profiles._cache = None
    rm = existing.rigmap_from_skeleton(m); drv = motion.Driver(motion.Rig(m.skeleton, rm))
    clips = {c.name: c for c in motion.standard_clips(drv)}
    for nm in ('walk', 'run'):
        cd = clips[nm]
        for k in range(6):
            q, t, s, W = drv.pose(cd.fn(k / 6 * cd.length, cd.length), dt=1 / 30)
            if not np.isfinite(W).all(): return 'non-finite pose in %s' % nm
    return None


stats = {'ok': 0}; worst = 0.0
err = run_once(json.dumps(good))
if err: print('FAIL: the unmutated curves: %s' % err); sys.exit(1)
for it in range(N):
    d = mutate(good)
    try: text = json.dumps(d, allow_nan=True)
    except (TypeError, ValueError): continue
    if rng.random() < 0.15:                                  # raw byte damage too
        b = bytearray(text, 'utf-8')
        for _ in range(rng.randint(1, 6)): b[rng.randrange(len(b))] = rng.randrange(32, 127)
        text = b.decode('utf-8', 'replace')
    t0 = time.time()
    try:
        err = run_once(text)
    except Exception as ex:                                  # nothing may escape: bad entries are dropped on load
        print('FAIL: %s: %s on iteration %d' % (type(ex).__name__, ex, it)); sys.exit(1)
    if err: print('FAIL: %s on iteration %d' % (err, it)); sys.exit(1)
    worst = max(worst, time.time() - t0); stats['ok'] += 1
print('fuzz_curves: %d iterations, all posed with finite numbers, slowest %.2f s: PASS' % (stats['ok'], worst))
