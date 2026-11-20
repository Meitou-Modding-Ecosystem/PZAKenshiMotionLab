# PZA's Kenshi Motion Lab - gait profiles: walk / run numbers measured from a skeleton's OWN animations (Kenshi's
# humans and animals), so procedural motion matches how the game moves that body. Author: PZA. GPLv3.
#
# Only numbers are stored (cycle length, stance share, stride, foot lift, pelvis drop, knee range, arm swing, speed),
# relative to leg length where it makes sense; no animation data is copied. Built by tools/build_gait_profiles.py into
# motionlab/data/kenshi_gaits.json; the driver looks a body up by its skeleton name, else uses the median of its plan.
import json, os
import numpy as np
from . import mathx as mx
from .model import Model, Skeleton

DATA = os.path.join(os.path.dirname(__file__), 'data', 'kenshi_gaits.json')
# curves measured from Kenshi's walk / run clips (tools/build_gait_profiles.py "<Kenshi>/data" rebuilds it): with it,
# people walk / run with Kenshi's own body turns, foot roll, arms and ankle path; without it (or with
# MOTIONLAB_NO_CURVES=1), the summary gait
CURVES = os.path.join(os.path.dirname(__file__), 'data', 'kenshi_curves.json')
CURVE_KEYS = ('turns', 'foot_roll', 'pelvis_y', 'arms', 'ankle_path')


def skeleton_model(path):
    """Skeleton + its animations from an Ogre .skeleton (metres), as a bones-only Model the rig mapper can read."""
    from .ogre import skeleton_read
    v, bones, parents, anims = skeleton_read.read(path)
    handles = sorted(bones); hidx = {h: i for i, h in enumerate(handles)}
    sk = Skeleton()
    for h in handles:
        nm, pos, q, sc = bones[h]
        sk.names.append(nm); sk.parents.append(hidx[parents[h]] if h in parents else -1)
        sk.rest_t.append(np.array(pos, float) / 10.0); sk.rest_q.append(mx.norm(np.array(q, float))); sk.rest_s.append(np.ones(3))
    m = Model(); m.skeleton = sk
    W = sk.world_rest(); m.verts = np.array([w[:3, 3] for w in W])
    m.verts[:, 1] -= 0.0
    m.normals = np.zeros_like(m.verts); m.tris = np.zeros((0, 3), int); m.uvs = np.zeros((len(m.verts), 2))
    m.joints = np.zeros((len(m.verts), 1), int); m.weights = np.ones((len(m.verts), 1))
    anims = {nm: (L, {hidx[h]: keys for h, keys in tr.items()}) for nm, L, tr in anims}
    return m, anims


def sampler(sk, anims, names):
    tracks = {}; length = 0.0
    for nm in names:
        if nm in anims:
            L, tr = anims[nm]; length = max(length, L); tracks.update(tr)
    def sample(t):
        out = []
        for i in range(len(sk.names)):
            tq, tt = mx.q_identity(), np.zeros(3)
            keys = tracks.get(i)
            if keys:
                ts = [k[0] for k in keys]; j = int(np.clip(np.searchsorted(ts, t) - 1, 0, len(keys) - 1)); j2 = min(j + 1, len(keys) - 1)
                u = 0.0 if j2 == j else float(np.clip((t - ts[j]) / max(ts[j2] - ts[j], 1e-9), 0, 1))
                tq = mx.q_slerp(mx.norm(np.array(keys[j][1], float)), mx.norm(np.array(keys[j2][1], float)), u)
                tt = (np.array(keys[j][2]) * (1 - u) + np.array(keys[j2][2]) * u) / 10.0
            # Ogre: animated = bind position + key translation, bind orientation * key rotation
            out.append(mx.trs(sk.rest_t[i] + tt, mx.q_mul(sk.rest_q[i], tq), np.ones(3)))
        return out
    return sample, length


def _angle(a, b):
    return float(np.degrees(np.arccos(np.clip(np.dot(mx.norm(a), mx.norm(b)), -1, 1))))


def measure(sk, rigmap, sample, length, fps=60):
    """The numbers of one cycle (any body the rig map describes)."""
    ix = {n: i for i, n in enumerate(sk.names)}
    Wr = sk.world_rest()
    root = ix[rigmap['root']]
    def tip_of(b):                                  # the toe / hoof tip: last bone down the foot chain (Nub included)
        while True:
            ch = sk.children(b)
            if len(ch) != 1: return b
            b = ch[0]
    legs = [(e, [ix[b] for b in e['bones']], ix[e['foot']] if e.get('foot') in ix else None) for e in rigmap['legs']]
    contact = {e['name']: tip_of(f) if f is not None else ch[-1] for e, ch, f in legs}
    def chain_len(ch):
        p = [Wr[c][:3, 3] for c in ch] + []
        return float(sum(np.linalg.norm(p[k + 1] - p[k]) for k in range(len(p) - 1)))
    # leg length = hip to ankle along the leg at rest
    lens = []
    for e, ch, f in legs:
        pts = [Wr[c][:3, 3] for c in ch] + ([Wr[f][:3, 3]] if f is not None else [])
        lens.append(float(sum(np.linalg.norm(pts[k + 1] - pts[k]) for k in range(len(pts) - 1))))
    L = float(np.median(lens)) if lens else 1.0
    n = max(8, int(length * fps)); ts = np.linspace(0, length, n, endpoint=False)
    rows = []
    for t in ts:
        W = sk.world(sample(t)); r = {'root': W[root][:3, 3].copy(), 'feet': {}, 'toe': {}, 'knee': {}, 'arm': {}, 'elbow': {}}
        for e, ch, f in legs:
            a, b = W[ch[0]][:3, 3], W[ch[1]][:3, 3]
            end = W[f][:3, 3] if f is not None else W[ch[-1]][:3, 3]
            r['knee'][e['name']] = 180.0 - _angle(a - b, end - b)
            r['feet'][e['name']] = end.copy(); r['toe'][e['name']] = W[contact[e['name']]][:3, 3].copy()
        for e in rigmap.get('arms', []):
            u, fa = W[ix[e['bones'][0]]][:3, 3], W[ix[e['bones'][1]]][:3, 3]
            ha = W[ix[e['hand']]][:3, 3] if e.get('hand') in ix else fa
            d = fa - u; r['arm'][e['side']] = float(np.degrees(np.arctan2(d[2], -d[1])))
            r['elbow'][e['side']] = 180.0 - _angle(u - fa, ha - fa)
        rows.append(r)
    out = {'cycle_s': float(length), 'leg_len_m': L}
    rest_root_y = Wr[root][1, 3]
    ry = np.array([r['root'][1] for r in rows])
    out['pelvis_drop'] = [float((rest_root_y - ry.max()) / L), float((rest_root_y - ry.min()) / L)]   # 0 = rest height
    duties, strides, lifts, speeds, kmin, kmax, centres = [], [], [], [], [], [], []
    for e, ch, f in legs:
        F = np.array([r['feet'][e['name']] for r in rows]); R = np.array([r['root'] for r in rows])
        rel = F - R
        y = F[:, 1]; low = y.min()
        C = np.array([r['toe'][e['name']] for r in rows]); cy = C[:, 1]
        vz = np.gradient(rel[:, 2]) * fps
        stance = (cy < cy.min() + 0.04 * L) & (vz < 0)           # the toe / hoof is down and the foot sweeps back
        duties.append(float(stance.mean()))
        strides.append(float((rel[:, 2].max() - rel[:, 2].min()) / L))
        rest_rel = Wr[f][2, 3] - Wr[root][2, 3] if f is not None else 0.0
        centres.append(float(((rel[:, 2].max() + rel[:, 2].min()) / 2 - rest_rel) / L))
        lifts.append(float((y.max() - low) / L))
        if stance.any(): speeds.append(float(-np.median(vz[stance])))
        k = np.array([r['knee'][e['name']] for r in rows]); kmin.append(float(k.min())); kmax.append(float(k.max()))
    out.update({'duty': float(np.median(duties)), 'stride': float(np.median(strides)), 'foot_centre': float(np.median(centres)), 'lift': float(np.median(lifts)),
                'speed_m_s': float(np.median(speeds)) if speeds else 0.0, 'knee_deg': [float(np.median(kmin)), float(np.median(kmax))]})
    if rigmap.get('arms'):
        a = np.array([r['arm'][e['side']] for r in rows for e in rigmap['arms']])
        el = np.array([r['elbow'][e['side']] for r in rows for e in rigmap['arms']])
        out['arm_deg'] = [float(a.min()), float(a.max())]; out['elbow_deg'] = [float(el.min()), float(el.max())]
    if rigmap.get('plan') == 'biped':
        try: out.update(_turns(sk, rigmap, sample, length, ts, fps, L))
        except (KeyError, ValueError, IndexError) as ex: out['turns_note'] = 'not measured: %s' % ex
    return out


def _rotvec(Rm):
    """Rotation vector (degrees) of a 3x3 rotation: x = pitch (+ leans forward / toe down), y = yaw, z = roll."""
    q = mx.norm(mx.m3_to_q(Rm)); w = float(np.clip(q[3], -1, 1))
    if w < 0: q, w = -q, -w
    s = np.sqrt(max(1 - w * w, 0.0)); ang = 2 * np.arccos(w)
    return np.degrees(q[:3] / s * ang) if s > 1e-9 else np.zeros(3)


def fourier(x, ph, n=2):
    """[c0, a1, b1, ..., an, bn] of x(ph) = c0 + sum a_k cos(2 pi k ph) + b_k sin(2 pi k ph) (least squares)."""
    cols = [np.ones_like(ph)]
    for k in range(1, n + 1): cols += [np.cos(2 * np.pi * k * ph), np.sin(2 * np.pi * k * ph)]
    A = np.stack(cols, axis=1); c, *_ = np.linalg.lstsq(A, np.asarray(x, float), rcond=None)
    return [float(v) for v in c]


def fourier_at(c, ph):
    v = c[0]
    for k in range(1, (len(c) - 1) // 2 + 1):
        v = v + c[2 * k - 1] * np.cos(2 * np.pi * k * ph) + c[2 * k] * np.sin(2 * np.pi * k * ph)
    return v


ROLL_SAMPLES = 9
R0_ = lambda Wr, b: Wr[b][:3, :3] / np.linalg.norm(Wr[b][:3, :3], axis=0)
Rw_ = lambda W, b: W[b][:3, :3] / np.linalg.norm(W[b][:3, :3], axis=0)


def _turns(sk, rigmap, sample, length, ts, fps, L):
    """How a person's body turns over the cycle (2026-11-19, the 'machine' look: ours never turned the pelvis or rolled
    the feet). Phase 0 = the left foot's heel strike. Pelvis and chest / head (each against the part below it) as
    2-harmonic Fourier numbers per axis (degrees); the foot's pitch and the toe's bend against the foot, sampled at
    ROLL_SAMPLES even steps through the stance and through the swing (stance and swing are timed separately because
    our stance is longer than Kenshi's run's)."""
    ix = {n: i for i, n in enumerate(sk.names)}; Wr = sk.world_rest()
    R0 = lambda b: Wr[ix[b]][:3, :3] / np.linalg.norm(Wr[ix[b]][:3, :3], axis=0)
    pel = ix['Bip01 Pelvis']; chest = ix[rigmap['spine_chain'][-1]] if rigmap.get('spine_chain') else ix['Bip01 Spine2']
    head = ix[rigmap['head']]
    legL = [e for e in rigmap['legs'] if e['side'] == 'L'][0]
    foot = ix[legL['foot']]; toes = sk.children(foot)
    if not toes: raise ValueError('no toe bone')
    toe = toes[0]
    arm_parts = {}                                                 # per side: shoulder (clavicle), upper arm, forearm, hand
    for e in rigmap.get('arms', []):
        up_, fo_ = ix[e['bones'][0]], ix[e['bones'][1]]
        arm_parts[e['side']] = {'clav': sk.parents[up_], 'upper': up_, 'fore': fo_, 'hand': ix[e['hand']] if e.get('hand') in ix else None}
    extra = [b for d in arm_parts.values() for b in d.values() if b is not None and b >= 0]
    rest = {b: (Wr[b][:3, :3] / np.linalg.norm(Wr[b][:3, :3], axis=0)) for b in [pel, chest, head, foot, toe] + extra}
    rows = []; frames = []
    for t in ts:
        W = sk.world(sample(t)); frames.append(np.array(W)); Rw = {b: W[b][:3, :3] / np.linalg.norm(W[b][:3, :3], axis=0) for b in rest}
        D = {b: Rw[b] @ rest[b].T for b in rest}                     # world turn of each part away from its rest pose
        arm = {}
        for sd, d in arm_parts.items():                            # each part against its parent (as the driver sets it)
            par = {'clav': sk.parents[d['clav']], 'upper': d['clav'], 'fore': d['upper'], 'hand': d['fore']}
            for k, b in d.items():
                if b is None or b < 0: continue
                pb = par[k]
                Dp = (Rw_(W, pb) @ R0_(Wr, pb).T) if pb >= 0 else np.eye(3)
                arm[sd + '_' + k] = _rotvec(Dp.T @ D[b])
        rows.append({'arm': arm, 'pel': _rotvec(D[pel]), 'chest': _rotvec(D[pel].T @ D[chest]), 'head': _rotvec(D[chest].T @ D[head]),
                     'foot': _rotvec(D[foot])[0], 'toe': _rotvec(D[foot].T @ D[toe])[0],
                     'ank': W[foot][1, 3], 'toe_y': W[toe][1, 3], 'z': W[foot][2, 3] - W[ix[rigmap['root']]][2, 3],
                     'root_y': W[ix[rigmap['root']]][1, 3] - Wr[ix[rigmap['root']]][1, 3]})
    n = len(rows)
    ay = np.array([r['ank'] for r in rows]); ty = np.array([r['toe_y'] for r in rows]); z = np.array([r['z'] for r in rows])
    vz = np.gradient(z) * fps
    stance = ((ay < ay.min() + 0.03 * L) | (ty < ty.min() + 0.04 * L)) & (vz < 0)
    if stance.all() or not stance.any(): raise ValueError('no clear stance')
    # the longest run of stance frames (cyclic) = the foot's contact; its first frame = heel strike
    best, start = 0, 0
    for i in range(n):
        if stance[i] and not stance[i - 1]:
            k = 0
            while stance[(i + k) % n] and k < n: k += 1
            if k > best: best, start = k, i
    ph = ((np.arange(n) - start) % n) / n
    duty = best / n
    pick = lambda key, ax: np.array([r[key][ax] for r in rows])
    def stretch(x, a, b, k=ROLL_SAMPLES):                          # x over phase [a, b] at k even steps
        u = np.linspace(0, 1, k); pp = a + u * (b - a)
        order = np.argsort(ph); phs = np.concatenate([ph[order], [1.0]]); vs = np.concatenate([x[order], x[order][:1]])
        return [float(v) for v in np.interp(pp, phs, vs)]
    out = {'turns': {part: {nm: fourier(pick(part, a), ph, n=4) for nm, a in (('pitch', 0), ('yaw', 1), ('roll', 2))}
                     for part in ('pel', 'chest', 'head')}}
    # the hips' height over the cycle (leg lengths against rest), 4 harmonics: a smooth double bounce
    out['pelvis_y'] = fourier(np.array([r['root_y'] for r in rows]) / L, ph, n=4)
    # the planted ankle sits a little under its rest height (leg lengths), and where the ankle travels (front / back of
    # its sweep, against the root, leg lengths): our feet aim for the same span
    out['ankle_sink'] = float((np.median(ay[stance]) - Wr[foot][1, 3]) / L)
    out['ankle_z'] = [float(z.min() / L), float(z.max() / L)]
    rz = z; ry = ay - Wr[foot][1, 3]                               # against the root / the rest ankle height
    out['ankle_path'] = {'stance_z': stretch(rz / L, 0.0, duty, 17), 'swing_z': stretch(rz / L, duty, 1.0, 17),
                         'stance_y': stretch(ry / L, 0.0, duty, 17), 'swing_y': stretch(ry / L, duty, 1.0, 17)}
    if rows[0]['arm']:                                             # the arms: measured too (they swing across the body,
        out['arms'] = {k: {nm: fourier([r['arm'][k][a] for r in rows], ph, n=4)   # the shoulders move)
                           for nm, a in (('pitch', 0), ('yaw', 1), ('roll', 2))} for k in rows[0]['arm']}
    fp = np.array([r['foot'] for r in rows]); tp = np.array([r['toe'] for r in rows])
    from . import metrics                                          # the Studio scorecard's numbers (test references)
    out['scorecard'], _ = metrics.cycle(np.array(frames), np.array(Wr), metrics.roles(sk.names, rigmap), fps)
    out['foot_roll'] = {'duty': float(duty), 'stance': stretch(fp, 0.0, duty), 'swing': stretch(fp, duty, 1.0),
                        'toe_stance': stretch(tp, 0.0, duty), 'toe_swing': stretch(tp, duty, 1.0)}
    return out


def _nums(v, n, lim):
    """v is a list of exactly n finite numbers, each within +-lim."""
    return (isinstance(v, list) and len(v) == n and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v)
            and bool(np.all(np.isfinite(v))) and float(np.max(np.abs(v))) <= lim)


def valid_curve(key, v):
    """A curve entry from the local file has exactly the shape the driver reads, with sane numbers (anything else is
    dropped and the summary gait is used): angles within 720 deg, lengths within 5 leg lengths."""
    try:
        if key in ('turns', 'arms'):
            parts = ('pel', 'chest', 'head') if key == 'turns' else None
            if not isinstance(v, dict) or not v or (parts and set(v) != set(parts)): return False
            return all(isinstance(d, dict) and set(d) == {'pitch', 'yaw', 'roll'} and all(_nums(c, 9, 720.0) for c in d.values())
                       for d in v.values())
        if key == 'foot_roll':
            return (isinstance(v, dict) and set(v) == {'duty', 'stance', 'swing', 'toe_stance', 'toe_swing'}
                    and isinstance(v['duty'], (int, float)) and 0.05 <= float(v['duty']) <= 0.95
                    and all(_nums(v[k], ROLL_SAMPLES, 720.0) for k in ('stance', 'swing', 'toe_stance', 'toe_swing')))
        if key == 'pelvis_y': return _nums(v, 9, 5.0)
        if key == 'ankle_path':
            return (isinstance(v, dict) and set(v) == {'stance_z', 'swing_z', 'stance_y', 'swing_y'}
                    and all(_nums(c, 17, 5.0) for c in v.values()) and v['stance_z'][0] - v['stance_z'][-1] > 0.05)
    except (TypeError, ValueError, KeyError):
        return False
    return False


def add_summaries(g):
    """Lossy summaries of a gait's curves (the fallback gait when the curves file is missing): the first
    harmonic (mean + one sine) of each body turn, the hips' mean and second harmonic, and four numbers of the foot
    roll (stance share, angle at heel strike, at toe-off, the swing's steepest). Nothing here rebuilds the clip."""
    if 'turns' in g:
        g['turns_h1'] = {pt: {a: [float(v) for v in c[:3]] for a, c in d.items()} for pt, d in g['turns'].items()}
    if 'arms' in g:
        g['arms_h1'] = {pt: {a: [float(v) for v in c[:3]] for a, c in d.items()} for pt, d in g['arms'].items()}
    if 'pelvis_y' in g:
        c = g['pelvis_y']; g['hips_h2'] = [float(c[0]), 0.0, 0.0, float(c[3]), float(c[4])]
    if 'foot_roll' in g:
        fr = g['foot_roll']
        g['roll_summary'] = {'duty': float(fr['duty']), 'strike': float(fr['stance'][0]), 'off': float(fr['stance'][-1]),
                             'swing_max': float(max(fr['swing']))}


def roll_from_summary(rs, n=ROLL_SAMPLES):
    """A foot roll of our own shape through the four summary numbers: heel strike toe-up, flat through mid-stance,
    heel rising to toe-off, toes pointing down early in the swing, back up for the next strike; the toes bend up
    against the foot while the heel rises."""
    st_, off, mx_ = rs['strike'], rs['off'], max(rs['swing_max'], rs['off'])
    u = np.linspace(0, 1, n)
    stance = np.interp(u, [0, 0.2, 0.6, 1.0], [st_, 0.0, 0.0, off])
    swing = np.interp(u, [0, 0.3, 0.75, 1.0], [off, mx_, 0.0, st_])
    toe = np.interp(u, [0, 0.6, 0.85, 1.0], [0.0, 0.0, -0.45 * max(off, 0.0), 0.0])
    return {'duty': float(rs['duty']), 'stance': [float(v) for v in stance], 'swing': [float(v) for v in swing],
            'toe_stance': [float(v) for v in toe], 'toe_swing': [0.0] * n}


def pick(anims, kind):
    """Animation names for a walk / run cycle (split lower / upper clips are combined)."""
    names = list(anims)
    exact = [n for n in names if n.lower() == kind]
    if exact: return exact
    low = [n for n in names if n.lower() == kind + ' lower']
    if low: return low + [n for n in names if n.lower() == kind + ' upper']
    cand = [n for n in names if n.lower().startswith(kind) and not any(x in n.lower() for x in ('carry', 'sword', 'limp', 'turn'))]
    return sorted(cand, key=len)[:1]


def profile_for_skeleton(path):
    from .existing import rigmap_from_skeleton
    m, anims = skeleton_model(path)
    rm = rigmap_from_skeleton(m)
    if rm is None: return None
    out = {'plan': rm['plan'], 'source': os.path.basename(path)}
    for kind in ('walk', 'run'):
        names = pick(anims, kind)
        if not names: continue
        smp, L = sampler(m.skeleton, anims, names)
        if L < 0.1: continue
        out[kind] = measure(m.skeleton, rm, smp, L); out[kind]['clips'] = names
    return out if ('walk' in out or 'run' in out) else None


_cache = None
def table():
    global _cache
    if _cache is None:
        _cache = {}
        if os.path.exists(DATA):
            with open(DATA, encoding='utf-8') as f: _cache = json.load(f)
        if os.path.exists(CURVES) and _cache and not os.environ.get('MOTIONLAB_NO_CURVES'):   # merge the curves in
            try:
                with open(CURVES, encoding='utf-8') as f: cv = json.load(f)
                for grp in ('skeletons', 'median'):
                    for key, kinds in cv.get(grp, {}).items():
                        if not isinstance(kinds, dict): continue
                        for kind, c in kinds.items():
                            tgt = _cache.get(grp, {}).get(key, {}).get(kind)
                            if isinstance(tgt, dict) and isinstance(c, dict):
                                tgt.update({k: v for k, v in c.items() if k in CURVE_KEYS and valid_curve(k, v)})
            except (ValueError, OSError, AttributeError, TypeError, RecursionError):
                pass                                            # a broken local file: the public numbers still work
    return _cache


def lookup(skeleton_name, plan):
    """The measured profile for a skeleton by file name, else the median profile of its plan, else None."""
    t = table()
    key = os.path.splitext(os.path.basename(skeleton_name or ''))[0].lower()
    if key in t.get('skeletons', {}): return t['skeletons'][key], key
    if plan in t.get('median', {}): return t['median'][plan], 'median of Kenshi %ss' % plan
    return None, None
