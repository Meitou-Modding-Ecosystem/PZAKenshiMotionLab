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
    return out


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
        _cache = json.load(open(DATA, encoding='utf-8')) if os.path.exists(DATA) else {}
    return _cache


def lookup(skeleton_name, plan):
    """The measured profile for a skeleton by file name, else the median profile of its plan, else None."""
    t = table()
    key = os.path.splitext(os.path.basename(skeleton_name or ''))[0].lower()
    if key in t.get('skeletons', {}): return t['skeletons'][key], key
    if plan in t.get('median', {}): return t['median'][plan], 'median of Kenshi %ss' % plan
    return None, None
