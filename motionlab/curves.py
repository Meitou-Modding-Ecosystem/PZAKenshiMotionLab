# PZA's Kenshi Motion Lab - drive a body from a REFERENCE CLIP's own curves (retargeting).
#
# A reference .glb (a skeleton-only clip, e.g. a Mixamo download converted by tools/fbx_to_glb.py with your own
# account) is read frame by frame into body curves that do not depend on its bone set or size:
#   pelvis path (relative to its standing height), the turn of the pelvis, chest and head (against its rest pose),
#   each foot's place and turn, each hand's place relative to its shoulder (in arm lengths), and which way the
#   knees and elbows point.
# Those curves then drive OUR body's own skeleton through the same driver as every other clip (leg / arm solvers,
# spine spread, the floor / knee / hand rules), scaled to our body. Nothing of the reference is stored by this
# code: run it on files you are allowed to use, and keep what it makes where their licence allows.
# Author: PZA. GPLv3.
import numpy as np
from . import mathx as mx
from .motion import ClipDef
from .reference import load_reference, roles

UP = np.array([0.0, 1.0, 0.0])


def _frame(x, y):
    """Rotation matrix whose X axis is x and Y axis is y (y made perpendicular)."""
    x = mx.norm(np.asarray(x, float)); y = np.asarray(y, float) - x * float(np.dot(x, y)); y = mx.norm(y)
    z = np.cross(x, y)
    return np.stack([x, y, z], axis=1)


def _q(Rm):
    return mx.norm(mx.m3_to_q(Rm))


def _hint(a, b, c):
    """Which way the middle joint b points away from the line a-c (unit), or None when the limb is straight."""
    d = mx.norm(c - a); v = (b - a) - d * float(np.dot(b - a, d)); n = np.linalg.norm(v)
    return v / n if n > 1e-4 else None


def extract(path, fps=30):
    """Body curves of a reference clip, sampled at fps. Facing is turned to +Z, distances are relative to its
    standing pelvis height (H) or arm length."""
    sk, clip, length = load_reference(path)
    if not (np.isfinite(length) and 0 < length <= 600): raise ValueError('reference clip length %r s is not usable' % length)
    R = roles(sk)
    need = ('hips', 'spine', 'chest', 'neck', 'head', 'l_thigh', 'l_knee', 'l_foot', 'l_toe', 'r_thigh', 'r_knee',
            'r_foot', 'r_toe', 'l_shoulder', 'l_elbow', 'l_hand', 'r_shoulder', 'r_elbow', 'r_hand')
    miss = [n for n in need if n not in R]
    if miss: raise ValueError('reference has no bone for: %s' % ', '.join(miss))
    from .bake import sample_read_clip
    Wr = sk.world_rest()
    P = lambda W, r: W[R[r]][:3, 3]
    # facing at rest -> +Z
    fwd = mx.norm(np.cross(P(Wr, 'l_thigh') - P(Wr, 'r_thigh'), UP)); yaw = np.arctan2(fwd[0], fwd[2])
    Ay = mx.q_to_m3(mx.q_axis_angle(UP, -yaw))
    al = lambda v: Ay @ v
    ground = min(P(Wr, 'l_toe')[1], P(Wr, 'r_toe')[1], P(Wr, 'l_foot')[1], P(Wr, 'r_foot')[1])
    H = float(P(Wr, 'hips')[1] - ground)
    reach = {s: float(np.linalg.norm(P(Wr, s + '_elbow') - P(Wr, s + '_shoulder')) +
                      np.linalg.norm(P(Wr, s + '_hand') - P(Wr, s + '_elbow'))) for s in 'lr'}
    def frames(W):
        g = {r: al(P(W, r)) for r in R}
        pel = _frame(g['l_thigh'] - g['r_thigh'], g['spine'] - g['hips'])
        chest = _frame(g['l_shoulder'] - g['r_shoulder'], g['neck'] - g['chest'])
        head = _frame(chest[:, 0], (g['head_top'] if 'head_top' in g else g['head'] + chest[:, 1]) - g['head'])
        feet = {s: _frame(pel[:, 0], g[s + '_toe'] - g[s + '_foot']) for s in 'lr'}
        return g, pel, chest, head, feet
    g0, pel0, chest0, head0, feet0 = frames(Wr)
    W0 = sk.world(sample_read_clip(sk, clip, 0.0)); o = al(P(W0, 'hips')); o[1] = 0.0
    ts = np.arange(0.0, length + 1e-9, 1.0 / fps)
    cur = {k: [] for k in ('pelvis', 'q_pel', 'q_chest', 'q_head')}
    for s in 'lr':
        for k in ('foot', 'q_foot', 'knee', 'hand', 'elbow', 'hand_y'): cur[s + '_' + k] = []
    for t in ts:
        W = sk.world(sample_read_clip(sk, clip, float(t)))
        g, pel, chest, head, feet = frames(W)
        p = g['hips'] - o; p[1] = g['hips'][1] - ground
        cur['pelvis'].append(p / H)
        cur['q_pel'].append(_q(pel @ pel0.T)); cur['q_chest'].append(_q(chest @ chest0.T)); cur['q_head'].append(_q(head @ head0.T))
        for s in 'lr':
            f = g[s + '_foot'] - o; f[1] = g[s + '_foot'][1] - ground
            cur[s + '_foot'].append(f / H)
            cur[s + '_q_foot'].append(_q(feet[s] @ feet0[s].T))
            cur[s + '_knee'].append(_hint(g[s + '_thigh'], g[s + '_knee'], g[s + '_foot']))
            cur[s + '_hand'].append((g[s + '_hand'] - g[s + '_shoulder']) / reach[s])
            cur[s + '_hand_y'].append((g[s + '_hand'][1] - ground) / H)
            cur[s + '_elbow'].append(_hint(g[s + '_shoulder'], g[s + '_elbow'], g[s + '_hand']))
    out = {'t': ts, 'length': float(ts[-1]), 'H': H,
           'ankle': {s: float(g0[s + '_foot'][1] - ground) / H for s in 'lr'},
           'foot_rest': {s: (g0[s + '_foot'] - g0['hips']) / H for s in 'lr'}}
    for k, v in cur.items():
        if k.endswith('_knee') or k.endswith('_elbow'):         # straight limb: keep the last known bend side
            last = None; vv = []
            for x in v:
                last = x if x is not None else last; vv.append(last)
            first = next((x for x in vv if x is not None), np.array([0.0, 0.0, 1.0]))
            v = [x if x is not None else first for x in vv]
        out[k] = np.array(v, float)
    for k, v in out.items():
        if isinstance(v, np.ndarray) and not np.isfinite(v).all(): raise ValueError('reference has broken numbers in %s' % k)
    if not (np.isfinite(H) and H > 1e-3): raise ValueError('reference standing height %r is not usable' % H)
    # quaternion sign continuity (so interpolation never takes the long way round)
    for k in [k for k in out if k.startswith('q_') or k.endswith('_q_foot')]:
        q = out[k]
        for i in range(1, len(q)):
            if np.dot(q[i], q[i - 1]) < 0: q[i] = -q[i]
    return out


def _at(cur, key, t):
    ts = cur['t']; v = cur[key]
    if t <= ts[0]: return v[0]
    if t >= ts[-1]: return v[-1]
    k = int(np.searchsorted(ts, t) - 1); u = (t - ts[k]) / (ts[k + 1] - ts[k])
    if key.startswith('q_') or key.endswith('_q_foot'): return mx.q_slerp(v[k], v[k + 1], u)
    return v[k] * (1 - u) + v[k + 1] * u


def curve_clip(drv, cur, name, note='retargeted from a reference clip'):
    """A ClipDef that drives drv's body (a biped: Bip01 people) by the curves from extract()."""
    if drv.plan != 'biped' or len(drv.legs) < 2: raise ValueError('reference clips drive biped rigs (people) only')
    r = drv.rig; root = r.b(drv.map['root'])
    P0 = r.Wrest[root][:3, 3].copy()
    # standing height measured the same way as the reference's: pelvis above the lowest foot / toe joint at rest
    feet_bones = [r.b(e['foot']) for e, c in drv.legs if e.get('foot') in r.idx]
    feet_bones += [k for k in range(r.n) if r.parents[k] in feet_bones]
    g0 = min(float(r.Wrest[k][1, 3]) for k in feet_bones) if feet_bones else 0.0
    Hu = float(P0[1]) - g0
    side = lambda e: e['side'].lower()
    ankle = {e['name']: float(drv.rest_tip[e['name']][1]) for e, c in drv.legs}

    def fn(t, T):
        p = _at(cur, 'pelvis', t)
        qp = _at(cur, 'q_pel', t); qc = _at(cur, 'q_chest', t); qh = _at(cur, 'q_head', t)
        sp = {'root_offset': (p[0] * Hu, g0 + p[1] * Hu - P0[1], p[2] * Hu), 'root_rot': qp,
              'body_rot': mx.norm(mx.q_mul(mx.q_conj(qp), qc)), 'head_rot': mx.norm(mx.q_mul(mx.q_conj(qc), qh)),
              'knee_floor': 0.035 * Hu, 'body_floor': 0.03 * Hu, 'feet': {'_stance': set()},
              'leg_bend': {}, 'foot_delta': {}, 'hands_rel': {}, 'arm_bend': {}, 'hand_floor': 0.03 * Hu}
        for e, c in drv.legs:
            s = side(e); f = _at(cur, s + '_foot', t)
            # our own foot spacing at rest, plus the reference foot's movement away from ITS rest place
            ours = drv.rest_tip[e['name']] - P0; ref = cur['foot_rest'][s]
            sp['feet'][e['name']] = np.array([P0[0] + ours[0] + (f[0] - ref[0]) * Hu,
                                              ankle[e['name']] + (f[1] - cur['ankle'][s]) * Hu,
                                              P0[2] + ours[2] + (f[2] - ref[2]) * Hu])
            sp['leg_bend'][e['name']] = _at(cur, s + '_knee', t)
            sp['foot_delta'][e['name']] = _at(cur, s + '_q_foot', t)
        for e, c in drv.arms:
            s = side(e)
            sp['hands_rel'][e['side']] = _at(cur, s + '_hand', t)
            sp['arm_bend'][e['side']] = _at(cur, s + '_elbow', t)
        return sp
    return ClipDef(name, cur['length'], fn, False, 0.0, note)
