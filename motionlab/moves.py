# PZA's Kenshi Motion Lab - knockdowns and get-ups for people (biped rigs), built from body STATES set here in code
# (how far the body is tipped, pelvis height and travel, feet planted or carried, hands on the ground, arm / spine /
# head angles); the leg and arm solvers do the rest. Timings and proportions come from measured reference clips
# (motionlab/reference.py: numbers only, no animation data). Author: PZA. GPLv3.
import numpy as np
from . import mathx as mx
from .motion import ClipDef

# measured (medians of the reference falls / get-ups; seconds, heights relative to the standing pelvis height H0)
FALL_BACK = dict(react=0.15, drop=0.75, bounce=0.06, settle=0.7, travel=-0.42, knee=110.0)
FALL_FRONT = dict(react=0.12, drop=0.65, bounce=0.07, settle=0.9, travel=0.72, knee=120.0)
# get-ups: the PHASE ORDER and supports follow the reference (hands down while sitting up / pushing up, knees down
# while kneeling), but the timing is game pace, about half the reference length (those take 5.5-5.7 s)
GETUP_BACK = dict(prop=1.0, rise=0.95, stand=0.35)        # prop up on hands / sit up, rise to a crouch, stand
GETUP_FRONT = dict(push=1.0, kneel=0.8, rise=0.9)         # push up to hands and knees, kneel, stand

KEYS = ('th', 'py', 'pz', 'fz', 'fy', 'fw', 'arm', 'fore', 'spine', 'head', 'hw', 'hz', 'hx', 'tk', 'fa')


def _lerp_state(a, b, u):
    return {k: a[k] + (b[k] - a[k]) * u for k in KEYS}


class Track:
    """Body states at times. Between keys a monotone cubic (no overshoot) keeps the motion flowing THROUGH a key
    instead of stopping at each one; a key may ask for its own entry: 'fall' (speeds up into it, like gravity: an
    impact), 'out' (fast start, settles into it), 'lin', or 'smooth' (flows through)."""

    def __init__(self, keys):
        self.keys = sorted(keys, key=lambda k: k[0])
        ts = np.array([k[0] for k in self.keys], float); self.ts = ts
        V = np.array([[k[1][c] for c in KEYS] for k in self.keys], float); self.V = V
        n = len(ts); h = np.diff(ts); h = np.where(h < 1e-9, 1e-9, h)
        d = np.diff(V, axis=0) / h[:, None]                       # slope of each segment
        m = np.zeros_like(V)                                       # Fritsch-Carlson tangents (first / last key: at rest)
        for i in range(1, n - 1):
            w1 = 2 * h[i] + h[i - 1]; w2 = h[i] + 2 * h[i - 1]
            a, b = d[i - 1], d[i]
            ok = (a * b) > 0
            m[i] = np.where(ok, (w1 + w2) / np.where(ok, w1 / np.where(ok, a, 1) + w2 / np.where(ok, b, 1), 1), 0.0)
        self.m0 = np.zeros_like(d); self.m1 = np.zeros_like(d)    # tangents at each segment's start / end
        for k in range(n - 1):
            ease = self.keys[k + 1][2]
            self.m0[k], self.m1[k] = {'fall': (0 * d[k], 2 * d[k]), 'out': (2 * d[k], 0 * d[k]),
                                      'lin': (d[k], d[k])}.get(ease, (m[k], m[k + 1]))
        self.h = h

    def at(self, t):
        ts = self.ts
        if t <= ts[0]: return dict(zip(KEYS, self.V[0]))
        if t >= ts[-1]: return dict(zip(KEYS, self.V[-1]))
        k = int(np.clip(np.searchsorted(ts, t) - 1, 0, len(ts) - 2))
        h = self.h[k]; u = (t - ts[k]) / h
        h00 = 2 * u ** 3 - 3 * u ** 2 + 1; h10 = u ** 3 - 2 * u ** 2 + u; h01 = -2 * u ** 3 + 3 * u ** 2; h11 = u ** 3 - u ** 2
        v = h00 * self.V[k] + h10 * h * self.m0[k] + h01 * self.V[k + 1] + h11 * h * self.m1[k]
        return dict(zip(KEYS, v))


# overlap: parts of the body reach each pose a little after the body does (seconds), and left / right differ, so the
# limbs never move as one block. Arm and leg amplitude differ slightly per side.
LAG = dict(spine=0.03, head=0.08, arm_L=0.05, arm_R=0.10, leg_L=0.0, leg_R=0.05)
SIDE_AMP = dict(L=1.0, R=0.9)


def state(**kw):
    s = dict(th=0.0, py=0.0, pz=0.0, fz=0.0, fy=0.0, fw=0.0, arm=0.0, fore=10.0, spine=0.0, head=0.0, hw=0.0, hz=0.0, hx=0.1, tk=0.0, fa=0.0)
    s.update(kw); return s


def biped_moves(drv):
    """Knockdowns and get-ups as ClipDefs for a biped driver (Kenshi people)."""
    if drv.plan != 'biped' or len(drv.legs) < 2: return []
    r = drv.rig; Wr = r.Wrest
    root = r.b(drv.map['root'])
    P0 = Wr[root][:3, 3].copy()                        # pelvis at rest (the standing height H0)
    H0 = float(P0[1]); L = drv.leg_len
    lie = 0.13 * H0                                    # pelvis height lying flat
    def st(**kw):                                      # a body state; default = standing at rest height
        kw.setdefault('py', H0); return state(**kw)
    ankle = {e['name']: float(drv.rest_tip[e['name']][1]) for e, c in drv.legs}
    shoulder = {e['side']: Wr[c[0]][:3, 3] - P0 for e, c in drv.arms}      # shoulder offset from the pelvis at rest
    reach = {e['side']: float(sum(r.length[b] for b in c)) for e, c in drv.arms}
    X = [1, 0, 0]

    def spec_from(S):
        """S(lag) = the body state lag seconds ago; the root follows S(0), the other parts their own lag."""
        s = S(0.0); sp_ = S(LAG['spine']); hd = S(LAG['head'])
        th = np.radians(s['th'])
        turn = mx.q_axis_angle(X, th)
        pel = np.array([P0[0], s['py'], P0[2] + s['pz']])
        feet = {'_stance': set()}
        for e, c in drv.legs:
            g = S(LAG['leg_' + e['side']]) if e['side'] in 'LR' else s
            rest = drv.rest_tip[e['name']]
            planted = rest + np.array([0.0, 0.0, g['fa']])     # fa: where the planted feet stand (after travel)
            carried = np.array([rest[0], max(ankle[e['name']], g['fy']), pel[2] + g['fz']])
            f = planted * (1 - g['fw']) + carried * g['fw']
            if g['tk'] > 0:                            # tucked: feet drawn up under the body, in the body's own frame
                tuck = pel + mx.q_rotate(turn, np.array([rest[0] - P0[0], -0.38 * L, 0.12 * L]))
                f = f * (1 - g['tk']) + tuck * g['tk']
            f[1] = max(f[1], ankle[e['name']])
            feet[e['name']] = f
        arms = {e['side']: S(LAG.get('arm_' + e['side'], 0.05)) for e, c in drv.arms}
        sp = {'root_offset': (0.0, s['py'] - P0[1], s['pz']), 'root_rot': turn, 'feet': feet,
              'arm_swing': {sd: a['arm'] * SIDE_AMP.get(sd, 1.0) for sd, a in arms.items()},
              'forearm_bend': {sd: a['fore'] * SIDE_AMP.get(sd, 1.0) for sd, a in arms.items()},
              'body_rot': mx.q_axis_angle(X, np.radians(sp_['spine'])), 'head_rot': mx.q_axis_angle(X, np.radians(hd['head'])),
              'feet_follow_body': (s['fw'] > 0.5 and s['th'] < 0) or s['tk'] > 0.5,   # on the back / tucked: feet go with the body
              'elbows_out': 0.8 if 45 < s['th'] < 135 else 0.0, 'knee_floor': 0.035 * H0, 'body_floor': 0.03 * H0}
        if s['hw'] > 0.5:                              # hands planted: under the shoulder, hz / hx in arm lengths
            sp['hands'] = {}
            for e, c in drv.arms:
                sd = e['side']; a_ = arms[sd]
                if a_['hw'] <= 0.5: continue           # this hand is not down yet / already up (its own lag)
                sh = pel + mx.q_rotate(mx.q_mul(turn, mx.q_axis_angle(X, np.radians(sp_['spine']))), shoulder[sd]); out = 1.0 if sd == 'L' else -1.0
                sp['hands'][sd] = np.array([sh[0] + out * a_['hx'] * reach[sd], 0.04 * H0, sh[2] + a_['hz'] * reach[sd]])
        return sp

    # knockdown backward: snap back, knees buckle, topple, land with a small bounce, settle, lie still
    fb = FALL_BACK; t1 = fb['react']; t2 = t1 + fb['drop']; trav = fb['travel'] * H0
    back_keys = [
        (0.0, st(), 'lin'),
        (t1, st(th=-6, head=-18, spine=-8, arm=-35, py=H0 - 0.02 * H0), 'out'),
        (t1 + 0.45 * fb['drop'], st(th=-35, py=0.62 * H0, pz=0.45 * trav, fw=0.3, fz=0.25 * L, arm=-80, fore=30, head=10, spine=-10), 'smooth'),
        (t2, st(th=-90, py=lie, pz=trav, fw=1.0, fz=0.95 * L, arm=-110, fore=20, head=15), 'fall'),
        (t2 + 0.12, st(th=-86, py=lie + fb['bounce'] * H0, pz=trav, fw=1.0, fz=0.95 * L, fy=0.06 * H0, arm=-70, fore=25, head=-5), 'out'),
        (t2 + fb['settle'], st(th=-90, py=lie, pz=trav, fw=1.0, fz=0.97 * L, arm=-4, fore=10, head=5), 'smooth'),
        (t2 + fb['settle'] + 0.6, st(th=-90, py=lie, pz=trav, fw=1.0, fz=0.97 * L, arm=-4, fore=10, head=5), 'lin'),
    ]
    # knockdown forward: jolt forward, knees give, pitch over, hands brace, land on the stomach
    ff = FALL_FRONT; t1 = ff['react']; t2 = t1 + ff['drop']; trav = ff['travel'] * H0
    front_keys = [
        (0.0, st(), 'lin'),
        (t1, st(th=8, head=12, spine=10, arm=10, py=H0 - 0.03 * H0), 'out'),
        (t1 + 0.5 * ff['drop'], st(th=40, py=0.6 * H0, pz=0.4 * trav, fw=0.3, fz=-0.2 * L, arm=-70, fore=20, spine=12), 'smooth'),
        (t2 - 0.05, st(th=80, py=0.22 * H0, pz=0.9 * trav, fw=0.9, fz=-0.85 * L, hw=1.0, hz=0.35, hx=0.15, spine=5, head=-20), 'fall'),
        (t2 + 0.1, st(th=90, py=lie, pz=trav, fw=1.0, fz=-0.95 * L, fy=0.05 * H0, hw=1.0, hz=0.1, hx=0.2, head=-25), 'out'),
        (t2 + ff['settle'], st(th=90, py=lie, pz=trav, fw=1.0, fz=-0.97 * L, arm=-6, fore=15, head=-30), 'smooth'),
        (t2 + ff['settle'] + 0.6, st(th=90, py=lie, pz=trav, fw=1.0, fz=-0.97 * L, arm=-6, fore=15, head=-30), 'lin'),
    ]
    # get up from the back: prop on the hands and sit up, pull the feet in, rise over them, stand
    gb = GETUP_BACK; a = gb['prop']; b = a + gb['rise']; c = b + gb['stand']
    getup_back_keys = [
        (0.0, st(th=-90, py=lie, fw=1.0, fz=0.97 * L, arm=-4, fore=10, head=5), 'lin'),
        (0.45 * a, st(th=-60, py=lie + 0.03 * H0, fw=1.0, fz=0.8 * L, hw=1.0, hz=-0.3, hx=0.15, head=15, spine=15), 'smooth'),
        (a, st(th=-40, py=0.15 * H0, fw=1.0, fz=0.35 * L, fy=0.0, hw=1.0, hz=-0.3, hx=0.15, head=10, spine=12), 'smooth'),
        (a + 0.45 * gb['rise'], st(th=25, py=0.42 * H0, fw=1.0, fz=0.05 * L, arm=-40, fore=60, head=-10, spine=15), 'smooth'),
        (b, st(th=12, py=0.82 * H0, fw=0.6, fz=0.02 * L, arm=-15, fore=30, spine=8), 'smooth'),
        (c, st(), 'smooth'),
    ]
    # get up from the stomach: push up to hands and knees, kneel, stand
    gf = GETUP_FRONT; a = gf['push']; b = a + gf['kneel']; c = b + gf['rise']
    getup_front_keys = [
        (0.0, st(th=90, py=lie, fw=1.0, fz=-0.97 * L, arm=-6, fore=15, head=-30), 'lin'),
        (0.35 * a, st(th=85, py=lie + 0.04 * H0, fw=1.0, fz=-0.9 * L, arm=-6, hw=1.0, hz=0.0, hx=0.2, head=-15), 'smooth'),
        (a, st(th=65, py=0.36 * H0, fw=1.0, fz=-0.45 * L, arm=-60, hw=1.0, hz=0.05, hx=0.12, head=-10), 'smooth'),
        (a + 0.3 * gf['kneel'], st(th=50, py=0.42 * H0, fw=1.0, fz=-0.3 * L, arm=-50, fore=20, hw=1.0, hz=0.0, hx=0.1, head=-5, spine=5), 'smooth'),
        (b, st(th=30, py=0.5 * H0, fw=1.0, fz=-0.1 * L, arm=-30, fore=40, head=0, spine=10), 'smooth'),
        (b + 0.55 * gf['rise'], st(th=12, py=0.8 * H0, fw=0.5, fz=0.0, arm=-10, fore=25, spine=6), 'smooth'),
        (c, st(), 'smooth'),
    ]
    # forward rolls: crouch, reach, hands down, tuck the head, roll over the back, feet under, rise; travel forward.
    # Key fractions and heights from the measured rolls (motionlab/reference.py measure_roll): turned 360, upside down
    # at ~0.5 of the clip, pelvis lowest (~0.14-0.17 of standing) on the back at ~0.6, standing again at ~0.9-0.96
    def roll_keys(T, trav, hop):
        f = lambda x: x * T
        k = [(0.0, st(), 'lin'),
             (f(0.18), st(th=18, py=0.70 * H0, pz=0.05 * trav, arm=-70, fore=20, spine=10, head=5), 'smooth')]
        if hop:                                         # the dive: spring up and forward off both feet first
            k.append((f(0.32), st(th=55, py=0.95 * H0, pz=0.30 * trav, fw=0.6, fz=-0.4 * L, fy=0.25 * H0, arm=-150, fore=10, spine=5), 'out'))
        k += [(f(0.42), st(th=95, py=0.50 * H0, pz=0.45 * trav, fw=0.5, fz=-0.5 * L, fy=0.15 * H0, tk=0.5,
                          hw=1.0, hz=0.25, hx=0.15, arm=-90, spine=55, head=45), 'fall' if hop else 'smooth'),
              (f(0.52), st(th=180, py=0.42 * H0, pz=0.55 * trav, tk=1.0, arm=-60, fore=90, spine=80, head=50), 'smooth'),
              (f(0.62), st(th=265, py=0.15 * H0, pz=0.68 * trav, tk=1.0, arm=-50, fore=90, spine=75, head=45), 'smooth'),
              (f(0.76), st(th=320, py=0.32 * H0, pz=0.85 * trav, tk=0.7, fa=0.9 * trav, arm=-70, fore=60, spine=35, head=20), 'smooth'),
              (f(0.88), st(th=345, py=0.62 * H0, pz=0.95 * trav, fa=trav, arm=-50, fore=30, spine=12, head=5), 'smooth'),
              (T, st(th=360, pz=trav, fa=trav), 'smooth')]
        return k
    roll_keys_fwd = roll_keys(1.3, 2.2 * H0, False)
    roll_keys_dive = roll_keys(1.8, 3.2 * H0, True)
    out = []
    for name, keys, note in (
            ('knockdown_back', back_keys, 'knocked down onto the back: snap back, knees buckle, topple, small bounce, lie still'),
            ('knockdown_front', front_keys, 'knocked down onto the stomach: knees give, pitch over, hands brace, land'),
            ('getup_back', getup_back_keys, 'get up from the back: prop on the hands, sit up, feet in, rise, stand'),
            ('getup_front', getup_front_keys, 'get up from the stomach: push up to hands and knees, kneel, stand'),
            ('roll_forward', roll_keys_fwd, 'forward roll from standing: crouch, hands down, tuck, over the back, rise'),
            ('roll_dive', roll_keys_dive, 'dive roll: spring forward, land on the hands, tuck, roll, rise')):
        tr = Track(keys)
        length = tr.keys[-1][0] + max(LAG.values()) + 0.1      # the trailing parts settle before the clip ends
        out.append(ClipDef(name, length, (lambda tr: (lambda t, T: spec_from(lambda lag: tr.at(t - lag))))(tr), False, 0.0, note))
    return out
