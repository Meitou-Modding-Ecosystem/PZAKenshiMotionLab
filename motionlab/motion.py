# PZA's Kenshi Motion Lab - the motion driver: poses a rig from code every frame. Author: PZA. GPLv3.
#
# A pose is built in four passes on top of the rest pose:
#   1. procedural rotations (root bob / pitch, body, head, abdomen, wings: spread + flap + twist + shrink scale)
#   2. forward kinematics
#   3. springs: antennae and the abdomen follow their procedural pose with lag (damped springs on the joints)
#   4. legs: foot targets from the gait (stance: planted, sliding back under an in-place body; swing: lifted arc
#      forward), solved with FABRIK from the carried rest pose (keeps each knee on its own side), then aimed.
# Bones keep their lengths; only the root (and body) translate; only wing roots scale.
import numpy as np
from . import mathx as mx


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0); return x * x * (3 - 2 * x)


class Rig:
    def __init__(self, sk, rigmap):
        self.sk = sk; self.map = rigmap; n = len(sk)
        self.n = n; self.names = sk.names; self.parents = sk.parents; self.order = sk.order()
        self.rest_t = np.array(sk.rest_t, float); self.rest_q = np.array([mx.norm(q) for q in sk.rest_q])
        self.rest_s = np.array(sk.rest_s, float)
        self.Wrest = sk.world_rest()
        self.qrest_w = np.array([mx.m3_to_q(w[:3, :3] / np.linalg.norm(w[:3, :3], axis=0)) for w in self.Wrest])
        self.idx = {nm: i for i, nm in enumerate(sk.names)}
        tails = rigmap.get('tails', {})
        self.dir_local = np.zeros((n, 3)); self.length = np.zeros(n)
        for i, nm in enumerate(sk.names):
            h = self.Wrest[i][:3, 3]
            tl = np.array(tails[nm]) if nm in tails else None
            if tl is None:
                ch = sk.children(i)
                tl = self.Wrest[ch[0]][:3, 3] if ch else h + self.Wrest[i][:3, 1] * 0.05
            d = tl - h; L = np.linalg.norm(d)
            self.length[i] = L
            self.dir_local[i] = mx.q_rotate(mx.q_conj(self.qrest_w[i]), d / max(L, 1e-9))
        self.rest_tail = {nm: np.array(tails[nm]) for nm in tails}
        H = max(1e-6, max(w[:3, 3][1] for w in self.Wrest))
        self.height = H

    def b(self, name): return self.idx[name]

    def fk(self, q, t, s):
        W = np.empty((self.n, 4, 4))
        for i in self.order:
            L = mx.trs(t[i], q[i], s[i]); p = self.parents[i]
            W[i] = W[p] @ L if p >= 0 else L
        return W

    def local_from_world_delta(self, i, D):
        """Local rotation of bone i that applies world-space rotation D (about its head) to its rest pose,
        expressed so it still follows its parent's motion."""
        p = self.parents[i]
        qpr = self.qrest_w[p] if p >= 0 else mx.q_identity()
        return mx.norm(mx.q_mul(mx.q_mul(mx.q_conj(qpr), D), self.qrest_w[i]))

    def aim_chain(self, chain, pts, q, t, s, W):
        """Turn each bone of the chain (minimal swing from where its parent carries it) to point at the next point."""
        for k, b in enumerate(chain):
            p = self.parents[b]
            if p >= 0:
                Wp = W[p]; R = Wp[:3, :3] / np.linalg.norm(Wp[:3, :3], axis=0); qp = mx.m3_to_q(R)
            else:
                Wp = np.eye(4); qp = mx.q_identity()
            base = mx.q_mul(qp, q[b])
            carried = mx.q_rotate(base, self.dir_local[b])
            want = pts[k + 1] - pts[k]
            if np.linalg.norm(want) > 1e-9:
                qw = mx.q_mul(mx.q_between(carried, want), base)
                q[b] = mx.norm(mx.q_mul(mx.q_conj(qp), qw))
            W[b] = Wp @ mx.trs(t[b], q[b], s[b])
            for c in self._desc(b):
                pc = self.parents[c]; W[c] = W[pc] @ mx.trs(t[c], q[c], s[c])

    def _desc(self, b):
        out = []; stack = list(self.sk.children(b))
        while stack:
            c = stack.pop(0); out.append(c); stack.extend(self.sk.children(c))
        return out

    def chain_points(self, chain, W):
        pts = [W[chain[0]][:3, 3]]
        for b in chain:
            R = W[b][:3, :3]
            pts.append(W[b][:3, 3] + R @ (self.dir_local[b] * self.length[b]))
        return np.array(pts)


def _rotate_about(points, pivot, qrot):
    R = mx.q_to_m3(qrot)
    return (np.asarray(points) - pivot) @ R.T + pivot


def leg_ik(pts, target, up=np.array([0.0, 1.0, 0.0]), knee=None):
    """Leg IK the way legs move: the whole leg swings at the hip (about the body's up axis) to face the target,
    then an upper part (hip..knee) and a lower part (knee..foot), each kept rigid, bend at the knee in the leg's own
    plane (the knee keeps the side it bends to at rest). Bones keep their lengths; the knee visibly works and the foot
    keeps its angle to the shin (no pivoting at the ankle)."""
    pts = np.array(pts, float); n = len(pts) - 1
    if n < 2: return fabrik(pts, np.linalg.norm(np.diff(pts, axis=0), axis=1), target)
    k = knee if knee is not None else (n + 1) // 2
    hip = pts[0].copy()
    # 1. hip swing: turn the leg about the vertical through the hip so the foot faces the target
    def flat(v): return v - up * (v @ up)
    a0, a1 = flat(pts[-1] - hip), flat(target - hip)
    if np.linalg.norm(a0) > 1e-6 and np.linalg.norm(a1) > 1e-6:
        pts = _rotate_about(pts, hip, mx.q_between(a0, a1))
    # 2. knee: two rigid parts, law of cosines in the plane of hip, knee and target
    La = np.linalg.norm(pts[k] - hip); Lb = np.linalg.norm(pts[-1] - pts[k])
    v = target - hip; d = np.linalg.norm(v)
    if d < 1e-9: return pts
    d = np.clip(d, abs(La - Lb) + 1e-4, La + Lb - 1e-4)
    u = v / np.linalg.norm(v)
    side = (pts[k] - hip) - u * ((pts[k] - hip) @ u)
    if np.linalg.norm(side) < 1e-6: side = up - u * (up @ u)
    nrm = side / np.linalg.norm(side)
    ca = np.clip((La * La + d * d - Lb * Lb) / (2 * La * d), -1, 1)
    knee_new = hip + La * (ca * u + np.sqrt(1 - ca * ca) * nrm)
    pts = _rotate_about(pts, hip, mx.q_between(pts[k] - hip, knee_new - hip))     # the lower leg rides along
    foot_new = hip + u * d
    pts[k:] = _rotate_about(pts[k:], pts[k], mx.q_between(pts[-1] - pts[k], foot_new - pts[k]))
    return pts


def fabrik(pts, lens, target, iters=16):
    pts = pts.copy(); base = pts[0].copy(); n = len(lens)
    if np.linalg.norm(target - base) >= lens.sum() - 1e-6:
        d = mx.norm(target - base)
        for i in range(n): pts[i + 1] = pts[i] + d * lens[i]
        return pts
    for _ in range(iters):
        pts[-1] = target
        for i in range(n - 1, -1, -1): pts[i] = pts[i + 1] + mx.norm(pts[i] - pts[i + 1]) * lens[i]
        pts[0] = base
        for i in range(n): pts[i + 1] = pts[i] + mx.norm(pts[i + 1] - pts[i]) * lens[i]
        if np.linalg.norm(pts[-1] - target) < 1e-4: break
    return pts


class Spring:
    """Damped springs on a chain's joints, following target points (the procedural pose) with lag."""

    def __init__(self, k, c):
        self.k, self.c = k, c; self.p = None; self.v = None

    def step(self, targets, lens, dt, sub=4):
        if self.p is None:
            self.p = targets.copy(); self.v = np.zeros_like(targets)
        h = dt / sub
        for _ in range(sub):
            a = self.k * (targets - self.p) - self.c * self.v
            a[0] = 0
            self.v += a * h; self.p += self.v * h
            self.p[0] = targets[0]; self.v[0] = 0
            for i in range(len(lens)):
                d = self.p[i + 1] - self.p[i]; L = np.linalg.norm(d)
                if L > 1e-9: self.p[i + 1] = self.p[i] + d / L * lens[i]
        return self.p


def leg_phases(rigmap):
    """Alternating gait phase per leg: neighbours along a side and across the body are half a cycle apart
    (tripod for six legs, trot for four, alternating pairs for eight, walk for two)."""
    ph = {}
    for e in rigmap['legs']:
        n = int(''.join(ch for ch in e['name'].split('_')[1] if ch.isdigit()) or 1)
        side = 0 if e['side'] == 'L' else 1
        ph[e['name']] = 0.5 * ((n - 1 + side) % 2)
    return ph


# ------------------------------------------------------------------ the driver

class Driver:
    def __init__(self, rig, ground_wings='hidden', spring_antenna=(70.0, 7.0), spring_abdomen=(160.0, 14.0)):
        self.rig = rig; self.map = rig.map; self.ground_wings = ground_wings
        self.phase = leg_phases(self.map)
        self.sa = spring_antenna; self.sb = spring_abdomen
        self.up = np.array([0.0, 1.0, 0.0])
        self.reset()
        r = rig
        self.legs = [(e, [r.b(x) for x in e['bones']]) for e in self.map['legs']]
        self.wings = [(e, [r.b(x) for x in e['bones']]) for e in self.map['wings']]
        self.ants = [(e, [r.b(x) for x in e['bones']]) for e in self.map['antennae']]
        self.abd = [r.b(x) for x in self.map.get('spine', [])]
        Wr = r.Wrest
        self.rest_tip = {e['name']: np.array(e['tip']) for e, _ in self.legs}
        reach = [np.linalg.norm((self.rest_tip[e['name']] - Wr[c[0]][:3, 3])[[0, 2]]) for e, c in self.legs] or [0.5]
        self.reach = float(np.median(reach))
        # the longest stride every planted leg can take (stance speed must be the same for all legs or feet slide);
        # straight-legged rigs need the body lowered a little while walking to get a stride at all
        self._Wr = Wr
        H_ = rig.height; self.crouch = 0.0
        for c in np.linspace(0, 0.15 * H_, 16):
            self.crouch = float(c)
            if self.stride_for(c) >= 0.45 * self.reach: break
        self.max_stride = self.stride_for(self.crouch)
        # standing: lower the body until every planted leg has a little bend (straight legs cannot work their knees)
        self.stand_crouch = 0.0
        for c in np.linspace(0, self.crouch, 12):
            self.stand_crouch = float(c)
            if self.stride_for(c) >= 0.15 * self.reach: break
        # running: crouch deeper (up to a quarter of the height) for a real stride instead of a shuffle
        self.run_crouch = self.crouch
        for c in np.linspace(self.crouch, 0.25 * H_, 16):
            self.run_crouch = float(c)
            if self.stride_for(c) >= 0.8 * self.reach: break

    def stride_for(self, crouch):
        """Longest stride every planted leg can take with the body lowered by `crouch` (stance speed must be the
        same for all legs or feet slide)."""
        rig = self.rig; Wr = self._Wr
        if True:
            lim = []
            for e, ch in self.legs:
                if e.get('lifted'): continue
                hip = Wr[ch[0]][:3, 3] - np.array([0, crouch, 0]); tip = self.rest_tip[e['name']]
                P0 = rig.chain_points(ch, Wr); kk = (len(ch) + 1) // 2
                Lt = 0.98 * (np.linalg.norm(P0[kk] - P0[0]) + np.linalg.norm(P0[-1] - P0[kk]))   # leg_ik's reach
                for sgn in (1, -1):
                    lo_, hi_ = 0.0, 2.0 * Lt
                    for _ in range(40):
                        mid = 0.5 * (lo_ + hi_)
                        if np.linalg.norm(tip + np.array([0, 0, sgn * mid]) - hip) <= Lt: lo_ = mid
                        else: hi_ = mid
                    lim.append(lo_)
            return 2.0 * min(lim) if lim else 0.5

    def reset(self):
        self.springs = {}

    def _spring(self, key, kc):
        if key not in self.springs: self.springs[key] = Spring(*kc)
        return self.springs[key]

    def pose(self, spec, dt=1 / 30):
        """spec: dict from a clip (see clips.py). Returns (q, t, s, W)."""
        r = self.rig
        q = r.rest_q.copy(); t = r.rest_t.copy(); s = r.rest_s.copy()
        root = r.b(self.map['root']); body = r.b(self.map['body']); head = r.b(self.map['head'])
        t[root] = r.rest_t[root] + np.asarray(spec.get('root_offset', (0, 0, 0)))
        if 'root_rot' in spec: q[root] = mx.norm(mx.q_mul(spec['root_rot'], r.rest_q[root]))
        if 'body_rot' in spec: q[body] = r.local_from_world_delta(body, spec['body_rot'])
        if 'body_offset' in spec: t[body] = r.rest_t[body] + spec['body_offset']
        if 'head_rot' in spec: q[head] = r.local_from_world_delta(head, spec['head_rot'])
        for k, b in enumerate(self.abd):
            if 'abdomen_rot' in spec and k < len(spec['abdomen_rot']):
                q[b] = r.local_from_world_delta(b, spec['abdomen_rot'][k])
        # wings
        w = spec.get('wings', {'scale': 0.0 if self.ground_wings == 'hidden' else 1.0, 'spread': 0, 'flap': 0, 'twist': 0})
        for e, ch in self.wings:
            sg = 1.0 if e['side'] == 'L' else -1.0
            rest_dir = mx.norm(np.array(e['tip']) - r.Wrest[ch[0]][:3, 3])
            spread_dir = mx.norm(np.array([sg * 1.0, 0.22, -0.30]))
            D = mx.q_slerp(mx.q_identity(), mx.q_between(rest_dir, spread_dir), float(w.get('spread', 0)))
            cur = mx.q_rotate(D, rest_dir)
            D = mx.q_mul(mx.q_axis_angle(cur, sg * w.get('twist', 0.0)), D)
            D = mx.q_mul(mx.q_axis_angle([0, 0, 1], sg * w.get('flap', 0.0)), D)
            q[ch[0]] = r.local_from_world_delta(ch[0], D)
            sc = max(1e-3, float(w.get('scale', 1.0)))
            s[ch[0]] = np.array([sc, sc, sc])
        W = r.fk(q, t, s)
        # springs: antennae and abdomen follow their procedural pose with lag
        dtt = spec.get('dt', dt)
        for key, ch, kc, extra in [('ant_' + e['name'], c, self.sa, spec.get('antenna_twitch')) for e, c in self.ants] + \
                                  ([('abdomen', self.abd, self.sb, None)] if self.abd else []):
            if extra is not None:
                for k, b in enumerate(ch):
                    q[b] = mx.norm(mx.q_mul(q[b], mx.q_axis_angle([1, 0, 0], extra * (0.6 + 0.4 * k))))
                W = r.fk(q, t, s)
            tg = r.chain_points(ch, W)
            lens = np.array([r.length[b] for b in ch])
            pts = self._spring(key, kc).step(tg, lens, dtt)
            r.aim_chain(ch, pts, q, t, s, W)
        # legs
        feet = spec.get('feet', {}); fold = float(spec.get('fold', 0.0))
        for e, ch in self.legs:
            tgt = feet.get(e['name'])
            if tgt is None and fold <= 0: continue
            if fold > 0 and tgt is not None: tgt = np.asarray(tgt)
            pts0 = r.chain_points(ch, W)
            if fold > 0:                                          # legs tucked under the body (flight)
                hip, tip = pts0[0], pts0[-1]
                ft = hip + (tip - hip) * np.array([0.6, 0.75, 0.6]) + np.array([0, 0.05 * r.height, 0])
                tgt = ft if tgt is None else (1 - fold) * np.asarray(tgt) + fold * ft
            lens = np.array([r.length[b] for b in ch])
            pts = leg_ik(pts0, np.asarray(tgt, float), self.up)
            r.aim_chain(ch, pts, q, t, s, W)
        return q, t, s, W


# ------------------------------------------------------------------ clips

class ClipDef:
    def __init__(self, name, length, fn, loop=True, speed=0.0, note=''):
        self.name, self.length, self.fn, self.loop, self.speed, self.note = name, length, fn, loop, speed, note


def gait_feet(drv, phase, stride, lift, duty, ground=0.0):
    """Foot targets; feet['_stance'] lists the legs that are planted this frame (for the slip check)."""
    feet = {'_stance': set()}
    for e, ch in drv.legs:
        p = (phase + drv.phase[e['name']]) % 1.0
        tip = drv.rest_tip[e['name']].copy()
        if e.get('lifted'): continue                             # lifted legs ride along with the body
        if p < duty:
            u = p / duty; off = stride * (0.5 - u); h = 0.0; feet['_stance'].add(e['name'])
        else:
            u = (p - duty) / (1 - duty); off = stride * (-0.5 + smoothstep(u)); h = lift * np.sin(np.pi * u)
        feet[e['name']] = np.array([tip[0], ground + max(tip[1], 0.0) + h, tip[2] + off])
    return feet


def standard_clips(drv, ground_wings='hidden'):
    H = drv.rig.height; reach = drv.reach
    hover_h = 0.55 * H
    flap_hz = 7.5
    hidden = 0.0 if ground_wings == 'hidden' else 1.0
    gw = {'scale': hidden, 'spread': 0.0, 'flap': 0.0, 'twist': 0.0}
    def ax(a, deg): return mx.q_axis_angle(a, np.radians(deg))
    X, Y, Z = [1, 0, 0], [0, 1, 0], [0, 0, 1]

    def idle(t, T):
        w = 2 * np.pi * t / T
        return {'root_offset': (0, -drv.stand_crouch + 0.012 * H * np.sin(w), 0), 'body_rot': ax(X, 1.5 * np.sin(w)),
                'head_rot': mx.q_mul(ax(Y, 6 * np.sin(w * 2 + 0.7)), ax(X, 3 * np.sin(w))),
                'abdomen_rot': [ax(X, 3 * np.sin(w + 0.8)), ax(X, 2 * np.sin(w + 1.2))],
                'antenna_twitch': np.radians(8) * np.sin(w * 3),
                'feet': gait_feet(drv, 0.0, 0.0, 0.0, 1.0), 'wings': gw}

    def walker(stride, lift, duty, bob, pitch, lean_head, crouch):
        def fn(t, T):
            ph = t / T
            w = 2 * np.pi * ph
            return {'root_offset': (0, -crouch + bob * H * np.cos(2 * w), 0),
                    'body_rot': mx.q_mul(ax(X, pitch), ax(Z, 2.0 * np.sin(w))),
                    'head_rot': ax(Y, lean_head * np.sin(w)),
                    'abdomen_rot': [ax(Y, 4 * np.sin(w - 0.6)), ax(X, 2 * np.cos(2 * w))],
                    'feet': gait_feet(drv, ph, stride, lift, duty), 'wings': gw}
        return fn

    def hover(t, T):
        w = 2 * np.pi * t / T; f = 2 * np.pi * flap_hz * t
        return {'root_offset': (0, hover_h + 0.04 * H * np.sin(w), 0), 'body_rot': ax(X, -8 + 3 * np.sin(w)),
                'head_rot': ax(X, 4 * np.sin(w + 1)),
                'abdomen_rot': [ax(X, -10 + 4 * np.sin(w + 0.5)), ax(X, -6 + 3 * np.sin(w + 0.9))],
                'wings': {'scale': 1.0, 'spread': 1.0, 'flap': np.radians(55) * np.sin(f), 'twist': np.radians(25) * np.cos(f)},
                'fold': 1.0}

    def takeoff(t, T):
        g = smoothstep((t - 0.10) / (T - 0.15))                  # body rise
        wg = smoothstep(t / 0.25)                                 # wings grow in 0.25 s
        f = 2 * np.pi * flap_hz * t
        feet = gait_feet(drv, 0.0, 0.0, 0.0, 1.0)
        return {'root_offset': (0, hover_h * g - (drv.stand_crouch + 0.04 * H * np.sin(np.pi * min(t / 0.2, 1))) * (1 - g), 0), 'fold': g,
                'body_rot': ax(X, -8 * g), 'abdomen_rot': [ax(X, -10 * g), ax(X, -6 * g)],
                'wings': {'scale': wg, 'spread': 1.0, 'flap': np.radians(55) * np.sin(f) * wg, 'twist': np.radians(25) * np.cos(f) * wg},
                'feet': feet}

    def land(t, T):
        g = 1 - smoothstep(t / (T - 0.25))                        # body comes down
        wg = 1 - smoothstep((t - (T - 0.25)) / 0.25)              # wings shrink in the last 0.25 s
        f = 2 * np.pi * flap_hz * t
        feet = gait_feet(drv, 0.0, 0.0, 0.0, 1.0)
        return {'root_offset': (0, hover_h * g - drv.stand_crouch * (1 - g), 0), 'body_rot': ax(X, -8 * g), 'fold': g,
                'abdomen_rot': [ax(X, -10 * g), ax(X, -6 * g)],
                'wings': {'scale': wg if ground_wings == 'hidden' else 1.0, 'spread': g if ground_wings != 'hidden' else 1.0,
                          'flap': np.radians(55) * np.sin(f) * (g if ground_wings != 'hidden' else wg),
                          'twist': np.radians(25) * np.cos(f) * wg}, 'feet': feet}

    def bite(t, T):
        # rear back with the head up, then lunge and snap the head down (mandibles first), feet stay planted
        wind = smoothstep(t / 0.30) * (1 - smoothstep((t - 0.30) / 0.10))
        snap = smoothstep((t - 0.30) / 0.10) * (1 - smoothstep((t - 0.55) / 0.40))
        return {'root_offset': (0, -drv.stand_crouch - 0.6 * drv.crouch * snap, -0.03 * H * wind + 0.05 * H * snap),
                'body_rot': ax(X, -6 * wind + 4 * snap),
                'head_rot': ax(X, -24 * wind + 34 * snap),
                'abdomen_rot': [ax(X, -6 * wind + 8 * snap), ax(X, 4 * snap)],
                'antenna_twitch': np.radians(-18) * wind + np.radians(12) * snap,
                'feet': gait_feet(drv, 0.0, 0.0, 0.0, 1.0), 'wings': gw}

    def sting(t, T):
        # short wind-up (sit back, abdomen up), a fast jab with the abdomen swung down and forward from the waist so
        # the tip thrusts out under the head, a short hold, recovery; feet stay planted
        wind = smoothstep(t / 0.30) * (1 - smoothstep((t - 0.30) / 0.08))
        jab = smoothstep((t - 0.30) / 0.10) * (1 - smoothstep((t - 0.58) / 0.42))
        lunge = smoothstep((t - 0.30) / 0.10) * (1 - smoothstep((t - 0.50) / 0.35))
        return {'root_offset': (0, -drv.stand_crouch - 0.6 * drv.crouch * jab, -0.04 * H * wind + 0.04 * H * lunge),
                'body_rot': ax(X, -6 * wind + 8 * jab),
                'head_rot': ax(X, -6 * wind + 10 * jab),
                'abdomen_rot': [ax(X, 14 * wind - 50 * jab), ax(X, 6 * wind - 28 * jab), ax(X, -12 * jab)],
                'antenna_twitch': np.radians(-15) * wind + np.radians(10) * jab,
                'feet': gait_feet(drv, 0.0, 0.0, 0.0, 1.0), 'wings': gw}

    def front_free(feet):                                        # the front pair rises with the body (forelegs up)
        out = dict(feet); out['_stance'] = set(feet['_stance'])
        for e, ch in drv.legs:
            if e['name'].endswith('1') and e['name'] in out:
                del out[e['name']]; out['_stance'].discard(e['name'])
        return out

    walk_T, walk_stride = 1.0, min(0.55 * reach, 0.8 * drv.max_stride)
    run_T, run_stride = 0.6, min(0.9 * reach, 0.85 * drv.stride_for(drv.run_crouch))
    return [
        ClipDef('idle', 2.4, idle, True, 0.0, 'breathing, head and antennae look around, feet planted'),
        ClipDef('walk', walk_T, walker(walk_stride, 0.10 * H, 0.55, 0.008, 2.0, 5, drv.crouch), True, walk_stride / (0.55 * walk_T),
                'alternating gait, feet planted in stance'),
        ClipDef('run', run_T, walker(run_stride, 0.12 * H, 0.45, 0.012, 7.0, 3, drv.run_crouch), True, run_stride / (0.45 * run_T),
                'longer strides from a deeper crouch, body pitched forward'),
        ClipDef('hover', 1.2, hover, True, 0.0, 'wingbeat %.1f Hz, legs folded, body bobs' % flap_hz),
        ClipDef('takeoff', 0.8, takeoff, False, 0.0, 'wings grow back in 0.25 s while the body lifts'),
        ClipDef('land', 0.8, land, False, 0.0, 'body comes down, wings shrink away in the last 0.25 s'),
        ClipDef('attack', 1.0, bite, False, 0.0, 'bite: rear back head up, lunge and snap the head down, recover'),
        ClipDef('sting', 1.2, sting, False, 0.0, 'sting: wind-up, fast jab with the abdomen swung under and forward, recover'),
    ]
