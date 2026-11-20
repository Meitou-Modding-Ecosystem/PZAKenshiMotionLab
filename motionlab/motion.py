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

ARM_HANG_MAX_DEG = 30.0        # people: an upper arm resting further than this from hanging down is lowered (T-pose)
FOOT_ROLL_MIN = 0.02           # feet roll when a toe sits this far (x height) ahead of its ankle, on ANY foot of the body


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0); return x * x * (3 - 2 * x)


def q_rotvec(v):
    """Quaternion of a rotation vector in degrees (x = pitch, + leans forward / toe down; y = yaw; z = roll)."""
    v = np.asarray(v, float); a = float(np.linalg.norm(v))
    return mx.q_identity() if a < 1e-9 else mx.q_axis_angle(v / a, np.radians(a))


def periodic_curve(knots, vals, x):
    """Smooth closed curve through (knots, vals) on [0, 1) (Catmull-Rom between neighbours, wrapping round)."""
    n = len(knots); x = x % 1.0
    k = int(np.searchsorted(knots, x, side='right') - 1) % n
    k1 = (k + 1) % n; span = (knots[k1] - knots[k]) % 1.0 or 1.0
    u = ((x - knots[k]) % 1.0) / span
    p0, p1, p2, p3 = vals[(k - 1) % n], vals[k], vals[k1], vals[(k + 2) % n]
    return 0.5 * ((2 * p1) + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u * u + (-p0 + 3 * p1 - 3 * p2 + p3) * u ** 3)


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
        self.arms_lowered = []                                    # rule counter: (upper arm, degrees lowered)
        if rigmap.get('plan') == 'biped': self.lower_arms(rigmap.get('arms', []))

    def lower_arms(self, arms, max_off=ARM_HANG_MAX_DEG):
        """Arm motion (swing, measured curves, moves) is a turn away from Kenshi's rest pose, where the arms hang
        straight down. A body exported in a T-pose (arms out to the sides) or an A-pose got the same turn from arms
        out, and its arms flew up over its head in the run. Here an upper arm resting more than max_off degrees from
        hanging is turned down (about the shoulder, the smallest turn) and the driver works from that pose instead.
        Only the driver's rest changes: the skeleton's bind pose and the skin stay as they are."""
        down = np.array([0.0, -1.0, 0.0]); moved = False
        for e in arms:
            bones = [self.idx[x] for x in e.get('bones', []) if x in self.idx]
            if not bones: continue
            up = bones[0]
            nxt = bones[1] if len(bones) > 1 else (self.idx.get(e.get('hand')) if e.get('hand') in self.idx else None)
            d = (self.Wrest[nxt][:3, 3] if nxt is not None else self.Wrest[up][:3, 3] + self.Wrest[up][:3, :3] @ self.dir_local[up]) - self.Wrest[up][:3, 3]
            if np.linalg.norm(d) < 1e-9: continue
            d = d / np.linalg.norm(d)
            off = float(np.degrees(np.arccos(np.clip(d @ down, -1.0, 1.0))))
            if off <= max_off: continue
            D = mx.q_between(d, down)
            self.rest_q[up] = self.local_from_world_delta(up, D)
            pivot = self.Wrest[up][:3, 3]
            for c in [up] + self._desc(up):                       # tails given in world space turn with the arm
                nm = self.names[c]
                if nm in self.rest_tail: self.rest_tail[nm] = pivot + mx.q_rotate(D, self.rest_tail[nm] - pivot)
            self.Wrest = self.fk(self.rest_q, self.rest_t, self.rest_s)
            self.qrest_w = np.array([mx.m3_to_q(w[:3, :3] / np.linalg.norm(w[:3, :3], axis=0)) for w in self.Wrest])
            self.arms_lowered.append((self.names[up], round(off, 1))); moved = True
        return moved

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

    def set_world_rot(self, b, qw, q, t, s, W):
        p = self.parents[b]
        if p >= 0:
            R = W[p][:3, :3] / np.linalg.norm(W[p][:3, :3], axis=0); qp = mx.m3_to_q(R); Wp = W[p]
        else:
            qp = mx.q_identity(); Wp = np.eye(4)
        q[b] = mx.norm(mx.q_mul(mx.q_conj(qp), qw))
        W[b] = Wp @ mx.trs(t[b], q[b], s[b])
        for c in self._desc(b):
            W[c] = W[self.parents[c]] @ mx.trs(t[c], q[c], s[c])

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


def leg_ik(pts, target, up=np.array([0.0, 1.0, 0.0]), knee=None, bend=None):
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
    Ltot = np.linalg.norm(pts[-1] - hip) + 1e-9
    if np.linalg.norm(a0) > 0.15 * Ltot and np.linalg.norm(a1) > 0.15 * Ltot:     # splayed legs (insects, crabs) turn at the hip
        pts = _rotate_about(pts, hip, mx.q_between(a0, a1))
    # 2. knee: two rigid parts, law of cosines in the plane of hip, knee and target
    La = np.linalg.norm(pts[k] - hip); Lb = np.linalg.norm(pts[-1] - pts[k])
    v = target - hip; d = np.linalg.norm(v)
    if d < 1e-9: return pts
    d = np.clip(d, abs(La - Lb) + 1e-4, La + Lb - 1e-4)
    u = v / np.linalg.norm(v)
    side = (pts[k] - hip) - u * ((pts[k] - hip) @ u)
    if bend is not None:                         # the leg's own bend direction (knee forward, elbow back): always
        b = np.asarray(bend, float); sb = b - u * (b @ u)  # (the rest knee's side flips with the target on straight legs)
        if np.linalg.norm(sb) > 1e-6: side = sb
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
        self.knee_floor_hits = 0; self.hand_releases = 0; self.body_floor_lifts = 0; self.swing_caps = 0; self.knee_floor_lifts = 0   # rule counters
        self.hip_caps = 0
        self._upper = None
        self.phase = leg_phases(self.map)
        self.sa = spring_antenna; self.sb = spring_abdomen
        self.up = np.array([0.0, 1.0, 0.0])
        self.reset()
        r = rig
        self.legs = [(e, [r.b(x) for x in e['bones']]) for e in self.map['legs']]
        self.wings = [(e, [r.b(x) for x in e['bones']]) for e in self.map['wings']]
        self.ants = [(e, [r.b(x) for x in e['bones']]) for e in self.map['antennae']]
        self.abd = [r.b(x) for x in self.map.get('spine', [])]
        self.arms = [(e, [r.b(x) for x in e['bones']]) for e in self.map.get('arms', [])]
        self.jaw = r.b(self.map['jaw']) if self.map.get('jaw') in r.idx else None
        self.plan = self.map.get('plan', 'insect')
        prof = self.map.get('gait_profile') or {}
        km = [prof[k]['knee_deg'][1] for k in ('walk', 'run') if k in prof]
        self.knee_max = (max(km) + 5.0) if km else None        # never fold a knee further than the clips do
        Wr = r.Wrest
        self.rest_tip = {e['name']: np.array(e['tip']) for e, _ in self.legs}
        # feet that roll (people): the toe bone under the foot, the ball (toe joint) and an estimated heel point, each
        # as an offset from the ankle at rest; the foot pivots about the heel at heel strike and the ball at toe-off
        self.toe, self.pivots = {}, {}
        cand = {}
        for e, _ in self.legs:
            fb = e.get('foot')
            if fb not in r.idx: continue
            kids = r.sk.children(r.b(fb))
            if not kids: continue
            tb = kids[0]; cand[e['name']] = (tb, Wr[tb][:3, 3] - self.rest_tip[e['name']])
        # whether feet roll is decided for the whole body, not foot by foot: a model a little lopsided (one spike foot
        # 3.7 cm ahead of its ankle, the other 3.1 cm, either side of the cut-off) rolled one foot and not the other,
        # and the gait built for rolling feet dragged the stiff one (1.2 m/s slip in the run)
        roll_body = any(ball[2] > FOOT_ROLL_MIN * rig.height for tb, ball in cand.values())   # a forward toe: a person's foot
        self.feet_unrolled = 0                                    # rule counter: feet that roll only because their pair does
        for nm, (tb, ball) in cand.items():
            if not roll_body or ball[2] <= 0.0: continue
            if ball[2] <= FOOT_ROLL_MIN * rig.height: self.feet_unrolled += 1
            self.toe[nm] = tb
            self.pivots[nm] = {'ball': ball, 'heel': np.array([0.0, ball[1], -0.35 * ball[2]])}
        reach = [np.linalg.norm((self.rest_tip[e['name']] - Wr[c[0]][:3, 3])[[0, 2]]) for e, c in self.legs] or [0.5]
        self.reach = float(np.median(reach))
        lens = [np.linalg.norm(np.diff(r.chain_points(c, Wr), axis=0), axis=1).sum() for e, c in self.legs] or [1.0]
        self.leg_len = float(np.median(lens))
        self.leg_lens = {e['name']: float(l) for (e, c), l in zip(self.legs, lens)}   # each leg's own (a dog's front legs are short)
        # Kenshi's animals hang their front legs off the NECK (Bip01 Neck -> Clavicle -> UpperArm): turning the neck there
        # swings the shoulders and lifts the planted front paws (the dog's bite: 3 cm off the ground, sliding 1.2 m/s),
        # so on such a body the whole head turn goes on the head bone
        nk = r.idx.get(self.map.get('neck')) if self.map.get('neck') in r.idx else None
        def _under(b, anc):
            while b >= 0:
                if b == anc: return True
                b = r.parents[b]
            return False
        self.neck_carries_legs = nk is not None and any(_under(c[0], nk) for e, c in self.legs)
        if self.reach < 0.3 * self.leg_len: self.reach = 0.6 * self.leg_len      # upright legs (feet under the hips)
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

    def reach_shift(self, want_z, drop=0.0, slack=0.98):
        """How far (up to want_z, signed) the body can shift forward / back with every planted foot at its rest place
        still in reach of its own hip, with the body lowered by `drop` (leg by leg: a dog's front legs are short)."""
        lo_, hi_ = 0.0, 1.0
        def ok(f):
            for e, ch in self.legs:
                if e.get('lifted'): continue
                P0 = self.rig.chain_points(ch, self._Wr); kk = (len(ch) + 1) // 2
                L = slack * (np.linalg.norm(P0[kk] - P0[0]) + np.linalg.norm(P0[-1] - P0[kk]))
                hip = P0[0] + np.array([0.0, -drop, f * want_z])
                if np.linalg.norm(self.rest_tip[e['name']] - hip) > L: return False
            return True
        if ok(1.0): return want_z
        for _ in range(30):
            mid = 0.5 * (lo_ + hi_)
            if ok(mid): lo_ = mid
            else: hi_ = mid
        return lo_ * want_z

    def vault(self, feet, slack=0.985, rot=None):
        """Highest root offset at which every planted foot is still within reach of its hip: the inverted-pendulum
        vault of a walk (hips rise over a straight stance leg, drop at double support). None when no foot is down.
        rot: the root's turn this frame (the hips swing with the pelvis)."""
        st = feet.get('_stance', set()); best = None
        rootp = self._Wr[self.rig.b(self.map['root'])][:3, 3]
        for e, ch in self.legs:
            if e['name'] not in st: continue
            P0 = self.rig.chain_points(ch, self._Wr); kk = (len(ch) + 1) // 2
            L = slack * (np.linalg.norm(P0[kk] - P0[0]) + np.linalg.norm(P0[-1] - P0[kk]))
            hip = P0[0] if rot is None else rootp + mx.q_rotate(rot, P0[0] - rootp)
            f = np.asarray(feet[e['name']])
            dh = (f - hip)[[0, 2]]; h2 = L * L - float(dh @ dh)
            if h2 <= 0: continue
            off = (f[1] + np.sqrt(h2)) - hip[1]
            best = off if best is None else min(best, off)
        return best

    def reset(self):
        self.springs = {}; self.knee_floor_hits = 0; self.hand_releases = 0; self.body_floor_lifts = 0; self.swing_caps = 0; self.knee_floor_lifts = 0
        self.hip_caps = 0

    def _placed_arms(self, spec, W):
        """Sides whose hand the arm solver will place this frame: a reference clip's hands, and hands put down that
        are within reach of their shoulder (a hand out of reach lets go and keeps its swing pose)."""
        out = set(spec.get('hands_rel', {}))
        want = dict(spec.get('hands', {})); want.update({k: None for k in spec.get('hands_down', {})})
        for e, ch in self.arms:
            if e['side'] not in want or len(ch) < 2: continue
            sh = self.rig.chain_points(ch, W)[0]; reach = sum(self.rig.length[b] for b in ch)
            tg = want[e['side']]
            if tg is None:
                hz, hx, hy = spec['hands_down'][e['side']]
                if abs(sh[1] - hy) < 0.98 * reach: out.add(e['side'])
            elif np.linalg.norm(np.asarray(tg, float) - sh) <= 0.98 * reach: out.add(e['side'])
        return frozenset(out)

    def _upper_bones(self, skip=frozenset()):
        """Every bone that is not part of a leg (legs reach the floor on purpose; their own rules keep them out of it),
        nor of an arm on a side in `skip`."""
        if getattr(self, '_upper', None) is None: self._upper = {}
        if skip not in self._upper:
            r = self.rig; skipb = set(b for e, ch in self.legs for b in ch)
            skipb |= set(b for e, ch in self.arms if e['side'] in skip for b in ch)
            out = []
            for b in range(r.n):
                p = b; inside = False
                while p >= 0:
                    if p in skipb: inside = True; break
                    p = r.parents[p]
                if not inside: out.append(b)
            self._upper[skip] = np.array(out, int)
        return self._upper[skip]

    def _spring(self, key, kc):
        if key not in self.springs: self.springs[key] = Spring(*kc)
        return self.springs[key]

    def pose(self, spec, dt=1 / 30):
        """spec: dict from a clip (see standard_clips below). Returns (q, t, s, W)."""
        r = self.rig
        q = r.rest_q.copy(); t = r.rest_t.copy(); s = r.rest_s.copy()
        root = r.b(self.map['root']); body = r.b(self.map['body']); head = r.b(self.map['head'])
        t[root] = r.rest_t[root] + np.asarray(spec.get('root_offset', (0, 0, 0)))
        if 'root_rot' in spec: q[root] = mx.norm(mx.q_mul(spec['root_rot'], r.rest_q[root]))
        if 'body_rot' in spec:
            chain = [r.b(n) for n in self.map.get('spine_chain', []) if n in r.idx]
            if len(chain) > 1:                                    # a bend spreads over the whole back, not one joint
                part = mx.q_slerp(mx.q_identity(), spec['body_rot'], 1.0 / len(chain))
                for b in chain: q[b] = r.local_from_world_delta(b, part)
            else: q[body] = r.local_from_world_delta(body, spec['body_rot'])
        if 'body_offset' in spec: t[body] = r.rest_t[body] + spec['body_offset']
        if 'head_rot' in spec:
            neck = self.map.get('neck')
            if neck in r.idx and not self.neck_carries_legs:      # half in the neck, half in the head
                half = mx.q_slerp(mx.q_identity(), spec['head_rot'], 0.5)
                q[r.b(neck)] = r.local_from_world_delta(r.b(neck), half); q[head] = r.local_from_world_delta(head, half)
            else: q[head] = r.local_from_world_delta(head, spec['head_rot'])
        if self.jaw is not None and spec.get('jaw_open'):
            q[self.jaw] = r.local_from_world_delta(self.jaw, mx.q_axis_angle([1, 0, 0], np.radians(spec['jaw_open'])))
        for e, ch in self.arms:                                   # arms swing about the shoulder's side axis
            a = spec.get('arm_swing', {}).get(e['side'], 0.0)
            q[ch[0]] = r.local_from_world_delta(ch[0], mx.q_axis_angle([1, 0, 0], np.radians(a)))
            if len(ch) > 1:
                fb = spec.get('forearm_bend', 0.0)
                if isinstance(fb, dict): fb = fb.get(e['side'], 0.0)
                q[ch[1]] = r.local_from_world_delta(ch[1], mx.q_axis_angle([1, 0, 0], np.radians(-fb)))
        ar = spec.get('arm_rel')
        if ar:                                                    # measured arms: shoulder, upper arm, forearm, hand,
            for e, ch in self.arms:                               # each turned against its parent
                sd = e['side']; up = ch[0]
                parts = [('clav', r.parents[up]), ('upper', up), ('fore', ch[1] if len(ch) > 1 else -1),
                         ('hand', r.b(e['hand']) if e.get('hand') in r.idx else -1)]
                for k, b in parts:
                    qd = ar.get(sd + '_' + k)
                    if qd is not None and b >= 0: q[b] = r.local_from_world_delta(b, qd)
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
        floor = spec.get('body_floor')
        if floor is not None:                                     # head / spine / arms never sink under the floor:
            ub = self._upper_bones(skip=self._placed_arms(spec, W))   # arms their own solver will place don't count
            low = float(W[ub, 1, 3].min())                     # lift the whole body by what is missing
            if low < floor:
                t[root] = t[root] + np.array([0.0, floor - low, 0.0])       # same frame as root_offset
                W = r.fk(q, t, s); self.body_floor_lifts += 1
        # springs: antennae and abdomen follow their procedural pose with lag
        dtt = spec.get('dt', dt)
        for key, ch, kc, extra in [('ant_' + e['name'], c, (600.0, 40.0) if e.get('stiff') else self.sa,
                                    None if e.get('stiff') else spec.get('antenna_twitch')) for e, c in self.ants] + \
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
        # a kneeling knee keeps where the leg solver put it (sideways along the floor); where the hips are too low for
        # the thigh to reach the floor there, the whole body is lifted just enough and the legs are solved again (a
        # knee coming forward under low hips used to flip from behind to in front in one frame)
        q_b, t_b, W_b = q.copy(), t.copy(), W.copy(); lifted = 0.0
        for attempt in range(4):
            need = 0.0
            for e, ch in self.legs:
                tgt = feet.get(e['name'])
                if tgt is None and fold <= 0: continue
                if fold > 0 and tgt is not None: tgt = np.asarray(tgt)
                pts0 = r.chain_points(ch, W)
                if fold > 0:                                          # legs tucked under the body (flight)
                    hip, tip = pts0[0], pts0[-1]
                    ft = hip + (tip - hip) * np.array([0.6, 0.75, 0.6]) + np.array([0, 0.05 * r.height, 0])
                    # a leg with no target of its own (a lifted leg riding along) eases from where the body carries it,
                    # by the same amount as the others: it went straight to the full tuck the instant the fold began
                    # (the bee's back legs: 13 m/s in one frame at take-off and landing)
                    tgt = (1 - fold) * (pts0[-1] if tgt is None else np.asarray(tgt)) + fold * ft
                lens = np.array([r.length[b] for b in ch])
                tgt = np.asarray(tgt, float)
                kmax = self.knee_max
                if kmax is not None:
                    kk = (len(ch) + 1) // 2
                    La = np.linalg.norm(pts0[kk] - pts0[0]); Lb = np.linalg.norm(pts0[-1] - pts0[kk])
                    inner = np.radians(180.0 - kmax)               # the knee's inner angle at its most bent
                    dmin = np.sqrt(La * La + Lb * Lb - 2 * La * Lb * np.cos(inner))
                    v = tgt - pts0[0]; dv = np.linalg.norm(v)
                    on_floor = tgt[1] <= self.rest_tip[e['name']][1] + 0.03 * r.height   # a foot ON the floor, not a swing foot
                    rr = spec.get('root_rot'); vb = mx.q_rotate(mx.q_conj(rr), v) if rr is not None else v   # in the body's frame
                    hd = float(np.hypot(vb[0], vb[2]))
                    if 1e-9 < dv < dmin and not on_floor and vb[1] < 0 and hd < dmin:
                        # a swinging / tucked foot too close under the hip: lower it along the body's own down, never
                        # push it round the hip (that flips it from behind to in front in one frame as it passes under)
                        vb = np.array([vb[0], -np.sqrt(dmin * dmin - hd * hd), vb[2]])
                        tgt = pts0[0] + (mx.q_rotate(rr, vb) if rr is not None else vb)
                        self.swing_caps += 1
                    elif 1e-9 < dv < dmin:
                        capped = pts0[0] + v / dv * dmin
                        if on_floor and capped[1] < tgt[1] and abs(v[1]) < dmin:   # never push it down into the floor: slide it
                            hzv = v.copy(); hzv[1] = 0.0                  # out along the ground at its own height instead
                            n = np.linalg.norm(hzv); hzv = hzv / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])
                            capped = np.array([pts0[0][0], tgt[1], pts0[0][2]]) + hzv * np.sqrt(dmin * dmin - v[1] * v[1])
                        tgt = capped
                turn = spec.get('root_rot')                            # a body lying down bends its knees its own way
                bend = e.get('bend')
                if bend is not None and turn is not None: bend = mx.q_rotate(turn, bend)
                lb = spec.get('leg_bend', {}).get(e['name'])           # a reference clip says where the knee points
                if lb is not None: bend = np.asarray(lb, float)
                pts = leg_ik(pts0, tgt, self.up, bend=bend)
                floor = spec.get('knee_floor')
                if floor is not None and len(pts) == 3 and pts[1][1] < floor:   # kneeling: the knee rests ON the floor
                    La_ = np.linalg.norm(pts0[1] - pts0[0]); Lb_ = np.linalg.norm(pts0[2] - pts0[1])
                    h = pts[1] - pts[0]; h[1] = 0.0; hl = float(np.linalg.norm(h))
                    if hl < La_:                              # hips too low for the thigh to reach the floor at this
                        need = max(need, float(floor) + np.sqrt(La_ * La_ - hl * hl) - pts[0][1])   # place: lift
                    pts = _knee_on_floor(pts, La_, Lb_, float(floor), float(tgt[1]), tgt)
                    self.knee_floor_hits += 1
                r.aim_chain(ch, pts, q, t, s, W)
                if e.get('foot') in r.idx:                            # keep the foot / hoof at its rest angle (flat)
                    fq = r.qrest_w[r.b(e['foot'])]
                    if turn is not None and spec.get('feet_follow_body'): fq = mx.q_mul(turn, fq)
                    fd = spec.get('foot_delta', {}).get(e['name'])     # a reference clip turns the foot itself
                    if fd is not None: fq = mx.norm(mx.q_mul(fd, r.qrest_w[r.b(e['foot'])]))
                    fp = feet.get('_pitch', {}).get(e['name'])         # a rolling foot (heel strike .. toe-off)
                    if fp is not None: fq = mx.norm(mx.q_mul(mx.q_axis_angle([1, 0, 0], np.radians(fp)), fq))
                    r.set_world_rot(r.b(e['foot']), fq, q, t, s, W)
                    tb = self.toe.get(e['name']); tp = feet.get('_toe', {}).get(e['name'])
                    if tb is not None and tp is not None:              # the toes bend against the foot (flat at push-off)
                        tq = mx.q_mul(mx.q_axis_angle([1, 0, 0], np.radians((fp or 0.0) + tp)), r.qrest_w[tb])
                        r.set_world_rot(tb, mx.norm(tq), q, t, s, W)
            if need <= 1e-4 or attempt == 3: break
            lifted += need; q, t, W = q_b.copy(), t_b.copy(), W_b.copy()
            t[root] = t[root] + np.array([0.0, lifted, 0.0]); W = r.fk(q, t, s)
        if lifted > 0: self.knee_floor_lifts += 1
        # hands planted on the ground (get-ups): arms solved like legs, elbows bending back
        hands = spec.get('hands', {}); rel = spec.get('hands_rel', {}); down = spec.get('hands_down', {})
        for e, ch in list(self.arms):
            tgt = hands.get(e['side'])
            if (tgt is None and e['side'] not in rel and e['side'] not in down) or len(ch) < 2: continue
            pts0 = r.chain_points(ch, W)
            reach = sum(r.length[b] for b in ch)
            if tgt is None and e['side'] in down:                 # a hand put down on the floor near its shoulder:
                hz, hx, hy = down[e['side']]                      # hz forward / hx outward in arm lengths
                out = 1.0 if e['side'] == 'L' else -1.0
                tgt = np.array([pts0[0][0] + out * hx * reach, hy, pts0[0][2] + hz * reach])
                hands[e['side']] = tgt; spec['hands'] = hands
            if tgt is None:                                       # a reference clip: hand relative to the shoulder,
                v = np.asarray(rel[e['side']], float); n = np.linalg.norm(v)   # in arm lengths (kept just in reach)
                if n > 0.97: v = v * (0.97 / n)
                tgt = pts0[0] + v * reach
                tgt[1] = max(tgt[1], float(spec.get('hand_floor', -1e9)))
                ab = spec.get('arm_bend', {}).get(e['side'])
                pts = leg_ik(pts0, tgt, self.up, bend=np.asarray(ab, float) if ab is not None else None)
                r.aim_chain(ch, pts, q, t, s, W)
                continue
            if np.linalg.norm(np.asarray(tgt, float) - pts0[0]) > 0.98 * reach:   # out of reach: the hand lets go
                hands.pop(e['side'], None); self.hand_releases += 1
                continue
            bend = np.array([0.0, 0.0, -1.0])
            if spec.get('root_rot') is not None: bend = mx.q_rotate(spec['root_rot'], bend)
            w = float(spec.get('elbows_out', 0.0))              # face down: elbows go out to the side, not up
            if w > 0:
                side = np.array([1.0 if e['side'] == 'L' else -1.0, 0.0, 0.0])
                bend = mx.norm((1 - w) * bend + w * side)
            pts = leg_ik(pts0, np.asarray(tgt, float), self.up, bend=bend)
            r.aim_chain(ch, pts, q, t, s, W)
        return q, t, s, W


def _knee_on_floor(pts, La, Lb, floor, ankle_y, target=None):
    """Re-place a two-part leg whose knee went under the floor: the knee sits on the floor in the same direction
    from the hip, the shin points at the foot's own target (so the foot stays where the leg would have put it and
    nothing jumps when the rule switches on or off), the ankle no lower than its target height."""
    hip, knee, ank = (np.asarray(p, float) for p in pts)
    d = hip[1] - floor
    h = knee - hip; h[1] = 0.0                         # the knee keeps where the leg solver put it, along the floor
    hl = np.linalg.norm(h); R = np.sqrt(max(La * La - d * d, 0.0))
    if hl > R and hl > 1e-9: h = h * (R / hl)          # (no farther than the thigh reaches at this hip height)
    k = hip + h; k[1] = floor if d < La else hip[1] - La
    aim = np.asarray(target, float) if target is not None else ank
    sd = aim - k
    if np.linalg.norm(sd) < 1e-9: sd = ank - k
    a = k + mx.norm(sd) * Lb
    if a[1] < ankle_y:                                  # keep the ankle off the floor: tip the shin up to it
        dy = min(ankle_y - k[1], Lb); hzv = sd.copy(); hzv[1] = 0.0
        nn = np.linalg.norm(hzv); hzv = hzv / nn if nn > 1e-9 else -hz
        a = k + hzv * np.sqrt(max(Lb * Lb - dy * dy, 0.0)); a[1] = k[1] + dy
    return np.array([hip, k, a])


# ------------------------------------------------------------------ clips

class ClipDef:
    def __init__(self, name, length, fn, loop=True, speed=0.0, note=''):
        self.name, self.length, self.fn, self.loop, self.speed, self.note = name, length, fn, loop, speed, note


SWING_MATCH = 0.6
HIP_MARGIN = 0.0
PUSH_ON = 1
RUN_PITCH = 20.0              # a person's run leans the chest forward (Kenshi's run, side by side)
RUN_LIFT_PEAK = 0.35           # a run's heel is highest 35% into the swing, behind the body
TURNS_ON = True                # people: pelvis / chest / head turns measured from Kenshi's clips (2026-11-19)
ROLL_ON = True                 # people: feet roll heel strike -> toe-off (2026-11-19)
ARMS_ON = True                 # people: shoulders / arms / hands measured from Kenshi's clips (2026-11-19)
PATH_ON = True                 # people: the ankle follows Kenshi's measured path (stance planted) (2026-11-19)


def roll_knots(roll, duty):
    """Our cycle's knots for a measured foot roll (stance samples over our stance, swing samples over our swing)."""
    n = len(roll['stance']); us = np.linspace(0, 1, n)
    kn = list(us[:-1] * duty) + list(duty + us[:-1] * (1 - duty))
    return (np.array(kn), np.array(list(roll['stance'][:-1]) + list(roll['swing'][:-1]), float),
            np.array(list(roll['toe_stance'][:-1]) + list(roll['toe_swing'][:-1]), float))


def roll_offset(piv, pitch):
    """How far the ankle moves when the foot pitches about its heel (toe up, pitch < 0) or ball (heel up, > 0) while
    that point stays put. Zero at pitch 0, so the switch between the two is seamless."""
    d = piv['ball'] if pitch > 0 else piv['heel']
    return mx.q_rotate(mx.q_axis_angle([1, 0, 0], np.radians(pitch)), -d) + d


def _cr(vals, u):
    """Catmull-Rom through evenly spaced samples on [0, 1] (ends held)."""
    v = np.asarray(vals, float); n = len(v) - 1; x = float(np.clip(u, 0, 1)) * n
    k = min(int(x), n - 1); t = x - k
    p0, p1, p2, p3 = v[max(k - 1, 0)], v[k], v[k + 1], v[min(k + 2, n)]
    return 0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3)


def gait_feet(drv, phase, stride, lift, duty, ground=0.0, centre=0.0, lift_peak=0.5, roll=None, path=None):
    """Foot targets; feet['_stance'] lists the legs that are planted this frame (for the slip check). roll: a measured
    foot roll (profiles._turns): the foot pitches heel strike -> flat -> toe-off about its heel / ball, which stays
    planted (feet['_pitch'], feet['_toe'], feet['_pivot'] say how, for the driver and the slip check)."""
    feet = {'_stance': set()}
    kn = pv = tv = None
    if roll is not None and drv.pivots:
        kn, pv, tv = roll_knots(roll, duty)
        feet['_pitch'], feet['_toe'], feet['_pivot'] = {}, {}, {}
    for e, ch in drv.legs:
        p = (phase + drv.phase[e['name']]) % 1.0
        tip = drv.rest_tip[e['name']].copy()
        if e.get('lifted'): continue                             # lifted legs ride along with the body
        if path is not None:                                     # measured (people): the ankle's own path
            pos = _path_foot(drv, e, tip, p, duty, path, ground, roll, kn, pv, tv, feet)
            if p < duty: feet['_stance'].add(e['name'])
            feet[e['name']] = pos; continue
        if p < duty:
            u = p / duty; off = stride * (0.5 - u); h = 0.0; feet['_stance'].add(e['name'])
        else:
            # the swing leaves and lands at the speed the planted foot slides back with (part SWING_MATCH of it): a
            # swing that starts from a standstill jerks the foot the instant it lifts
            u = (p - duty) / (1 - duty); m = -SWING_MATCH * stride * (1 - duty) / max(duty, 1e-6)
            h00 = 2 * u ** 3 - 3 * u ** 2 + 1; h10 = u ** 3 - 2 * u ** 2 + u; h01 = -2 * u ** 3 + 3 * u ** 2; h11 = u ** 3 - u ** 2
            off = -0.5 * stride * h00 + m * h10 + 0.5 * stride * h01 + m * h11
            # the foot is highest EARLY in the swing (the heel kicks up behind) and comes through low; peaking at
            # mid-swing lifted the knee high in front ('high-knees' running). lift_peak = where in the swing.
            # warp w(u) = u + c u (1 - u), w(lift_peak) = 0.5: smooth, finite slope at both ends (a power curve
            # here yanked the foot up the instant it lifted)
            lp = float(np.clip(lift_peak, 0.27, 0.73)); c = (0.5 - lp) / (lp * (1 - lp))
            # never more than 0.45 of THIS leg's length: one lift for every leg aimed a dog's short front paws at its
            # shoulders, out of reach (the knee rule then snapped them down 33 cm in one frame)
            le = min(lift, 0.45 * drv.leg_lens.get(e['name'], lift / 0.45))
            h = le * np.sin(np.pi * (u + c * u * (1 - u)))
        pos = np.array([tip[0], ground + max(tip[1], 0.0) + h, tip[2] + centre + off])
        if roll is not None and e['name'] in drv.pivots:
            piv = drv.pivots[e['name']]
            fp = float(periodic_curve(kn, pv, p)); feet['_pitch'][e['name']] = fp
            feet['_toe'][e['name']] = float(periodic_curve(kn, tv, p))
            # in the stance the heel / ball really is the pivot (stays planted); in the swing that hold fades out
            # and back (cos^2) so the measured ankle lift is not counted twice
            # (in the air the offset holds its toe-off / heel-strike value while it fades: following a fast-turning
            # foot through 100+ deg there whipped the ankle round, 12 m/s in one frame)
            if p < duty: pos = pos + roll_offset(piv, fp)
            else:
                us = (p - duty) / (1 - duty)                      # only the first / last quarter of the swing
                wgt = float(1 - smoothstep(us / 0.25)) if us < 0.5 else float(smoothstep((us - 0.75) / 0.25))
                hold = float(periodic_curve(kn, pv, duty - 1e-9)) if us < 0.5 else float(periodic_curve(kn, pv, 0.0))
                pos = pos + wgt * roll_offset(piv, hold)
            feet['_pivot'][e['name']] = (piv['ball'] if fp > 0 else piv['heel']) if p < duty else None
        feet[e['name']] = pos
    return feet


def _path_foot(drv, e, tip, p, duty, path, ground, roll, kn, pv, tv, feet):
    """One foot on a path measured from Kenshi's own clip (profiles._turns 'ankle_path', leg lengths against the root /
    the rest ankle): planted in the stance (sliding back at the gait's speed, rolling about its heel / ball, which stay
    put), the measured swing (heel kick, lift, reach) bent just enough to start and end where the stance does."""
    L = drv.leg_len; nm = e['name']; piv = drv.pivots.get(nm)
    rz = drv._Wr[drv.rig.b(drv.map['root'])][2, 3]
    fp = float(periodic_curve(kn, pv, p)) if roll is not None else 0.0
    ro = lambda pitch: roll_offset(piv, pitch) if (piv is not None and roll is not None) else np.zeros(3)
    p_end = float(periodic_curve(kn, pv, duty - 1e-9)) if roll is not None else 0.0
    p_beg = float(periodic_curve(kn, pv, 0.0)) if roll is not None else 0.0
    # the measured ankle ends already include the foot's roll: the planted point slides from where the ankle starts
    # minus its roll there to where it ends minus its roll there (that slide is the true ground speed)
    z0 = rz + path['stance_z'][0] * L - ro(p_beg)[2]; z1 = rz + path['stance_z'][-1] * L - ro(p_end)[2]
    def stance_at(u, pitch):
        return np.array([tip[0], tip[1] + ground, z0 + (z1 - z0) * u]) + ro(pitch)
    if roll is not None and piv is not None:
        feet['_pitch'][nm] = fp; feet['_toe'][nm] = float(periodic_curve(kn, tv, p))
    if p < duty:
        if piv is not None and roll is not None: feet['_pivot'][nm] = piv['ball'] if fp > 0 else piv['heel']
        return stance_at(p / duty, fp)
    if roll is not None: feet['_pivot'][nm] = None
    u = (p - duty) / (1 - duty)
    meas = lambda uu: np.array([tip[0], tip[1] + _cr(path['swing_y'], uu) * L, rz + _cr(path['swing_z'], uu) * L])
    d0 = stance_at(1.0, p_end) - meas(0.0); d1 = stance_at(0.0, p_beg) - meas(1.0)
    sm = smoothstep(u)
    return meas(u) + d0 * (1 - sm) + d1 * sm


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
        return {'arm_swing': {e['side']: 2.0 * np.sin(w + (0 if e['side'] == 'L' else 1.3)) for e, c in drv.arms}, 'forearm_bend': 8.0,
                'root_offset': (0, -drv.stand_crouch + 0.012 * H * np.sin(w), 0), 'body_rot': ax(X, 1.5 * np.sin(w)),
                'head_rot': mx.q_mul(ax(Y, 6 * np.sin(w * 2 + 0.7)), ax(X, 3 * np.sin(w))),
                'abdomen_rot': [ax(X, 3 * np.sin(w + 0.8)), ax(X, 2 * np.sin(w + 1.2))],
                'antenna_twitch': np.radians(8) * np.sin(w * 3),
                'feet': gait_feet(drv, 0.0, 0.0, 0.0, 1.0), 'wings': gw}

    def arm_swing(ph, amp):
        """Each arm swings against the leg on its own side (forward while that foot is back)."""
        out = {}
        for e, ch in drv.arms:
            same = [l for l, c in drv.legs if l['side'] == e['side']]
            p0 = drv.phase[sorted(same, key=lambda l: l['name'])[0]['name']] if same else 0.0
            out[e['side']] = amp * np.cos(2 * np.pi * (ph + p0))
        return out

    def walker(stride, lift, duty, bob, pitch, lean_head, crouch, arm_amp=0.0, fore=0.0, vault=False, air=0.0,
               ymax=0.0, slack=0.985, arm_bias=0.0, fore_amp=0.0, centre=0.0, all_legs=False, lift_peak=0.5, push_from=1.0,
               turns=None, roll=None, hips=None, arms=None, path=None, ground=0.0):
        cache = {}

        def fn(t, T):
            sp = raw(t, T)
            if not vault: return sp
            # the hips' height, smoothed round the cycle (sigma 30 ms): the vault alone sags and then snaps back up at
            # each toe-off. A planted foot the smoothed hips can no longer reach is coming off its toes: it leaves the
            # stance early instead of dragging the hips down.
            if T not in cache:
                N = 360; ys = np.array([raw(k / N * T, T)['root_offset'][1] for k in range(N)])
                sig = max(1.0, 0.03 / T * N); x = np.arange(-int(4 * sig), int(4 * sig) + 1)
                g = np.exp(-0.5 * (x / sig) ** 2); g /= g.sum()
                sm = np.array([float(np.dot(g, ys[(k + x) % N])) for k in range(N)])
                # never above the vault (the knee keeps Kenshi's least bend), but over a few samples: a one-sample
                # dip in the raw vault (no foot counted for an instant) would come straight through as a jolt
                wn = max(2, int(0.012 / T * N)); win = np.arange(-wn, wn + 1)
                dil = np.array([ys[(k + win) % N].max() for k in range(N)])
                cap = np.array([dil[(k + win) % N].min() for k in range(N)])   # fills narrow dips, keeps the peaks
                cache[T] = np.minimum(sm, cap + HIP_MARGIN * drv.leg_len)
                # where the reach limit is lower than the wanted curve (a measured one, or the smoothed one), take the
                # shortfall over the cycle, widen it (30 ms) and smooth it (20 ms), and lower the curve by that: a cap
                # that bites becomes a smooth dip instead of a kink. min(curve, reach) frame by frame cut corners into
                # people's hips and left a 24 cm one-frame step in the dog's run where its flight phase starts.
                if True:
                    if hips is not None:
                        from .profiles import fourier_at
                        yf = np.array([float(fourier_at(hips, k / N)) * drv.leg_len for k in range(N)]); lim = ys
                    else:
                        yf = sm; lim = cap + HIP_MARGIN * drv.leg_len
                    d = np.maximum(yf - lim, 0.0)
                    w2 = max(2, int(0.03 / T * N)); win2 = np.arange(-w2, w2 + 1)
                    dd = np.array([d[(k + win2) % N].max() for k in range(N)])
                    sig2 = max(1.0, 0.02 / T * N); x2 = np.arange(-int(4 * sig2), int(4 * sig2) + 1)
                    g2 = np.exp(-0.5 * (x2 / sig2) ** 2); g2 /= g2.sum()
                    ds = np.array([float(np.dot(g2, dd[(k + x2) % N])) for k in range(N)])
                    cache[T] = np.minimum(yf - ds, lim + 0.002 * drv.leg_len)   # never more than 2 mm past the reach
            tab = cache[T]; N = len(tab); f = (t / T % 1.0) * N; k0 = int(f) % N
            y = float(tab[k0] + (tab[(k0 + 1) % N] - tab[k0]) * (f - int(f)))
            feet = dict(sp['feet']); st = set(feet.get('_stance', ()))
            for nm in list(st):
                one = drv.vault({'_stance': {nm}, nm: feet[nm]}, slack, rot=sp.get('root_rot'))
                if one is None or one < y - 1e-4: st.discard(nm)
            feet['_stance'] = st
            sp = dict(sp); sp['feet'] = feet; sp['root_offset'] = (sp['root_offset'][0], y, sp['root_offset'][2])
            return sp

        def raw(t, T):
            ph = t / T
            w = 2 * np.pi * ph
            feet = gait_feet(drv, ph, stride, lift, duty, ground=ground, centre=centre, lift_peak=lift_peak, roll=roll, path=path)
            rr = None
            if turns is not None:                       # measured: the pelvis turns toward the leading leg, the chest
                from .profiles import fourier_at          # turns against it, the head keeps looking ahead
                tv = {pt: [fourier_at(turns[pt][a], ph) for a in ('pitch', 'yaw', 'roll')] for pt in ('pel', 'chest', 'head')}
                rr = q_rotvec(tv['pel'])
            if vault:                                   # people: hips ride over the stance leg, a hop when no foot is down
                vf = feet
                if all_legs:                            # the swinging leg reaching for its next step counts too
                    vf = dict(feet); vf['_stance'] = {e['name'] for e, c in drv.legs if e['name'] in feet}
                if push_from < 1.0:                     # late in its stance a foot is rolling onto its toes and pushing
                    vf = dict(vf); vf['_stance'] = set(vf['_stance'])   # off: it no longer pulls the hips down
                    for e, c in drv.legs:
                        pp = (ph + drv.phase[e['name']]) % 1.0
                        if pp < duty and pp / duty > push_from: vf['_stance'].discard(e['name'])
                v = drv.vault(vf, slack, rot=rr)
                if hips is not None:                    # measured: Kenshi's own hip height over the cycle; the reach
                    from .profiles import fourier_at      # limit only caps it (and is counted when it does)
                    y = float(fourier_at(hips, ph)) * drv.leg_len
                    if v is not None and v < y: y = v; drv.hip_caps += 1
                elif v is not None: y = min(v, ymax)
                else:                                   # airborne: a small hop, never higher than the feet can follow
                    allf = dict(feet); allf['_stance'] = {e['name'] for e, c in drv.legs if e['name'] in feet}
                    va = drv.vault(allf, rot=rr)
                    y = ymax + air * H * np.sin(np.pi * ((ph * 2) % 1.0))
                    if va is not None: y = min(y, va)
            else:
                y = -crouch + bob * H * np.cos(2 * w)
            sw = {k: v + arm_bias for k, v in arm_swing(ph, arm_amp).items()}
            # the elbow bends most as the arm comes FORWARD (fist up by the chest, Kenshi's run) and opens as it goes
            # back; arm_swing is + = back, so forward is the negative side (this sign was flipped: a straight arm
            # thrown forward, 'Forrest Gump')
            fb = {k: fore - fore_amp * (v - arm_bias) / max(arm_amp, 1e-6) for k, v in sw.items()}
            sp = {'arm_swing': sw, 'forearm_bend': fb, 'root_offset': (0, y, 0),
                  'body_rot': mx.q_mul(ax(X, pitch), ax(Z, 2.0 * np.sin(w))),
                  'head_rot': ax(Y, lean_head * np.sin(w)),
                  'abdomen_rot': [ax(Y, 4 * np.sin(w - 0.6)), ax(X, 2 * np.cos(2 * w))],
                  'feet': feet, 'wings': gw}
            if rr is not None:
                sp.update(root_rot=rr, body_rot=q_rotvec(tv['chest']), head_rot=q_rotvec(tv['head']))
            if arms is not None:
                from .profiles import fourier_at
                sp['arm_rel'] = {k: q_rotvec([fourier_at(v[a], ph) for a in ('pitch', 'yaw', 'roll')]) for k, v in arms.items()}
            return sp
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
        # only as far as EVERY planted foot still reaches (leg by leg: the dog's short front legs lost their paws, which
        # lifted 3 cm and slid at 1.2 m/s, when one stride number stood for all legs)
        lunge = drv.reach_shift(min(0.05 * H, 0.35 * drv.stride_for(drv.stand_crouch + 0.6 * drv.crouch)), drv.stand_crouch + 0.6 * drv.crouch)
        back = -drv.reach_shift(-min(0.03 * H, 0.35 * drv.stride_for(drv.stand_crouch)), drv.stand_crouch)
        return {'root_offset': (0, -drv.stand_crouch - 0.6 * drv.crouch * snap, -back * wind + lunge * snap),
                'body_rot': ax(X, (-6 * wind + 4 * snap) * (0.0 if drv.plan == 'quadruped' else 1.0)),   # quadrupeds: front legs hang off the spine
                'head_rot': ax(X, -24 * wind + 34 * snap),
                'abdomen_rot': [ax(X, -6 * wind + 8 * snap), ax(X, 4 * snap)],
                'antenna_twitch': np.radians(-18) * wind + np.radians(12) * snap,
                'jaw_open': 28 * wind + 6 * snap,
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

    prof = drv.map.get('gait_profile')
    if prof and drv.plan in ('biped', 'quadruped') and ('walk' in prof or 'run' in prof):
        # numbers measured from Kenshi's own walk / run clips for this skeleton (or the median of its kind):
        # cycle, stride, foot lift, pelvis height, straightest knee, arm swing and elbow (motionlab/profiles.py)
        L = drv.leg_len
        def from_prof(g, kind):
            dmin, dmax = (0.45, 0.68) if kind == 'walk' else (0.3, 0.5)
            T = float(g['cycle_s']); duty = float(np.clip(g['duty'], dmin, dmax))
            ymax = -float(g['pelvis_drop'][0]) * L
            # stance slack from the straightest knee in the clip, with this body's own thigh / shin lengths
            km = np.radians(float(g['knee_deg'][0])); sl = []
            for e, ch in drv.legs:
                P = drv.rig.chain_points(ch, drv._Wr); kk = (len(ch) + 1) // 2
                La, Lb = np.linalg.norm(P[kk] - P[0]), np.linalg.norm(P[-1] - P[kk])
                sl.append(np.sqrt(La * La + Lb * Lb + 2 * La * Lb * np.cos(km)) / (La + Lb))
            slack = float(np.clip(np.median(sl), 0.6, 0.995))
            low = max(float(g['pelvis_drop'][1]), 0.03) * L          # the pelvis dips this low to reach the long steps
            stride = min(float(g['stride']) * L, 0.95 * drv.stride_for(low))
            lift = float(g['lift']) * L
            if drv.plan == 'quadruped':      # a four-legged run's measured 'lift' includes the leap of the whole body, which
                lift = min(lift, 0.5 * L)    # the gait adds on its own: the paws were aimed above the shoulders, out of reach
            pitch = 2.0 if kind == 'walk' else (RUN_PITCH if drv.plan == 'biped' else 4.0)
            kw = dict(vault=True, ymax=ymax, slack=slack, centre=float(g.get('foot_centre', 0.0)) * L, all_legs=True,
                      lift_peak=RUN_LIFT_PEAK if kind == 'run' else 0.5,
                      # Kenshi's run plants a foot for ~15% of the stride, ours for 30%+ (the IK needs it): the second
                      # part of our stance is the toe push and no longer holds the hips down
                      push_from=min(1.0, max(0.3, float(g['duty']) / duty)) if (kind == 'run' and PUSH_ON) else 1.0)
            if drv.plan == 'biped' and 'turns' not in g and 'turns_h1' in g:   # no curves file: the summaries
                from .profiles import roll_from_summary                          # (one sine per turn, our own foot roll)
                g = dict(g, turns=g['turns_h1'])
                if 'roll_summary' in g: g['foot_roll'] = roll_from_summary(g['roll_summary'])
                if 'hips_h2' in g: g['pelvis_y'] = g['hips_h2']
                if 'arms_h1' in g: g['arms'] = g['arms_h1']
                # the summary's lift is Kenshi's whole ankle-height range, made on its own (longer) foot path; on our
                # shorter one the heel came within 3 cm of hip height under the summary's hip curve and the knee rule
                # clamped it (a 10 m/s snap): keep the heel a knee-fold below the hips
                lift = min(lift, 0.8 * float(g['lift']) * L)
            if drv.plan == 'biped' and 'turns' in g and TURNS_ON:
                kw.update(turns=g['turns'])
                pitch = float(g['turns']['pel']['pitch'][0] + g['turns']['chest']['pitch'][0])   # the chest's mean lean
            if drv.plan == 'biped' and 'foot_roll' in g and ROLL_ON:
                kw.update(roll=g['foot_roll'])
                if 'pelvis_y' in g:                                 # rolling feet always reach: the shape comes from Kenshi
                    kw.update(hips=g['pelvis_y'], slack=0.998)      # (its legs reached those hips: the cap is only a net)
                kw.update(ground=float(g.get('ankle_sink', 0.0)) * L)   # the planted ankle sits a little low
                if 'ankle_path' in g and PATH_ON:                   # and the feet follow Kenshi's own ankle path, at its
                    ap = g['ankle_path']                            # stance share (so its speed, too)
                    duty = float(g['foot_roll']['duty']); stride = (ap['stance_z'][0] - ap['stance_z'][-1]) * L
                    pv_ = next(iter(drv.pivots.values()), None)     # the planted point's slide (see _path_foot)
                    if pv_ is not None:
                        kn_, pvv, _ = roll_knots(g['foot_roll'], duty)
                        stride -= roll_offset(pv_, float(periodic_curve(kn_, pvv, 0.0)))[2] -                                   roll_offset(pv_, float(periodic_curve(kn_, pvv, duty - 1e-9)))[2]
                    kw.update(path=ap, ground=float(g.get('ankle_sink', 0.0)) * L)
            if drv.plan == 'biped' and 'arms' in g and ARMS_ON and drv.arms: kw.update(arms=g['arms'])
            if 'arm_deg' in g and drv.arms:
                lo_, hi_ = g['arm_deg']; e0, e1 = g.get('elbow_deg', [10, 30])
                kw.update(arm_amp=(hi_ - lo_) / 2, arm_bias=-(hi_ + lo_) / 2 - pitch, fore=(e0 + e1) / 2, fore_amp=(e1 - e0) / 2)
            return T, duty, stride, walker(stride, lift, duty, 0.0, pitch, 2, 0.0, **kw)
        walk_g = prof.get('walk') or prof.get('run'); run_g = prof.get('run') or prof.get('walk')
        walk_T, walk_duty, walk_stride, walk = from_prof(walk_g, 'walk')
        run_T, run_duty, run_stride, run = from_prof(run_g, 'run')
    elif drv.plan == 'biped':
        # people without a profile: Kenshi-like defaults (pelvis a little low, knees never locked)
        L = drv.leg_len
        walk_T, walk_duty, walk_ymax = 1.4, 0.6, -0.03 * L
        walk_stride = min(0.84 * L, 0.95 * drv.stride_for(-walk_ymax + 0.03 * L))
        run_T, run_duty, run_ymax = 0.6, 0.35, -0.066 * L
        run_stride = min(1.0 * L, 0.95 * drv.stride_for(-run_ymax + 0.04 * L))
        walk = walker(walk_stride, 0.22 * L, walk_duty, 0.0, 2.0, 2, 0.0, arm_amp=22, fore=18, fore_amp=9, arm_bias=-1.5,
                      vault=True, ymax=walk_ymax, slack=0.994)
        run = walker(run_stride, 0.6 * L, run_duty, 0.0, 9.0, 2, 0.0, arm_amp=62, fore=90, fore_amp=38, arm_bias=20,
                     vault=True, ymax=run_ymax, slack=0.98)
    else:
        walk_T, walk_duty = 1.0, 0.55; walk_stride = min(0.55 * reach, 0.8 * drv.max_stride)
        run_T, run_duty = 0.6, 0.45; run_stride = min(0.9 * reach, 0.85 * drv.stride_for(drv.run_crouch))
        walk = walker(walk_stride, 0.10 * H, walk_duty, 0.008, 2.0, 5, drv.crouch)
        run = walker(run_stride, 0.12 * H, run_duty, 0.012, 7.0, 3, drv.run_crouch)
    gait = {'insect': 'alternating gait (tripod for six legs)', 'quadruped': 'trot (diagonal pairs)', 'biped': 'walk, arms swing'}.get(drv.plan, 'alternating gait')
    clips = [
        ClipDef('idle', 2.4, idle, True, 0.0, 'breathing, head looks around, feet planted'),
        ClipDef('walk', walk_T, walk, True, walk_stride / (walk_duty * walk_T), gait + ', feet planted in stance'),
        ClipDef('run', run_T, run, True, run_stride / (run_duty * run_T), 'longer strides, body pitched forward'),
        ClipDef('hover', 1.2, hover, True, 0.0, 'wingbeat %.1f Hz, legs folded, body bobs' % flap_hz),
        ClipDef('takeoff', 0.8, takeoff, False, 0.0, 'wings grow back in 0.25 s while the body lifts'),
        ClipDef('land', 0.8, land, False, 0.0, 'body comes down, wings shrink away in the last 0.25 s'),
        ClipDef('attack', 1.0, bite, False, 0.0, 'bite: rear back head up, lunge and snap the head down, recover'),
        ClipDef('sting', 1.2, sting, False, 0.0, 'sting: wind-up, fast jab with the abdomen swung under and forward, recover'),
    ]
    if drv.plan == 'biped':                     # knockdowns and get-ups (motionlab/moves.py)
        from .moves import biped_moves, moving_jumps, dance_moves
        clips += biped_moves(drv)
        clips += moving_jumps(drv, {c.name: c for c in clips})
        clips += dance_moves(drv)
    return [c for c in clips if clip_fits(drv, c.name)]


def clip_fits(drv, name):
    """Which clips make sense for this body: flight needs wings, the sting an insect abdomen, the bite a head on a
    four-legged or many-legged body (people get idle, walk and run for now)."""
    if name in ('hover', 'takeoff', 'land'): return bool(drv.wings)
    if name == 'sting': return drv.plan == 'insect' and bool(drv.abd)
    if name == 'attack': return drv.plan != 'biped'
    return True
