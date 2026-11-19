# PZA's Kenshi Motion Lab - how natural a cycle moves, in numbers (the Studio scorecard and tests/test_naturalness.py).
# The same numbers are measured on our clips and on a reference (Kenshi's own clips for the same skeleton), so a
# "machine" look shows up as a number: a pelvis that never turns, feet that never roll, hip height with corners.
# Author: PZA. GPLv3.
import numpy as np
from . import mathx as mx


def _rot(W):
    R = W[:3, :3]; return R / np.linalg.norm(R, axis=0)


def rotvec_deg(Rm):
    """Rotation vector (degrees) of a 3x3 rotation: x = pitch (+ leans forward / toe down), y = yaw, z = roll."""
    q = mx.norm(mx.m3_to_q(Rm)); w = float(np.clip(q[3], -1, 1))
    if w < 0: q, w = -q, -w
    s = np.sqrt(max(1 - w * w, 0.0)); ang = 2 * np.arccos(w)
    return np.degrees(q[:3] / s * ang) if s > 1e-9 else np.zeros(3)


def corners_pct(x):
    """Share (%) of a cyclic curve's motion above its 2nd harmonic: a smooth double bounce is ~2-3%, a curve built
    from max / min limits (kinks) is 20-40%."""
    x = np.asarray(x, float); F = np.abs(np.fft.rfft(x - x.mean())) ** 2
    return float(100 * F[3:].sum() / max(F[1:].sum(), 1e-12))


def jerk_rms(P, fps):
    """RMS jerk (m/s^3) of a cyclic path of points (n, 3)."""
    d = lambda x: (np.roll(x, -1, 0) - np.roll(x, 1, 0)) * fps / 2
    return float(np.sqrt((np.linalg.norm(d(d(d(np.asarray(P, float)))), axis=1) ** 2).mean()))


def roles(names, rigmap=None):
    """Bone indices the scorecard reads: pelvis, chest, head, left foot, left toe, left hand (Bip01 names, or the
    rig map's own roles)."""
    ix = {n: i for i, n in enumerate(names)}
    rm = rigmap or {}
    def first(*cands):
        for c in cands:
            if c and c in ix: return ix[c]
        return None
    legL = [e for e in rm.get('legs', []) if e.get('side') == 'L']
    armL = [e for e in rm.get('arms', []) if e.get('side') == 'L']
    out = {'pelvis': first('Bip01 Pelvis', rm.get('root')),
           'chest': first('Bip01 Spine2', (rm.get('spine_chain') or [None])[-1], rm.get('body')),
           'head': first('Bip01 Head', rm.get('head')),
           'foot': first('Bip01 L Foot', legL[0].get('foot') if legL else None),
           'toe': first('Bip01 L Toe0'),
           'hand': first('Bip01 L Hand', armL[0].get('hand') if armL else None),
           'upperarm': first('Bip01 L UpperArm', armL[0]['bones'][0] if armL else None),
           'forearm': first('Bip01 L Forearm', armL[0]['bones'][1] if armL and len(armL[0]['bones']) > 1 else None)}
    return {k: v for k, v in out.items() if v is not None}


def cycle(frames, rest, R, fps):
    """Numbers and curves of one cycle. frames: (n, bones, 4, 4) world matrices over exactly one cycle; rest: the
    rest world matrices; R: roles(). Angles are each part's turn away from its rest pose (chest against the pelvis)."""
    F = np.asarray(frames); n = len(F)
    rel = {k: np.array([_rot(F[i, b]) @ _rot(rest[b]).T for i in range(n)]) for k, b in R.items()}
    rv = lambda M: np.array([rotvec_deg(m) for m in M])
    curves = {}
    P = F[:, R['pelvis'], :3, 3]
    curves['pelvis height (cm)'] = 100 * (P[:, 1] - P[:, 1].mean())
    pel = rv(rel['pelvis']); curves['pelvis yaw (deg)'] = pel[:, 1]; curves['pelvis roll (deg)'] = pel[:, 2]
    if 'chest' in R:
        ch = rv(np.einsum('nji,njk->nik', rel['pelvis'], rel['chest']))     # chest against the pelvis
        curves['chest yaw against pelvis (deg)'] = ch[:, 1]
    if 'foot' in R: curves['foot pitch (deg, + toe down)'] = rv(rel['foot'])[:, 0]
    if 'head' in R:
        Hh = F[:, R['head'], :3, 3]; curves['head height (cm)'] = 100 * (Hh[:, 1] - Hh[:, 1].mean())
    out = {'pelvis yaw range (deg)': float(np.ptp(pel[:, 1])), 'pelvis roll range (deg)': float(np.ptp(pel[:, 2]))}
    if 'chest yaw against pelvis (deg)' in curves: out['chest counter-turn range (deg)'] = float(np.ptp(curves['chest yaw against pelvis (deg)']))
    if 'foot pitch (deg, + toe down)' in curves: out['foot roll range (deg)'] = float(np.ptp(curves['foot pitch (deg, + toe down)']))
    if 'chest' in R: out['chest lean (deg, mean)'] = float(rv(rel['chest'])[:, 0].mean())
    if all(k in R for k in ('upperarm', 'forearm', 'hand')):
        u, f, h = (F[:, R[k], :3, 3] for k in ('upperarm', 'forearm', 'hand'))
        d = f - u; sw = np.degrees(np.arctan2(d[:, 2], -d[:, 1])); curves['arm swing (deg, + forward)'] = sw
        a_ = (u - f) / np.linalg.norm(u - f, axis=1)[:, None]; b_ = (h - f) / np.linalg.norm(h - f, axis=1)[:, None]
        el = 180 - np.degrees(np.arccos(np.clip(np.einsum('ij,ij->i', a_, b_), -1, 1))); curves['elbow bend (deg)'] = el
        out['arm swing range (deg)'] = float(np.ptp(sw)); out['elbow bend range (deg)'] = float(np.ptp(el))
    out['pelvis bob (cm)'] = float(np.ptp(P[:, 1]) * 100)
    if 'head' in R: out['head bob / pelvis bob'] = float(np.ptp(Hh[:, 1]) / max(np.ptp(P[:, 1]), 1e-6))
    out['hip-height corners (%)'] = corners_pct(P[:, 1])
    drift = np.outer(np.arange(n) / n, P[-1] - P[0]) if n > 1 else 0.0     # in-place clips: no travel to remove
    out['pelvis jerk (m/s3)'] = jerk_rms(P - drift, fps)
    return out, curves


# how each number is graded against the reference: 'match' = within +-25% green, +-50% amber; 'low' = lower is
# better (green up to 1.5x the reference or the floor, amber up to 3x)
GRADING = {'pelvis yaw range (deg)': 'match', 'pelvis roll range (deg)': 'match', 'chest counter-turn range (deg)': 'match',
           'foot roll range (deg)': 'match', 'pelvis bob (cm)': 'match', 'head bob / pelvis bob': 'match',
           'hip-height corners (%)': ('low', 8.0), 'pelvis jerk (m/s3)': ('low', 0.0),
           'chest lean (deg, mean)': ('abs', 5.0), 'arm swing range (deg)': 'match', 'elbow bend range (deg)': 'match',
           'ground speed (m/s)': 'match'}


def grade(key, ours, ref):
    g = GRADING.get(key, 'match')
    if ref is None or not np.isfinite(ref): return 'none'
    if g == 'match':
        if abs(ref) < 1e-6: return 'green' if abs(ours) < 1.0 else 'amber'
        r = ours / ref
        return 'green' if 0.75 <= r <= 1.25 else ('amber' if 0.5 <= r <= 1.5 else 'red')
    if g[0] == 'abs':                                   # degrees apart: green within g[1], amber within 2x
        dd = abs(ours - ref); return 'green' if dd <= g[1] else ('amber' if dd <= 2 * g[1] else 'red')
    floor = g[1]; lim = max(1.5 * ref, floor)
    return 'green' if ours <= lim else ('amber' if ours <= max(3 * ref, 2 * floor) else 'red')


def align(a, b):
    """Cyclic shift (frames) that best lines curve b up with curve a (same length)."""
    a = np.asarray(a, float) - np.mean(a); b = np.asarray(b, float) - np.mean(b)
    c = [float(np.dot(a, np.roll(b, k))) for k in range(len(a))]
    return int(np.argmax(c))
