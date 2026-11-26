# PZA's Kenshi Motion Lab - clip sequencing with inertialized joins. Author: PZA. GPLv3.
#
# Chains multiple clips into one continuous Clip (smoothly decaying pose offsets at every cut so there is no snap,
# keeping forward root travel continuous). Can be written to .glb, exported to .skeleton, or rendered to a GIF.
#   python -m motionlab sequence <rigged.glb> --steps "walk:2,run:1.5:0.3,jump_run:0.1" -o <out.glb>
import numpy as np
from . import bake, mathx as mx, motion


def parse_steps(spec):
    """Parse a step list from a string ('walk:2,run:1.5:0.3,jump_run::0.1') or list of dicts."""
    if isinstance(spec, (list, tuple)):
        out = []
        for s in spec:
            if isinstance(s, str):
                out.extend(parse_steps(s))
            elif isinstance(s, dict):
                if 'clip' not in s:
                    raise ValueError('sequence step missing "clip": %r' % (s,))
                out.append({
                    'clip': str(s['clip']),
                    'duration': None if s.get('duration') is None else float(s['duration']),
                    'blend': float(s.get('blend', 0.25)),
                })
            else:
                raise ValueError('bad sequence step %r' % (s,))
        return out
    out = []
    for tok in str(spec).split(','):
        tok = tok.strip()
        if not tok:
            continue
        parts = [p.strip() for p in tok.split(':')]
        name = parts[0]
        dur = float(parts[1]) if len(parts) > 1 and parts[1] else None
        blend = float(parts[2]) if len(parts) > 2 and parts[2] else 0.25
        out.append({'clip': name, 'duration': dur, 'blend': blend})
    if not out:
        raise ValueError('empty sequence specification')
    return out


def _ease(x):
    u = np.clip(x, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def _sample_clip(drv, cd, dur, fps):
    drv.reset()
    n = max(2, int(round(dur * fps)))
    dt = 1.0 / fps
    if cd.loop:
        # Settle springs over one warm-up cycle before sampling
        nw = max(2, int(round(cd.length * fps)))
        for k in range(nw):
            drv.pose(cd.fn((k * dt) % cd.length, cd.length), dt=dt)
    Q, T, S = [], [], []
    for k in range(n):
        t = (k * dt) % cd.length if cd.loop else min(k * dt, cd.length)
        q, tt, s, _ = drv.pose(cd.fn(t, cd.length), dt=dt)
        Q.append(q.copy())
        T.append(tt.copy())
        S.append(s.copy())
    return np.array(Q), np.array(T), np.array(S)


def build_sequence(drv, steps, fps=30, name='sequence', travel=True, clip_map=None):
    """Chain `steps` into one inertialized bake.Clip. Returns (Clip, metadata dict)."""
    parsed = parse_steps(steps)
    clips = clip_map if clip_map is not None else {c.name: c for c in motion.standard_clips(drv)}
    r = drv.rig
    root = r.b(drv.map['root'])
    Qs, Ts, Ss = [], [], []
    joins = []
    segments = []
    z0 = 0.0
    prevQ = prevT = prevS = None
    dt_step = 1.0 / fps

    for idx, st in enumerate(parsed):
        cn = st['clip']
        if cn not in clips:
            raise ValueError('unknown clip %r in sequence (known: %s)' % (cn, ', '.join(sorted(clips))))
        cd = clips[cn]
        dur = st['duration'] if st['duration'] is not None else cd.length
        if dur <= 0:
            raise ValueError('step %d (%s): duration must be positive' % (idx, cn))
        blend = max(0.0, float(st['blend']))
        Q, T, S = _sample_clip(drv, cd, dur, fps)

        if travel:
            trav = np.arange(len(T)) * dt_step * cd.speed
            T[:, root, 2] += z0 + trav

        if prevQ is not None and blend > 0.0:
            dq = np.array([mx.q_mul(prevQ[b], mx.q_conj(Q[0][b])) for b in range(r.n)])
            dt_ = prevT - T[0]
            if travel:
                dt_[root, 2] = 0.0  # keep forward travel continuous across the join
            ds_ = prevS - S[0]
            nb = int(round(blend * fps))
            if nb > 0:
                for k in range(min(nb, len(Q))):
                    w = 1.0 - float(_ease(k / nb))
                    for b in range(r.n):
                        Q[k][b] = mx.norm(mx.q_mul(mx.q_slerp(mx.q_identity(), dq[b], w), Q[k][b]))
                    T[k] = T[k] + dt_ * w
                    S[k] = S[k] + ds_ * w
                joins.append(len(Qs))

        start_frame = len(Qs)
        Qs.extend(Q)
        Ts.extend(T)
        Ss.extend(S)
        segments.append({'clip': cn, 'start_frame': start_frame, 'frames': len(Q), 'duration_s': round(len(Q) * dt_step, 4)})
        prevQ, prevT, prevS = Q[-1].copy(), T[-1].copy(), S[-1].copy()
        if travel:
            z0 = float(T[-1][root, 2] + (cd.speed * dt_step if cd.speed else 0.0) - r.rest_t[root][2])

    Q_arr = np.array(Qs)
    T_arr = np.array(Ts)
    S_arr = np.array(Ss)
    for k in range(1, len(Q_arr)):
        flip = np.einsum('bi,bi->b', Q_arr[k], Q_arr[k - 1]) < 0
        Q_arr[k][flip] *= -1
    times = np.arange(len(Q_arr), dtype=np.float64) * dt_step
    tracks = {}
    wing_roots = {r.b(e['bones'][0]) for e in r.map.get('wings', [])}
    body_i = r.b(r.map['body'])
    for b in range(r.n):
        tr = {'r': Q_arr[:, b]}
        if b == root or b == body_i or np.ptp(T_arr[:, b], axis=0).max() > 1e-6:
            tr['t'] = T_arr[:, b]
        if b in wing_roots or np.ptp(S_arr[:, b], axis=0).max() > 1e-6:
            tr['s'] = S_arr[:, b]
        tracks[b] = tr
    desc = ' -> '.join(s['clip'] for s in parsed)
    clip = bake.Clip(name, times, tracks, loop=False, speed=0.0, note='sequence: ' + desc)
    meta = {'name': name, 'fps': fps, 'frames': len(Q_arr), 'length_s': round(float(times[-1]), 4),
            'joins': joins, 'segments': segments, 'travel': travel}
    return clip, meta
