# PZA's Kenshi Motion Lab - checks that log what the driver did (a rule that never reports is a failed test).
# Author: PZA. GPLv3.
import numpy as np


def foot_check(drv, cd, fps=30):
    """Per leg: worst IK miss (foot vs its target) and worst slip of a planted foot against the ground
    (in-place clip: a planted foot must move back at exactly the clip's speed)."""
    rig = drv.rig; drv.reset()
    n = max(2, int(round(cd.length * fps)))
    miss, slip, planted = {}, {}, {}
    prev = None
    for k in range(2 * n):
        t = (k % n) / fps if cd.loop else min(k, n) / fps
        spec = cd.fn(t, cd.length); q, tt, s, W = drv.pose(spec, dt=1 / fps)
        feet = spec.get('feet', {}); st = feet.get('_stance', set()) if spec.get('fold', 0) <= 0 else set()
        cur = {e['name']: rig.chain_points(ch, W)[-1] for e, ch in drv.legs}
        if k >= n or not cd.loop:
            for e, ch in drv.legs:
                nm = e['name']; tg = feet.get(nm)
                if tg is not None and spec.get("fold", 0) <= 0 and nm in st:   # planted feet only (swing targets may be capped)
                    miss[nm] = max(miss.get(nm, 0.0), float(np.linalg.norm(cur[nm] - tg)))
                if prev is not None and nm in st and nm in prev[1]:
                    v = (cur[nm] - prev[0][nm]) * fps + np.array([0, 0, cd.speed])
                    slip[nm] = max(slip.get(nm, 0.0), float(np.linalg.norm(v)))
                    planted[nm] = planted.get(nm, 0) + 1
        prev = (cur, st)
    return {'ik_miss_m': miss, 'slip_m_per_s': slip, 'planted_frames': planted}


def move_check(drv, cd, fps=30):
    """A one-shot move (knockdown / get-up): pelvis height at the start, lowest and end (relative to the rest
    height), the lowest joint against the ground, and the worst miss of a planted hand."""
    rig = drv.rig; drv.reset()
    root = rig.b(drv.map['root']); H0 = float(rig.Wrest[root][:3, 3][1])
    n = max(2, int(round(cd.length * fps)))
    py, low, hand_miss = [], 0.0, 0.0
    for k in range(n + 1):
        spec = cd.fn(min(k / fps, cd.length), cd.length); q, t, s, W = drv.pose(spec, dt=1 / fps)
        py.append(float(W[root][:3, 3][1]) / H0)
        low = min(low, float(W[:, 1, 3].min()) / H0)
        for e, ch in drv.arms:
            tg = spec.get('hands', {}).get(e['side'])
            if tg is not None:
                hand_miss = max(hand_miss, float(np.linalg.norm(rig.chain_points(ch, W)[-1] - tg)))
    return {'pelvis_start': py[0], 'pelvis_min': min(py), 'pelvis_end': py[-1], 'lowest_joint': low,
            'hand_miss_m': hand_miss, 'frames': n + 1,
            'knee_floor_hits': drv.knee_floor_hits, 'hand_releases': drv.hand_releases,
            'body_floor_lifts': drv.body_floor_lifts}
