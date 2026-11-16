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
                if tg is not None and spec.get('fold', 0) <= 0:
                    miss[nm] = max(miss.get(nm, 0.0), float(np.linalg.norm(cur[nm] - tg)))
                if prev is not None and nm in st and nm in prev[1]:
                    v = (cur[nm] - prev[0][nm]) * fps + np.array([0, 0, cd.speed])
                    slip[nm] = max(slip.get(nm, 0.0), float(np.linalg.norm(v)))
                    planted[nm] = planted.get(nm, 0) + 1
        prev = (cur, st)
    return {'ik_miss_m': miss, 'slip_m_per_s': slip, 'planted_frames': planted}
