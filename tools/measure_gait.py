# PZA's Kenshi Motion Lab - measure a walk / run cycle, from Kenshi's own .skeleton animations or from our baked .glb
# clips, with the same numbers for both (to tune our gait against the game's). Author: PZA. GPLv3.
#   python tools/measure_gait.py --kenshi <file.skeleton> "walk lower" ["walk upper"]
#   python tools/measure_gait.py --glb <rigged.glb> walk
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from motionlab import mathx as mx


def _angle(a, b):
    return float(np.degrees(np.arccos(np.clip(np.dot(mx.norm(a), mx.norm(b)), -1, 1))))


def measure(sk, sample, length, fps=60):
    """sample(t) -> list of local 4x4 per bone. Prints knee, hip, stride, lift, arm, elbow numbers."""
    ix = {n: i for i, n in enumerate(sk.names)}
    g = lambda W, n: W[ix[n]][:3, 3]
    rows = []
    for t in np.linspace(0, length, int(length * fps), endpoint=False):
        W = sk.world(sample(t))
        r = {}
        for s in 'LR':
            th, ca, ft = g(W, 'Bip01 %s Thigh' % s), g(W, 'Bip01 %s Calf' % s), g(W, 'Bip01 %s Foot' % s)
            r['knee_' + s] = 180.0 - _angle(th - ca, ft - ca)          # 0 = straight
            r['foot_z_' + s] = ft[2] - g(W, 'Bip01 Pelvis')[2]; r['foot_y_' + s] = ft[1]
            if 'Bip01 %s UpperArm' % s in ix:
                ua, fa, ha = g(W, 'Bip01 %s UpperArm' % s), g(W, 'Bip01 %s Forearm' % s), g(W, 'Bip01 %s Hand' % s)
                d = fa - ua; r['arm_' + s] = float(np.degrees(np.arctan2(d[2], -d[1])))   # + = forward
                r['elbow_' + s] = 180.0 - _angle(ua - fa, ha - fa)
        r['hip_y'] = g(W, 'Bip01 Pelvis')[1]; r['root_z'] = g(W, 'Bip01')[2]
        rows.append(r)
    def col(k): return np.array([x[k] for x in rows if k in x])
    out = {}
    for s in 'L':
        k = col('knee_' + s); out['knee bend deg (min / max)'] = (k.min(), k.max())
        z = col('foot_z_' + s); out['foot ahead of pelvis m (back / front)'] = (z.min(), z.max())
        y = col('foot_y_' + s); out['ankle height m (low / high)'] = (y.min(), y.max())
        if len(col('arm_' + s)):
            a = col('arm_' + s); out['upper arm swing deg (back / front)'] = (a.min(), a.max())
            e = col('elbow_' + s); out['elbow bend deg (min / max)'] = (e.min(), e.max())
    h = col('hip_y'); out['pelvis height m (low / high)'] = (h.min(), h.max())
    rz = col('root_z'); out['root moves forward m'] = (rz.min(), rz.max())
    return out


def kenshi_sampler(path, names):
    from motionlab.ogre import skeleton_read
    v, bones, parents, anims = skeleton_read.read(path)
    handles = sorted(bones); hidx = {h: i for i, h in enumerate(handles)}
    from motionlab.model import Skeleton
    sk = Skeleton()
    for h in handles:
        nm, pos, q, sc = bones[h]
        sk.names.append(nm); sk.parents.append(hidx[parents[h]] if h in parents else -1)
        sk.rest_t.append(np.array(pos, float) / 10.0); sk.rest_q.append(mx.norm(np.array(q, float))); sk.rest_s.append(np.ones(3))
    tracks = {}; length = 0
    for nm, L, tr in anims:
        if nm in names:
            length = max(length, L)
            for h, keys in tr.items(): tracks[hidx[h]] = keys
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
    return sk, sample, length


def glb_sampler(path, clip):
    from motionlab import gltf, bake
    m = gltf.load_glb(path); rc = gltf.read_clips(path)[clip]
    length = max(c['times'][-1] for c in rc.values())
    return m.skeleton, (lambda t: bake.sample_read_clip(m.skeleton, rc, t)), length


if __name__ == '__main__':
    if sys.argv[1] == '--kenshi': sk, smp, L = kenshi_sampler(sys.argv[2], sys.argv[3:])
    else: sk, smp, L = glb_sampler(sys.argv[2], sys.argv[3])
    print('cycle %.2f s' % L)
    for k, (a, b) in measure(sk, smp, L).items(): print('  %-38s %7.3f  %7.3f' % (k, a, b))
