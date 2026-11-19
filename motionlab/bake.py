# PZA's Kenshi Motion Lab - bake driver motion into keyframed clips, play clips back, render preview GIFs.
# Author: PZA. GPLv3.
import numpy as np
from . import mathx as mx


class Clip:
    def __init__(self, name, times, tracks, loop=True, speed=0.0, note=''):
        self.name, self.times, self.tracks, self.loop, self.speed, self.note = name, times, tracks, loop, speed, note


def bake(driver, cd, fps=30, warm_cycles=2):
    """Sample a ClipDef at fps. Looping clips run warm-up cycles first so springs settle into the loop; the last
    key equals the first cycle's start, so the loop has no seam. Every bone gets rotation keys; the root and body
    translation; wing roots scale."""
    rig = driver.rig
    n = max(2, int(round(cd.length * fps)))
    times = np.linspace(0, cd.length, n + 1)
    driver.reset()
    if cd.loop:
        for c in range(warm_cycles):
            for tt in times[:-1]: driver.pose(cd.fn(tt, cd.length), dt=1 / fps)
    else:
        for _ in range(int(0.5 * fps)): driver.pose(cd.fn(0.0, cd.length), dt=1 / fps)   # settle springs at the start pose
    Q, T, S = [], [], []
    for tt in times:
        q, t, s, _ = driver.pose(cd.fn(tt % cd.length if (cd.loop and tt >= cd.length) else tt, cd.length), dt=1 / fps)
        Q.append(q.copy()); T.append(t.copy()); S.append(s.copy())
    Q = np.array(Q); T = np.array(T); S = np.array(S)
    if cd.loop:                                   # close the loop exactly
        Q[-1] = Q[0]; T[-1] = T[0]; S[-1] = S[0]
    for k in range(1, len(Q)):                    # quaternion continuity for linear interpolation
        flip = np.einsum('bi,bi->b', Q[k], Q[k - 1]) < 0
        Q[k][flip] *= -1
    tracks = {}
    names = rig.names
    wing_roots = {rig.b(e['bones'][0]) for e in rig.map['wings']}
    for b in range(rig.n):
        tr = {'r': Q[:, b]}
        if b == rig.b(rig.map['root']) or b == rig.b(rig.map['body']) or np.ptp(T[:, b], axis=0).max() > 1e-6: tr['t'] = T[:, b]
        if b in wing_roots or np.ptp(S[:, b], axis=0).max() > 1e-6: tr['s'] = S[:, b]
        tracks[b] = tr
    return Clip(cd.name, times, tracks, cd.loop, cd.speed, cd.note)


def sample_read_clip(sk, clip_nodes, t):
    """Locals of every bone at time t from a clip read back from a .glb (gltf.read_clips), rest where no track."""
    n = len(sk)
    q = [np.array(x, float) for x in sk.rest_q]; tr = [np.array(x, float) for x in sk.rest_t]; sc = [np.array(x, float) for x in sk.rest_s]
    for i, nm in enumerate(sk.names):
        ch = clip_nodes.get(nm)
        if not ch: continue
        def at(path):
            ts = ch.get('t_' + path, ch.get('times')); v = ch[path]
            if len(ts) < 2: return v[0], v[0], 0.0
            k = int(np.clip(np.searchsorted(ts, t) - 1, 0, len(ts) - 2))
            return v[k], v[k + 1], float(np.clip((t - ts[k]) / max(ts[k + 1] - ts[k], 1e-9), 0, 1))
        if 'rotation' in ch: a, b, u = at('rotation'); q[i] = mx.q_slerp(a, b, u)
        if 'translation' in ch: a, b, u = at('translation'); tr[i] = a * (1 - u) + b * u
        if 'scale' in ch: a, b, u = at('scale'); sc[i] = a * (1 - u) + b * u
    return [mx.trs(tr[i], q[i], sc[i]) for i in range(n)]


def gif_from_clip(model, sk, clip_nodes, length, path, fps=30, speed=0.0, cam=None, repeats=1, label=None, bones=False, yaw=38.0, pitch=14.0):
    from PIL import ImageDraw
    from . import render
    lo, hi = model.bounds()
    centre = np.array([0, 0.5 * (hi[1] - lo[1]), 0])
    cam = cam or render.Camera(centre, yaw, pitch, max(1.45 * float(np.linalg.norm(hi - lo)), 2.3 * float(hi[1] - lo[1])), size=(480, 360))
    R = render.Renderer(model, cam)
    r0 = sk.parents.index(-1)
    # follow a body that travels (rolls, knockdowns): keep the root centred and scroll the floor instead
    x0 = sk.world(sample_read_clip(sk, clip_nodes, 0.0))[r0][:3, 3]
    y_rest = float(sk.world_rest()[r0][1, 3])
    frames = []
    n = int(round(length * fps))
    for r_ in range(repeats):
        for k in range(n):
            t = k / fps
            W = sk.world(sample_read_clip(sk, clip_nodes, t))
            P, N = model.skin(W)
            T = (r_ * n + k) / fps
            d = W[r0][:3, 3] - x0; dy = float(W[r0][1, 3]) - y_rest; d[1] = 0.0
            # up / down: the camera rises with a body above its standing height (hover, a jump, a ledge, a high fall)
            lift = max(0.0, dy - 0.1 * float(hi[1] - lo[1]))
            if float(np.abs(d).max()) > 1e-6 or lift > 0: P = P - d - np.array([0.0, lift, 0.0])
            gx, gz = float(d[0]), float(d[2])
            ov = (lambda img, z, pr, W=W: render.draw_bones(img, z, pr, W, sk.parents)) if bones else None
            im = R.frame(P, ground_y=-lift, ground_offset=(gx, speed * T + gz), overlay=ov)
            if label:
                ImageDraw.Draw(im).text((10, 8), label, fill=(40, 34, 28))
            frames.append(im)
    render.save_gif(frames, path, fps)
    return len(frames)
