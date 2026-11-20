# PZA's Kenshi Motion Lab - a procedural test creature (no third-party assets). Author: PZA. GPLv3.
import numpy as np
from motionlab.model import Model


def _ellipsoid(c, r, nu=24, nv=14):
    V, T = [], []
    for i in range(nv + 1):
        th = np.pi * i / nv
        for j in range(nu):
            ph = 2 * np.pi * j / nu
            V.append(c + r * np.array([np.sin(th) * np.cos(ph), np.cos(th), np.sin(th) * np.sin(ph)]))
    for i in range(nv):
        for j in range(nu):
            a = i * nu + j; b = i * nu + (j + 1) % nu; c_ = a + nu; d = b + nu
            T += [(a, c_, b), (b, c_, d)]
    return np.array(V), np.array(T)


def _tube(pts, rad, seg=8, sub=4):
    pts = np.array(pts, float)
    P = [pts[0]]
    for a, b in zip(pts[:-1], pts[1:]):
        for k in range(1, sub + 1): P.append(a + (b - a) * k / sub)
    P = np.array(P); V, T = [], []
    for i, p in enumerate(P):
        d = P[min(i + 1, len(P) - 1)] - P[max(i - 1, 0)]; d /= np.linalg.norm(d)
        u = np.cross(d, [0, 1, 0]) if abs(d[1]) < 0.9 else np.cross(d, [1, 0, 0]); u /= np.linalg.norm(u); w = np.cross(d, u)
        r = rad * (1 - 0.5 * i / len(P))
        for j in range(seg):
            a = 2 * np.pi * j / seg; V.append(p + r * (np.cos(a) * u + np.sin(a) * w))
    for i in range(len(P) - 1):
        for j in range(seg):
            a = i * seg + j; b = i * seg + (j + 1) % seg
            T += [(a, a + seg, b), (b, a + seg, b + seg)]
    tipc = len(V); V.append(P[-1]); T += [(len(P) * seg - seg + j, tipc, len(P) * seg - seg + (j + 1) % seg) for j in range(seg)]
    return np.array(V), np.array(T)


def _plate(c, along, side, L, Wd, nl=8, nw=4, th=0.01):
    V, T = [], []
    for s in (-1, 1):
        base = len(V)
        for i in range(nl + 1):
            for j in range(nw + 1):
                u = i / nl; v = (j / nw - 0.5) * Wd * (1 - 0.6 * u)
                V.append(c + along * L * u + side * v + np.array([0, s * th, 0]))
        for i in range(nl):
            for j in range(nw):
                a = base + i * (nw + 1) + j
                T += [(a, a + nw + 1, a + 1), (a + 1, a + nw + 1, a + nw + 2)]
    return np.array(V), np.array(T)


def make_hexapod(yaw_deg=0.0):
    parts = []
    parts.append(_ellipsoid(np.array([0, 1.0, 0.0]), np.array([0.35, 0.3, 0.55])))      # thorax
    parts.append(_ellipsoid(np.array([0, 1.0, -0.95]), np.array([0.3, 0.28, 0.5])))     # abdomen
    parts.append(_ellipsoid(np.array([0, 1.05, 0.72]), np.array([0.22, 0.2, 0.22])))    # head
    for side in (1, -1):
        for k, z in enumerate((0.35, 0.0, -0.35)):
            hip = np.array([side * 0.25, 0.9, z])
            knee = np.array([side * 0.75, 1.25, z + 0.15 * (1 - k)])
            foot = np.array([side * 1.05, 0.0, z + 0.25 * (1 - k)])
            parts.append(_tube([hip, knee, foot], 0.06))
        parts.append(_plate(np.array([side * 0.12, 1.32, -0.15]), np.array([side * 0.25, 0.05, -0.97]) / 1.0,
                            np.array([1, 0, 0]), 0.95, 0.35))
        parts.append(_tube([np.array([side * 0.08, 1.2, 0.85]), np.array([side * 0.2, 1.55, 1.1]), np.array([side * 0.32, 1.6, 1.4])], 0.025))
    V, T = [], []; base = 0
    for v, t in parts: V.append(v); T.append(t + base); base += len(v)
    V = np.concatenate(V); T = np.concatenate(T)
    a = np.radians(yaw_deg); R = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    V = V @ R.T
    m = Model(); m.name = 'test_hexapod'
    m.verts = V; m.tris = T
    n = np.zeros_like(V)
    fn = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
    for k in range(3): np.add.at(n, T[:, k], fn)
    m.normals = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    m.uvs = np.zeros((len(V), 2)); m.parts = [{'material': None, 'tri_start': 0, 'tri_count': len(T)}]
    return m


QUAD = {  # a four-legged body shaped like Kenshi's animals (rounded, about 0.6 of a dog): the front legs hang off the
    # NECK, whose pivot sits well ahead of and above the shoulders (a nod swings them), and are shorter than the hind
    # legs; the hind "foot" is the hock, high off the ground
    'Bip01': (None, (0, 0.62, -0.36)), 'Bip01 Pelvis': ('Bip01', (0, 0.62, -0.36)),
    'Bip01 L Thigh': ('Bip01 Pelvis', (0.11, 0.62, -0.36)), 'Bip01 L Calf': ('Bip01 L Thigh', (0.19, 0.36, -0.26)),
    'Bip01 L Foot': ('Bip01 L Calf', (0.16, 0.27, -0.45)), 'Bip01 L Toe0': ('Bip01 L Foot', (0.16, 0.02, -0.42)),
    'Bip01 R Thigh': ('Bip01 Pelvis', (-0.11, 0.62, -0.36)), 'Bip01 R Calf': ('Bip01 R Thigh', (-0.19, 0.36, -0.26)),
    'Bip01 R Foot': ('Bip01 R Calf', (-0.16, 0.27, -0.45)), 'Bip01 R Toe0': ('Bip01 R Foot', (-0.16, 0.02, -0.42)),
    'Bip01 Spine': ('Bip01 Pelvis', (0, 0.63, -0.2)), 'Bip01 Spine1': ('Bip01 Spine', (0, 0.64, 0.0)),
    'Bip01 Spine2': ('Bip01 Spine1', (0, 0.65, 0.16)), 'Bip01 Neck': ('Bip01 Spine2', (0, 0.74, 0.41)),
    'Bip01 Head': ('Bip01 Neck', (0, 0.77, 0.5)),
    'Bip01 L Clavicle': ('Bip01 Neck', (0.07, 0.7, 0.2)), 'Bip01 L UpperArm': ('Bip01 L Clavicle', (0.13, 0.46, 0.25)),
    'Bip01 L Forearm': ('Bip01 L UpperArm', (0.14, 0.37, 0.18)), 'Bip01 L Hand': ('Bip01 L Forearm', (0.2, 0.11, 0.25)),
    'Bip01 R Clavicle': ('Bip01 Neck', (-0.07, 0.7, 0.2)), 'Bip01 R UpperArm': ('Bip01 R Clavicle', (-0.13, 0.46, 0.25)),
    'Bip01 R Forearm': ('Bip01 R UpperArm', (-0.14, 0.37, 0.18)), 'Bip01 R Hand': ('Bip01 R Forearm', (-0.2, 0.11, 0.25)),
}
QUAD_ENDS = {'Bip01 L Toe0': (0.16, 0.02, -0.36), 'Bip01 R Toe0': (-0.16, 0.02, -0.36), 'Bip01 Head': (0, 0.78, 0.68),
             'Bip01 L Hand': (0.2, 0.02, 0.3), 'Bip01 R Hand': (-0.2, 0.02, 0.3)}


def make_quadruped():
    """A stick four-legged animal rigged like Kenshi's dog (Bip01 names; front legs under the neck, shorter than the
    hind legs): tube per bone, skinned rigidly."""
    return _stick(QUAD, QUAD_ENDS, 'test_quadruped')


def make_biped(tpose=False, toe_r=0.15, toe_l=0.15):
    """A stick person rigged like Kenshi's characters (3ds Max Biped names, Bip01 ...): tube per bone, skinned rigidly.
    tpose: arms straight out to the sides (as Meshy / Mixamo export them) instead of hanging; toe_l / toe_r: how far each
    toe sits ahead of its ankle (a lopsided model's feet differ)."""
    from motionlab.model import Skeleton
    J = {  # name: (parent, world position)
        'Bip01': (None, (0, 1.0, 0)), 'Bip01 Pelvis': ('Bip01', (0, 1.0, 0)),
        'Bip01 L Thigh': ('Bip01 Pelvis', (0.1, 1.0, 0)), 'Bip01 L Calf': ('Bip01 L Thigh', (0.1, 0.55, 0.01)),
        'Bip01 L Foot': ('Bip01 L Calf', (0.1, 0.09, 0)), 'Bip01 L Toe0': ('Bip01 L Foot', (0.1, 0.02, 0.15)),
        'Bip01 R Thigh': ('Bip01 Pelvis', (-0.1, 1.0, 0)), 'Bip01 R Calf': ('Bip01 R Thigh', (-0.1, 0.55, 0.01)),
        'Bip01 R Foot': ('Bip01 R Calf', (-0.1, 0.09, 0)), 'Bip01 R Toe0': ('Bip01 R Foot', (-0.1, 0.02, 0.15)),
        'Bip01 Spine': ('Bip01 Pelvis', (0, 1.05, 0)), 'Bip01 Spine1': ('Bip01 Spine', (0, 1.25, 0)),
        'Bip01 Spine2': ('Bip01 Spine1', (0, 1.4, 0)), 'Bip01 Neck': ('Bip01 Spine2', (0, 1.55, 0)),
        'Bip01 Head': ('Bip01 Neck', (0, 1.65, 0.02)),
        'Bip01 L Clavicle': ('Bip01 Spine2', (0.05, 1.5, 0)), 'Bip01 L UpperArm': ('Bip01 L Clavicle', (0.18, 1.5, 0)),
        'Bip01 L Forearm': ('Bip01 L UpperArm', (0.2, 1.2, 0)), 'Bip01 L Hand': ('Bip01 L Forearm', (0.21, 0.92, 0.02)),
        'Bip01 R Clavicle': ('Bip01 Spine2', (-0.05, 1.5, 0)), 'Bip01 R UpperArm': ('Bip01 R Clavicle', (-0.18, 1.5, 0)),
        'Bip01 R Forearm': ('Bip01 R UpperArm', (-0.2, 1.2, 0)), 'Bip01 R Hand': ('Bip01 R Forearm', (-0.21, 0.92, 0.02)),
    }
    ends = {'Bip01 L Toe0': (0.1, 0.02, 0.24), 'Bip01 R Toe0': (-0.1, 0.02, toe_r + 0.09), 'Bip01 Head': (0, 1.85, 0.02),
            'Bip01 L Hand': (0.21, 0.8, 0.03), 'Bip01 R Hand': (-0.21, 0.8, 0.03)}
    J['Bip01 R Toe0'] = ('Bip01 R Foot', (-0.1, 0.02, toe_r)); J['Bip01 L Toe0'] = ('Bip01 L Foot', (0.1, 0.02, toe_l))
    ends['Bip01 L Toe0'] = (0.1, 0.02, toe_l + 0.09)
    if tpose:
        for sg, sd in ((1, 'L'), (-1, 'R')):
            J['Bip01 %s Forearm' % sd] = ('Bip01 %s UpperArm' % sd, (sg * 0.48, 1.5, 0))
            J['Bip01 %s Hand' % sd] = ('Bip01 %s Forearm' % sd, (sg * 0.76, 1.5, 0.02))
            ends['Bip01 %s Hand' % sd] = (sg * 0.88, 1.5, 0.03)
    return _stick(J, ends, 'test_biped')


def _stick(J, ends, name):
    """Skeleton from {name: (parent, world position)} + a tube per bone, skinned rigidly."""
    from motionlab.model import Skeleton
    sk = Skeleton(); idx = {}
    for nm, (par, p) in J.items():
        idx[nm] = sk.add(nm, idx[par] if par else -1, np.array(p, float))
    V, T, Jn = [], [], []; base = 0
    for nm, (par, p) in J.items():
        kids = [k for k, (pp, _) in J.items() if pp == nm]
        tail = J[kids[0]][1] if len(kids) == 1 else ends.get(nm)
        if tail is None or np.linalg.norm(np.subtract(tail, p)) < 1e-3: continue
        v, t = _tube([np.array(p, float), np.array(tail, float)], 0.05 if 'Spine' in nm or nm == 'Bip01 Pelvis' else 0.035, sub=3)
        V.append(v); T.append(t + base); Jn += [idx[nm]] * len(v); base += len(v)
    m = Model(); m.name = name
    m.verts = np.concatenate(V); m.tris = np.concatenate(T)
    n = np.zeros_like(m.verts); fn = np.cross(m.verts[m.tris[:, 1]] - m.verts[m.tris[:, 0]], m.verts[m.tris[:, 2]] - m.verts[m.tris[:, 0]])
    for k in range(3): np.add.at(n, m.tris[:, k], fn)
    m.normals = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    m.uvs = np.zeros((len(m.verts), 2)); m.parts = [{'material': None, 'tri_start': 0, 'tri_count': len(m.tris)}]
    m.skeleton = sk; m.joints = np.array(Jn)[:, None].repeat(4, 1) * 0 + np.array(Jn)[:, None]
    m.weights = np.zeros((len(m.verts), 4)); m.weights[:, 0] = 1.0
    return m
