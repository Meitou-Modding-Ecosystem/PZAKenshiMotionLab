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
