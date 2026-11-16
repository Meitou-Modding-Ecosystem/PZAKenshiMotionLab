# PZA's Kenshi Motion Lab - Kenshi / Ogre .mesh (+ .skeleton) into a Model. Author: PZA. GPLv3.
# The readers next to this file are PZA's handoff tools (mesh_read.py, skeleton_read.py), copied.
import os
import numpy as np
from .. import mathx as mx
from ..model import Model, Skeleton
from . import mesh_read, skeleton_read

KENSHI_UNITS_PER_METRE = 10.0


def _sub_arrays(g):
    n = g['count']
    pos = nrm = uv = None
    for el in g['elements']:
        src, typ, sem, off, idx = el
        if sem == 1 and pos is None: pos = np.array([mesh_read.element(g, i, el)[:3] for i in range(n)], float)
        elif sem == 4 and nrm is None: nrm = np.array([mesh_read.element(g, i, el)[:3] for i in range(n)], float)
        elif sem == 7 and uv is None and idx == 0: uv = np.array([mesh_read.element(g, i, el)[:2] for i in range(n)], float)
    if pos is None: raise ValueError('submesh has no positions')
    if nrm is None: nrm = np.tile([0, 1, 0], (n, 1)).astype(float)
    if uv is None: uv = np.zeros((n, 2))
    return pos, nrm, uv


def load_mesh(path, skeleton_path=None, units_per_metre=KENSHI_UNITS_PER_METRE):
    m = mesh_read.read(path)
    P, N, UV, T, parts, BW = [], [], [], [], [], []
    base = 0; shared_done = False
    for s in m['subs']:
        g = s['geometry'] or m['shared']
        if g is None: continue
        if s['geometry'] is None:
            if shared_done:
                parts.append({'material': None, 'tri_start': sum(len(x) for x in T), 'tri_count': len(s['indices']) // 3})
                T.append(np.array(s['indices'], np.int64).reshape(-1, 3) + shared_base); continue
            shared_done = True; shared_base = base
        pos, nrm, uv = _sub_arrays(g)
        parts.append({'material': None, 'tri_start': sum(len(x) for x in T), 'tri_count': len(s['indices']) // 3, 'ogre_material': s['material']})
        T.append(np.array(s['indices'], np.int64).reshape(-1, 3) + base)
        for vi, bone, w in s['bones']: BW.append((vi + base, bone, w))
        P.append(pos); N.append(nrm); UV.append(uv); base += len(pos)
    mo = Model()
    mo.name = os.path.splitext(os.path.basename(path))[0]
    k = 1.0 / units_per_metre
    mo.verts = np.concatenate(P) * k; mo.normals = mx.norm(np.concatenate(N)); mo.uvs = np.concatenate(UV); mo.tris = np.concatenate(T)
    mo.parts = parts; mo.source = {'kind': 'ogre', 'path': path, 'units_per_metre': units_per_metre}
    sp = skeleton_path
    if sp is None and m.get('skeleton'):
        cand = os.path.join(os.path.dirname(path), os.path.basename(m['skeleton']))
        sp = cand if os.path.exists(cand) else None
    if sp is None:                                      # older meshes: the link is not always read; try <name>.skeleton
        cand = os.path.splitext(path)[0] + '.skeleton'
        sp = cand if os.path.exists(cand) else None
    if sp and BW:
        ver, bones, parents, anims = skeleton_read.read(sp)
        handles = sorted(bones)
        hidx = {h: i for i, h in enumerate(handles)}
        sk = Skeleton()
        for h in handles:
            name, pos, qwxyz, sc = bones[h]
            sk.names.append(name); sk.parents.append(hidx[parents[h]] if h in parents else -1)
            sk.rest_t.append(np.array(pos, float) * k); sk.rest_q.append(mx.q_from_wxyz(qwxyz)); sk.rest_s.append(np.array(sc, float))
        nv = len(mo.verts)
        J = np.zeros((nv, 4), np.int64); Wt = np.zeros((nv, 4))
        cnt = np.zeros(nv, int)
        for vi, bone, w in BW:
            c = cnt[vi]
            if c < 4 and bone in hidx: J[vi, c] = hidx[bone]; Wt[vi, c] = w; cnt[vi] += 1
        s_ = Wt.sum(1, keepdims=True); Wt = np.where(s_ > 0, Wt / np.maximum(s_, 1e-12), 0)
        mo.skeleton, mo.joints, mo.weights = sk, J, Wt
        mo.ogre_animations = [a[0] for a in anims]
    return mo
