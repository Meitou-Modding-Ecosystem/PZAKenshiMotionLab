# PZA's Kenshi Motion Lab - the in-memory creature: geometry in world rest space + skeleton + skin. Author: PZA. GPLv3.
import numpy as np
from . import mathx as mx


class Skeleton:
    """Bones in parent-before-child order is NOT assumed; world() walks parents.
    rest_t / rest_q / rest_s are each bone's local transform relative to its parent (glTF / Ogre style)."""

    def __init__(self):
        self.names, self.parents, self.rest_t, self.rest_q, self.rest_s = [], [], [], [], []
        self.root_world = None  # optional per-bone 4x4 applied above parentless bones

    def __len__(self): return len(self.names)

    def index(self, name): return self.names.index(name)

    def order(self):
        seen, out = set(), []
        def visit(i, d=0):
            if i in seen: return
            if d > 1024: raise ValueError('skeleton has a cycle')
            p = self.parents[i]
            if p >= 0: visit(p, d + 1)
            seen.add(i); out.append(i)
        for i in range(len(self.names)): visit(i)
        return out

    def world(self, local=None):
        """World 4x4 per bone; `local` = list of 4x4 local matrices (default: rest)."""
        n = len(self.names)
        L = local if local is not None else [mx.trs(self.rest_t[i], self.rest_q[i], self.rest_s[i]) for i in range(n)]
        W = [None] * n
        for i in self.order():
            p = self.parents[i]
            if p >= 0: W[i] = W[p] @ L[i]
            else:
                rw = self.root_world[i] if self.root_world is not None else np.eye(4)
                W[i] = rw @ L[i]
        return np.array(W)

    def world_rest(self): return self.world()

    def add(self, name, parent, world_pos, world_q=None, world_cache=None):
        """Append a bone given its WORLD rest position (and optional world rotation)."""
        Wp = world_cache if world_cache is not None else (self.world_rest() if len(self.names) else None)
        wq = mx.q_identity() if world_q is None else world_q
        Wb = mx.trs(world_pos, wq)
        L = np.linalg.inv(Wp[parent]) @ Wb if parent >= 0 else Wb
        t, q, s = mx.decompose(L)
        self.names.append(name); self.parents.append(parent)
        self.rest_t.append(t); self.rest_q.append(q); self.rest_s.append(np.ones(3))
        if self.root_world is not None: self.root_world.append(np.eye(4))
        return len(self.names) - 1

    def children(self, i): return [k for k, p in enumerate(self.parents) if p == i]


class Model:
    def __init__(self):
        self.name = 'creature'
        self.verts = self.normals = self.uvs = self.tris = None
        self.parts = []; self.extra_attrs = {}
        self.skeleton = None; self.joints = None; self.weights = None; self.ibm = None
        self.texture = {'factor': (0.75, 0.72, 0.65, 1.0), 'image': None}
        self.source = None

    def skin(self, W, skeleton=None, joints=None, weights=None):
        """Linear-blend skinning of the world-rest geometry with bone world matrices W."""
        sk = skeleton or self.skeleton
        J = self.joints if joints is None else joints
        Wt = self.weights if weights is None else weights
        ibm = np.array([np.linalg.inv(w) for w in sk.world_rest()])
        S = np.einsum('bij,bjk->bik', W, ibm)
        hp = np.c_[self.verts, np.ones(len(self.verts))]
        P = np.zeros_like(self.verts); N = np.zeros_like(self.normals)
        for k in range(J.shape[1]):
            w = Wt[:, k:k + 1]
            if not w.any(): continue
            M = S[J[:, k]]
            P += w * np.einsum('nij,nj->ni', M[:, :3, :], hp)
            N += w * np.einsum('nij,nj->ni', M[:, :3, :3], self.normals)
        tot = Wt.sum(1)
        free = tot <= 1e-9
        P[free] = self.verts[free]; N[free] = self.normals[free]
        return P, mx.norm(N)

    def bounds(self):
        return self.verts.min(0), self.verts.max(0)
