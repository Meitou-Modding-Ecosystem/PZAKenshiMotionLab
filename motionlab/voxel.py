# PZA's Kenshi Motion Lab - voxel body analysis (numpy only): solid from a non-watertight mesh, thick body by opening,
# limbs as connected parts of the rest, centre lines by BFS levels. Author: PZA. GPLv3.
from collections import deque
import numpy as np


def _shift_or(a, axis, r):
    out = a.copy()
    for s in range(1, r + 1):
        sl_p = [slice(None)] * 3; sl_m = [slice(None)] * 3
        dst_p = [slice(None)] * 3; dst_m = [slice(None)] * 3
        sl_p[axis] = slice(0, -s); dst_p[axis] = slice(s, None)
        sl_m[axis] = slice(s, None); dst_m[axis] = slice(0, -s)
        out[tuple(dst_p)] |= a[tuple(sl_p)]; out[tuple(dst_m)] |= a[tuple(sl_m)]
    return out


def dilate(a, r=1):
    for ax in range(3): a = _shift_or(a, ax, r)
    return a


def erode(a, r=1):
    return ~dilate(~a, r)


def dilate6(a):
    out = a.copy()
    out[1:] |= a[:-1]; out[:-1] |= a[1:]
    out[:, 1:] |= a[:, :-1]; out[:, :-1] |= a[:, 1:]
    out[:, :, 1:] |= a[:, :, :-1]; out[:, :, :-1] |= a[:, :, 1:]
    return out


def sphere_erode(a, r):
    """Erode by a ball of radius r voxels (approximated by r single steps alternating cube / cross)."""
    for i in range(r):
        a = ~dilate(~a, 1) if i % 2 == 0 else ~dilate6(~a)
    return a


def sphere_dilate(a, r):
    for i in range(r):
        a = dilate(a, 1) if i % 2 == 0 else dilate6(a)
    return a


class Grid:
    def __init__(self, P, tris, voxel, pad=3):
        self.vs = voxel
        self.lo = P.min(0) - pad * voxel
        self.shape = tuple(np.ceil((P.max(0) - self.lo) / voxel).astype(int) + pad + 1)
        surf = np.zeros(self.shape, bool)
        tri = P[tris]
        L = np.max(np.linalg.norm(tri - np.roll(tri, 1, axis=1), axis=2), axis=1)
        nsub = np.clip(np.ceil(L / (0.5 * voxel)).astype(int), 1, 64)
        for n in np.unique(nsub):
            sel = nsub == n
            B = np.array([((n - i - j) / n, i / n, j / n) for i in range(n + 1) for j in range(n + 1 - i)])
            X = np.einsum('sk,tkc->tsc', B, tri[sel]).reshape(-1, 3)
            ijk = self.index(X)
            surf[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True
        self.surface = surf
        closed = dilate(surf, 1)                       # closes small holes in the shell
        outside = np.zeros(self.shape, bool)
        outside[0] = outside[-1] = True; outside[:, 0] = outside[:, -1] = True; outside[:, :, 0] = outside[:, :, -1] = True
        outside &= ~closed
        free = ~closed
        for _ in range(4 * max(self.shape)):
            nxt = dilate6(outside) & free
            if nxt.sum() == outside.sum(): break
            outside = nxt
        self.solid = erode(~outside, 1) | surf         # undo the closing's 1-voxel growth, keep the shell itself

    def index(self, X):
        return np.clip(((np.asarray(X) - self.lo) / self.vs).astype(int), 0, np.array(self.shape) - 1)

    def centre(self, ijk):
        return self.lo + (np.asarray(ijk) + 0.5) * self.vs

    def at(self, mask, X):
        ijk = self.index(X); return mask[ijk[..., 0], ijk[..., 1], ijk[..., 2]]


def components(mask):
    """6-connected components of a boolean grid (BFS). Returns list of (N,3) int arrays, largest first."""
    seen = np.zeros(mask.shape, bool)
    pts = np.argwhere(mask)
    comps = []
    sh = mask.shape
    for p in pts:
        p = tuple(p)
        if seen[p]: continue
        q = deque([p]); seen[p] = True; cur = []
        while q:
            x, y, z = q.popleft(); cur.append((x, y, z))
            for dx, dy, dz in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
                a, b, c = x + dx, y + dy, z + dz
                if 0 <= a < sh[0] and 0 <= b < sh[1] and 0 <= c < sh[2] and mask[a, b, c] and not seen[a, b, c]:
                    seen[a, b, c] = True; q.append((a, b, c))
        comps.append(np.array(cur))
    comps.sort(key=len, reverse=True)
    return comps


def bfs_levels(comp, seeds):
    """BFS distance (26-neighbour steps weighted 1 / sqrt2 / sqrt3 approximated by Dijkstra-free BFS on 26) inside a
    component from seed voxels. Returns dict voxel -> distance in voxels (float)."""
    import heapq
    inside = set(map(tuple, comp))
    d = {}
    h = []
    for s in seeds:
        s = tuple(s); d[s] = 0.0; h.append((0.0, s))
    heapq.heapify(h)
    nb = [(a, b, c, (a * a + b * b + c * c) ** 0.5) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1) if a or b or c]
    while h:
        dv, v = heapq.heappop(h)
        if dv > d.get(v, 1e18): continue
        for a, b, c, w in nb:
            u = (v[0] + a, v[1] + b, v[2] + c)
            if u in inside:
                nd = dv + w
                if nd < d.get(u, 1e18): d[u] = nd; heapq.heappush(h, (nd, u))
    return d
