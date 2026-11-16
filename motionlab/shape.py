# PZA's Kenshi Motion Lab - mesh shape analysis for the auto-rig: weld, surface graph, geodesic distances, extremities,
# limb centre lines from geodesic rings. Author: PZA. GPLv3. Pure numpy + heapq (no scipy).
import heapq
import numpy as np


class Surface:
    def __init__(self, model, weld=1e-5):
        P = model.verts
        key = np.round(P / weld).astype(np.int64)
        u, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
        self.inv = inv.reshape(-1)                 # original vertex -> welded vertex
        self.P = P[first]                          # welded positions
        T = self.inv[model.tris]
        e = np.r_[T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]
        e = np.sort(e, 1); e = e[e[:, 0] != e[:, 1]]
        e = np.unique(e, axis=0)
        a = np.r_[e[:, 0], e[:, 1]]; b = np.r_[e[:, 1], e[:, 0]]
        o = np.argsort(a, kind='stable'); a, b = a[o], b[o]
        self.ptr = np.searchsorted(a, np.arange(len(self.P) + 1))
        self.nb = b
        self.w = np.linalg.norm(self.P[a] - self.P[b], axis=1)
        self.n = len(self.P)

    def geodesic(self, sources, limit=np.inf, pred=False):
        d = np.full(self.n, np.inf); pr = np.full(self.n, -1, np.int64)
        h = []
        for s in np.atleast_1d(sources):
            d[s] = 0.0; h.append((0.0, int(s)))
        heapq.heapify(h)
        ptr, nb, w = self.ptr, self.nb, self.w
        while h:
            dv, v = heapq.heappop(h)
            if dv > d[v] or dv > limit: continue
            for k in range(ptr[v], ptr[v + 1]):
                u = nb[k]; nd = dv + w[k]
                if nd < d[u]:
                    d[u] = nd; pr[u] = v; heapq.heappush(h, (nd, u))
        return (d, pr) if pred else d

    @staticmethod
    def path(pr, v):
        out = [v]
        while pr[out[-1]] >= 0 and len(out) < 1_000_000: out.append(int(pr[out[-1]]))
        return out

    def nearest(self, x):
        return int(np.argmin(np.linalg.norm(self.P - np.asarray(x), axis=1)))

    def extremities(self, dist, radius):
        """Vertices that are the geodesic-farthest point within `radius` of themselves (tips of limbs)."""
        order = np.argsort(-dist)
        taken = np.zeros(self.n, bool); tips = []
        for v in order:
            if not np.isfinite(dist[v]) or taken[v]: continue
            tips.append(int(v))
            near = self.geodesic(v, radius)
            taken |= np.isfinite(near)
        return tips

    def limb_line(self, tip, step, max_len, grow=1.8, min_rings=3):
        """Centre line from a tip inward: centroids of geodesic rings (level sets) around the tip.
        Stops where the ring gets much wider than the limb so far (the limb meets the body).
        Returns (points tip->root, ring radii)."""
        d = self.geodesic(tip, max_len + step)
        pts, rad = [], []
        k = 0
        while (k + 1) * step <= max_len:
            sel = (d >= k * step) & (d < (k + 1) * step)
            if sel.sum() < 3: k += 1; continue
            Q = self.P[sel]; c = Q.mean(0)
            r = float(np.median(np.linalg.norm(Q - c, axis=1)))
            if len(rad) >= min_rings and r > grow * max(np.median(rad), 1e-6) + 0.02 * step:
                break
            pts.append(c); rad.append(r); k += 1
        return np.array(pts), np.array(rad)
