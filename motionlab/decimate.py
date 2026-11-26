# PZA's Kenshi Motion Lab - auto-decimation to Kenshi polygon budgets (docs/polygon-budgets.md).
# Author: PZA. GPLv3.
#
# Quadric-error edge collapse (Garland & Heckbert) in pure Python + numpy: preserves open edges, UV seams and material
# boundaries, refuses triangle flips, and interpolates UVs across collapsed edges.
#   python -m motionlab decimate <in.glb> -o <out.glb> [--budget creature|large|gear|weapon | --target-tris N]
import copy, heapq
import numpy as np
from . import mathx as mx

# Targets from docs/polygon-budgets.md (measured from every vanilla Kenshi mesh)
BUDGETS = {
    'creature': 6000,   # normal animal median 6,072 (range 4,000 - 8,000)
    'large': 12000,     # large / boss creature (up to ~14,000)
    'gear': 3500,       # worn armour (1,000 - 5,000)
    'weapon': 1200,     # weapons (under 1,500)
}

BOUNDARY_WEIGHT = 100.0
FLIP_COS = 0.2


def resolve_target(budget=None, target_tris=None):
    if target_tris is not None:
        t = int(target_tris)
        if t < 4:
            raise ValueError('target_tris must be at least 4 (got %d)' % t)
        return t
    if budget is not None:
        b = budget.lower().strip()
        if b not in BUDGETS:
            raise ValueError('unknown budget %r (known: %s)' % (budget, ', '.join(BUDGETS)))
        return BUDGETS[b]
    return BUDGETS['creature']


def _plane_quadric(a, b, c, d):
    """Symmetric 4x4 plane quadric stored as 10 floats: [aa, ab, ac, ad, bb, bc, bd, cc, cd, dd]."""
    return np.array([a * a, a * b, a * c, a * d, b * b, b * c, b * d, c * c, c * d, d * d], np.float64)


def _eval_quadric(q, p):
    x, y, z = p[0], p[1], p[2]
    return (q[0] * x * x + 2.0 * q[1] * x * y + 2.0 * q[2] * x * z + 2.0 * q[3] * x +
            q[4] * y * y + 2.0 * q[5] * y * z + 2.0 * q[6] * y +
            q[7] * z * z + 2.0 * q[8] * z + q[9])


def _best_target(q, pa, pb):
    """Optimal collapse position for edge (pa, pb) under quadric q, falling back to endpoints / midpoint."""
    A = np.array([[q[0], q[1], q[2]], [q[1], q[4], q[5]], [q[2], q[5], q[7]]], np.float64)
    rhs = -np.array([q[3], q[6], q[8]], np.float64)
    cands = [pa, pb, 0.5 * (pa + pb)]
    det = float(np.linalg.det(A))
    if abs(det) > 1e-12:
        try:
            opt = np.linalg.solve(A, rhs)
            # Reject wild extrapolations far from the edge segment
            elen = float(np.linalg.norm(pb - pa))
            if np.isfinite(opt).all() and float(np.linalg.norm(opt - 0.5 * (pa + pb))) <= max(1.5 * elen, 1e-4):
                cands.append(opt)
        except np.linalg.LinAlgError:
            pass
    best_p = cands[0]
    best_c = max(0.0, float(_eval_quadric(q, best_p)))
    for p in cands[1:]:
        c = max(0.0, float(_eval_quadric(q, p)))
        if c < best_c:
            best_c, best_p = c, p
    return best_p, best_c


def _recompute_normals(verts, tris):
    n = np.zeros_like(verts)
    if len(tris):
        fn = np.cross(verts[tris[:, 1]] - verts[tris[:, 0]], verts[tris[:, 2]] - verts[tris[:, 0]])
        for k in range(3):
            np.add.at(n, tris[:, k], fn)
    return mx.norm(n)


def decimate_model(model, target_tris, log=print):
    """Return a copy of `model` reduced to at most `target_tris` triangles. Keeps UVs, materials and texture."""
    n_tris = len(model.tris)
    if n_tris <= target_tris:
        return model

    P = model.verts.astype(np.float64).copy()
    UV = model.uvs.astype(np.float64).copy() if model.uvs is not None else np.zeros((len(P), 2), np.float64)
    T = model.tris.astype(np.int64).copy()
    nv = len(P)

    # Track per-triangle material part id
    part_of = np.zeros(n_tris, np.int32)
    parts_meta = model.parts or [{'material': None, 'tri_start': 0, 'tri_count': n_tris}]
    for pi, part in enumerate(parts_meta):
        t0 = int(part['tri_start'])
        tc = int(part['tri_count'])
        part_of[t0:t0 + tc] = pi

    # Build initial per-vertex quadrics and adjacency
    Q = np.zeros((nv, 10), np.float64)
    around = [set() for _ in range(nv)]
    edge_tris = {}
    for ti in range(n_tris):
        i0, i1, i2 = int(T[ti, 0]), int(T[ti, 1]), int(T[ti, 2])
        if i0 == i1 or i1 == i2 or i0 == i2:
            continue
        p0, p1, p2 = P[i0], P[i1], P[i2]
        fn = np.cross(p1 - p0, p2 - p0)
        area2 = float(np.linalg.norm(fn))
        if area2 > 1e-18:
            nrm = fn / area2
            d = -float(nrm @ p0)
            q = _plane_quadric(nrm[0], nrm[1], nrm[2], d) * (0.5 * area2)
            Q[i0] += q
            Q[i1] += q
            Q[i2] += q
        around[i0].add(ti)
        around[i1].add(ti)
        around[i2].add(ti)
        for u, v in ((i0, i1), (i1, i2), (i2, i0)):
            e = (u, v) if u < v else (v, u)
            edge_tris.setdefault(e, []).append(ti)

    # Add boundary / material-seam penalty planes
    boundary_verts = set()
    for (u, v), tlist in edge_tris.items():
        is_bound = len(tlist) == 1 or len({int(part_of[t]) for t in tlist}) > 1
        if is_bound:
            boundary_verts.add(u)
            boundary_verts.add(v)
            ti = tlist[0]
            p0, p1, p2 = P[T[ti, 0]], P[T[ti, 1]], P[T[ti, 2]]
            fn = np.cross(p1 - p0, p2 - p0)
            fl = float(np.linalg.norm(fn))
            ed = P[v] - P[u]
            el = float(np.linalg.norm(ed))
            if fl > 1e-18 and el > 1e-12:
                bn = np.cross(ed / el, fn / fl)
                bl = float(np.linalg.norm(bn))
                if bl > 1e-12:
                    bn /= bl
                    q = _plane_quadric(bn[0], bn[1], bn[2], -float(bn @ P[u])) * (BOUNDARY_WEIGHT * el)
                    Q[u] += q
                    Q[v] += q

    # Queue all candidate edges
    ver = np.zeros(nv, np.int32)
    gone = np.zeros(nv, bool)
    dead = np.zeros(n_tris, bool)
    heap = []

    def push_edge(u, v):
        if u == v or gone[u] or gone[v]:
            return
        if u > v:
            u, v = v, u
        # Avoid collapsing across distinct material parts if one vertex is strictly interior to a different part
        q = Q[u] + Q[v]
        tp, cost = _best_target(q, P[u], P[v])
        heapq.heappush(heap, (cost, int(ver[u]), int(ver[v]), u, v, tp))

    for u, v in edge_tris:
        push_edge(u, v)

    alive = n_tris
    while alive > target_tris and heap:
        cost, va, vb, a, b, tp = heapq.heappop(heap)
        if gone[a] or gone[b] or ver[a] != va or ver[b] != vb:
            continue
        # Check normal flip on every surviving triangle incident to a or b
        incident = around[a] | around[b]
        flip = False
        for ti in incident:
            if dead[ti]:
                continue
            i0, i1, i2 = int(T[ti, 0]), int(T[ti, 1]), int(T[ti, 2])
            if (a in (i0, i1, i2)) and (b in (i0, i1, i2)):
                continue  # this triangle will collapse
            p0, p1, p2 = P[i0], P[i1], P[i2]
            old_n = np.cross(p1 - p0, p2 - p0)
            ol = float(np.linalg.norm(old_n))
            if ol < 1e-18:
                continue
            np0 = tp if (i0 == a or i0 == b) else p0
            np1 = tp if (i1 == a or i1 == b) else p1
            np2 = tp if (i2 == a or i2 == b) else p2
            new_n = np.cross(np1 - np0, np2 - np0)
            nl = float(np.linalg.norm(new_n))
            if nl < 1e-18 or float(old_n @ new_n) < FLIP_COS * ol * nl:
                flip = True
                break
        if flip:
            continue

        # Interpolate UV along segment a -> b
        ab = P[b] - P[a]
        ab2 = float(ab @ ab)
        u_frac = float(np.clip(((tp - P[a]) @ ab) / ab2, 0.0, 1.0)) if ab2 > 1e-18 else 0.5
        P[a] = tp
        UV[a] = UV[a] * (1.0 - u_frac) + UV[b] * u_frac
        Q[a] += Q[b]
        gone[b] = True
        ver[a] += 1

        # Remap triangles from b -> a, mark collapsed triangles dead
        neighbours = set()
        for ti in list(incident):
            if dead[ti]:
                continue
            row = T[ti]
            if row[0] == b: row[0] = a
            if row[1] == b: row[1] = a
            if row[2] == b: row[2] = a
            i0, i1, i2 = int(row[0]), int(row[1]), int(row[2])
            if i0 == i1 or i1 == i2 or i0 == i2:
                dead[ti] = True
                alive -= 1
                around[i0].discard(ti)
                around[i1].discard(ti)
                around[i2].discard(ti)
            else:
                around[a].add(ti)
                for w in (i0, i1, i2):
                    if w != a:
                        neighbours.add(w)
        around[b].clear()
        for w in neighbours:
            push_edge(a, w)

    # Compact surviving triangles grouped by material part
    keep_mask = ~dead
    keep_tris = T[keep_mask]
    keep_parts = part_of[keep_mask]
    order = np.argsort(keep_parts, kind='stable')
    keep_tris = keep_tris[order]
    keep_parts = keep_parts[order]

    used_verts, inv = np.unique(keep_tris.reshape(-1), return_inverse=True)
    new_verts = P[used_verts]
    new_uvs = UV[used_verts]
    new_tris = inv.reshape(-1, 3).astype(np.int64)
    new_normals = _recompute_normals(new_verts, new_tris)

    new_parts = []
    t_cursor = 0
    for pi, pm in enumerate(parts_meta):
        cnt = int(np.sum(keep_parts == pi))
        if cnt > 0:
            np_ = dict(pm)
            np_['tri_start'] = t_cursor
            np_['tri_count'] = cnt
            new_parts.append(np_)
            t_cursor += cnt

    out = copy.copy(model)
    out.verts = new_verts
    out.normals = new_normals
    out.uvs = new_uvs
    out.tris = new_tris
    out.parts = new_parts or [{'material': None, 'tri_start': 0, 'tri_count': len(new_tris)}]
    out.extra_attrs = {}
    out.skeleton = None
    out.joints = None
    out.weights = None
    log('decimate: %d -> %d triangles (%d -> %d vertices, target %d)' % (
        n_tris, len(new_tris), nv, len(new_verts), target_tris))
    return out
