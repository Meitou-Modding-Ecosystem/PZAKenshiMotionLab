# PZA's Kenshi Motion Lab - auto-rig: find the body plan from the mesh shape, place a clean named skeleton, bind it.
# Author: PZA. GPLv3.
#
# Steps (each one writes what it found into the landmarks, which a modder can edit and feed back with --landmarks):
#   1. frame     : up = +Y, forward = the body's long axis turned to +Z; the mesh is turned / moved so the body
#                  stands on y = 0, centred, facing +Z (glTF's forward). The turn is reported.
#   2. limbs     : geodesic distance on the surface from the body core; its far tips are limb ends. Tips whose paths to
#                  the core join far from the core are one limb (claws of one foot). Each limb's centre line comes
#                  from geodesic rings around its tip, up to where the ring widens into the body.
#   3. classify  : legs (tips near the ground), wings (tips high and behind), antennae (tips high and in front).
#   4. spine     : the body left after removing limbs, sliced along the forward axis (head / thorax / abdomen).
#   5. skeleton  : root, body, abdomen_01.., neck, head, leg_<L|R><n>_01.., wing_<L|R>_01.., antenna_<L|R>_01..
#   6. bind      : geodesic distance from each bone (seeded along the bone), 3 weights per vertex (Kenshi's GPU reads 3).
import copy, json
import numpy as np
from . import mathx as mx
from .model import Skeleton
from .shape import Surface
from . import voxel

VERSION = '0.1.0'


# ------------------------------------------------------------------ frame

def transformed(model, R, t):
    """Copy of the model with every point p -> R p + t (normals / tangents turned)."""
    m = copy.copy(model)
    m.verts = model.verts @ R.T + t
    m.normals = model.normals @ R.T
    m.extra_attrs = {}
    for k, v in (model.extra_attrs or {}).items():
        v = v.copy()
        if k.startswith('TANGENT'): v[:, :3] = v[:, :3] @ R.T
        m.extra_attrs[k] = v
    m.skeleton = None; m.joints = None; m.weights = None
    return m


def yaw_matrix(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


# ------------------------------------------------------------------ limbs

def _lca(pr, a, b):
    seen = set(); x = a
    while x >= 0: seen.add(x); x = pr[x]
    x = b
    while x >= 0 and x not in seen: x = pr[x]
    return x


def find_limbs(S, d, pr, size, min_frac=0.45, same_frac=0.5):
    dmax = np.max(d[np.isfinite(d)])
    tips = [t for t in S.extremities(d, 0.05 * size) if d[t] >= min_frac * dmax]
    groups = []
    for t in tips:
        for g in groups:
            j = _lca(pr, t, g[0])
            if j >= 0 and d[j] >= same_frac * min(d[t], d[g[0]]):
                g.append(t); break
        else:
            groups.append([t])
    return groups


def limb_centre(S, tip, size, max_len):
    pts, rad = S.limb_line(tip, step=0.025 * size, max_len=max_len, grow=2.2, min_rings=4)
    return pts, rad


def fit_polyline(points, k):
    """Choose k+1 of the points (first and last kept) so the k segments fit the polyline best (DP, least squares)."""
    n = len(points)
    if n <= k + 1:
        idx = np.linspace(0, n - 1, k + 1)
        return np.array([points[int(round(i))] if float(i).is_integer() else
                         points[int(i)] * (1 - (i % 1)) + points[int(i) + 1] * (i % 1) for i in idx])
    def cost(i, j):
        a, b = points[i], points[j]; ab = b - a; L = ab @ ab
        if L < 1e-12: return 0.0
        Q = points[i:j + 1]; t = np.clip((Q - a) @ ab / L, 0, 1)
        return float(np.sum(np.linalg.norm(Q - (a + t[:, None] * ab), axis=1) ** 2))
    INF = 1e18
    C = np.full((k + 1, n), INF); B = np.zeros((k + 1, n), int); C[0, 0] = 0
    minseg = max(1, n // (3 * k))
    for s in range(1, k + 1):
        for j in range(1, n):
            best, arg = INF, -1
            for i in range(0, j - minseg + 1):
                if C[s - 1, i] >= INF: continue
                c = C[s - 1, i] + cost(i, j)
                if c < best: best, arg = c, i
            C[s, j] = best; B[s, j] = arg
    idx = [n - 1]
    for s in range(k, 0, -1): idx.append(B[s, idx[-1]])
    return points[idx[::-1]]


# ------------------------------------------------------------------ analyse

def analyse(model, plan='insect', legs_bones=4, log=print):
    """Return (straightened model, landmarks dict)."""
    m = model
    S = Surface(m)
    lo, hi = S.P.min(0), S.P.max(0); H = hi[1] - lo[1]
    upper = S.P[S.P[:, 1] > lo[1] + 0.55 * H]
    c = upper.mean(0)
    X = upper[:, [0, 2]] - c[[0, 2]]
    w, v = np.linalg.eigh(X.T @ X); ax = v[:, -1]
    if ax[1] < 0: ax = -ax
    yaw = np.arctan2(ax[0], ax[1])
    R = yaw_matrix(-yaw); t = -(R @ np.array([c[0], 0, c[2]])); t[1] = -lo[1]
    turn = {'yaw_deg': float(np.degrees(-yaw)), 'move': [float(x) for x in t]}
    for it in range(2):
        m2 = transformed(model, R, t)
        S = Surface(m2)
        lm = _analyse_frame(m2, S, legs_bones, log)
        feet = [l for l in lm['legs']]
        Ls = sorted([l for l in feet if l['side'] == 'L'], key=lambda l: -l['points'][-1][2])
        Rs = sorted([l for l in feet if l['side'] == 'R'], key=lambda l: -l['points'][-1][2])
        if it == 1 or not Ls or not Rs: break
        mids = np.array([(np.array(a['points'][-1]) + np.array(b['points'][-1])) / 2 for a, b in zip(Ls, Rs)])
        hips = np.array([(np.array(a['points'][0]) + np.array(b['points'][0])) / 2 for a, b in zip(Ls, Rs)])
        q = np.r_[mids, hips]
        if len(q) >= 2:
            A = np.c_[np.ones(len(q)), q[:, 2]]
            (a0, a1), *_ = np.linalg.lstsq(A, q[:, 0], rcond=None)
            dyaw = np.arctan(a1)
            R2 = yaw_matrix(-dyaw)
            R = R2 @ R; t = R2 @ (t - np.array([a0, 0, 0]))
            t[1] = 0
            turn = {'yaw_deg': float(np.degrees(np.arctan2(R[0, 2], R[0, 0]))), 'move': [float(x) for x in t]}
            log('frame: mirror fit of %d leg pairs: turned %.1f deg more, centred %.3f' % (len(mids), np.degrees(-dyaw), -a0))
    m2.verts[:, 1] -= m2.verts[:, 1].min(); t[1] -= 0  # keep feet on y=0
    lm['frame'] = {'turn_about_y_deg': turn['yaw_deg'], 'R': R.tolist(), 't': [float(x) for x in t],
                   'note': 'mesh turned so the body faces +Z, centred on the leg pairs, lowest point on y=0'}
    lm['plan'] = plan; lm['version'] = VERSION
    return m2, lm


_NB26 = [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1) if a or b or c]


def _descent(tip, dist):
    path = [tip]; cur = tip
    for _ in range(100000):
        best = cur
        for a, b, c in _NB26:
            u = (cur[0] + a, cur[1] + b, cur[2] + c)
            if u in dist and dist[u] < dist[best]: best = u
        if best == cur: break
        cur = best; path.append(cur)
    return path


def _limb_tips(comp_set, dist, min_len_vox, min_offshoot_vox, join_vox=3):
    """Far ends of one limb part, farthest first. A later end counts only as a real offshoot: its descent path must run
    min_offshoot_vox before it comes within join_vox of a path already taken (spikes along a leg, claws of one foot
    and the corners of a wing sheet do not)."""
    order = sorted(dist.items(), key=lambda kv: -kv[1])
    tips = []; onpath = set()
    for v, dv in order:
        if dv < min_len_vox: break
        if v in onpath: continue
        path = _descent(v, dist)
        join = None
        for p in path:
            if p in onpath: join = p; break
        offshoot = dv - (dist[join] if join is not None else 0.0)
        if join is None or offshoot >= min_offshoot_vox:
            tips.append(v)
        # mark this path (thickened) as taken either way, so its neighbours are judged against it
        for p in path:
            for a in range(-join_vox, join_vox + 1):
                for b in range(-join_vox, join_vox + 1):
                    for c in range(-join_vox, join_vox + 1):
                        onpath.add((p[0] + a, p[1] + b, p[2] + c))
    return tips


def _trace(tip, dist, G, band_vox=1.5, radius_vox=5.0):
    """Centre line tip -> attachment: steepest descent over the distance field, then the centroid of the voxels in
    each distance band near the path."""
    path = [tip]; cur = tip
    nb = [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1) if a or b or c]
    for _ in range(100000):
        best = cur
        for a, b, c in nb:
            u = (cur[0] + a, cur[1] + b, cur[2] + c)
            if u in dist and dist[u] < dist[best]: best = u
        if best == cur: break
        cur = best; path.append(cur)
    pd = np.array([dist[p] for p in path]); path = np.array(path, float)
    keys = np.array(list(dist.keys()), float); vals = np.array(list(dist.values()))
    out = []
    for lev in np.arange(dist[tip], -0.01, -band_vox):
        p = path[int(np.argmin(np.abs(pd - lev)))]
        sel = (np.abs(vals - lev) <= band_vox) & (np.sum((keys - p) ** 2, 1) <= radius_vox ** 2)
        out.append(keys[sel].mean(0) if sel.sum() >= 2 else p)
    return np.array([G.centre(x) for x in out])


def _smooth(points, it=2):
    p = np.array(points, float)
    for _ in range(it):
        q = p.copy(); q[1:-1] = 0.25 * p[:-2] + 0.5 * p[1:-1] + 0.25 * p[2:]; p = q
    return p


def _analyse_frame(m, S, legs_bones, log):
    P = m.verts
    lo, hi = P.min(0), P.max(0); H = hi[1] - lo[1]; size = float(np.linalg.norm(hi - lo))
    vs = size / 140.0
    G = voxel.Grid(P, m.tris, vs)
    r = max(2, int(round(0.03 * size / vs)))
    body = voxel.sphere_dilate(voxel.sphere_erode(G.solid, r), r) & G.solid
    keep = voxel.components(body)
    body = np.zeros_like(body)
    if keep:
        k = keep[0]; body[k[:, 0], k[:, 1], k[:, 2]] = True                # the body is the largest thick part
    near_body = voxel.dilate(body, 1)
    rest = G.solid & ~near_body
    attach_zone = voxel.dilate(near_body, 1)
    comps = voxel.components(rest)
    labels = np.full(G.shape, -1, np.int32)
    labels[body] = -2
    for ci, comp in enumerate(comps): labels[comp[:, 0], comp[:, 1], comp[:, 2]] = ci
    legs, wings, ants, other = [], [], [], []
    min_len = 0.12 * H / vs
    sup = 0.12 * H / vs
    for ci, comp in enumerate(comps):
        if len(comp) < 30: continue
        cs = set(map(tuple, comp))
        seeds = [v for v in cs if attach_zone[v]]
        if not seeds:                                   # a separate piece (e.g. a wing modelled apart): attach it
            grow = attach_zone                          # where it comes closest to the body
            for _ in range(int(0.15 * size / vs)):
                grow = voxel.dilate(grow, 1)
                seeds = [v for v in cs if grow[v]]
                if seeds: break
        if not seeds: continue
        dist = voxel.bfs_levels(comp, seeds)
        for tip in _limb_tips(cs, dist, min_len, 0.18 * H / vs):
            line = _smooth(_trace(tip, dist, G))[::-1]                           # attachment -> tip
            if G.centre(tip)[1] < lo[1] + 0.12 * H:                              # foot: end on the lowest claw
                cl = np.array([v for v in cs if (v[0]-tip[0])**2 + (v[1]-tip[1])**2 + (v[2]-tip[2])**2 <= sup ** 2])
                line[-1] = G.centre(cl[np.argmin(cl[:, 1])])
            p = line[-1]; root = line[0]
            length = float(np.sum(np.linalg.norm(np.diff(line, axis=0), axis=1)))
            e = {'points': line.tolist(), 'length': length, 'part_voxels': len(comp), '_part': ci}
            if p[1] < lo[1] + 0.3 * H: legs.append(e)
            elif root[1] > lo[1] + 0.55 * H and p[2] < root[2] + 0.05 * size and p[1] > lo[1] + 0.45 * H: wings.append(e)
            elif p[2] > 0 and p[1] > lo[1] + 0.4 * H: ants.append(e)
            else: other.append(e)
    # spine: centroid of body voxels per slice along z, smoothed
    bv = G.centre(np.argwhere(body))
    zmin, zmax = bv[:, 2].min(), bv[:, 2].max()
    nb = 28; edges = np.linspace(zmin, zmax, nb + 1); spine, area = [], []
    for i in range(nb):
        s = bv[(bv[:, 2] >= edges[i]) & (bv[:, 2] <= edges[i + 1])]
        if len(s) >= 3:
            spine.append([float(np.mean(s[:, 0])), float(np.mean(s[:, 1])), float((edges[i] + edges[i + 1]) / 2)]); area.append(len(s))
    spine = _smooth(np.array(spine), 3); area = np.array(area, float)
    # head / thorax / abdomen: the thorax is the thickest stretch; neck and waist are where the smoothed
    # cross-section falls below a fraction of its peak, going forward and back from the peak
    zs = spine[:, 2]
    a_s = np.convolve(np.r_[area[:1].repeat(2), area, area[-1:].repeat(2)], np.ones(5) / 5, 'valid')
    pk = int(np.argmax(a_s)); amax = a_s[pk]
    i = pk
    while i + 1 < len(zs) and a_s[i + 1] >= 0.5 * amax: i += 1
    neck_z = float(zs[min(i + 1, len(zs) - 1)] - 0.5 * (zs[1] - zs[0]))
    i = pk
    while i - 1 >= 0 and a_s[i - 1] >= 0.6 * amax: i -= 1
    waist_z = float(zs[max(i - 1, 0)] + 0.5 * (zs[1] - zs[0]))
    # insect rule: the walking legs hang from the thorax, so the waist sits just behind the rearmost planted hip;
    # and the abdomen is at least 30 % of the body (a waist found far back leaves the abdomen bones a stub)
    L_body = float(zmax - zmin)
    planted = [l['points'][0][2] for l in legs if l['points'][-1][1] <= lo[1] + 0.15 * H]
    cands = [waist_z, float(zmin + 0.30 * L_body)]
    if planted: cands.append(float(min(planted) - 0.05 * L_body))
    waist_z = min(max(cands), neck_z - 0.2 * L_body)
    def cx_at(z): return float(np.interp(z, zs, spine[:, 0]))
    def side_of(e):
        return 'L' if e['points'][-1][0] > cx_at(e['points'][0][2]) else 'R'
    for l in legs: l['side'] = side_of(l)
    for side in 'LR':
        sl = sorted([l for l in legs if l['side'] == side], key=lambda l: -l['points'][0][2])
        for i, l in enumerate(sl): l['name'] = 'leg_%s%d' % (side, i + 1)
    # wings / antennae: keep the long ones (spikes are short), pair them by their roots' x
    def keep_long(grp, frac, key='length'):
        if not grp: return grp
        mx_ = max(e[key] for e in grp)
        return sorted([e for e in grp if e[key] >= frac * mx_], key=lambda e: -e[key])
    # insect wings hinge on the top of the thorax's rear part, even when the mesh lays them over the abdomen;
    # a wing's size is its reach from that hinge (where it happens to touch the body says little)
    wz = waist_z + 0.3 * (neck_z - waist_z)
    sl = bv[np.abs(bv[:, 2] - wz) <= 2 * vs]
    if not len(sl): sl = bv
    top = float(np.percentile(sl[:, 1], 95)); half = 0.5 * float(np.percentile(sl[:, 0], 95) - np.percentile(sl[:, 0], 5))
    def hinge_for(e):
        sgn = 1 if e['points'][-1][0] > cx_at(wz) else -1
        return [cx_at(wz) + sgn * 0.35 * half, top - 0.5 * r * vs, wz]
    wings = [e for e in wings if e['points'][-1][2] < wz]       # a wing reaches back from its hinge
    for e in wings: e['reach'] = float(np.linalg.norm(np.array(e['points'][-1]) - hinge_for(e)))
    wings = keep_long(wings, 0.75, 'reach'); ants = keep_long(ants, 0.6)
    for grp, nm in ((wings, 'wing'), (ants, 'antenna')):
        if len(grp) == 2:
            a_, b_ = sorted(grp, key=lambda e: e['points'][0][0] + e['points'][-1][0])
            a_['side'], b_['side'] = 'R', 'L'
        else:
            for e in grp: e['side'] = side_of(e)
        for side in 'LR':
            sl = [e for e in grp if e['side'] == side]
            for i, e in enumerate(sl): e['name'] = '%s_%s' % (nm, side) if i == 0 else '%s_%s%d' % (nm, side, i + 1)
    for e in wings:
        hinge = hinge_for(e)
        if np.linalg.norm(np.array(e['points'][0]) - hinge) > 2 * vs: e['points'] = [hinge] + e['points']
        e['hinge'] = hinge
    lm = {'height': float(H), 'size': size, 'voxel': vs, 'body_open_radius_vox': r, 'spine': spine.tolist(),
          'spine_area': area.tolist(), 'neck_z': neck_z, 'waist_z': waist_z,
          'body_z': [float(zmin), float(zmax)],
          'legs': sorted(legs, key=lambda e: e['name']), 'wings': wings, 'antennae': ants,
          'other': [{'tip': e['points'][-1], 'length': e['length']} for e in other]}
    lm['_grid'] = G; lm['_body'] = body; lm['_labels'] = labels
    log('analyse: %d legs, %d wings, %d antennae, %d other limb ends ignored (body opening %d voxels of %.3f)' % (
        len(legs), len(wings), len(ants), len(other), r, vs))
    return lm


def public(lm):
    """Landmarks without the private vertex masks (for JSON)."""
    out = {}
    for k, v in lm.items():
        if k.startswith('_'): continue
        if isinstance(v, list) and v and isinstance(v[0], dict):
            out[k] = [{a: b for a, b in e.items() if not a.startswith('_')} for e in v]
        else: out[k] = v
    return out
