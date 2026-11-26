# PZA's Kenshi Motion Lab - build a clean named skeleton from landmarks, and bind the mesh to it. Author: PZA. GPLv3.
#
# Bone names (the driver and the MeitouMaestro descriptor find bones by these):
#   root                      on the ground under the thorax (moves the whole creature)
#   body                      thorax (legs and wings hang from it)
#   head                      head (antennae hang from it)
#   abdomen_01 .. _NN         abdomen / tail chain, front to back
#   leg_<L|R><n>_01 .. _NN    legs, n = 1 front .. rear; L = the creature's left (+X when it faces +Z)
#   wing_<L|R>_01 .. _NN      wings, hinge first
#   antenna_<L|R>_01 .. _NN   antennae, base first
# Every bone's local +Y points down the bone (to its tail), like Blender and Ogre exports.
import numpy as np
from . import mathx as mx
from .model import Skeleton
from .autorig import fit_polyline
from . import voxel

KENSHI_MAX_BONES_PER_SUBMESH = 60
KENSHI_WEIGHTS_PER_VERTEX = 3
KENSHI_UNITS_PER_METRE = 10.0


def _seg_dist(P, a, b):
    ab = b - a; L = float(ab @ ab)
    if L < 1e-12: return np.linalg.norm(P - a, axis=1)
    t = np.clip((P - a) @ ab / L, 0, 1)
    return np.linalg.norm(P - (a + t[:, None] * ab), axis=1)


def build_skeleton(lm, leg_bones=4, wing_bones=2, antenna_bones=2, abdomen_bones=3, arm_bones=2, log=print):
    sp = np.array(lm['spine'])
    vertical = lm.get('spine_axis') == 'y'
    sk = Skeleton(); tails = {}; heads = {}
    def add(name, parent, head, tail):
        d = np.asarray(tail) - np.asarray(head)
        q = mx.q_between([0, 1, 0], d) if np.linalg.norm(d) > 1e-9 else mx.q_identity()
        i = sk.add(name, parent, np.asarray(head, float), q)
        tails[name] = [float(x) for x in tail]; heads[name] = [float(x) for x in head]
        return i
    if vertical:
        ys = sp[:, 1]
        def Sy(y): return np.array([np.interp(y, ys, sp[:, 0]), y, np.interp(y, ys, sp[:, 2])])
        ymin, ymax = lm['body_z']; neck, waist = lm['neck_z'], lm['waist_z']
        root = add('root', -1, [Sy(waist)[0], 0.0, Sy(waist)[2]], [Sy(waist)[0], 0.0, Sy(waist)[2] + 0.1 * (ymax - ymin)])
        body = add('body', root, Sy(waist), Sy(neck))
        head = add('head', body, Sy(neck), Sy(ymax))
        abd = []
        spine_bones = [body, head]
        seg = {body: (Sy(waist), Sy(neck))}
    else:
        zs = sp[:, 2]
        def S(z): return np.array([np.interp(z, zs, sp[:, 0]), np.interp(z, zs, sp[:, 1]), z])
        zmin, zmax = lm['body_z']; neck, waist = lm['neck_z'], lm['waist_z']
        zc = 0.5 * (neck + waist)
        root = add('root', -1, [S(zc)[0], 0.0, zc], [S(zc)[0], 0.0, zc + 0.1 * (zmax - zmin)])
        body = add('body', root, S(zc), S(neck))
        head = add('head', body, S(neck), S(zmax))
        zz = np.linspace(waist, zmin, abdomen_bones + 1)
        abd = []; par = body
        for k in range(abdomen_bones):
            par = add('abdomen_%02d' % (k + 1), par, S(zz[k]), S(zz[k + 1])); abd.append(par)
        spine_bones = [body, head] + abd
        seg = {body: (S(waist), S(neck))}       # the thorax bone covers waist..neck for this choice (its pivot is mid-thorax)
    def nearest_spine(p):
        best, bi = 1e18, body
        for b in spine_bones:
            nm = sk.names[b]
            a_, b_ = seg.get(b, (np.array(heads[nm]), np.array(tails[nm])))
            dd = float(_seg_dist(np.asarray([p]), a_, b_)[0])
            if dd < best: best, bi = dd, b
        return bi
    rigmap = {'plan': lm.get('plan', 'insect'), 'units': 'metres', 'forward': '+Z', 'up': '+Y',
              'root': 'root', 'body': 'body', 'head': 'head', 'spine': [sk.names[b] for b in abd],
              'legs': [], 'arms': [], 'wings': [], 'antennae': []}
    H = lm['height']
    for kind, nb, key, parent_rule in (('leg', leg_bones, 'legs', 'nearest'), ('arm', arm_bones, 'arms', 'body'),
                                       ('wing', wing_bones, 'wings', 'body'), ('antenna', antenna_bones, 'antennae', 'head')):
        for e in lm.get(key, []):
            pts = np.array(e['points'])
            j = fit_polyline(pts, nb)
            par = nearest_spine(j[0]) if parent_rule == 'nearest' else (body if parent_rule == 'body' else head)
            names = []
            for k in range(nb):
                nm = '%s_%02d' % (e['name'], k + 1)
                par = add(nm, par, j[k], j[k + 1]); names.append(nm)
            ent = {'name': e['name'], 'side': e['side'], 'bones': names, 'tip': [float(x) for x in j[-1]],
                   'parent': sk.names[sk.parents[sk.index(names[0])]]}
            if kind == 'leg': ent['lifted'] = bool(j[-1][1] > 0.15 * H)
            rigmap[key].append(ent)
    rigmap['tails'] = tails
    log('skeleton: %d bones (%d legs x %d, %d arms x %d, %d wings x %d, %d antennae x %d, abdomen %d)' % (
        len(sk), len(lm['legs']), leg_bones, len(lm.get('arms', [])), arm_bones, len(lm['wings']), wing_bones,
        len(lm['antennae']), antenna_bones, len(abd)))
    return sk, rigmap


def bind(model, sk, rigmap, lm, surface=None, smooth_iters=2, power=4.0, max_influences=KENSHI_WEIGHTS_PER_VERTEX, log=print):
    """Weights by distance to bone segments, limited to the bones of the vertex's own voxel part (a leg's vertices
    only see that leg's bones and the body bone it hangs from), smoothed over the surface, pruned to 3."""
    P = model.verts; n = len(P); B = len(sk)
    W = sk.world_rest()
    heads = np.array([W[i][:3, 3] for i in range(B)])
    tails = np.array([rigmap['tails'][nm] for nm in sk.names])
    G = lm['_grid']; labels = lm['_labels']
    # part label per vertex: nearest labelled voxel within 3 voxels
    ijk = G.index(P); lab = np.full(n, -1, np.int64)
    offs = sorted([(a, b, c) for a in range(-3, 4) for b in range(-3, 4) for c in range(-3, 4)], key=lambda o: o[0]**2 + o[1]**2 + o[2]**2)
    sh = np.array(labels.shape)
    for a, b, c in offs:
        need = lab == -1
        if not need.any(): break
        q = np.clip(ijk[need] + [a, b, c], 0, sh - 1)
        v = labels[q[:, 0], q[:, 1], q[:, 2]]
        idx = np.nonzero(need)[0]; lab[idx[v != -1]] = v[v != -1]
    names = sk.names
    spine = {names.index(x) for x in ['body', 'head'] + rigmap['spine']}
    body_allowed = set(spine)
    part_bones = {}
    for key in ('legs', 'arms', 'wings', 'antennae'):
        for ent, e in zip(rigmap.get(key, []), lm.get(key, [])):
            bs = [names.index(x) for x in ent['bones']]
            body_allowed.add(bs[0])
            part_bones.setdefault(e['_part'], set()).update(bs + [names.index(ent['parent'])])
    allow = np.zeros((n, B), bool)
    allow[:, list(body_allowed)] = True                       # body vertices, unknown vertices, spikes
    for part, bs in part_bones.items():
        sel = lab == part
        allow[sel] = False; allow[np.ix_(sel, list(bs))] = True
    allow[:, names.index('root')] = False
    D = np.stack([_seg_dist(P, heads[i], tails[i]) for i in range(B)], 1)
    eps = 0.01 * lm['size']
    Wt = np.where(allow, 1.0 / (D + eps) ** power, 0.0)
    Wt /= np.maximum(Wt.sum(1, keepdims=True), 1e-30)
    if smooth_iters and surface is not None:
        inv = surface.inv
        Ww = np.zeros((surface.n, B)); cnt = np.zeros(surface.n)
        np.add.at(Ww, inv, Wt); np.add.at(cnt, inv, 1); Ww /= cnt[:, None]
        a = np.repeat(np.arange(surface.n), np.diff(surface.ptr)); b = surface.nb
        Aw = np.zeros((surface.n, B))
        for _ in range(smooth_iters):
            Aw[:] = 0; np.add.at(Aw, a, Ww[b])
            deg = np.diff(surface.ptr)[:, None]
            Ww = 0.5 * Ww + 0.5 * Aw / np.maximum(deg, 1)
        Wt = np.where(allow, Ww[inv], 0.0)
    order = np.argsort(-Wt, 1)[:, :max_influences]
    J = order; Wk = np.take_along_axis(Wt, order, 1)
    dropped = 1.0 - Wk.sum(1) / np.maximum(Wt.sum(1), 1e-30)
    Wk /= np.maximum(Wk.sum(1, keepdims=True), 1e-30)
    stats = {'vertices': n, 'max_weight_dropped_by_prune': float(dropped.max()), 'unweighted': int((Wk.sum(1) <= 0).sum()),
             'vertices_per_part': {}}
    log('bind: %d vertices, 3 weights each, most weight dropped by the prune to 3: %.3f' % (n, stats['max_weight_dropped_by_prune']))
    return J.astype(np.int64), Wk, stats


def report(model, sk, rigmap, J, Wk, stats, lm, source_info=None):
    used = sorted(set(J[Wk > 1e-6].tolist()))
    H = lm['height']
    lines = []
    lines.append("# PZA's Kenshi Motion Lab: rig report for '%s'" % model.name)
    lines.append('')
    if source_info: lines += ['Source: %s' % source_info, '']
    fr = lm.get('frame', {})
    lines.append('| Check | Value | Kenshi | OK |')
    lines.append('|---|---|---|---|')
    nb = len(sk)
    lines.append('| Bones in the skeleton | %d | %d per submesh (skin shader) | %s |' % (nb, KENSHI_MAX_BONES_PER_SUBMESH, 'yes' if nb <= KENSHI_MAX_BONES_PER_SUBMESH else 'NO: split the mesh into submeshes'))
    lines.append('| Bones that carry weights | %d | %d per submesh | %s |' % (len(used), KENSHI_MAX_BONES_PER_SUBMESH, 'yes' if len(used) <= KENSHI_MAX_BONES_PER_SUBMESH else 'NO'))
    lines.append('| Weights per vertex | %d | %d (the GPU reads 3; a 4th is ignored) | yes |' % (J.shape[1], KENSHI_WEIGHTS_PER_VERTEX))
    lines.append('| Weight dropped by pruning to 3 (worst vertex) | %.1f %% | - | %s |' % (100 * stats['max_weight_dropped_by_prune'], 'yes' if stats['max_weight_dropped_by_prune'] < 0.25 else 'check'))
    lines.append('| Vertices without weights | %d | 0 | %s |' % (stats['unweighted'], 'yes' if stats['unweighted'] == 0 else 'NO'))
    lines.append('| Height / length | %.2f m / %.2f m | x%d in Kenshi units: %.1f / %.1f | - |' % (
        H, lm['body_z'][1] - lm['body_z'][0], KENSHI_UNITS_PER_METRE, H * 10, (lm['body_z'][1] - lm['body_z'][0]) * 10))
    lines.append('| Mesh turned to face +Z | %.1f deg about Y, centred, feet on y=0 | - | - |' % fr.get('turn_about_y_deg', 0))
    lines.append('| Vertices / triangles | %d / %d | - | - |' % (len(model.verts), len(model.tris)))
    lines.append('')
    lines.append('## Limbs found')
    lines.append('| Limb | Bones | Hangs from | Tip (m) | Note |')
    lines.append('|---|---|---|---|---|')
    for key in ('legs', 'arms', 'wings', 'antennae'):
        for e in rigmap.get(key, []):
            note = 'lifted: its tip rests off the ground, it swings but does not plant' if e.get('lifted') else ''
            lines.append('| %s | %s | %s | (%.2f, %.2f, %.2f) | %s |' % (e['name'], ', '.join(e['bones']), e['parent'], *e['tip'], note))
    lines.append('')
    lines.append('Spine: body (thorax) %.2f..%.2f m along +Z, head from %.2f, abdomen behind %.2f (%s).' % (
        lm['waist_z'], lm['neck_z'], lm['neck_z'], lm['waist_z'], ', '.join(rigmap['spine'])))
    if lm.get('other'): lines.append('Ignored limb ends (short spikes): %d.' % len(lm['other']))
    return '\n'.join(lines) + '\n'
