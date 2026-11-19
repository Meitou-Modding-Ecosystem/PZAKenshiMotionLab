# PZA's Kenshi Motion Lab - baked clips (.glb) -> a Kenshi animation file (.skeleton). Author: PZA. GPLv3.
#
#   python -m motionlab.kenshi_export <clips.glb> --skeleton <male_skeleton.skeleton> -o <out.skeleton>
#                                     [--clips a,b,c] [--prefix pza_] [--fps 30] [--travel keep|remove] [--contract out.json]
#
# Kenshi links extra animation files to a body's skeleton (ANIMATION_FILE records -> addLinkedSkeletonAnimationSource)
# and Ogre applies their tracks BY BONE HANDLE. So the file written here starts with the target skeleton's own bone and
# parent chunks copied byte for byte (same handles, same bind pose), followed by the new animations only.
#
# Ogre's keyframes are relative to the bind pose: drawn orientation = bind q * key q, drawn position = bind t + key t
# (parent space, Kenshi units = decimetres). The .glb holds parent-space local transforms in metres, so
#   key q = conj(bind q) * local q,    key t = (local t - bind t) * units.
#
# --travel keep (default): the root bone's horizontal travel stays in the Bip01 track. Both games treat a RELOCATING
# record's Bip01 track the same way: only its height is drawn and the ground part moves the character (Meitou:
# AnimationMask.Relocates / AnimationSystem.Relocate, along the walkable surface; Kenshi: ANIMATION `relocates`). The
# contract marks such clips `relocates` so the game data sets that flag. --travel remove takes the travel out as a
# straight line first -> last frame (sway and bob stay), for a record that must play in place.
import argparse, json, os, struct, sys
import numpy as np
from . import mathx as mx
from . import gltf
from .ogre import skeleton_read

UNITS = 10.0                       # Kenshi units per metre (decimetres)
HEADER, BLENDMODE, BONE, BONE_PARENT, ANIMATION, ANIM_BASEINFO, TRACK, KEYFRAME, LINK = \
    0x1000, 0x1010, 0x2000, 0x3000, 0x4000, 0x4010, 0x4100, 0x4110, 0x5000
RELOCATE_MIN_M = 0.05             # a clip whose root travels further than this is a relocating one
MAX_CLIP_S = 60.0                  # caps: a runaway length or key count is refused, never written
MAX_KEYS = 20000


class ExportError(ValueError):
    pass


# ------------------------------------------------------------------ the target skeleton
def skeleton_prefix(data):
    """Bytes of `data` (a .skeleton) up to its first animation / link chunk: header, blend mode, bones, parents."""
    p = 0
    if len(data) < 2 or struct.unpack_from('<H', data, 0)[0] != HEADER: raise ExportError('not an Ogre .skeleton')
    e = data.find(b'\n', 2)
    if e < 0: raise ExportError('no version string')
    p = e + 1
    known = {BLENDMODE, BONE, BONE_PARENT, ANIMATION, ANIM_BASEINFO, TRACK, KEYFRAME, LINK}
    def next_is_chunk(q): return q + 2 > len(data) or struct.unpack_from('<H', data, q)[0] in known
    while p + 6 <= len(data):
        cid = struct.unpack_from('<H', data, p)[0]
        if cid in (ANIMATION, LINK): return data[:p]
        q = p + 6
        if cid == BLENDMODE: q += 2
        elif cid == BONE:
            e = data.find(b'\n', q)
            if e < 0: raise ExportError('bone name runs off the end')
            q = e + 1 + 2 + 28
            if not next_is_chunk(q): q += 12
        elif cid == BONE_PARENT: q += 4
        else: raise ExportError('unexpected chunk 0x%X before the animations' % cid)
        if q > len(data): raise ExportError('chunk runs off the end')
        p = q
    return data[:p]


def bind_pose(skel_path):
    """{bone name: (handle, bind t in metres, bind q xyzw, parent handle or None)} of a Kenshi .skeleton."""
    v, bones, parents, anims = skeleton_read.read(skel_path)
    out = {}
    for h, (name, pos, q, sc) in bones.items():
        out[name] = (h, np.array(pos, float) / UNITS, mx.norm(np.array(q, float)), parents.get(h))
    return out


# ------------------------------------------------------------------ the clips
def _rest_locals(js):
    """Local rest t / q of every named glTF node."""
    out = {}
    for nd in js.get('nodes', []):
        nm = nd.get('name')
        if nm is None: continue
        out[nm] = (np.array(nd.get('translation', (0, 0, 0)), float), mx.norm(np.array(nd.get('rotation', (0, 0, 0, 1)), float)))
    return out


def _sample_t(times, vals, t):
    return np.array([np.interp(t, times, vals[:, c]) for c in range(vals.shape[1])])


def _sample_q(times, vals, t):
    if len(times) == 1 or t <= times[0]: return mx.norm(vals[0])
    if t >= times[-1]: return mx.norm(vals[-1])
    j = int(np.searchsorted(times, t) - 1); j = max(0, min(j, len(times) - 2))
    u = (t - times[j]) / max(times[j + 1] - times[j], 1e-12)
    return mx.q_slerp(mx.norm(vals[j]), mx.norm(vals[j + 1]), float(np.clip(u, 0, 1)))


def clip_locals(glb_path, fps=30.0):
    """{clip: (times, {bone name: (t[n,3] metres, q[n,4] xyzw)})}, every bone of the .glb sampled on one even grid."""
    js, _ = gltf.read_glb_raw(glb_path)
    rest = _rest_locals(js)
    out = {}
    for name, chans in gltf.read_clips(glb_path).items():
        length = max((float(c['times'][-1]) for c in chans.values() if len(c.get('times', ()))), default=0.0)
        if not np.isfinite(length) or length <= 0 or length > MAX_CLIP_S: raise ExportError('%s: bad clip length %r' % (name, length))
        n = int(round(length * fps)) + 1
        if n > MAX_KEYS: raise ExportError('%s: too many keys' % name)
        ts = np.linspace(0.0, length, n)
        bones = {}
        for bn, c in chans.items():
            if bn not in rest: continue
            rt, rq = rest[bn]
            T = np.array([_sample_t(c['t_translation'], c['translation'], t) for t in ts]) if 'translation' in c else np.tile(rt, (n, 1))
            Q = np.array([_sample_q(c['t_rotation'], c['rotation'], t) for t in ts]) if 'rotation' in c else np.tile(rq, (n, 1))
            if 'scale' in c and np.abs(c['scale'] - 1.0).max() > 1e-3:
                raise ExportError('%s: bone %s scales; Kenshi scale keys are unverified, refused' % (name, bn))
            if not (np.isfinite(T).all() and np.isfinite(Q).all()): raise ExportError('%s: non-finite key on %s' % (name, bn))
            bones[bn] = (T, Q)
        out[name] = (ts, bones)
    return out


def remove_travel(bones, roots):
    """Take each root bone's horizontal (x, z) travel out as a straight line first -> last frame. Returns the travel
    (metres, model space) that was removed."""
    travel = np.zeros(3)
    for r in roots:
        if r not in bones: continue
        T, Q = bones[r]
        d = T[-1] - T[0]; d[1] = 0.0
        u = np.linspace(0.0, 1.0, len(T))[:, None]
        bones[r] = (T - u * d[None, :], Q)
        travel = d
    return travel


# ------------------------------------------------------------------ writing
def _chunk(cid, body):
    return struct.pack('<HI', cid, 6 + len(body)) + body


def _str(s):
    b = s.encode('latin-1')
    if b'\n' in b or b'\0' in b: raise ExportError('bad name %r' % s)
    return b + b'\n'


def write_skeleton(skel_data, anims, path):
    """skel_data: bytes of the target .skeleton. anims: [(name, length, {handle: [(time, q xyzw, t units)]})]."""
    out = bytearray(skeleton_prefix(skel_data))
    for name, length, tracks in anims:
        body = bytearray(_str(name) + struct.pack('<f', length))
        for h in sorted(tracks):
            tb = bytearray(struct.pack('<H', h))
            for (t, q, tr) in tracks[h]:
                kb = struct.pack('<f4f3f', t, *q, *tr)          # no scale: 38-byte keyframes, Ogre reads unit scale
                tb += _chunk(KEYFRAME, kb)
            body += _chunk(TRACK, bytes(tb))
        out += _chunk(ANIMATION, bytes(body))
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f: f.write(out)
    os.replace(tmp, path)
    return len(out)


def build(glb_path, skel_path, clips=None, prefix='pza_', fps=30.0, travel='keep'):
    """-> (animations for write_skeleton, {clip: {'travel_m': [x, y, z], 'kenshi_name': ...}})."""
    bind = bind_pose(skel_path)
    roots = [n for n, b in bind.items() if b[3] is None]
    got = clip_locals(glb_path, fps)
    names = clips or list(got)
    anims, info = [], {}
    for cn in names:
        if cn not in got: raise ExportError('no clip %r in %s (has: %s)' % (cn, glb_path, ', '.join(got)))
        ts, bones = got[cn]
        missing = [b for b in bones if b not in bind]
        if missing: raise ExportError('%s: bones not in the target skeleton: %s' % (cn, ', '.join(missing[:5])))
        if travel == 'remove': mv = remove_travel(bones, roots)
        else:
            mv = np.zeros(3)
            for r in roots:
                if r in bones: mv = bones[r][0][-1] - bones[r][0][0]; mv[1] = 0.0
        tracks = {}
        for bn, (T, Q) in bones.items():
            h, bt, bq, _ = bind[bn]; ib = mx.q_conj(bq); keys = []
            for k, t in enumerate(ts):
                kq = mx.norm(mx.q_mul(ib, Q[k]))
                if kq[3] < 0: kq = -kq
                keys.append((float(t), tuple(float(x) for x in kq), tuple(float(x) for x in (T[k] - bt) * UNITS)))
            tracks[h] = keys
        kn = prefix + cn
        anims.append((kn, float(ts[-1]), tracks))
        info[cn] = {'kenshi_name': kn, 'length_s': round(float(ts[-1]), 4), 'keys_per_s': fps,
                    'travel_m': [round(float(x), 4) for x in mv], 'travel_removed': travel == 'remove',
                    'relocates': travel == 'keep' and float(np.hypot(mv[0], mv[2])) > RELOCATE_MIN_M}
    return anims, info


# ------------------------------------------------------------------ reading it back (Ogre's rules) and measuring
def sampler(skel_path, anim_name):
    """(names, parents, sample(t) -> world 4x4 per bone in metres, length): Ogre's playback of one clip of a file
    that holds the bones (the exported file), bind + key per bone, linear / slerp between keys."""
    v, bones, parents, anims = skeleton_read.read(skel_path)
    hs = sorted(bones); ix = {h: i for i, h in enumerate(hs)}
    names = [bones[h][0] for h in hs]; par = [ix[parents[h]] if h in parents else -1 for h in hs]
    bt = [np.array(bones[h][1], float) / UNITS for h in hs]; bq = [mx.norm(np.array(bones[h][2], float)) for h in hs]
    an = [a for a in anims if a[0] == anim_name]
    if not an: raise ExportError('no animation %r' % anim_name)
    _, length, tracks = an[0]
    order = []; seen = set()
    def visit(i):
        if i in seen: return
        if par[i] >= 0: visit(par[i])
        seen.add(i); order.append(i)
    for i in range(len(hs)): visit(i)
    def sample(t):
        W = [None] * len(hs)
        for i in order:
            keys = tracks.get(hs[i]); kq, kt = mx.q_identity(), np.zeros(3)
            if keys:
                tt = [k[0] for k in keys]; j = int(np.clip(np.searchsorted(tt, t) - 1, 0, len(keys) - 1)); j2 = min(j + 1, len(keys) - 1)
                u = 0.0 if j2 == j else float(np.clip((t - tt[j]) / max(tt[j2] - tt[j], 1e-9), 0, 1))
                kq = mx.q_slerp(mx.norm(np.array(keys[j][1], float)), mx.norm(np.array(keys[j2][1], float)), u)
                kt = (np.array(keys[j][2]) * (1 - u) + np.array(keys[j2][2]) * u) / UNITS
            L = mx.trs(bt[i] + kt, mx.q_mul(bq[i], kq), np.ones(3))
            W[i] = W[par[i]] @ L if par[i] >= 0 else L
        return np.array(W)
    return names, par, sample, float(length)


FEET = (('Bip01 L Foot', 'Bip01 L Toe0'), ('Bip01 R Foot', 'Bip01 R Toe0'))


def measure(skel_path, anim_name, levels=(0.0,), fps=60.0, tol=0.03):
    """Contract numbers read from the FILE (what Ogre will play): take-off / landing times, air time, pelvis rise and
    foot clearance. A foot is on the ground when its foot or toe bone is within `tol` m of that bone's lowest height in
    the clip plus one of `levels` (the floor heights the clip stands on: 0, and the ledge for a drop)."""
    names, par, smp, L = sampler(skel_path, anim_name)
    ix = {n: i for i, n in enumerate(names)}
    feet = [(ix[a], ix[b]) for a, b in FEET if a in ix and b in ix]
    pel = ix.get('Bip01 Pelvis', 0)
    n = int(L * fps) + 1; ts = np.linspace(0, L, n)
    Y = np.array([[smp(t)[i][1, 3] for pair in feet for i in pair] + [smp(t)[pel][1, 3]] for t in ts])
    bones_y = Y[:, :-1]; py = Y[:, -1]
    low = bones_y.min(axis=0)
    def touching(k):
        for f in range(len(feet)):
            for b in (2 * f, 2 * f + 1):
                if any(abs(bones_y[k, b] - (low[b] + lv)) < tol for lv in levels): return True
        return False
    on = np.array([touching(k) for k in range(n)])
    air = ~on
    # the longest airborne run is the flight (a swing foot never makes both feet leave the floor in a gait)
    best, cur, start = (0, 0), 0, 0
    for k in range(n):
        if air[k]:
            if cur == 0: start = k
            cur += 1
            if cur > best[1] - best[0]: best = (start, k + 1)
        else: cur = 0
    res = {'length_s': round(L, 4)}
    if best[1] > best[0]:
        k0, k1 = best
        res['takeoff_s'] = round(float(ts[k0]), 3) if k0 > 0 else None          # None: the clip starts in the air
        res['landing_s'] = round(float(ts[k1]), 3) if k1 < n else None
        res['air_s'] = round(float(ts[min(k1, n - 1)] - ts[k0]), 3)
        res['rise_m'] = round(float(py[k0:k1].max() - py[0]), 3)
        gap = (bones_y[k0:k1] - low[None, :]).min(axis=1)        # the lowest foot / toe bone above its floor, per frame
        res['clearance_m'] = round(float(gap.max()), 3)           # at the top of the flight
    else:
        res.update(takeoff_s=None, landing_s=None, air_s=0.0, rise_m=0.0, clearance_m=0.0)
    res['pelvis_start_m'] = round(float(py[0]), 3); res['pelvis_end_m'] = round(float(py[-1]), 3)
    return res


CONTRACT_FORMAT = 'pza.jump.contract'
# what each clip is for (the game side picks by kind); clips not listed are 'other'
KINDS = {'jump_up': 'standing', 'jump_forward': 'standing', 'jump_walk': 'stride', 'jump_run': 'stride',
         'jump_down': 'drop', 'land_soft': 'landing', 'land_hard': 'landing'}
FROM_GAIT = {'jump_walk': 'walk', 'jump_run': 'run'}


def contract(skel_paths, info, levels_of=None, landing=None):
    """The shared jump contract both games read: per clip its Kenshi name, kind, timings and heights MEASURED from the
    written file (`measure`), and the travel / relocates flags from `build`. skel_paths: {'male': path, 'female': path}."""
    levels_of = levels_of or {}
    first = next(iter(skel_paths.values()))
    clips = {}
    for cn, inf in info.items():
        m = measure(first, inf['kenshi_name'], levels=levels_of.get(cn, (0.0,)))
        per = {b: measure(p, inf['kenshi_name'], levels=levels_of.get(cn, (0.0,))) for b, p in skel_paths.items()}
        spread = max(abs((per[b].get(k) or 0) - (m.get(k) or 0)) for b in per for k in ('takeoff_s', 'landing_s', 'air_s'))
        clips[cn] = {'kenshi_name': inf['kenshi_name'], 'kind': KINDS.get(cn, 'other'), 'from_gait': FROM_GAIT.get(cn),
                     'loop': False, 'length_s': inf['length_s'],
                     'takeoff_s': m['takeoff_s'], 'landing_s': m['landing_s'], 'air_s': m['air_s'],
                     'rise_m': m['rise_m'], 'clearance_m': m['clearance_m'],
                     'start_floor_m': max(levels_of.get(cn, (0.0,))) if KINDS.get(cn) == 'drop' else 0.0,
                     'starts_in_air': m['takeoff_s'] is None,
                     'travel_m': inf['travel_m'], 'relocates': inf['relocates'],
                     'bodies_differ_s': round(spread, 3)}
    return {'format': CONTRACT_FORMAT, 'version': 1, 'units': 'metres, seconds; travel in model space (+Z = front)',
            'skeletons': {b: os.path.basename(p) for b, p in skel_paths.items()},
            'landing': landing or {}, 'clips': clips}


def main(argv=None):
    ap = argparse.ArgumentParser(prog='python -m motionlab.kenshi_export', description=__doc__.split('\n')[0] if __doc__ else None)
    ap.add_argument('glb'); ap.add_argument('--skeleton', required=True); ap.add_argument('-o', '--out', required=True)
    ap.add_argument('--clips', default=None); ap.add_argument('--prefix', default='pza_'); ap.add_argument('--fps', type=float, default=30.0)
    ap.add_argument('--travel', choices=('remove', 'keep'), default='keep'); ap.add_argument('--contract', default=None)
    a = ap.parse_args(argv)
    clips = [c.strip() for c in a.clips.split(',')] if a.clips else None
    anims, info = build(a.glb, a.skeleton, clips, a.prefix, a.fps, a.travel)
    size = write_skeleton(open(a.skeleton, 'rb').read(), anims, a.out)
    print('wrote %s (%.1f KB, %d animations)' % (a.out, size / 1024, len(anims)))
    if a.contract:
        with open(a.contract, 'w', encoding='utf-8') as f: json.dump(contract({'body': a.out}, info), f, indent=1)
    return 0


if __name__ == '__main__':
    sys.exit(main())
