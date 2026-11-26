# PZA's Kenshi Motion Lab - direct .glb -> Kenshi/Meitou skinned asset exporter (.mesh + .skeleton + .dds).
# Author: PZA. GPLv3.
#
#   python -m motionlab export <rigged.glb> -o <out_dir> [--name stem] [--clips a,b] [--fps 30] [--max-texture 2048]
#
# Writes a ready-to-load Ogre binary [MeshSerializer_v1.8] skinned mesh (position, normal, UV, Kenshi-handed tangent,
# up to 3 bone weights per vertex, skeleton link), a fresh Ogre [Serializer_v1.80] skeleton with all baked clips, and
# Kenshi-packed DDS textures (_d.dds BC3 with gloss in alpha, _n.dds BC1/BC3 with Y-flipped green, _m.dds BC1 metalness).
# Pure Python + numpy (Pillow optional, used only when decoding embedded PNG/JPEG textures from the .glb).
import argparse, io, json, os, struct, sys
import numpy as np
from . import gltf, kenshi_export as ke, mathx as mx
from .rigbuild import KENSHI_MAX_BONES_PER_SUBMESH, KENSHI_UNITS_PER_METRE, KENSHI_WEIGHTS_PER_VERTEX

# Ogre MeshSerializer_v1.8 chunk ids
M_HEADER = 0x1000
M_MESH = 0x3000
M_SUBMESH = 0x4000
M_SUBMESH_OPERATION = 0x4010
M_SUBMESH_BONE_ASSIGNMENT = 0x4100
M_GEOMETRY = 0x5000
M_GEOMETRY_VERTEX_DECLARATION = 0x5100
M_GEOMETRY_VERTEX_ELEMENT = 0x5110
M_GEOMETRY_VERTEX_BUFFER = 0x5200
M_GEOMETRY_VERTEX_BUFFER_DATA = 0x5210
M_MESH_SKELETON_LINK = 0x6000
M_MESH_BOUNDS = 0x9000

MESH_VERSION = '[MeshSerializer_v1.8]'
SKEL_VERSION = '[Serializer_v1.80]'
DEFAULT_MATERIAL = 'Character'
VERTEX_SIZE = 48  # pos(12) + normal(12) + uv(8) + tangent(16)


class AssetExportError(ValueError):
    pass


def _chunk(cid, body):
    return struct.pack('<HI', cid, 6 + len(body)) + body


def _ostr(s):
    b = s.encode('latin-1')
    if b'\n' in b or b'\0' in b:
        raise AssetExportError('invalid Ogre name %r' % s)
    return b + b'\n'


# ------------------------------------------------------------------ tangents (Kenshi handedness)

def compute_tangents(verts, normals, uvs, tris):
    """Per-vertex tangent (N, 4) with handedness w such that cross(T, N) * w aligns with +V (Kenshi / Ogre objects & skin)."""
    n = len(verts)
    t_acc = np.zeros((n, 3), np.float64)
    b_acc = np.zeros((n, 3), np.float64)
    if len(tris):
        i0, i1, i2 = tris[:, 0], tris[:, 1], tris[:, 2]
        e1 = verts[i1] - verts[i0]
        e2 = verts[i2] - verts[i0]
        du1 = uvs[i1, 0] - uvs[i0, 0]
        dv1 = uvs[i1, 1] - uvs[i0, 1]
        du2 = uvs[i2, 0] - uvs[i0, 0]
        dv2 = uvs[i2, 1] - uvs[i0, 1]
        det = du1 * dv2 - du2 * dv1
        good = np.abs(det) > 1e-12
        inv = np.where(good, 1.0 / np.where(good, det, 1.0), 0.0)
        tan = (e1 * dv2[:, None] - e2 * dv1[:, None]) * inv[:, None]
        bit = (e2 * du1[:, None] - e1 * du2[:, None]) * inv[:, None]
        for idx in (i0, i1, i2):
            np.add.at(t_acc, idx, tan)
            np.add.at(b_acc, idx, bit)
    N = mx.norm(normals)
    # Gram-Schmidt orthogonalize T against N
    T = t_acc - N * np.sum(N * t_acc, axis=1, keepdims=True)
    tl = np.linalg.norm(T, axis=1, keepdims=True)
    # fallback basis where UV is degenerate
    fb = np.where(np.abs(N[:, 1:2]) < 0.9, np.array([[0.0, 1.0, 0.0]]), np.array([[1.0, 0.0, 0.0]]))
    fb_t = mx.norm(np.cross(fb, N))
    T = np.where(tl > 1e-9, T / np.maximum(tl, 1e-9), fb_t)
    # Kenshi binormal = cross(T, N) * w -> w = sign(dot(cross(T, N), B))
    c = np.cross(T, N)
    w = np.where(np.sum(c * b_acc, axis=1) < 0.0, -1.0, 1.0)
    return np.c_[T, w].astype(np.float32)


# ------------------------------------------------------------------ Ogre skinned .mesh writer

def _submesh_slice(model, part):
    """Extract compact vertex/index/skin arrays for one submesh part."""
    t0 = int(part['tri_start'])
    tc = int(part['tri_count'])
    tri = model.tris[t0:t0 + tc]
    if len(tri) == 0:
        return None
    used, inv = np.unique(tri.reshape(-1), return_inverse=True)
    local_tris = inv.reshape(-1, 3)
    verts = model.verts[used]
    normals = model.normals[used]
    uvs = model.uvs[used]
    joints = model.joints[used] if model.joints is not None else None
    weights = model.weights[used] if model.weights is not None else None
    return verts, normals, uvs, local_tris, joints, weights


def write_ogre_mesh(model, skel_name, path, material_name=None, units=KENSHI_UNITS_PER_METRE):
    """Write a [MeshSerializer_v1.8] binary .mesh (skinned when model.skeleton is set, else static)."""
    if model.verts is None or len(model.verts) == 0 or model.tris is None or len(model.tris) == 0:
        raise AssetExportError('model has no geometry')
    skinned = model.skeleton is not None and model.joints is not None and model.weights is not None
    parts = model.parts or [{'material': None, 'tri_start': 0, 'tri_count': len(model.tris)}]

    sub_chunks = bytearray()
    bone_counts = []
    for pi, part in enumerate(parts):
        sl = _submesh_slice(model, part)
        if sl is None:
            continue
        V, N, UV, T, J, W = sl
        nv = len(V)
        # Scale positions to Kenshi decimetres
        Vs = (V * units).astype(np.float32)
        Ns = mx.norm(N).astype(np.float32)
        UVs = UV.astype(np.float32)
        Tans = compute_tangents(V, N, UV, T)

        mat = material_name or part.get('ogre_material') or (
            ('%s_%d' % (DEFAULT_MATERIAL, pi)) if pi > 0 else DEFAULT_MATERIAL
        )
        sb = bytearray(_ostr(mat))
        sb += struct.pack('<B', 0)  # own vertices, not shared
        idx_flat = T.reshape(-1)
        wide = nv > 65535
        sb += struct.pack('<IB', len(idx_flat), 1 if wide else 0)
        sb += idx_flat.astype('<u4' if wide else '<u2').tobytes()

        # M_GEOMETRY
        decl = bytearray()
        for typ, sem, off in ((2, 1, 0), (2, 4, 12), (1, 7, 24), (3, 9, 32)):
            decl += _chunk(M_GEOMETRY_VERTEX_ELEMENT, struct.pack('<HHHHH', 0, typ, sem, off, 0))
        vbuf_arr = np.empty((nv, 12), dtype='<f4')
        vbuf_arr[:, 0:3] = Vs
        vbuf_arr[:, 3:6] = Ns
        vbuf_arr[:, 6:8] = UVs
        vbuf_arr[:, 8:12] = Tans
        vbuf = _chunk(M_GEOMETRY_VERTEX_BUFFER, struct.pack('<HH', 0, VERTEX_SIZE) + _chunk(M_GEOMETRY_VERTEX_BUFFER_DATA, vbuf_arr.tobytes()))
        geom = struct.pack('<I', nv) + _chunk(M_GEOMETRY_VERTEX_DECLARATION, bytes(decl)) + vbuf
        sb += _chunk(M_GEOMETRY, geom)
        sb += _chunk(M_SUBMESH_OPERATION, struct.pack('<H', 4))  # triangle list

        if skinned:
            # Prune to Kenshi's 3 strongest weights per vertex and normalize
            k_max = min(KENSHI_WEIGHTS_PER_VERTEX, J.shape[1])
            order = np.argsort(-W, axis=1)[:, :k_max]
            Jp = np.take_along_axis(J, order, axis=1)
            Wp = np.take_along_axis(W, order, axis=1)
            wsum = Wp.sum(axis=1, keepdims=True)
            Wp = np.where(wsum > 1e-9, Wp / np.maximum(wsum, 1e-9), 0.0)
            used_bones = set(int(b) for b in Jp[Wp > 1e-6].tolist())
            if len(used_bones) > KENSHI_MAX_BONES_PER_SUBMESH:
                raise AssetExportError('submesh %d uses %d bones (Kenshi limit is %d)' % (
                    pi, len(used_bones), KENSHI_MAX_BONES_PER_SUBMESH))
            bone_counts.append(len(used_bones))
            # Pack all non-zero bone assignments vectorized
            vi_grid = np.broadcast_to(np.arange(nv, dtype=np.uint32)[:, None], Jp.shape)
            mask = Wp > 1e-6
            vis = vi_grid[mask]
            bns = Jp[mask].astype(np.uint16)
            wts = Wp[mask].astype(np.float32)
            # Each M_SUBMESH_BONE_ASSIGNMENT chunk is 6-byte header + 10-byte payload (<HI + <IHf)
            dt = np.dtype([('cid', '<u2'), ('ln', '<u4'), ('vi', '<u4'), ('bn', '<u2'), ('wt', '<f4')])
            rec = np.empty(len(vis), dtype=dt)
            rec['cid'] = M_SUBMESH_BONE_ASSIGNMENT
            rec['ln'] = 16
            rec['vi'] = vis
            rec['bn'] = bns
            rec['wt'] = wts
            sb += rec.tobytes()

        sub_chunks += _chunk(M_SUBMESH, bytes(sb))

    mesh_body = bytearray(struct.pack('<B', 1 if skinned else 0))
    mesh_body += sub_chunks
    if skinned and skel_name:
        mesh_body += _chunk(M_MESH_SKELETON_LINK, _ostr(os.path.basename(skel_name)))
    lo = (model.verts.min(0) * units).astype(np.float32)
    hi = (model.verts.max(0) * units).astype(np.float32)
    radius = float(np.linalg.norm(hi - lo) * 0.5)
    mesh_body += _chunk(M_MESH_BOUNDS, struct.pack('<7f', float(lo[0]), float(lo[1]), float(lo[2]),
                                                   float(hi[0]), float(hi[1]), float(hi[2]), radius))

    out = bytearray(struct.pack('<H', M_HEADER) + _ostr(MESH_VERSION))
    out += _chunk(M_MESH, bytes(mesh_body))
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        f.write(out)
    os.replace(tmp, path)
    return {'bytes': len(out), 'submeshes': len(parts), 'skinned': skinned,
            'max_bones_per_submesh': max(bone_counts, default=0)}


# ------------------------------------------------------------------ fresh Ogre .skeleton writer

def skeleton_bytes(sk, units=KENSHI_UNITS_PER_METRE):
    """Serialize a MotionLab Skeleton to Ogre [Serializer_v1.80] bone + parent chunks (decimetres)."""
    out = bytearray(struct.pack('<H', ke.HEADER) + _ostr(SKEL_VERSION))
    out += _chunk(ke.BLENDMODE, struct.pack('<H', 0))
    for i, nm in enumerate(sk.names):
        t = np.asarray(sk.rest_t[i], float) * units
        q = mx.norm(np.asarray(sk.rest_q[i], float))
        body = _ostr(nm) + struct.pack('<H3f4f', i, float(t[0]), float(t[1]), float(t[2]),
                                       float(q[0]), float(q[1]), float(q[2]), float(q[3]))
        out += _chunk(ke.BONE, body)
    for i, p in enumerate(sk.parents):
        if p >= 0:
            out += _chunk(ke.BONE_PARENT, struct.pack('<HH', i, p))
    return bytes(out)


def _clips_to_ogre(glb_path, sk, clips=None, prefix='', fps=30.0, travel='keep', drop_scale=True):
    """Convert .glb clips against a fresh Skeleton into Ogre animation tuples for write_skeleton."""
    js, _ = gltf.read_glb_raw(glb_path)
    rest = {}
    for nd in js.get('nodes', []):
        nm = nd.get('name')
        if nm is not None:
            rest[nm] = (np.array(nd.get('translation', (0, 0, 0)), float),
                        mx.norm(np.array(nd.get('rotation', (0, 0, 0, 1)), float)))
    raw_clips = gltf.read_clips(glb_path)
    names = clips if clips is not None else list(raw_clips)
    bind = {nm: (i, np.asarray(sk.rest_t[i], float), mx.norm(np.asarray(sk.rest_q[i], float)), sk.parents[i])
            for i, nm in enumerate(sk.names)}
    roots = [nm for nm, b in bind.items() if b[3] < 0]
    anims, info = [], {}
    scale_dropped = []
    for cn in names:
        if cn not in raw_clips:
            raise AssetExportError('no clip %r in %s (has: %s)' % (cn, glb_path, ', '.join(raw_clips)))
        chans = raw_clips[cn]
        length = max((float(c['times'][-1]) for c in chans.values() if len(c.get('times', ()))), default=0.0)
        if not np.isfinite(length) or length <= 0 or length > ke.MAX_CLIP_S:
            raise AssetExportError('%s: bad clip length %r' % (cn, length))
        n = int(round(length * fps)) + 1
        if n > ke.MAX_KEYS:
            raise AssetExportError('%s: too many keys (%d)' % (cn, n))
        ts = np.linspace(0.0, length, n)
        bones = {}
        for bn, c in chans.items():
            if bn not in bind or bn not in rest:
                continue
            rt, rq = rest[bn]
            T = np.array([ke._sample_t(c['t_translation'], c['translation'], t) for t in ts]) if 'translation' in c else np.tile(rt, (n, 1))
            Q = np.array([ke._sample_q(c['t_rotation'], c['rotation'], t) for t in ts]) if 'rotation' in c else np.tile(rq, (n, 1))
            if 'scale' in c and np.abs(c['scale'] - 1.0).max() > 1e-3:
                if not drop_scale:
                    raise AssetExportError('%s: bone %s scales; Kenshi scale keys are unverified' % (cn, bn))
                scale_dropped.append((cn, bn))
            bones[bn] = (T, Q)
        if travel == 'remove':
            mv = ke.remove_travel(bones, roots)
        else:
            mv = np.zeros(3)
            for r in roots:
                if r in bones:
                    mv = bones[r][0][-1] - bones[r][0][0]
                    mv[1] = 0.0
        tracks = {}
        for bn, (T, Q) in bones.items():
            h, bt, bq, _ = bind[bn]
            ib = mx.q_conj(bq)
            keys = []
            for k, t in enumerate(ts):
                kq = mx.norm(mx.q_mul(ib, Q[k]))
                if kq[3] < 0:
                    kq = -kq
                keys.append((float(t), tuple(float(x) for x in kq), tuple(float(x) for x in (T[k] - bt) * ke.UNITS)))
            tracks[h] = keys
        kn = prefix + cn
        anims.append((kn, float(ts[-1]), tracks))
        info[cn] = {'kenshi_name': kn, 'length_s': round(float(ts[-1]), 4), 'keys_per_s': fps,
                    'travel_m': [round(float(x), 4) for x in mv],
                    'relocates': travel == 'keep' and float(np.hypot(mv[0], mv[2])) > ke.RELOCATE_MIN_M}
    return anims, info, scale_dropped


# ------------------------------------------------------------------ pure-numpy BC1 / BC3 DDS encoder

def _pow2_side(v, max_side):
    s = 4
    while s < v and s < max_side:
        s <<= 1
    return min(s, max(4, max_side))


def _resize_nearest(arr, h, w):
    """Simple numpy area/bilinear-free resize for fallback when Pillow is absent."""
    sh, sw = arr.shape[:2]
    if sh == h and sw == w:
        return arr
    yi = np.clip((np.arange(h) * sh) // h, 0, sh - 1)
    xi = np.clip((np.arange(w) * sw) // w, 0, sw - 1)
    return arr[yi][:, xi]


def _encode_bc1_level(rgba):
    """Encode (H, W, 4) uint8 RGBA into BC1/DXT1 raw bytes (8 bytes per 4x4 block)."""
    h, w = rgba.shape[:2]
    ph = max(4, ((h + 3) // 4) * 4)
    pw = max(4, ((w + 3) // 4) * 4)
    if ph != h or pw != w:
        pad = np.zeros((ph, pw, 4), np.uint8)
        pad[:h, :w] = rgba
        if h < ph:
            pad[h:, :w] = rgba[-1:, :w]
        if w < pw:
            pad[:, w:] = pad[:, w - 1:w]
        rgba = pad
    # Reshape to (N_blocks, 16, 3) float32
    blk = rgba[:, :, :3].reshape(ph // 4, 4, pw // 4, 4, 3).transpose(0, 2, 1, 3, 4).reshape(-1, 16, 3).astype(np.float32)
    mn = blk.min(axis=1)
    mx_ = blk.max(axis=1)
    # Slight inset (1/16) reduces outlier distortion
    inset = (mx_ - mn) * (1.0 / 16.0)
    c0f = np.clip(mx_ - inset, 0.0, 255.0)
    c1f = np.clip(mn + inset, 0.0, 255.0)

    def to565(c):
        r = np.round(c[:, 0] * (31.0 / 255.0)).astype(np.uint16)
        g = np.round(c[:, 1] * (63.0 / 255.0)).astype(np.uint16)
        b = np.round(c[:, 2] * (31.0 / 255.0)).astype(np.uint16)
        return (r << 11) | (g << 5) | b

    def from565(v):
        r = ((v >> 11) & 31).astype(np.float32) * (255.0 / 31.0)
        g = ((v >> 5) & 63).astype(np.float32) * (255.0 / 63.0)
        b = (v & 31).astype(np.float32) * (255.0 / 31.0)
        return np.stack([r, g, b], axis=-1)

    c0 = to565(c0f)
    c1 = to565(c1f)
    # Enforce c0 > c1 for 4-colour opaque mode (if equal, all pixels pick index 0)
    swap = c0 < c1
    c0, c1 = np.where(swap, c1, c0), np.where(swap, c0, c1)
    p0 = from565(c0)
    p1 = from565(c1)
    pal = np.stack([p0, p1, (2.0 * p0 + p1) / 3.0, (p0 + 2.0 * p1) / 3.0], axis=1)  # (N, 4, 3)
    diff = blk[:, :, None, :] - pal[:, None, :, :]  # (N, 16, 4, 3)
    idx = np.argmin(np.sum(diff * diff, axis=-1), axis=-1).astype(np.uint32)  # (N, 16)
    # When c0 == c1, index 0 is exact
    idx = np.where((c0 == c1)[:, None], np.uint32(0), idx)
    shifts = (np.arange(16, dtype=np.uint32) * 2)[None, :]
    bits = np.bitwise_or.reduce(idx << shifts, axis=1)
    out = np.empty(len(c0), dtype=np.dtype([('c0', '<u2'), ('c1', '<u2'), ('bits', '<u4')]))
    out['c0'] = c0
    out['c1'] = c1
    out['bits'] = bits
    return out.tobytes()


def _encode_bc3_level(rgba):
    """Encode (H, W, 4) uint8 RGBA into BC3/DXT5 raw bytes (16 bytes per 4x4 block)."""
    h, w = rgba.shape[:2]
    ph = max(4, ((h + 3) // 4) * 4)
    pw = max(4, ((w + 3) // 4) * 4)
    if ph != h or pw != w:
        pad = np.zeros((ph, pw, 4), np.uint8)
        pad[:h, :w] = rgba
        if h < ph:
            pad[h:, :w] = rgba[-1:, :w]
        if w < pw:
            pad[:, w:] = pad[:, w - 1:w]
        rgba = pad
    ablk = rgba[:, :, 3].reshape(ph // 4, 4, pw // 4, 4).transpose(0, 2, 1, 3).reshape(-1, 16).astype(np.float32)
    a0 = np.round(ablk.max(axis=1)).astype(np.uint8)
    a1 = np.round(ablk.min(axis=1)).astype(np.uint8)
    # Ensure a0 > a1 for 8-level interpolation (if equal, idx 0 is exact)
    same = a0 == a1
    a0f = a0.astype(np.float32)
    a1f = a1.astype(np.float32)
    apal = np.stack([
        a0f, a1f,
        (6.0 * a0f + 1.0 * a1f) / 7.0,
        (5.0 * a0f + 2.0 * a1f) / 7.0,
        (4.0 * a0f + 3.0 * a1f) / 7.0,
        (3.0 * a0f + 4.0 * a1f) / 7.0,
        (2.0 * a0f + 5.0 * a1f) / 7.0,
        (1.0 * a0f + 6.0 * a1f) / 7.0,
    ], axis=1)  # (N, 8)
    aidx = np.argmin(np.abs(ablk[:, :, None] - apal[:, None, :]), axis=-1).astype(np.uint64)
    aidx = np.where(same[:, None], np.uint64(0), aidx)
    shifts = (np.arange(16, dtype=np.uint64) * 3)[None, :]
    abits = np.bitwise_or.reduce(aidx << shifts, axis=1)  # 48 bits in uint64
    ab_bytes = abits.view(np.uint8).reshape(-1, 8)[:, :6]
    alpha_block = np.concatenate([a0[:, None], a1[:, None], ab_bytes], axis=1)  # (N, 8) uint8

    color_raw = np.frombuffer(_encode_bc1_level(rgba), dtype=np.uint8).reshape(-1, 8)
    return np.concatenate([alpha_block, color_raw], axis=1).tobytes()


def write_dds(rgba, path, fmt='DXT5'):
    """Write a legacy DDS (DXT1 / DXT5) with full mip chain from an (H, W, 4) uint8 RGBA array."""
    if fmt not in ('DXT1', 'DXT5'):
        raise AssetExportError('unsupported DDS format %r' % fmt)
    cur = np.asarray(rgba, np.uint8)
    if cur.ndim != 3 or cur.shape[2] != 4:
        raise AssetExportError('DDS expects (H, W, 4) RGBA array')
    h0, w0 = cur.shape[:2]
    levels = []
    enc = _encode_bc1_level if fmt == 'DXT1' else _encode_bc3_level
    while True:
        levels.append(enc(cur))
        ch, cw = cur.shape[:2]
        if ch == 1 and cw == 1:
            break
        nh, nw = max(1, ch // 2), max(1, cw // 2)
        if ch >= 2 and cw >= 2:
            cur = cur[:nh * 2, :nw * 2].reshape(nh, 2, nw, 2, 4).mean(axis=(1, 3)).round().astype(np.uint8)
        else:
            cur = _resize_nearest(cur, nh, nw)

    flags = 0x1 | 0x2 | 0x4 | 0x1000 | 0x20000 | 0x80000  # CAPS|HEIGHT|WIDTH|PIXELFORMAT|MIPMAPCOUNT|LINEARSIZE
    caps = 0x8 | 0x1000 | 0x400000  # COMPLEX|TEXTURE|MIPMAP
    hdr = bytearray(b'DDS ')
    hdr += struct.pack('<7I', 124, flags, h0, w0, len(levels[0]), 0, len(levels))
    hdr += b'\x00' * 44
    hdr += struct.pack('<II', 32, 0x4) + fmt.encode('ascii') + b'\x00' * 20
    hdr += struct.pack('<I', caps) + b'\x00' * 16
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        f.write(hdr)
        for lv in levels:
            f.write(lv)
    os.replace(tmp, path)
    return len(hdr) + sum(len(lv) for lv in levels)


# ------------------------------------------------------------------ glTF PBR -> Kenshi DDS textures

def _glb_image_rgba(js, binc, tex_ref):
    if not isinstance(tex_ref, dict) or 'index' not in tex_ref:
        return None
    try:
        from PIL import Image
    except ImportError:
        return None
    try:
        tex = js['textures'][tex_ref['index']]
        img = js['images'][tex['source']]
        if 'bufferView' not in img:
            return None
        bv = js['bufferViews'][img['bufferView']]
        off = int(bv.get('byteOffset', 0))
        raw = binc[off:off + int(bv['byteLength'])]
        return np.asarray(Image.open(io.BytesIO(raw)).convert('RGBA'), np.uint8)
    except Exception:
        return None


def _resize_rgba(arr, h, w):
    if arr is None:
        return None
    if arr.shape[0] == h and arr.shape[1] == w:
        return arr
    try:
        from PIL import Image
        return np.asarray(Image.fromarray(arr, 'RGBA').resize((w, h), Image.Resampling.BILINEAR), np.uint8)
    except ImportError:
        return _resize_nearest(arr, h, w)


def export_textures(glb_path, out_dir, stem, max_size=2048):
    """Extract PBR textures from the first material of glb_path and write Kenshi _d.dds, _n.dds, _m.dds."""
    js, binc = gltf.read_glb_raw(glb_path)
    mats = js.get('materials') or []
    mat = mats[0] if mats else {}
    pbr = mat.get('pbrMetallicRoughness') or {}
    base_factor = tuple(float(x) for x in pbr.get('baseColorFactor', (0.75, 0.72, 0.65, 1.0)))
    rough_factor = float(pbr.get('roughnessFactor', 0.85))
    metal_factor = float(pbr.get('metallicFactor', 0.0))
    cutout = mat.get('alphaMode', 'OPAQUE') in ('MASK', 'BLEND')

    base_img = _glb_image_rgba(js, binc, pbr.get('baseColorTexture'))
    mr_img = _glb_image_rgba(js, binc, pbr.get('metallicRoughnessTexture'))
    nrm_img = _glb_image_rgba(js, binc, mat.get('normalTexture'))

    def pick_size(*imgs):
        h = max((im.shape[0] for im in imgs if im is not None), default=4)
        w = max((im.shape[1] for im in imgs if im is not None), default=4)
        return _pow2_side(h, max_size), _pow2_side(w, max_size)

    # Diffuse: RGB = baseColor * linear_to_srgb(factor), A = gloss (1 - roughness)
    dh, dw = pick_size(base_img, mr_img)
    bc = _resize_rgba(base_img, dh, dw)
    mr = _resize_rgba(mr_img, dh, dw)
    tint = np.array([pow(max(0.0, min(1.0, c)), 1.0 / 2.2) for c in base_factor[:3]], np.float32)
    diff = np.empty((dh, dw, 4), np.uint8)
    rgb = (bc[:, :, :3].astype(np.float32) if bc is not None else np.full((dh, dw, 3), 255.0, np.float32))
    diff[:, :, :3] = np.clip(np.round(rgb * tint[None, None, :]), 0, 255).astype(np.uint8)
    rough = ((mr[:, :, 1].astype(np.float32) / 255.0) if mr is not None else np.ones((dh, dw), np.float32)) * rough_factor
    diff[:, :, 3] = np.clip(np.round((1.0 - rough) * 255.0), 0, 255).astype(np.uint8)

    # Normal: RGB = normal with green flipped (255 - G), A = cutout alpha or 255
    nh, nw = pick_size(nrm_img, base_img if cutout else None)
    nm = _resize_rgba(nrm_img, nh, nw)
    bc_n = _resize_rgba(base_img, nh, nw) if cutout else None
    nrm = np.empty((nh, nw, 4), np.uint8)
    if nm is not None:
        nrm[:, :, 0] = nm[:, :, 0]
        nrm[:, :, 1] = (255 - nm[:, :, 1].astype(np.int16)).astype(np.uint8)
        nrm[:, :, 2] = nm[:, :, 2]
    else:
        nrm[:, :, 0] = 128
        nrm[:, :, 1] = 128
        nrm[:, :, 2] = 255
    if cutout:
        a_src = bc_n[:, :, 3].astype(np.float32) if bc_n is not None else np.full((nh, nw), 255.0, np.float32)
        nrm[:, :, 3] = np.clip(np.round(a_src * base_factor[3]), 0, 255).astype(np.uint8)
    else:
        nrm[:, :, 3] = 255

    # Metalness: R=G=B = mr.B * metallicFactor, A = 255
    mh, mw = pick_size(mr_img)
    mr_m = _resize_rgba(mr_img, mh, mw)
    met = np.empty((mh, mw, 4), np.uint8)
    mval = ((mr_m[:, :, 2].astype(np.float32) if mr_m is not None else np.full((mh, mw), 255.0, np.float32)) * metal_factor)
    mbyte = np.clip(np.round(mval), 0, 255).astype(np.uint8)
    met[:, :, 0] = met[:, :, 1] = met[:, :, 2] = mbyte
    met[:, :, 3] = 255

    dp = os.path.join(out_dir, stem + '_d.dds')
    np_ = os.path.join(out_dir, stem + '_n.dds')
    mp = os.path.join(out_dir, stem + '_m.dds')
    write_dds(diff, dp, 'DXT5')
    write_dds(nrm, np_, 'DXT5' if cutout else 'DXT1')
    write_dds(met, mp, 'DXT1')
    return {'diffuse': dp, 'normal': np_, 'metal': mp, 'cutout': cutout,
            'sizes': {'diffuse': [dw, dh], 'normal': [nw, nh], 'metal': [mw, mh]}}


# ------------------------------------------------------------------ full asset export

def export_asset(glb_path, out_dir, name=None, clips=None, prefix='', fps=30.0,
                 travel='keep', max_texture=2048, drop_scale=True, log=print):
    """Export a rigged (or static) .glb directly to Kenshi/Meitou .mesh + .skeleton + .dds + .asset.json."""
    os.makedirs(out_dir, exist_ok=True)
    model = gltf.load_glb(glb_path)
    stem = name or os.path.splitext(os.path.basename(glb_path))[0].replace(' ', '_')
    mesh_path = os.path.join(out_dir, stem + '.mesh')
    skel_path = os.path.join(out_dir, stem + '.skeleton') if model.skeleton is not None else None

    mesh_info = write_ogre_mesh(model, skel_path, mesh_path, material_name=stem)
    skel_info = None
    clip_info = {}
    scale_dropped = []
    if skel_path is not None:
        base = skeleton_bytes(model.skeleton)
        raw_clips = gltf.read_clips(glb_path)
        if raw_clips:
            anims, clip_info, scale_dropped = _clips_to_ogre(
                glb_path, model.skeleton, clips=clips, prefix=prefix, fps=fps, travel=travel, drop_scale=drop_scale)
        else:
            anims = []
        skel_size = ke.write_skeleton(base, anims, skel_path)
        skel_info = {'path': skel_path, 'bytes': skel_size, 'bones': len(model.skeleton), 'animations': len(anims)}
        if scale_dropped:
            log('note: dropped bone scale keys on %d (clip, bone) track(s); Kenshi scale keys are unverified' % len(scale_dropped))

    tex_info = export_textures(glb_path, out_dir, stem, max_size=max_texture)
    manifest = {
        'format': 'pza.kenshi.asset', 'version': 1, 'name': stem,
        'source_glb': os.path.basename(glb_path),
        'mesh': {'path': mesh_path, 'vertices': len(model.verts), 'triangles': len(model.tris), **mesh_info},
        'skeleton': skel_info,
        'textures': tex_info,
        'clips': clip_info,
        'scale_keys_dropped': len(scale_dropped),
    }
    manifest_path = os.path.join(out_dir, stem + '.asset.json')
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=1)
    manifest['manifest'] = manifest_path
    log('exported %s: .mesh (%.1f KB, %d tris)%s + 3 DDS textures' % (
        stem, mesh_info['bytes'] / 1024.0, len(model.tris),
        (', .skeleton (%.1f KB, %d bones, %d clips)' % (skel_info['bytes'] / 1024.0, skel_info['bones'], skel_info['animations'])) if skel_info else ''))
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser(prog='python -m motionlab.asset_export',
                                 description='Export a rigged .glb directly to Kenshi .mesh + .skeleton + .dds')
    ap.add_argument('glb')
    ap.add_argument('-o', '--out', required=True)
    ap.add_argument('--name', default=None)
    ap.add_argument('--clips', default=None)
    ap.add_argument('--prefix', default='')
    ap.add_argument('--fps', type=float, default=30.0)
    ap.add_argument('--travel', choices=('keep', 'remove'), default='keep')
    ap.add_argument('--max-texture', type=int, default=2048)
    a = ap.parse_args(argv)
    clips = [c.strip() for c in a.clips.split(',') if c.strip()] if a.clips else None
    export_asset(a.glb, a.out, name=a.name, clips=clips, prefix=a.prefix, fps=a.fps,
                 travel=a.travel, max_texture=a.max_texture)
    return 0


if __name__ == '__main__':
    sys.exit(main())
