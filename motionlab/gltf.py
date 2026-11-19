# PZA's Kenshi Motion Lab - glTF 2.0 binary (.glb) reader and writer. Author: PZA. GPLv3.
# Reads one skinned (or plain) mesh into a Model; writes a Model + baked clips back as a new .glb that keeps the
# source's materials, textures and extra vertex attributes (tangents, extra UVs) untouched.
import json, struct
import numpy as np
from . import mathx as mx
from .model import Model, Skeleton

CT = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
NC = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4, 'MAT2': 4, 'MAT3': 9, 'MAT4': 16}
MAX_GLB = 512 * 1024 * 1024


class GlbError(ValueError):
    pass


def _at(lst, i, what):
    """lst[i] for an index read from the file: refused unless it is a real, in-range, non-negative index."""
    if not isinstance(lst, list) or isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < len(lst):
        raise GlbError('%s index %r is out of range' % (what, i))
    return lst[i]


def read_glb_raw(path_or_bytes):
    if isinstance(path_or_bytes, (bytes, bytearray)):
        d = path_or_bytes
    else:
        import os
        if os.path.getsize(path_or_bytes) > MAX_GLB: raise GlbError('file is larger than %d MB' % (MAX_GLB >> 20))
        with open(path_or_bytes, 'rb') as f: d = f.read()
    if len(d) > MAX_GLB: raise GlbError('file is larger than %d MB' % (MAX_GLB >> 20))
    if len(d) < 20 or d[:4] != b'glTF': raise GlbError('not a .glb file (no glTF magic)')
    ver, total = struct.unpack_from('<II', d, 4)
    if ver != 2: raise GlbError('glTF version %d, only 2 is supported' % ver)
    if total > len(d): raise GlbError('length field %d is past the end of the file (%d bytes)' % (total, len(d)))
    p = 12; js = None; binc = b''
    while p + 8 <= total:
        ln, typ = struct.unpack_from('<II', d, p); p += 8
        if ln > total - p: raise GlbError('chunk of %d bytes runs past the file' % ln)
        c = d[p:p + ln]; p += (ln + 3) & ~3
        if typ == 0x4E4F534A: js = json.loads(c.decode('utf-8'))
        elif typ == 0x004E4942: binc = bytes(c)
    if js is None: raise GlbError('no JSON chunk')
    return js, binc


def accessor(js, binc, i):
    a = _at(js.get('accessors'), i, 'accessor')
    if a.get('type') not in NC or a.get('componentType') not in CT: raise GlbError('accessor %d: unknown type' % i)
    n = NC[a['type']]; dt = np.dtype(CT[a['componentType']]); cnt = int(a['count'])
    if cnt < 0 or cnt > 50_000_000: raise GlbError('accessor %d: bad count %d' % (i, cnt))
    if 'bufferView' not in a:
        out = np.zeros((cnt, n), dt)
    else:
        bv = _at(js.get('bufferViews'), a['bufferView'], 'bufferView')
        if bv.get('buffer', 0) != 0: raise GlbError('external buffers are not supported (GLB only)')
        off = int(bv.get('byteOffset', 0)) + int(a.get('byteOffset', 0))
        stride = int(bv.get('byteStride', 0)) or n * dt.itemsize
        need = off + stride * (cnt - 1) + n * dt.itemsize if cnt else off
        if off < 0 or need > len(binc): raise GlbError('accessor %d runs past the binary chunk' % i)
        if stride < n * dt.itemsize: raise GlbError('accessor %d: byteStride %d is smaller than one element' % (i, stride))
        if stride == n * dt.itemsize:
            out = np.frombuffer(binc, dt, cnt * n, off).reshape(cnt, n).copy()
        else:                                   # interleaved: one strided view, no per-element loop
            out = np.ndarray(shape=(cnt, n), dtype=dt, buffer=binc, offset=off, strides=(stride, dt.itemsize)).copy()
    if 'sparse' in a: raise GlbError('sparse accessors are not supported')
    if a.get('normalized') and dt.kind in 'ui':
        out = out.astype(np.float64) / np.iinfo(dt).max
    return out


def node_local(nd):
    if 'matrix' in nd: return np.array(nd['matrix'], float).reshape(4, 4).T
    return mx.trs(nd.get('translation', (0, 0, 0)), nd.get('rotation', (0, 0, 0, 1)), nd.get('scale', (1, 1, 1)))


def node_worlds(js):
    nodes = js.get('nodes', [])
    par = [-1] * len(nodes)
    for i, nd in enumerate(nodes):
        for c in nd.get('children', []):
            if not 0 <= c < len(nodes): raise GlbError('node %d has a bad child %d' % (i, c))
            par[c] = i
    W = [None] * len(nodes)
    def w(i, depth=0):
        if depth > 512: raise GlbError('node hierarchy has a cycle')
        if W[i] is None:
            W[i] = node_local(nodes[i]) if par[i] < 0 else w(par[i], depth + 1) @ node_local(nodes[i])
        return W[i]
    for i in range(len(nodes)): w(i)
    return W, par


def load_glb(path):
    js, binc = read_glb_raw(path)
    W, par = node_worlds(js)
    nodes = js.get('nodes', [])
    mesh_nodes = [i for i, nd in enumerate(nodes) if 'mesh' in nd]
    if not mesh_nodes: raise GlbError('the file holds no mesh')
    skinned = [i for i in mesh_nodes if 'skin' in nodes[i]]
    mi = (skinned or mesh_nodes)[0]
    mnode = nodes[mi]; mesh = _at(js.get('meshes'), mnode['mesh'], 'mesh')
    pos, nrm, uv, tris, J, Wt, parts, extra = [], [], [], [], [], [], [], {}
    base = 0
    for pi, prim in enumerate(mesh['primitives']):
        if prim.get('mode', 4) != 4: continue
        at = prim['attributes']
        P = accessor(js, binc, at['POSITION']).astype(np.float64)
        nv = len(P)
        N = accessor(js, binc, at['NORMAL']).astype(np.float64) if 'NORMAL' in at else np.zeros((nv, 3))
        T = accessor(js, binc, at['TEXCOORD_0']).astype(np.float64) if 'TEXCOORD_0' in at else np.zeros((nv, 2))
        I = accessor(js, binc, prim['indices']).reshape(-1).astype(np.int64) if 'indices' in prim else np.arange(nv)
        if len(I) % 3 or (len(I) and (I.min() < 0 or I.max() >= nv)): raise GlbError('primitive %d: bad indices' % pi)
        for k in at:
            if k not in ('POSITION', 'NORMAL', 'TEXCOORD_0') and not k.startswith(('JOINTS_', 'WEIGHTS_')):
                extra.setdefault(k, []).append(accessor(js, binc, at[k]).astype(np.float64))
        if 'skin' in mnode:
            jk = sorted(a for a in at if a.startswith('JOINTS_')); wk = sorted(a for a in at if a.startswith('WEIGHTS_'))
            if len(jk) != len(wk): raise GlbError('primitive %d: JOINTS and WEIGHTS sets do not pair up' % pi)
            if jk:
                jj = np.concatenate([accessor(js, binc, at[k]).astype(np.int64) for k in jk], 1)
                ww = np.concatenate([accessor(js, binc, at[k]).astype(np.float64) for k in wk], 1)
                if len(jj) != nv or len(ww) != nv or jj.shape != ww.shape: raise GlbError('primitive %d: skin data does not match its vertices' % pi)
            else:                               # an unskinned primitive in a skinned mesh: no weights (follows its node)
                jj = np.zeros((nv, 4), np.int64); ww = np.zeros((nv, 4))
            J.append(jj); Wt.append(ww)
        parts.append({'material': prim.get('material'), 'tri_start': len(tris) and sum(len(t) for t in tris), 'tri_count': len(I) // 3})
        pos.append(P); nrm.append(N); uv.append(T); tris.append(I.reshape(-1, 3) + base); base += nv
    if not pos: raise GlbError('no triangle primitives in the mesh')
    m = Model()
    m.verts = np.concatenate(pos); m.normals = np.concatenate(nrm); m.uvs = np.concatenate(uv); m.tris = np.concatenate(tris)
    m.parts = parts
    m.extra_attrs = {k: np.concatenate(v) for k, v in extra.items() if sum(len(x) for x in v) == len(m.verts)}
    m.source = {'kind': 'glb', 'json': js, 'bin': binc, 'mesh_name': mesh.get('name', 'mesh')}
    m.name = mesh.get('name') or mnode.get('name') or 'creature'
    if 'skin' in mnode and J and any(w.any() for w in Wt):
        skin = _at(js.get('skins'), mnode['skin'], 'skin')
        joints = skin.get('joints') or []
        for j in joints: _at(nodes, j, 'joint node')
        if not joints: raise GlbError('the skin has no joints')
        ibm = accessor(js, binc, skin['inverseBindMatrices']).astype(np.float64).reshape(-1, 4, 4).transpose(0, 2, 1) \
            if 'inverseBindMatrices' in skin else np.array([np.eye(4)] * len(joints))
        if len(ibm) != len(joints): raise GlbError('the skin has %d joints but %d bind matrices' % (len(joints), len(ibm)))
        jset = set(joints)
        sk = Skeleton(); sk.tails = {}
        for j in joints:
            nd = nodes[j]
            sk.names.append(nd.get('name', 'joint_%d' % j))
            tl = (nd.get('extras') or {}).get('motionlab_tail')
            if isinstance(tl, list) and len(tl) == 3: sk.tails[sk.names[-1]] = [float(x) for x in tl]
            p = par[j]
            while p >= 0 and p not in jset: p = par[p]
            sk.parents.append(joints.index(p) if p >= 0 else -1)
            t, q, s = mx.decompose(node_local(nd)) if 'matrix' in nd else (
                np.array(nd.get('translation', (0, 0, 0)), float), np.array(nd.get('rotation', (0, 0, 0, 1)), float),
                np.array(nd.get('scale', (1, 1, 1)), float))
            sk.rest_t.append(t); sk.rest_q.append(mx.norm(q)); sk.rest_s.append(s)
        # world rest of the skeleton root's non-joint ancestors
        sk.root_world = []
        for k, j in enumerate(joints):
            sk.root_world.append(W[par[j]] if (sk.parents[k] < 0 and par[j] >= 0) else np.eye(4))
        K = max(x.shape[1] for x in J)
        Jm = np.concatenate([np.pad(x, ((0, 0), (0, K - x.shape[1]))) for x in J])
        Wm = np.concatenate([np.pad(x, ((0, 0), (0, K - x.shape[1]))) for x in Wt])
        if Jm.min() < 0: raise GlbError('a vertex uses a negative joint index')
        s = Wm.sum(1, keepdims=True); Wm = np.where(s > 0, Wm / np.maximum(s, 1e-12), 0)
        if Jm.max() >= len(joints): raise GlbError('a vertex uses joint %d, the skin has %d' % (Jm.max(), len(joints)))
        # bring the mesh into world rest space (bind pose as the file shows it), so every later rig is in one space
        Wj = sk.world_rest()
        S = np.einsum('bij,bjk->bik', Wj, ibm)
        Pw = np.zeros_like(m.verts); Nw = np.zeros_like(m.normals)
        hp = np.c_[m.verts, np.ones(len(m.verts))]
        for k in range(Jm.shape[1]):
            M = S[Jm[:, k]]
            Pw += Wm[:, k:k + 1] * np.einsum('nij,nj->ni', M, hp)[:, :3]
            Nw += Wm[:, k:k + 1] * np.einsum('nij,nj->ni', M[:, :3, :3], m.normals)
        unw = s[:, 0] <= 0
        if unw.any():
            Mw = W[mi]; Pw[unw] = (hp[unw] @ Mw.T)[:, :3]; Nw[unw] = m.normals[unw] @ Mw[:3, :3].T
        m.verts = Pw; m.normals = mx.norm(Nw)
        m.skeleton = sk; m.joints = Jm; m.weights = Wm
        try:
            m.ibm = np.array([np.linalg.inv(w) for w in Wj])
        except np.linalg.LinAlgError:
            raise GlbError('a joint has a zero scale (its rest matrix cannot be inverted)')
    else:
        Mw = W[mi]
        m.verts = (np.c_[m.verts, np.ones(len(m.verts))] @ Mw.T)[:, :3]
        m.normals = mx.norm(m.normals @ Mw[:3, :3].T)
    m.texture = base_color_image(js, binc, parts[0]['material'] if parts else None)
    return m


def base_color_image(js, binc, mat):
    m = {'factor': (0.75, 0.72, 0.65, 1.0), 'image': None}
    try:
        if mat is None: return m
        pbr = js['materials'][mat].get('pbrMetallicRoughness', {})
        m['factor'] = tuple(pbr.get('baseColorFactor', (1, 1, 1, 1)))
        if 'baseColorTexture' in pbr:
            img = js['images'][js['textures'][pbr['baseColorTexture']['index']]['source']]
            if 'bufferView' in img:
                import io
                from PIL import Image
                bv = js['bufferViews'][img['bufferView']]
                o = bv.get('byteOffset', 0)
                im = Image.open(io.BytesIO(binc[o:o + bv['byteLength']])).convert('RGB')
                im.thumbnail((1024, 1024))
                m['image'] = np.asarray(im, np.float32) / 255.0
    except Exception:
        pass
    return m


# ---------------------------------------------------------------- writer

class _Bin:
    def __init__(self):
        self.data = bytearray(); self.views = []; self.accessors = []

    def view(self, raw, target=None):
        while len(self.data) % 4: self.data.append(0)
        v = {'buffer': 0, 'byteOffset': len(self.data), 'byteLength': len(raw)}
        if target: v['target'] = target
        self.data += raw; self.views.append(v); return len(self.views) - 1

    def acc(self, arr, ctype, typ, target=None, minmax=False, normalized=False):
        arr = np.ascontiguousarray(arr, dtype=CT[ctype])
        n = NC[typ]
        flat = arr.reshape(-1, n)
        a = {'bufferView': self.view(flat.tobytes(), target), 'componentType': ctype, 'count': len(flat), 'type': typ}
        if normalized: a['normalized'] = True
        if minmax:
            a['min'] = [float(x) for x in flat.min(0)]; a['max'] = [float(x) for x in flat.max(0)]
        self.accessors.append(a); return len(self.accessors) - 1


def write_glb(path, model, clips=(), skeleton=None, joints=None, weights=None, tails=None):
    """Write `model` (world-rest geometry) skinned to `skeleton` (default model.skeleton) with `clips`
    (list of bake.Clip). Materials and images are copied from the source .glb when there is one."""
    sk = skeleton or model.skeleton
    J = model.joints if joints is None else joints
    Wt = model.weights if weights is None else weights
    b = _Bin(); js = {'asset': {'version': '2.0', 'generator': "PZA's Kenshi Motion Lab"}, 'scene': 0}
    src = model.source if model.source and model.source.get('kind') == 'glb' else None
    # materials, textures, samplers, images (copied; image bytes re-packed into our buffer)
    if src:
        sj, sb = src['json'], src['bin']
        imgs = []
        for img in sj.get('images', []):
            img = dict(img)
            if 'bufferView' in img:
                bv = sj['bufferViews'][img['bufferView']]; o = bv.get('byteOffset', 0)
                img['bufferView'] = b.view(sb[o:o + bv['byteLength']])
            imgs.append(img)
        for k, v in (('images', imgs), ('textures', sj.get('textures')), ('samplers', sj.get('samplers')),
                     ('materials', sj.get('materials'))):
            if v: js[k] = v
    nv = len(model.verts)
    attrs = {'POSITION': b.acc(model.verts, 5126, 'VEC3', 34962, minmax=True),
             'NORMAL': b.acc(mx.norm(model.normals), 5126, 'VEC3', 34962),
             'TEXCOORD_0': b.acc(model.uvs, 5126, 'VEC2', 34962)}
    for k, v in (model.extra_attrs or {}).items():
        typ = {1: 'SCALAR', 2: 'VEC2', 3: 'VEC3', 4: 'VEC4'}.get(v.shape[1])
        if typ is None: continue                 # matrix-shaped extras are not vertex attributes we carry over
        attrs[k] = b.acc(v, 5126, typ, 34962)
    nodes = []
    if sk is not None:
        K = 4
        Jp = np.zeros((nv, K), np.uint16); Wp = np.zeros((nv, K), np.float32)
        kk = min(K, J.shape[1]); Jp[:, :kk] = J[:, :kk]; Wp[:, :kk] = Wt[:, :kk]
        s = Wp.sum(1, keepdims=True); Wp = np.where(s > 0, Wp / np.maximum(s, 1e-12), 0)
        attrs['JOINTS_0'] = b.acc(Jp, 5123, 'VEC4', 34962)
        attrs['WEIGHTS_0'] = b.acc(Wp, 5126, 'VEC4', 34962)
    prims = []
    for part in model.parts:
        I = model.tris[part['tri_start']:part['tri_start'] + part['tri_count']].reshape(-1)
        pr = {'attributes': attrs, 'indices': b.acc(I.astype(np.uint32), 5125, 'SCALAR', 34963), 'mode': 4}
        if part.get('material') is not None and 'materials' in js: pr['material'] = part['material']
        prims.append(pr)
    js['meshes'] = [{'name': model.name, 'primitives': prims}]
    root = {'name': model.name, 'children': []}; nodes.append(root)
    mesh_node = {'name': model.name + '_mesh', 'mesh': 0}; nodes.append(mesh_node); root['children'].append(1)
    if sk is not None:
        first = len(nodes)
        for i, nm in enumerate(sk.names):
            nd = {'name': nm, 'translation': [float(x) for x in sk.rest_t[i]],
                  'rotation': [float(x) for x in mx.norm(sk.rest_q[i])]}
            if np.any(np.abs(np.asarray(sk.rest_s[i]) - 1) > 1e-9): nd['scale'] = [float(x) for x in sk.rest_s[i]]
            tl = (tails or {}).get(nm)
            if tl is not None: nd['extras'] = {'motionlab_tail': [float(x) for x in tl]}   # bone end, world rest (m)
            nodes.append(nd)
        for i, p in enumerate(sk.parents):
            if p < 0: root['children'].append(first + i)
            else: nodes[first + p].setdefault('children', []).append(first + i)
        Wr = sk.world_rest()
        ibm = np.array([np.linalg.inv(w) for w in Wr])
        js['skins'] = [{'name': model.name + '_rig', 'joints': list(range(first, first + len(sk.names))),
                        'inverseBindMatrices': b.acc(ibm.transpose(0, 2, 1), 5126, 'MAT4'),
                        'skeleton': first + [i for i, p in enumerate(sk.parents) if p < 0][0]}]
        mesh_node['skin'] = 0
        anims = []
        for clip in clips:
            ch, sm = [], []
            tin = b.acc(clip.times.astype(np.float32), 5126, 'SCALAR', minmax=True)
            for bi, tr in clip.tracks.items():
                for path_, arr, typ in (('rotation', tr.get('r'), 'VEC4'), ('translation', tr.get('t'), 'VEC3'),
                                        ('scale', tr.get('s'), 'VEC3')):
                    if arr is None: continue
                    sm.append({'input': tin, 'output': b.acc(np.asarray(arr, np.float32), 5126, typ), 'interpolation': 'LINEAR'})
                    ch.append({'sampler': len(sm) - 1, 'target': {'node': first + bi, 'path': path_}})
            anims.append({'name': clip.name, 'channels': ch, 'samplers': sm})
        if anims: js['animations'] = anims
    js['nodes'] = nodes; js['scenes'] = [{'name': 'Scene', 'nodes': [0]}]
    js['bufferViews'] = b.views; js['accessors'] = b.accessors
    while len(b.data) % 4: b.data.append(0)
    js['buffers'] = [{'byteLength': len(b.data)}]
    jb = json.dumps(js, separators=(',', ':')).encode('utf-8')
    jb += b' ' * ((4 - len(jb) % 4) % 4)
    total = 12 + 8 + len(jb) + 8 + len(b.data)
    out = struct.pack('<4sII', b'glTF', 2, total) + struct.pack('<II', len(jb), 0x4E4F534A) + jb + \
        struct.pack('<II', len(b.data), 0x004E4942) + bytes(b.data)
    tmp = str(path) + '.tmp'
    with open(tmp, 'wb') as f: f.write(out)
    import os
    os.replace(tmp, path)
    return total


def read_clips(path):
    """Baked clips of a .glb as {name: {node_name: {'times':..., 'rotation'/'translation'/'scale': array}}}."""
    js, binc = read_glb_raw(path)
    out = {}
    for an in js.get('animations', []):
        d = {}
        for ch in an['channels']:
            sm = an['samplers'][ch['sampler']]
            nm = js['nodes'][ch['target']['node']].get('name', str(ch['target']['node']))
            e = d.setdefault(nm, {})
            path = ch['target']['path']
            if path not in ('rotation', 'translation', 'scale'): continue
            if sm.get('interpolation', 'LINEAR') == 'CUBICSPLINE': raise GlbError('cubic-spline animation is not supported')
            t = accessor(js, binc, sm['input'])[:, 0].astype(np.float64)
            v = accessor(js, binc, sm['output']).astype(np.float64)
            if len(v) != len(t): raise GlbError('animation channel has %d keys for %d times' % (len(v), len(t)))
            e['t_' + path] = t; e[path] = v           # each channel keeps its own key times (exporters differ)
            if len(t) >= len(e.get('times', ())): e['times'] = t
        out[an.get('name', 'clip')] = d
    return out
