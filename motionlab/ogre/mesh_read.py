# Minimal reader for Ogre binary .mesh files (MeshSerializer_v1.100 / v1.8 / v1.40). Author: PZA. Read-only:
# returns submeshes, vertex elements, bone assignments and the skeleton link (run it on a file for a summary).
# usage: python mesh_read.py <file.mesh> [--verts N]
import struct, sys

M_HEADER = 0x1000
M_MESH = 0x3000
M_SUBMESH = 0x4000
M_SUBMESH_OPERATION = 0x4010
M_SUBMESH_BONE_ASSIGNMENT = 0x4100
M_SUBMESH_TEXTURE_ALIAS = 0x4200
M_GEOMETRY = 0x5000
M_GEOMETRY_VERTEX_DECLARATION = 0x5100
M_GEOMETRY_VERTEX_ELEMENT = 0x5110
M_GEOMETRY_VERTEX_BUFFER = 0x5200
M_GEOMETRY_VERTEX_BUFFER_DATA = 0x5210
M_MESH_SKELETON_LINK = 0x6000
M_MESH_BONE_ASSIGNMENT = 0x7000
M_MESH_BOUNDS = 0x9000
M_SUBMESH_NAME_TABLE = 0xA000

SEMANTICS = {1: 'position', 2: 'blend_weights', 3: 'blend_indices', 4: 'normal', 5: 'diffuse',
             6: 'specular', 7: 'uv', 8: 'binormal', 9: 'tangent'}
# vertex element type -> (struct format, byte size)
TYPES = {0: ('f', 4), 1: ('2f', 8), 2: ('3f', 12), 3: ('4f', 16), 4: ('I', 4), 5: ('h', 2),
         6: ('2h', 4), 7: ('3h', 6), 8: ('4h', 8), 9: ('4B', 4), 10: ('4B', 4), 11: ('4B', 4)}

class R:
    def __init__(self, d): self.d, self.p = d, 0
    def u16(self): v = struct.unpack_from('<H', self.d, self.p)[0]; self.p += 2; return v
    def u32(self): v = struct.unpack_from('<I', self.d, self.p)[0]; self.p += 4; return v
    def f32(self): v = struct.unpack_from('<f', self.d, self.p)[0]; self.p += 4; return v
    def b(self): v = self.d[self.p] != 0; self.p += 1; return v
    def s(self):
        e = self.d.index(b'\n', self.p); v = self.d[self.p:e].decode('latin-1'); self.p = e + 1; return v
    def raw(self, n): v = self.d[self.p:self.p + n]; self.p += n; return v
    def chunk(self):
        if self.p + 6 > len(self.d): return None, 0
        cid = self.u16(); ln = self.u32(); return cid, ln

def read_geometry(r, end):
    g = {'count': r.u32(), 'elements': [], 'buffers': {}}
    while r.p < end:
        start = r.p
        cid, ln = r.chunk()
        if cid == M_GEOMETRY_VERTEX_DECLARATION:
            dend = start + ln
            while r.p < dend:
                s2 = r.p; c2, l2 = r.chunk()
                if c2 != M_GEOMETRY_VERTEX_ELEMENT: r.p = s2; break
                src, typ, sem, off, idx = r.u16(), r.u16(), r.u16(), r.u16(), r.u16()
                g['elements'].append((src, typ, sem, off, idx))
            r.p = dend
        elif cid == M_GEOMETRY_VERTEX_BUFFER:
            bind, vsize = r.u16(), r.u16()
            c2, l2 = r.chunk()
            if c2 != M_GEOMETRY_VERTEX_BUFFER_DATA: raise ValueError('vertex buffer without data chunk (0x%X)' % c2)
            g['buffers'][bind] = (vsize, r.raw(vsize * g['count']))
            r.p = start + ln
        else:
            r.p = start; break
    return g

def element(g, vi, el):
    src, typ, sem, off, idx = el
    vsize, data = g['buffers'][src]
    fmt, size = TYPES[typ]
    return struct.unpack_from('<' + fmt, data, vi * vsize + off)

def read(path):
    d = open(path, 'rb').read()
    r = R(d)
    if r.u16() != M_HEADER: raise ValueError('not an Ogre .mesh file')
    version = r.s()
    mesh = {'version': version, 'subs': [], 'shared': None, 'skeleton': None, 'mesh_bones': 0, 'shared_bones': []}
    cid, ln = r.chunk()
    if cid != M_MESH: raise ValueError('no mesh chunk (0x%X)' % (cid or 0))
    mend = r.p - 6 + ln
    mesh['animated'] = r.b()
    while r.p < mend:
        start = r.p
        cid, ln = r.chunk()
        if cid is None: break
        if cid == M_SUBMESH:
            sub = {'material': r.s(), 'shared': r.b()}
            n = r.u32(); i32 = r.b()
            sub['indices'] = struct.unpack_from('<%d%s' % (n, 'I' if i32 else 'H'), d, r.p); r.p += n * (4 if i32 else 2)
            sub['geometry'] = None; sub['bones'] = []; sub['op'] = 4      # 4 = triangle list (Ogre's default)
            send = start + ln
            while r.p < send:
                s2 = r.p; c2, l2 = r.chunk()
                if c2 == M_GEOMETRY: sub['geometry'] = read_geometry(r, s2 + l2); r.p = s2 + l2
                elif c2 == M_SUBMESH_BONE_ASSIGNMENT:
                    sub['bones'].append((r.u32(), r.u16(), r.f32())); r.p = s2 + l2
                elif c2 == M_SUBMESH_OPERATION:
                    sub['op'] = r.u16(); r.p = s2 + l2
                else: r.p = s2 + l2
            mesh['subs'].append(sub)
            r.p = send
        elif cid == M_GEOMETRY:
            mesh['shared'] = read_geometry(r, start + ln); r.p = start + ln
        elif cid == M_MESH_SKELETON_LINK:
            mesh['skeleton'] = r.s(); r.p = start + ln
        elif cid == M_MESH_BONE_ASSIGNMENT:
            mesh['mesh_bones'] += 1; mesh['shared_bones'].append((r.u32(), r.u16(), r.f32())); r.p = start + ln
        else:
            r.p = start + ln
    return mesh

def main():
    path = sys.argv[1]
    nv = int(sys.argv[sys.argv.index('--verts') + 1]) if '--verts' in sys.argv else 2
    m = read(path)
    print('%s: %s, skeleton=%s, skeletally animated=%s, shared geometry=%s, mesh-level bone assignments=%d' % (
        path.split('\\')[-1].split('/')[-1], m['version'], m['skeleton'], m['animated'],
        m['shared']['count'] if m['shared'] else None, m['mesh_bones']))
    for i, s in enumerate(m['subs']):
        g = s['geometry'] or m['shared']
        sems = ['%s(t%d,src%d)' % (SEMANTICS.get(e[2], e[2]), e[1], e[0]) for e in g['elements']] if g else []
        print('  sub %d: material=%s verts=%s tris=%d bone assignments=%d elements=%s' % (
            i, s['material'], g['count'] if g else '-', len(s['indices']) // 3, len(s['bones']), ', '.join(sems)))
        if g:
            for vi in range(min(nv, g['count'])):
                vals = []
                for el in g['elements']:
                    v = element(g, vi, el)
                    vals.append('%s=%s' % (SEMANTICS.get(el[2], el[2]), tuple(round(x, 3) if isinstance(x, float) else x for x in v)))
                print('    v%d: %s' % (vi, '  '.join(vals)))
            bw = {}
            for vi, bone, w in s['bones']: bw.setdefault(vi, []).append(w)
            if bw:
                mx = max(len(x) for x in bw.values())
                print('    weights per vertex: max %d, verts with weights %d of %d' % (mx, len(bw), g['count']))

if __name__ == '__main__':
    main()
