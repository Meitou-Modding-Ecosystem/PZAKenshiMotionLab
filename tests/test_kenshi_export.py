# PZA's Kenshi Motion Lab - baked clips -> Kenshi animation file (.skeleton): written bytes, round trip, contract.
# Author: PZA. GPLv3.   python tests/test_kenshi_export.py      (no game files: a procedural Bip01 body)
import os, sys, struct, tempfile, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..')); sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from synthetic import make_biped
from motionlab import pipeline, kenshi_export as ke, mathx as mx
from motionlab.ogre import skeleton_read

fails = []
def ok(c, msg):
    print(('PASS ' if c else 'FAIL ') + msg)
    if not c: fails.append(msg)


def kenshi_skeleton_bytes(sk, version='[Serializer_v1.80]'):
    """A bones-only Ogre .skeleton like Kenshi's (decimetres, handle = bone index, Ogre's bone length quirk)."""
    out = bytearray(struct.pack('<H', 0x1000) + version.encode() + b'\n')
    out += struct.pack('<HIH', 0x1010, 8, 0)
    for i, nm in enumerate(sk.names):
        t = np.array(sk.rest_t[i]) * 10.0; q = mx.norm(sk.rest_q[i])
        out += struct.pack('<HI', 0x2000, 6 + 2 + 12 + 16) + nm.encode() + b'\n' + struct.pack('<H3f4f', i, *t, *q)
    for i, p in enumerate(sk.parents):
        if p >= 0: out += struct.pack('<HIHH', 0x3000, 10, i, p)
    return bytes(out)


d = tempfile.mkdtemp()
m = make_biped(); src = os.path.join(d, 'biped.glb')
from motionlab import gltf
gltf.write_glb(src, m)
r = pipeline.rig_file(src, d, log=lambda *a: None, png=False, clips='walk,run,jump_up,jump_forward,jump_walk')
skp = os.path.join(d, 'body.skeleton'); base = kenshi_skeleton_bytes(m.skeleton)
open(skp, 'wb').write(base)

# 1. the prefix is the bones exactly; a file with animations keeps only its bones
ok(ke.skeleton_prefix(base) == base, 'bones-only skeleton: prefix is the whole file')
anims, info = ke.build(r['glb'], skp, ['jump_up', 'jump_forward', 'jump_walk'])
out = os.path.join(d, 'pza_jumps.skeleton')
ke.write_skeleton(base, anims, out)
data = open(out, 'rb').read()
ok(data.startswith(base) and ke.skeleton_prefix(data) == base, 'written file starts with the body bones byte for byte')
v, bones, parents, an = skeleton_read.read(out)
ok([a[0] for a in an] == ['pza_jump_up', 'pza_jump_forward', 'pza_jump_walk'], 'animations named with the prefix: %s' % [a[0] for a in an])
ok(all(len(tr) == len(m.skeleton.names) for _, _, tr in an), 'every bone has a track')
# keyframes are Ogre's 38-byte chunks (no scale): the reader would mis-step on anything else
p = len(base) + 6; nm_end = data.index(b'\n', p); p = nm_end + 1 + 4 + 6 + 2
ok(struct.unpack_from('<HI', data, p) == (0x4110, 38), 'first keyframe chunk id 0x4110, length 38')

# 2. Ogre's playback of the file = the .glb's pose, every bone, every frame (travel kept)
got = ke.clip_locals(r['glb'], 30.0); worst = 0.0
for cn in ('jump_up', 'jump_forward', 'jump_walk'):
    ts, bl = got[cn]; names, par, smp, L = ke.sampler(out, 'pza_' + cn)
    ix = {n: i for i, n in enumerate(m.skeleton.names)}
    for k in range(0, len(ts), 3):
        Wf = smp(float(ts[k])); Wg = [None] * len(names)
        for i in range(len(names)):
            T, Q = bl[names[i]] if names[i] in bl else (None, None)
            Lm = mx.trs(T[k], Q[k], np.ones(3)) if T is not None else mx.trs(m.skeleton.rest_t[ix[names[i]]], m.skeleton.rest_q[ix[names[i]]], np.ones(3))
            Wg[i] = Wg[par[i]] @ Lm if par[i] >= 0 else Lm
        worst = max(worst, float(np.abs(np.array(Wg)[:, :3, 3] - Wf[:, :3, 3]).max()))
ok(worst < 1e-4, 'file plays back the .glb pose (worst %.6f m)' % worst)

# 3. contract numbers measured from the file match the Motion Lab's own jump timings
C = ke.contract({'body': out}, info)
ju, jf, jw = C['clips']['jump_up'], C['clips']['jump_forward'], C['clips']['jump_walk']
ok(abs(ju['takeoff_s'] - 0.67) < 0.04 and abs(ju['air_s'] - 0.55) < 0.05, 'jump_up: take-off %.3f s, air %.3f s' % (ju['takeoff_s'], ju['air_s']))
ok(ju['rise_m'] > 0.2 and ju['clearance_m'] > 0.2, 'jump_up: rise %.2f m, feet clear %.2f m' % (ju['rise_m'], ju['clearance_m']))
H0 = m.skeleton.world_rest()[m.skeleton.index('Bip01')][1, 3]
ok(jf['relocates'] and not ju['relocates'] and abs(jf['travel_m'][2] - 1.8 * H0) < 0.06,
   'jump_forward relocates %.2f m forward (1.8 x %.2f), jump_up stays put' % (jf['travel_m'][2], H0))
ok(jw['kind'] == 'stride' and jw['from_gait'] == 'walk' and abs(jw['takeoff_s'] - 0.10) < 0.03, 'jump_walk: stride from the walk, off at %.3f s' % jw['takeoff_s'])
ok(json.loads(json.dumps(C))['format'] == ke.CONTRACT_FORMAT, 'contract is plain JSON')

# 4. travel removed: the root ends where it starts, the height kept
an2, info2 = ke.build(r['glb'], skp, ['jump_forward'], travel='remove')
ok(abs(info2['jump_forward']['travel_m'][2] - jf['travel_m'][2]) < 1e-6 and not info2['jump_forward']['relocates'], 'remove: travel reported, not relocating')
h = m.skeleton.names.index('Bip01'); keys = an2[0][2][h]
ok(abs(keys[-1][2][2] - keys[0][2][2]) < 1e-4 and max(k[2][1] for k in keys) > 1.0, 'remove: root back at its start, height still in the track')

# 5. refusals
try:
    ke.build(r['glb'], skp, ['no_such']); ok(False, 'unknown clip refused')
except ke.ExportError: ok(True, 'unknown clip refused')
small = make_biped(); keep = [i for i, n in enumerate(small.skeleton.names) if 'Toe' not in n]
sk2 = type(small.skeleton)()
for i in keep:
    sk2.names.append(small.skeleton.names[i]); sk2.parents.append(keep.index(small.skeleton.parents[i]) if small.skeleton.parents[i] >= 0 else -1)
    sk2.rest_t.append(small.skeleton.rest_t[i]); sk2.rest_q.append(small.skeleton.rest_q[i]); sk2.rest_s.append(np.ones(3))
sk2p = os.path.join(d, 'short.skeleton'); open(sk2p, 'wb').write(kenshi_skeleton_bytes(sk2))
try:
    ke.build(r['glb'], sk2p, ['jump_up']); ok(False, 'a target skeleton missing bones is refused')
except ke.ExportError as e: ok('not in the target skeleton' in str(e), 'a target skeleton missing bones is refused')
for bad in (b'', b'\x00\x10', b'\x00\x10[Serializer_v1.80]', b'\x01\x00junk'):
    try: ke.skeleton_prefix(bad); ok(len(bad) > 2 and bad[:2] == b'\x00\x10', 'prefix of %r' % bad[:8])
    except ke.ExportError: ok(True, 'bad file refused: %r' % bad[:8])

print('\n%d failed' % len(fails) if fails else '\nall passed')
sys.exit(1 if fails else 0)
