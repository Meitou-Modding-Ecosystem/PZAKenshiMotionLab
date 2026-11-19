# PZA's Kenshi Motion Lab - fuzz the Kenshi .skeleton exporter. Author: PZA. GPLv3.
#   python tests/fuzz/fuzz_kenshi_export.py [cases]
# 1. skeleton_prefix on mutated real-layout files: returns a prefix or raises ExportError, nothing else, never hangs.
# 2. random poses through write_skeleton -> skeleton_read -> sampler: Ogre's playback rebuilds the pose (< 0.1 mm).
# 3. planted bugs (each must be caught by step 2): key rotation without the bind inverse, translation not in decimetres,
#    quaternion written w-first.
import os, sys, random, struct, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import numpy as np
from motionlab import kenshi_export as ke, mathx as mx

N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
rng = random.Random(1919); nrng = np.random.default_rng(1919)
fails = []


def rand_q():
    return mx.norm(nrng.normal(size=4))


def skeleton(nb):
    names = ['Bip01'] + ['b%d' % i for i in range(1, nb)]
    par = [-1] + [rng.randrange(0, i) for i in range(1, nb)]
    t = [nrng.normal(size=3) * 0.3 for _ in range(nb)]; q = [rand_q() for _ in range(nb)]
    out = bytearray(struct.pack('<H', 0x1000) + b'[Serializer_v1.80]\n' + struct.pack('<HIH', 0x1010, 8, 0))
    for i in range(nb):
        out += struct.pack('<HI', 0x2000, 36) + names[i].encode() + b'\n' + struct.pack('<H3f4f', i, *(t[i] * 10), *q[i])
    for i in range(1, nb): out += struct.pack('<HIHH', 0x3000, 10, i, par[i])
    return bytes(out), names, par, t, q


def world(names, par, Lt, Lq):
    W = [None] * len(names)
    for i in range(len(names)):
        L = mx.trs(Lt[i], Lq[i], np.ones(3)); W[i] = W[par[i]] @ L if par[i] >= 0 else L
    return np.array(W)


def roundtrip(bug=None):
    """Worst joint error (m) of one random clip through the file, with an optional planted bug in the key maths."""
    nb = rng.randrange(2, 12); data, names, par, bt, bq = skeleton(nb)
    n = rng.randrange(2, 8); ts = np.linspace(0, rng.uniform(0.1, 3.0), n)
    Lt = [[bt[i] + nrng.normal(size=3) * 0.2 for i in range(nb)] for _ in range(n)]
    Lq = [[rand_q() for i in range(nb)] for _ in range(n)]
    tracks = {}
    for i in range(nb):
        keys = []
        for k in range(n):
            kq = mx.norm(mx.q_mul(mx.q_conj(bq[i]), Lq[k][i])) if bug != 'no-bind-inverse' else Lq[k][i]
            kt = (Lt[k][i] - bt[i]) * (ke.UNITS if bug != 'metres' else 1.0)
            if bug == 'w-first': kq = np.array([kq[3], kq[0], kq[1], kq[2]])
            keys.append((float(ts[k]), tuple(kq), tuple(kt)))
        tracks[i] = keys
    p = os.path.join(tempfile.gettempdir(), 'pza_fuzz_kenshi_export.skeleton')
    ke.write_skeleton(data, [('clip', float(ts[-1]), tracks)], p)
    nm, pr, smp, L = ke.sampler(p, 'clip')
    worst = 0.0
    for k in range(n):
        worst = max(worst, float(np.abs(smp(float(ts[k]))[:, :3, 3] - world(names, par, Lt[k], Lq[k])[:, :3, 3]).max()))
    return worst


# 1. mutated files
for c in range(N):
    data = bytearray(skeleton(rng.randrange(1, 8))[0])
    for _ in range(rng.randrange(1, 6)):
        op = rng.random()
        if op < 0.4 and data: data[rng.randrange(len(data))] = rng.randrange(256)
        elif op < 0.7: data = data[:rng.randrange(len(data) + 1)]
        else: data[rng.randrange(len(data) + 1):0] = bytes(rng.randrange(256) for _ in range(rng.randrange(1, 9)))
    try:
        pre = ke.skeleton_prefix(bytes(data))
        if not bytes(data).startswith(pre): fails.append('prefix is not a prefix (case %d)' % c)
    except ke.ExportError: pass
    except Exception as e: fails.append('case %d: %s: %s' % (c, type(e).__name__, e))
# 2. round trips
worst = max(roundtrip() for _ in range(max(20, N // 10)))
if worst > 1e-4: fails.append('round trip error %.6f m' % worst)
# 3. planted bugs
for bug in ('no-bind-inverse', 'metres', 'w-first'):
    caught = max(roundtrip(bug) for _ in range(5)) > 1e-3
    print(('caught ' if caught else 'MISSED ') + bug)
    if not caught: fails.append('planted bug not caught: ' + bug)
print('fuzz_kenshi_export: %d mutated files, round trip worst %.7f m' % (N, worst))
for f in fails[:20]: print('FAIL', f)
sys.exit(1 if fails else 0)
