# PZA's Kenshi Motion Lab - reference-clip fuzzer: mutated / truncated reference .glb files given to the curve
# extraction (motionlab/curves.py, motionlab/reference.py) must be refused cleanly (an exception) or read, never
# hang, never allocate wildly, never crash. Author: PZA. GPLv3.
#   python tests/fuzz/fuzz_reference.py [iterations]
import copy, os, sys, random, time, tempfile, json, struct
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from motionlab import gltf, existing, motion, bake, curves
from synthetic import make_biped

N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
m = make_biped()
drv = motion.Driver(motion.Rig(m.skeleton, existing.rigmap_from_skeleton(m)))
cd = [c for c in motion.standard_clips(drv) if c.name == 'getup_back'][0]
clip = bake.bake(drv, cd, fps=15)
src = open(os.path.join(os.path.dirname(__file__), '..', 'test_curves.py'), encoding='utf-8').read()
ns = {}; exec(src[src.index('TO_MIXAMO = '):src.index('m = make_biped()')], {}, ns)
ref = copy.deepcopy(m); ref.skeleton = copy.deepcopy(m.skeleton)
ref.skeleton.names = ['mixamorig:' + ns['TO_MIXAMO'].get(n, n.replace(' ', '_')) for n in m.skeleton.names]
td = tempfile.mkdtemp(); base = os.path.join(td, 'ref.glb'); gltf.write_glb(base, ref, [clip])
good = open(base, 'rb').read()
cur = curves.extract(base, fps=15)                      # the unmutated file reads
assert all(np.isfinite(v).all() for k, v in cur.items() if isinstance(v, np.ndarray))
# crafted: a clip just over the 600 s limit, and one with a broken time, must be refused (not read and sampled)
for label, scale in (('700 s clip (limit 600)', 700.0 / max(clip.times[-1], 1e-9)), ('NaN time', float('nan'))):
    bad = copy.copy(clip); bad.times = np.array(clip.times, float) * scale
    bp = os.path.join(td, 'crafted.glb'); gltf.write_glb(bp, ref, [bad])
    try:
        curves.extract(bp, fps=15); print('FAIL: crafted %s was read' % label); sys.exit(1)
    except (ValueError, gltf.GlbError, KeyError, IndexError, TypeError, OverflowError):
        pass
print('crafted cases: PASS')
rng = random.Random(4321)
ok_rejects = (gltf.GlbError, ValueError, KeyError, IndexError, TypeError, struct.error, json.JSONDecodeError,
              UnicodeDecodeError, OverflowError, AttributeError, ZeroDivisionError)
stats = {'read': 0, 'refused': 0}; worst = 0.0
path = os.path.join(td, 'm.glb')
for it in range(N):
    d = bytearray(good)
    kind = rng.randrange(3)
    if kind == 0:
        for _ in range(rng.randint(1, 16)): d[rng.randrange(len(d))] = rng.randrange(256)
    elif kind == 1:
        d = d[:rng.randrange(len(d))]
    else:
        jl = struct.unpack_from('<I', d, 12)[0]; p = 20 + rng.randrange(jl)
        d[p] = rng.choice(b'0123456789-+.e,:[]{}"')
    with open(path, 'wb') as f: f.write(d)
    t = time.time()
    try:
        c = curves.extract(path, fps=15)
        assert c['length'] <= 600 and all(np.isfinite(v).all() for v in c.values() if isinstance(v, np.ndarray)),             'a read that worked must stay sane (iteration %d)' % it
        stats['read'] += 1
    except ok_rejects:
        stats['refused'] += 1
    except (MemoryError, RecursionError) as ex:
        print('FAIL: %s on iteration %d' % (type(ex).__name__, it)); sys.exit(1)
    worst = max(worst, time.time() - t)
    if time.time() - t > 20: print('FAIL: iteration %d took %.1f s' % (it, time.time() - t)); sys.exit(1)
print('fuzz_reference: %d iterations, %d read, %d refused, slowest %.2f s: PASS' % (N, stats['read'], stats['refused'], worst))
