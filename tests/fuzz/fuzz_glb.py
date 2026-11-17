# PZA's Kenshi Motion Lab - GLB reader fuzzer: mutated / truncated files must be refused cleanly (an exception),
# never hang, never allocate wildly, never crash the interpreter. Author: PZA. GPLv3.
#   python tests/fuzz/fuzz_glb.py [iterations]
import os, sys, random, time, tempfile, json, struct
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from motionlab import gltf, autorig, rigbuild, shape, motion, bake
from synthetic import make_hexapod

N = int(sys.argv[1]) if len(sys.argv) > 1 else 400
src = make_hexapod()
m, lm = autorig.analyse(src, log=lambda *a: None)
sk, rm = rigbuild.build_skeleton(lm, log=lambda *a: None)
J, Wk, _ = rigbuild.bind(m, sk, rm, lm, None, log=lambda *a: None)
m.skeleton, m.joints, m.weights = sk, J, Wk
drv = motion.Driver(motion.Rig(sk, rm))
clip = bake.bake(drv, [d for d in motion.standard_clips(drv) if d.name == 'walk'][0])
td = tempfile.mkdtemp()
base = os.path.join(td, 'base.glb'); gltf.write_glb(base, m, [clip])
good = open(base, 'rb').read()
gltf.load_glb(good); gltf.read_clips(good)
rng = random.Random(1234)
ok_rejects = (gltf.GlbError, ValueError, KeyError, IndexError, TypeError, struct.error, json.JSONDecodeError,
              UnicodeDecodeError, OverflowError, AttributeError, ZeroDivisionError)
stats = {'loaded': 0, 'rejected': 0}; worst = 0.0
for it in range(N):
    d = bytearray(good)
    kind = rng.randrange(4)
    if kind == 0:
        for _ in range(rng.randint(1, 16)): d[rng.randrange(len(d))] = rng.randrange(256)
    elif kind == 1:
        d = d[:rng.randrange(len(d))]
    elif kind == 2:                                    # mutate the JSON chunk's numbers / text
        jl = struct.unpack_from('<I', d, 12)[0]; p = 20 + rng.randrange(jl)
        d[p] = rng.choice(b'0123456789-+.e,:[]{}"')
    else:                                              # header / chunk length fields
        off = rng.choice([8, 12, 16, 20 + struct.unpack_from('<I', d, 12)[0]])
        if off + 4 <= len(d): struct.pack_into('<I', d, off, rng.choice([0, 1, 0x7fffffff, 0xffffffff, rng.randrange(1 << 32)]))
    t = time.time()
    try:
        gltf.load_glb(bytes(d)); gltf.read_clips(bytes(d)); stats['loaded'] += 1
    except ok_rejects:
        stats['rejected'] += 1
    except MemoryError:
        print('FAIL: MemoryError on iteration', it); sys.exit(1)
    worst = max(worst, time.time() - t)
    if time.time() - t > 10: print('FAIL: iteration %d took %.1f s' % (it, time.time() - t)); sys.exit(1)
print('fuzz_glb: %d iterations, %d loaded, %d refused, slowest %.2f s: PASS' % (N, stats['loaded'], stats['rejected'], worst))

# crafted cases from the pre-release audit: each must be refused as GlbError (or read correctly), never another crash
def rebuild(js, binc):
    jb = json.dumps(js).encode(); jb += b' ' * ((4 - len(jb) % 4) % 4)
    total = 12 + 8 + len(jb) + 8 + len(binc)
    return struct.pack('<4sII', b'glTF', 2, total) + struct.pack('<II', len(jb), 0x4E4F534A) + jb + struct.pack('<II', len(binc), 0x004E4942) + binc
js0, bin0 = gltf.read_glb_raw(good)
def case(name, edit, expect_ok=False):
    js = json.loads(json.dumps(js0)); edit(js)
    try:
        gltf.load_glb(rebuild(js, bin0)); res = 'loaded'
    except gltf.GlbError as ex: res = 'refused (%s)' % ex
    except Exception as ex: print('FAIL crafted %s: %s %s' % (name, type(ex).__name__, ex)); sys.exit(1)
    if expect_ok and res != 'loaded': print('FAIL crafted %s: %s' % (name, res)); sys.exit(1)
    print('  crafted %-28s %s' % (name, res))
prim = lambda js: js['meshes'][0]['primitives'][0]
case('negative accessor index', lambda js: prim(js)['attributes'].__setitem__('POSITION', -1))
case('JOINTS without WEIGHTS', lambda js: prim(js)['attributes'].pop('WEIGHTS_0'))
case('skin with no joint data', lambda js: [prim(js)['attributes'].pop(k) for k in ('JOINTS_0', 'WEIGHTS_0')], expect_ok=True)
case('negative joint node', lambda js: js['skins'][0]['joints'].__setitem__(0, -1))
case('stride smaller than element', lambda js: js['bufferViews'][js['accessors'][prim(js)['attributes']['POSITION']]['bufferView']].__setitem__('byteStride', 4))
case('zero-scale joint', lambda js: js['nodes'][js['skins'][0]['joints'][0]].__setitem__('scale', [0, 0, 0]))
case('bool as index', lambda js: prim(js).__setitem__('indices', True))
print('crafted cases: PASS')
