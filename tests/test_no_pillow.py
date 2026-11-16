# PZA's Kenshi Motion Lab - the rig pipeline must run without Pillow (Blender ships numpy, not Pillow).
# Author: PZA. GPLv3.   python tests/test_no_pillow.py
import builtins, os, sys, tempfile
_real = builtins.__import__
def _no_pil(name, *a, **k):
    if name == 'PIL' or name.startswith('PIL.'): raise ImportError('Pillow blocked for this test')
    return _real(name, *a, **k)
builtins.__import__ = _no_pil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..')); sys.path.insert(0, os.path.dirname(__file__))
from synthetic import make_hexapod
from motionlab import gltf, pipeline

d = tempfile.mkdtemp()
src = os.path.join(d, 'hexapod.glb')
gltf.write_glb(src, make_hexapod(15.0), skeleton=None)            # an unrigged model, like a Meshy export
r = pipeline.rig_file(src, d, log=lambda *a: None, clips='walk,hover', gifs=True)
ok = os.path.exists(r['glb']) and r['png'] is None and r['gifs'] == [] and r['clips'] == ['walk', 'hover']
print('PASS' if ok else 'FAIL', 'rig without Pillow: %s, bones %d, clips %s, png %s' % (os.path.basename(r['glb']), r['bones'], r['clips'], r['png']))
sys.exit(0 if ok else 1)
