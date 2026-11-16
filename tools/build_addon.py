# PZA's Kenshi Motion Lab - build the Blender add-on (extension zip): the add-on folder + the motionlab package.
# Author: PZA. GPLv3.   python tools/build_addon.py   ->  out/addon/pza_motion_lab/ and out/addon/pza_motion_lab-<ver>.zip
import os, re, shutil, zipfile
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, 'blender_addon', 'pza_motion_lab')
OUT = os.path.join(ROOT, 'out', 'addon')
dst = os.path.join(OUT, 'pza_motion_lab')
shutil.rmtree(dst, ignore_errors=True)
shutil.copytree(SRC, dst, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
shutil.copytree(os.path.join(ROOT, 'motionlab'), os.path.join(dst, 'motionlab'), ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
shutil.copy(os.path.join(ROOT, 'LICENSE'), os.path.join(dst, 'LICENSE'))
ver = re.search(r'^version = "([^"]+)"', open(os.path.join(SRC, 'blender_manifest.toml'), encoding='utf-8').read(), re.M).group(1)
z = os.path.join(OUT, 'pza_motion_lab-%s.zip' % ver)
with zipfile.ZipFile(z, 'w', zipfile.ZIP_DEFLATED) as f:
    for base, _, files in os.walk(dst):
        for fn in files:
            p = os.path.join(base, fn); f.write(p, os.path.relpath(p, dst))
print('built', z, '(%d KB)' % (os.path.getsize(z) // 1024))
