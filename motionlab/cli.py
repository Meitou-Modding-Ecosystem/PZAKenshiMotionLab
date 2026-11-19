# PZA's Kenshi Motion Lab - command line. Author: PZA. GPLv3.
#   python -m motionlab rig   <model.glb | model.mesh> -o <out folder> [--gifs] [--landmarks edited.json]
#   python -m motionlab view  <rigged.glb> [--clip walk] -o <file.gif>
#   python -m motionlab info  <model.glb | model.mesh>
#   python -m motionlab studio <model.glb | model.mesh> [--against <Kenshi .skeleton>] -o <out folder>
import argparse, json, os, sys, time
import numpy as np


def _load(path, skeleton=None):
    from .pipeline import _load as L
    return L(path, skeleton)


def cmd_rig(a):
    from .pipeline import rig_file
    kw = {k: getattr(a, k) for k in ('skeleton', 'name', 'plan', 'landmarks', 'leg_bones', 'wing_bones', 'antenna_bones',
                                     'abdomen_bones', 'ground_wings', 'clips', 'fps', 'gifs', 'own_rig', 'references')}
    rig_file(a.model, a.out, **kw)


def cmd_view(a):
    from . import gltf, bake
    m = gltf.load_glb(a.model)
    if m.skeleton is None: sys.exit('no skin in %s' % a.model)
    rc = gltf.read_clips(a.model)
    if not rc: sys.exit('no animations in %s' % a.model)
    name = a.clip or sorted(rc)[0]
    if name not in rc: sys.exit('no clip %r (has: %s)' % (name, ', '.join(sorted(rc))))
    length = max(ch['times'][-1] for ch in rc[name].values())
    bake.gif_from_clip(m, m.skeleton, rc[name], length, a.out, fps=a.fps, label=name, bones=a.bones, yaw=a.yaw, pitch=a.pitch)
    print('wrote', a.out)


def cmd_info(a):
    m = _load(a.model, a.skeleton)
    print('%s: %d vertices, %d triangles' % (a.model, len(m.verts), len(m.tris)))
    lo, hi = m.bounds(); print('bounds (m): %s .. %s' % (np.round(lo, 3), np.round(hi, 3)))
    if m.skeleton is not None:
        nz = (m.weights > 1e-6).sum(1)
        print('skeleton: %d bones; weights per vertex: max %d (Kenshi reads 3)' % (len(m.skeleton), nz.max()))
        print('bones:', ', '.join(m.skeleton.names))
    if a.model.lower().endswith('.glb'):
        from .gltf import read_clips
        rc = read_clips(a.model); print('clips:', ', '.join(rc) or 'none')


def cmd_studio(a):
    from . import studio
    m = _load(a.model, a.skeleton)
    if m.skeleton is None: sys.exit('no skeleton in %s (give --skeleton)' % a.model)
    studio.build(m, a.out, against=a.against, clips=tuple(c.strip() for c in a.clips.split(',') if c.strip()), gifs=not a.no_gifs)


def main(argv=None):
    ap = argparse.ArgumentParser(prog='motionlab', description="PZA's Kenshi Motion Lab: rig and animate models from code")
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('rig', help='auto-rig a model, bake clips, write a rigged .glb + report (+ GIFs)')
    r.add_argument('model'); r.add_argument('-o', '--out', required=True)
    r.add_argument('--skeleton', help='.skeleton for a .mesh (default: the one it links, if next to it)')
    r.add_argument('--name'); r.add_argument('--plan', default='insect', choices=['insect'])
    r.add_argument('--landmarks', help='edited <name>.landmarks.json from an earlier run')
    r.add_argument('--leg-bones', type=int, default=4); r.add_argument('--wing-bones', type=int, default=2)
    r.add_argument('--antenna-bones', type=int, default=2); r.add_argument('--abdomen-bones', type=int, default=3)
    r.add_argument('--ground-wings', default='hidden', choices=['hidden', 'folded'])
    r.add_argument('--clips', help='comma list (default all: idle,walk,run,hover,takeoff,land,attack,sting)')
    r.add_argument('--fps', type=int, default=30); r.add_argument('--gifs', action='store_true')
    r.add_argument('--own-rig', dest='own_rig', default='auto', choices=['auto', 'yes', 'no'],
                   help='keep the model own skeleton when it is one we know (Kenshi Bip01, Motion Lab names); default auto')
    r.add_argument('--reference', dest='references', action='append', metavar='CLIP.glb',
                   help='a skeleton-only reference clip (e.g. tools/fbx_to_glb.py output): retargeted onto a person '
                        'rig as clip ref_<file name>; repeat for more. Use only files you may use; outputs stay yours to keep')
    v = sub.add_parser('view', help='render a clip of a rigged .glb to a GIF')
    v.add_argument('model'); v.add_argument('-o', '--out', required=True); v.add_argument('--clip')
    v.add_argument('--fps', type=int, default=30); v.add_argument('--bones', action='store_true')
    v.add_argument('--yaw', type=float, default=38.0, help='camera angle around the model (90 = side)'); v.add_argument('--pitch', type=float, default=14.0)
    i = sub.add_parser('info', help='what a model holds (bones, weights, clips)')
    i.add_argument('model'); i.add_argument('--skeleton')
    st = sub.add_parser('studio', help='a local page: our walk / run next to a reference, curves and a scorecard')
    st.add_argument('model'); st.add_argument('-o', '--out', required=True); st.add_argument('--skeleton')
    st.add_argument('--against', help="a Kenshi .skeleton with this body's walk / run clips (pictures and curves); "
                                      "without it the scorecard uses the numbers measured from Kenshi's clips")
    st.add_argument('--clips', default='walk,run'); st.add_argument('--no-gifs', action='store_true')
    a = ap.parse_args(argv)
    {'rig': cmd_rig, 'view': cmd_view, 'info': cmd_info, 'studio': cmd_studio}[a.cmd](a)


if __name__ == '__main__':
    main()
