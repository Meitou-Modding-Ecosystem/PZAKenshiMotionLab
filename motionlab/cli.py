# PZA's Kenshi Motion Lab - command line. Author: PZA. GPLv3.
#   python -m motionlab rig      <model.glb | model.mesh> -o <out folder> [--plan insect|quadruped|biped|crab|bird|auto] [--budget creature] [--gifs]
#   python -m motionlab export   <rigged.glb> -o <out folder> [--name stem] [--clips a,b] [--max-texture 2048]
#   python -m motionlab decimate <model.glb> -o <out.glb> [--budget creature|large|gear|weapon | --target-tris N]
#   python -m motionlab sequence <rigged.glb> --steps "walk:2,run:1.5:0.3" -o <out.glb> [--gif out.gif]
#   python -m motionlab view     <rigged.glb> [--clip walk] -o <file.gif>
#   python -m motionlab info     <model.glb | model.mesh>
#   python -m motionlab studio   <model.glb | model.mesh> [--against <Kenshi .skeleton>] -o <out folder>
import argparse, json, os, sys
import numpy as np


def _load(path, skeleton=None):
    from .pipeline import _load as L
    return L(path, skeleton)


def cmd_rig(a):
    from .pipeline import rig_file
    kw = {k: getattr(a, k) for k in ('skeleton', 'name', 'plan', 'landmarks', 'leg_bones', 'wing_bones', 'antenna_bones',
                                     'abdomen_bones', 'arm_bones', 'ground_wings', 'clips', 'fps', 'gifs', 'own_rig',
                                     'references', 'budget', 'target_tris')}
    rig_file(a.model, a.out, **kw)


def cmd_export(a):
    from . import asset_export
    clips = [c.strip() for c in a.clips.split(',') if c.strip()] if a.clips else None
    asset_export.export_asset(a.model, a.out, name=a.name, clips=clips, prefix=a.prefix,
                              fps=a.fps, travel=a.travel, max_texture=a.max_texture)


def cmd_decimate(a):
    from . import decimate, gltf
    m = _load(a.model, a.skeleton)
    target = decimate.resolve_target(a.budget, a.target_tris)
    m2 = decimate.decimate_model(m, target)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    sz = gltf.write_glb(a.out, m2)
    print('wrote %s (%.1f KB, %d tris)' % (a.out, sz / 1024.0, len(m2.tris)))


def cmd_sequence(a):
    from . import bake, existing, gltf, motion, sequence
    m = _load(a.model, a.skeleton)
    if m.skeleton is None:
        sys.exit('no skeleton in %s (rig it first or pass --skeleton)' % a.model)
    rm = existing.rigmap_from_skeleton(m)
    if rm is None:
        sys.exit('unrecognised skeleton in %s' % a.model)
    drv = motion.Driver(motion.Rig(m.skeleton, rm), ground_wings=a.ground_wings)
    clip, meta = sequence.build_sequence(drv, a.steps, fps=a.fps, name=a.name, travel=not a.in_place)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    sz = gltf.write_glb(a.out, m, [clip], tails=rm.get('tails'))
    print('wrote %s (%.1f KB, %d frames, %.2f s, %d joins)' % (
        a.out, sz / 1024.0, meta['frames'], meta['length_s'], len(meta['joins'])))
    if a.gif:
        rc = gltf.read_clips(a.out)
        back = gltf.load_glb(a.out)
        bake.gif_from_clip(back, back.skeleton, rc[clip.name], float(clip.times[-1]), a.gif, fps=a.fps, label=clip.name)
        print('wrote', a.gif)


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
    r.add_argument('--name')
    r.add_argument('--plan', default='insect', choices=['insect', 'quadruped', 'biped', 'crab', 'bird', 'auto'])
    r.add_argument('--landmarks', help='edited <name>.landmarks.json from an earlier run')
    r.add_argument('--leg-bones', type=int, default=4); r.add_argument('--wing-bones', type=int, default=2)
    r.add_argument('--antenna-bones', type=int, default=2); r.add_argument('--abdomen-bones', type=int, default=3)
    r.add_argument('--arm-bones', type=int, default=2)
    r.add_argument('--ground-wings', default='hidden', choices=['hidden', 'folded'])
    r.add_argument('--clips', help='comma list (default all fitting clips)')
    r.add_argument('--fps', type=int, default=30); r.add_argument('--gifs', action='store_true')
    r.add_argument('--own-rig', dest='own_rig', default='auto', choices=['auto', 'yes', 'no'],
                   help='keep the model own skeleton when it is one we know (Kenshi Bip01, Motion Lab names); default auto')
    r.add_argument('--budget', choices=['creature', 'large', 'gear', 'weapon'],
                   help='decimate before auto-rigging to a Kenshi polygon budget (docs/polygon-budgets.md)')
    r.add_argument('--target-tris', type=int, dest='target_tris', help='explicit triangle cap before auto-rigging')
    r.add_argument('--reference', dest='references', action='append', metavar='CLIP.glb',
                   help='a skeleton-only reference clip: retargeted onto a person rig as clip ref_<file name>')

    ex = sub.add_parser('export', help='export a rigged .glb directly to Kenshi .mesh + .skeleton + .dds')
    ex.add_argument('model'); ex.add_argument('-o', '--out', required=True)
    ex.add_argument('--name'); ex.add_argument('--clips'); ex.add_argument('--prefix', default='')
    ex.add_argument('--fps', type=float, default=30.0)
    ex.add_argument('--travel', choices=('keep', 'remove'), default='keep')
    ex.add_argument('--max-texture', type=int, default=2048, dest='max_texture')

    dc = sub.add_parser('decimate', help='reduce a .glb mesh to a Kenshi polygon budget')
    dc.add_argument('model'); dc.add_argument('-o', '--out', required=True)
    dc.add_argument('--skeleton')
    dc.add_argument('--budget', choices=['creature', 'large', 'gear', 'weapon'])
    dc.add_argument('--target-tris', type=int, dest='target_tris')

    sq = sub.add_parser('sequence', help='chain clips into one continuous clip with inertialized joins')
    sq.add_argument('model'); sq.add_argument('-o', '--out', required=True)
    sq.add_argument('--skeleton')
    sq.add_argument('--steps', required=True, help='comma list of clip[:duration_s[:blend_s]], e.g. walk:2,run:1.5:0.3,jump_run::0.1')
    sq.add_argument('--name', default='sequence'); sq.add_argument('--fps', type=int, default=30)
    sq.add_argument('--ground-wings', default='hidden', choices=['hidden', 'folded'])
    sq.add_argument('--in-place', action='store_true', dest='in_place')
    sq.add_argument('--gif', help='optional preview GIF path')

    v = sub.add_parser('view', help='render a clip of a rigged .glb to a GIF')
    v.add_argument('model'); v.add_argument('-o', '--out', required=True); v.add_argument('--clip')
    v.add_argument('--fps', type=int, default=30); v.add_argument('--bones', action='store_true')
    v.add_argument('--yaw', type=float, default=38.0, help='camera angle around the model (90 = side)'); v.add_argument('--pitch', type=float, default=14.0)

    i = sub.add_parser('info', help='what a model holds (bones, weights, clips)')
    i.add_argument('model'); i.add_argument('--skeleton')

    st = sub.add_parser('studio', help='a local page: our walk / run next to a reference, curves and a scorecard')
    st.add_argument('model'); st.add_argument('-o', '--out', required=True); st.add_argument('--skeleton')
    st.add_argument('--against', help="a Kenshi .skeleton with this body's walk / run clips")
    st.add_argument('--clips', default='walk,run'); st.add_argument('--no-gifs', action='store_true')

    a = ap.parse_args(argv)
    {'rig': cmd_rig, 'export': cmd_export, 'decimate': cmd_decimate, 'sequence': cmd_sequence,
     'view': cmd_view, 'info': cmd_info, 'studio': cmd_studio}[a.cmd](a)


if __name__ == '__main__':
    main()
