# PZA's Kenshi Motion Lab - the rig pipeline as a library (the command line and the Blender add-on call it).
# Author: PZA. GPLv3.
import json, os, time
import numpy as np


def _load(path, skeleton=None):
    if path.lower().endswith('.mesh'):
        from .ogre.load import load_mesh
        return load_mesh(path, skeleton)
    from .gltf import load_glb
    return load_glb(path)


def _landmark_png(model, lm, path):
    from PIL import Image, ImageDraw
    from . import render
    cols = {'legs': (220, 30, 30), 'wings': (30, 90, 230), 'antennae': (20, 160, 40)}
    def ov(img, z, pr):
        pil = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)); dr = ImageDraw.Draw(pil)
        for k, c in cols.items():
            for e in lm[k]:
                uv, _ = pr(np.array(e['points'])); dr.line([tuple(p) for p in uv], fill=c, width=5)
                dr.text(tuple(uv[-1] + 6), e['name'], fill=(0, 0, 0))
        uv, _ = pr(np.array(lm['spine'])); dr.line([tuple(p) for p in uv], fill=(200, 0, 200), width=5)
        img[:] = np.asarray(pil, np.float32) / 255
    lo, hi = model.bounds(); c = np.array([0, 0.45 * (hi[1] - lo[1]), 0]); d = 1.6 * float(np.linalg.norm(hi - lo))
    ims = [render.Renderer(model, render.Camera(c, yaw, p, d, size=(560, 420))).frame(model.verts, overlay=ov)
           for yaw, p in ((35, 20), (90, 2), (180, 89.9), (0, 8))]
    big = Image.new('RGB', (1120, 840))
    for i, im in enumerate(ims): big.paste(im, ((i % 2) * 560, (i // 2) * 420))
    big.save(path)


def _apply_edits(lm, edited, log):
    """Take limb points / names from an edited landmarks file; keep the voxel analysis for binding."""
    by = {e['name']: e for k in ('legs', 'wings', 'antennae') for e in lm[k]}
    G = lm['_grid']; labels = lm['_labels']
    for k in ('legs', 'wings', 'antennae'):
        if k not in edited: continue
        out = []
        for e in edited[k]:
            ne = dict(e)
            if e['name'] in by: ne['_part'] = by[e['name']]['_part']
            else:
                mid = np.array(e['points'])[len(e['points']) // 2]
                ijk = G.index(mid); ne['_part'] = int(labels[tuple(ijk)])
            ne.setdefault('length', float(np.sum(np.linalg.norm(np.diff(np.array(e['points']), axis=0), axis=1))))
            out.append(ne)
        lm[k] = out
    for k in ('spine', 'neck_z', 'waist_z', 'body_z'):
        if k in edited: lm[k] = edited[k]
    log('landmarks: edits applied (%d legs, %d wings, %d antennae)' % (len(lm['legs']), len(lm['wings']), len(lm['antennae'])))


class Options:
    """Settings of one rig run (the command line and the Blender add-on fill these)."""
    def __init__(self, **kw):
        self.skeleton = None; self.name = None; self.plan = 'insect'; self.landmarks = None
        self.leg_bones = 4; self.wing_bones = 2; self.antenna_bones = 2; self.abdomen_bones = 3
        self.ground_wings = 'hidden'; self.clips = None; self.fps = 30; self.gifs = False; self.png = True
        self.references = None     # reference clip .glb files: each becomes a clip retargeted from its curves (people)
        self.own_rig = 'auto'      # 'auto': keep a skeleton we recognise (Kenshi Bip01, Motion Lab names); 'yes' / 'no'
        for k, v in kw.items(): setattr(self, k, v)


def have_pillow():
    try:
        import PIL  # noqa: F401
        return True
    except ImportError:
        return False


def _rig_lm(model, rigmap, drv):
    """Limb lines of an own rig, in the landmarks shape the picture code draws."""
    r = drv.rig; W = r.Wrest
    def pts(bones): return r.chain_points([r.b(b) for b in bones], W).tolist()
    sp = [W[r.b(rigmap['body'])][:3, 3].tolist(), W[r.b(rigmap['head'])][:3, 3].tolist()]
    return {'legs': [{'name': e['name'], 'points': pts(e['bones'])} for e in rigmap['legs']] +
                    [{'name': e['name'], 'points': pts(e['bones'])} for e in rigmap.get('arms', [])],
            'wings': [{'name': e['name'], 'points': pts(e['bones'])} for e in rigmap['wings']],
            'antennae': [{'name': e['name'], 'points': pts(e['bones'])} for e in rigmap['antennae']],
            'spine': sp + (pts(rigmap['spine']) if rigmap['spine'] else [])}


def _own_report(model, sk, rigmap, st, source):
    from .rigbuild import KENSHI_MAX_BONES_PER_SUBMESH as MB, KENSHI_UNITS_PER_METRE as U
    lo, hi = model.bounds(); H = float(hi[1] - lo[1])
    L = ["# PZA's Kenshi Motion Lab: report for '%s' (its own rig)" % model.name, '', 'Source: %s' % source, '',
         '| Check | Value | Kenshi | OK |', '|---|---|---|---|',
         '| Rig | %s bones, %s body (kept as it is) | - | - |' % (rigmap.get('source_rig', '?'), rigmap.get('plan', '?')),
         '| Bones in the skeleton | %d | %d per submesh | %s |' % (len(sk), MB, 'yes' if len(sk) <= MB else 'check per submesh'),
         '| Weights per vertex | up to %d | 3 | %s |' % (st['weights_per_vertex_max'], 'yes' if st['weights_per_vertex_max'] <= 3 else 'NO'),
         '| Vertices without weights | %d | 0 | %s |' % (st['unweighted'], 'yes' if st['unweighted'] == 0 else 'check'),
         '| Height | %.2f m | x%d in Kenshi units: %.1f | - |' % (H, U, H * U),
         '| Gait numbers | %s | - | - |' % ("measured from Kenshi's own walk / run clips: " + rigmap['gait_profile_from']
                                           if rigmap.get('gait_profile_from') else 'built-in defaults'),
         '', '## Bones the motion drives',
         '| Role | Bones |', '|---|---|']
    for e in rigmap['legs']: L.append('| %s | %s (foot kept level: %s) |' % (e['name'], ', '.join(e['bones']), e.get('foot', '-')))
    for e in rigmap.get('arms', []): L.append('| %s | %s (swing) |' % (e['name'], ', '.join(e['bones'])))
    if rigmap['spine']: L.append('| tail (springs) | %s |' % ', '.join(rigmap['spine']))
    for e in rigmap['antennae']: L.append('| %s (springs%s) | %s |' % (e['name'], ', stiff' if e.get('stiff') else '', ', '.join(e['bones'])))
    if rigmap.get('jaw'): L.append('| jaw (opens in the bite) | %s |' % rigmap['jaw'])
    L.append('| body / head / root | %s / %s / %s |' % (rigmap['body'], rigmap['head'], rigmap['root']))
    return chr(10).join(L) + chr(10)


def rig_file(model_path, out_dir, log=print, **kw):
    """Auto-rig a model file, bake the clips, write <name>_rigged.glb + report + landmarks + motion map (+ PNG, GIFs
    when Pillow is there). Returns a dict of the written paths and the run's numbers."""
    a = Options(**kw); a.model = model_path; a.out = out_dir
    from . import autorig, rigbuild, shape, motion, bake, gltf, check
    t0 = time.time()
    os.makedirs(a.out, exist_ok=True)
    src = _load(a.model, a.skeleton)
    from . import existing
    own = existing.rigmap_from_skeleton(src) if (a.own_rig != 'no' and not a.landmarks) else None
    if a.own_rig == 'yes' and own is None:
        raise ValueError('the model has no skeleton the Motion Lab recognises (Kenshi Bip01 or Motion Lab names)')
    if src.skeleton is None: what = ''
    elif own: what = ', own skeleton %d bones (kept: %s, %s)' % (len(src.skeleton), own['source_rig'], own['plan'])
    else: what = ', own skeleton %d bones (not one we recognise: replaced)' % len(src.skeleton)
    log('loaded %s: %d vertices, %d triangles%s' % (os.path.basename(a.model), len(src.verts), len(src.tris), what))
    name = a.name or os.path.splitext(os.path.basename(a.model))[0].replace(' ', '_')
    if own:
        # the model's own rig: skeleton and skin stay exactly as they are, only motion is added
        model, rigmap, lm = src, own, None
        sk, J, Wk = src.skeleton, src.joints, src.weights
        nz = (Wk > 1e-6).sum(1)
        st = {'vertices': len(src.verts), 'max_weight_dropped_by_prune': 0.0, 'unweighted': int((nz == 0).sum()),
              'weights_per_vertex_max': int(nz.max())}
        log('own rig: %d legs, %d arms, %d spring chains, tail %d bones, jaw %s' % (len(own['legs']), len(own['arms']),
            len(own['antennae']), len(own['spine']), 'yes' if own.get('jaw') else 'no'))
    else:
        model, lm = autorig.analyse(src, plan=a.plan, legs_bones=a.leg_bones, log=log)
        if a.landmarks:
            if isinstance(a.landmarks, dict): ed = a.landmarks
            else:
                with open(a.landmarks, encoding='utf-8') as f: ed = json.load(f)
            _apply_edits(lm, ed, log)
        sk, rigmap = rigbuild.build_skeleton(lm, leg_bones=a.leg_bones, wing_bones=a.wing_bones,
                                             antenna_bones=a.antenna_bones, abdomen_bones=a.abdomen_bones, log=log)
        J, Wk, st = rigbuild.bind(model, sk, rigmap, lm, shape.Surface(model), log=log)
        model.skeleton, model.joints, model.weights = sk, J, Wk
    model.name = name
    rig = motion.Rig(sk, rigmap); drv = motion.Driver(rig, ground_wings=a.ground_wings)
    for b, deg in rig.arms_lowered:
        log('rest pose: %s rested %.0f degrees from hanging (a T- or A-pose): the motion lowers it first' % (b, deg))
    defs = motion.standard_clips(drv, ground_wings=a.ground_wings)
    if a.clips:
        want = a.clips.split(',') if isinstance(a.clips, str) else list(a.clips)
        defs = [d for d in defs if d.name in want]
    for ref in (a.references or []):                       # retarget reference clips (local files you may use)
        from . import curves
        nm = 'ref_' + os.path.splitext(os.path.basename(ref))[0]
        defs.append(curves.curve_clip(drv, curves.extract(ref, fps=a.fps), nm))
        log('reference %s -> clip %s' % (os.path.basename(ref), nm))
    clips = []; checks = {}
    for d in defs:
        clips.append(bake.bake(drv, d, fps=a.fps))
        checks[d.name] = check.foot_check(drv, d, fps=a.fps)
    glb = os.path.join(a.out, name + '_rigged.glb')
    size = gltf.write_glb(glb, model, clips, tails=rigmap.get('tails'))
    log('wrote %s (%.1f MB, %d bones, %d clips)' % (glb, size / 1e6, len(sk), len(clips)))
    # read it back: the clips in the file are what the GIFs show
    rc = gltf.read_clips(glb)
    back = gltf.load_glb(glb)
    rest_err = float(np.abs(back.verts - model.verts).max())
    lmj = os.path.join(a.out, name + '.landmarks.json')
    if lm is not None:
        with open(lmj, 'w', encoding='utf-8') as f: json.dump(autorig.public(lm), f, indent=1)
    else: lmj = None
    draw = lm if lm is not None else _rig_lm(model, rigmap, drv)
    rmj = os.path.join(a.out, name + '.motion.json')
    with open(rmj, 'w', encoding='utf-8') as f:
        json.dump({'name': name, 'tool': "PZA's Kenshi Motion Lab", 'version': autorig.VERSION, **rigmap,
                   'clips': [{'name': c.name, 'length_s': float(c.times[-1]), 'loop': c.loop, 'ground_speed_m_s': c.speed,
                              'note': c.note} for c in clips], 'crouch_m': drv.crouch, 'ground_wings': a.ground_wings},
                  f, indent=1)
    png = os.path.join(a.out, name + '_landmarks.png')
    if a.png and have_pillow(): _landmark_png(model, draw, png)
    else: png = None
    if lm is not None: rep = rigbuild.report(model, sk, rigmap, J, Wk, st, lm, source_info=os.path.basename(a.model))
    else: rep = _own_report(model, sk, rigmap, st, os.path.basename(a.model))
    rep += '\n## Clips (baked at %d fps, in place: the game moves the creature)\n' % a.fps
    rep += '| Clip | Length | Loop | Ground speed | Worst foot IK miss | Worst planted-foot slip | Note |\n|---|---|---|---|---|---|---|\n'
    for c in clips:
        ck = checks[c.name]
        miss = max(ck['ik_miss_m'].values(), default=0.0); slip = max(ck['slip_m_per_s'].values(), default=0.0)
        rep += '| %s | %.2f s | %s | %.2f m/s (%.1f Kenshi units/s) | %.3f m | %.3f m/s | %s |\n' % (
            c.name, c.times[-1], 'yes' if c.loop else 'no', c.speed, c.speed * 10, miss, slip, c.note)
    for b, deg in rig.arms_lowered:
        rep += ('\nArms: %s rests %.0f degrees from hanging down (a T- or A-pose); Kenshi rests its arms hanging, so '
                'every clip lowers it first (the bind pose in the file is unchanged).\n' % (b, deg))
    rep += '\nWalking crouch: the body is lowered %.2f m while walking so the legs have room for a stride.\n' % drv.crouch
    rep += 'Round trip: the written .glb read back, rest pose max vertex difference %.2e m; %d clips found in the file.\n' % (rest_err, len(rc))
    if drv.wings: rep += ('\nWings on the ground: %s (wings shrink away on landing and grow back on take-off, 0.25 s, by '
            'scaling the wing root bones). Whether Kenshi plays bone SCALE keys is not verified yet; fallback: fold them.\n') % a.ground_wings
    rpt = os.path.join(a.out, name + '_report.md')
    with open(rpt, 'w', encoding='utf-8') as f: f.write(rep)
    gifs = []
    if a.gifs and have_pillow():
        for c in clips:
            p = os.path.join(a.out, '%s_%s.gif' % (name, c.name))
            bake.gif_from_clip(back, back.skeleton, rc[c.name], c.times[-1], p, fps=a.fps, speed=c.speed,
                               repeats=2 if c.loop and c.times[-1] < 1.5 else 1, label='%s - %s' % (name, c.name))
            log('gif %s' % p); gifs.append(p)
    log('done in %.1f s' % (time.time() - t0))
    return {'name': name, 'glb': glb, 'report': rpt, 'report_text': rep, 'landmarks': lmj, 'motion': rmj, 'png': png,
            'gifs': gifs, 'frame': (lm or {}).get('frame', {}), 'own_rig': lm is None, 'plan': rigmap.get('plan'), 'bones': len(sk), 'clips': [c.name for c in clips]}


