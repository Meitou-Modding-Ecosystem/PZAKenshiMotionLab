# PZA's Kenshi Motion Lab - Studio: one local page that shows how natural our clips move next to a reference.
#   python -m motionlab studio <model.mesh | .glb> [--skeleton s] [--against <Kenshi .skeleton with clips>]
#                              [--clips walk,run] -o out/studio
# For each gait: our clip and the reference side by side (same body, same side camera, frame-locked GIF), our curves
# over the reference's (pelvis height / yaw / roll, chest counter-turn, foot roll, head height), and a scorecard graded
# green / amber / red (motionlab/metrics.py). Without --against the reference is the numbers measured from Kenshi's
# clips for this skeleton (motionlab/data/kenshi_gaits.json): scorecard only, no reference picture or curves.
# The page is written locally; a page made with Kenshi's own clips holds pictures of them: keep it to yourself.
# Author: PZA. GPLv3.
import base64, html, io, os
import numpy as np
from . import metrics, motion, existing, profiles

FPS = 30


def our_frames(drv, cd, fps=FPS):
    """World matrices of every bone over one loop (the 2nd pass, so springs have settled)."""
    n = max(2, int(round(cd.length * fps))); drv.reset(); out = []
    for k in range(2 * n):
        q, t, s, W = drv.pose(cd.fn((k % n) / fps * (cd.length * fps / n), cd.length), dt=1 / fps)
        if k >= n: out.append(np.array(W))
    return np.array(out)


def reference_frames(names, skeleton_path, kind, fps=FPS, n=None):
    """World matrices (in OUR bone order, by name) of a Kenshi clip of that skeleton, resampled to n frames."""
    m, anims = profiles.skeleton_model(skeleton_path)
    picked = profiles.pick(anims, kind)
    if not picked: return None
    smp, L = profiles.sampler(m.skeleton, anims, picked)
    n = n or max(2, int(round(L * fps)))
    ix = {nm: i for i, nm in enumerate(m.skeleton.names)}
    rest = m.skeleton.world_rest(); out = []
    for k in range(n):
        W = m.skeleton.world(smp(k / n * L))
        out.append(np.array([W[ix[nm]] if nm in ix else np.eye(4) for nm in names]))
    return np.array(out), np.array([rest[ix[nm]] if nm in ix else np.eye(4) for nm in names]), L


def _render(model, frames_list, labels, root, fps=FPS, size=(330, 300)):
    """A side-by-side GIF (bytes): each body in place (root's travel removed), side camera."""
    from PIL import Image, ImageDraw
    from . import render
    lo, hi = model.bounds(); h = float(hi[1] - lo[1])
    cam = render.Camera(np.array([0, 0.5 * h, 0]), 90.0, 8.0, 2.6 * h, size=size)
    R = render.Renderer(model, cam, ss=1)
    n = min(len(f) for f in frames_list); imgs = []
    for k in range(n):
        row = Image.new('RGB', (size[0] * len(frames_list), size[1]))
        for j, F in enumerate(frames_list):
            W = F[k]; P, N = model.skin(W)
            d = W[root][:3, 3].copy(); d[1] = 0.0; P = P - d
            im = R.frame(P); ImageDraw.Draw(im).text((8, 6), labels[j], fill=(40, 34, 28)); row.paste(im, (size[0] * j, 0))
        imgs.append(row)
    buf = io.BytesIO()
    imgs[0].save(buf, format='GIF', save_all=True, append_images=imgs[1:], duration=int(round(1000 / fps)), loop=0, disposal=2)
    return buf.getvalue()


def _svg(title, ours, ref):
    """A small line chart: reference (grey) behind ours (accent), x = one cycle."""
    W_, H_, pad = 300, 120, 18
    vals = np.concatenate([ours] + ([ref] if ref is not None else []))
    lo, hi = float(vals.min()), float(vals.max()); span = max(hi - lo, 1e-6)
    def path(v):
        x = pad + np.arange(len(v)) / max(len(v) - 1, 1) * (W_ - 2 * pad)
        y = H_ - pad - (np.asarray(v) - lo) / span * (H_ - 2 * pad)
        return ' '.join('%.1f,%.1f' % p for p in zip(x, y))
    s = ['<svg viewBox="0 0 %d %d" role="img" aria-label="%s">' % (W_, H_, html.escape(title))]
    s.append('<line x1="%d" y1="%d" x2="%d" y2="%d" class="axis"/>' % (pad, H_ - pad, W_ - pad, H_ - pad))
    if ref is not None: s.append('<polyline class="ref" points="%s"/>' % path(ref))
    s.append('<polyline class="ours" points="%s"/>' % path(ours))
    s.append('<text x="%d" y="12" class="lbl">%s</text>' % (pad, html.escape(title)))
    s.append('<text x="%d" y="%d" class="tick">%.0f</text><text x="%d" y="%d" class="tick">%.0f</text>' % (2, pad + 4, hi, 2, H_ - pad, lo))
    s.append('</svg>')
    return ''.join(s)


def _resample(v, n):
    v = np.asarray(v, float); x = np.linspace(0, 1, len(v), endpoint=False)
    return np.interp(np.linspace(0, 1, n, endpoint=False), x, v, period=1.0)


def build(model, out_dir, against=None, clips=('walk', 'run'), log=print, gifs=True):
    """Write <out_dir>/studio.html. Returns {clip: {metric: (ours, reference, grade)}}."""
    rm = existing.rigmap_from_skeleton(model)
    if rm is None: raise ValueError('the Studio reads bodies with a Kenshi-style (Bip01) skeleton')
    drv = motion.Driver(motion.Rig(model.skeleton, rm)); rig = drv.rig
    R = metrics.roles(rig.names, rm); root = rig.b(rm['root'])
    prof = rm.get('gait_profile') or {}
    by = {c.name: c for c in motion.standard_clips(drv)}
    result = {}; sections = []
    for name in clips:
        cd = by.get(name)
        if cd is None or not cd.loop: log('studio: no looping clip %r, skipped' % name); continue
        F = our_frames(drv, cd)
        ours, oc = metrics.cycle(F, rig.Wrest, R, FPS)
        ref, rc, RF = None, None, None
        if against:
            got = reference_frames(rig.names, against, name, n=len(F))
            if got is not None:
                RF, Rrest, L = got
                ref, rc = metrics.cycle(RF, Rrest, R, len(RF) / L)
        if ref is None and name in prof: ref = prof[name].get('scorecard')
        ours['ground speed (m/s)'] = float(cd.speed)                 # the clip's planted-foot speed vs the reference's
        if ref is not None and name in prof and 'speed_m_s' in prof[name]: ref = dict(ref, **{'ground speed (m/s)': float(prof[name]['speed_m_s'])})
        shift = 0
        if rc is not None:                                       # line the reference's cycle up with ours (by phase)
            key = 'foot pitch (deg, + toe down)' if 'foot pitch (deg, + toe down)' in oc else 'pelvis height (cm)'
            shift = metrics.align(oc[key], _resample(rc[key], len(oc[key])))
            if RF is not None and len(RF) == len(F): RF = np.roll(RF, shift, axis=0)
        rows = {}
        for k, v in ours.items():
            rv = (ref or {}).get(k); rows[k] = (v, rv, metrics.grade(k, v, rv))
        result[name] = rows
        log('studio %s: ' % name + ', '.join('%s %s' % (k.split(' (')[0], g) for k, (a, b, g) in rows.items()))
        # page section
        sec = ['<section><h2>%s</h2>' % html.escape(name)]
        if gifs:
            fl = [F] + ([RF] if RF is not None else []); lb = ['ours'] + (['reference'] if RF is not None else [])
            g = base64.b64encode(_render(model, fl, lb, root)).decode()
            sec.append('<img class="gif" alt="%s side by side" src="data:image/gif;base64,%s">' % (html.escape(name), g))
        sec.append('<table><thead><tr><th>Measure</th><th>Reference</th><th>Ours</th><th></th></tr></thead><tbody>')
        for k, (a, b, g) in rows.items():
            sec.append('<tr><td>%s</td><td>%s</td><td>%.1f</td><td><span class="dot %s" title="%s"></span></td></tr>'
                       % (html.escape(k), '-' if b is None else '%.1f' % b, a, g, g))
        sec.append('</tbody></table><div class="charts">')
        for k, v in oc.items():
            r_ = np.roll(_resample(rc[k], len(v)), shift) if (rc is not None and k in rc) else None
            sec.append(_svg(k, v, r_))
        sec.append('</div></section>')
        sections.append(''.join(sec))
    os.makedirs(out_dir, exist_ok=True)
    page = PAGE.replace('{{SECTIONS}}', '\n'.join(sections)).replace('{{REF}}', html.escape(
        'Kenshi clips from ' + os.path.basename(against) if against else
        "numbers measured from Kenshi's clips (" + str(rm.get('gait_profile_from', 'none')) + ')'))
    path = os.path.join(out_dir, 'studio.html')
    with open(path, 'w', encoding='utf-8') as f: f.write(page)
    log('wrote %s' % path)
    return result


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Motion Lab Studio</title><style>
:root{--bg:#f4efe6;--fg:#2b241d;--muted:#7a6e60;--card:#fffaf2;--line:#d8cdbd;--ours:#c0562b;--ref:#a59886;--green:#3f8f4f;--amber:#d39b27;--red:#c23b32}
@media (prefers-color-scheme:dark){:root{--bg:#1d1915;--fg:#eee5d8;--muted:#a89a88;--card:#28221c;--line:#43392e;--ours:#ef7a48;--ref:#7d7163}}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,sans-serif}
main{max-width:1000px;margin:0 auto;padding:16px}h1{font-size:22px;margin:8px 0}h2{font-size:18px;margin:0 0 10px;text-transform:capitalize}
p.sub{color:var(--muted);margin:0 0 16px}section{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;margin:0 0 18px}
img.gif{max-width:100%;border-radius:6px;display:block;margin:0 0 12px}table{border-collapse:collapse;width:100%;margin:0 0 12px}
td,th{padding:5px 8px;border-bottom:1px solid var(--line);text-align:left}td:nth-child(2),td:nth-child(3){font-variant-numeric:tabular-nums}
th{color:var(--muted);font-weight:600}.dot{display:inline-block;width:12px;height:12px;border-radius:50%}
.dot.green{background:var(--green)}.dot.amber{background:var(--amber)}.dot.red{background:var(--red)}.dot.none{background:var(--line)}
.charts{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:10px}
svg{width:100%;height:auto;background:var(--bg);border-radius:6px}.axis{stroke:var(--line)}
polyline{fill:none;stroke-width:2}.ours{stroke:var(--ours)}.ref{stroke:var(--ref);stroke-dasharray:4 3}
.lbl{fill:var(--fg);font-size:11px}.tick{fill:var(--muted);font-size:9px}.key span{display:inline-block;width:18px;height:3px;vertical-align:middle;margin:0 4px 0 10px}
</style></head><body><main><h1>Motion Lab Studio</h1>
<p class="sub">Reference: {{REF}}. <span class="key"><span style="background:var(--ours)"></span>ours<span style="background:var(--ref)"></span>reference</span>.
Green: within 25% of the reference (smoothness: at most 1.5x); amber: within 50%; red: further.</p>
{{SECTIONS}}
</main></body></html>"""
