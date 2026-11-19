# PZA's Kenshi Motion Lab - a small software renderer (numpy + Pillow) for preview frames and GIFs. Author: PZA. GPLv3.
# Dense triangle splatting with a z-buffer, rendered at 2x and downsampled: no GPU, no OpenGL, same picture everywhere.
import numpy as np
from PIL import Image, ImageDraw
from . import mathx as mx

# barycentric sample pattern per triangle (vertices, edge midpoints, centre, thirds)
_B = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [.5, .5, 0], [0, .5, .5], [.5, 0, .5], [1 / 3, 1 / 3, 1 / 3],
               [2 / 3, 1 / 6, 1 / 6], [1 / 6, 2 / 3, 1 / 6], [1 / 6, 1 / 6, 2 / 3]])


class Camera:
    def __init__(self, target, yaw_deg=35.0, pitch_deg=18.0, dist=4.0, fov_deg=35.0, size=(480, 360)):
        self.target = np.asarray(target, float); self.yaw = yaw_deg; self.pitch = pitch_deg
        self.dist = dist; self.fov = fov_deg; self.size = size

    def basis(self):
        y, p = np.radians(self.yaw), np.radians(self.pitch)
        fwd = -np.array([np.sin(y) * np.cos(p), np.sin(p), np.cos(y) * np.cos(p)])  # camera looks along fwd
        eye = self.target - fwd * self.dist
        right = mx.norm(np.cross(fwd, [0, 1, 0])); up = np.cross(right, fwd)
        return eye, fwd, right, up

    def project(self, P, scale=1):
        eye, fwd, right, up = self.basis()
        d = P - eye
        z = d @ fwd; x = d @ right; y = d @ up
        w, h = self.size[0] * scale, self.size[1] * scale
        f = 0.5 * h / np.tan(np.radians(self.fov) / 2)
        z = np.maximum(z, 1e-3)
        return np.c_[w / 2 + f * x / z, h / 2 - f * y / z], z


def _sample_tex(tex, uv):
    img = tex.get('image'); fac = np.asarray(tex.get('factor', (1, 1, 1, 1))[:3], np.float32)
    if img is None: return np.broadcast_to(fac, (len(uv), 3))
    h, w = img.shape[:2]
    u = np.mod(uv[:, 0], 1.0); v = np.mod(uv[:, 1], 1.0)
    return img[np.clip((v * h).astype(int), 0, h - 1), np.clip((u * w).astype(int), 0, w - 1)] * fac


class Renderer:
    def __init__(self, model, cam, bg=(236, 230, 218), ss=2):
        self.m = model; self.cam = cam; self.bg = bg; self.ss = ss
        self._lat = {}

    def _lattice(self, n):
        if n not in self._lat:
            pts = [((n - i - j) / n, i / n, j / n) for i in range(n + 1) for j in range(n + 1 - i)]
            self._lat[n] = np.array(pts)
        return self._lat[n]

    def frame(self, P, N=None, ground_y=0.0, ground_offset=(0.0, 0.0), overlay=None, shadow=True, grid=True):
        cam = self.cam; s = self.ss; W, H = cam.size[0] * s, cam.size[1] * s
        img = np.empty((H, W, 3), np.float32); img[:] = np.asarray(self.bg, np.float32) / 255
        zbuf = np.full(H * W, np.inf)
        if grid: self._ground(img, zbuf, ground_y, ground_offset)
        T = self.m.tris
        tri = P[T]                                                           # (t, 3, 3)
        fn = mx.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]))
        if shadow: self._shadow(img, zbuf, P, ground_y)
        light = mx.norm(np.array([0.4, 0.9, 0.5]))
        eye = cam.basis()[0]
        view = mx.norm(eye - tri.mean(1))
        fn = np.where((np.einsum('ij,ij->i', fn, view) < 0)[:, None], -fn, fn)  # two-sided
        lam = 0.35 + 0.65 * np.clip(fn @ light, 0, 1)
        uv2, _ = cam.project(tri.reshape(-1, 3), s); uv2 = uv2.reshape(-1, 3, 2)
        L = np.max(np.linalg.norm(uv2 - np.roll(uv2, 1, axis=1), axis=2), axis=1)
        nsub = np.clip(np.ceil(L / 1.2).astype(int), 1, 24)
        T = self.m.tris
        for n in np.unique(nsub):
            sel = np.nonzero(nsub == n)[0]
            B = self._lattice(int(n))
            S = np.einsum('sk,tkc->tsc', B, tri[sel]).reshape(-1, 3)
            UV = np.einsum('sk,tkc->tsc', B, self.m.uvs[T[sel]]).reshape(-1, 2)
            col = _sample_tex(self.m.texture, UV) * np.repeat(lam[sel], len(B))[:, None]
            self._splat(img, zbuf, S, col, r=1)
        if overlay is not None: overlay(img, zbuf, lambda X: cam.project(X, s))
        im = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
        return im.resize(cam.size, Image.LANCZOS) if s > 1 else im

    def _splat(self, img, zbuf, X, col, r=1):
        H, W = img.shape[:2]
        uv, z = self.cam.project(X, self.ss)
        px = uv[:, 0].astype(int); py = uv[:, 1].astype(int)
        offs = [(dx, dy) for dx in range(-r + 1, r + 1) for dy in range(-r + 1, r + 1)] if r > 0 else [(0, 0)]
        for dx, dy in offs:
            x = px + dx; y = py + dy
            ok = (x >= 0) & (x < W) & (y >= 0) & (y < H)
            idx = y[ok] * W + x[ok]; zz = z[ok]; cc = col[ok]
            if idx.size == 0: continue                    # nothing of it on screen this frame
            o = np.lexsort((zz, idx))
            idx, zz, cc = idx[o], zz[o], cc[o]
            first = np.r_[True, idx[1:] != idx[:-1]]
            idx, zz, cc = idx[first], zz[first], cc[first]
            win = zz < zbuf[idx]
            zbuf[idx[win]] = zz[win]
            img.reshape(-1, 3)[idx[win]] = cc[win]

    def _ground(self, img, zbuf, gy, off):
        H, W = img.shape[:2]
        eye, fwd, right, up = self.cam.basis()
        f = 0.5 * H / np.tan(np.radians(self.cam.fov) / 2)
        ys, xs = np.mgrid[0:H, 0:W]
        d = (fwd[None, None] + right[None, None] * ((xs - W / 2) / f)[..., None] - up[None, None] * ((ys - H / 2) / f)[..., None])
        t = (gy - eye[1]) / np.where(np.abs(d[..., 1]) < 1e-9, -1e-9, d[..., 1])
        hit = t > 0
        X = eye[None, None] + d * t[..., None]
        gx = X[..., 0] + off[0]; gz = X[..., 2] + off[1]
        cell = (np.floor(gx / 0.5).astype(int) + np.floor(gz / 0.5).astype(int)) & 1
        line = (np.abs(gx / 0.5 - np.round(gx / 0.5)) < 0.03) | (np.abs(gz / 0.5 - np.round(gz / 0.5)) < 0.03)
        fade = np.exp(-np.maximum(t, 0) / 14.0)[..., None]
        base = np.where(cell[..., None] == 1, np.array([0.80, 0.74, 0.63]), np.array([0.84, 0.79, 0.69]))
        base = np.where(line[..., None], np.array([0.70, 0.64, 0.54]), base)
        bg = np.asarray(self.bg, np.float32) / 255
        g = base * fade + bg * (1 - fade)
        img[hit] = g[hit]
        zbuf[hit.reshape(-1)] = (t * np.einsum('ijk,k->ij', d, fwd))[hit].reshape(-1) + 1e6  # always behind the body

    def _shadow(self, img, zbuf, P, gy):
        Q = P[::3].copy(); Q[:, 1] = gy + 0.002
        H, W = img.shape[:2]
        uv, z = self.cam.project(Q, self.ss)
        px = np.clip(uv[:, 0].astype(int), 0, W - 1); py = np.clip(uv[:, 1].astype(int), 0, H - 1)
        mask = np.zeros((H, W), bool); mask[py, px] = True
        from PIL import ImageFilter
        mimg = Image.fromarray(mask.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.GaussianBlur(4 * self.ss))
        a = np.asarray(mimg, np.float32)[..., None] / 255 * 0.35
        img[:] = img * (1 - a)


def draw_bones(img, zbuf, proj, W, parents, color=(0.85, 0.1, 0.1), highlight=None):
    """Overlay bones (always on top) on a float image."""
    from PIL import Image as _I
    H, Wd = img.shape[:2]
    pil = _I.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)); dr = ImageDraw.Draw(pil)
    pts, _ = proj(W[:, :3, 3])
    for i, p in enumerate(parents):
        c = tuple(int(255 * x) for x in ((highlight or {}).get(i, color)))
        if p >= 0: dr.line([tuple(pts[p]), tuple(pts[i])], fill=c, width=3)
        dr.ellipse([pts[i][0] - 4, pts[i][1] - 4, pts[i][0] + 4, pts[i][1] + 4], fill=c)
    img[:] = np.asarray(pil, np.float32) / 255


def save_gif(frames, path, fps=30):
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=int(round(1000 / fps)), loop=0,
                   optimize=False, disposal=2)
