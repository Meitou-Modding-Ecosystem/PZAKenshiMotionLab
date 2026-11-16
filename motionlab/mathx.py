# PZA's Kenshi Motion Lab - small vector / quaternion / matrix helpers (numpy). Author: PZA. GPLv3.
# Quaternions are (x, y, z, w) like glTF; Ogre wants (w, x, y, z): use q_to_wxyz / q_from_wxyz at the edge.
import numpy as np

def norm(v, eps=1e-12):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, eps)

def q_identity():
    return np.array([0.0, 0.0, 0.0, 1.0])

def q_mul(a, b):
    ax, ay, az, aw = a; bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw,
                     aw * bw - ax * bx - ay * by - az * bz])

def q_conj(q):
    return np.array([-q[0], -q[1], -q[2], q[3]])

def q_axis_angle(axis, ang):
    axis = norm(axis); s = np.sin(ang / 2.0)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, np.cos(ang / 2.0)])

def q_rotate(q, v):
    return q_to_m3(q) @ np.asarray(v, dtype=np.float64)

def q_between(a, b):
    """Shortest rotation taking direction a to direction b."""
    a = norm(a); b = norm(b)
    d = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if d > 1.0 - 1e-10: return q_identity()
    if d < -1.0 + 1e-10:
        ax = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(ax) < 1e-6: ax = np.cross(a, [0.0, 1.0, 0.0])
        return q_axis_angle(ax, np.pi)
    ax = np.cross(a, b)
    q = np.array([ax[0], ax[1], ax[2], 1.0 + d])
    return q / np.linalg.norm(q)

def q_slerp(a, b, t):
    a = np.asarray(a, float); b = np.asarray(b, float)
    d = float(np.dot(a, b))
    if d < 0: b = -b; d = -d
    if d > 0.9995:
        r = a + t * (b - a); return r / np.linalg.norm(r)
    th = np.arccos(d); s = np.sin(th)
    return (np.sin((1 - t) * th) * a + np.sin(t * th) * b) / s

def q_to_m3(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])

def m3_to_q(m):
    m = np.asarray(m, float); t = np.trace(m)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2; w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s; y = (m[0, 2] - m[2, 0]) / s; z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w = (m[2, 1] - m[1, 2]) / s; x = 0.25 * s; y = (m[0, 1] + m[1, 0]) / s; z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w = (m[0, 2] - m[2, 0]) / s; x = (m[0, 1] + m[1, 0]) / s; y = 0.25 * s; z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w = (m[1, 0] - m[0, 1]) / s; x = (m[0, 2] + m[2, 0]) / s; y = (m[1, 2] + m[2, 1]) / s; z = 0.25 * s
    q = np.array([x, y, z, w]); return q / np.linalg.norm(q)

def trs(t=(0, 0, 0), q=(0, 0, 0, 1), s=(1, 1, 1)):
    m = np.eye(4)
    m[:3, :3] = q_to_m3(q) * np.asarray(s, float)[None, :]
    m[:3, 3] = t
    return m

def decompose(m):
    t = m[:3, 3].copy()
    sx, sy, sz = (np.linalg.norm(m[:3, i]) for i in range(3))
    r = m[:3, :3] / np.array([sx, sy, sz])[None, :]
    if np.linalg.det(r) < 0: sx = -sx; r[:, 0] = -r[:, 0]
    return t, m3_to_q(r), np.array([sx, sy, sz])

def q_to_wxyz(q): return (float(q[3]), float(q[0]), float(q[1]), float(q[2]))
def q_from_wxyz(q): return np.array([q[1], q[2], q[3], q[0]], float)
