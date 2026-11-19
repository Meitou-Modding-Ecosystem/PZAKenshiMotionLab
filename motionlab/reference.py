# PZA's Kenshi Motion Lab - reference clips (e.g. Mixamo downloads converted with tools/fbx_to_glb.py): load a
# skeleton-only .glb, map its bones to roles, and measure falls / knockdowns and get-ups. Only NUMBERS come out
# (timings, heights relative to the body, angles, contact times); no animation data is kept. Author: PZA. GPLv3.
import numpy as np
from . import mathx as mx
from .gltf import read_glb_raw, read_clips, node_worlds, node_local, GlbError, _at
from .model import Skeleton

MIXAMO = {  # role -> bone name suffix after "mixamorig:"
    'hips': 'Hips', 'spine': 'Spine', 'chest': 'Spine2', 'neck': 'Neck', 'head': 'Head', 'head_top': 'HeadTop_End',
    'l_thigh': 'LeftUpLeg', 'l_knee': 'LeftLeg', 'l_foot': 'LeftFoot', 'l_toe': 'LeftToeBase',
    'r_thigh': 'RightUpLeg', 'r_knee': 'RightLeg', 'r_foot': 'RightFoot', 'r_toe': 'RightToeBase',
    'l_shoulder': 'LeftArm', 'l_elbow': 'LeftForeArm', 'l_hand': 'LeftHand',
    'r_shoulder': 'RightArm', 'r_elbow': 'RightForeArm', 'r_hand': 'RightHand',
}


def load_reference(path):
    """(Skeleton, clip nodes, length in s) of a skeleton-only .glb. Parent nodes above the skin's joints (Mixamo's
    'Armature', which converts centimetres / Z up) are folded into the root's world transform."""
    js, binc = read_glb_raw(path)
    W, par = node_worlds(js)
    nodes = js.get('nodes', [])
    if not js.get('skins'): raise GlbError('no skeleton in the file')
    joints = js['skins'][0].get('joints') or []
    jset = set(joints)
    sk = Skeleton(); sk.root_world = []
    for j in joints:
        nd = _at(nodes, j, 'joint node')
        sk.names.append(nd.get('name', 'joint_%d' % j))
        p = par[j]
        while p >= 0 and p not in jset: p = par[p]
        sk.parents.append(joints.index(p) if p >= 0 else -1)
        t, q, s = mx.decompose(node_local(nd))
        sk.rest_t.append(t); sk.rest_q.append(q); sk.rest_s.append(s)
        sk.root_world.append(W[par[j]] if (sk.parents[-1] < 0 and par[j] >= 0) else np.eye(4))
    clips = read_clips(path)
    if not clips: raise GlbError('no animation in the file')
    clip = next(iter(clips.values()))
    length = max(float(ch['times'][-1]) for ch in clip.values())
    return sk, clip, length


def roles(sk, table=MIXAMO):
    out = {}
    for role, suffix in table.items():
        hit = [i for i, n in enumerate(sk.names) if n == suffix or n.endswith(':' + suffix)]
        if hit: out[role] = hit[0]
    return out


def sample_positions(sk, clip, length, fps=60):
    """World positions of the role bones at fps: {role: (frames, 3)} plus times."""
    from .bake import sample_read_clip
    R = roles(sk)
    ts = np.linspace(0, length, int(round(length * fps)) + 1)
    pos = {r: [] for r in R}
    for t in ts:
        W = sk.world(sample_read_clip(sk, clip, t))
        for r, i in R.items(): pos[r].append(W[i][:3, 3].copy())
    return ts, {r: np.array(v) for r, v in pos.items()}


def _first(ts, mask):
    i = np.argmax(mask) if mask.any() else -1
    return float(ts[i]) if i >= 0 and mask[i] else None


def measure_fall(ts, P):
    """A fall / knockdown: everything relative to the standing pelvis height H0 at the first frame."""
    hips = P['hips']; ground = min(P['l_toe'][:, 1].min(), P['r_toe'][:, 1].min(), hips[:, 1].min()) if 'l_toe' in P else 0.0
    y = hips[:, 1] - ground; H0 = float(y[0])
    head_y = P['head'][:, 1] - ground
    fwd0 = mx.norm(np.cross(P['l_thigh'][0] - P['r_thigh'][0], [0, 1, 0]))     # facing at the start (left x up = forward)
    torso = P['neck'] - P['hips']
    pitch = np.degrees(np.arccos(np.clip(mx.norm(torso)[:, 1], -1, 1)))         # 0 = upright, 90 = lying
    t_drop = _first(ts, y < 0.95 * H0)
    lying = y < 0.30 * H0
    t_ground = _first(ts, lying)
    v = np.gradient(y, ts)
    settle = None
    if t_ground is not None:
        after = ts >= t_ground
        still = after & (np.abs(v) < 0.05 * H0)
        # settled = still for the rest of the clip
        for k in np.nonzero(after)[0]:
            if still[k:].all(): settle = float(ts[k]); break
    bounce = 0.0
    if t_ground is not None:
        k0 = int(np.argmax(lying)); y_after = y[k0:]
        kmin = int(np.argmin(y_after[: max(2, int(0.4 * 60))]))
        bounce = float(max(0.0, y_after[kmin:].max() - y_after[kmin]) / H0)
    travel = hips[-1] - hips[0]; travel[1] = 0
    along = float(travel @ fwd0) / H0
    chest_up = None
    if 'chest' in P and 'l_shoulder' in P:
        # lying on the back = the chest faces up at the end
        across = P['l_shoulder'][-1] - P['r_shoulder'][-1]; spine = P['neck'][-1] - P['hips'][-1]
        front = mx.norm(np.cross(across, spine)); chest_up = float(front[1])
    hand_ground = {}
    for h in ('l_hand', 'r_hand'):
        if h in P: hand_ground[h] = _first(ts, (P[h][:, 1] - ground) < 0.12 * H0)
    knee = []
    for s in 'lr':
        a, b, c = P[s + '_thigh'], P[s + '_knee'], P[s + '_foot']
        u = a - b; w = c - b
        knee.append(180 - np.degrees(np.arccos(np.clip(np.einsum('ij,ij->i', u, w) / np.linalg.norm(u, axis=1) / np.linalg.norm(w, axis=1), -1, 1))))
    knee = np.max(knee, axis=0)
    drop_time = (t_ground - t_drop) if (t_ground is not None and t_drop is not None) else None
    return {'pelvis_height_m': H0, 'length_s': float(ts[-1]),
            't_start_drop_s': t_drop, 't_ground_s': t_ground, 'drop_time_s': drop_time, 't_settled_s': settle,
            'bounce': round(bounce, 3), 'travel_along_facing': round(along, 3),
            'ends_on': None if chest_up is None else ('back' if chest_up > 0.3 else 'stomach' if chest_up < -0.3 else 'side'),
            'torso_pitch_end_deg': round(float(pitch[-1]), 1), 'knee_bend_max_deg': round(float(knee.max()), 1),
            'hands_down_s': hand_ground, 'head_lowest': round(float(head_y.min() / H0), 3),
            'pelvis_curve': [round(float(x / H0), 3) for x in y[:: max(1, len(y) // 40)]]}


def measure_getup(ts, P):
    """A get-up: phases by pelvis height (relative to the final standing height), supports by contact."""
    hips = P['hips']; ground = min(P['l_toe'][:, 1].min(), P['r_toe'][:, 1].min())
    y = hips[:, 1] - ground; Hs = float(y[-1])
    phases = {}
    for name, lo in (('off_ground', 0.30), ('half_up', 0.55), ('crouch', 0.80), ('standing', 0.95)):
        phases[name] = _first(ts, y >= lo * Hs)
    contacts = {}
    for part in ('l_hand', 'r_hand', 'l_knee', 'r_knee'):
        if part in P:
            down = (P[part][:, 1] - ground) < 0.12 * Hs
            # intervals when the part carries weight
            iv = []; k = 0
            while k < len(down):
                if down[k]:
                    j = k
                    while j + 1 < len(down) and down[j + 1]: j += 1
                    if ts[j] - ts[k] > 0.1: iv.append([round(float(ts[k]), 2), round(float(ts[j]), 2)])
                    k = j + 1
                else: k += 1
            contacts[part] = iv
    torso = P['neck'] - P['hips']
    pitch = np.degrees(np.arccos(np.clip(mx.norm(torso)[:, 1], -1, 1)))
    yaw = np.degrees(np.unwrap(np.arctan2(torso[:, 0], torso[:, 2])))
    return {'stand_pelvis_height_m': Hs, 'length_s': float(ts[-1]), 'phase_times_s': phases, 'contacts': contacts,
            'torso_pitch_start_deg': round(float(pitch[0]), 1), 'torso_pitch_at_half_up_deg':
                round(float(pitch[np.argmax(y >= 0.55 * Hs)]), 1) if phases['half_up'] else None,
            'pelvis_curve': [round(float(x / Hs), 3) for x in y[:: max(1, len(y) // 40)]]}


def measure_roll(ts, P):
    """A forward roll: the body's pitch turned through (summed about the side axis, 360 = one full turn), when it is
    upside down, pelvis and head heights, how tight the tuck is, travel, and hand / back contacts."""
    hips = P['hips']; ground = min(P['l_toe'][:, 1].min(), P['r_toe'][:, 1].min(), P['head'][:, 1].min())
    y = hips[:, 1] - ground; H0 = float(y[0])
    side = mx.norm(P['l_thigh'] - P['r_thigh'])
    torso = P['neck'] - P['hips']
    fwd0 = mx.norm(np.cross(side[0], [0, 1, 0]))
    # pitch angle of the torso in the plane (forward, up): 0 = upright, +90 = leaning flat forward, 180 = upside down
    ang = np.unwrap(np.arctan2(torso @ fwd0, torso[:, 1]))
    ang = np.degrees(ang - ang[0])
    inv = ang > 135
    tuck = np.linalg.norm(0.5 * (P['l_knee'] + P['r_knee']) - P['chest'], axis=1) / H0
    travel = hips[-1] - hips[0]; travel[1] = 0
    back_down = (P['chest'][:, 1] - ground) < 0.25 * H0
    hd = {h: _first(ts, (P[h][:, 1] - ground) < 0.12 * H0) for h in ('l_hand', 'r_hand') if h in P}
    t_inv = _first(ts, inv)
    t_back = [round(float(t), 2) for t in (ts[back_down][0], ts[back_down][-1])] if back_down.any() else None
    t_up = None
    if t_inv is not None:
        k = int(np.argmax(inv)); after = np.nonzero((y >= 0.9 * H0) & (np.arange(len(y)) > k))[0]
        t_up = float(ts[after[0]]) if len(after) else None
    return {'pelvis_height_m': H0, 'length_s': float(ts[-1]), 'turned_deg': round(float(ang.max()), 1),
            't_hands_down_s': hd, 't_upside_down_s': t_inv, 'upside_down_for_s': round(float(inv.sum() * (ts[1] - ts[0])), 3),
            'back_on_ground_s': t_back, 't_standing_again_s': t_up,
            'pelvis_min': round(float(y.min() / H0), 3), 't_pelvis_min_s': float(ts[np.argmin(y)]),
            'head_lowest': round(float((P['head'][:, 1] - ground).min() / H0), 3),
            'tuck_min': round(float(tuck.min()), 3), 'travel_along_facing': round(float(travel @ fwd0) / H0, 3),
            'pelvis_curve': [round(float(x / H0), 3) for x in y[:: max(1, len(y) // 40)]],
            'pitch_curve_deg': [round(float(x)) for x in ang[:: max(1, len(ang) // 40)]]}
