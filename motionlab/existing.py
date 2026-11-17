# PZA's Kenshi Motion Lab - use a model's OWN skeleton instead of auto-rigging it. Author: PZA. GPLv3.
#
# Kenshi's humans and animals are rigged with 3ds Max Biped-style bones ("Bip01 L Thigh", "Bip01 R UpperArm",
# "Bip01 Tail1", ...). This maps those bones to the roles the motion driver understands, so the same gait, springs and
# clips run on them with their own skeleton and skin untouched. Models already rigged by the Motion Lab (leg_L1_01 ...)
# map back too, so a rigged .glb can be re-animated.
#
#   legs     Thigh -> Calf (hind legs / human legs; the Foot bone is kept level) and, when the hand reaches the ground,
#            UpperArm -> Forearm (front legs; the Hand is the foot)
#   arms     UpperArm -> Forearm when the hand hangs free (bipeds): they swing against the legs
#   spine    Tail chain (springs), head = Head, jaw = Jaw (opens in the bite), ears / ponytails = spring chains
import re
import numpy as np


def _chain_from(sk, start):
    out = [start]
    while True:
        ch = sk.children(out[-1])
        ch = [c for c in ch if not sk.names[c].lower().endswith('nub')]
        if len(ch) != 1: break
        out.append(ch[0])
    return out


def _bip01_map(model):
    sk = model.skeleton; names = sk.names
    W = sk.world_rest()
    pos = lambda i: W[i][:3, 3]
    lo, hi = model.bounds(); H = float(hi[1] - lo[1])
    find = {}
    for i, nm in enumerate(names):
        m = re.match(r'^Bip\d+\s*(.*)$', nm, re.I)
        if m: find[m.group(1).strip().lower()] = i
    if 'l thigh' not in find or 'r thigh' not in find: return None
    root = next(i for i, p in enumerate(sk.parents) if p < 0)
    body = find.get('spine', find.get('pelvis', root))
    head = find.get('head')
    rm = {'plan': 'biped', 'units': 'metres', 'forward': '+Z', 'up': '+Y', 'root': names[root], 'body': names[body],
          'head': names[head] if head is not None else names[body], 'spine': [], 'legs': [], 'wings': [], 'antennae': [],
          'arms': [], 'source_rig': 'Bip01'}
    def leg(side, upper, lower, foot, n, bend):
        u, l = find.get('%s %s' % (side, upper)), find.get('%s %s' % (side, lower))
        f = find.get('%s %s' % (side, foot))
        if u is None or l is None or f is None: return None
        return {'name': 'leg_%s%d' % (side.upper(), n), 'side': side.upper(), 'bones': [names[u], names[l]],
                'foot': names[f], 'tip': [float(x) for x in pos(f)], 'parent': names[sk.parents[u]] if sk.parents[u] >= 0 else names[root],
                'lifted': False, 'bend': bend}
    # front limbs: legs if the hand comes near the ground, else arms
    front = []
    for side in ('l', 'r'):
        a = find.get('%s upperarm' % side); fa = find.get('%s forearm' % side); h = find.get('%s hand' % side)
        if a is not None and fa is not None and h is not None:
            front.append((side, a, fa, h, pos(h)[1] - lo[1] < 0.3 * H))
    quad = len(front) == 2 and all(f[4] for f in front)
    for side in ('l', 'r'):
        n = 2 if quad else 1
        e = leg(side, 'thigh', 'calf', 'foot', n, [0.0, 0.0, 1.0])          # knees (stifles) point forward
        if e: rm['legs'].append(e)
    for side, a, fa, h, on_ground in front:
        if quad:
            rm['legs'].append({'name': 'leg_%s1' % side.upper(), 'side': side.upper(), 'bones': [names[a], names[fa]],
                               'foot': names[h], 'tip': [float(x) for x in pos(h)], 'parent': names[sk.parents[a]],
                               'lifted': False, 'bend': [0.0, 0.0, -1.0]})           # elbows point back
        else:
            rm['arms'].append({'name': 'arm_%s' % side.upper(), 'side': side.upper(), 'bones': [names[a], names[fa]],
                               'hand': names[h]})
    rm['plan'] = 'quadruped' if quad else 'biped'
    if 'tail' in find: rm['spine'] = [names[i] for i in _chain_from(sk, find['tail'])]
    jaw = find.get('jaw')
    if jaw is not None: rm['jaw'] = names[jaw]
    for key, i in find.items():                                   # ears / ponytails: springy chains on the head
        if (key.startswith('ear_') or key.startswith('ponytail')) and sk.parents[i] == head and not key.endswith('nub'):
            ch = _chain_from(sk, i)
            x = pos(i)[0]
            rm['antennae'].append({'name': 'ear_%s_%d' % ('L' if x > 0 else 'R', len(rm['antennae'])), 'side': 'L' if x > 0 else 'R',
                                   'bones': [names[c] for c in ch], 'tip': [float(v) for v in pos(ch[-1])],
                                   'parent': names[head], 'stiff': key.startswith('ponytail')})
    rm['tails'] = {}
    return rm


def _motionlab_map(model):
    """A skeleton the Motion Lab made (or one named the same way)."""
    sk = model.skeleton; names = set(sk.names)
    if not {'root', 'body', 'head'} <= names: return None
    W = sk.world_rest()
    rm = {'plan': 'insect', 'units': 'metres', 'forward': '+Z', 'up': '+Y', 'root': 'root', 'body': 'body', 'head': 'head',
          'spine': sorted(n for n in names if re.match(r'^abdomen_\d+$', n)), 'legs': [], 'wings': [], 'antennae': [],
          'arms': [], 'tails': {}, 'source_rig': 'motionlab'}
    groups = {}
    for n in sk.names:
        m = re.match(r'^((leg|wing|antenna)_[LR]\d*)_(\d+)$', n)
        if m: groups.setdefault((m.group(2), m.group(1)), []).append(n)
    lo, hi = model.bounds(); H = float(hi[1] - lo[1])
    for (kind, nm), bones in sorted(groups.items()):
        bones.sort()
        last = sk.index(bones[-1]); tip = W[last][:3, 3] + W[last][:3, 1] * 0.0
        ch = sk.children(last)
        e = {'name': nm, 'side': nm.split('_')[1][0], 'bones': bones, 'tip': [float(x) for x in tip],
             'parent': sk.names[sk.parents[sk.index(bones[0])]]}
        if kind == 'leg': e['lifted'] = bool(tip[1] - lo[1] > 0.15 * H)
        rm[{'leg': 'legs', 'wing': 'wings', 'antenna': 'antennae'}[kind]].append(e)
    return rm if rm['legs'] else None


def rigmap_from_skeleton(model):
    """Role map for the model's own skeleton, or None when its bones are not a layout we know."""
    if model.skeleton is None: return None
    for f in (_bip01_map, _motionlab_map):
        rm = f(model)
        if rm and rm['legs']:
            if rm['plan'] in ('biped', 'quadruped'):
                from .profiles import lookup
                prof, src = lookup(getattr(model, 'skeleton_file', None), rm['plan'])
                if prof: rm['gait_profile'] = prof; rm['gait_profile_from'] = src
            return rm
    return None
