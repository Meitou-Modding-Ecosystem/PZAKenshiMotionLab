# Ogre binary .skeleton reader: bones, hierarchy, animations. Author: PZA.
# usage: python -I ogre_skeleton_read.py <file.skeleton> [animation substring]
import struct, sys

HEADER, BLENDMODE, BONE, BONE_PARENT, ANIMATION, ANIM_BASEINFO, TRACK, KEYFRAME, LINK = \
    0x1000, 0x1010, 0x2000, 0x3000, 0x4000, 0x4010, 0x4100, 0x4110, 0x5000

def read(path):
    d = open(path, 'rb').read()
    p = 0
    def u16():
        nonlocal p; v = struct.unpack_from('<H', d, p)[0]; p += 2; return v
    def u32():
        nonlocal p; v = struct.unpack_from('<I', d, p)[0]; p += 4; return v
    def f(n):
        nonlocal p; v = struct.unpack_from('<%df' % n, d, p); p += 4 * n; return v
    def s():
        nonlocal p; e = d.index(b'\n', p); v = d[p:e].decode('latin-1'); p = e + 1; return v
    assert u16() == HEADER
    version = s()
    bones, parents, anims = {}, {}, []
    KNOWN = {BLENDMODE, BONE, BONE_PARENT, ANIMATION, ANIM_BASEINFO, TRACK, KEYFRAME, LINK}
    def next_is_chunk():
        return p + 2 > len(d) or struct.unpack_from('<H', d, p)[0] in KNOWN
    # chunk lengths in Kenshi's skeletons are unreliable (Ogre's writer miscounts), so parse field by field
    bh = None
    while p + 6 <= len(d):
        cid = u16(); ln = u32()
        if cid == BLENDMODE:
            u16()
        elif cid == BONE:
            name = s(); h = u16(); pos = f(3); q = f(4)
            sc = (1.0, 1.0, 1.0) if next_is_chunk() else f(3)
            bones[h] = (name, pos, q, sc)
        elif cid == BONE_PARENT:
            c = u16(); par = u16(); parents[c] = par
        elif cid == ANIMATION:
            name = s(); length = f(1)[0]
            anims.append((name, length, {}))
        elif cid == ANIM_BASEINFO:
            s(); f(1)
        elif cid == TRACK:
            bh = u16(); anims[-1][2][bh] = []
        elif cid == KEYFRAME:
            t = f(1)[0]; q = f(4); tr = f(3)
            sc = (1.0, 1.0, 1.0) if next_is_chunk() else f(3)
            anims[-1][2][bh].append((t, q, tr, sc))
        elif cid == LINK:
            s(); f(1)
        else:
            raise ValueError('unknown chunk 0x%X at %d' % (cid, p - 6))
    return version, bones, parents, anims

if __name__ == '__main__':
    v, bones, parents, anims = read(sys.argv[1])
    print(v, '| bones', len(bones), '| animations', len(anims))
    for h in sorted(bones)[:6]:
        n, pos, q, sc = bones[h]
        print('  bone %d %s parent=%s pos=%s' % (h, n, bones[parents[h]][0] if h in parents else '-', tuple(round(x, 3) for x in pos)))
    want = sys.argv[2].lower() if len(sys.argv) > 2 else None
    for name, length, tracks in anims:
        if want is None or want in name.lower():
            nk = sum(len(k) for k in tracks.values())
            print('  anim %-40s %.2fs tracks=%d keys=%d' % (name, length, len(tracks), nk))
