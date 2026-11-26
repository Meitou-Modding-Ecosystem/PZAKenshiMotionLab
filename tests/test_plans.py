# PZA's Kenshi Motion Lab - auto-rig body plans: quadruped, biped, crab, bird (+ auto). Author: PZA. GPLv3.
#   python tests/test_plans.py     (no game files, no Pillow required)
import json, os, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))
from synthetic import make_unrigged_quadruped, make_unrigged_biped, make_unrigged_crab, make_unrigged_bird
from motionlab import asset_export as ae, gltf, pipeline

fails = []
def ok(c, msg):
    print(('PASS ' if c else 'FAIL ') + msg)
    if not c:
        fails.append(msg)


d = tempfile.mkdtemp()

# 1. Quadruped
q_src = os.path.join(d, 'quad.glb')
gltf.write_glb(q_src, make_unrigged_quadruped())
rq = pipeline.rig_file(q_src, os.path.join(d, 'q'), plan='quadruped', log=lambda *a: None, png=False, clips='idle,walk,run,attack')
lmq = json.load(open(rq['landmarks'], encoding='utf-8'))
ok(rq['plan'] == 'quadruped' and len(lmq['legs']) == 4, 'quadruped: 4 legs found (%d)' % len(lmq['legs']))
ok(len(lmq['wings']) == 0, 'quadruped: 0 wings')
ok(set(rq['clips']) == {'idle', 'walk', 'run', 'attack'}, 'quadruped clips: %s' % rq['clips'])

# 2. Biped (also test plan='auto' picking biped for upright proportions)
b_src = os.path.join(d, 'biped.glb')
gltf.write_glb(b_src, make_unrigged_biped())
rb = pipeline.rig_file(b_src, os.path.join(d, 'b'), plan='auto', log=lambda *a: None, png=False, clips='idle,walk,run')
lmb = json.load(open(rb['landmarks'], encoding='utf-8'))
ok(rb['plan'] == 'biped' and len(lmb['legs']) == 2, 'biped (auto): 2 legs found (%d)' % len(lmb['legs']))
ok(len(lmb.get('arms', [])) == 2, 'biped (auto): 2 arms found (%d)' % len(lmb.get('arms', [])))
ok(set(rb['clips']) == {'idle', 'walk', 'run'}, 'biped clips: %s' % rb['clips'])

# 3. Crab (claws classified as arms, NOT wings/antennae)
c_src = os.path.join(d, 'crab.glb')
gltf.write_glb(c_src, make_unrigged_crab())
rc = pipeline.rig_file(c_src, os.path.join(d, 'c'), plan='crab', log=lambda *a: None, png=False, clips='idle,walk,run,attack')
lmc = json.load(open(rc['landmarks'], encoding='utf-8'))
ok(rc['plan'] == 'crab' and len(lmc['legs']) >= 4, 'crab: >=4 walking legs found (%d)' % len(lmc['legs']))
ok(len(lmc.get('arms', [])) == 2 and len(lmc['wings']) == 0 and len(lmc['antennae']) == 0,
   'crab: 2 claws as arms, 0 wings, 0 antennae (got arms=%d wings=%d ants=%d)' % (
       len(lmc.get('arms', [])), len(lmc['wings']), len(lmc['antennae'])))

# 4. Bird (2 legs, 2 wings -> flight clips)
f_src = os.path.join(d, 'bird.glb')
gltf.write_glb(f_src, make_unrigged_bird())
rf = pipeline.rig_file(f_src, os.path.join(d, 'f'), plan='bird', log=lambda *a: None, png=False, clips='idle,walk,hover,takeoff,land')
lmf = json.load(open(rf['landmarks'], encoding='utf-8'))
ok(rf['plan'] == 'bird' and len(lmf['legs']) == 2 and len(lmf['wings']) == 2,
   'bird: 2 legs and 2 wings (got legs=%d wings=%d)' % (len(lmf['legs']), len(lmf['wings'])))
ok('hover' in rf['clips'] and 'takeoff' in rf['clips'], 'bird has flight clips: %s' % rf['clips'])

# 5. Direct Kenshi asset export works on one of the new plans (quadruped)
exp = ae.export_asset(rq['glb'], os.path.join(d, 'exp_q'), name='wild_hound', log=lambda *a: None)
ok(os.path.isfile(exp['mesh']['path']) and os.path.isfile(exp['skeleton']['path']), 'quadruped exports directly to Kenshi .mesh + .skeleton')

print('\n%d failed' % len(fails) if fails else '\nall passed')
sys.exit(1 if fails else 0)
