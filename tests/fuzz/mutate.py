# PZA's Kenshi Motion Lab - mutation check: plant known bugs in a scratch copy of the code and make sure the fuzzers
# catch every one (a fuzzer that passes with the bug planted proves nothing). Author: PZA. GPLv3.
#   python tests/fuzz/mutate.py
import os, shutil, subprocess, sys, tempfile
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# (file, exact text, replacement, fuzzer + args that must FAIL with the bug planted)
MUTANTS = [
    ('motionlab/profiles.py', "def valid_curve(key, v):\n",
     "def valid_curve(key, v):\n    return True\n", ['tests/fuzz/fuzz_curves.py', '60']),
    ('motionlab/profiles.py', "and bool(np.all(np.isfinite(v))) and float(np.max(np.abs(v))) <= lim)",
     "and True)", ['tests/fuzz/fuzz_curves.py', '120']),
    ('motionlab/curves.py', "if not (np.isfinite(length) and 0 < length <= 600): raise ValueError",
     "if False: raise ValueError", ['tests/fuzz/fuzz_reference.py', '300']),
    # the 2026-11-19 benchmark fixes (tests/test_animals.py must notice each one undone)
    ('motionlab/motion.py', "if neck in r.idx and not self.neck_carries_legs:", "if neck in r.idx:", ['tests/test_animals.py']),
    ('motionlab/motion.py', "tgt = (1 - fold) * (pts0[-1] if tgt is None else np.asarray(tgt)) + fold * ft",
     "tgt = ft if tgt is None else (1 - fold) * np.asarray(tgt) + fold * ft", ['tests/test_animals.py']),
    ('motionlab/motion.py', "le = min(lift, 0.45 * drv.leg_lens.get(e['name'], lift / 0.45))", "le = lift", ['tests/test_animals.py']),
    ('motionlab/motion.py', "                if True:\n                    if hips is not None:",
     "                if hips is not None:\n                    if hips is not None:", ['tests/test_animals.py']),
]

caught = 0
for i, (rel, old, new, cmd) in enumerate(MUTANTS):
    td = tempfile.mkdtemp(prefix='mlmut%d_' % i)
    for sub in ('motionlab', 'tests'):
        shutil.copytree(os.path.join(ROOT, sub), os.path.join(td, sub), ignore=shutil.ignore_patterns('__pycache__', 'kenshi_curves.json'))
    p = os.path.join(td, rel); src = open(p, encoding='utf-8').read()
    if src.count(old) != 1: print('MUTANT %d: anchor not found in %s (update this file)' % (i, rel)); sys.exit(2)
    open(p, 'w', encoding='utf-8').write(src.replace(old, new))
    r = subprocess.run([sys.executable] + [os.path.join(td, cmd[0])] + cmd[1:], cwd=td, capture_output=True, text=True, timeout=1800)
    ok = r.returncode != 0
    caught += ok
    print('mutant %d (%s): %s' % (i, rel, 'CAUGHT' if ok else 'MISSED  <- the fuzzer did not notice'))
    shutil.rmtree(td, ignore_errors=True)
print('mutation check: %d / %d caught: %s' % (caught, len(MUTANTS), 'PASS' if caught == len(MUTANTS) else 'FAIL'))
sys.exit(0 if caught == len(MUTANTS) else 1)
