# PZA's Kenshi Motion Lab

Author: PZA. Licence: GPLv3 (see `LICENSE`). **Status: beta (0.2.0).** It works end to end on our test bodies and creatures; it
has not been tried in Kenshi itself yet (see *Known gaps*). Feedback and broken models are welcome as issues.

Rig and animate a model **from code** instead of hand-animating it. Give it a model (a Meshy / Blender `.glb`, or a
Kenshi `.mesh` + `.skeleton`) and it:

1. **finds the body plan** from the mesh shape: legs, wings, antennae, head, thorax, abdomen;
2. **places a clean, named skeleton** and binds the mesh (3 weights per vertex, like Kenshi);
3. **drives the bones from code**: a foot-planting gait for any number of legs (tripod for six, trot for four),
   a wingbeat, wings that **shrink away on landing and grow back on take-off (0.25 s)**, springy antennae and abdomen;
4. **bakes clips** (idle, walk, run, hover, takeoff, land, attack = a bite, sting) into a new `.glb`, with a report and preview GIFs.

Two ways in:

- **A model that is already rigged the Kenshi way** (Kenshi's humans of every race, its animals, anything exported with
  the same `Bip01 ...` bone names) **keeps its own skeleton and skin**. The Motion Lab recognises legs, arms, tail, ears
  and jaw by their names and adds the motion: people get idle, walk and run (arms swing) plus **knocked down onto the
  back / onto the stomach, getting up from each, a forward roll, a dive roll, a standing jump, a long jump, a jump
  down from a ledge, two landings and a quick jump in stride from the walk or run**; four-legged animals get idle, walk, run (a trot) and a bite.
- **A model with no rig, or a rig it does not know** (a Meshy export), is **auto-rigged** from its shape. Today that
  auto-rig knows insect-like bodies (legs, wings, antennae, abdomen); more body plans later.

**Gaits measured from Kenshi.** The walk and run numbers (cycle length, stride, foot lift, how low the pelvis rides,
how far the knee straightens and bends, arm swing, elbow) are measured from Kenshi's own walk / run animations for every
skeleton that has them (`tools/build_gait_profiles.py`: tables in `motionlab/data/kenshi_gaits.json` and `kenshi_curves.json`). A model on one of those skeletons moves by its own numbers; other people and animals use the median.
`tools/measure_gait.py` prints the same numbers for Kenshi's clips and for ours side by side. For Kenshi's human walk
ours now comes out at: knee 8-83 degrees (Kenshi 12-73), pelvis 0.93-0.97 m (0.92-0.97), arm swing -23 / +24 degrees
(-21 / +24). Four-legged animals are close but not exact yet (the dog walks a little lower than Kenshi's).

**Natural walk and run (2026-11-19).** The old walk and run looked machine-made, and the numbers said why: the
pelvis never turned, the feet never rolled (0 degrees, against Kenshi's 15-42 and ~90), and the hip height was built
from reach limits (40% of its motion in sharp corners, Kenshi's 3%). People now:
- turn the pelvis toward the leading leg, turn the chest against it and keep the head looking ahead;
- roll each foot heel strike -> flat -> toe-off about the heel and the ball, which stay planted (the toes bend at
  push-off);
- ride the hips on a smooth double bounce, with the reach limit only a safety net that dips smoothly when it bites;
- move the shoulders and swing the arms across the body.

Where the shapes come from:
- **`motionlab/data/kenshi_curves.json`:** curves measured from Kenshi's walk / run (body turns, arms, foot roll, the
  ankle's path) for every skeleton that has them, plus the median for people. With them, our walk / run match
  Kenshi's on every Studio number. `python tools/build_gait_profiles.py "<Kenshi>/data"` rebuilds the file. It is
  checked on load: wrong shapes or numbers are dropped, and that gait falls back to the summaries.
- **`kenshi_gaits.json`:** summaries (the mean and one sine per body turn, the hips' second harmonic, four numbers of
  the foot roll) drive our own shapes when the curves file is missing, or with `MOTIONLAB_NO_CURVES=1`.

**Motion Lab Studio: see how natural a clip moves.**

```bash
python -m motionlab studio "<Kenshi>/data/character/meshes/human/human_male.mesh" --skeleton "<Kenshi>/data/character/meshes/male_skeleton/male_skeleton.skeleton" --against "<Kenshi>/data/character/meshes/male_skeleton/male_skeleton.skeleton" -o out/studio
```

`out/studio/studio.html` shows, for the walk and run:
- ours and Kenshi's on the same body side by side, lined up by phase;
- our curves over Kenshi's (pelvis height / yaw / roll, chest counter-turn, foot roll, head height, arm swing, elbow);
- a scorecard graded green / amber / red: pelvis turn and roll, chest counter-turn, foot roll, chest lean, arm swing,
  elbow, hip bob, head bob, hip-height corners, jerk, ground speed.

Without `--against` it grades against the numbers in `kenshi_gaits.json` (no pictures of Kenshi's clips). A page made
with `--against` holds pictures of Kenshi's own animation: keep it to yourself. `tests/test_naturalness.py` holds the
same numbers as gates, so a change that looks better on one clip can't quietly break another.

**Retarget a reference clip onto a Kenshi body (`--reference`).** Give the rig command (or the add-on's *Reference
clips* folder) skeleton-only reference clips, e.g. Mixamo downloads you converted with `tools/fbx_to_glb.py`, and each
becomes a clip `ref_<file name>` on the body's OWN Kenshi skeleton. `motionlab/curves.py` reads the reference frame by
frame into curves that do not depend on its bones or size (pelvis path against its standing height, the turn of
pelvis, chest and head, where each foot and hand is, which way knees and elbows point) and the usual driver plays them
on the Kenshi rig, scaled to it, with the floor / knee rules on. This is the most natural-looking motion the Motion
Lab makes. **The repo ships only the code:** no reference files and nothing made from them. Use files you are allowed
to use, and keep what you make from them where their licence allows (Mixamo's terms cover use in your own games).

```bash
python -m motionlab rig human_male.mesh --skeleton male_skeleton.skeleton -o out/human --reference my_clips/roll.glb --reference my_clips/jab.glb
```

**Knockdowns, get-ups and rolls learned from reference clips.** `motionlab/reference.py` reads a skeleton-only reference
`.glb` (for example a Mixamo download converted by `tools/fbx_to_glb.py`, with your own Adobe account) and measures it:
when the pelvis starts to drop and reaches the ground, the bounce, how far the body travels, which side it ends on,
when hands and knees touch down; for rolls how far the body turns, when it is upside down and on its back. Only those numbers go into the code (`motionlab/moves.py`); no animation data and no
reference files are in this repo. The moves are built from body states (how far the body is tipped, pelvis height,
feet planted or carried, hands on the ground) and the leg / arm solvers do the rest. Rules that keep them honest, each
counted by `check.move_check`: a kneeling knee rests on the floor instead of going through it, a planted hand lets go
once the body rises out of its reach, a deep crouch slides the feet out along the ground instead of into it, the head / spine / arms never sink under
the floor (the body is lifted instead). The preview GIFs follow a body that travels (rolls, knockdowns). To keep them from looking mechanical: the keys are joined by a
monotone cubic, so the body flows through a pose instead of stopping at each one (impacts keep their sudden stop); the
head, spine, arms and legs reach each pose a little after the body, left and right at different times and sizes; a
bend spreads over all the spine bones and the neck. The
get-ups follow the reference's phases (hands down while sitting or pushing up, knees down while kneeling) at game pace,
about half the reference length.

## In Blender (the easy way)

1. Download `pza_motion_lab-<version>.zip` from the **Releases** page (or build it: `python tools/build_addon.py`,
   it lands in `out/addon/`).
2. Blender 4.2 or newer: **Edit > Preferences > Get Extensions > (v menu, top right) > Install from Disk**, pick the zip.
3. Import your model (File > Import > glTF), click its **mesh**, press **N**, open the **Motion Lab** tab:
   - **1. Rig > Auto-rig & Animate**: the model is rigged and gets every clip as an action (a few seconds; the
     original is hidden, the result is in a "Motion Lab: <name>" collection). **Options** has the wings mode, bone
     counts, which clips, the output folder.
   - **2. Clips**: press a clip to play it. **Show report** opens the report (Kenshi limits, limbs, clip speeds).
   - **3. Fix a limb**: **Show limb markers** puts a small sphere on every point of every limb it found (big spheres =
     hip and tip). Move them; delete all of a limb's spheres to drop that limb; then **Rebuild from markers**.
   - **4. Kenshi**: export with the Kenshi IO add-on (`.mesh` + `.skeleton`).

No extra installs: Blender ships numpy, which is all the add-on needs.

## Command line

Needs Python 3.10+ with numpy and Pillow (`pip install numpy pillow`; Pillow only for the GIFs and pictures).

```bash
python -m motionlab rig "My Creature.glb" -o out/my_creature --gifs
```

In `out/my_creature/` you get:

| File | What |
|---|---|
| `<name>_rigged.glb` | the mesh (same materials and textures), the new skeleton, the skin, and every clip as an animation |
| `<name>_report.md` | bone count and weights against Kenshi's limits, the limbs found, the clips with their ground speed and foot checks |
| `<name>_landmarks.png` | four views with the limbs it found drawn on: **look at this first** |
| `<name>.landmarks.json` | what it found, editable (see below) |
| `<name>.motion.json` | which bone does what (legs, wings, antennae, spine) + clip list: for other tools (MeitouMaestro later) |
| `<name>_<clip>.gif` | each clip, rendered from the clip **read back out of the .glb** (what the file holds, not a preview of the code) |

Other commands:

```bash
python -m motionlab view out/my_creature/My_Creature_rigged.glb --clip walk -o walk.gif
```

```bash
python -m motionlab info "My Creature.glb"
```

A Kenshi model straight from the game files (rig kept; `--own-rig no` forces the auto-rig):

```bash
python -m motionlab rig "<Kenshi>/data/character/meshes/human/human_male.mesh" --skeleton "<Kenshi>/data/character/meshes/male_skeleton/male_skeleton.skeleton" -o out/human --gifs
```

Options for `rig`: `--ground-wings hidden|folded` (default hidden: wings vanish on the ground),
`--leg-bones 4 --wing-bones 2 --antenna-bones 2 --abdomen-bones 3`, `--clips walk,idle`, `--fps 30`,
`--skeleton file.skeleton` for a `.mesh`, `--landmarks edited.json`.

## How it works

**Frame.** Up is +Y. The body's long axis is turned to face +Z (glTF's forward), centred on the leg pairs, and its
lowest point is put on the ground (y = 0). The report says how much it turned (generated models are often exported a
little turned, which can make their own rig look lopsided).

**Body and limbs.** The mesh is voxelised and filled (it does not need to be watertight). Opening the solid (erode,
then dilate) keeps the thick body and removes everything thin; what is left falls apart into limbs. Inside each limb,
distances from where it meets the body give its far ends; an end counts only if it is a real offshoot (spikes along a
leg, claws of one foot and the corners of a wing do not). Each limb's centre line follows those distances from tip to
body. A limb is a **leg** if it ends near the ground, a **wing** if it starts high and reaches back, an **antenna**
if it reaches forward from high up. Head, thorax and abdomen come from the body's cross-section along its length.

**Skeleton** (names the driver and other tools look for):

| Bone | What |
|---|---|
| `root` | on the ground under the thorax; moves the whole creature |
| `body` | thorax; legs and wings hang from it |
| `head` | head; antennae hang from it |
| `abdomen_01..` | abdomen / tail, front to back |
| `leg_L1_01..04` | legs: L/R = the creature's left / right, 1 = front; `_01` at the hip |
| `wing_L_01..02` | wings: `_01` at the hinge on top of the thorax |
| `antenna_L_01..02` | antennae: `_01` at the base |

**Weights.** Each vertex only sees the bones of its own limb plus the body bone that limb hangs from, so a leg never
pulls body skin. Weight falls off with distance to each bone, is smoothed over the surface and pruned to the 3
strongest (Kenshi's GPU reads 3).

**Motion.**
- *Gait*: neighbouring legs (along a side and across the body) are half a cycle apart: tripod for six legs. A planted
  foot stays planted: the clips are in place, so a planted foot slides back at exactly the clip's ground speed
  (written in the report: set the creature's walk speed to match). Each leg swings at the hip toward its foot
  target, then bends at the knee (upper and lower leg kept rigid, the knee stays on the side it bends to at rest), so
  knees visibly work and feet keep their angle to the shin. The stride is capped so every leg can reach. Straight-legged
  models stand with a slight bend and crouch more while walking and running to get a real stride.
- *Lifted legs*: a leg whose tip rests off the ground (a raised hind leg, say) rides along and does not plant.
- *Wings*: spread out sideways, beat at 7.5 Hz (4 frames per beat at 30 fps), with a twist for a figure-eight. On the
  ground they are scaled to nothing at their hinge (easier than unfurling them); take-off grows them back in 0.25 s,
  landing shrinks them in the last 0.25 s.
- *Springs*: antennae and the abdomen follow their pose with lag (damped springs), so they swing after the body moves.
  Looping clips run two warm-up cycles first, so the springs loop without a seam.

**Checks.** Each clip logs its worst foot IK miss and the worst slip of a planted foot against the ground; the
`tests/` run the whole pipeline on a procedural six-legged creature (no third-party files needed).

## Fixing what it found

Open `<name>_landmarks.png`. If a limb is wrong (a spike taken for a leg, a wing hinge in the wrong place), edit
`<name>.landmarks.json`: each limb is a `name`, a `side` and its `points` (from the body out to the tip, metres, in the
turned frame). Delete a wrong limb, move points, or add one. Then run again with the edits:

```bash
python -m motionlab rig "My Creature.glb" -o out/my_creature --landmarks out/my_creature/My_Creature.landmarks.json --gifs
```

## Into Kenshi

- **Scale:** the `.glb` is in metres; Kenshi works in about 10 units per metre (x10 on export).
- **Limits:** 60 bones per submesh, 3 weights per vertex; the report checks both.
- **Export:** open the rigged `.glb` in Blender (File > Import > glTF), then export `.mesh` + `.skeleton` with the
  Kenshi IO Continued add-on. The clips arrive as Blender actions. `tools/blender_check.py` shows what Blender sees:

```bash
blender -b --factory-startup --python tools/blender_check.py -- out/my_creature/My_Creature_rigged.glb
```

- **Not verified yet:** whether Kenshi plays bone *scale* keys (the wing shrink). If it does not, use
  `--ground-wings folded` (wings stay folded on the ground and spread for flight).
- **Materials:** Kenshi has no metallic-roughness; colour and normal maps carry over.

## Known gaps (beta)

- **Not tried in Kenshi yet.** The clips are checked in Blender and by our own tests; the trip through Kenshi IO into
  the game is next.
- Whether Kenshi plays bone *scale* keys (the wing shrink) is unknown: `--ground-wings folded` is the fallback.
- People: idle, walk, run, two knockdowns, two get-ups, two forward rolls, jumps (up, forward, down from a 1.22 m
  ledge), landings (soft, hard) and one dance for fun (`hip_thrust`, 110 BPM loop); no combat moves yet. Jump flights use real gravity (metres). Kenshi has no jump
  of its own: these clips are for mods that add one (Meitou, Part B). The preview draws no ledge under jump_down.
- **Quick jumps on the move (`jump_walk`, `jump_run`):** no crouch: the planted foot pushes off 0.10 s into the clip,
  the body flies 0.30 s (walk) / 0.38 s (run) with the legs held in their split, lands on the other foot and carries
  on at the gait's speed. Each starts and ends on the same frame of our walk / run (which match Kenshi's measured
  gaits), so it blends in and out. A whole jump cannot fit in 300 ms: that much air time only lifts the body ~11 cm.
- Smoothness rule (tested): the hips never change speed by more than 1 m/s from one 1/60 s frame to the next in the
  walk, run and in-stride jumps. The run used to sag and snap back at each toe-off (6 m/s in one frame); the hip
  height is now smoothed round the cycle and a foot the hips can no longer reach comes off its toes.
- Feet (tested in every clip: no foot changes speed by more than 8.5 m/s from one 1/60 s frame to the next): a swing
  foot lifted close under the hip is lowered along the body's own down instead of being pushed round the hip (it
  used to flip from behind to in front in one frame); the swing leaves and lands at part of the ground speed; a
  kneeling knee keeps where the leg put it and, where the hips are too low for the thigh, the body is lifted
  instead of the knee flipping sides; the stomach get-up's hands-and-knees hips sit one thigh length up.
- **Animals and flyers (fixed 2026-11-19, found by a benchmark over Kenshi's dog, the human bodies and an
  auto-rigged insect):**
  - Kenshi's animals hang their front legs off the neck, so a head turn there swung the shoulders: the dog's bite
    lifted its planted front paws and slid them. On such a body the whole head turn now goes on the head bone.
  - One swing height for every leg aimed the dog's short front paws at its shoulders in the run, and the knee rule
    then snapped them down (30 m/s in one frame). Each leg now swings at most 0.45 of its own length.
  - The hips no longer take a one-frame step where a run's flight phase starts. The reach limit dips the hip curve
    smoothly for every body, not only for people. The dog's run still turns over sharply at the top of each bound:
    2.8 m/s per frame, against 3.0 in Kenshi's own dog run.
  - A leg that rides along with the body (lifted) now tucks for flight as gradually as the others. It used to jump
    straight to the full tuck: 13 m/s at take-off and landing.
  - The floor check measures the body only. Kenshi's held-item sockets (Prop1 / Prop2, children of the hands, no
    skin) sit about 20 cm past the palm, so a hand planted in a get-up or roll puts them under the floor. That
    matters only for a held weapon, and `move_check` reports it apart as `socket_lowest`.
- **Ground speed (open):** with the measured curves, our run's planted foot slides at 8.3 m/s (male) / 8.2 (female),
  Kenshi's ankle at 6.6 / 6.4. Our heel and ball stay exactly planted while the foot rolls 70 degrees onto its toes;
  Kenshi's ankle number does not count that roll. Which one the game's run speed matches is unchecked until the clip
  plays in Kenshi (the Studio shows it amber for the female run).
- (Before the 2026-11-19 natural walk / run, kept for the history:) The run, side by side with Kenshi's own on the same body: the elbow bends most as the arm comes FORWARD (fist up
  by the chest; a flipped sign threw a straight arm forward), the chest leans 20 degrees, the heel kicks highest 35%
  into the swing (behind the body) and comes through low, and the back foot only pushes off in the second half of
  its stance (Kenshi plants it for ~15% of the stride, ours for 30%) instead of dragging the hips down. Moves that travel
  (knockdowns, rolls) carry the distance in the root bone; whether Kenshi wants that or an in-place clip is untested. Four-legged animals: idle, walk, run
  and a bite. The knockdowns end in one fixed lying pose; Kenshi's ragdoll and its own get-up timing are not matched.

## Limits

- The auto-rig (for models without a known rig) knows one body plan, `insect`. Kenshi's own crab, run through the
  auto-rig, gets its four walking legs but its claws are taken for wings and antennae; with its own Bip01 rig kept (the
  default) it walks as a four-legged body.
- One mesh per file is read (the first skinned one, else the first mesh).
- Rendering is a small software renderer (numpy): fine for checking motion, not a beauty render.

## Tests

```bash
python tests/test_pipeline.py
```

```bash
python tests/test_no_pillow.py
```

```bash
python tests/test_own_rig.py
```

```bash
python tests/fuzz/fuzz_glb.py 400
```

```bash
python tests/test_curves.py
```

```bash
python tests/fuzz/fuzz_reference.py 300
```

```bash
python tests/test_naturalness.py
```

```bash
python tests/fuzz/fuzz_curves.py 150
```

```bash
python tests/test_animals.py
```

```bash
python tests/fuzz/mutate.py
```

`mutate.py` plants known bugs in a scratch copy and checks the fuzzers catch each one. Set `MOTIONLAB_NO_CURVES=1` to
run any test without the curves file (the summary gait).

The Blender add-on has a headless test (`tests/blender/test_addon.py`, run it with `BLENDER_USER_RESOURCES` pointing at
an empty folder so it never touches your own Blender setup).

## Later: live in Meitou

The same driver can run every frame in Meitou instead of baking: a Meitou framework mod with a `maestro.json`
descriptor, where authors assign leg / wing bones in MeitouMaestro and the creature animates live. `<name>.motion.json`
is the bone-role map that descriptor starts from.

## Credits

- Kenshi `.mesh` / `.skeleton` export: the **Kenshi IO Continued** Blender add-on by Kindrad (GPLv3, on Codeberg).
- Kenshi is by Lo-Fi Games. This tool ships no game files.
