# PZA's Kenshi Motion Lab

Author: PZA. Licence: GPLv3 (see `LICENSE`). **Status: beta (0.1.0).** It works end to end on our test creatures; it
has not been tried in Kenshi itself yet (see *Known gaps*). Feedback and broken models are welcome as issues.

Rig and animate a model **from code** instead of hand-animating it. Give it a model (a Meshy / Blender `.glb`, or a
Kenshi `.mesh` + `.skeleton`) and it:

1. **finds the body plan** from the mesh shape: legs, wings, antennae, head, thorax, abdomen;
2. **places a clean, named skeleton** and binds the mesh (3 weights per vertex, like Kenshi);
3. **drives the bones from code**: a foot-planting gait for any number of legs (tripod for six, trot for four),
   a wingbeat, wings that **shrink away on landing and grow back on take-off (0.25 s)**, springy antennae and abdomen;
4. **bakes clips** (idle, walk, run, hover, takeoff, land, attack = a bite, sting) into a new `.glb`, with a report and preview GIFs.

Only the `insect` body plan exists today (six-legged, winged creatures and their relatives). Bipeds (humans) and
quadrupeds are planned: the skinning, driver, springs, IK and baking are already plan-independent.

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

Options for `rig`: `--ground-wings hidden|folded` (default hidden: wings vanish on the ground),
`--leg-bones 4 --wing-bones 2 --antenna-bones 2 --abdomen-bones 3`, `--clips walk,idle`, `--fps 30`,
`--skeleton file.skeleton` for a `.mesh`, `--landmarks edited.json`.

## How it works

**Frame.** Up is +Y. The body's long axis is turned to face +Z (glTF's forward), centred on the leg pairs, and its
lowest point is put on the ground (y = 0). The report says how much it turned (a Meshy model we tested was turned about 22
degrees in its file, which made its own rig look lopsided).

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
- *Lifted legs*: a leg whose tip rests off the ground (the bee's raised hind leg) rides along and does not plant.
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
- One body plan (`insect`); see *Limits*.

## Limits

- One body plan (`insect`). On Kenshi's crab it finds the four walking legs but takes the claws for wings and
  antennae: fix with landmark edits until a `crab` / `quadruped` / `biped` plan exists.
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
python tests/fuzz/fuzz_glb.py 400
```

The Blender add-on has a headless test (`tests/blender/test_addon.py`, run it with `BLENDER_USER_RESOURCES` pointing at
an empty folder so it never touches your own Blender setup).

## Later: live in Meitou

The same driver can run every frame in Meitou instead of baking: a Meitou framework mod with a `maestro.json`
descriptor, where authors assign leg / wing bones in MeitouMaestro and the creature animates live. `<name>.motion.json`
is the bone-role map that descriptor starts from.

## Credits

- Kenshi `.mesh` / `.skeleton` export: the **Kenshi IO Continued** Blender add-on by Kindrad (GPLv3, on Codeberg).
- Kenshi is by Lo-Fi Games. This tool ships no game files.
