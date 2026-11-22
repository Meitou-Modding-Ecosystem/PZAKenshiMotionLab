# Polygon budgets for Kenshi creatures and gear

Author: PZA. Measured 2026-11-22 from the game's own files (Kenshi 1.0.68, `data\animal`, `data\character`,
`data\items`): the triangle count of every vanilla `.mesh`, summed over its submeshes. Use it to pick a target before
you decimate a model for Kenshi or Meitou.

## What the game itself uses

| Vanilla meshes | Count | Median | 90% are under | Largest |
|---|---|---|---|---|
| **Animals / creatures** | 35 | **6,072** | 8,960 | Big-Bones 26,076; Crimper 22,740; pack beast 9,316 |
| Bodies, robots, skeletons | 72 | 3,034 | 12,851 | Skeleton MkII (Screamer) 24,578 |
| Armour | 222 | 1,208 | 4,566 | Iron Clad Jacket (female) 14,087; Hack Stopper Jacket 13,401 |
| Hair | 108 | 1,272 | 2,460 | 6,424 |
| Weapons | 112 | 314 | 1,192 | Spring-Bat 2,818 |

All counts are triangles. Kenshi's own crab is 8,322.

## Targets

- **A normal creature: 4,000 to 8,000 triangles.** That is where nearly every vanilla animal sits.
- **A large or boss creature: up to about 14,000.** Only two vanilla animals go past 20,000 (Big-Bones and the Crimper).
- **Worn gear:** about 1,000 to 5,000 per piece; the heaviest vanilla armour is 14,000.
- **Weapons:** under 1,500; the largest vanilla weapon is under 3,000.

A model generated at 20,000+ triangles (typical for AI mesh tools) will work but sits with the game's very heaviest
meshes. Decimating it to 8,000 to 10,000 puts it with the normal animals.

## The hard limits are bones and weights, not triangles

- **60 bones per submesh**: Kenshi's skin shader (`data\materials\deferred\skin.hlsl`, `#define BONES 60`). The rig
  report's "bones per submesh" row checks it.
- **3 weights per vertex**: the GPU reads 3; a 4th is ignored. `motionlab rig` prunes to 3 and reports the most weight
  any vertex lost.
- Vertex count is higher than the triangle count suggests: UV seams and hard edges split vertices, so a 22,000-triangle
  model can carry 48,000 vertices. That is normal.

## How the numbers were made

A small reader walked every `.mesh` under the folders above and summed each submesh's index count / 3 (Kenshi meshes
are triangle lists): the full-detail mesh only, not its lower LOD levels. Meshes with no triangles were left out. Nothing from the game is copied here: only
the counts and file names.
