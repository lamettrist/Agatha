# Soft-body evaluation (EXPERIMENTAL)

**EXPERIMENTAL; rigid boxes are PRESCRIBED from the ground-truth trajectory (teacher-forced obstacles); only the soft-body vertices are predicted.**

Checkpoint `models/runs/soft_v1/best.pt` ({'kind': 'soft_gnn', 'config': {'width': 128, 'rounds': 6, 'box_radius': 0.08, 'centre': True}, 'epoch': 16, 'params': 1141507}); data `data/external_v2/meshgraphnets_soft` split `test`, 10 episodes, 60 steps of 0.05 s; rollouts are seeded with the true frames (0, 1) and predict frames 2..T

Engines:
- `physics_reference`: PyBullet ground truth scored with the same metrics (floor for penetration / strain)
- `hold_initial`: trivial baseline: the last seeded frame is held
- `rigid_freefall`: trivial baseline: rigid free-fall of the seeded shape (COM velocity + gravity) until the table top / floor (+ --baseline-clearance, the collision margin), then stop
- `soft_raw`: SoftGNN alone: learned model only, no projection
- `soft_projected`: SoftGNN + non-learned project_soft after every step
- `soft_projected_margin`: SoftGNN + project_soft keeping vertices one collision margin outside surfaces
- `soft_projected_sleep`: SoftGNN + project_soft + NON-LEARNED sleep rule (a supported body whose vertices all move less than --sleep-move per step is held in place)

## all (10 episodes)

| engine | vert 0.5s (mm) | vert 1s | vert 2s | vert final | COM 1s (mm) | COM 2s | shape 2s (mm) | final-rest COM (mm) | pen capped mean (mm) | pen >5mm (%) | pen max (mm) | strain err | diverged | CPU ms/step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| physics_reference | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | - | 0.00 | 0.00 | 0.0 | 0.0000 | 0 | - |
| hold_initial | 282.8 | 256.9 | 269.6 | 288.9 | 255.7 | 268.2 | 139.3 | - | 0.00 | 0.01 | 26.2 | 0.0249 | 0 | 0.00 |
| rigid_freefall | 739.9 | 722.0 | 726.8 | 719.0 | 717.6 | 719.6 | 139.3 | - | 0.01 | 0.04 | 23.2 | 0.0249 | 0 | 0.28 |
| soft_raw | 9.8 | 8.7 | 9.1 | 19.0 | 4.1 | 4.7 | 7.8 | - | 0.01 | 0.06 | 18.0 | 0.0267 | 0 | 14.52 |
| soft_projected | 9.7 | 8.0 | 8.5 | 19.1 | 3.6 | 4.0 | 7.5 | - | 0.00 | 0.00 | 0.0 | 0.0265 | 0 | 18.74 |
| soft_projected_margin | 9.5 | 8.3 | 8.4 | 18.5 | 3.8 | 4.1 | 7.4 | - | 0.00 | 0.00 | 0.0 | 0.0268 | 0 | 19.04 |
| soft_projected_sleep | 9.7 | 8.0 | 8.5 | 19.2 | 3.6 | 4.0 | 7.5 | - | 0.00 | 0.00 | 0.0 | 0.0265 | 0 | 20.03 |

## kind:mgn_sphere_simple (10 episodes)

| engine | vert 0.5s (mm) | vert 1s | vert 2s | vert final | COM 1s (mm) | COM 2s | shape 2s (mm) | final-rest COM (mm) | pen capped mean (mm) | pen >5mm (%) | pen max (mm) | strain err | diverged | CPU ms/step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| physics_reference | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | - | 0.00 | 0.00 | 0.0 | 0.0000 | 0 | - |
| hold_initial | 282.8 | 256.9 | 269.6 | 288.9 | 255.7 | 268.2 | 139.3 | - | 0.00 | 0.01 | 26.2 | 0.0249 | 0 | 0.00 |
| rigid_freefall | 739.9 | 722.0 | 726.8 | 719.0 | 717.6 | 719.6 | 139.3 | - | 0.01 | 0.04 | 23.2 | 0.0249 | 0 | 0.28 |
| soft_raw | 9.8 | 8.7 | 9.1 | 19.0 | 4.1 | 4.7 | 7.8 | - | 0.01 | 0.06 | 18.0 | 0.0267 | 0 | 14.52 |
| soft_projected | 9.7 | 8.0 | 8.5 | 19.1 | 3.6 | 4.0 | 7.5 | - | 0.00 | 0.00 | 0.0 | 0.0265 | 0 | 18.74 |
| soft_projected_margin | 9.5 | 8.3 | 8.4 | 18.5 | 3.8 | 4.1 | 7.4 | - | 0.00 | 0.00 | 0.0 | 0.0268 | 0 | 19.04 |
| soft_projected_sleep | 9.7 | 8.0 | 8.5 | 19.2 | 3.6 | 4.0 | 7.5 | - | 0.00 | 0.00 | 0.0 | 0.0265 | 0 | 20.03 |

## shape:cloth (10 episodes)

| engine | vert 0.5s (mm) | vert 1s | vert 2s | vert final | COM 1s (mm) | COM 2s | shape 2s (mm) | final-rest COM (mm) | pen capped mean (mm) | pen >5mm (%) | pen max (mm) | strain err | diverged | CPU ms/step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| physics_reference | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | - | 0.00 | 0.00 | 0.0 | 0.0000 | 0 | - |
| hold_initial | 282.8 | 256.9 | 269.6 | 288.9 | 255.7 | 268.2 | 139.3 | - | 0.00 | 0.01 | 26.2 | 0.0249 | 0 | 0.00 |
| rigid_freefall | 739.9 | 722.0 | 726.8 | 719.0 | 717.6 | 719.6 | 139.3 | - | 0.01 | 0.04 | 23.2 | 0.0249 | 0 | 0.28 |
| soft_raw | 9.8 | 8.7 | 9.1 | 19.0 | 4.1 | 4.7 | 7.8 | - | 0.01 | 0.06 | 18.0 | 0.0267 | 0 | 14.52 |
| soft_projected | 9.7 | 8.0 | 8.5 | 19.1 | 3.6 | 4.0 | 7.5 | - | 0.00 | 0.00 | 0.0 | 0.0265 | 0 | 18.74 |
| soft_projected_margin | 9.5 | 8.3 | 8.4 | 18.5 | 3.8 | 4.1 | 7.4 | - | 0.00 | 0.00 | 0.0 | 0.0268 | 0 | 19.04 |
| soft_projected_sleep | 9.7 | 8.0 | 8.5 | 19.2 | 3.6 | 4.0 | 7.5 | - | 0.00 | 0.00 | 0.0 | 0.0265 | 0 | 20.03 |

## with_rigid_box (10 episodes)

| engine | vert 0.5s (mm) | vert 1s | vert 2s | vert final | COM 1s (mm) | COM 2s | shape 2s (mm) | final-rest COM (mm) | pen capped mean (mm) | pen >5mm (%) | pen max (mm) | strain err | diverged | CPU ms/step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| physics_reference | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | - | 0.00 | 0.00 | 0.0 | 0.0000 | 0 | - |
| hold_initial | 282.8 | 256.9 | 269.6 | 288.9 | 255.7 | 268.2 | 139.3 | - | 0.00 | 0.01 | 26.2 | 0.0249 | 0 | 0.00 |
| rigid_freefall | 739.9 | 722.0 | 726.8 | 719.0 | 717.6 | 719.6 | 139.3 | - | 0.01 | 0.04 | 23.2 | 0.0249 | 0 | 0.28 |
| soft_raw | 9.8 | 8.7 | 9.1 | 19.0 | 4.1 | 4.7 | 7.8 | - | 0.01 | 0.06 | 18.0 | 0.0267 | 0 | 14.52 |
| soft_projected | 9.7 | 8.0 | 8.5 | 19.1 | 3.6 | 4.0 | 7.5 | - | 0.00 | 0.00 | 0.0 | 0.0265 | 0 | 18.74 |
| soft_projected_margin | 9.5 | 8.3 | 8.4 | 18.5 | 3.8 | 4.1 | 7.4 | - | 0.00 | 0.00 | 0.0 | 0.0268 | 0 | 19.04 |
| soft_projected_sleep | 9.7 | 8.0 | 8.5 | 19.2 | 3.6 | 4.0 | 7.5 | - | 0.00 | 0.00 | 0.0 | 0.0265 | 0 | 20.03 |

Notes:
- EXPERIMENTAL research prototype. rigid boxes are PRESCRIBED from the ground-truth trajectory (teacher-forced obstacles); only the soft-body vertices are predicted in every engine.
- COM is the unweighted mean of the soft vertices; shape error removes each trajectory's COM translation (no rotation alignment).
- Penetration and strain error are averaged over the predicted frames only; physics_reference shows what the ground truth itself scores (PyBullet soft nodes hover about one collision margin above surfaces, so its penetration is near zero).
- The penetration column is the mean vertex depth with each vertex capped at 50 mm (one fall-through rollout cannot dominate); the JSON also holds the uncapped mean.
- final-rest COM error is the last-frame COM error over episodes whose ground truth is at rest at the last frame.
- Pinned (kinematic) vertices, if the data has any, are prescribed from the ground truth in the model engines and count in the vertex means (with zero error there); the trivial baselines do not move them either way.
- 2.0 pinned vertices per episode on average.
- CPU ms/step: batch 1, first episode of the split (128 vertices), 8 torch threads, float32, median over 40 autoregressive steps.
