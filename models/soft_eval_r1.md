# Soft-body evaluation (EXPERIMENTAL)

**EXPERIMENTAL; rigid boxes are PRESCRIBED from the ground-truth trajectory (teacher-forced obstacles); only the soft-body vertices are predicted.**

Checkpoint `models/runs/soft_v1/best.pt` ({'kind': 'soft_gnn', 'config': {'width': 128, 'rounds': 6, 'box_radius': 0.08, 'centre': True}, 'epoch': 16, 'params': 1141507}); data `data/soft_v1` split `test`, 752 episodes, 60 steps of 0.05 s; rollouts are seeded with the true frames (0, 1) and predict frames 2..T

Engines:
- `physics_reference`: PyBullet ground truth scored with the same metrics (floor for penetration / strain)
- `hold_initial`: trivial baseline: the last seeded frame is held
- `rigid_freefall`: trivial baseline: rigid free-fall of the seeded shape (COM velocity + gravity) until the table top / floor (+ --baseline-clearance, the collision margin), then stop
- `soft_raw`: SoftGNN alone: learned model only, no projection
- `soft_projected`: SoftGNN + non-learned project_soft after every step
- `soft_projected_margin`: SoftGNN + project_soft keeping vertices one collision margin outside surfaces
- `soft_projected_sleep`: SoftGNN + project_soft + NON-LEARNED sleep rule (a supported body whose vertices all move less than --sleep-move per step is held in place)

## all (752 episodes)

| engine | vert 0.5s (mm) | vert 1s | vert 2s | vert final | COM 1s (mm) | COM 2s | shape 2s (mm) | final-rest COM (mm) | pen capped mean (mm) | pen >5mm (%) | pen max (mm) | strain err | diverged | CPU ms/step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| physics_reference | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.00 | 0.03 | 24.3 | 0.0000 | 0 | - |
| hold_initial | 235.2 | 286.4 | 300.1 | 308.2 | 285.7 | 299.2 | 16.0 | 296.8 | 0.03 | 0.16 | 57.3 | 0.0094 | 0 | 0.00 |
| rigid_freefall | 146.0 | 170.2 | 172.3 | 177.4 | 168.9 | 170.3 | 16.0 | 169.2 | 0.59 | 3.17 | 78.7 | 0.0094 | 0 | 0.28 |
| soft_raw | 24.6 | 38.4 | 47.2 | 55.1 | 36.8 | 45.2 | 9.9 | 46.6 | 0.01 | 0.09 | 51.8 | 0.0121 | 0 | 13.34 |
| soft_projected | 24.5 | 38.4 | 47.4 | 54.9 | 36.9 | 45.5 | 9.9 | 46.4 | 0.00 | 0.00 | 2.2 | 0.0125 | 0 | 17.44 |
| soft_projected_margin | 24.5 | 38.5 | 46.8 | 55.5 | 37.0 | 44.9 | 9.8 | 47.0 | 0.00 | 0.00 | 7.3 | 0.0128 | 0 | 17.41 |
| soft_projected_sleep | 24.5 | 38.6 | 47.9 | 55.3 | 37.1 | 45.9 | 10.2 | 46.9 | 0.00 | 0.00 | 2.2 | 0.0124 | 0 | 18.85 |

## kind:box_on_soft (123 episodes)

| engine | vert 0.5s (mm) | vert 1s | vert 2s | vert final | COM 1s (mm) | COM 2s | shape 2s (mm) | final-rest COM (mm) | pen capped mean (mm) | pen >5mm (%) | pen max (mm) | strain err | diverged | CPU ms/step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| physics_reference | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.00 | 0.02 | 13.7 | 0.0000 | 0 | - |
| hold_initial | 30.0 | 38.2 | 41.9 | 42.7 | 36.9 | 40.5 | 8.7 | 41.7 | 0.18 | 0.99 | 57.3 | 0.0076 | 0 | 0.00 |
| rigid_freefall | 30.0 | 38.2 | 41.9 | 42.7 | 36.9 | 40.5 | 8.7 | 41.7 | 0.18 | 1.01 | 57.3 | 0.0076 | 0 | 0.28 |
| soft_raw | 23.6 | 37.3 | 42.3 | 41.7 | 35.7 | 40.7 | 8.8 | 40.7 | 0.04 | 0.26 | 51.8 | 0.0129 | 0 | 13.34 |
| soft_projected | 23.5 | 37.1 | 41.9 | 42.0 | 35.5 | 40.2 | 9.0 | 41.1 | 0.00 | 0.00 | 2.2 | 0.0144 | 0 | 17.44 |
| soft_projected_margin | 23.4 | 36.6 | 40.9 | 41.3 | 35.0 | 39.2 | 9.0 | 40.3 | 0.00 | 0.01 | 7.3 | 0.0151 | 0 | 17.41 |
| soft_projected_sleep | 23.6 | 37.4 | 42.3 | 42.5 | 35.7 | 40.6 | 9.1 | 41.5 | 0.00 | 0.00 | 2.2 | 0.0152 | 0 | 18.85 |

## kind:cloth_drape_box (50 episodes)

| engine | vert 0.5s (mm) | vert 1s | vert 2s | vert final | COM 1s (mm) | COM 2s | shape 2s (mm) | final-rest COM (mm) | pen capped mean (mm) | pen >5mm (%) | pen max (mm) | strain err | diverged | CPU ms/step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| physics_reference | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.00 | 0.00 | 0.3 | 0.0000 | 0 | - |
| hold_initial | 101.3 | 113.1 | 127.2 | 135.8 | 112.3 | 125.5 | 35.1 | 125.6 | 0.00 | 0.00 | 0.0 | 0.0104 | 0 | 0.00 |
| rigid_freefall | 104.2 | 96.3 | 91.2 | 89.1 | 93.6 | 85.3 | 35.1 | 72.7 | 0.56 | 5.81 | 73.9 | 0.0104 | 0 | 0.28 |
| soft_raw | 8.1 | 13.5 | 24.8 | 33.5 | 10.8 | 20.2 | 13.5 | 17.0 | 0.00 | 0.00 | 3.1 | 0.0120 | 0 | 13.34 |
| soft_projected | 8.1 | 13.5 | 24.8 | 33.5 | 10.8 | 20.2 | 13.5 | 17.0 | 0.00 | 0.00 | 0.0 | 0.0120 | 0 | 17.44 |
| soft_projected_margin | 8.1 | 13.5 | 24.7 | 33.4 | 10.8 | 20.2 | 13.5 | 16.9 | 0.00 | 0.00 | 0.0 | 0.0121 | 0 | 17.41 |
| soft_projected_sleep | 8.2 | 14.3 | 27.3 | 37.3 | 11.6 | 22.6 | 15.4 | 21.3 | 0.00 | 0.00 | 0.8 | 0.0106 | 0 | 18.85 |

## kind:cloth_drape_edge (32 episodes)

| engine | vert 0.5s (mm) | vert 1s | vert 2s | vert final | COM 1s (mm) | COM 2s | shape 2s (mm) | final-rest COM (mm) | pen capped mean (mm) | pen >5mm (%) | pen max (mm) | strain err | diverged | CPU ms/step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| physics_reference | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.00 | 0.00 | 0.7 | 0.0000 | 0 | - |
| hold_initial | 61.6 | 78.4 | 166.9 | 224.7 | 75.4 | 162.5 | 36.4 | 193.9 | 0.00 | 0.00 | 0.0 | 0.0063 | 0 | 0.00 |
| rigid_freefall | 163.8 | 161.4 | 106.8 | 134.5 | 158.2 | 98.9 | 36.4 | 68.6 | 0.04 | 0.15 | 58.4 | 0.0063 | 0 | 0.28 |
| soft_raw | 5.9 | 14.0 | 62.4 | 100.7 | 11.7 | 58.3 | 16.6 | 41.8 | 0.00 | 0.00 | 4.6 | 0.0093 | 0 | 13.34 |
| soft_projected | 5.8 | 13.7 | 62.4 | 100.7 | 11.4 | 58.3 | 16.5 | 41.9 | 0.00 | 0.00 | 0.0 | 0.0092 | 0 | 17.44 |
| soft_projected_margin | 5.2 | 11.2 | 31.6 | 80.5 | 9.6 | 27.7 | 15.3 | 12.0 | 0.00 | 0.00 | 0.0 | 0.0090 | 0 | 17.41 |
| soft_projected_sleep | 5.9 | 14.2 | 63.9 | 103.4 | 11.9 | 59.5 | 18.1 | 44.2 | 0.00 | 0.00 | 0.0 | 0.0085 | 0 | 18.85 |

## kind:cloth_drop (32 episodes)

| engine | vert 0.5s (mm) | vert 1s | vert 2s | vert final | COM 1s (mm) | COM 2s | shape 2s (mm) | final-rest COM (mm) | pen capped mean (mm) | pen >5mm (%) | pen max (mm) | strain err | diverged | CPU ms/step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| physics_reference | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.00 | 0.00 | 1.6 | 0.0000 | 0 | - |
| hold_initial | 187.0 | 215.0 | 215.6 | 215.8 | 214.8 | 215.4 | 21.4 | 214.4 | 0.00 | 0.00 | 0.0 | 0.0006 | 0 | 0.00 |
| rigid_freefall | 69.1 | 96.4 | 96.9 | 97.1 | 95.8 | 96.3 | 21.4 | 97.0 | 0.00 | 0.00 | 0.0 | 0.0006 | 0 | 0.28 |
| soft_raw | 9.2 | 11.5 | 11.4 | 11.0 | 10.7 | 10.8 | 3.0 | 10.5 | 0.01 | 0.06 | 18.9 | 0.0089 | 0 | 13.34 |
| soft_projected | 9.2 | 11.4 | 12.5 | 12.1 | 10.7 | 11.9 | 2.9 | 11.6 | 0.00 | 0.00 | 0.0 | 0.0087 | 0 | 17.44 |
| soft_projected_margin | 9.1 | 11.3 | 12.8 | 12.4 | 10.9 | 12.4 | 2.7 | 12.1 | 0.00 | 0.00 | 0.0 | 0.0086 | 0 | 17.41 |
| soft_projected_sleep | 9.3 | 12.0 | 13.7 | 13.8 | 11.1 | 13.0 | 3.6 | 13.3 | 0.00 | 0.00 | 0.0 | 0.0094 | 0 | 18.85 |

## kind:soft_drop (176 episodes)

| engine | vert 0.5s (mm) | vert 1s | vert 2s | vert final | COM 1s (mm) | COM 2s | shape 2s (mm) | final-rest COM (mm) | pen capped mean (mm) | pen >5mm (%) | pen max (mm) | strain err | diverged | CPU ms/step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| physics_reference | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.01 | 0.09 | 24.3 | 0.0000 | 0 | - |
| hold_initial | 225.5 | 234.8 | 239.8 | 247.8 | 234.2 | 239.0 | 18.3 | 238.8 | 0.00 | 0.00 | 0.0 | 0.0111 | 0 | 0.00 |
| rigid_freefall | 37.0 | 45.4 | 51.9 | 60.6 | 43.4 | 49.4 | 18.3 | 51.2 | 0.00 | 0.00 | 1.9 | 0.0111 | 0 | 0.28 |
| soft_raw | 15.3 | 18.9 | 24.6 | 33.1 | 17.4 | 22.7 | 9.5 | 25.4 | 0.00 | 0.02 | 10.1 | 0.0116 | 0 | 13.34 |
| soft_projected | 15.0 | 18.8 | 24.3 | 32.9 | 17.3 | 22.5 | 9.5 | 25.2 | 0.00 | 0.00 | 0.0 | 0.0116 | 0 | 17.44 |
| soft_projected_margin | 15.0 | 18.7 | 24.4 | 32.9 | 17.1 | 22.5 | 9.4 | 25.2 | 0.00 | 0.00 | 0.0 | 0.0118 | 0 | 17.41 |
| soft_projected_sleep | 15.1 | 19.0 | 24.8 | 33.4 | 17.5 | 22.9 | 9.7 | 25.7 | 0.00 | 0.00 | 0.0 | 0.0121 | 0 | 18.85 |

## kind:soft_edge_fall (77 episodes)

| engine | vert 0.5s (mm) | vert 1s | vert 2s | vert final | COM 1s (mm) | COM 2s | shape 2s (mm) | final-rest COM (mm) | pen capped mean (mm) | pen >5mm (%) | pen max (mm) | strain err | diverged | CPU ms/step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| physics_reference | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.00 | 0.02 | 16.0 | 0.0000 | 0 | - |
| hold_initial | 659.1 | 849.1 | 877.1 | 883.5 | 848.7 | 876.7 | 15.0 | 855.4 | 0.00 | 0.00 | 0.0 | 0.0103 | 0 | 0.00 |
| rigid_freefall | 496.3 | 495.1 | 487.6 | 486.6 | 494.6 | 486.9 | 15.0 | 483.7 | 0.04 | 0.19 | 77.7 | 0.0103 | 0 | 0.28 |
| soft_raw | 63.0 | 81.7 | 90.9 | 97.3 | 80.4 | 89.1 | 11.4 | 98.9 | 0.02 | 0.18 | 13.6 | 0.0144 | 0 | 13.34 |
| soft_projected | 62.6 | 82.4 | 91.4 | 97.5 | 81.1 | 89.7 | 11.4 | 98.7 | 0.00 | 0.00 | 0.0 | 0.0144 | 0 | 17.44 |
| soft_projected_margin | 62.6 | 81.8 | 91.3 | 97.3 | 80.5 | 89.6 | 11.4 | 98.5 | 0.00 | 0.00 | 0.0 | 0.0149 | 0 | 17.41 |
| soft_projected_sleep | 62.7 | 82.4 | 91.3 | 97.2 | 81.1 | 89.5 | 11.3 | 98.3 | 0.00 | 0.00 | 0.0 | 0.0142 | 0 | 18.85 |

## kind:soft_launch (126 episodes)

| engine | vert 0.5s (mm) | vert 1s | vert 2s | vert final | COM 1s (mm) | COM 2s | shape 2s (mm) | final-rest COM (mm) | pen capped mean (mm) | pen >5mm (%) | pen max (mm) | strain err | diverged | CPU ms/step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| physics_reference | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.00 | 0.02 | 16.1 | 0.0000 | 0 | - |
| hold_initial | 362.9 | 466.4 | 470.6 | 474.8 | 466.1 | 470.2 | 9.5 | 458.0 | 0.00 | 0.01 | 5.8 | 0.0104 | 0 | 0.00 |
| rigid_freefall | 297.3 | 396.7 | 400.9 | 405.1 | 396.3 | 400.4 | 9.5 | 395.6 | 0.00 | 0.00 | 5.6 | 0.0104 | 0 | 0.28 |
| soft_raw | 28.5 | 41.3 | 44.2 | 51.4 | 39.9 | 42.3 | 9.0 | 47.6 | 0.01 | 0.09 | 16.5 | 0.0119 | 0 | 13.34 |
| soft_projected | 28.5 | 41.1 | 44.0 | 51.2 | 39.7 | 42.2 | 9.0 | 47.5 | 0.00 | 0.00 | 0.0 | 0.0120 | 0 | 17.44 |
| soft_projected_margin | 29.0 | 41.4 | 43.7 | 50.9 | 40.0 | 41.9 | 9.2 | 47.2 | 0.00 | 0.00 | 0.0 | 0.0125 | 0 | 17.41 |
| soft_projected_sleep | 28.5 | 41.1 | 44.1 | 51.3 | 39.7 | 42.2 | 9.1 | 47.5 | 0.00 | 0.00 | 0.0 | 0.0123 | 0 | 18.85 |

## kind:soft_on_boxes (86 episodes)

| engine | vert 0.5s (mm) | vert 1s | vert 2s | vert final | COM 1s (mm) | COM 2s | shape 2s (mm) | final-rest COM (mm) | pen capped mean (mm) | pen >5mm (%) | pen max (mm) | strain err | diverged | CPU ms/step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| physics_reference | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.00 | 0.01 | 16.2 | 0.0000 | 0 | - |
| hold_initial | 277.6 | 348.0 | 378.5 | 392.5 | 347.6 | 378.0 | 18.6 | 401.8 | 0.00 | 0.00 | 0.0 | 0.0123 | 0 | 0.00 |
| rigid_freefall | 129.4 | 161.5 | 183.6 | 194.0 | 160.3 | 182.1 | 18.6 | 184.2 | 4.50 | 22.65 | 78.7 | 0.0123 | 0 | 0.28 |
| soft_raw | 40.3 | 90.3 | 110.0 | 126.0 | 88.4 | 108.2 | 13.7 | 121.4 | 0.01 | 0.05 | 32.5 | 0.0159 | 0 | 13.34 |
| soft_projected | 40.3 | 90.6 | 112.5 | 124.3 | 88.8 | 110.8 | 13.3 | 118.7 | 0.00 | 0.00 | 0.0 | 0.0166 | 0 | 17.44 |
| soft_projected_margin | 40.3 | 93.4 | 120.0 | 137.9 | 91.8 | 118.4 | 12.7 | 137.0 | 0.00 | 0.00 | 0.0 | 0.0167 | 0 | 17.41 |
| soft_projected_sleep | 40.3 | 90.7 | 112.6 | 121.8 | 88.9 | 110.9 | 13.3 | 118.9 | 0.00 | 0.00 | 1.5 | 0.0163 | 0 | 18.85 |

## kind:soft_rest (50 episodes)

| engine | vert 0.5s (mm) | vert 1s | vert 2s | vert final | COM 1s (mm) | COM 2s | shape 2s (mm) | final-rest COM (mm) | pen capped mean (mm) | pen >5mm (%) | pen max (mm) | strain err | diverged | CPU ms/step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| physics_reference | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.00 | 0.01 | 11.3 | 0.0000 | 0 | - |
| hold_initial | 2.5 | 4.7 | 7.1 | 8.7 | 4.2 | 6.4 | 4.3 | 8.0 | 0.00 | 0.00 | 0.0 | 0.0047 | 0 | 0.00 |
| rigid_freefall | 2.5 | 4.7 | 7.1 | 8.7 | 4.2 | 6.4 | 4.3 | 8.0 | 0.00 | 0.00 | 0.0 | 0.0047 | 0 | 0.28 |
| soft_raw | 2.0 | 4.1 | 6.8 | 8.6 | 3.5 | 5.9 | 4.3 | 7.7 | 0.00 | 0.00 | 0.0 | 0.0068 | 0 | 13.34 |
| soft_projected | 2.0 | 4.1 | 6.8 | 8.6 | 3.5 | 5.9 | 4.3 | 7.7 | 0.00 | 0.00 | 0.0 | 0.0068 | 0 | 17.44 |
| soft_projected_margin | 2.0 | 4.1 | 6.7 | 8.6 | 3.5 | 5.8 | 4.3 | 7.7 | 0.00 | 0.00 | 0.0 | 0.0069 | 0 | 17.41 |
| soft_projected_sleep | 2.1 | 4.2 | 6.6 | 8.2 | 3.7 | 5.9 | 4.1 | 7.6 | 0.00 | 0.00 | 0.0 | 0.0044 | 0 | 18.85 |

## shape:ball (243 episodes)

| engine | vert 0.5s (mm) | vert 1s | vert 2s | vert final | COM 1s (mm) | COM 2s | shape 2s (mm) | final-rest COM (mm) | pen capped mean (mm) | pen >5mm (%) | pen max (mm) | strain err | diverged | CPU ms/step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| physics_reference | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.01 | 0.08 | 24.3 | 0.0000 | 0 | - |
| hold_initial | 251.9 | 298.3 | 305.3 | 308.8 | 297.4 | 304.2 | 15.5 | 294.9 | 0.05 | 0.26 | 55.9 | 0.0113 | 0 | 0.00 |
| rigid_freefall | 140.5 | 176.7 | 181.0 | 185.2 | 175.3 | 179.2 | 15.5 | 181.8 | 0.70 | 3.46 | 78.7 | 0.0113 | 0 | 0.28 |
| soft_raw | 28.4 | 48.0 | 50.9 | 56.8 | 45.9 | 48.1 | 13.6 | 55.0 | 0.01 | 0.09 | 32.5 | 0.0141 | 0 | 13.34 |
| soft_projected | 28.1 | 47.9 | 50.6 | 56.6 | 45.8 | 47.8 | 13.5 | 54.5 | 0.00 | 0.00 | 0.0 | 0.0144 | 0 | 17.44 |
| soft_projected_margin | 28.4 | 48.5 | 51.7 | 57.8 | 46.4 | 49.0 | 13.5 | 55.9 | 0.00 | 0.00 | 0.0 | 0.0149 | 0 | 17.41 |
| soft_projected_sleep | 28.2 | 48.1 | 50.7 | 56.7 | 45.9 | 47.9 | 13.6 | 54.6 | 0.00 | 0.00 | 1.5 | 0.0146 | 0 | 18.85 |

## shape:cloth (114 episodes)

| engine | vert 0.5s (mm) | vert 1s | vert 2s | vert final | COM 1s (mm) | COM 2s | shape 2s (mm) | final-rest COM (mm) | pen capped mean (mm) | pen >5mm (%) | pen max (mm) | strain err | diverged | CPU ms/step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| physics_reference | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.00 | 0.00 | 1.6 | 0.0000 | 0 | - |
| hold_initial | 114.2 | 132.0 | 163.2 | 183.2 | 130.7 | 161.1 | 31.6 | 175.2 | 0.00 | 0.00 | 0.0 | 0.0065 | 0 | 0.00 |
| rigid_freefall | 111.1 | 114.6 | 97.2 | 104.1 | 112.4 | 92.2 | 31.6 | 80.8 | 0.26 | 2.59 | 73.9 | 0.0065 | 0 | 0.28 |
| soft_raw | 7.8 | 13.1 | 31.6 | 46.1 | 11.0 | 28.3 | 11.4 | 20.5 | 0.00 | 0.02 | 18.9 | 0.0104 | 0 | 13.34 |
| soft_projected | 7.8 | 13.0 | 31.9 | 46.4 | 11.0 | 28.6 | 11.4 | 21.0 | 0.00 | 0.00 | 0.0 | 0.0103 | 0 | 17.44 |
| soft_projected_margin | 7.6 | 12.2 | 23.3 | 40.7 | 10.5 | 20.1 | 11.0 | 13.9 | 0.00 | 0.00 | 0.0 | 0.0103 | 0 | 17.41 |
| soft_projected_sleep | 7.9 | 13.6 | 33.8 | 49.3 | 11.5 | 30.3 | 12.8 | 23.8 | 0.00 | 0.00 | 0.8 | 0.0097 | 0 | 18.85 |

## shape:cube (395 episodes)

| engine | vert 0.5s (mm) | vert 1s | vert 2s | vert final | COM 1s (mm) | COM 2s | shape 2s (mm) | final-rest COM (mm) | pen capped mean (mm) | pen >5mm (%) | pen max (mm) | strain err | diverged | CPU ms/step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| physics_reference | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.00 | 0.01 | 22.8 | 0.0000 | 0 | - |
| hold_initial | 259.9 | 323.7 | 336.5 | 344.0 | 323.2 | 336.0 | 11.9 | 326.3 | 0.02 | 0.15 | 57.3 | 0.0090 | 0 | 0.00 |
| rigid_freefall | 159.5 | 182.2 | 188.6 | 193.8 | 181.2 | 187.5 | 11.9 | 181.5 | 0.61 | 3.16 | 77.7 | 0.0090 | 0 | 0.28 |
| soft_raw | 27.1 | 39.7 | 49.4 | 56.6 | 38.7 | 48.4 | 7.2 | 47.3 | 0.02 | 0.11 | 51.8 | 0.0114 | 0 | 13.34 |
| soft_projected | 27.0 | 39.8 | 50.0 | 56.4 | 38.8 | 49.0 | 7.2 | 47.1 | 0.00 | 0.00 | 2.2 | 0.0119 | 0 | 17.44 |
| soft_projected_margin | 27.0 | 39.9 | 50.5 | 58.3 | 38.9 | 49.5 | 7.1 | 48.9 | 0.00 | 0.00 | 7.3 | 0.0122 | 0 | 17.41 |
| soft_projected_sleep | 27.1 | 40.0 | 50.2 | 56.1 | 39.0 | 49.2 | 7.3 | 47.3 | 0.00 | 0.00 | 2.2 | 0.0119 | 0 | 18.85 |

## with_rigid_box (259 episodes)

| engine | vert 0.5s (mm) | vert 1s | vert 2s | vert final | COM 1s (mm) | COM 2s | shape 2s (mm) | final-rest COM (mm) | pen capped mean (mm) | pen >5mm (%) | pen max (mm) | strain err | diverged | CPU ms/step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| physics_reference | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.00 | 0.01 | 16.2 | 0.0000 | 0 | - |
| hold_initial | 126.0 | 155.5 | 170.1 | 176.8 | 154.6 | 169.0 | 17.1 | 158.6 | 0.08 | 0.47 | 57.3 | 0.0097 | 0 | 0.00 |
| rigid_freefall | 77.3 | 90.4 | 98.5 | 101.9 | 88.8 | 96.2 | 17.1 | 87.6 | 1.69 | 9.12 | 78.7 | 0.0097 | 0 | 0.28 |
| soft_raw | 26.1 | 50.3 | 61.4 | 68.1 | 48.4 | 59.1 | 11.3 | 60.5 | 0.02 | 0.14 | 51.8 | 0.0137 | 0 | 13.34 |
| soft_projected | 26.1 | 50.3 | 62.1 | 67.7 | 48.4 | 59.8 | 11.3 | 59.9 | 0.00 | 0.00 | 2.2 | 0.0147 | 0 | 17.44 |
| soft_projected_margin | 26.1 | 51.0 | 64.0 | 71.9 | 49.2 | 61.8 | 11.1 | 64.8 | 0.00 | 0.01 | 7.3 | 0.0150 | 0 | 17.41 |
| soft_projected_sleep | 26.2 | 50.6 | 62.8 | 67.8 | 48.7 | 60.5 | 11.7 | 60.9 | 0.00 | 0.00 | 2.2 | 0.0147 | 0 | 18.85 |

## no_rigid_box (493 episodes)

| engine | vert 0.5s (mm) | vert 1s | vert 2s | vert final | COM 1s (mm) | COM 2s | shape 2s (mm) | final-rest COM (mm) | pen capped mean (mm) | pen >5mm (%) | pen max (mm) | strain err | diverged | CPU ms/step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| physics_reference | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.01 | 0.04 | 24.3 | 0.0000 | 0 | - |
| hold_initial | 292.6 | 355.1 | 368.4 | 377.3 | 354.5 | 367.6 | 15.5 | 361.4 | 0.00 | 0.00 | 5.8 | 0.0092 | 0 | 0.00 |
| rigid_freefall | 182.1 | 212.1 | 211.1 | 217.1 | 210.9 | 209.3 | 15.5 | 207.3 | 0.01 | 0.04 | 77.7 | 0.0092 | 0 | 0.28 |
| soft_raw | 23.8 | 32.1 | 39.7 | 48.3 | 30.8 | 37.9 | 9.2 | 40.1 | 0.01 | 0.06 | 18.9 | 0.0113 | 0 | 13.34 |
| soft_projected | 23.6 | 32.1 | 39.8 | 48.2 | 30.8 | 38.0 | 9.1 | 40.1 | 0.00 | 0.00 | 0.0 | 0.0113 | 0 | 17.44 |
| soft_projected_margin | 23.7 | 31.9 | 37.7 | 46.9 | 30.6 | 36.0 | 9.1 | 38.7 | 0.00 | 0.00 | 0.0 | 0.0116 | 0 | 17.41 |
| soft_projected_sleep | 23.7 | 32.3 | 40.1 | 48.7 | 30.9 | 38.2 | 9.4 | 40.4 | 0.00 | 0.00 | 0.0 | 0.0113 | 0 | 18.85 |

Notes:
- EXPERIMENTAL research prototype. rigid boxes are PRESCRIBED from the ground-truth trajectory (teacher-forced obstacles); only the soft-body vertices are predicted in every engine.
- COM is the unweighted mean of the soft vertices; shape error removes each trajectory's COM translation (no rotation alignment).
- Penetration and strain error are averaged over the predicted frames only; physics_reference shows what the ground truth itself scores (PyBullet soft nodes hover about one collision margin above surfaces, so its penetration is near zero).
- The penetration column is the mean vertex depth with each vertex capped at 50 mm (one fall-through rollout cannot dominate); the JSON also holds the uncapped mean.
- final-rest COM error is the last-frame COM error over episodes whose ground truth is at rest at the last frame.
- Pinned (kinematic) vertices, if the data has any, are prescribed from the ground truth in the model engines and count in the vertex means (with zero error there); the trivial baselines do not move them either way.
- CPU ms/step: batch 1, first episode of the split (64 vertices), 8 torch threads, float32, median over 40 autoregressive steps.
