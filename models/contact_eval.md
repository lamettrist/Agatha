# Contact-v2 evaluation

Data `data/v2` split `test`: 4998 episodes, 4 s autoregressive rollouts (live_physics 2468, chaotic 2350, gentle 2648).

Engines:
- `physics_reference`: PyBullet ground truth scored with the same metrics (floor for penetration / rest drift)
- `old_mlp`: old shipped model alone (mlp_multistep)
- `old_graph`: old graph_impulse model alone
- `old_mlp_projected`: old mlp_multistep + the non-learned contact projection (attribution only)
- `new_raw`: ContactGNN alone: learned model only, no projection
- `new_projected`: ContactGNN + non-learned contact projection after every step (configuration that would ship)

## live_physics / all (2468 episodes)

| engine | pos 0.5s (m) | pos 1s | pos 2s | pos 4s | pos 2s active | ori 2s (deg) | fell acc | fell P | fell R | pen mean (mm) | pen capped (mm) | pen >5mm (%) | pen max (mm) | rest drift (mm/s) | diverged | out of bounds | ms/step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| physics_reference | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00 | 1.000 | 1.000 | 1.000 | 0.12 | 0.11 | 0.29 | 79.9 | 0.26 | 0 | 0 | - |
| old_mlp | 0.0895 | 0.1700 | 0.2764 | 0.4923 | 0.5108 | 34.91 | 0.853 | 0.508 | 0.797 | 47.82 | 20.12 | 64.03 | 32367.3 | 121.44 | 0 | 167 | 12.13 |
| old_graph | 0.1869 | 0.4132 | 0.8058 | 1.4885 | 1.1658 | 102.60 | 0.688 | 0.289 | 0.732 | 257.16 | 18.28 | 53.75 | 61658.8 | 446.61 | 0 | 173 | 24.95 |
| old_mlp_projected | 0.0773 | 0.1485 | 0.2748 | 0.5250 | 0.5740 | 31.69 | 0.865 | 0.537 | 0.768 | 0.22 | 0.21 | 0.89 | 80.0 | 128.02 | 0 | 61 | 123.76 |
| new_raw | 0.0240 | 0.0434 | 0.0601 | 0.0643 | 0.1630 | 14.61 | 0.979 | 0.923 | 0.938 | 0.39 | 0.38 | 0.93 | 79.9 | 6.03 | 0 | 0 | 370.63 |
| new_projected | 0.0252 | 0.0469 | 0.0637 | 0.0672 | 0.1728 | 15.00 | 0.978 | 0.916 | 0.943 | 0.05 | 0.05 | 0.08 | 74.6 | 5.54 | 0 | 0 | 632.21 |

## live_physics / gentle (1273 episodes)

| engine | pos 0.5s (m) | pos 1s | pos 2s | pos 4s | pos 2s active | ori 2s (deg) | fell acc | fell P | fell R | pen mean (mm) | pen capped (mm) | pen >5mm (%) | pen max (mm) | rest drift (mm/s) | diverged | out of bounds | ms/step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| physics_reference | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00 | 1.000 | - | - | 0.09 | 0.08 | 0.11 | 76.9 | 0.19 | 0 | 0 | - |
| old_mlp | 0.0310 | 0.0717 | 0.1434 | 0.2648 | 0.1467 | 17.53 | 0.883 | 0.000 | - | 20.59 | 17.52 | 61.20 | 5606.8 | 77.68 | 0 | 8 | 12.13 |
| old_graph | 0.1268 | 0.3114 | 0.6536 | 1.2378 | 0.9258 | 99.37 | 0.702 | 0.000 | - | 200.61 | 18.32 | 54.28 | 61601.0 | 390.94 | 0 | 67 | 24.95 |
| old_mlp_projected | 0.0200 | 0.0457 | 0.1042 | 0.2012 | 0.1137 | 13.26 | 0.905 | 0.000 | - | 0.12 | 0.12 | 0.53 | 79.8 | 61.39 | 0 | 2 | 123.76 |
| new_raw | 0.0017 | 0.0022 | 0.0028 | 0.0035 | 0.0117 | 0.68 | 1.000 | 0.000 | - | 0.24 | 0.24 | 0.15 | 74.3 | 0.73 | 0 | 0 | 370.63 |
| new_projected | 0.0019 | 0.0023 | 0.0027 | 0.0028 | 0.0138 | 0.50 | 0.999 | 0.000 | - | 0.03 | 0.03 | 0.06 | 72.2 | 0.28 | 0 | 0 | 632.21 |

## live_physics / chaotic (1195 episodes)

| engine | pos 0.5s (m) | pos 1s | pos 2s | pos 4s | pos 2s active | ori 2s (deg) | fell acc | fell P | fell R | pen mean (mm) | pen capped (mm) | pen >5mm (%) | pen max (mm) | rest drift (mm/s) | diverged | out of bounds | ms/step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| physics_reference | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00 | 1.000 | 1.000 | 1.000 | 0.14 | 0.13 | 0.45 | 79.9 | 0.33 | 0 | 0 | - |
| old_mlp | 0.1450 | 0.2635 | 0.4028 | 0.7085 | 0.5889 | 51.43 | 0.824 | 0.668 | 0.797 | 73.68 | 22.59 | 66.71 | 32367.3 | 168.98 | 0 | 159 | 12.13 |
| old_graph | 0.2440 | 0.5099 | 0.9504 | 1.7267 | 1.2173 | 105.66 | 0.674 | 0.465 | 0.732 | 310.89 | 18.25 | 53.25 | 61658.8 | 507.10 | 0 | 106 | 24.95 |
| old_mlp_projected | 0.1316 | 0.2461 | 0.4368 | 0.8326 | 0.6728 | 49.20 | 0.827 | 0.684 | 0.768 | 0.31 | 0.29 | 1.22 | 80.0 | 200.43 | 0 | 59 | 123.76 |
| new_raw | 0.0453 | 0.0826 | 0.1145 | 0.1220 | 0.1954 | 27.85 | 0.959 | 0.924 | 0.938 | 0.53 | 0.52 | 1.67 | 79.9 | 11.79 | 0 | 0 | 370.63 |
| new_projected | 0.0474 | 0.0893 | 0.1217 | 0.1283 | 0.2069 | 28.77 | 0.958 | 0.918 | 0.943 | 0.07 | 0.07 | 0.09 | 74.6 | 11.26 | 0 | 0 | 632.21 |

## all_physics / all (4998 episodes)

| engine | pos 0.5s (m) | pos 1s | pos 2s | pos 4s | pos 2s active | ori 2s (deg) | fell acc | fell P | fell R | pen mean (mm) | pen capped (mm) | pen >5mm (%) | pen max (mm) | rest drift (mm/s) | diverged | out of bounds | ms/step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| physics_reference | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00 | 1.000 | 1.000 | 1.000 | 0.14 | 0.13 | 0.29 | 80.0 | 0.24 | 0 | 0 | - |
| old_mlp | 0.0919 | 0.1780 | 0.2925 | 0.5058 | 0.5336 | 35.53 | 0.829 | 0.451 | 0.779 | 45.19 | 20.87 | 64.94 | 32367.3 | 126.27 | 0 | 330 | 12.13 |
| old_graph | 0.1935 | 0.4318 | 0.8410 | 1.5292 | 1.2198 | 103.18 | 0.670 | 0.271 | 0.744 | 268.76 | 18.64 | 54.47 | 65221.4 | 454.98 | 0 | 374 | 24.95 |
| old_mlp_projected | 0.0786 | 0.1528 | 0.2870 | 0.5537 | 0.5924 | 32.68 | 0.849 | 0.490 | 0.748 | 0.22 | 0.21 | 0.94 | 101.4 | 137.10 | 0 | 137 | 123.76 |
| new_raw | 0.0227 | 0.0417 | 0.0574 | 0.0617 | 0.1595 | 13.47 | 0.979 | 0.924 | 0.936 | 0.38 | 0.38 | 0.85 | 80.0 | 5.91 | 0 | 0 | 370.63 |
| new_projected | 0.0237 | 0.0447 | 0.0608 | 0.0643 | 0.1692 | 13.86 | 0.979 | 0.916 | 0.940 | 0.04 | 0.04 | 0.06 | 77.7 | 5.57 | 0 | 0 | 632.21 |

## all_physics / gentle (2648 episodes)

| engine | pos 0.5s (m) | pos 1s | pos 2s | pos 4s | pos 2s active | ori 2s (deg) | fell acc | fell P | fell R | pen mean (mm) | pen capped (mm) | pen >5mm (%) | pen max (mm) | rest drift (mm/s) | diverged | out of bounds | ms/step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| physics_reference | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00 | 1.000 | - | - | 0.08 | 0.07 | 0.08 | 80.0 | 0.18 | 0 | 0 | - |
| old_mlp | 0.0346 | 0.0807 | 0.1600 | 0.2927 | 0.1723 | 19.04 | 0.857 | 0.000 | - | 22.28 | 18.86 | 63.42 | 5611.5 | 85.97 | 0 | 17 | 12.13 |
| old_graph | 0.1258 | 0.3144 | 0.6679 | 1.2528 | 0.9112 | 99.96 | 0.681 | 0.000 | - | 199.12 | 18.75 | 55.08 | 65221.4 | 391.06 | 0 | 154 | 24.95 |
| old_mlp_projected | 0.0224 | 0.0523 | 0.1204 | 0.2308 | 0.1347 | 15.31 | 0.885 | 0.000 | - | 0.12 | 0.12 | 0.55 | 79.9 | 70.74 | 0 | 3 | 123.76 |
| new_raw | 0.0017 | 0.0022 | 0.0028 | 0.0036 | 0.0118 | 0.71 | 1.000 | 0.000 | - | 0.23 | 0.23 | 0.10 | 74.3 | 0.73 | 0 | 0 | 370.63 |
| new_projected | 0.0018 | 0.0024 | 0.0027 | 0.0028 | 0.0142 | 0.56 | 0.999 | 0.000 | - | 0.02 | 0.02 | 0.03 | 72.2 | 0.28 | 0 | 0 | 632.21 |

## all_physics / chaotic (2350 episodes)

| engine | pos 0.5s (m) | pos 1s | pos 2s | pos 4s | pos 2s active | ori 2s (deg) | fell acc | fell P | fell R | pen mean (mm) | pen capped (mm) | pen >5mm (%) | pen max (mm) | rest drift (mm/s) | diverged | out of bounds | ms/step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| physics_reference | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00 | 1.000 | 1.000 | 1.000 | 0.21 | 0.18 | 0.51 | 80.0 | 0.31 | 0 | 0 | - |
| old_mlp | 0.1498 | 0.2762 | 0.4264 | 0.7208 | 0.6160 | 52.18 | 0.800 | 0.629 | 0.779 | 68.31 | 22.90 | 66.47 | 32367.3 | 172.58 | 0 | 313 | 12.13 |
| old_graph | 0.2617 | 0.5502 | 1.0157 | 1.8081 | 1.2903 | 106.43 | 0.659 | 0.451 | 0.744 | 339.04 | 18.53 | 53.85 | 62005.2 | 528.43 | 0 | 220 | 24.95 |
| old_mlp_projected | 0.1352 | 0.2542 | 0.4552 | 0.8796 | 0.6968 | 50.22 | 0.813 | 0.660 | 0.748 | 0.33 | 0.30 | 1.33 | 101.4 | 213.36 | 0 | 134 | 123.76 |
| new_raw | 0.0439 | 0.0816 | 0.1125 | 0.1204 | 0.1932 | 26.34 | 0.959 | 0.924 | 0.936 | 0.54 | 0.52 | 1.62 | 80.0 | 11.85 | 0 | 0 | 370.63 |
| new_projected | 0.0458 | 0.0875 | 0.1194 | 0.1263 | 0.2046 | 27.29 | 0.957 | 0.917 | 0.940 | 0.07 | 0.06 | 0.09 | 77.7 | 11.65 | 0 | 0 | 632.21 |

## Per scenario kind (all physics): pos err @2s (m) / mean penetration (mm)

| kind | n | physics_reference | old_mlp | old_graph | old_mlp_projected | new_raw | new_projected |
|---|---|---|---|---|---|---|---|
| collision | 293 | 0.0000 / 0.09 | 0.5220 / 85.39 | 1.4412 / 471.53 | 0.4387 / 0.67 | 0.1360 / 0.50 | 0.1373 / 0.10 |
| desk | 453 | 0.0000 / 0.62 | 0.1286 / 25.22 | 0.4996 / 16.45 | 0.1179 / 0.11 | 0.0145 / 0.50 | 0.0171 / 0.13 |
| drop | 238 | 0.0000 / 0.10 | 0.3259 / 21.73 | 0.6980 / 208.33 | 0.2787 / 0.22 | 0.0764 / 0.49 | 0.0820 / 0.02 |
| edge_fall | 266 | 0.0000 / 0.09 | 0.2962 / 40.24 | 0.6983 / 68.92 | 0.2854 / 0.13 | 0.0454 / 0.29 | 0.0467 / 0.01 |
| gentle_push | 608 | 0.0000 / 0.05 | 0.1832 / 24.71 | 0.7363 / 263.85 | 0.1429 / 0.13 | 0.0044 / 0.21 | 0.0045 / 0.02 |
| legacy_layout | 600 | 0.0000 / 0.10 | 0.1797 / 57.74 | 0.4675 / 17.97 | 0.1734 / 0.23 | 0.0763 / 0.46 | 0.0835 / 0.05 |
| noclip | 162 | 0.0000 / 0.15 | 0.2561 / 67.07 | 1.7020 / 1559.75 | 0.2547 / 0.42 | 0.0457 / 0.34 | 0.0478 / 0.09 |
| rest | 770 | 0.0000 / 0.10 | 0.1913 / 25.21 | 0.7130 / 295.76 | 0.1510 / 0.17 | 0.0010 / 0.25 | 0.0004 / 0.00 |
| slide_stop | 486 | 0.0000 / 0.03 | 0.1835 / 26.31 | 0.9166 / 227.45 | 0.1283 / 0.09 | 0.0068 / 0.17 | 0.0070 / 0.03 |
| stack_collapse | 277 | 0.0000 / 0.13 | 0.3113 / 50.57 | 1.1050 / 448.23 | 0.2883 / 0.22 | 0.0674 / 0.52 | 0.0704 / 0.07 |
| tilt_slide | 343 | 0.0000 / 0.20 | 1.0359 / 156.44 | 1.2812 / 390.36 | 1.4143 / 0.44 | 0.2937 / 0.81 | 0.3144 / 0.06 |
| topple | 326 | 0.0000 / 0.06 | 0.2735 / 22.98 | 0.8302 / 131.08 | 0.2408 / 0.17 | 0.0566 / 0.40 | 0.0588 / 0.03 |
| toss | 176 | 0.0000 / 0.08 | 0.3136 / 29.76 | 0.7049 / 164.22 | 0.2968 / 0.15 | 0.0679 / 0.31 | 0.0736 / 0.02 |

## ship_decision (subset live_physics, baseline old_mlp)

ship iff, vs old_mlp: chaotic pos err @2s improves AND mean penetration improves AND gentle pos err @2s <= old * 1.02; AND (when old_mlp_projected was evaluated) vs old_mlp_projected: chaotic pos err @2s improves AND gentle pos err @2s <= old_projected * 1.02 AND capped mean penetration <= old_projected + 1 mm. Gates use point estimates; 'significant' reports whether the paired-bootstrap 95% CI agrees. new_projected = learned ContactGNN + NON-LEARNED contact projection (what would ship); new_raw = the learned model ALONE. The penetration gate against raw old_mlp can be passed by the projection alone, so the gain is only attributed to the learned model when beats_old_projected is true (same projection on the old model).

| variant | includes projection | baseline | metric | old | new | delta | 95% CI | pass | CI significant |
|---|---|---|---|---|---|---|---|---|---|
| new_projected | True | old_mlp | chaotic_pos_err_2s_m | 0.40279 | 0.12165 | -0.28114 | [-0.30847, -0.25747] | True | True |
| new_projected | True | old_mlp | mean_penetration_m | 0.04782 | 0.00005 | -0.04776 | [-0.05256, -0.04298] | True | True |
| new_projected | True | old_mlp | gentle_pos_err_2s_m | 0.14337 | 0.00269 | -0.14068 | [-0.15134, -0.12980] | True | True |
| new_projected | True | old_mlp | mean_penetration_capped_50mm_m | 0.02012 | 0.00005 | -0.02007 | [-0.02045, -0.01966] | True (not gating) | True |
| new_projected | True | old_mlp | chaotic_active_pos_err_2s_m | 0.58886 | 0.20691 | -0.38195 | [-0.42345, -0.34151] | True (not gating) | True |
| new_projected | True | old_mlp_projected | chaotic_pos_err_2s_m | 0.43677 | 0.12165 | -0.31512 | [-0.35916, -0.27315] | True | True |
| new_projected | True | old_mlp_projected | gentle_pos_err_2s_m | 0.10422 | 0.00269 | -0.10153 | [-0.11185, -0.09183] | True | True |
| new_projected | True | old_mlp_projected | mean_penetration_capped_50mm_m | 0.00021 | 0.00005 | -0.00016 | [-0.00019, -0.00012] | True | True |
| new_projected | True | | **ship** (vs old_mlp only: True; beats old_mlp_projected: True) | | | | | **True** | |
| new_raw | False | old_mlp | chaotic_pos_err_2s_m | 0.40279 | 0.11451 | -0.28828 | [-0.31605, -0.26444] | True | True |
| new_raw | False | old_mlp | mean_penetration_m | 0.04782 | 0.00039 | -0.04743 | [-0.05223, -0.04266] | True | True |
| new_raw | False | old_mlp | gentle_pos_err_2s_m | 0.14337 | 0.00284 | -0.14053 | [-0.15138, -0.12972] | True | True |
| new_raw | False | old_mlp | mean_penetration_capped_50mm_m | 0.02012 | 0.00038 | -0.01974 | [-0.02011, -0.01933] | True (not gating) | True |
| new_raw | False | old_mlp | chaotic_active_pos_err_2s_m | 0.58886 | 0.19544 | -0.39342 | [-0.43640, -0.35422] | True (not gating) | True |
| new_raw | False | old_mlp_projected | chaotic_pos_err_2s_m | 0.43677 | 0.11451 | -0.32226 | [-0.36832, -0.28015] | True | True |
| new_raw | False | old_mlp_projected | gentle_pos_err_2s_m | 0.10422 | 0.00284 | -0.10138 | [-0.11160, -0.09152] | True | True |
| new_raw | False | old_mlp_projected | mean_penetration_capped_50mm_m | 0.00021 | 0.00038 | 0.00018 | [0.00014, 0.00022] | True | False |
| new_raw | False | | **ship** (vs old_mlp only: True; beats old_mlp_projected: True) | | | | | **True** | |

## ship_decision_all_physics (subset all_physics, baseline old_mlp)

ship iff, vs old_mlp: chaotic pos err @2s improves AND mean penetration improves AND gentle pos err @2s <= old * 1.02; AND (when old_mlp_projected was evaluated) vs old_mlp_projected: chaotic pos err @2s improves AND gentle pos err @2s <= old_projected * 1.02 AND capped mean penetration <= old_projected + 1 mm. Gates use point estimates; 'significant' reports whether the paired-bootstrap 95% CI agrees. new_projected = learned ContactGNN + NON-LEARNED contact projection (what would ship); new_raw = the learned model ALONE. The penetration gate against raw old_mlp can be passed by the projection alone, so the gain is only attributed to the learned model when beats_old_projected is true (same projection on the old model).

| variant | includes projection | baseline | metric | old | new | delta | 95% CI | pass | CI significant |
|---|---|---|---|---|---|---|---|---|---|
| new_projected | True | old_mlp | chaotic_pos_err_2s_m | 0.42635 | 0.11939 | -0.30696 | [-0.32530, -0.28941] | True | True |
| new_projected | True | old_mlp | mean_penetration_m | 0.04519 | 0.00004 | -0.04515 | [-0.04840, -0.04216] | True | True |
| new_projected | True | old_mlp | gentle_pos_err_2s_m | 0.15996 | 0.00266 | -0.15729 | [-0.16468, -0.15005] | True | True |
| new_projected | True | old_mlp | mean_penetration_capped_50mm_m | 0.02087 | 0.00004 | -0.02083 | [-0.02111, -0.02056] | True (not gating) | True |
| new_projected | True | old_mlp | chaotic_active_pos_err_2s_m | 0.61604 | 0.20457 | -0.41147 | [-0.44340, -0.38475] | True (not gating) | True |
| new_projected | True | old_mlp_projected | chaotic_pos_err_2s_m | 0.45519 | 0.11939 | -0.33580 | [-0.36597, -0.30823] | True | True |
| new_projected | True | old_mlp_projected | gentle_pos_err_2s_m | 0.12035 | 0.00266 | -0.11769 | [-0.12510, -0.11078] | True | True |
| new_projected | True | old_mlp_projected | mean_penetration_capped_50mm_m | 0.00021 | 0.00004 | -0.00017 | [-0.00019, -0.00014] | True | True |
| new_projected | True | | **ship** (vs old_mlp only: True; beats old_mlp_projected: True) | | | | | **True** | |
| new_raw | False | old_mlp | chaotic_pos_err_2s_m | 0.42635 | 0.11251 | -0.31384 | [-0.33270, -0.29622] | True | True |
| new_raw | False | old_mlp | mean_penetration_m | 0.04519 | 0.00038 | -0.04481 | [-0.04806, -0.04184] | True | True |
| new_raw | False | old_mlp | gentle_pos_err_2s_m | 0.15996 | 0.00281 | -0.15715 | [-0.16457, -0.15000] | True | True |
| new_raw | False | old_mlp | mean_penetration_capped_50mm_m | 0.02087 | 0.00038 | -0.02049 | [-0.02077, -0.02024] | True (not gating) | True |
| new_raw | False | old_mlp | chaotic_active_pos_err_2s_m | 0.61604 | 0.19318 | -0.42285 | [-0.45479, -0.39579] | True (not gating) | True |
| new_raw | False | old_mlp_projected | chaotic_pos_err_2s_m | 0.45519 | 0.11251 | -0.34268 | [-0.37263, -0.31455] | True | True |
| new_raw | False | old_mlp_projected | gentle_pos_err_2s_m | 0.12035 | 0.00281 | -0.11755 | [-0.12498, -0.11072] | True | True |
| new_raw | False | old_mlp_projected | mean_penetration_capped_50mm_m | 0.00021 | 0.00038 | 0.00017 | [0.00014, 0.00020] | True | False |
| new_raw | False | | **ship** (vs old_mlp only: True; beats old_mlp_projected: True) | | | | | **True** | |

Notes:
