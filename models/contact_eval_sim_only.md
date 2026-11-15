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
| old_mlp | 0.0895 | 0.1700 | 0.2764 | 0.4923 | 0.5108 | 34.91 | 0.853 | 0.508 | 0.797 | 47.82 | 20.12 | 64.03 | 32367.3 | 121.44 | 0 | 167 | 1.66 |
| old_graph | 0.1869 | 0.4132 | 0.8058 | 1.4885 | 1.1658 | 102.60 | 0.688 | 0.289 | 0.732 | 257.16 | 18.28 | 53.75 | 61658.8 | 446.61 | 0 | 173 | 2.57 |
| old_mlp_projected | 0.0773 | 0.1485 | 0.2748 | 0.5250 | 0.5740 | 31.69 | 0.865 | 0.537 | 0.768 | 0.22 | 0.21 | 0.89 | 80.0 | 128.02 | 0 | 61 | 8.22 |
| new_raw | 0.0240 | 0.0438 | 0.0597 | 0.0634 | 0.1621 | 14.34 | 0.979 | 0.924 | 0.936 | 0.39 | 0.39 | 0.94 | 79.5 | 5.72 | 0 | 0 | 4.56 |
| new_projected | 0.0250 | 0.0467 | 0.0634 | 0.0670 | 0.1725 | 14.87 | 0.978 | 0.916 | 0.942 | 0.06 | 0.05 | 0.11 | 79.1 | 5.48 | 0 | 0 | 10.77 |

## live_physics / gentle (1273 episodes)

| engine | pos 0.5s (m) | pos 1s | pos 2s | pos 4s | pos 2s active | ori 2s (deg) | fell acc | fell P | fell R | pen mean (mm) | pen capped (mm) | pen >5mm (%) | pen max (mm) | rest drift (mm/s) | diverged | out of bounds | ms/step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| physics_reference | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00 | 1.000 | - | - | 0.09 | 0.08 | 0.11 | 76.9 | 0.19 | 0 | 0 | - |
| old_mlp | 0.0310 | 0.0717 | 0.1434 | 0.2648 | 0.1467 | 17.53 | 0.883 | 0.000 | - | 20.59 | 17.52 | 61.20 | 5606.8 | 77.68 | 0 | 8 | 1.66 |
| old_graph | 0.1268 | 0.3114 | 0.6536 | 1.2378 | 0.9258 | 99.37 | 0.702 | 0.000 | - | 200.61 | 18.32 | 54.28 | 61601.0 | 390.94 | 0 | 67 | 2.57 |
| old_mlp_projected | 0.0200 | 0.0457 | 0.1042 | 0.2012 | 0.1137 | 13.26 | 0.905 | 0.000 | - | 0.12 | 0.12 | 0.53 | 79.8 | 61.39 | 0 | 2 | 8.22 |
| new_raw | 0.0017 | 0.0020 | 0.0024 | 0.0031 | 0.0122 | 0.59 | 1.000 | 0.000 | - | 0.25 | 0.25 | 0.14 | 69.1 | 0.61 | 0 | 0 | 4.56 |
| new_projected | 0.0018 | 0.0025 | 0.0028 | 0.0029 | 0.0152 | 0.51 | 0.999 | 0.000 | - | 0.03 | 0.03 | 0.06 | 68.6 | 0.30 | 0 | 0 | 10.77 |

## live_physics / chaotic (1195 episodes)

| engine | pos 0.5s (m) | pos 1s | pos 2s | pos 4s | pos 2s active | ori 2s (deg) | fell acc | fell P | fell R | pen mean (mm) | pen capped (mm) | pen >5mm (%) | pen max (mm) | rest drift (mm/s) | diverged | out of bounds | ms/step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| physics_reference | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00 | 1.000 | 1.000 | 1.000 | 0.14 | 0.13 | 0.45 | 79.9 | 0.33 | 0 | 0 | - |
| old_mlp | 0.1450 | 0.2635 | 0.4028 | 0.7085 | 0.5889 | 51.43 | 0.824 | 0.668 | 0.797 | 73.68 | 22.59 | 66.71 | 32367.3 | 168.98 | 0 | 159 | 1.66 |
| old_graph | 0.2440 | 0.5099 | 0.9504 | 1.7267 | 1.2173 | 105.66 | 0.674 | 0.465 | 0.732 | 310.89 | 18.25 | 53.25 | 61658.8 | 507.10 | 0 | 106 | 2.57 |
| old_mlp_projected | 0.1316 | 0.2461 | 0.4368 | 0.8326 | 0.6728 | 49.20 | 0.827 | 0.684 | 0.768 | 0.31 | 0.29 | 1.22 | 80.0 | 200.43 | 0 | 59 | 8.22 |
| new_raw | 0.0452 | 0.0834 | 0.1141 | 0.1207 | 0.1943 | 27.40 | 0.959 | 0.925 | 0.936 | 0.53 | 0.52 | 1.70 | 79.5 | 11.28 | 0 | 0 | 4.56 |
| new_projected | 0.0470 | 0.0886 | 0.1209 | 0.1279 | 0.2063 | 28.51 | 0.958 | 0.917 | 0.942 | 0.08 | 0.08 | 0.16 | 79.1 | 11.11 | 0 | 0 | 10.77 |

## all_physics / all (4998 episodes)

| engine | pos 0.5s (m) | pos 1s | pos 2s | pos 4s | pos 2s active | ori 2s (deg) | fell acc | fell P | fell R | pen mean (mm) | pen capped (mm) | pen >5mm (%) | pen max (mm) | rest drift (mm/s) | diverged | out of bounds | ms/step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| physics_reference | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00 | 1.000 | 1.000 | 1.000 | 0.14 | 0.13 | 0.29 | 80.0 | 0.24 | 0 | 0 | - |
| old_mlp | 0.0919 | 0.1780 | 0.2925 | 0.5058 | 0.5336 | 35.53 | 0.829 | 0.451 | 0.779 | 45.19 | 20.87 | 64.94 | 32367.3 | 126.27 | 0 | 330 | 1.66 |
| old_graph | 0.1935 | 0.4318 | 0.8410 | 1.5292 | 1.2198 | 103.18 | 0.670 | 0.271 | 0.744 | 268.76 | 18.64 | 54.47 | 65221.4 | 454.98 | 0 | 374 | 2.57 |
| old_mlp_projected | 0.0786 | 0.1528 | 0.2870 | 0.5537 | 0.5924 | 32.68 | 0.849 | 0.490 | 0.748 | 0.22 | 0.21 | 0.94 | 101.4 | 137.10 | 0 | 137 | 8.22 |
| new_raw | 0.0227 | 0.0420 | 0.0572 | 0.0612 | 0.1592 | 13.30 | 0.980 | 0.924 | 0.937 | 0.38 | 0.37 | 0.85 | 79.8 | 5.82 | 0 | 0 | 4.56 |
| new_projected | 0.0235 | 0.0444 | 0.0605 | 0.0641 | 0.1689 | 13.82 | 0.978 | 0.917 | 0.938 | 0.05 | 0.04 | 0.09 | 79.1 | 5.46 | 0 | 0 | 10.77 |

## all_physics / gentle (2648 episodes)

| engine | pos 0.5s (m) | pos 1s | pos 2s | pos 4s | pos 2s active | ori 2s (deg) | fell acc | fell P | fell R | pen mean (mm) | pen capped (mm) | pen >5mm (%) | pen max (mm) | rest drift (mm/s) | diverged | out of bounds | ms/step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| physics_reference | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00 | 1.000 | - | - | 0.08 | 0.07 | 0.08 | 80.0 | 0.18 | 0 | 0 | - |
| old_mlp | 0.0346 | 0.0807 | 0.1600 | 0.2927 | 0.1723 | 19.04 | 0.857 | 0.000 | - | 22.28 | 18.86 | 63.42 | 5611.5 | 85.97 | 0 | 17 | 1.66 |
| old_graph | 0.1258 | 0.3144 | 0.6679 | 1.2528 | 0.9112 | 99.96 | 0.681 | 0.000 | - | 199.12 | 18.75 | 55.08 | 65221.4 | 391.06 | 0 | 154 | 2.57 |
| old_mlp_projected | 0.0224 | 0.0523 | 0.1204 | 0.2308 | 0.1347 | 15.31 | 0.885 | 0.000 | - | 0.12 | 0.12 | 0.55 | 79.9 | 70.74 | 0 | 3 | 8.22 |
| new_raw | 0.0017 | 0.0021 | 0.0025 | 0.0034 | 0.0118 | 0.66 | 1.000 | 0.000 | - | 0.23 | 0.23 | 0.08 | 69.1 | 0.68 | 0 | 0 | 4.56 |
| new_projected | 0.0018 | 0.0025 | 0.0027 | 0.0028 | 0.0149 | 0.55 | 1.000 | 0.000 | - | 0.02 | 0.02 | 0.03 | 68.6 | 0.28 | 0 | 0 | 10.77 |

## all_physics / chaotic (2350 episodes)

| engine | pos 0.5s (m) | pos 1s | pos 2s | pos 4s | pos 2s active | ori 2s (deg) | fell acc | fell P | fell R | pen mean (mm) | pen capped (mm) | pen >5mm (%) | pen max (mm) | rest drift (mm/s) | diverged | out of bounds | ms/step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| physics_reference | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00 | 1.000 | 1.000 | 1.000 | 0.21 | 0.18 | 0.51 | 80.0 | 0.31 | 0 | 0 | - |
| old_mlp | 0.1498 | 0.2762 | 0.4264 | 0.7208 | 0.6160 | 52.18 | 0.800 | 0.629 | 0.779 | 68.31 | 22.90 | 66.47 | 32367.3 | 172.58 | 0 | 313 | 1.66 |
| old_graph | 0.2617 | 0.5502 | 1.0157 | 1.8081 | 1.2903 | 106.43 | 0.659 | 0.451 | 0.744 | 339.04 | 18.53 | 53.85 | 62005.2 | 528.43 | 0 | 220 | 2.57 |
| old_mlp_projected | 0.1352 | 0.2542 | 0.4552 | 0.8796 | 0.6968 | 50.22 | 0.813 | 0.660 | 0.748 | 0.33 | 0.30 | 1.33 | 101.4 | 213.36 | 0 | 134 | 8.22 |
| new_raw | 0.0439 | 0.0822 | 0.1125 | 0.1196 | 0.1929 | 26.07 | 0.959 | 0.925 | 0.937 | 0.53 | 0.52 | 1.62 | 79.8 | 11.73 | 0 | 0 | 4.56 |
| new_projected | 0.0455 | 0.0868 | 0.1189 | 0.1260 | 0.2041 | 27.21 | 0.957 | 0.918 | 0.938 | 0.07 | 0.07 | 0.14 | 79.1 | 11.41 | 0 | 0 | 10.77 |

## Per scenario kind (all physics): pos err @2s (m) / mean penetration (mm)

| kind | n | physics_reference | old_mlp | old_graph | old_mlp_projected | new_raw | new_projected |
|---|---|---|---|---|---|---|---|
| collision | 293 | 0.0000 / 0.09 | 0.5220 / 85.39 | 1.4412 / 471.53 | 0.4387 / 0.67 | 0.1382 / 0.51 | 0.1417 / 0.09 |
| desk | 453 | 0.0000 / 0.62 | 0.1286 / 25.22 | 0.4996 / 16.45 | 0.1179 / 0.11 | 0.0148 / 0.46 | 0.0170 / 0.13 |
| drop | 238 | 0.0000 / 0.10 | 0.3259 / 21.73 | 0.6980 / 208.33 | 0.2787 / 0.22 | 0.0754 / 0.48 | 0.0805 / 0.04 |
| edge_fall | 266 | 0.0000 / 0.09 | 0.2962 / 40.24 | 0.6983 / 68.92 | 0.2854 / 0.13 | 0.0454 / 0.30 | 0.0477 / 0.00 |
| gentle_push | 608 | 0.0000 / 0.05 | 0.1832 / 24.71 | 0.7363 / 263.85 | 0.1429 / 0.13 | 0.0045 / 0.21 | 0.0044 / 0.02 |
| legacy_layout | 600 | 0.0000 / 0.10 | 0.1797 / 57.74 | 0.4675 / 17.97 | 0.1734 / 0.23 | 0.0776 / 0.47 | 0.0816 / 0.05 |
| noclip | 162 | 0.0000 / 0.15 | 0.2561 / 67.07 | 1.7020 / 1559.75 | 0.2547 / 0.42 | 0.0448 / 0.34 | 0.0480 / 0.08 |
| rest | 770 | 0.0000 / 0.10 | 0.1913 / 25.21 | 0.7130 / 295.76 | 0.1510 / 0.17 | 0.0010 / 0.26 | 0.0004 / 0.00 |
| slide_stop | 486 | 0.0000 / 0.03 | 0.1835 / 26.31 | 0.9166 / 227.45 | 0.1283 / 0.09 | 0.0067 / 0.17 | 0.0062 / 0.03 |
| stack_collapse | 277 | 0.0000 / 0.13 | 0.3113 / 50.57 | 1.1050 / 448.23 | 0.2883 / 0.22 | 0.0651 / 0.54 | 0.0721 / 0.08 |
| tilt_slide | 343 | 0.0000 / 0.20 | 1.0359 / 156.44 | 1.2812 / 390.36 | 1.4143 / 0.44 | 0.2897 / 0.80 | 0.3085 / 0.08 |
| topple | 326 | 0.0000 / 0.06 | 0.2735 / 22.98 | 0.8302 / 131.08 | 0.2408 / 0.17 | 0.0580 / 0.39 | 0.0585 / 0.03 |
| toss | 176 | 0.0000 / 0.08 | 0.3136 / 29.76 | 0.7049 / 164.22 | 0.2968 / 0.15 | 0.0652 / 0.32 | 0.0753 / 0.02 |

## ship_decision (subset live_physics, baseline old_mlp)

ship iff, vs old_mlp: chaotic pos err @2s improves AND mean penetration improves AND gentle pos err @2s <= old * 1.02; AND (when old_mlp_projected was evaluated) vs old_mlp_projected: chaotic pos err @2s improves AND gentle pos err @2s <= old_projected * 1.02 AND capped mean penetration <= old_projected + 1 mm. Gates use point estimates; 'significant' reports whether the paired-bootstrap 95% CI agrees. new_projected = learned ContactGNN + NON-LEARNED contact projection (what would ship); new_raw = the learned model ALONE. The penetration gate against raw old_mlp can be passed by the projection alone, so the gain is only attributed to the learned model when beats_old_projected is true (same projection on the old model).

| variant | includes projection | baseline | metric | old | new | delta | 95% CI | pass | CI significant |
|---|---|---|---|---|---|---|---|---|---|
| new_projected | True | old_mlp | chaotic_pos_err_2s_m | 0.40279 | 0.12093 | -0.28187 | [-0.30936, -0.25777] | True | True |
| new_projected | True | old_mlp | mean_penetration_m | 0.04782 | 0.00006 | -0.04776 | [-0.05255, -0.04298] | True | True |
| new_projected | True | old_mlp | gentle_pos_err_2s_m | 0.14337 | 0.00277 | -0.14060 | [-0.15134, -0.12981] | True | True |
| new_projected | True | old_mlp | mean_penetration_capped_50mm_m | 0.02012 | 0.00005 | -0.02007 | [-0.02045, -0.01966] | True (not gating) | True |
| new_projected | True | old_mlp | chaotic_active_pos_err_2s_m | 0.58886 | 0.20627 | -0.38259 | [-0.42304, -0.34260] | True (not gating) | True |
| new_projected | True | old_mlp_projected | chaotic_pos_err_2s_m | 0.43677 | 0.12093 | -0.31585 | [-0.36070, -0.27367] | True | True |
| new_projected | True | old_mlp_projected | gentle_pos_err_2s_m | 0.10422 | 0.00277 | -0.10146 | [-0.11134, -0.09167] | True | True |
| new_projected | True | old_mlp_projected | mean_penetration_capped_50mm_m | 0.00021 | 0.00005 | -0.00015 | [-0.00019, -0.00012] | True | True |
| new_projected | True | | **ship** (vs old_mlp only: True; beats old_mlp_projected: True) | | | | | **True** | |
| new_raw | False | old_mlp | chaotic_pos_err_2s_m | 0.40279 | 0.11406 | -0.28873 | [-0.31726, -0.26469] | True | True |
| new_raw | False | old_mlp | mean_penetration_m | 0.04782 | 0.00039 | -0.04742 | [-0.05223, -0.04265] | True | True |
| new_raw | False | old_mlp | gentle_pos_err_2s_m | 0.14337 | 0.00240 | -0.14097 | [-0.15147, -0.13022] | True | True |
| new_raw | False | old_mlp | mean_penetration_capped_50mm_m | 0.02012 | 0.00039 | -0.01973 | [-0.02011, -0.01933] | True (not gating) | True |
| new_raw | False | old_mlp | chaotic_active_pos_err_2s_m | 0.58886 | 0.19429 | -0.39458 | [-0.43798, -0.35491] | True (not gating) | True |
| new_raw | False | old_mlp_projected | chaotic_pos_err_2s_m | 0.43677 | 0.11406 | -0.32271 | [-0.36961, -0.28145] | True | True |
| new_raw | False | old_mlp_projected | gentle_pos_err_2s_m | 0.10422 | 0.00240 | -0.10183 | [-0.11211, -0.09202] | True | True |
| new_raw | False | old_mlp_projected | mean_penetration_capped_50mm_m | 0.00021 | 0.00039 | 0.00018 | [0.00014, 0.00022] | True | False |
| new_raw | False | | **ship** (vs old_mlp only: True; beats old_mlp_projected: True) | | | | | **True** | |

## ship_decision_all_physics (subset all_physics, baseline old_mlp)

ship iff, vs old_mlp: chaotic pos err @2s improves AND mean penetration improves AND gentle pos err @2s <= old * 1.02; AND (when old_mlp_projected was evaluated) vs old_mlp_projected: chaotic pos err @2s improves AND gentle pos err @2s <= old_projected * 1.02 AND capped mean penetration <= old_projected + 1 mm. Gates use point estimates; 'significant' reports whether the paired-bootstrap 95% CI agrees. new_projected = learned ContactGNN + NON-LEARNED contact projection (what would ship); new_raw = the learned model ALONE. The penetration gate against raw old_mlp can be passed by the projection alone, so the gain is only attributed to the learned model when beats_old_projected is true (same projection on the old model).

| variant | includes projection | baseline | metric | old | new | delta | 95% CI | pass | CI significant |
|---|---|---|---|---|---|---|---|---|---|
| new_projected | True | old_mlp | chaotic_pos_err_2s_m | 0.42635 | 0.11889 | -0.30746 | [-0.32565, -0.29038] | True | True |
| new_projected | True | old_mlp | mean_penetration_m | 0.04519 | 0.00005 | -0.04514 | [-0.04839, -0.04216] | True | True |
| new_projected | True | old_mlp | gentle_pos_err_2s_m | 0.15996 | 0.00265 | -0.15730 | [-0.16474, -0.14998] | True | True |
| new_projected | True | old_mlp | mean_penetration_capped_50mm_m | 0.02087 | 0.00004 | -0.02083 | [-0.02111, -0.02056] | True (not gating) | True |
| new_projected | True | old_mlp | chaotic_active_pos_err_2s_m | 0.61604 | 0.20407 | -0.41197 | [-0.44354, -0.38490] | True (not gating) | True |
| new_projected | True | old_mlp_projected | chaotic_pos_err_2s_m | 0.45519 | 0.11889 | -0.33630 | [-0.36666, -0.30798] | True | True |
| new_projected | True | old_mlp_projected | gentle_pos_err_2s_m | 0.12035 | 0.00265 | -0.11770 | [-0.12510, -0.11094] | True | True |
| new_projected | True | old_mlp_projected | mean_penetration_capped_50mm_m | 0.00021 | 0.00004 | -0.00017 | [-0.00019, -0.00014] | True | True |
| new_projected | True | | **ship** (vs old_mlp only: True; beats old_mlp_projected: True) | | | | | **True** | |
| new_raw | False | old_mlp | chaotic_pos_err_2s_m | 0.42635 | 0.11247 | -0.31388 | [-0.33218, -0.29641] | True | True |
| new_raw | False | old_mlp | mean_penetration_m | 0.04519 | 0.00038 | -0.04481 | [-0.04807, -0.04184] | True | True |
| new_raw | False | old_mlp | gentle_pos_err_2s_m | 0.15996 | 0.00251 | -0.15745 | [-0.16479, -0.15011] | True | True |
| new_raw | False | old_mlp | mean_penetration_capped_50mm_m | 0.02087 | 0.00037 | -0.02050 | [-0.02077, -0.02024] | True (not gating) | True |
| new_raw | False | old_mlp | chaotic_active_pos_err_2s_m | 0.61604 | 0.19289 | -0.42315 | [-0.45442, -0.39630] | True (not gating) | True |
| new_raw | False | old_mlp_projected | chaotic_pos_err_2s_m | 0.45519 | 0.11247 | -0.34271 | [-0.37338, -0.31456] | True | True |
| new_raw | False | old_mlp_projected | gentle_pos_err_2s_m | 0.12035 | 0.00251 | -0.11784 | [-0.12528, -0.11101] | True | True |
| new_raw | False | old_mlp_projected | mean_penetration_capped_50mm_m | 0.00021 | 0.00037 | 0.00016 | [0.00014, 0.00019] | True | False |
| new_raw | False | | **ship** (vs old_mlp only: True; beats old_mlp_projected: True) | | | | | **True** | |

Notes:
