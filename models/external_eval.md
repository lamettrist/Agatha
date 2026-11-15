# External-data evaluation (APPROXIMATE)

> **APPROXIMATE: external data converted with documented approximations (scripts/prepare_external_data.py). Kubric / CLEVRER / Physion are simulator output from other generators; the `real_*` datasets are real-world pose tracks of ONE tossed box with ASSUMED friction / restitution. None of this is a validation of the product.**

New model: `/home/chintd2/agatha/models/contact_gnn.pt` (epoch 246, file written 2026-11-15 06:35:20, sha256 e4b5e4a3a905); old model: `/home/chintd2/agatha/models/mlp_multistep.pt`. Autoregressive rollouts from the first converted frame, never teacher-forced. Licences: `DATA_LICENSES.md`. Entries kept by `--merge` from an earlier run state their own checkpoint below the dataset heading.

Model / projection code: `/home/chintd2/agatha/agatha/contact_gnn.py` sha256 7f4ce7fcf5f5 (the `*_projected` rows depend on `contact_gnn.project` in exactly this version).

Engines:
- `reference`: converted ground truth scored with the same metrics
- `hold_initial`: trivial baseline: every object stays at its initial pose
- `agatha_physics`: Agatha's PyBullet box engine from the same converted initial state (how far the box approximation + a different simulator are from the source, without any learned model)
- `old_mlp`: old shipped model alone (mlp_multistep)
- `old_mlp_projected`: old mlp_multistep + the non-learned contact projection (attribution only)
- `new_raw`: ContactGNN alone: learned model only, no projection
- `new_projected`: ContactGNN + non-learned contact projection after every step (shipped configuration)

Subsets:
- `all_objects`: all dynamic objects
- `box_like_objects`: true cuboids, in any episode (neighbours may be non-box shapes)
- `box_like_only_episodes`: all dynamic objects of episodes that contain only true cuboids

Columns: position error (m) at 0.5 s / 1 s / the last frame ("rest pos", all objects) and at the last frame for the objects that are at rest in the ground truth ("settled"); tipped / moved-far accuracy with (TP/FP/FN/TN), `maj` = accuracy of always predicting the majority class; `fell` false alarms (the ground truth never falls: there is no edge); penetration in mm.

## clevrer

Evaluated 2026-11-15 05:52:42 with contact_gnn checkpoint sha256 7c53e6b60107 (epoch 115, file written 2026-11-15 05:27:01), contact_gnn.py sha256 7f4ce7fcf5f5.

1000 episodes, 4955 dynamic objects (1505 box-like, 4 box-like-only episodes); horizon 4-4 s; ground truth: tipped 0.000, moved far 0.733, settled at the end 0.259. Preferred ground mapping for new_projected (lower rest-position error, all objects): **slab**.

### clevrer / ground = slab / all_objects (4955 objects in 1000 episodes)

| engine | pos 0.5s | pos 1s | rest pos | rest pos median | rest pos settled (n) | ori final (deg) | tipped acc (TP/FP/FN/TN) | tipped maj | moved-far acc (TP/FP/FN/TN) | fell false alarms | pen mean (mm) | pen capped (mm) | diverged |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| reference | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 (1281) | 0.0 | 1.000 (1/0/0/4954) | 1.000 | 1.000 (3634/0/0/1321) | 0 | 1.20 | 1.20 | 0 |
| hold_initial | 0.252 | 0.481 | 1.359 | 1.293 | 0.216 (1281) | 10.9 | 1.000 (0/0/1/4954) | 1.000 | 0.267 (0/0/3634/1321) | 0 | 0.08 | 0.08 | 0 |
| agatha_physics | 0.007 | 0.030 | 0.455 | 0.439 | 0.048 (1281) | 27.3 | 1.000 (0/0/1/4954) | 1.000 | 0.881 (3059/13/575/1308) | 0 | 0.04 | 0.04 | 0 |
| old_mlp | 0.378 | 0.855 | 7.345 | 6.620 | 6.829 (1281) | 136.3 | 0.000 (1/4954/0/0) | 1.000 | 0.733 (3634/1321/0/0) | 379 | 1412.69 | 38.96 | 0 |
| old_mlp_projected | 0.317 | 0.748 | 3.950 | 2.363 | 2.399 (1281) | 102.3 | 0.768 (0/1151/1/3803) | 1.000 | 0.733 (3634/1321/0/0) | 1110 | 0.00 | 0.00 | 0 |
| new_raw | 0.025 | 0.097 | 0.879 | 0.830 | 0.104 (1281) | 14.5 | 1.000 (0/0/1/4954) | 1.000 | 0.795 (2626/6/1008/1315) | 0 | 0.17 | 0.17 | 0 |
| new_projected | 0.025 | 0.094 | 0.869 | 0.826 | 0.099 (1281) | 14.5 | 1.000 (0/0/1/4954) | 1.000 | 0.795 (2624/6/1010/1315) | 0 | 0.01 | 0.01 | 0 |

### clevrer / ground = slab / box_like_objects (1505 objects in 832 episodes)

| engine | pos 0.5s | pos 1s | rest pos | rest pos median | rest pos settled (n) | ori final (deg) | tipped acc (TP/FP/FN/TN) | tipped maj | moved-far acc (TP/FP/FN/TN) | fell false alarms | pen mean (mm) | pen capped (mm) | diverged |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| reference | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 (443) | 0.0 | 1.000 (1/0/0/1504) | 0.999 | 1.000 (1081/0/0/424) | 0 | 1.07 | 1.07 | 0 |
| hold_initial | 0.252 | 0.476 | 1.269 | 1.234 | 0.293 (443) | 35.7 | 0.999 (0/0/1/1504) | 0.999 | 0.282 (0/0/1081/424) | 0 | 0.11 | 0.11 | 0 |
| agatha_physics | 0.006 | 0.025 | 0.369 | 0.365 | 0.067 (443) | 35.3 | 0.999 (0/0/1/1504) | 0.999 | 0.894 (922/1/159/423) | 0 | 0.04 | 0.04 | 0 |
| old_mlp | 0.379 | 0.853 | 7.297 | 6.727 | 6.738 (443) | 124.5 | 0.001 (1/1504/0/0) | 0.999 | 0.718 (1081/424/0/0) | 103 | 1403.60 | 38.55 | 0 |
| old_mlp_projected | 0.318 | 0.743 | 3.962 | 2.440 | 2.643 (443) | 104.1 | 0.761 (0/358/1/1146) | 0.999 | 0.718 (1081/424/0/0) | 341 | 0.00 | 0.00 | 0 |
| new_raw | 0.019 | 0.071 | 0.728 | 0.702 | 0.120 (443) | 37.6 | 0.999 (0/0/1/1504) | 0.999 | 0.817 (806/1/275/423) | 0 | 0.19 | 0.19 | 0 |
| new_projected | 0.018 | 0.070 | 0.720 | 0.696 | 0.117 (443) | 37.6 | 0.999 (0/0/1/1504) | 0.999 | 0.817 (807/1/274/423) | 0 | 0.01 | 0.01 | 0 |

### clevrer / ground = slab / box_like_only_episodes (17 objects in 4 episodes)

| engine | pos 0.5s | pos 1s | rest pos | rest pos median | rest pos settled (n) | ori final (deg) | tipped acc (TP/FP/FN/TN) | tipped maj | moved-far acc (TP/FP/FN/TN) | fell false alarms | pen mean (mm) | pen capped (mm) | diverged |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| reference | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 (2) | 0.0 | 1.000 (0/0/0/17) | 1.000 | 1.000 (16/0/0/1) | 0 | 0.31 | 0.31 | 0 |
| hold_initial | 0.330 | 0.631 | 1.766 | 1.966 | 0.792 (2) | 18.2 | 1.000 (0/0/0/17) | 1.000 | 0.059 (0/0/16/1) | 0 | 0.12 | 0.12 | 0 |
| agatha_physics | 0.007 | 0.028 | 0.457 | 0.513 | 0.067 (2) | 15.0 | 1.000 (0/0/0/17) | 1.000 | 0.765 (12/0/4/1) | 0 | 0.04 | 0.04 | 0 |
| old_mlp | 0.360 | 0.829 | 6.081 | 5.891 | 5.470 (2) | 129.5 | 0.000 (0/17/0/0) | 1.000 | 0.941 (16/1/0/0) | 2 | 1060.85 | 37.39 | 0 |
| old_mlp_projected | 0.302 | 0.709 | 3.083 | 1.787 | 0.986 (2) | 105.3 | 0.824 (0/3/0/14) | 1.000 | 0.941 (16/1/0/0) | 3 | 0.01 | 0.01 | 0 |
| new_raw | 0.024 | 0.093 | 1.007 | 1.016 | 0.341 (2) | 19.6 | 1.000 (0/0/0/17) | 1.000 | 0.765 (12/0/4/1) | 0 | 0.17 | 0.17 | 0 |
| new_projected | 0.024 | 0.090 | 0.995 | 1.018 | 0.341 (2) | 19.9 | 1.000 (0/0/0/17) | 1.000 | 0.765 (12/0/4/1) | 0 | 0.09 | 0.09 | 0 |

### clevrer / ground = floor / all_objects (4955 objects in 1000 episodes)

| engine | pos 0.5s | pos 1s | rest pos | rest pos median | rest pos settled (n) | ori final (deg) | tipped acc (TP/FP/FN/TN) | tipped maj | moved-far acc (TP/FP/FN/TN) | fell false alarms | pen mean (mm) | pen capped (mm) | diverged |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| reference | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 (1281) | 0.0 | 1.000 (1/0/0/4954) | 1.000 | 1.000 (3634/0/0/1321) | - | 1.20 | 1.20 | 0 |
| hold_initial | 0.252 | 0.481 | 1.359 | 1.293 | 0.216 (1281) | 10.9 | 1.000 (0/0/1/4954) | 1.000 | 0.267 (0/0/3634/1321) | - | 0.08 | 0.08 | 0 |
| agatha_physics | 0.038 | 0.148 | 1.024 | 0.946 | 0.122 (1281) | 11.3 | 1.000 (0/0/1/4954) | 1.000 | 0.793 (2612/6/1022/1315) | - | 0.02 | 0.02 | 0 |
| old_mlp | 0.182 | 0.418 | 1.359 | 1.257 | 0.318 (1281) | 25.1 | 0.991 (0/45/1/4909) | 1.000 | 0.519 (1288/39/2346/1282) | - | 3.95 | 1.99 | 0 |
| old_mlp_projected | 0.181 | 0.415 | 1.347 | 1.247 | 0.284 (1281) | 22.3 | 0.993 (0/34/1/4920) | 1.000 | 0.524 (1310/35/2324/1286) | - | 0.00 | 0.00 | 0 |
| new_raw | 0.047 | 0.177 | 1.069 | 0.988 | 0.137 (1281) | 12.4 | 1.000 (0/0/1/4954) | 1.000 | 0.793 (2612/6/1022/1315) | - | 0.18 | 0.18 | 0 |
| new_projected | 0.046 | 0.176 | 1.068 | 0.989 | 0.135 (1281) | 12.4 | 1.000 (0/0/1/4954) | 1.000 | 0.793 (2612/6/1022/1315) | - | 0.00 | 0.00 | 0 |

### clevrer / ground = floor / box_like_objects (1505 objects in 832 episodes)

| engine | pos 0.5s | pos 1s | rest pos | rest pos median | rest pos settled (n) | ori final (deg) | tipped acc (TP/FP/FN/TN) | tipped maj | moved-far acc (TP/FP/FN/TN) | fell false alarms | pen mean (mm) | pen capped (mm) | diverged |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| reference | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 (443) | 0.0 | 1.000 (1/0/0/1504) | 0.999 | 1.000 (1081/0/0/424) | - | 1.07 | 1.07 | 0 |
| hold_initial | 0.252 | 0.476 | 1.269 | 1.234 | 0.293 (443) | 35.7 | 0.999 (0/0/1/1504) | 0.999 | 0.282 (0/0/1081/424) | - | 0.11 | 0.11 | 0 |
| agatha_physics | 0.037 | 0.141 | 0.923 | 0.877 | 0.165 (443) | 35.8 | 0.999 (0/0/1/1504) | 0.999 | 0.815 (803/1/278/423) | - | 0.02 | 0.02 | 0 |
| old_mlp | 0.185 | 0.417 | 1.270 | 1.209 | 0.379 (443) | 42.9 | 0.997 (0/4/1/1500) | 0.999 | 0.474 (297/7/784/417) | - | 4.10 | 1.78 | 0 |
| old_mlp_projected | 0.183 | 0.412 | 1.253 | 1.206 | 0.338 (443) | 41.1 | 0.999 (0/0/1/1504) | 0.999 | 0.476 (301/8/780/416) | - | 0.00 | 0.00 | 0 |
| new_raw | 0.042 | 0.159 | 0.956 | 0.903 | 0.177 (443) | 36.4 | 0.999 (0/0/1/1504) | 0.999 | 0.815 (803/1/278/423) | - | 0.13 | 0.13 | 0 |
| new_projected | 0.042 | 0.159 | 0.956 | 0.902 | 0.177 (443) | 36.4 | 0.999 (0/0/1/1504) | 0.999 | 0.815 (803/1/278/423) | - | 0.01 | 0.01 | 0 |

### clevrer / ground = floor / box_like_only_episodes (17 objects in 4 episodes)

| engine | pos 0.5s | pos 1s | rest pos | rest pos median | rest pos settled (n) | ori final (deg) | tipped acc (TP/FP/FN/TN) | tipped maj | moved-far acc (TP/FP/FN/TN) | fell false alarms | pen mean (mm) | pen capped (mm) | diverged |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| reference | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 (2) | 0.0 | 1.000 (0/0/0/17) | 1.000 | 1.000 (16/0/0/1) | - | 0.31 | 0.31 | 0 |
| hold_initial | 0.330 | 0.631 | 1.766 | 1.966 | 0.792 (2) | 18.2 | 1.000 (0/0/0/17) | 1.000 | 0.059 (0/0/16/1) | - | 0.12 | 0.12 | 0 |
| agatha_physics | 0.048 | 0.188 | 1.295 | 1.287 | 0.437 (2) | 19.1 | 1.000 (0/0/0/17) | 1.000 | 0.765 (12/0/4/1) | - | 0.03 | 0.03 | 0 |
| old_mlp | 0.239 | 0.548 | 1.741 | 1.965 | 0.838 (2) | 25.1 | 1.000 (0/0/0/17) | 1.000 | 0.294 (4/0/12/1) | - | 0.15 | 0.15 | 0 |
| old_mlp_projected | 0.237 | 0.543 | 1.739 | 1.969 | 0.839 (2) | 24.6 | 1.000 (0/0/0/17) | 1.000 | 0.294 (4/0/12/1) | - | 0.00 | 0.00 | 0 |
| new_raw | 0.055 | 0.211 | 1.339 | 1.353 | 0.461 (2) | 18.8 | 1.000 (0/0/0/17) | 1.000 | 0.765 (12/0/4/1) | - | 0.12 | 0.12 | 0 |
| new_projected | 0.054 | 0.211 | 1.339 | 1.353 | 0.461 (2) | 18.8 | 1.000 (0/0/0/17) | 1.000 | 0.765 (12/0/4/1) | - | 0.00 | 0.00 | 0 |

### clevrer / paired episode bootstrap (B - A; negative = engine B is better)

| ground | subset | metric | A -> B | A | B | delta | 95% CI | episodes |
|---|---|---|---|---|---|---|---|---|
| slab | all_objects | rest_pos_err_m | old_mlp -> new_projected | 7.345 | 0.869 | -6.477 | [-6.624, -6.319] | 1000 |
| slab | all_objects | rest_pos_err_m | old_mlp -> new_raw | 7.345 | 0.879 | -6.466 | [-6.614, -6.309] | 1000 |
| slab | all_objects | rest_pos_err_m | old_mlp_projected -> new_projected | 3.950 | 0.869 | -3.081 | [-3.177, -2.994] | 1000 |
| slab | all_objects | rest_pos_err_m | hold_initial -> new_projected | 1.359 | 0.869 | -0.491 | [-0.501, -0.479] | 1000 |
| slab | all_objects | rest_pos_err_m | agatha_physics -> new_projected | 0.455 | 0.869 | 0.413 | [0.402, 0.424] | 1000 |
| slab | box_like_objects | rest_pos_err_m | old_mlp -> new_projected | 7.297 | 0.720 | -6.577 | [-6.762, -6.380] | 832 |
| slab | box_like_objects | rest_pos_err_m | old_mlp -> new_raw | 7.297 | 0.728 | -6.569 | [-6.754, -6.372] | 832 |
| slab | box_like_objects | rest_pos_err_m | old_mlp_projected -> new_projected | 3.962 | 0.720 | -3.243 | [-3.416, -3.072] | 832 |
| slab | box_like_objects | rest_pos_err_m | hold_initial -> new_projected | 1.269 | 0.720 | -0.549 | [-0.576, -0.520] | 832 |
| slab | box_like_objects | rest_pos_err_m | agatha_physics -> new_projected | 0.369 | 0.720 | 0.351 | [0.332, 0.369] | 832 |
| slab | box_like_only_episodes | rest_pos_err_m | old_mlp -> new_projected | 6.081 | 0.995 | -5.087 | [-6.409, -4.130] | 4 |
| slab | box_like_only_episodes | rest_pos_err_m | old_mlp -> new_raw | 6.081 | 1.007 | -5.075 | [-6.398, -4.117] | 4 |
| slab | box_like_only_episodes | rest_pos_err_m | old_mlp_projected -> new_projected | 3.083 | 0.995 | -2.089 | [-3.026, -1.256] | 4 |
| slab | box_like_only_episodes | rest_pos_err_m | hold_initial -> new_projected | 1.766 | 0.995 | -0.772 | [-0.810, -0.729] | 4 |
| slab | box_like_only_episodes | rest_pos_err_m | agatha_physics -> new_projected | 0.457 | 0.995 | 0.538 | [0.399, 0.695] | 4 |
| floor | all_objects | rest_pos_err_m | old_mlp -> new_projected | 1.359 | 1.068 | -0.291 | [-0.298, -0.283] | 1000 |
| floor | all_objects | rest_pos_err_m | old_mlp -> new_raw | 1.359 | 1.069 | -0.290 | [-0.297, -0.283] | 1000 |
| floor | all_objects | rest_pos_err_m | old_mlp_projected -> new_projected | 1.347 | 1.068 | -0.279 | [-0.286, -0.271] | 1000 |
| floor | all_objects | rest_pos_err_m | hold_initial -> new_projected | 1.359 | 1.068 | -0.291 | [-0.297, -0.285] | 1000 |
| floor | all_objects | rest_pos_err_m | agatha_physics -> new_projected | 1.024 | 1.068 | 0.045 | [0.043, 0.046] | 1000 |
| floor | box_like_objects | rest_pos_err_m | old_mlp -> new_projected | 1.270 | 0.956 | -0.314 | [-0.328, -0.301] | 832 |
| floor | box_like_objects | rest_pos_err_m | old_mlp -> new_raw | 1.270 | 0.956 | -0.314 | [-0.328, -0.301] | 832 |
| floor | box_like_objects | rest_pos_err_m | old_mlp_projected -> new_projected | 1.253 | 0.956 | -0.297 | [-0.311, -0.284] | 832 |
| floor | box_like_objects | rest_pos_err_m | hold_initial -> new_projected | 1.269 | 0.956 | -0.313 | [-0.327, -0.298] | 832 |
| floor | box_like_objects | rest_pos_err_m | agatha_physics -> new_projected | 0.923 | 0.956 | 0.033 | [0.031, 0.035] | 832 |
| floor | box_like_only_episodes | rest_pos_err_m | old_mlp -> new_projected | 1.741 | 1.339 | -0.402 | [-0.461, -0.351] | 4 |
| floor | box_like_only_episodes | rest_pos_err_m | old_mlp -> new_raw | 1.741 | 1.339 | -0.403 | [-0.461, -0.351] | 4 |
| floor | box_like_only_episodes | rest_pos_err_m | old_mlp_projected -> new_projected | 1.739 | 1.339 | -0.400 | [-0.459, -0.346] | 4 |
| floor | box_like_only_episodes | rest_pos_err_m | hold_initial -> new_projected | 1.766 | 1.339 | -0.427 | [-0.466, -0.402] | 4 |
| floor | box_like_only_episodes | rest_pos_err_m | agatha_physics -> new_projected | 1.295 | 1.339 | 0.044 | [0.038, 0.053] | 4 |

## kubric_movi_a

Evaluated 2026-11-15 05:52:42 with contact_gnn checkpoint sha256 7c53e6b60107 (epoch 115, file written 2026-11-15 05:27:01), contact_gnn.py sha256 7f4ce7fcf5f5.

1137 episodes, 7414 dynamic objects (2457 box-like, 6 box-like-only episodes); horizon 1.15-1.15 s; ground truth: tipped 0.597, moved far 1.000, settled at the end 0.089. Preferred ground mapping for new_projected (lower rest-position error, all objects): **slab**.

### kubric_movi_a / ground = slab / all_objects (7414 objects in 1137 episodes)

| engine | pos 0.5s | pos 1s | rest pos | rest pos median | rest pos settled (n) | ori final (deg) | tipped acc (TP/FP/FN/TN) | tipped maj | moved-far acc (TP/FP/FN/TN) | fell false alarms | pen mean (mm) | pen capped (mm) | diverged |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| reference | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 (660) | 0.0 | 1.000 (4429/0/0/2985) | 0.597 | 1.000 (7414/0/0/0) | 0 | 13.15 | 10.83 | 0 |
| hold_initial | 1.321 | 1.897 | 2.032 | 1.849 | 1.142 (660) | 71.5 | 0.403 (0/0/4429/2985) | 0.597 | 0.000 (0/0/7414/0) | 0 | 6.86 | 5.08 | 0 |
| agatha_physics | 0.213 | 0.557 | 0.649 | 0.489 | 0.356 (660) | 88.7 | 0.780 (3819/1024/610/1961) | 0.597 | 1.000 (7414/0/0/0) | 1 | 0.34 | 0.34 | 0 |
| old_mlp | 0.925 | 1.455 | 1.589 | 1.442 | 0.729 (660) | 91.1 | 0.484 (2827/2224/1602/761) | 0.597 | 0.999 (7408/0/6/0) | 0 | 33.52 | 15.38 | 0 |
| old_mlp_projected | 0.926 | 1.454 | 1.588 | 1.441 | 0.722 (660) | 90.3 | 0.524 (2855/1955/1574/1030) | 0.597 | 0.999 (7409/0/5/0) | 0 | 0.17 | 0.17 | 0 |
| new_raw | 0.310 | 0.786 | 0.911 | 0.769 | 0.408 (660) | 78.6 | 0.742 (3088/570/1341/2415) | 0.597 | 1.000 (7414/0/0/0) | 0 | 5.74 | 5.05 | 0 |
| new_projected | 0.289 | 0.738 | 0.857 | 0.709 | 0.391 (660) | 76.6 | 0.780 (3209/413/1220/2572) | 0.597 | 1.000 (7414/0/0/0) | 0 | 0.11 | 0.11 | 0 |

### kubric_movi_a / ground = slab / box_like_objects (2457 objects in 1001 episodes)

| engine | pos 0.5s | pos 1s | rest pos | rest pos median | rest pos settled (n) | ori final (deg) | tipped acc (TP/FP/FN/TN) | tipped maj | moved-far acc (TP/FP/FN/TN) | fell false alarms | pen mean (mm) | pen capped (mm) | diverged |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| reference | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 (551) | 0.0 | 1.000 (2054/0/0/403) | 0.836 | 1.000 (2457/0/0/0) | 0 | 4.64 | 3.97 | 0 |
| hold_initial | 1.312 | 1.882 | 2.001 | 1.796 | 1.150 (551) | 99.9 | 0.164 (0/0/2054/403) | 0.836 | 0.000 (0/0/2457/0) | 0 | 3.29 | 2.79 | 0 |
| agatha_physics | 0.171 | 0.451 | 0.518 | 0.369 | 0.350 (551) | 87.1 | 0.891 (1872/87/182/316) | 0.836 | 1.000 (2457/0/0/0) | 0 | 0.33 | 0.33 | 0 |
| old_mlp | 0.919 | 1.440 | 1.554 | 1.377 | 0.737 (551) | 107.9 | 0.593 (1327/274/727/129) | 0.836 | 0.999 (2455/0/2/0) | 0 | 32.48 | 15.12 | 0 |
| old_mlp_projected | 0.921 | 1.440 | 1.552 | 1.379 | 0.730 (551) | 107.8 | 0.610 (1345/249/709/154) | 0.836 | 0.999 (2455/0/2/0) | 0 | 0.16 | 0.16 | 0 |
| new_raw | 0.302 | 0.739 | 0.840 | 0.693 | 0.409 (551) | 97.1 | 0.741 (1502/84/552/319) | 0.836 | 1.000 (2457/0/0/0) | 0 | 6.04 | 5.35 | 0 |
| new_projected | 0.273 | 0.674 | 0.767 | 0.630 | 0.394 (551) | 97.4 | 0.760 (1559/94/495/309) | 0.836 | 1.000 (2457/0/0/0) | 0 | 0.11 | 0.11 | 0 |

### kubric_movi_a / ground = slab / box_like_only_episodes (19 objects in 6 episodes)

| engine | pos 0.5s | pos 1s | rest pos | rest pos median | rest pos settled (n) | ori final (deg) | tipped acc (TP/FP/FN/TN) | tipped maj | moved-far acc (TP/FP/FN/TN) | fell false alarms | pen mean (mm) | pen capped (mm) | diverged |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| reference | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 (2) | 0.0 | 1.000 (14/0/0/5) | 0.737 | 1.000 (19/0/0/0) | 0 | 0.26 | 0.26 | 0 |
| hold_initial | 1.722 | 2.653 | 2.864 | 2.676 | 1.220 (2) | 113.6 | 0.263 (0/0/14/5) | 0.737 | 0.000 (0/0/19/0) | 0 | 0.00 | 0.00 | 0 |
| agatha_physics | 0.083 | 0.256 | 0.302 | 0.284 | 0.214 (2) | 104.9 | 0.947 (13/0/1/5) | 0.737 | 1.000 (19/0/0/0) | 0 | 0.27 | 0.27 | 0 |
| old_mlp | 1.227 | 2.101 | 2.304 | 2.185 | 0.885 (2) | 126.5 | 0.526 (9/4/5/1) | 0.737 | 1.000 (19/0/0/0) | 0 | 34.92 | 10.79 | 0 |
| old_mlp_projected | 1.212 | 2.062 | 2.261 | 2.185 | 0.885 (2) | 126.4 | 0.474 (8/4/6/1) | 0.737 | 1.000 (19/0/0/0) | 0 | 0.00 | 0.00 | 0 |
| new_raw | 0.232 | 0.648 | 0.777 | 0.720 | 0.340 (2) | 110.3 | 0.737 (10/1/4/4) | 0.737 | 1.000 (19/0/0/0) | 0 | 6.82 | 6.12 | 0 |
| new_projected | 0.182 | 0.523 | 0.624 | 0.485 | 0.313 (2) | 110.2 | 0.737 (12/3/2/2) | 0.737 | 1.000 (19/0/0/0) | 0 | 0.00 | 0.00 | 0 |

### kubric_movi_a / ground = floor / all_objects (7414 objects in 1137 episodes)

| engine | pos 0.5s | pos 1s | rest pos | rest pos median | rest pos settled (n) | ori final (deg) | tipped acc (TP/FP/FN/TN) | tipped maj | moved-far acc (TP/FP/FN/TN) | fell false alarms | pen mean (mm) | pen capped (mm) | diverged |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| reference | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 (660) | 0.0 | 1.000 (4429/0/0/2985) | 0.597 | 1.000 (7414/0/0/0) | - | 15.52 | 11.98 | 0 |
| hold_initial | 1.321 | 1.897 | 2.032 | 1.849 | 1.142 (660) | 71.5 | 0.403 (0/0/4429/2985) | 0.597 | 0.000 (0/0/7414/0) | - | 6.86 | 5.08 | 0 |
| agatha_physics | 0.234 | 0.657 | 0.779 | 0.662 | 0.325 (660) | 85.4 | 0.734 (3327/867/1102/2118) | 0.597 | 1.000 (7414/0/0/0) | - | 0.64 | 0.64 | 0 |
| old_mlp | 0.454 | 1.030 | 1.177 | 1.037 | 0.535 (660) | 94.1 | 0.614 (3666/2099/763/886) | 0.597 | 0.999 (7405/0/9/0) | - | 65.42 | 28.85 | 0 |
| old_mlp_projected | 0.413 | 0.972 | 1.118 | 0.990 | 0.486 (660) | 93.0 | 0.665 (3708/1760/721/1225) | 0.597 | 0.999 (7407/0/7/0) | - | 0.43 | 0.39 | 0 |
| new_raw | 0.311 | 0.808 | 0.940 | 0.797 | 0.404 (660) | 78.5 | 0.731 (2980/545/1449/2440) | 0.597 | 1.000 (7414/0/0/0) | - | 4.92 | 4.22 | 0 |
| new_projected | 0.292 | 0.770 | 0.899 | 0.762 | 0.383 (660) | 76.5 | 0.765 (3081/392/1348/2593) | 0.597 | 1.000 (7414/0/0/0) | - | 0.11 | 0.11 | 0 |

### kubric_movi_a / ground = floor / box_like_objects (2457 objects in 1001 episodes)

| engine | pos 0.5s | pos 1s | rest pos | rest pos median | rest pos settled (n) | ori final (deg) | tipped acc (TP/FP/FN/TN) | tipped maj | moved-far acc (TP/FP/FN/TN) | fell false alarms | pen mean (mm) | pen capped (mm) | diverged |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| reference | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 (551) | 0.0 | 1.000 (2054/0/0/403) | 0.836 | 1.000 (2457/0/0/0) | - | 4.64 | 3.97 | 0 |
| hold_initial | 1.312 | 1.882 | 2.001 | 1.796 | 1.150 (551) | 99.9 | 0.164 (0/0/2054/403) | 0.836 | 0.000 (0/0/2457/0) | - | 3.29 | 2.79 | 0 |
| agatha_physics | 0.197 | 0.569 | 0.668 | 0.560 | 0.323 (551) | 91.1 | 0.796 (1614/61/440/342) | 0.836 | 1.000 (2457/0/0/0) | - | 0.63 | 0.63 | 0 |
| old_mlp | 0.455 | 1.012 | 1.140 | 0.996 | 0.544 (551) | 110.9 | 0.723 (1703/329/351/74) | 0.836 | 0.999 (2454/0/3/0) | - | 65.68 | 28.88 | 0 |
| old_mlp_projected | 0.408 | 0.951 | 1.078 | 0.935 | 0.485 (551) | 110.6 | 0.740 (1737/322/317/81) | 0.836 | 0.998 (2453/0/4/0) | - | 0.41 | 0.39 | 0 |
| new_raw | 0.302 | 0.758 | 0.867 | 0.727 | 0.407 (551) | 97.3 | 0.731 (1468/76/586/327) | 0.836 | 1.000 (2457/0/0/0) | - | 5.19 | 4.46 | 0 |
| new_projected | 0.275 | 0.703 | 0.808 | 0.676 | 0.385 (551) | 97.3 | 0.753 (1518/72/536/331) | 0.836 | 1.000 (2457/0/0/0) | - | 0.11 | 0.11 | 0 |

### kubric_movi_a / ground = floor / box_like_only_episodes (19 objects in 6 episodes)

| engine | pos 0.5s | pos 1s | rest pos | rest pos median | rest pos settled (n) | ori final (deg) | tipped acc (TP/FP/FN/TN) | tipped maj | moved-far acc (TP/FP/FN/TN) | fell false alarms | pen mean (mm) | pen capped (mm) | diverged |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| reference | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 (2) | 0.0 | 1.000 (14/0/0/5) | 0.737 | 1.000 (19/0/0/0) | - | 0.26 | 0.26 | 0 |
| hold_initial | 1.722 | 2.653 | 2.864 | 2.676 | 1.220 (2) | 113.6 | 0.263 (0/0/14/5) | 0.737 | 0.000 (0/0/19/0) | - | 0.00 | 0.00 | 0 |
| agatha_physics | 0.141 | 0.567 | 0.707 | 0.709 | 0.348 (2) | 101.7 | 0.789 (11/1/3/4) | 0.737 | 1.000 (19/0/0/0) | - | 0.66 | 0.66 | 0 |
| old_mlp | 0.537 | 1.300 | 1.510 | 1.347 | 0.482 (2) | 116.9 | 0.632 (11/4/3/1) | 0.737 | 1.000 (19/0/0/0) | - | 46.57 | 21.19 | 0 |
| old_mlp_projected | 0.439 | 1.138 | 1.329 | 1.294 | 0.465 (2) | 113.9 | 0.579 (10/4/4/1) | 0.737 | 1.000 (19/0/0/0) | - | 0.00 | 0.00 | 0 |
| new_raw | 0.240 | 0.719 | 0.872 | 0.812 | 0.344 (2) | 105.1 | 0.842 (12/1/2/4) | 0.737 | 1.000 (19/0/0/0) | - | 3.76 | 3.26 | 0 |
| new_projected | 0.191 | 0.621 | 0.757 | 0.661 | 0.323 (2) | 106.2 | 0.737 (11/2/3/3) | 0.737 | 1.000 (19/0/0/0) | - | 0.00 | 0.00 | 0 |

### kubric_movi_a / paired episode bootstrap (B - A; negative = engine B is better)

| ground | subset | metric | A -> B | A | B | delta | 95% CI | episodes |
|---|---|---|---|---|---|---|---|---|
| slab | all_objects | rest_pos_err_m | old_mlp -> new_projected | 1.589 | 0.857 | -0.732 | [-0.757, -0.712] | 1137 |
| slab | all_objects | rest_pos_err_m | old_mlp -> new_raw | 1.589 | 0.911 | -0.678 | [-0.701, -0.658] | 1137 |
| slab | all_objects | rest_pos_err_m | old_mlp_projected -> new_projected | 1.588 | 0.857 | -0.731 | [-0.756, -0.710] | 1137 |
| slab | all_objects | rest_pos_err_m | hold_initial -> new_projected | 2.032 | 0.857 | -1.175 | [-1.204, -1.152] | 1137 |
| slab | all_objects | rest_pos_err_m | agatha_physics -> new_projected | 0.649 | 0.857 | 0.208 | [0.196, 0.222] | 1137 |
| slab | box_like_objects | rest_pos_err_m | old_mlp -> new_projected | 1.554 | 0.767 | -0.786 | [-0.824, -0.752] | 1001 |
| slab | box_like_objects | rest_pos_err_m | old_mlp -> new_raw | 1.554 | 0.840 | -0.714 | [-0.748, -0.680] | 1001 |
| slab | box_like_objects | rest_pos_err_m | old_mlp_projected -> new_projected | 1.552 | 0.767 | -0.785 | [-0.825, -0.750] | 1001 |
| slab | box_like_objects | rest_pos_err_m | hold_initial -> new_projected | 2.001 | 0.767 | -1.234 | [-1.275, -1.196] | 1001 |
| slab | box_like_objects | rest_pos_err_m | agatha_physics -> new_projected | 0.518 | 0.767 | 0.250 | [0.228, 0.269] | 1001 |
| slab | box_like_only_episodes | rest_pos_err_m | old_mlp -> new_projected | 2.304 | 0.624 | -1.680 | [-2.050, -1.247] | 6 |
| slab | box_like_only_episodes | rest_pos_err_m | old_mlp -> new_raw | 2.304 | 0.777 | -1.527 | [-1.761, -1.171] | 6 |
| slab | box_like_only_episodes | rest_pos_err_m | old_mlp_projected -> new_projected | 2.261 | 0.624 | -1.637 | [-1.971, -1.188] | 6 |
| slab | box_like_only_episodes | rest_pos_err_m | hold_initial -> new_projected | 2.864 | 0.624 | -2.240 | [-2.649, -1.680] | 6 |
| slab | box_like_only_episodes | rest_pos_err_m | agatha_physics -> new_projected | 0.302 | 0.624 | 0.322 | [0.163, 0.520] | 6 |
| floor | all_objects | rest_pos_err_m | old_mlp -> new_projected | 1.177 | 0.899 | -0.278 | [-0.291, -0.266] | 1137 |
| floor | all_objects | rest_pos_err_m | old_mlp -> new_raw | 1.177 | 0.940 | -0.237 | [-0.250, -0.225] | 1137 |
| floor | all_objects | rest_pos_err_m | old_mlp_projected -> new_projected | 1.118 | 0.899 | -0.219 | [-0.233, -0.205] | 1137 |
| floor | all_objects | rest_pos_err_m | hold_initial -> new_projected | 2.032 | 0.899 | -1.133 | [-1.159, -1.112] | 1137 |
| floor | all_objects | rest_pos_err_m | agatha_physics -> new_projected | 0.779 | 0.899 | 0.120 | [0.108, 0.132] | 1137 |
| floor | box_like_objects | rest_pos_err_m | old_mlp -> new_projected | 1.140 | 0.808 | -0.332 | [-0.356, -0.310] | 1001 |
| floor | box_like_objects | rest_pos_err_m | old_mlp -> new_raw | 1.140 | 0.867 | -0.273 | [-0.295, -0.251] | 1001 |
| floor | box_like_objects | rest_pos_err_m | old_mlp_projected -> new_projected | 1.078 | 0.808 | -0.270 | [-0.295, -0.247] | 1001 |
| floor | box_like_objects | rest_pos_err_m | hold_initial -> new_projected | 2.001 | 0.808 | -1.194 | [-1.230, -1.158] | 1001 |
| floor | box_like_objects | rest_pos_err_m | agatha_physics -> new_projected | 0.668 | 0.808 | 0.139 | [0.120, 0.157] | 1001 |
| floor | box_like_only_episodes | rest_pos_err_m | old_mlp -> new_projected | 1.510 | 0.757 | -0.753 | [-1.034, -0.316] | 6 |
| floor | box_like_only_episodes | rest_pos_err_m | old_mlp -> new_raw | 1.510 | 0.872 | -0.638 | [-0.865, -0.274] | 6 |
| floor | box_like_only_episodes | rest_pos_err_m | old_mlp_projected -> new_projected | 1.329 | 0.757 | -0.572 | [-0.828, -0.222] | 6 |
| floor | box_like_only_episodes | rest_pos_err_m | hold_initial -> new_projected | 2.864 | 0.757 | -2.107 | [-2.493, -1.570] | 6 |
| floor | box_like_only_episodes | rest_pos_err_m | agatha_physics -> new_projected | 0.707 | 0.757 | 0.050 | [-0.072, 0.214] | 6 |

## physion

Evaluated 2026-11-15 05:52:42 with contact_gnn checkpoint sha256 7c53e6b60107 (epoch 115, file written 2026-11-15 05:27:01), contact_gnn.py sha256 7f4ce7fcf5f5.

10 episodes, 47 dynamic objects (43 box-like, 8 box-like-only episodes); horizon 0.8-4 s; ground truth: tipped 0.851, moved far 0.191, settled at the end 0.957. Preferred ground mapping for new_projected (lower rest-position error, all objects): **floor**.

### physion / ground = slab / all_objects (47 objects in 10 episodes)

| engine | pos 0.5s | pos 1s | rest pos | rest pos median | rest pos settled (n) | ori final (deg) | tipped acc (TP/FP/FN/TN) | tipped maj | moved-far acc (TP/FP/FN/TN) | fell false alarms | pen mean (mm) | pen capped (mm) | diverged |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| reference | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 (45) | 0.0 | 1.000 (40/0/0/7) | 0.851 | 1.000 (9/0/0/38) | 0 | 1.79 | 1.63 | 0 |
| hold_initial | 0.131 | 0.255 | 0.376 | 0.181 | 0.309 (45) | 75.4 | 0.149 (0/0/40/7) | 0.851 | 0.809 (0/0/9/38) | 0 | 0.29 | 0.29 | 0 |
| agatha_physics | 0.185 | 0.400 | 0.856 | 0.097 | 0.805 (45) | 41.0 | 0.851 (35/2/5/5) | 0.851 | 0.872 (9/6/0/32) | 3 | 0.30 | 0.30 | 0 |
| old_mlp | 0.352 | 0.833 | 2.080 | 1.297 | 2.043 (45) | 112.6 | 0.787 (37/7/3/0) | 0.851 | 0.191 (9/38/0/0) | 0 | 419.37 | 28.01 | 0 |
| old_mlp_projected | 0.306 | 0.824 | 2.173 | 1.268 | 2.112 (45) | 81.4 | 0.426 (16/3/24/4) | 0.851 | 0.191 (9/38/0/0) | 6 | 0.56 | 0.51 | 0 |
| new_raw | 0.173 | 0.335 | 0.615 | 0.162 | 0.550 (45) | 71.8 | 0.362 (13/3/27/4) | 0.851 | 0.894 (9/5/0/33) | 2 | 4.23 | 4.15 | 0 |
| new_projected | 0.175 | 0.370 | 0.578 | 0.164 | 0.529 (45) | 73.8 | 0.383 (14/3/26/4) | 0.851 | 0.915 (9/4/0/34) | 2 | 0.04 | 0.04 | 0 |

### physion / ground = slab / box_like_objects (43 objects in 10 episodes)

| engine | pos 0.5s | pos 1s | rest pos | rest pos median | rest pos settled (n) | ori final (deg) | tipped acc (TP/FP/FN/TN) | tipped maj | moved-far acc (TP/FP/FN/TN) | fell false alarms | pen mean (mm) | pen capped (mm) | diverged |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| reference | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 (43) | 0.0 | 1.000 (39/0/0/4) | 0.907 | 1.000 (5/0/0/38) | 0 | 0.44 | 0.44 | 0 |
| hold_initial | 0.092 | 0.169 | 0.251 | 0.177 | 0.251 (43) | 79.7 | 0.093 (0/0/39/4) | 0.907 | 0.884 (0/0/5/38) | 0 | 0.26 | 0.26 | 0 |
| agatha_physics | 0.146 | 0.302 | 0.515 | 0.089 | 0.515 (43) | 37.6 | 0.884 (34/0/5/4) | 0.907 | 0.860 (5/6/0/32) | 1 | 0.28 | 0.28 | 0 |
| old_mlp | 0.328 | 0.802 | 1.851 | 1.120 | 1.851 (43) | 111.5 | 0.837 (36/4/3/0) | 0.907 | 0.116 (5/38/0/0) | 0 | 381.72 | 27.15 | 0 |
| old_mlp_projected | 0.283 | 0.777 | 1.937 | 1.049 | 1.937 (43) | 78.1 | 0.419 (15/1/24/3) | 0.907 | 0.116 (5/38/0/0) | 5 | 0.55 | 0.50 | 0 |
| new_raw | 0.149 | 0.284 | 0.431 | 0.158 | 0.431 (43) | 69.8 | 0.349 (12/1/27/3) | 0.907 | 0.884 (5/5/0/33) | 1 | 4.72 | 4.65 | 0 |
| new_projected | 0.146 | 0.306 | 0.441 | 0.162 | 0.441 (43) | 71.7 | 0.372 (13/1/26/3) | 0.907 | 0.907 (5/4/0/34) | 1 | 0.02 | 0.02 | 0 |

### physion / ground = slab / box_like_only_episodes (34 objects in 8 episodes)

| engine | pos 0.5s | pos 1s | rest pos | rest pos median | rest pos settled (n) | ori final (deg) | tipped acc (TP/FP/FN/TN) | tipped maj | moved-far acc (TP/FP/FN/TN) | fell false alarms | pen mean (mm) | pen capped (mm) | diverged |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| reference | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 (34) | 0.0 | 1.000 (32/0/0/2) | 0.941 | 1.000 (0/0/0/34) | 0 | 0.06 | 0.06 | 0 |
| hold_initial | 0.090 | 0.137 | 0.161 | 0.169 | 0.161 (34) | 76.2 | 0.059 (0/0/32/2) | 0.941 | 1.000 (0/0/0/34) | 0 | 0.00 | 0.00 | 0 |
| agatha_physics | 0.061 | 0.111 | 0.120 | 0.058 | 0.120 (34) | 20.7 | 0.912 (29/0/3/2) | 0.941 | 0.912 (0/3/0/31) | 0 | 0.28 | 0.28 | 0 |
| old_mlp | 0.318 | 0.759 | 1.011 | 0.943 | 1.011 (34) | 107.6 | 0.853 (29/2/3/0) | 0.941 | 0.000 (0/34/0/0) | 0 | 41.41 | 18.75 | 0 |
| old_mlp_projected | 0.276 | 0.724 | 0.975 | 0.910 | 0.975 (34) | 67.5 | 0.294 (8/0/24/2) | 0.941 | 0.000 (0/34/0/0) | 0 | 0.01 | 0.01 | 0 |
| new_raw | 0.053 | 0.102 | 0.124 | 0.142 | 0.124 (34) | 62.7 | 0.294 (8/0/24/2) | 0.941 | 0.971 (0/1/0/33) | 0 | 0.34 | 0.34 | 0 |
| new_projected | 0.051 | 0.097 | 0.120 | 0.129 | 0.120 (34) | 61.8 | 0.294 (8/0/24/2) | 0.941 | 1.000 (0/0/0/34) | 0 | 0.00 | 0.00 | 0 |

### physion / ground = floor / all_objects (47 objects in 10 episodes)

| engine | pos 0.5s | pos 1s | rest pos | rest pos median | rest pos settled (n) | ori final (deg) | tipped acc (TP/FP/FN/TN) | tipped maj | moved-far acc (TP/FP/FN/TN) | fell false alarms | pen mean (mm) | pen capped (mm) | diverged |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| reference | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 (45) | 0.0 | 1.000 (40/0/0/7) | 0.851 | 1.000 (9/0/0/38) | - | 3.63 | 1.97 | 0 |
| hold_initial | 0.131 | 0.255 | 0.376 | 0.181 | 0.309 (45) | 75.4 | 0.149 (0/0/40/7) | 0.851 | 0.809 (0/0/9/38) | - | 0.29 | 0.29 | 0 |
| agatha_physics | 0.184 | 0.362 | 0.873 | 0.097 | 0.809 (45) | 41.8 | 0.872 (35/1/5/6) | 0.851 | 0.872 (9/6/0/32) | - | 0.27 | 0.27 | 0 |
| old_mlp | 0.219 | 0.341 | 0.836 | 0.211 | 0.804 (45) | 80.4 | 0.128 (1/2/39/5) | 0.851 | 0.957 (8/1/1/37) | - | 218.22 | 26.63 | 0 |
| old_mlp_projected | 0.165 | 0.349 | 0.759 | 0.188 | 0.692 (45) | 81.9 | 0.234 (6/2/34/5) | 0.851 | 0.936 (9/3/0/35) | - | 2.08 | 1.67 | 0 |
| new_raw | 0.179 | 0.349 | 0.561 | 0.163 | 0.503 (45) | 70.4 | 0.362 (11/1/29/6) | 0.851 | 0.894 (9/5/0/33) | - | 1.51 | 1.48 | 0 |
| new_projected | 0.176 | 0.386 | 0.532 | 0.162 | 0.481 (45) | 67.9 | 0.404 (13/1/27/6) | 0.851 | 0.894 (9/5/0/33) | - | 0.05 | 0.05 | 0 |

### physion / ground = floor / box_like_objects (43 objects in 10 episodes)

| engine | pos 0.5s | pos 1s | rest pos | rest pos median | rest pos settled (n) | ori final (deg) | tipped acc (TP/FP/FN/TN) | tipped maj | moved-far acc (TP/FP/FN/TN) | fell false alarms | pen mean (mm) | pen capped (mm) | diverged |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| reference | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 (43) | 0.0 | 1.000 (39/0/0/4) | 0.907 | 1.000 (5/0/0/38) | - | 0.44 | 0.44 | 0 |
| hold_initial | 0.092 | 0.169 | 0.251 | 0.177 | 0.251 (43) | 79.7 | 0.093 (0/0/39/4) | 0.907 | 0.884 (0/0/5/38) | - | 0.26 | 0.26 | 0 |
| agatha_physics | 0.145 | 0.269 | 0.523 | 0.089 | 0.523 (43) | 38.4 | 0.884 (34/0/5/4) | 0.907 | 0.860 (5/6/0/32) | - | 0.28 | 0.28 | 0 |
| old_mlp | 0.136 | 0.211 | 0.258 | 0.211 | 0.258 (43) | 81.4 | 0.093 (1/1/38/3) | 0.907 | 0.953 (4/1/1/37) | - | 52.36 | 24.85 | 0 |
| old_mlp_projected | 0.107 | 0.211 | 0.272 | 0.183 | 0.272 (43) | 81.6 | 0.209 (5/0/34/4) | 0.907 | 0.930 (5/3/0/35) | - | 1.59 | 1.26 | 0 |
| new_raw | 0.154 | 0.296 | 0.420 | 0.161 | 0.420 (43) | 69.8 | 0.326 (10/0/29/4) | 0.907 | 0.884 (5/5/0/33) | - | 1.63 | 1.62 | 0 |
| new_projected | 0.145 | 0.316 | 0.409 | 0.158 | 0.409 (43) | 67.7 | 0.372 (12/0/27/4) | 0.907 | 0.884 (5/5/0/33) | - | 0.05 | 0.05 | 0 |

### physion / ground = floor / box_like_only_episodes (34 objects in 8 episodes)

| engine | pos 0.5s | pos 1s | rest pos | rest pos median | rest pos settled (n) | ori final (deg) | tipped acc (TP/FP/FN/TN) | tipped maj | moved-far acc (TP/FP/FN/TN) | fell false alarms | pen mean (mm) | pen capped (mm) | diverged |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| reference | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 (34) | 0.0 | 1.000 (32/0/0/2) | 0.941 | 1.000 (0/0/0/34) | - | 0.06 | 0.06 | 0 |
| hold_initial | 0.090 | 0.137 | 0.161 | 0.169 | 0.161 (34) | 76.2 | 0.059 (0/0/32/2) | 0.941 | 1.000 (0/0/0/34) | - | 0.00 | 0.00 | 0 |
| agatha_physics | 0.061 | 0.111 | 0.120 | 0.058 | 0.120 (34) | 20.7 | 0.912 (29/0/3/2) | 0.941 | 0.912 (0/3/0/31) | - | 0.28 | 0.28 | 0 |
| old_mlp | 0.087 | 0.149 | 0.183 | 0.204 | 0.183 (34) | 78.7 | 0.088 (1/0/31/2) | 0.941 | 1.000 (0/0/0/34) | - | 5.09 | 5.09 | 0 |
| old_mlp_projected | 0.076 | 0.133 | 0.164 | 0.175 | 0.164 (34) | 75.7 | 0.059 (0/0/32/2) | 0.941 | 1.000 (0/0/0/34) | - | 0.02 | 0.02 | 0 |
| new_raw | 0.055 | 0.104 | 0.126 | 0.139 | 0.126 (34) | 62.4 | 0.294 (8/0/24/2) | 0.941 | 0.971 (0/1/0/33) | - | 0.34 | 0.34 | 0 |
| new_projected | 0.055 | 0.103 | 0.124 | 0.138 | 0.124 (34) | 61.9 | 0.294 (8/0/24/2) | 0.941 | 0.971 (0/1/0/33) | - | 0.08 | 0.08 | 0 |

### physion / paired episode bootstrap (B - A; negative = engine B is better)

| ground | subset | metric | A -> B | A | B | delta | 95% CI | episodes |
|---|---|---|---|---|---|---|---|---|
| slab | all_objects | rest_pos_err_m | old_mlp -> new_projected | 2.080 | 0.578 | -1.502 | [-2.268, -0.805] | 10 |
| slab | all_objects | rest_pos_err_m | old_mlp -> new_raw | 2.080 | 0.615 | -1.465 | [-2.235, -0.804] | 10 |
| slab | all_objects | rest_pos_err_m | old_mlp_projected -> new_projected | 2.173 | 0.578 | -1.596 | [-2.463, -0.777] | 10 |
| slab | all_objects | rest_pos_err_m | hold_initial -> new_projected | 0.376 | 0.578 | 0.202 | [-0.045, 0.552] | 10 |
| slab | all_objects | rest_pos_err_m | agatha_physics -> new_projected | 0.856 | 0.578 | -0.278 | [-0.628, 0.008] | 10 |
| slab | box_like_objects | rest_pos_err_m | old_mlp -> new_projected | 1.851 | 0.441 | -1.411 | [-2.117, -0.805] | 10 |
| slab | box_like_objects | rest_pos_err_m | old_mlp -> new_raw | 1.851 | 0.431 | -1.420 | [-2.147, -0.804] | 10 |
| slab | box_like_objects | rest_pos_err_m | old_mlp_projected -> new_projected | 1.937 | 0.441 | -1.496 | [-2.554, -0.775] | 10 |
| slab | box_like_objects | rest_pos_err_m | hold_initial -> new_projected | 0.251 | 0.441 | 0.189 | [-0.045, 0.595] | 10 |
| slab | box_like_objects | rest_pos_err_m | agatha_physics -> new_projected | 0.515 | 0.441 | -0.074 | [-0.330, 0.089] | 10 |
| slab | box_like_only_episodes | rest_pos_err_m | old_mlp -> new_projected | 1.011 | 0.120 | -0.891 | [-1.114, -0.650] | 8 |
| slab | box_like_only_episodes | rest_pos_err_m | old_mlp -> new_raw | 1.011 | 0.124 | -0.887 | [-1.111, -0.649] | 8 |
| slab | box_like_only_episodes | rest_pos_err_m | old_mlp_projected -> new_projected | 0.975 | 0.120 | -0.856 | [-1.082, -0.613] | 8 |
| slab | box_like_only_episodes | rest_pos_err_m | hold_initial -> new_projected | 0.161 | 0.120 | -0.042 | [-0.052, -0.034] | 8 |
| slab | box_like_only_episodes | rest_pos_err_m | agatha_physics -> new_projected | 0.120 | 0.120 | -0.000 | [-0.034, 0.032] | 8 |
| floor | all_objects | rest_pos_err_m | old_mlp -> new_projected | 0.836 | 0.532 | -0.305 | [-0.917, 0.070] | 10 |
| floor | all_objects | rest_pos_err_m | old_mlp -> new_raw | 0.836 | 0.561 | -0.275 | [-0.811, 0.054] | 10 |
| floor | all_objects | rest_pos_err_m | old_mlp_projected -> new_projected | 0.759 | 0.532 | -0.227 | [-0.670, 0.041] | 10 |
| floor | all_objects | rest_pos_err_m | hold_initial -> new_projected | 0.376 | 0.532 | 0.156 | [-0.041, 0.414] | 10 |
| floor | all_objects | rest_pos_err_m | agatha_physics -> new_projected | 0.873 | 0.532 | -0.341 | [-0.808, 0.012] | 10 |
| floor | box_like_objects | rest_pos_err_m | old_mlp -> new_projected | 0.258 | 0.409 | 0.151 | [-0.063, 0.511] | 10 |
| floor | box_like_objects | rest_pos_err_m | old_mlp -> new_raw | 0.258 | 0.420 | 0.162 | [-0.063, 0.541] | 10 |
| floor | box_like_objects | rest_pos_err_m | old_mlp_projected -> new_projected | 0.272 | 0.409 | 0.137 | [-0.045, 0.416] | 10 |
| floor | box_like_objects | rest_pos_err_m | hold_initial -> new_projected | 0.251 | 0.409 | 0.157 | [-0.041, 0.484] | 10 |
| floor | box_like_objects | rest_pos_err_m | agatha_physics -> new_projected | 0.523 | 0.409 | -0.114 | [-0.476, 0.111] | 10 |
| floor | box_like_only_episodes | rest_pos_err_m | old_mlp -> new_projected | 0.183 | 0.124 | -0.059 | [-0.074, -0.044] | 8 |
| floor | box_like_only_episodes | rest_pos_err_m | old_mlp -> new_raw | 0.183 | 0.126 | -0.057 | [-0.072, -0.041] | 8 |
| floor | box_like_only_episodes | rest_pos_err_m | old_mlp_projected -> new_projected | 0.164 | 0.124 | -0.040 | [-0.057, -0.026] | 8 |
| floor | box_like_only_episodes | rest_pos_err_m | hold_initial -> new_projected | 0.161 | 0.124 | -0.037 | [-0.048, -0.029] | 8 |
| floor | box_like_only_episodes | rest_pos_err_m | agatha_physics -> new_projected | 0.120 | 0.124 | 0.004 | [-0.033, 0.043] | 8 |

## real_iam_box_toss (REAL WORLD, approximate)

> REAL-WORLD pose tracks (motion capture / AprilTags) of a single tossed box; friction and restitution are assumed (see `materials`), velocities are finite differences, 20 Hz resampling. APPROXIMATE.

Evaluated 2026-11-15 07:37:53 with contact_gnn checkpoint sha256 e4b5e4a3a905 (epoch 246, file written 2026-11-15 06:35:20), contact_gnn.py sha256 7f4ce7fcf5f5.

49 trajectories, 1294 transitions at 20 Hz, horizon 1.15-1.85 s; OptiTrack motion capture, 360 Hz; object: Box009 (foam-filled carton), 0.1644 x 0.1202 x 0.1253 m (Agatha x, y, z), 0.285 kg; surface: idle rubber-belt conveyor. Assumed materials (default): object friction 0.500, support friction 0.500 (effective 0.250), restitution 0.000 (effective 0.000). Ground truth rests on the same face that was down at release in 1.000 of the trajectories (= the rest-face accuracy of `hold_initial`). Tables larger than the training range: 48.

Columns: position error (m) at 0.25 / 0.5 / 1 s (n = trajectories that last that long) and at the last frame (rest); orientation error at the last frame; rest-face accuracy (same body face down as in the measurement); `fell` false alarms (the real box never leaves the surface); penetration (mm).

### real_iam_box_toss / ground = slab (49 trajectories)

| engine | pos 0.25s | pos 0.5s | pos 1s (n) | rest pos mean | rest pos median | ori final (deg) | rest-face acc | fell false alarms | pen mean (mm) | pen capped (mm) | diverged |
|---|---|---|---|---|---|---|---|---|---|---|---|
| reference | 0.000 | 0.000 | 0.000 (49) | 0.000 | 0.000 | 0.0 | 1.000 | 0 | 1.48 | 1.48 | 0 |
| hold_initial | 0.583 | 0.918 | 1.029 (49) | 1.028 | 1.037 | 35.7 | 1.000 | 0 | 0.00 | 0.00 | 0 |
| agatha_physics | 0.021 | 0.104 | 0.360 (49) | 0.476 | 0.352 | 63.8 | 0.633 | 13 | 0.47 | 0.47 | 0 |
| old_mlp | 0.055 | 0.209 | 0.640 (49) | 0.651 | 0.891 | 86.8 | 0.224 | 28 | 33.49 | 28.71 | 0 |
| old_mlp_projected | 0.055 | 0.134 | 0.347 (49) | 0.439 | 0.280 | 51.0 | 0.796 | 10 | 0.00 | 0.00 | 0 |
| new_raw | 0.014 | 0.078 | 0.309 (49) | 0.372 | 0.294 | 48.1 | 0.837 | 8 | 0.16 | 0.16 | 0 |
| new_projected | 0.014 | 0.078 | 0.308 (49) | 0.371 | 0.296 | 47.9 | 0.837 | 8 | 0.00 | 0.00 | 0 |

### real_iam_box_toss / ground = floor (49 trajectories)

| engine | pos 0.25s | pos 0.5s | pos 1s (n) | rest pos mean | rest pos median | ori final (deg) | rest-face acc | fell false alarms | pen mean (mm) | pen capped (mm) | diverged |
|---|---|---|---|---|---|---|---|---|---|---|---|
| reference | 0.000 | 0.000 | 0.000 (49) | 0.000 | 0.000 | 0.0 | 1.000 | - | 1.48 | 1.48 | 0 |
| hold_initial | 0.583 | 0.918 | 1.029 (49) | 1.028 | 1.037 | 35.7 | 1.000 | - | 0.00 | 0.00 | 0 |
| agatha_physics | 0.021 | 0.104 | 0.354 (49) | 0.359 | 0.352 | 29.0 | 1.000 | - | 0.44 | 0.44 | 0 |
| old_mlp | 0.124 | 0.243 | 0.301 (49) | 0.297 | 0.242 | 46.3 | 0.980 | - | 0.00 | 0.00 | 0 |
| old_mlp_projected | 0.124 | 0.243 | 0.301 (49) | 0.297 | 0.242 | 46.3 | 0.980 | - | 0.00 | 0.00 | 0 |
| new_raw | 0.023 | 0.060 | 0.292 (49) | 0.296 | 0.297 | 31.2 | 1.000 | - | 0.06 | 0.06 | 0 |
| new_projected | 0.023 | 0.060 | 0.292 (49) | 0.295 | 0.297 | 31.2 | 1.000 | - | 0.00 | 0.00 | 0 |

### real_iam_box_toss / paired episode bootstrap (B - A; negative = engine B is better)

| ground | subset | metric | A -> B | A | B | delta | 95% CI | episodes |
|---|---|---|---|---|---|---|---|---|
| slab | all_objects | rest_pos_err_m | old_mlp -> new_projected | 0.651 | 0.371 | -0.280 | [-0.388, -0.169] | 49 |
| slab | all_objects | ori_err_final_deg | old_mlp -> new_projected | 86.8 | 47.9 | -38.9 | [-51.7, -25.6] | 49 |
| slab | all_objects | rest_face_error_rate | old_mlp -> new_projected | 0.776 | 0.163 | -0.612 | [-0.755, -0.469] | 49 |
| slab | all_objects | rest_pos_err_m | old_mlp_projected -> new_projected | 0.439 | 0.371 | -0.068 | [-0.174, 0.043] | 49 |
| slab | all_objects | ori_err_final_deg | old_mlp_projected -> new_projected | 51.0 | 47.9 | -3.0 | [-16.8, 12.0] | 49 |
| slab | all_objects | rest_face_error_rate | old_mlp_projected -> new_projected | 0.204 | 0.163 | -0.041 | [-0.184, 0.102] | 49 |
| slab | all_objects | rest_pos_err_m | agatha_physics -> new_projected | 0.476 | 0.371 | -0.105 | [-0.144, -0.072] | 49 |
| slab | all_objects | ori_err_final_deg | agatha_physics -> new_projected | 63.8 | 47.9 | -15.8 | [-26.9, -5.8] | 49 |
| slab | all_objects | rest_face_error_rate | agatha_physics -> new_projected | 0.367 | 0.163 | -0.204 | [-0.306, -0.102] | 49 |
| slab | all_objects | rest_pos_err_m | agatha_physics -> new_raw | 0.476 | 0.372 | -0.104 | [-0.142, -0.069] | 49 |
| slab | all_objects | ori_err_final_deg | agatha_physics -> new_raw | 63.8 | 48.1 | -15.7 | [-26.9, -5.8] | 49 |
| slab | all_objects | rest_face_error_rate | agatha_physics -> new_raw | 0.367 | 0.163 | -0.204 | [-0.306, -0.102] | 49 |
| slab | all_objects | rest_pos_err_m | hold_initial -> new_projected | 1.028 | 0.371 | -0.658 | [-0.709, -0.596] | 49 |
| slab | all_objects | ori_err_final_deg | hold_initial -> new_projected | 35.7 | 47.9 | 12.2 | [2.1, 24.1] | 49 |
| slab | all_objects | rest_face_error_rate | hold_initial -> new_projected | 0.000 | 0.163 | 0.163 | [0.061, 0.265] | 49 |
| slab | all_objects | rest_pos_err_m | hold_initial -> agatha_physics | 1.028 | 0.476 | -0.552 | [-0.618, -0.477] | 49 |
| slab | all_objects | ori_err_final_deg | hold_initial -> agatha_physics | 35.7 | 63.8 | 28.0 | [15.0, 42.3] | 49 |
| slab | all_objects | rest_face_error_rate | hold_initial -> agatha_physics | 0.000 | 0.367 | 0.367 | [0.245, 0.510] | 49 |
| floor | all_objects | rest_pos_err_m | old_mlp -> new_projected | 0.297 | 0.295 | -0.002 | [-0.041, 0.032] | 49 |
| floor | all_objects | ori_err_final_deg | old_mlp -> new_projected | 46.3 | 31.2 | -15.1 | [-18.9, -11.3] | 49 |
| floor | all_objects | rest_face_error_rate | old_mlp -> new_projected | 0.020 | 0.000 | -0.020 | [-0.061, 0.000] | 49 |
| floor | all_objects | rest_pos_err_m | old_mlp_projected -> new_projected | 0.297 | 0.295 | -0.002 | [-0.041, 0.032] | 49 |
| floor | all_objects | ori_err_final_deg | old_mlp_projected -> new_projected | 46.3 | 31.2 | -15.1 | [-18.9, -11.3] | 49 |
| floor | all_objects | rest_face_error_rate | old_mlp_projected -> new_projected | 0.020 | 0.000 | -0.020 | [-0.061, 0.000] | 49 |
| floor | all_objects | rest_pos_err_m | agatha_physics -> new_projected | 0.359 | 0.295 | -0.064 | [-0.070, -0.058] | 49 |
| floor | all_objects | ori_err_final_deg | agatha_physics -> new_projected | 29.0 | 31.2 | 2.2 | [0.5, 3.9] | 49 |
| floor | all_objects | rest_face_error_rate | agatha_physics -> new_projected | 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 49 |
| floor | all_objects | rest_pos_err_m | agatha_physics -> new_raw | 0.359 | 0.296 | -0.063 | [-0.070, -0.057] | 49 |
| floor | all_objects | ori_err_final_deg | agatha_physics -> new_raw | 29.0 | 31.2 | 2.2 | [0.5, 3.9] | 49 |
| floor | all_objects | rest_face_error_rate | agatha_physics -> new_raw | 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 49 |
| floor | all_objects | rest_pos_err_m | hold_initial -> new_projected | 1.028 | 0.295 | -0.733 | [-0.763, -0.701] | 49 |
| floor | all_objects | ori_err_final_deg | hold_initial -> new_projected | 35.7 | 31.2 | -4.5 | [-7.6, -1.4] | 49 |
| floor | all_objects | rest_face_error_rate | hold_initial -> new_projected | 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 49 |
| floor | all_objects | rest_pos_err_m | hold_initial -> agatha_physics | 1.028 | 0.359 | -0.669 | [-0.697, -0.637] | 49 |
| floor | all_objects | ori_err_final_deg | hold_initial -> agatha_physics | 35.7 | 29.0 | -6.7 | [-9.4, -4.1] | 49 |
| floor | all_objects | rest_face_error_rate | hold_initial -> agatha_physics | 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 49 |

### real_iam_box_toss / PyBullet friction x restitution grid search (simulator vs reality, no learned model)

> APPROXIMATE. PyBullet (Agatha box engine: uniform box, 240 Hz, damping 0.04) on an 8 m slab (no reachable edge) from the measured initial state; effective values are pair products (object = support = sqrt(value)); selected on the even episodes, reported on the odd ones.

Fit on 25 trajectories, reported on the other 24. Objective: mean position error over the rollout, fit episodes.

| parameters | mu_eff | e_eff | fit rollout pos (m) | holdout rollout pos (m) | holdout rest pos mean (m) | holdout rest pos median (m) | holdout ori final (deg) | holdout rest-face acc |
|---|---|---|---|---|---|---|---|---|
| grid best (rollout objective) | 0.350 | 0.100 | 0.041 | 0.039 | 0.056 | 0.047 | 23.0 | 1.000 |
| grid best by rest position | 0.350 | 0.100 | 0.041 | 0.039 | 0.056 | 0.047 | 23.0 | 1.000 |
| default_mu0.25_e0 | 0.250 | 0.000 | 0.205 | 0.190 | 0.346 | 0.342 | 23.4 | 1.000 |
| sliding_deceleration_friction_e0 | 0.344 | 0.000 | 0.047 | 0.042 | 0.062 | 0.058 | 23.5 | 1.000 |

Grid best minus default on the holdout trajectories (paired bootstrap): rest position -0.290 m [-0.309, -0.265], rollout position -0.151 m [-0.161, -0.138].

Holdout rest-position error (m) over the grid (rows mu_eff, columns e_eff):

| mu_eff \ e_eff | 0 | 0.05 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 |
|---|---|---|---|---|---|---|---|
| 0.05 | 1.651 | 1.651 | 1.651 | 1.652 | 1.652 | 1.652 | 1.652 |
| 0.1 | 1.300 | 1.300 | 1.300 | 1.300 | 1.301 | 1.301 | 1.301 |
| 0.15 | 0.946 | 0.947 | 0.947 | 0.947 | 0.948 | 0.948 | 0.949 |
| 0.2 | 0.610 | 0.611 | 0.611 | 0.611 | 0.612 | 0.613 | 0.615 |
| 0.25 | 0.346 | 0.346 | 0.346 | 0.347 | 0.348 | 0.349 | 0.351 |
| 0.3 | 0.165 | 0.165 | 0.166 | 0.167 | 0.167 | 0.169 | 0.171 |
| 0.35 | 0.055 | 0.055 | 0.056 | 0.057 | 0.057 | 0.059 | 0.061 |
| 0.4 | 0.081 | 0.082 | 0.081 | 0.082 | 0.081 | 0.081 | 0.081 |
| 0.5 | 0.212 | 0.212 | 0.212 | 0.212 | 0.210 | 0.209 | 0.207 |
| 0.6 | 0.303 | 0.304 | 0.304 | 0.303 | 0.301 | 0.298 | 0.293 |
| 0.8 | 0.414 | 0.413 | 0.412 | 0.412 | 0.410 | 0.422 | 0.410 |
| 1 | 0.483 | 0.482 | 0.481 | 0.475 | 0.481 | 0.485 | 0.491 |

## real_iam_box_toss_gridfit (REAL WORLD, approximate)

> REAL-WORLD pose tracks (motion capture / AprilTags) of a single tossed box; friction and restitution are assumed (see `materials`), velocities are finite differences, 20 Hz resampling. APPROXIMATE.

Evaluated 2026-11-15 07:37:53 with contact_gnn checkpoint sha256 e4b5e4a3a905 (epoch 246, file written 2026-11-15 06:35:20), contact_gnn.py sha256 7f4ce7fcf5f5.

49 trajectories, 1294 transitions at 20 Hz, horizon 1.15-1.85 s; OptiTrack motion capture, 360 Hz; object: Box009 (foam-filled carton), 0.1644 x 0.1202 x 0.1253 m (Agatha x, y, z), 0.285 kg; surface: idle rubber-belt conveyor. Assumed materials (PyBullet grid-search optimum (chosen on the even trajectories)): object friction 0.592, support friction 0.592 (effective 0.350), restitution 0.316 (effective 0.100). Ground truth rests on the same face that was down at release in 1.000 of the trajectories (= the rest-face accuracy of `hold_initial`). Tables larger than the training range: 48.

Columns: position error (m) at 0.25 / 0.5 / 1 s (n = trajectories that last that long) and at the last frame (rest); orientation error at the last frame; rest-face accuracy (same body face down as in the measurement); `fell` false alarms (the real box never leaves the surface); penetration (mm).

### real_iam_box_toss_gridfit / ground = slab (49 trajectories)

| engine | pos 0.25s | pos 0.5s | pos 1s (n) | rest pos mean | rest pos median | ori final (deg) | rest-face acc | fell false alarms | pen mean (mm) | pen capped (mm) | diverged |
|---|---|---|---|---|---|---|---|---|---|---|---|
| reference | 0.000 | 0.000 | 0.000 (49) | 0.000 | 0.000 | 0.0 | 1.000 | 0 | 1.48 | 1.48 | 0 |
| hold_initial | 0.583 | 0.918 | 1.029 (49) | 1.028 | 1.037 | 35.7 | 1.000 | 0 | 0.00 | 0.00 | 0 |
| agatha_physics | 0.015 | 0.032 | 0.057 (49) | 0.057 | 0.045 | 28.3 | 1.000 | 0 | 0.31 | 0.31 | 0 |
| old_mlp | 0.065 | 0.192 | 0.456 (49) | 0.473 | 0.317 | 70.7 | 0.469 | 2 | 35.47 | 29.92 | 0 |
| old_mlp_projected | 0.065 | 0.139 | 0.220 (49) | 0.237 | 0.173 | 40.6 | 0.959 | 2 | 0.01 | 0.01 | 0 |
| new_raw | 0.019 | 0.031 | 0.059 (49) | 0.059 | 0.050 | 30.8 | 1.000 | 0 | 0.07 | 0.07 | 0 |
| new_projected | 0.019 | 0.031 | 0.058 (49) | 0.058 | 0.050 | 30.8 | 1.000 | 0 | 0.00 | 0.00 | 0 |

### real_iam_box_toss_gridfit / ground = floor (49 trajectories)

| engine | pos 0.25s | pos 0.5s | pos 1s (n) | rest pos mean | rest pos median | ori final (deg) | rest-face acc | fell false alarms | pen mean (mm) | pen capped (mm) | diverged |
|---|---|---|---|---|---|---|---|---|---|---|---|
| reference | 0.000 | 0.000 | 0.000 (49) | 0.000 | 0.000 | 0.0 | 1.000 | - | 1.48 | 1.48 | 0 |
| hold_initial | 0.583 | 0.918 | 1.029 (49) | 1.028 | 1.037 | 35.7 | 1.000 | - | 0.00 | 0.00 | 0 |
| agatha_physics | 0.015 | 0.032 | 0.057 (49) | 0.057 | 0.045 | 29.0 | 1.000 | - | 0.37 | 0.37 | 0 |
| old_mlp | 0.151 | 0.354 | 0.447 (49) | 0.444 | 0.424 | 43.4 | 1.000 | - | 0.02 | 0.02 | 0 |
| old_mlp_projected | 0.151 | 0.354 | 0.447 (49) | 0.444 | 0.424 | 43.4 | 1.000 | - | 0.00 | 0.00 | 0 |
| new_raw | 0.045 | 0.069 | 0.088 (49) | 0.088 | 0.084 | 31.5 | 1.000 | - | 0.08 | 0.08 | 0 |
| new_projected | 0.045 | 0.069 | 0.089 (49) | 0.088 | 0.085 | 31.5 | 1.000 | - | 0.00 | 0.00 | 0 |

### real_iam_box_toss_gridfit / paired episode bootstrap (B - A; negative = engine B is better)

| ground | subset | metric | A -> B | A | B | delta | 95% CI | episodes |
|---|---|---|---|---|---|---|---|---|
| slab | all_objects | rest_pos_err_m | old_mlp -> new_projected | 0.473 | 0.058 | -0.415 | [-0.493, -0.332] | 49 |
| slab | all_objects | ori_err_final_deg | old_mlp -> new_projected | 70.7 | 30.8 | -39.9 | [-47.2, -32.7] | 49 |
| slab | all_objects | rest_face_error_rate | old_mlp -> new_projected | 0.531 | 0.000 | -0.531 | [-0.653, -0.388] | 49 |
| slab | all_objects | rest_pos_err_m | old_mlp_projected -> new_projected | 0.237 | 0.058 | -0.179 | [-0.233, -0.130] | 49 |
| slab | all_objects | ori_err_final_deg | old_mlp_projected -> new_projected | 40.6 | 30.8 | -9.8 | [-15.0, -5.5] | 49 |
| slab | all_objects | rest_face_error_rate | old_mlp_projected -> new_projected | 0.041 | 0.000 | -0.041 | [-0.102, 0.000] | 49 |
| slab | all_objects | rest_pos_err_m | agatha_physics -> new_projected | 0.057 | 0.058 | 0.001 | [-0.005, 0.008] | 49 |
| slab | all_objects | ori_err_final_deg | agatha_physics -> new_projected | 28.3 | 30.8 | 2.5 | [0.6, 4.7] | 49 |
| slab | all_objects | rest_face_error_rate | agatha_physics -> new_projected | 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 49 |
| slab | all_objects | rest_pos_err_m | agatha_physics -> new_raw | 0.057 | 0.059 | 0.001 | [-0.005, 0.008] | 49 |
| slab | all_objects | ori_err_final_deg | agatha_physics -> new_raw | 28.3 | 30.8 | 2.5 | [0.7, 4.7] | 49 |
| slab | all_objects | rest_face_error_rate | agatha_physics -> new_raw | 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 49 |
| slab | all_objects | rest_pos_err_m | hold_initial -> new_projected | 1.028 | 0.058 | -0.970 | [-0.998, -0.937] | 49 |
| slab | all_objects | ori_err_final_deg | hold_initial -> new_projected | 35.7 | 30.8 | -4.9 | [-7.1, -2.8] | 49 |
| slab | all_objects | rest_face_error_rate | hold_initial -> new_projected | 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 49 |
| slab | all_objects | rest_pos_err_m | hold_initial -> agatha_physics | 1.028 | 0.057 | -0.971 | [-1.000, -0.937] | 49 |
| slab | all_objects | ori_err_final_deg | hold_initial -> agatha_physics | 35.7 | 28.3 | -7.4 | [-10.0, -5.0] | 49 |
| slab | all_objects | rest_face_error_rate | hold_initial -> agatha_physics | 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 49 |
| floor | all_objects | rest_pos_err_m | old_mlp -> new_projected | 0.444 | 0.088 | -0.356 | [-0.375, -0.336] | 49 |
| floor | all_objects | ori_err_final_deg | old_mlp -> new_projected | 43.4 | 31.5 | -12.0 | [-15.4, -8.5] | 49 |
| floor | all_objects | rest_face_error_rate | old_mlp -> new_projected | 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 49 |
| floor | all_objects | rest_pos_err_m | old_mlp_projected -> new_projected | 0.444 | 0.088 | -0.356 | [-0.375, -0.336] | 49 |
| floor | all_objects | ori_err_final_deg | old_mlp_projected -> new_projected | 43.4 | 31.5 | -12.0 | [-15.4, -8.5] | 49 |
| floor | all_objects | rest_face_error_rate | old_mlp_projected -> new_projected | 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 49 |
| floor | all_objects | rest_pos_err_m | agatha_physics -> new_projected | 0.057 | 0.088 | 0.032 | [0.014, 0.049] | 49 |
| floor | all_objects | ori_err_final_deg | agatha_physics -> new_projected | 29.0 | 31.5 | 2.4 | [0.5, 4.6] | 49 |
| floor | all_objects | rest_face_error_rate | agatha_physics -> new_projected | 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 49 |
| floor | all_objects | rest_pos_err_m | agatha_physics -> new_raw | 0.057 | 0.088 | 0.031 | [0.014, 0.048] | 49 |
| floor | all_objects | ori_err_final_deg | agatha_physics -> new_raw | 29.0 | 31.5 | 2.4 | [0.5, 4.6] | 49 |
| floor | all_objects | rest_face_error_rate | agatha_physics -> new_raw | 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 49 |
| floor | all_objects | rest_pos_err_m | hold_initial -> new_projected | 1.028 | 0.088 | -0.940 | [-0.967, -0.910] | 49 |
| floor | all_objects | ori_err_final_deg | hold_initial -> new_projected | 35.7 | 31.5 | -4.3 | [-6.8, -1.5] | 49 |
| floor | all_objects | rest_face_error_rate | hold_initial -> new_projected | 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 49 |
| floor | all_objects | rest_pos_err_m | hold_initial -> agatha_physics | 1.028 | 0.057 | -0.972 | [-1.000, -0.938] | 49 |
| floor | all_objects | ori_err_final_deg | hold_initial -> agatha_physics | 35.7 | 29.0 | -6.7 | [-9.0, -4.5] | 49 |
| floor | all_objects | rest_face_error_rate | hold_initial -> agatha_physics | 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 49 |

## real_iam_box_toss_calib (REAL WORLD, approximate)

> REAL-WORLD pose tracks (motion capture / AprilTags) of a single tossed box; friction and restitution are assumed (see `materials`), velocities are finite differences, 20 Hz resampling. APPROXIMATE.

Evaluated 2026-11-15 07:37:53 with contact_gnn checkpoint sha256 e4b5e4a3a905 (epoch 246, file written 2026-11-15 06:35:20), contact_gnn.py sha256 7f4ce7fcf5f5.

49 trajectories, 1294 transitions at 20 Hz, horizon 1.15-1.85 s; OptiTrack motion capture, 360 Hz; object: Box009 (foam-filled carton), 0.1644 x 0.1202 x 0.1253 m (Agatha x, y, z), 0.285 kg; surface: idle rubber-belt conveyor. Assumed materials (calibrated from sliding deceleration): object friction 0.587, support friction 0.587 (effective 0.344), restitution 0.000 (effective 0.000). Ground truth rests on the same face that was down at release in 1.000 of the trajectories (= the rest-face accuracy of `hold_initial`). Tables larger than the training range: 48.

Columns: position error (m) at 0.25 / 0.5 / 1 s (n = trajectories that last that long) and at the last frame (rest); orientation error at the last frame; rest-face accuracy (same body face down as in the measurement); `fell` false alarms (the real box never leaves the surface); penetration (mm).

### real_iam_box_toss_calib / ground = slab (49 trajectories)

| engine | pos 0.25s | pos 0.5s | pos 1s (n) | rest pos mean | rest pos median | ori final (deg) | rest-face acc | fell false alarms | pen mean (mm) | pen capped (mm) | diverged |
|---|---|---|---|---|---|---|---|---|---|---|---|
| reference | 0.000 | 0.000 | 0.000 (49) | 0.000 | 0.000 | 0.0 | 1.000 | 0 | 1.48 | 1.48 | 0 |
| hold_initial | 0.583 | 0.918 | 1.029 (49) | 1.028 | 1.037 | 35.7 | 1.000 | 0 | 0.00 | 0.00 | 0 |
| agatha_physics | 0.016 | 0.034 | 0.066 (49) | 0.066 | 0.055 | 29.0 | 1.000 | 0 | 0.39 | 0.39 | 0 |
| old_mlp | 0.065 | 0.192 | 0.475 (49) | 0.490 | 0.317 | 71.9 | 0.449 | 4 | 35.44 | 30.03 | 0 |
| old_mlp_projected | 0.064 | 0.137 | 0.220 (49) | 0.239 | 0.185 | 40.6 | 0.959 | 2 | 0.00 | 0.00 | 0 |
| new_raw | 0.019 | 0.031 | 0.062 (49) | 0.062 | 0.054 | 30.8 | 1.000 | 0 | 0.07 | 0.07 | 0 |
| new_projected | 0.019 | 0.031 | 0.062 (49) | 0.062 | 0.054 | 30.7 | 1.000 | 0 | 0.00 | 0.00 | 0 |

### real_iam_box_toss_calib / ground = floor (49 trajectories)

| engine | pos 0.25s | pos 0.5s | pos 1s (n) | rest pos mean | rest pos median | ori final (deg) | rest-face acc | fell false alarms | pen mean (mm) | pen capped (mm) | diverged |
|---|---|---|---|---|---|---|---|---|---|---|---|
| reference | 0.000 | 0.000 | 0.000 (49) | 0.000 | 0.000 | 0.0 | 1.000 | - | 1.48 | 1.48 | 0 |
| hold_initial | 0.583 | 0.918 | 1.029 (49) | 1.028 | 1.037 | 35.7 | 1.000 | - | 0.00 | 0.00 | 0 |
| agatha_physics | 0.016 | 0.034 | 0.066 (49) | 0.066 | 0.055 | 29.0 | 1.000 | - | 0.39 | 0.39 | 0 |
| old_mlp | 0.150 | 0.348 | 0.440 (49) | 0.437 | 0.415 | 43.6 | 1.000 | - | 0.01 | 0.01 | 0 |
| old_mlp_projected | 0.150 | 0.348 | 0.440 (49) | 0.437 | 0.415 | 43.6 | 1.000 | - | 0.00 | 0.00 | 0 |
| new_raw | 0.044 | 0.064 | 0.077 (49) | 0.077 | 0.073 | 31.4 | 1.000 | - | 0.07 | 0.07 | 0 |
| new_projected | 0.044 | 0.064 | 0.078 (49) | 0.078 | 0.074 | 31.4 | 1.000 | - | 0.00 | 0.00 | 0 |

### real_iam_box_toss_calib / paired episode bootstrap (B - A; negative = engine B is better)

| ground | subset | metric | A -> B | A | B | delta | 95% CI | episodes |
|---|---|---|---|---|---|---|---|---|
| slab | all_objects | rest_pos_err_m | old_mlp -> new_projected | 0.490 | 0.062 | -0.428 | [-0.509, -0.342] | 49 |
| slab | all_objects | ori_err_final_deg | old_mlp -> new_projected | 71.9 | 30.7 | -41.2 | [-48.5, -33.8] | 49 |
| slab | all_objects | rest_face_error_rate | old_mlp -> new_projected | 0.551 | 0.000 | -0.551 | [-0.673, -0.408] | 49 |
| slab | all_objects | rest_pos_err_m | old_mlp_projected -> new_projected | 0.239 | 0.062 | -0.177 | [-0.232, -0.127] | 49 |
| slab | all_objects | ori_err_final_deg | old_mlp_projected -> new_projected | 40.6 | 30.7 | -9.8 | [-15.0, -5.5] | 49 |
| slab | all_objects | rest_face_error_rate | old_mlp_projected -> new_projected | 0.041 | 0.000 | -0.041 | [-0.102, 0.000] | 49 |
| slab | all_objects | rest_pos_err_m | agatha_physics -> new_projected | 0.066 | 0.062 | -0.004 | [-0.011, 0.004] | 49 |
| slab | all_objects | ori_err_final_deg | agatha_physics -> new_projected | 29.0 | 30.7 | 1.7 | [0.3, 3.3] | 49 |
| slab | all_objects | rest_face_error_rate | agatha_physics -> new_projected | 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 49 |
| slab | all_objects | rest_pos_err_m | agatha_physics -> new_raw | 0.066 | 0.062 | -0.004 | [-0.011, 0.004] | 49 |
| slab | all_objects | ori_err_final_deg | agatha_physics -> new_raw | 29.0 | 30.8 | 1.8 | [0.3, 3.3] | 49 |
| slab | all_objects | rest_face_error_rate | agatha_physics -> new_raw | 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 49 |
| slab | all_objects | rest_pos_err_m | hold_initial -> new_projected | 1.028 | 0.062 | -0.967 | [-0.995, -0.933] | 49 |
| slab | all_objects | ori_err_final_deg | hold_initial -> new_projected | 35.7 | 30.7 | -5.0 | [-7.2, -2.9] | 49 |
| slab | all_objects | rest_face_error_rate | hold_initial -> new_projected | 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 49 |
| slab | all_objects | rest_pos_err_m | hold_initial -> agatha_physics | 1.028 | 0.066 | -0.963 | [-0.992, -0.928] | 49 |
| slab | all_objects | ori_err_final_deg | hold_initial -> agatha_physics | 35.7 | 29.0 | -6.7 | [-9.0, -4.6] | 49 |
| slab | all_objects | rest_face_error_rate | hold_initial -> agatha_physics | 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 49 |
| floor | all_objects | rest_pos_err_m | old_mlp -> new_projected | 0.437 | 0.078 | -0.360 | [-0.380, -0.340] | 49 |
| floor | all_objects | ori_err_final_deg | old_mlp -> new_projected | 43.6 | 31.4 | -12.2 | [-15.6, -8.7] | 49 |
| floor | all_objects | rest_face_error_rate | old_mlp -> new_projected | 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 49 |
| floor | all_objects | rest_pos_err_m | old_mlp_projected -> new_projected | 0.437 | 0.078 | -0.360 | [-0.380, -0.340] | 49 |
| floor | all_objects | ori_err_final_deg | old_mlp_projected -> new_projected | 43.6 | 31.4 | -12.2 | [-15.6, -8.7] | 49 |
| floor | all_objects | rest_face_error_rate | old_mlp_projected -> new_projected | 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 49 |
| floor | all_objects | rest_pos_err_m | agatha_physics -> new_projected | 0.066 | 0.078 | 0.012 | [-0.006, 0.030] | 49 |
| floor | all_objects | ori_err_final_deg | agatha_physics -> new_projected | 29.0 | 31.4 | 2.4 | [0.5, 4.6] | 49 |
| floor | all_objects | rest_face_error_rate | agatha_physics -> new_projected | 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 49 |
| floor | all_objects | rest_pos_err_m | agatha_physics -> new_raw | 0.066 | 0.077 | 0.012 | [-0.007, 0.030] | 49 |
| floor | all_objects | ori_err_final_deg | agatha_physics -> new_raw | 29.0 | 31.4 | 2.4 | [0.5, 4.6] | 49 |
| floor | all_objects | rest_face_error_rate | agatha_physics -> new_raw | 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 49 |
| floor | all_objects | rest_pos_err_m | hold_initial -> new_projected | 1.028 | 0.078 | -0.951 | [-0.978, -0.920] | 49 |
| floor | all_objects | ori_err_final_deg | hold_initial -> new_projected | 35.7 | 31.4 | -4.3 | [-6.9, -1.6] | 49 |
| floor | all_objects | rest_face_error_rate | hold_initial -> new_projected | 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 49 |
| floor | all_objects | rest_pos_err_m | hold_initial -> agatha_physics | 1.028 | 0.066 | -0.963 | [-0.992, -0.928] | 49 |
| floor | all_objects | ori_err_final_deg | hold_initial -> agatha_physics | 35.7 | 29.0 | -6.7 | [-9.0, -4.6] | 49 |
| floor | all_objects | rest_face_error_rate | hold_initial -> agatha_physics | 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 49 |

## real_contactnets_cube (REAL WORLD, approximate)

> REAL-WORLD pose tracks (motion capture / AprilTags) of a single tossed box; friction and restitution are assumed (see `materials`), velocities are finite differences, 20 Hz resampling. APPROXIMATE.

Evaluated 2026-11-15 07:37:53 with contact_gnn checkpoint sha256 e4b5e4a3a905 (epoch 246, file written 2026-11-15 06:35:20), contact_gnn.py sha256 7f4ce7fcf5f5.

570 trajectories, 7434 transitions at 20 Hz, horizon 0.5-2.95 s; AprilTag pose tracking, 148 Hz; object: acrylic cube, 0.1048 x 0.1048 x 0.1048 m (Agatha x, y, z), 0.370 kg; surface: flat wooden surface. Assumed materials (default): object friction 0.500, support friction 0.500 (effective 0.250), restitution 0.000 (effective 0.000). Ground truth rests on the same face that was down at release in 0.433 of the trajectories (= the rest-face accuracy of `hold_initial`). Tables larger than the training range: 0.

Columns: position error (m) at 0.25 / 0.5 / 1 s (n = trajectories that last that long) and at the last frame (rest); orientation error at the last frame; rest-face accuracy (same body face down as in the measurement); `fell` false alarms (the real box never leaves the surface); penetration (mm).

### real_contactnets_cube / ground = slab (570 trajectories)

| engine | pos 0.25s | pos 0.5s | pos 1s (n) | rest pos mean | rest pos median | ori final (deg) | rest-face acc | fell false alarms | pen mean (mm) | pen capped (mm) | diverged |
|---|---|---|---|---|---|---|---|---|---|---|---|
| reference | 0.000 | 0.000 | 0.000 (10) | 0.000 | 0.000 | 0.0 | 1.000 | 0 | 0.25 | 0.25 | 0 |
| hold_initial | 0.256 | 0.382 | 0.499 (10) | 0.400 | 0.396 | 61.0 | 0.433 | 0 | 0.00 | 0.00 | 0 |
| agatha_physics | 0.025 | 0.085 | 0.114 (10) | 0.103 | 0.100 | 24.6 | 0.853 | 0 | 0.44 | 0.44 | 0 |
| old_mlp | 0.057 | 0.138 | 0.201 (10) | 0.153 | 0.147 | 53.1 | 0.640 | 0 | 23.85 | 20.42 | 0 |
| old_mlp_projected | 0.051 | 0.133 | 0.193 (10) | 0.149 | 0.145 | 51.1 | 0.607 | 0 | 0.00 | 0.00 | 0 |
| new_raw | 0.025 | 0.085 | 0.115 (10) | 0.105 | 0.102 | 30.6 | 0.804 | 0 | 0.32 | 0.32 | 0 |
| new_projected | 0.025 | 0.083 | 0.112 (10) | 0.101 | 0.099 | 31.4 | 0.800 | 0 | 0.00 | 0.00 | 0 |

### real_contactnets_cube / ground = floor (570 trajectories)

| engine | pos 0.25s | pos 0.5s | pos 1s (n) | rest pos mean | rest pos median | ori final (deg) | rest-face acc | fell false alarms | pen mean (mm) | pen capped (mm) | diverged |
|---|---|---|---|---|---|---|---|---|---|---|---|
| reference | 0.000 | 0.000 | 0.000 (10) | 0.000 | 0.000 | 0.0 | 1.000 | - | 0.25 | 0.25 | 0 |
| hold_initial | 0.256 | 0.382 | 0.499 (10) | 0.400 | 0.396 | 61.0 | 0.433 | - | 0.00 | 0.00 | 0 |
| agatha_physics | 0.025 | 0.085 | 0.114 (10) | 0.103 | 0.100 | 24.6 | 0.853 | - | 0.44 | 0.44 | 0 |
| old_mlp | 0.105 | 0.200 | 0.287 (10) | 0.215 | 0.211 | 53.3 | 0.574 | - | 0.09 | 0.09 | 0 |
| old_mlp_projected | 0.105 | 0.200 | 0.287 (10) | 0.215 | 0.211 | 53.3 | 0.574 | - | 0.00 | 0.00 | 0 |
| new_raw | 0.044 | 0.131 | 0.182 (10) | 0.150 | 0.145 | 46.1 | 0.611 | - | 0.26 | 0.26 | 0 |
| new_projected | 0.044 | 0.130 | 0.177 (10) | 0.149 | 0.144 | 46.3 | 0.612 | - | 0.00 | 0.00 | 0 |

### real_contactnets_cube / paired episode bootstrap (B - A; negative = engine B is better)

| ground | subset | metric | A -> B | A | B | delta | 95% CI | episodes |
|---|---|---|---|---|---|---|---|---|
| slab | all_objects | rest_pos_err_m | old_mlp -> new_projected | 0.153 | 0.101 | -0.051 | [-0.056, -0.047] | 570 |
| slab | all_objects | ori_err_final_deg | old_mlp -> new_projected | 53.1 | 31.4 | -21.7 | [-24.0, -19.4] | 570 |
| slab | all_objects | rest_face_error_rate | old_mlp -> new_projected | 0.360 | 0.200 | -0.160 | [-0.200, -0.116] | 570 |
| slab | all_objects | rest_pos_err_m | old_mlp_projected -> new_projected | 0.149 | 0.101 | -0.047 | [-0.051, -0.043] | 570 |
| slab | all_objects | ori_err_final_deg | old_mlp_projected -> new_projected | 51.1 | 31.4 | -19.7 | [-21.9, -17.4] | 570 |
| slab | all_objects | rest_face_error_rate | old_mlp_projected -> new_projected | 0.393 | 0.200 | -0.193 | [-0.235, -0.154] | 570 |
| slab | all_objects | rest_pos_err_m | agatha_physics -> new_projected | 0.103 | 0.101 | -0.002 | [-0.004, 0.001] | 570 |
| slab | all_objects | ori_err_final_deg | agatha_physics -> new_projected | 24.6 | 31.4 | 6.7 | [4.6, 8.9] | 570 |
| slab | all_objects | rest_face_error_rate | agatha_physics -> new_projected | 0.147 | 0.200 | 0.053 | [0.025, 0.082] | 570 |
| slab | all_objects | rest_pos_err_m | agatha_physics -> new_raw | 0.103 | 0.105 | 0.001 | [-0.001, 0.004] | 570 |
| slab | all_objects | ori_err_final_deg | agatha_physics -> new_raw | 24.6 | 30.6 | 6.0 | [3.8, 8.2] | 570 |
| slab | all_objects | rest_face_error_rate | agatha_physics -> new_raw | 0.147 | 0.196 | 0.049 | [0.021, 0.077] | 570 |
| slab | all_objects | rest_pos_err_m | hold_initial -> new_projected | 0.400 | 0.101 | -0.299 | [-0.303, -0.294] | 570 |
| slab | all_objects | ori_err_final_deg | hold_initial -> new_projected | 61.0 | 31.4 | -29.7 | [-32.2, -26.9] | 570 |
| slab | all_objects | rest_face_error_rate | hold_initial -> new_projected | 0.567 | 0.200 | -0.367 | [-0.414, -0.319] | 570 |
| slab | all_objects | rest_pos_err_m | hold_initial -> agatha_physics | 0.400 | 0.103 | -0.297 | [-0.301, -0.293] | 570 |
| slab | all_objects | ori_err_final_deg | hold_initial -> agatha_physics | 61.0 | 24.6 | -36.4 | [-39.0, -33.8] | 570 |
| slab | all_objects | rest_face_error_rate | hold_initial -> agatha_physics | 0.567 | 0.147 | -0.419 | [-0.465, -0.372] | 570 |
| floor | all_objects | rest_pos_err_m | old_mlp -> new_projected | 0.215 | 0.149 | -0.067 | [-0.070, -0.063] | 570 |
| floor | all_objects | ori_err_final_deg | old_mlp -> new_projected | 53.3 | 46.3 | -7.1 | [-9.5, -4.8] | 570 |
| floor | all_objects | rest_face_error_rate | old_mlp -> new_projected | 0.426 | 0.388 | -0.039 | [-0.065, -0.011] | 570 |
| floor | all_objects | rest_pos_err_m | old_mlp_projected -> new_projected | 0.215 | 0.149 | -0.067 | [-0.070, -0.063] | 570 |
| floor | all_objects | ori_err_final_deg | old_mlp_projected -> new_projected | 53.3 | 46.3 | -7.1 | [-9.5, -4.8] | 570 |
| floor | all_objects | rest_face_error_rate | old_mlp_projected -> new_projected | 0.426 | 0.388 | -0.039 | [-0.065, -0.011] | 570 |
| floor | all_objects | rest_pos_err_m | agatha_physics -> new_projected | 0.103 | 0.149 | 0.045 | [0.043, 0.048] | 570 |
| floor | all_objects | ori_err_final_deg | agatha_physics -> new_projected | 24.6 | 46.3 | 21.6 | [18.2, 25.1] | 570 |
| floor | all_objects | rest_face_error_rate | agatha_physics -> new_projected | 0.147 | 0.388 | 0.240 | [0.195, 0.288] | 570 |
| floor | all_objects | rest_pos_err_m | agatha_physics -> new_raw | 0.103 | 0.150 | 0.046 | [0.044, 0.049] | 570 |
| floor | all_objects | ori_err_final_deg | agatha_physics -> new_raw | 24.6 | 46.1 | 21.4 | [18.0, 24.8] | 570 |
| floor | all_objects | rest_face_error_rate | agatha_physics -> new_raw | 0.147 | 0.389 | 0.242 | [0.198, 0.286] | 570 |
| floor | all_objects | rest_pos_err_m | hold_initial -> new_projected | 0.400 | 0.149 | -0.252 | [-0.255, -0.248] | 570 |
| floor | all_objects | ori_err_final_deg | hold_initial -> new_projected | 61.0 | 46.3 | -14.7 | [-16.8, -12.7] | 570 |
| floor | all_objects | rest_face_error_rate | hold_initial -> new_projected | 0.567 | 0.388 | -0.179 | [-0.211, -0.146] | 570 |
| floor | all_objects | rest_pos_err_m | hold_initial -> agatha_physics | 0.400 | 0.103 | -0.297 | [-0.301, -0.293] | 570 |
| floor | all_objects | ori_err_final_deg | hold_initial -> agatha_physics | 61.0 | 24.6 | -36.4 | [-39.0, -33.8] | 570 |
| floor | all_objects | rest_face_error_rate | hold_initial -> agatha_physics | 0.567 | 0.147 | -0.419 | [-0.465, -0.372] | 570 |

### real_contactnets_cube / PyBullet friction x restitution grid search (simulator vs reality, no learned model)

> APPROXIMATE. PyBullet (Agatha box engine: uniform box, 240 Hz, damping 0.04) on an 8 m slab (no reachable edge) from the measured initial state; effective values are pair products (object = support = sqrt(value)); selected on the even episodes, reported on the odd ones.

Fit on 285 trajectories, reported on the other 285. Objective: mean position error over the rollout, fit episodes.

| parameters | mu_eff | e_eff | fit rollout pos (m) | holdout rollout pos (m) | holdout rest pos mean (m) | holdout rest pos median (m) | holdout ori final (deg) | holdout rest-face acc |
|---|---|---|---|---|---|---|---|---|
| grid best (rollout objective) | 0.150 | 0.000 | 0.019 | 0.018 | 0.043 | 0.040 | 30.9 | 0.782 |
| grid best by rest position | 0.200 | 0.300 | 0.022 | 0.022 | 0.042 | 0.038 | 30.0 | 0.842 |
| default_mu0.25_e0 | 0.250 | 0.000 | 0.049 | 0.051 | 0.105 | 0.103 | 24.5 | 0.856 |
| sliding_deceleration_friction_e0 | 0.218 | 0.000 | 0.034 | 0.036 | 0.075 | 0.071 | 23.2 | 0.874 |
| dataset_authors_simulation_values | 0.180 | 0.125 | 0.017 | 0.017 | 0.029 | 0.023 | 27.7 | 0.828 |

Grid best minus default on the holdout trajectories (paired bootstrap): rest position -0.062 m [-0.066, -0.057], rollout position -0.032 m [-0.034, -0.031].

Holdout rest-position error (m) over the grid (rows mu_eff, columns e_eff):

| mu_eff \ e_eff | 0 | 0.05 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 |
|---|---|---|---|---|---|---|---|
| 0.05 | 0.230 | 0.231 | 0.231 | 0.233 | 0.235 | 0.237 | 0.241 |
| 0.1 | 0.130 | 0.132 | 0.133 | 0.136 | 0.140 | 0.146 | 0.154 |
| 0.15 | 0.043 | 0.045 | 0.047 | 0.051 | 0.057 | 0.066 | 0.081 |
| 0.2 | 0.053 | 0.052 | 0.049 | 0.046 | 0.042 | 0.043 | 0.052 |
| 0.25 | 0.105 | 0.103 | 0.100 | 0.092 | 0.084 | 0.077 | 0.073 |
| 0.3 | 0.137 | 0.135 | 0.132 | 0.123 | 0.114 | 0.102 | 0.093 |
| 0.35 | 0.159 | 0.156 | 0.152 | 0.142 | 0.131 | 0.119 | 0.111 |
| 0.4 | 0.173 | 0.169 | 0.166 | 0.156 | 0.144 | 0.135 | 0.130 |
| 0.5 | 0.188 | 0.184 | 0.179 | 0.170 | 0.160 | 0.156 | 0.158 |
| 0.6 | 0.198 | 0.193 | 0.190 | 0.183 | 0.174 | 0.172 | 0.178 |
| 0.8 | 0.206 | 0.202 | 0.198 | 0.191 | 0.183 | 0.180 | 0.185 |
| 1 | 0.206 | 0.203 | 0.199 | 0.192 | 0.184 | 0.180 | 0.185 |

## real_contactnets_cube_gridfit (REAL WORLD, approximate)

> REAL-WORLD pose tracks (motion capture / AprilTags) of a single tossed box; friction and restitution are assumed (see `materials`), velocities are finite differences, 20 Hz resampling. APPROXIMATE.

Evaluated 2026-11-15 07:37:53 with contact_gnn checkpoint sha256 e4b5e4a3a905 (epoch 246, file written 2026-11-15 06:35:20), contact_gnn.py sha256 7f4ce7fcf5f5.

570 trajectories, 7434 transitions at 20 Hz, horizon 0.5-2.95 s; AprilTag pose tracking, 148 Hz; object: acrylic cube, 0.1048 x 0.1048 x 0.1048 m (Agatha x, y, z), 0.370 kg; surface: flat wooden surface. Assumed materials (PyBullet grid-search optimum (chosen on the even trajectories)): object friction 0.387, support friction 0.387 (effective 0.150), restitution 0.000 (effective 0.000). Ground truth rests on the same face that was down at release in 0.433 of the trajectories (= the rest-face accuracy of `hold_initial`). Tables larger than the training range: 0.

Columns: position error (m) at 0.25 / 0.5 / 1 s (n = trajectories that last that long) and at the last frame (rest); orientation error at the last frame; rest-face accuracy (same body face down as in the measurement); `fell` false alarms (the real box never leaves the surface); penetration (mm).

### real_contactnets_cube_gridfit / ground = slab (570 trajectories)

| engine | pos 0.25s | pos 0.5s | pos 1s (n) | rest pos mean | rest pos median | ori final (deg) | rest-face acc | fell false alarms | pen mean (mm) | pen capped (mm) | diverged |
|---|---|---|---|---|---|---|---|---|---|---|---|
| reference | 0.000 | 0.000 | 0.000 (10) | 0.000 | 0.000 | 0.0 | 1.000 | 0 | 0.25 | 0.25 | 0 |
| hold_initial | 0.256 | 0.382 | 0.499 (10) | 0.400 | 0.396 | 61.0 | 0.433 | 0 | 0.00 | 0.00 | 0 |
| agatha_physics | 0.013 | 0.026 | 0.084 (10) | 0.043 | 0.041 | 30.5 | 0.779 | 0 | 0.55 | 0.55 | 0 |
| old_mlp | 0.053 | 0.129 | 0.247 (10) | 0.147 | 0.136 | 53.2 | 0.630 | 0 | 27.91 | 23.55 | 0 |
| old_mlp_projected | 0.043 | 0.112 | 0.175 (10) | 0.127 | 0.121 | 50.5 | 0.604 | 0 | 0.00 | 0.00 | 0 |
| new_raw | 0.015 | 0.033 | 0.084 (10) | 0.048 | 0.045 | 38.4 | 0.696 | 0 | 0.39 | 0.39 | 0 |
| new_projected | 0.016 | 0.034 | 0.075 (10) | 0.049 | 0.047 | 38.4 | 0.700 | 0 | 0.00 | 0.00 | 0 |

### real_contactnets_cube_gridfit / ground = floor (570 trajectories)

| engine | pos 0.25s | pos 0.5s | pos 1s (n) | rest pos mean | rest pos median | ori final (deg) | rest-face acc | fell false alarms | pen mean (mm) | pen capped (mm) | diverged |
|---|---|---|---|---|---|---|---|---|---|---|---|
| reference | 0.000 | 0.000 | 0.000 (10) | 0.000 | 0.000 | 0.0 | 1.000 | - | 0.25 | 0.25 | 0 |
| hold_initial | 0.256 | 0.382 | 0.499 (10) | 0.400 | 0.396 | 61.0 | 0.433 | - | 0.00 | 0.00 | 0 |
| agatha_physics | 0.013 | 0.026 | 0.084 (10) | 0.043 | 0.041 | 30.5 | 0.779 | - | 0.56 | 0.56 | 0 |
| old_mlp | 0.095 | 0.175 | 0.251 (10) | 0.188 | 0.187 | 53.2 | 0.582 | - | 0.05 | 0.05 | 0 |
| old_mlp_projected | 0.095 | 0.175 | 0.251 (10) | 0.188 | 0.187 | 53.2 | 0.582 | - | 0.00 | 0.00 | 0 |
| new_raw | 0.019 | 0.034 | 0.080 (10) | 0.044 | 0.040 | 53.9 | 0.500 | - | 0.26 | 0.26 | 0 |
| new_projected | 0.020 | 0.034 | 0.082 (10) | 0.044 | 0.040 | 54.0 | 0.505 | - | 0.00 | 0.00 | 0 |

### real_contactnets_cube_gridfit / paired episode bootstrap (B - A; negative = engine B is better)

| ground | subset | metric | A -> B | A | B | delta | 95% CI | episodes |
|---|---|---|---|---|---|---|---|---|
| slab | all_objects | rest_pos_err_m | old_mlp -> new_projected | 0.147 | 0.049 | -0.098 | [-0.104, -0.092] | 570 |
| slab | all_objects | ori_err_final_deg | old_mlp -> new_projected | 53.2 | 38.4 | -14.8 | [-17.4, -12.1] | 570 |
| slab | all_objects | rest_face_error_rate | old_mlp -> new_projected | 0.370 | 0.300 | -0.070 | [-0.109, -0.026] | 570 |
| slab | all_objects | rest_pos_err_m | old_mlp_projected -> new_projected | 0.127 | 0.049 | -0.078 | [-0.082, -0.073] | 570 |
| slab | all_objects | ori_err_final_deg | old_mlp_projected -> new_projected | 50.5 | 38.4 | -12.2 | [-14.3, -9.9] | 570 |
| slab | all_objects | rest_face_error_rate | old_mlp_projected -> new_projected | 0.396 | 0.300 | -0.096 | [-0.130, -0.065] | 570 |
| slab | all_objects | rest_pos_err_m | agatha_physics -> new_projected | 0.043 | 0.049 | 0.006 | [0.004, 0.008] | 570 |
| slab | all_objects | ori_err_final_deg | agatha_physics -> new_projected | 30.5 | 38.4 | 7.9 | [5.7, 10.2] | 570 |
| slab | all_objects | rest_face_error_rate | agatha_physics -> new_projected | 0.221 | 0.300 | 0.079 | [0.053, 0.109] | 570 |
| slab | all_objects | rest_pos_err_m | agatha_physics -> new_raw | 0.043 | 0.048 | 0.004 | [0.002, 0.006] | 570 |
| slab | all_objects | ori_err_final_deg | agatha_physics -> new_raw | 30.5 | 38.4 | 7.9 | [5.8, 10.1] | 570 |
| slab | all_objects | rest_face_error_rate | agatha_physics -> new_raw | 0.221 | 0.304 | 0.082 | [0.054, 0.112] | 570 |
| slab | all_objects | rest_pos_err_m | hold_initial -> new_projected | 0.400 | 0.049 | -0.351 | [-0.356, -0.345] | 570 |
| slab | all_objects | ori_err_final_deg | hold_initial -> new_projected | 61.0 | 38.4 | -22.6 | [-24.8, -20.2] | 570 |
| slab | all_objects | rest_face_error_rate | hold_initial -> new_projected | 0.567 | 0.300 | -0.267 | [-0.302, -0.226] | 570 |
| slab | all_objects | rest_pos_err_m | hold_initial -> agatha_physics | 0.400 | 0.043 | -0.357 | [-0.362, -0.352] | 570 |
| slab | all_objects | ori_err_final_deg | hold_initial -> agatha_physics | 61.0 | 30.5 | -30.5 | [-32.6, -28.1] | 570 |
| slab | all_objects | rest_face_error_rate | hold_initial -> agatha_physics | 0.567 | 0.221 | -0.346 | [-0.384, -0.302] | 570 |
| floor | all_objects | rest_pos_err_m | old_mlp -> new_projected | 0.188 | 0.044 | -0.144 | [-0.149, -0.140] | 570 |
| floor | all_objects | ori_err_final_deg | old_mlp -> new_projected | 53.2 | 54.0 | 0.8 | [-1.8, 3.4] | 570 |
| floor | all_objects | rest_face_error_rate | old_mlp -> new_projected | 0.418 | 0.495 | 0.077 | [0.049, 0.105] | 570 |
| floor | all_objects | rest_pos_err_m | old_mlp_projected -> new_projected | 0.188 | 0.044 | -0.144 | [-0.149, -0.140] | 570 |
| floor | all_objects | ori_err_final_deg | old_mlp_projected -> new_projected | 53.2 | 54.0 | 0.8 | [-1.8, 3.4] | 570 |
| floor | all_objects | rest_face_error_rate | old_mlp_projected -> new_projected | 0.418 | 0.495 | 0.077 | [0.049, 0.105] | 570 |
| floor | all_objects | rest_pos_err_m | agatha_physics -> new_projected | 0.043 | 0.044 | 0.001 | [-0.001, 0.003] | 570 |
| floor | all_objects | ori_err_final_deg | agatha_physics -> new_projected | 30.5 | 54.0 | 23.5 | [20.5, 26.3] | 570 |
| floor | all_objects | rest_face_error_rate | agatha_physics -> new_projected | 0.221 | 0.495 | 0.274 | [0.233, 0.312] | 570 |
| floor | all_objects | rest_pos_err_m | agatha_physics -> new_raw | 0.043 | 0.044 | 0.000 | [-0.002, 0.002] | 570 |
| floor | all_objects | ori_err_final_deg | agatha_physics -> new_raw | 30.5 | 53.9 | 23.4 | [20.3, 26.2] | 570 |
| floor | all_objects | rest_face_error_rate | agatha_physics -> new_raw | 0.221 | 0.500 | 0.279 | [0.239, 0.319] | 570 |
| floor | all_objects | rest_pos_err_m | hold_initial -> new_projected | 0.400 | 0.044 | -0.356 | [-0.360, -0.351] | 570 |
| floor | all_objects | ori_err_final_deg | hold_initial -> new_projected | 61.0 | 54.0 | -7.0 | [-9.1, -5.0] | 570 |
| floor | all_objects | rest_face_error_rate | hold_initial -> new_projected | 0.567 | 0.495 | -0.072 | [-0.093, -0.049] | 570 |
| floor | all_objects | rest_pos_err_m | hold_initial -> agatha_physics | 0.400 | 0.043 | -0.357 | [-0.362, -0.352] | 570 |
| floor | all_objects | ori_err_final_deg | hold_initial -> agatha_physics | 61.0 | 30.5 | -30.5 | [-32.6, -28.1] | 570 |
| floor | all_objects | rest_face_error_rate | hold_initial -> agatha_physics | 0.567 | 0.221 | -0.346 | [-0.384, -0.302] | 570 |

## real_contactnets_cube_calib (REAL WORLD, approximate)

> REAL-WORLD pose tracks (motion capture / AprilTags) of a single tossed box; friction and restitution are assumed (see `materials`), velocities are finite differences, 20 Hz resampling. APPROXIMATE.

Evaluated 2026-11-15 07:37:53 with contact_gnn checkpoint sha256 e4b5e4a3a905 (epoch 246, file written 2026-11-15 06:35:20), contact_gnn.py sha256 7f4ce7fcf5f5.

570 trajectories, 7434 transitions at 20 Hz, horizon 0.5-2.95 s; AprilTag pose tracking, 148 Hz; object: acrylic cube, 0.1048 x 0.1048 x 0.1048 m (Agatha x, y, z), 0.370 kg; surface: flat wooden surface. Assumed materials (calibrated from sliding deceleration): object friction 0.467, support friction 0.467 (effective 0.218), restitution 0.000 (effective 0.000). Ground truth rests on the same face that was down at release in 0.433 of the trajectories (= the rest-face accuracy of `hold_initial`). Tables larger than the training range: 0.

Columns: position error (m) at 0.25 / 0.5 / 1 s (n = trajectories that last that long) and at the last frame (rest); orientation error at the last frame; rest-face accuracy (same body face down as in the measurement); `fell` false alarms (the real box never leaves the surface); penetration (mm).

### real_contactnets_cube_calib / ground = slab (570 trajectories)

| engine | pos 0.25s | pos 0.5s | pos 1s (n) | rest pos mean | rest pos median | ori final (deg) | rest-face acc | fell false alarms | pen mean (mm) | pen capped (mm) | diverged |
|---|---|---|---|---|---|---|---|---|---|---|---|
| reference | 0.000 | 0.000 | 0.000 (10) | 0.000 | 0.000 | 0.0 | 1.000 | 0 | 0.25 | 0.25 | 0 |
| hold_initial | 0.256 | 0.382 | 0.499 (10) | 0.400 | 0.396 | 61.0 | 0.433 | 0 | 0.00 | 0.00 | 0 |
| agatha_physics | 0.020 | 0.057 | 0.077 (10) | 0.073 | 0.070 | 24.1 | 0.856 | 0 | 0.47 | 0.47 | 0 |
| old_mlp | 0.056 | 0.135 | 0.197 (10) | 0.150 | 0.144 | 53.1 | 0.646 | 0 | 25.19 | 21.43 | 0 |
| old_mlp_projected | 0.049 | 0.127 | 0.187 (10) | 0.142 | 0.138 | 50.9 | 0.609 | 0 | 0.00 | 0.00 | 0 |
| new_raw | 0.021 | 0.060 | 0.084 (10) | 0.077 | 0.074 | 31.1 | 0.791 | 0 | 0.31 | 0.31 | 0 |
| new_projected | 0.021 | 0.058 | 0.082 (10) | 0.075 | 0.073 | 31.4 | 0.788 | 0 | 0.00 | 0.00 | 0 |

### real_contactnets_cube_calib / ground = floor (570 trajectories)

| engine | pos 0.25s | pos 0.5s | pos 1s (n) | rest pos mean | rest pos median | ori final (deg) | rest-face acc | fell false alarms | pen mean (mm) | pen capped (mm) | diverged |
|---|---|---|---|---|---|---|---|---|---|---|---|
| reference | 0.000 | 0.000 | 0.000 (10) | 0.000 | 0.000 | 0.0 | 1.000 | - | 0.25 | 0.25 | 0 |
| hold_initial | 0.256 | 0.382 | 0.499 (10) | 0.400 | 0.396 | 61.0 | 0.433 | - | 0.00 | 0.00 | 0 |
| agatha_physics | 0.020 | 0.057 | 0.077 (10) | 0.073 | 0.070 | 24.1 | 0.856 | - | 0.47 | 0.47 | 0 |
| old_mlp | 0.102 | 0.192 | 0.277 (10) | 0.208 | 0.204 | 53.3 | 0.579 | - | 0.08 | 0.08 | 0 |
| old_mlp_projected | 0.102 | 0.192 | 0.277 (10) | 0.208 | 0.204 | 53.3 | 0.579 | - | 0.00 | 0.00 | 0 |
| new_raw | 0.035 | 0.098 | 0.141 (10) | 0.117 | 0.113 | 49.2 | 0.570 | - | 0.23 | 0.23 | 0 |
| new_projected | 0.035 | 0.098 | 0.140 (10) | 0.117 | 0.113 | 49.1 | 0.575 | - | 0.00 | 0.00 | 0 |

### real_contactnets_cube_calib / paired episode bootstrap (B - A; negative = engine B is better)

| ground | subset | metric | A -> B | A | B | delta | 95% CI | episodes |
|---|---|---|---|---|---|---|---|---|
| slab | all_objects | rest_pos_err_m | old_mlp -> new_projected | 0.150 | 0.075 | -0.075 | [-0.079, -0.070] | 570 |
| slab | all_objects | ori_err_final_deg | old_mlp -> new_projected | 53.1 | 31.4 | -21.7 | [-23.9, -19.3] | 570 |
| slab | all_objects | rest_face_error_rate | old_mlp -> new_projected | 0.354 | 0.212 | -0.142 | [-0.184, -0.102] | 570 |
| slab | all_objects | rest_pos_err_m | old_mlp_projected -> new_projected | 0.142 | 0.075 | -0.067 | [-0.071, -0.062] | 570 |
| slab | all_objects | ori_err_final_deg | old_mlp_projected -> new_projected | 50.9 | 31.4 | -19.5 | [-21.7, -17.3] | 570 |
| slab | all_objects | rest_face_error_rate | old_mlp_projected -> new_projected | 0.391 | 0.212 | -0.179 | [-0.219, -0.140] | 570 |
| slab | all_objects | rest_pos_err_m | agatha_physics -> new_projected | 0.073 | 0.075 | 0.002 | [0.000, 0.005] | 570 |
| slab | all_objects | ori_err_final_deg | agatha_physics -> new_projected | 24.1 | 31.4 | 7.3 | [5.2, 9.6] | 570 |
| slab | all_objects | rest_face_error_rate | agatha_physics -> new_projected | 0.144 | 0.212 | 0.068 | [0.042, 0.095] | 570 |
| slab | all_objects | rest_pos_err_m | agatha_physics -> new_raw | 0.073 | 0.077 | 0.004 | [0.002, 0.007] | 570 |
| slab | all_objects | ori_err_final_deg | agatha_physics -> new_raw | 24.1 | 31.1 | 7.0 | [4.8, 9.2] | 570 |
| slab | all_objects | rest_face_error_rate | agatha_physics -> new_raw | 0.144 | 0.209 | 0.065 | [0.039, 0.090] | 570 |
| slab | all_objects | rest_pos_err_m | hold_initial -> new_projected | 0.400 | 0.075 | -0.325 | [-0.330, -0.320] | 570 |
| slab | all_objects | ori_err_final_deg | hold_initial -> new_projected | 61.0 | 31.4 | -29.6 | [-32.2, -26.9] | 570 |
| slab | all_objects | rest_face_error_rate | hold_initial -> new_projected | 0.567 | 0.212 | -0.354 | [-0.398, -0.309] | 570 |
| slab | all_objects | rest_pos_err_m | hold_initial -> agatha_physics | 0.400 | 0.073 | -0.328 | [-0.332, -0.323] | 570 |
| slab | all_objects | ori_err_final_deg | hold_initial -> agatha_physics | 61.0 | 24.1 | -36.9 | [-39.3, -34.4] | 570 |
| slab | all_objects | rest_face_error_rate | hold_initial -> agatha_physics | 0.567 | 0.144 | -0.423 | [-0.467, -0.375] | 570 |
| floor | all_objects | rest_pos_err_m | old_mlp -> new_projected | 0.208 | 0.117 | -0.091 | [-0.095, -0.087] | 570 |
| floor | all_objects | ori_err_final_deg | old_mlp -> new_projected | 53.3 | 49.1 | -4.2 | [-6.6, -1.7] | 570 |
| floor | all_objects | rest_face_error_rate | old_mlp -> new_projected | 0.421 | 0.425 | 0.004 | [-0.023, 0.030] | 570 |
| floor | all_objects | rest_pos_err_m | old_mlp_projected -> new_projected | 0.208 | 0.117 | -0.091 | [-0.095, -0.087] | 570 |
| floor | all_objects | ori_err_final_deg | old_mlp_projected -> new_projected | 53.3 | 49.1 | -4.2 | [-6.6, -1.7] | 570 |
| floor | all_objects | rest_face_error_rate | old_mlp_projected -> new_projected | 0.421 | 0.425 | 0.004 | [-0.023, 0.030] | 570 |
| floor | all_objects | rest_pos_err_m | agatha_physics -> new_projected | 0.073 | 0.117 | 0.044 | [0.041, 0.047] | 570 |
| floor | all_objects | ori_err_final_deg | agatha_physics -> new_projected | 24.1 | 49.1 | 25.0 | [21.7, 28.4] | 570 |
| floor | all_objects | rest_face_error_rate | agatha_physics -> new_projected | 0.144 | 0.425 | 0.281 | [0.235, 0.323] | 570 |
| floor | all_objects | rest_pos_err_m | agatha_physics -> new_raw | 0.073 | 0.117 | 0.044 | [0.041, 0.047] | 570 |
| floor | all_objects | ori_err_final_deg | agatha_physics -> new_raw | 24.1 | 49.2 | 25.1 | [21.8, 28.5] | 570 |
| floor | all_objects | rest_face_error_rate | agatha_physics -> new_raw | 0.144 | 0.430 | 0.286 | [0.242, 0.330] | 570 |
| floor | all_objects | rest_pos_err_m | hold_initial -> new_projected | 0.400 | 0.117 | -0.284 | [-0.288, -0.280] | 570 |
| floor | all_objects | ori_err_final_deg | hold_initial -> new_projected | 61.0 | 49.1 | -11.9 | [-13.9, -9.9] | 570 |
| floor | all_objects | rest_face_error_rate | hold_initial -> new_projected | 0.567 | 0.425 | -0.142 | [-0.170, -0.114] | 570 |
| floor | all_objects | rest_pos_err_m | hold_initial -> agatha_physics | 0.400 | 0.073 | -0.328 | [-0.332, -0.323] | 570 |
| floor | all_objects | ori_err_final_deg | hold_initial -> agatha_physics | 61.0 | 24.1 | -36.9 | [-39.3, -34.4] | 570 |
| floor | all_objects | rest_face_error_rate | hold_initial -> agatha_physics | 0.567 | 0.144 | -0.423 | [-0.467, -0.375] | 570 |

## Notes

- APPROXIMATE: external data converted with documented approximations (scripts/prepare_external_data.py). Kubric / CLEVRER / Physion are simulator output from other generators; the `real_*` datasets are real-world pose tracks of ONE tossed box with ASSUMED friction / restitution. None of this is a validation of the product.
- Kubric / CLEVRER / Physion are simulators (PyBullet, Blender/Bullet, TDW/PhysX). The `real_*` datasets are real-world pose tracks (I.AM. motion capture, ContactNets AprilTags) of a single tossed box; no video is evaluated anywhere.
- Real-world data: friction / restitution are assumed (default 0.5 x 0.5 / 0, `_calib` = friction from the measured sliding deceleration, `_gridfit` = PyBullet grid-search optimum, half of the trajectories were used to pick it). `agatha_physics` is the simulator baseline: its error is the simulator-vs-reality gap under those assumptions, and the learned models were trained to imitate that simulator.
- Real-world data covers one box alone on a flat surface: no box-box contact, no stacking, no edges.
- Non-box shapes are replaced by their bounding boxes; the `reference` row shows how much penetration that alone creates.
- `tipped` is the relative rule (initial up axis rotated > 35 deg), not agatha.physics.summarize's absolute rule.
- `rest pos` is the position error at the last available frame; many Kubric / CLEVRER objects are still moving then, see the 'settled' column for the objects that are at rest in the ground truth.
- Physion: 10 trials only; every number there is anecdotal.
- CLEVRER friction was calibrated from the data set itself (median sliding deceleration), sizes / masses / time base are assumed.
- Chaotic rollouts (tosses, collapses) are sensitive to initial conditions; a position error of several object sizes is expected even for a perfect model of a slightly different simulator.
