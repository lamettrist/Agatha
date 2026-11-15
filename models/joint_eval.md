# Joint (articulated body) evaluation - EXPERIMENTAL

> EXPERIMENTAL: articulated-body (joint) extension track. Not the shipped model; the shipped box model is models/contact_gnn.pt. Synthetic PyBullet box-link articulations only.

Model `models/runs/joints_r2/best.pt` (config {'width': 128, 'rounds': 8, 'edge_radius': 0.12, 'joint_features': 80, 'shared_rounds': 4}, epoch 12, 2,010,892 parameters).

Data `data/joints_v1` split `test`: 1996 episodes, 4 s autoregressive rollouts, 12 body slots, 5271 joints (3814 hinge, 1151 ball, 162 prismatic, 144 fixed).

Engines:
- `physics_reference`: PyBullet ground truth scored with the same metrics (floor for constraint / penetration / rest metrics)
- `joint_raw`: JointGNN alone: learned model only, no projection
- `joint_projected`: JointGNN + joint_gnn.project_joints after every step (non-learned contact + joint projection)
- `contact_projected`: baseline: box model contact_gnn + contact_gnn.project, joints ignored
- `hold_initial`: trivial baseline: every body keeps its initial pose (zero velocity)

## overall (1996 episodes)

| engine | pos 0.5s (m) | pos 1s | pos 2s | pos 4s | pos 2s articulated | pos 2s free | ori 2s (deg) | hinge err 1s (rad) | hinge err 2s | slide err 2s (m) | gap mean (mm) | gap max (mm) | off-axis mean (deg) | limit viol (%) | limit viol > slop (%) | pen mean (mm) | pen >5mm (%) | fell acc | rest drift (mm/s) | diverged | ms/step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| physics_reference | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00 | 0.000 | 0.000 | 0.0000 | 0.00 | 0.0 | 0.00 | 1.95 | 1.29 | 0.05 | 0.15 | 1.000 | 0.32 | 0 | - |
| joint_raw | 0.0388 | 0.0825 | 0.1081 | 0.1105 | 0.1412 | 0.0533 | 32.30 | 0.320 | 0.338 | 0.0260 | 0.95 | 80.8 | 0.61 | 0.80 | 0.22 | 0.34 | 1.31 | 0.964 | 6.85 | 0 | - |
| joint_projected | 0.0396 | 0.0831 | 0.1049 | 0.1072 | 0.1365 | 0.0526 | 33.28 | 0.336 | 0.358 | 0.0250 | 0.00 | 0.0 | 0.00 | 4.65 | 0.64 | 0.41 | 1.39 | 0.964 | 5.89 | 0 | - |
| contact_projected | 0.1353 | 0.2158 | 0.2555 | 0.2574 | 0.3721 | 0.0628 | 53.60 | 0.902 | 0.945 | 0.2019 | 307.68 | 5562.8 | 31.90 | 22.28 | 21.40 | 0.02 | 0.05 | 0.880 | 11.04 | 0 | - |
| hold_initial | 0.2536 | 0.3720 | 0.4089 | 0.4103 | 0.5559 | 0.1662 | 53.36 | 0.603 | 0.623 | 0.0543 | 0.00 | 0.0 | 0.00 | 0.00 | 0.00 | 0.17 | 0.39 | 0.767 | 0.00 | 0 | - |

## gentle (696 episodes)

| engine | pos 0.5s (m) | pos 1s | pos 2s | pos 4s | pos 2s articulated | pos 2s free | ori 2s (deg) | hinge err 1s (rad) | hinge err 2s | slide err 2s (m) | gap mean (mm) | gap max (mm) | off-axis mean (deg) | limit viol (%) | limit viol > slop (%) | pen mean (mm) | pen >5mm (%) | fell acc | rest drift (mm/s) | diverged | ms/step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| physics_reference | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00 | 0.000 | 0.000 | 0.0000 | 0.00 | 0.0 | 0.00 | 0.50 | 0.22 | 0.03 | 0.01 | 1.000 | 0.26 | 0 | - |
| joint_raw | 0.0055 | 0.0072 | 0.0089 | 0.0099 | 0.0148 | 0.0035 | 4.08 | 0.100 | 0.128 | 0.0225 | 0.65 | 62.6 | 0.44 | 0.67 | 0.31 | 0.15 | 0.08 | 0.998 | 1.25 | 0 | - |
| joint_projected | 0.0058 | 0.0075 | 0.0103 | 0.0114 | 0.0182 | 0.0033 | 4.42 | 0.106 | 0.136 | 0.0211 | 0.00 | 0.0 | 0.00 | 4.18 | 0.40 | 0.22 | 0.47 | 0.996 | 1.12 | 0 | - |
| contact_projected | 0.0427 | 0.0540 | 0.0577 | 0.0589 | 0.1161 | 0.0052 | 14.60 | 0.582 | 0.616 | 0.1352 | 118.53 | 1296.4 | 13.87 | 17.54 | 16.37 | 0.01 | 0.00 | 0.970 | 3.38 | 0 | - |
| hold_initial | 0.0429 | 0.0412 | 0.0424 | 0.0427 | 0.0644 | 0.0226 | 10.89 | 0.458 | 0.492 | 0.0454 | 0.00 | 0.0 | 0.00 | 0.00 | 0.00 | 0.16 | 0.46 | 1.000 | 0.00 | 0 | - |

## chaotic (1300 episodes)

| engine | pos 0.5s (m) | pos 1s | pos 2s | pos 4s | pos 2s articulated | pos 2s free | ori 2s (deg) | hinge err 1s (rad) | hinge err 2s | slide err 2s (m) | gap mean (mm) | gap max (mm) | off-axis mean (deg) | limit viol (%) | limit viol > slop (%) | pen mean (mm) | pen >5mm (%) | fell acc | rest drift (mm/s) | diverged | ms/step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| physics_reference | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00 | 0.000 | 0.000 | 0.0000 | 0.00 | 0.0 | 0.00 | 2.31 | 1.56 | 0.06 | 0.20 | 1.000 | 0.34 | 0 | - |
| joint_raw | 0.0504 | 0.1089 | 0.1429 | 0.1458 | 0.1723 | 0.0817 | 42.20 | 0.371 | 0.387 | 0.0300 | 1.02 | 80.8 | 0.65 | 0.84 | 0.20 | 0.41 | 1.75 | 0.952 | 9.23 | 0 | - |
| joint_projected | 0.0514 | 0.1096 | 0.1380 | 0.1409 | 0.1656 | 0.0807 | 43.41 | 0.389 | 0.409 | 0.0293 | 0.00 | 0.0 | 0.00 | 4.77 | 0.70 | 0.48 | 1.71 | 0.953 | 7.91 | 0 | - |
| contact_projected | 0.1678 | 0.2725 | 0.3249 | 0.3271 | 0.4352 | 0.0956 | 67.28 | 0.975 | 1.021 | 0.2774 | 349.10 | 5562.8 | 36.49 | 23.47 | 22.66 | 0.03 | 0.06 | 0.848 | 14.29 | 0 | - |
| hold_initial | 0.3275 | 0.4880 | 0.5375 | 0.5393 | 0.6769 | 0.2478 | 68.26 | 0.636 | 0.653 | 0.0644 | 0.00 | 0.0 | 0.00 | 0.00 | 0.00 | 0.17 | 0.36 | 0.685 | 0.00 | 0 | - |

## Per scenario kind: pos err 2s (m) / hinge err 2s (rad) / mean anchor gap (mm) / mean penetration (mm)

| kind | n | physics_reference | joint_raw | joint_projected | contact_projected | hold_initial |
|---|---|---|---|---|---|---|
| box_lid | 134 | 0.0000 / 0.000 / 0.00 / 0.04 | 0.0735 / 0.230 / 0.73 / 0.25 | 0.0781 / 0.257 / 0.00 / 0.27 | 0.1303 / 0.701 / 246.12 / 0.01 | 0.2968 / 0.643 / 0.00 / 0.00 |
| chain | 265 | 0.0000 / 0.000 / 0.00 / 0.05 | 0.1142 / 0.377 / 0.89 / 0.20 | 0.1225 / 0.417 / 0.00 / 0.43 | 0.3177 / 0.763 / 259.64 / 0.01 | 0.4607 / 0.528 / 0.00 / 0.01 |
| door | 153 | 0.0000 / 0.000 / 0.00 / 0.04 | 0.0623 / 0.469 / 0.75 / 0.34 | 0.0620 / 0.450 / 0.00 / 0.39 | 0.1311 / 0.892 / 176.03 / 0.05 | 0.1466 / 0.946 / 0.00 / 0.00 |
| drawer | 162 | 0.0000 / - / 0.00 / 0.04 | 0.0858 / - / 2.07 / 0.21 | 0.0750 / - / 0.00 / 0.25 | 0.1709 / - / 223.29 / 0.02 | 0.2840 / - / 0.00 / 0.00 |
| laptop | 238 | 0.0000 / 0.000 / 0.00 / 0.04 | 0.0674 / 0.162 / 0.55 / 0.22 | 0.0627 / 0.157 / 0.00 / 0.30 | 0.1114 / 0.651 / 171.96 / 0.01 | 0.2875 / 0.580 / 0.00 / 0.00 |
| pendulum | 253 | 0.0000 / 0.000 / 0.00 / 0.02 | 0.0384 / 0.129 / 1.00 / 0.22 | 0.0366 / 0.143 / 0.00 / 0.36 | 0.3039 / 1.438 / 400.37 / 0.03 | 0.1646 / 0.870 / 0.00 / 0.00 |
| plain_collision | 18 | 0.0000 / - / - / 0.07 | 0.1126 / - / - / 0.50 | 0.1169 / - / - / 0.05 | 0.1167 / - / - / 0.05 | 0.2919 / - / - / 0.00 |
| plain_desk | 20 | 0.0000 / - / - / 0.04 | 0.0255 / - / - / 0.19 | 0.0204 / - / - / 0.00 | 0.0204 / - / - / 0.00 | 0.1050 / - / - / 2.56 |
| plain_drop | 21 | 0.0000 / - / - / 0.16 | 0.1273 / - / - / 0.64 | 0.1273 / - / - / 0.04 | 0.1290 / - / - / 0.04 | 0.2604 / - / - / 0.00 |
| plain_edge_fall | 17 | 0.0000 / - / - / 0.19 | 0.0478 / - / - / 0.31 | 0.0499 / - / - / 0.00 | 0.0499 / - / - / 0.00 | 0.2747 / - / - / 0.00 |
| plain_gentle_push | 43 | 0.0000 / - / - / 0.03 | 0.0059 / - / - / 0.21 | 0.0054 / - / - / 0.00 | 0.0054 / - / - / 0.00 | 0.0267 / - / - / 0.00 |
| plain_legacy_layout | 32 | 0.0000 / - / - / 0.09 | 0.0901 / - / - / 0.51 | 0.1203 / - / - / 0.02 | 0.1202 / - / - / 0.02 | 0.2682 / - / - / 5.49 |
| plain_noclip | 6 | 0.0000 / - / - / 0.18 | 0.0945 / - / - / 0.45 | 0.0779 / - / - / 0.03 | 0.0778 / - / - / 0.03 | 0.8102 / - / - / 0.00 |
| plain_rest | 44 | 0.0000 / - / - / 0.10 | 0.0010 / - / - / 0.31 | 0.0003 / - / - / 0.00 | 0.0003 / - / - / 0.00 | 0.0003 / - / - / 0.00 |
| plain_slide_stop | 27 | 0.0000 / - / - / 0.04 | 0.0115 / - / - / 0.19 | 0.0185 / - / - / 0.02 | 0.0185 / - / - / 0.02 | 0.0555 / - / - / 0.00 |
| plain_stack_collapse | 17 | 0.0000 / - / - / 0.16 | 0.1159 / - / - / 0.56 | 0.1104 / - / - / 0.10 | 0.1096 / - / - / 0.11 | 0.3268 / - / - / 0.80 |
| plain_tilt_slide | 10 | 0.0000 / - / - / 0.21 | 0.3817 / - / - / 0.75 | 0.3844 / - / - / 0.12 | 0.3933 / - / - / 0.10 | 1.0909 / - / - / 12.20 |
| plain_topple | 15 | 0.0000 / - / - / 0.06 | 0.0393 / - / - / 0.34 | 0.0399 / - / - / 0.06 | 0.0399 / - / - / 0.06 | 0.2004 / - / - / 0.34 |
| plain_toss | 11 | 0.0000 / - / - / 0.13 | 0.0392 / - / - / 0.26 | 0.0483 / - / - / 0.06 | 0.0483 / - / - / 0.06 | 0.3700 / - / - / 2.80 |
| ragdoll | 270 | 0.0000 / 0.000 / 0.00 / 0.06 | 0.1997 / 0.403 / 0.98 / 0.51 | 0.1861 / 0.426 / 0.00 / 0.62 | 0.4285 / 1.045 / 385.84 / 0.03 | 0.7410 / 0.557 / 0.00 / 0.02 |
| robot_arm | 240 | 0.0000 / 0.000 / 0.00 / 0.04 | 0.0331 / 0.253 / 0.87 / 0.32 | 0.0333 / 0.265 / 0.00 / 0.53 | 0.1421 / 0.840 / 164.42 / 0.01 | 0.1591 / 0.736 / 0.00 / 0.00 |

## Box-only regression check (`data/v2` split `test`, 4998 episodes, zero joints)

| engine | subset | pos 1s (m) | pos 2s | pos 4s | ori 2s (deg) | fell acc | fell P | fell R | pen mean (mm) | pen >5mm (%) | rest drift (mm/s) | diverged |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| physics_reference | overall | 0.0000 | 0.0000 | 0.0000 | 0.00 | 1.000 | 1.000 | 1.000 | 0.14 | 0.29 | 0.24 | 0 |
| physics_reference | gentle | 0.0000 | 0.0000 | 0.0000 | 0.00 | 1.000 | - | - | 0.08 | 0.08 | 0.18 | 0 |
| physics_reference | chaotic | 0.0000 | 0.0000 | 0.0000 | 0.00 | 1.000 | 1.000 | 1.000 | 0.21 | 0.51 | 0.31 | 0 |
| joint_raw | overall | 0.0420 | 0.0572 | 0.0612 | 13.30 | 0.980 | 0.924 | 0.937 | 0.38 | 0.85 | 5.82 | 0 |
| joint_raw | gentle | 0.0021 | 0.0025 | 0.0034 | 0.66 | 1.000 | 0.000 | - | 0.23 | 0.08 | 0.68 | 0 |
| joint_raw | chaotic | 0.0822 | 0.1125 | 0.1196 | 26.07 | 0.959 | 0.925 | 0.937 | 0.53 | 1.62 | 11.74 | 0 |
| joint_projected | overall | 0.0444 | 0.0603 | 0.0640 | 13.76 | 0.979 | 0.917 | 0.939 | 0.05 | 0.09 | 5.47 | 0 |
| joint_projected | gentle | 0.0025 | 0.0027 | 0.0028 | 0.55 | 0.999 | 0.000 | - | 0.02 | 0.03 | 0.30 | 0 |
| joint_projected | chaotic | 0.0867 | 0.1184 | 0.1256 | 27.08 | 0.958 | 0.918 | 0.939 | 0.07 | 0.14 | 11.42 | 0 |
| contact_projected | overall | 0.0444 | 0.0607 | 0.0644 | 13.82 | 0.979 | 0.917 | 0.940 | 0.05 | 0.09 | 5.52 | 0 |
| contact_projected | gentle | 0.0025 | 0.0027 | 0.0029 | 0.56 | 0.999 | 0.000 | - | 0.02 | 0.03 | 0.30 | 0 |
| contact_projected | chaotic | 0.0868 | 0.1192 | 0.1265 | 27.20 | 0.958 | 0.918 | 0.940 | 0.07 | 0.15 | 11.52 | 0 |
| contact_raw | overall | 0.0420 | 0.0572 | 0.0612 | 13.30 | 0.980 | 0.924 | 0.937 | 0.38 | 0.85 | 5.82 | 0 |
| contact_raw | gentle | 0.0021 | 0.0025 | 0.0034 | 0.66 | 1.000 | 0.000 | - | 0.23 | 0.08 | 0.68 | 0 |
| contact_raw | chaotic | 0.0822 | 0.1125 | 0.1196 | 26.07 | 0.959 | 0.925 | 0.937 | 0.53 | 1.62 | 11.73 | 0 |

Paired bootstrap, JointGNN (zero joints) minus contact_gnn (negative = JointGNN better):

| pair | metric | contact_gnn | joint_gnn | delta | 95% CI |
|---|---|---|---|---|---|
| projected | pos_err_2s_m | 0.06068 | 0.06026 | -0.00041 | [-0.00087, -0.00000] |
| projected | chaotic_pos_err_2s_m | 0.11917 | 0.11837 | -0.00080 | [-0.00169, -0.00004] |
| projected | gentle_pos_err_2s_m | 0.00273 | 0.00270 | -0.00003 | [-0.00008, 0.00000] |
| projected | mean_penetration_m | 0.00005 | 0.00005 | -0.00000 | [-0.00000, 0.00000] |
| projected | rest_drift_mps | 0.00552 | 0.00547 | -0.00005 | [-0.00015, 0.00004] |
| raw | pos_err_2s_m | 0.05723 | 0.05724 | 0.00000 | [-0.00000, 0.00001] |
| raw | chaotic_pos_err_2s_m | 0.11246 | 0.11247 | 0.00001 | [-0.00000, 0.00002] |
| raw | gentle_pos_err_2s_m | 0.00251 | 0.00251 | -0.00000 | [-0.00000, 0.00000] |
| raw | mean_penetration_m | 0.00038 | 0.00038 | -0.00000 | [-0.00000, 0.00000] |
| raw | rest_drift_mps | 0.00582 | 0.00582 | 0.00000 | [-0.00000, 0.00001] |

One-step |JointGNN(zero joints) - ContactGNN| on 256 box states: max 1.907e-06, mean 2.148e-09 (exactly zero is only expected for a warm start that has not been trained further).

Notes:
- EXPERIMENTAL: articulated-body (joint) extension track. Not the shipped model; the shipped box model is models/contact_gnn.pt. Synthetic PyBullet box-link articulations only.
- Autoregressive rollouts from the true t=0 state over the full episode; never teacher-forced.
- joint_projected applies joint_gnn.project_joints (non-learned contact + joint projection) after every step; joint_raw is the learned model alone.
- contact_projected ignores the joints entirely (the shipped box model on the same bodies): the gap to it is what the joint edges buy.
- Joint coordinate error compares joint_gnn.joint_coordinates of the predicted and of the TRUE states (hinge error wrapped to +-pi); ball and fixed joints have no coordinate. data_checks reports how well that function reproduces the dataset joint_pos.
- Anchor gap / off-axis / limit metrics come from joint_gnn.joint_violation; physics_reference shows PyBullet's own residual.
- Limit violation = joint-frames with limit excess > 0.02 rad (hinge) or 0.002 m (prismatic); "> slop" counts only excess beyond joint_gnn.LIMIT_SLOP = 0.05 rad / 0.005 m, the tolerance project_joints itself clamps to (Bullet limits are soft, so physics_reference is non-zero in the strict column).
- Penetration = contact_gnn corner-in-box depth (support, floor, other boxes) with pairs directly connected by a joint excluded.
- fell / rest drift follow scripts/eval_contact.py (dataset fell flag and rest frames).
- Latency: batch 1, CPU (4 threads), 60 autoregressive steps on the first episode with a joint; shared machine, indicative only.
