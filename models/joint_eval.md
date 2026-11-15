# Joint (articulated body) evaluation - EXPERIMENTAL

> EXPERIMENTAL: articulated-body (joint) extension track. Not the shipped model; the shipped box model is models/contact_gnn.pt. Synthetic PyBullet box-link articulations only.

Model `models/runs/joint_v1/best.pt` (config {'width': 128, 'rounds': 6, 'edge_radius': 0.12, 'joint_features': 80, 'shared_rounds': 4}, epoch 27, 1,548,300 parameters).

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
| joint_raw | 0.0408 | 0.0888 | 0.1155 | 0.1176 | 0.1537 | 0.0524 | 33.59 | 0.332 | 0.343 | 0.0254 | 1.37 | 86.2 | 0.69 | 0.93 | 0.24 | 0.43 | 1.93 | 0.964 | 9.15 | 0 | 8.51 |
| joint_projected | 0.0417 | 0.0898 | 0.1137 | 0.1163 | 0.1487 | 0.0559 | 34.32 | 0.358 | 0.391 | 0.0256 | 0.00 | 0.0 | 0.00 | 4.87 | 0.77 | 0.51 | 1.92 | 0.964 | 8.18 | 0 | 19.67 |
| contact_projected | 0.1353 | 0.2157 | 0.2548 | 0.2563 | 0.3708 | 0.0633 | 53.54 | 0.897 | 0.945 | 0.2036 | 303.99 | 4384.2 | 31.70 | 22.49 | 21.59 | 0.02 | 0.05 | 0.880 | 10.74 | 0 | 10.68 |
| hold_initial | 0.2536 | 0.3720 | 0.4089 | 0.4103 | 0.5559 | 0.1662 | 53.36 | 0.603 | 0.623 | 0.0543 | 0.00 | 0.0 | 0.00 | 0.00 | 0.00 | 0.17 | 0.39 | 0.767 | 0.00 | 0 | 0.03 |

## gentle (696 episodes)

| engine | pos 0.5s (m) | pos 1s | pos 2s | pos 4s | pos 2s articulated | pos 2s free | ori 2s (deg) | hinge err 1s (rad) | hinge err 2s | slide err 2s (m) | gap mean (mm) | gap max (mm) | off-axis mean (deg) | limit viol (%) | limit viol > slop (%) | pen mean (mm) | pen >5mm (%) | fell acc | rest drift (mm/s) | diverged | ms/step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| physics_reference | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00 | 0.000 | 0.000 | 0.0000 | 0.00 | 0.0 | 0.00 | 0.50 | 0.22 | 0.03 | 0.01 | 1.000 | 0.26 | 0 | - |
| joint_raw | 0.0065 | 0.0092 | 0.0127 | 0.0140 | 0.0233 | 0.0030 | 5.01 | 0.107 | 0.143 | 0.0222 | 1.07 | 78.4 | 0.51 | 0.56 | 0.17 | 0.18 | 0.25 | 0.994 | 1.41 | 0 | 8.51 |
| joint_projected | 0.0065 | 0.0119 | 0.0149 | 0.0176 | 0.0273 | 0.0038 | 5.62 | 0.131 | 0.170 | 0.0216 | 0.00 | 0.0 | 0.00 | 3.23 | 0.30 | 0.26 | 0.55 | 0.990 | 2.16 | 0 | 19.67 |
| contact_projected | 0.0426 | 0.0538 | 0.0576 | 0.0588 | 0.1158 | 0.0052 | 14.57 | 0.572 | 0.606 | 0.1356 | 118.08 | 1296.3 | 13.41 | 17.30 | 16.16 | 0.01 | 0.00 | 0.970 | 3.38 | 0 | 10.68 |
| hold_initial | 0.0429 | 0.0412 | 0.0424 | 0.0427 | 0.0644 | 0.0226 | 10.89 | 0.458 | 0.492 | 0.0454 | 0.00 | 0.0 | 0.00 | 0.00 | 0.00 | 0.16 | 0.46 | 1.000 | 0.00 | 0 | 0.03 |

## chaotic (1300 episodes)

| engine | pos 0.5s (m) | pos 1s | pos 2s | pos 4s | pos 2s articulated | pos 2s free | ori 2s (deg) | hinge err 1s (rad) | hinge err 2s | slide err 2s (m) | gap mean (mm) | gap max (mm) | off-axis mean (deg) | limit viol (%) | limit viol > slop (%) | pen mean (mm) | pen >5mm (%) | fell acc | rest drift (mm/s) | diverged | ms/step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| physics_reference | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00 | 0.000 | 0.000 | 0.0000 | 0.00 | 0.0 | 0.00 | 2.31 | 1.56 | 0.06 | 0.20 | 1.000 | 0.34 | 0 | - |
| joint_raw | 0.0528 | 0.1167 | 0.1516 | 0.1540 | 0.1858 | 0.0804 | 43.62 | 0.384 | 0.389 | 0.0290 | 1.44 | 86.2 | 0.74 | 1.02 | 0.26 | 0.52 | 2.52 | 0.954 | 12.43 | 0 | 8.51 |
| joint_projected | 0.0541 | 0.1171 | 0.1483 | 0.1509 | 0.1785 | 0.0855 | 44.39 | 0.410 | 0.442 | 0.0301 | 0.00 | 0.0 | 0.00 | 5.28 | 0.89 | 0.60 | 2.40 | 0.955 | 10.74 | 0 | 19.67 |
| contact_projected | 0.1679 | 0.2725 | 0.3240 | 0.3256 | 0.4336 | 0.0964 | 67.21 | 0.972 | 1.023 | 0.2804 | 344.71 | 4384.2 | 36.35 | 23.79 | 22.96 | 0.03 | 0.07 | 0.849 | 13.85 | 0 | 10.68 |
| hold_initial | 0.3275 | 0.4880 | 0.5375 | 0.5393 | 0.6769 | 0.2478 | 68.26 | 0.636 | 0.653 | 0.0644 | 0.00 | 0.0 | 0.00 | 0.00 | 0.00 | 0.17 | 0.36 | 0.685 | 0.00 | 0 | 0.03 |

## Per scenario kind: pos err 2s (m) / hinge err 2s (rad) / mean anchor gap (mm) / mean penetration (mm)

| kind | n | physics_reference | joint_raw | joint_projected | contact_projected | hold_initial |
|---|---|---|---|---|---|---|
| box_lid | 134 | 0.0000 / 0.000 / 0.00 / 0.04 | 0.0753 / 0.243 / 0.97 / 0.30 | 0.0710 / 0.323 / 0.00 / 0.36 | 0.1307 / 0.719 / 247.09 / 0.01 | 0.2968 / 0.643 / 0.00 / 0.00 |
| chain | 265 | 0.0000 / 0.000 / 0.00 / 0.05 | 0.1253 / 0.405 / 1.29 / 0.22 | 0.1259 / 0.469 / 0.00 / 0.53 | 0.3172 / 0.766 / 259.76 / 0.01 | 0.4607 / 0.528 / 0.00 / 0.01 |
| door | 153 | 0.0000 / 0.000 / 0.00 / 0.04 | 0.0705 / 0.518 / 0.98 / 0.46 | 0.0709 / 0.509 / 0.00 / 0.40 | 0.1323 / 0.893 / 174.74 / 0.05 | 0.1466 / 0.946 / 0.00 / 0.00 |
| drawer | 162 | 0.0000 / - / 0.00 / 0.04 | 0.0829 / - / 4.16 / 0.24 | 0.0805 / - / 0.00 / 0.26 | 0.1721 / - / 226.31 / 0.02 | 0.2840 / - / 0.00 / 0.00 |
| laptop | 238 | 0.0000 / 0.000 / 0.00 / 0.04 | 0.0684 / 0.138 / 0.76 / 0.29 | 0.0712 / 0.173 / 0.00 / 0.38 | 0.1115 / 0.629 / 172.55 / 0.01 | 0.2875 / 0.580 / 0.00 / 0.00 |
| pendulum | 253 | 0.0000 / 0.000 / 0.00 / 0.02 | 0.0351 / 0.124 / 1.18 / 0.25 | 0.0355 / 0.125 / 0.00 / 0.36 | 0.3040 / 1.434 / 400.47 / 0.03 | 0.1646 / 0.870 / 0.00 / 0.00 |
| plain_collision | 18 | 0.0000 / - / - / 0.07 | 0.1126 / - / - / 0.50 | 0.1168 / - / - / 0.06 | 0.1170 / - / - / 0.05 | 0.2919 / - / - / 0.00 |
| plain_desk | 20 | 0.0000 / - / - / 0.04 | 0.0255 / - / - / 0.19 | 0.0204 / - / - / 0.00 | 0.0204 / - / - / 0.00 | 0.1050 / - / - / 2.56 |
| plain_drop | 21 | 0.0000 / - / - / 0.16 | 0.1273 / - / - / 0.64 | 0.1312 / - / - / 0.04 | 0.1294 / - / - / 0.07 | 0.2604 / - / - / 0.00 |
| plain_edge_fall | 17 | 0.0000 / - / - / 0.19 | 0.0478 / - / - / 0.31 | 0.0499 / - / - / 0.00 | 0.0499 / - / - / 0.00 | 0.2747 / - / - / 0.00 |
| plain_gentle_push | 43 | 0.0000 / - / - / 0.03 | 0.0059 / - / - / 0.21 | 0.0054 / - / - / 0.00 | 0.0054 / - / - / 0.00 | 0.0267 / - / - / 0.00 |
| plain_legacy_layout | 32 | 0.0000 / - / - / 0.09 | 0.0901 / - / - / 0.51 | 0.1204 / - / - / 0.02 | 0.1201 / - / - / 0.02 | 0.2682 / - / - / 5.49 |
| plain_noclip | 6 | 0.0000 / - / - / 0.18 | 0.0945 / - / - / 0.45 | 0.0778 / - / - / 0.03 | 0.0778 / - / - / 0.03 | 0.8102 / - / - / 0.00 |
| plain_rest | 44 | 0.0000 / - / - / 0.10 | 0.0010 / - / - / 0.31 | 0.0003 / - / - / 0.00 | 0.0003 / - / - / 0.00 | 0.0003 / - / - / 0.00 |
| plain_slide_stop | 27 | 0.0000 / - / - / 0.04 | 0.0115 / - / - / 0.19 | 0.0185 / - / - / 0.02 | 0.0185 / - / - / 0.02 | 0.0555 / - / - / 0.00 |
| plain_stack_collapse | 17 | 0.0000 / - / - / 0.16 | 0.1159 / - / - / 0.56 | 0.1092 / - / - / 0.10 | 0.1104 / - / - / 0.12 | 0.3268 / - / - / 0.80 |
| plain_tilt_slide | 10 | 0.0000 / - / - / 0.21 | 0.3816 / - / - / 0.75 | 0.3893 / - / - / 0.10 | 0.3948 / - / - / 0.12 | 1.0909 / - / - / 12.20 |
| plain_topple | 15 | 0.0000 / - / - / 0.06 | 0.0393 / - / - / 0.34 | 0.0426 / - / - / 0.06 | 0.0426 / - / - / 0.06 | 0.2004 / - / - / 0.34 |
| plain_toss | 11 | 0.0000 / - / - / 0.13 | 0.0392 / - / - / 0.26 | 0.0483 / - / - / 0.06 | 0.0483 / - / - / 0.06 | 0.3700 / - / - / 2.80 |
| ragdoll | 270 | 0.0000 / 0.000 / 0.00 / 0.06 | 0.2195 / 0.407 / 1.41 / 0.68 | 0.2102 / 0.460 / 0.00 / 0.78 | 0.4259 / 1.052 / 377.54 / 0.03 | 0.7410 / 0.557 / 0.00 / 0.02 |
| robot_arm | 240 | 0.0000 / 0.000 / 0.00 / 0.04 | 0.0306 / 0.244 / 1.21 / 0.47 | 0.0341 / 0.291 / 0.00 / 0.76 | 0.1422 / 0.827 / 164.43 / 0.01 | 0.1591 / 0.736 / 0.00 / 0.00 |

## Box-only regression check (`data/v2` split `test`, 4998 episodes, zero joints)

| engine | subset | pos 1s (m) | pos 2s | pos 4s | ori 2s (deg) | fell acc | fell P | fell R | pen mean (mm) | pen >5mm (%) | rest drift (mm/s) | diverged |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| physics_reference | overall | 0.0000 | 0.0000 | 0.0000 | 0.00 | 1.000 | 1.000 | 1.000 | 0.14 | 0.29 | 0.24 | 0 |
| physics_reference | gentle | 0.0000 | 0.0000 | 0.0000 | 0.00 | 1.000 | - | - | 0.08 | 0.08 | 0.18 | 0 |
| physics_reference | chaotic | 0.0000 | 0.0000 | 0.0000 | 0.00 | 1.000 | 1.000 | 1.000 | 0.21 | 0.51 | 0.31 | 0 |
| joint_raw | overall | 0.0420 | 0.0572 | 0.0612 | 13.30 | 0.980 | 0.924 | 0.937 | 0.38 | 0.85 | 5.82 | 0 |
| joint_raw | gentle | 0.0021 | 0.0025 | 0.0034 | 0.66 | 1.000 | 0.000 | - | 0.23 | 0.08 | 0.68 | 0 |
| joint_raw | chaotic | 0.0822 | 0.1125 | 0.1196 | 26.07 | 0.959 | 0.925 | 0.937 | 0.53 | 1.62 | 11.73 | 0 |
| joint_projected | overall | 0.0444 | 0.0605 | 0.0641 | 13.80 | 0.978 | 0.916 | 0.939 | 0.05 | 0.09 | 5.49 | 0 |
| joint_projected | gentle | 0.0025 | 0.0028 | 0.0029 | 0.55 | 0.999 | 0.000 | - | 0.02 | 0.03 | 0.31 | 0 |
| joint_projected | chaotic | 0.0867 | 0.1187 | 0.1258 | 27.17 | 0.957 | 0.918 | 0.939 | 0.07 | 0.15 | 11.45 | 0 |
| contact_projected | overall | 0.0444 | 0.0605 | 0.0641 | 13.80 | 0.979 | 0.917 | 0.938 | 0.05 | 0.09 | 5.48 | 0 |
| contact_projected | gentle | 0.0025 | 0.0027 | 0.0029 | 0.56 | 0.999 | 0.000 | - | 0.02 | 0.03 | 0.30 | 0 |
| contact_projected | chaotic | 0.0867 | 0.1188 | 0.1259 | 27.18 | 0.957 | 0.919 | 0.938 | 0.07 | 0.16 | 11.42 | 0 |
| contact_raw | overall | 0.0420 | 0.0572 | 0.0612 | 13.30 | 0.980 | 0.924 | 0.937 | 0.38 | 0.85 | 5.82 | 0 |
| contact_raw | gentle | 0.0021 | 0.0025 | 0.0034 | 0.66 | 1.000 | 0.000 | - | 0.23 | 0.08 | 0.68 | 0 |
| contact_raw | chaotic | 0.0822 | 0.1125 | 0.1196 | 26.07 | 0.959 | 0.925 | 0.937 | 0.53 | 1.62 | 11.73 | 0 |

Paired bootstrap, JointGNN (zero joints) minus contact_gnn (negative = JointGNN better):

| pair | metric | contact_gnn | joint_gnn | delta | 95% CI |
|---|---|---|---|---|---|
| projected | pos_err_2s_m | 0.06049 | 0.06047 | -0.00002 | [-0.00047, 0.00039] |
| projected | chaotic_pos_err_2s_m | 0.11878 | 0.11871 | -0.00007 | [-0.00093, 0.00080] |
| projected | gentle_pos_err_2s_m | 0.00272 | 0.00275 | 0.00003 | [-0.00000, 0.00008] |
| projected | mean_penetration_m | 0.00005 | 0.00005 | 0.00000 | [-0.00000, 0.00000] |
| projected | rest_drift_mps | 0.00548 | 0.00549 | 0.00002 | [-0.00008, 0.00012] |
| raw | pos_err_2s_m | 0.05723 | 0.05724 | 0.00000 | [0.00000, 0.00001] |
| raw | chaotic_pos_err_2s_m | 0.11247 | 0.11247 | 0.00001 | [-0.00000, 0.00001] |
| raw | gentle_pos_err_2s_m | 0.00251 | 0.00251 | 0.00000 | [-0.00000, 0.00000] |
| raw | mean_penetration_m | 0.00038 | 0.00038 | -0.00000 | [-0.00000, 0.00000] |
| raw | rest_drift_mps | 0.00582 | 0.00582 | 0.00000 | [-0.00000, 0.00000] |

One-step |JointGNN(zero joints) - ContactGNN| on 256 box states: max 1.192e-06, mean 2.162e-09 (exactly zero is only expected for a warm start that has not been trained further).

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
