# RBO articulated objects: kinematic consistency of the joint rows (APPROXIMATE, hand-driven real data)

APPROXIMATE conversion of REAL, HAND-DRIVEN mocap recordings (RBO articulated objects, CC BY 4.0). External forces are not observed: valid for kinematic-consistency checks only, NOT for scoring free dynamics.

Recordings converted: 10 (33 windows of 40 steps at 20 Hz); skipped: 1.

| recording | joint | type | frames | anchor gap mm (mean / p95 / max) | off-axis deg (mean / p95 / max) | coordinate range | vs dataset joint_states (mean / max) | vs NEGATED (mean) |
|---|---|---|---|---|---|---|---|---|
| cabinet03 | j_0_1 | prismatic | 824 | 2.27 / 3.17 / 4.03 | 0.42 / 1.25 / 1.95 | -0.0092 .. 0.0365 m | 0.000 / 0.000 mm | 11.74 mm |
| cabinet03 | j_0_2 | prismatic | 824 | 1.49 / 2.26 / 4.00 | 0.60 / 0.83 / 1.84 | -0.0024 .. 0.0458 m | 0.000 / 0.000 mm | 14.88 mm |
| ikea08 | j_0_1 | prismatic | 270 | 2.77 / 4.83 / 16.67 | 2.03 / 3.11 / 3.88 | -0.0089 .. 0.2155 m | 0.000 / 0.000 mm | 150.23 mm |
| ikea08 | j_0_2 | hinge | 270 | 2.86 / 4.63 / 19.44 | 0.19 / 0.48 / 0.89 | 0.0224 .. 0.0994 rad | 0.310 / 3.519 deg | 5.12 deg |
| ikea09 | j_0_1 | prismatic | 294 | 2.41 / 3.27 / 4.53 | 2.15 / 2.73 / 4.40 | -0.0147 .. 0.0023 m | 0.000 / 0.000 mm | 12.63 mm |
| ikea09 | j_0_2 | hinge | 294 | 7.61 / 17.25 / 18.11 | 0.54 / 1.70 / 2.36 | -1.3111 .. 0.0656 rad | 1.287 / 3.466 deg | 47.20 deg |
| laptop04 | j_0_1 | hinge | 643 | 7.32 / 14.74 / 16.72 | 2.20 / 3.96 / 5.03 | 0.0331 .. 1.9965 rad | 2.080 / 6.210 deg | 105.87 deg |
| laptop07 | j_0_1 | hinge | 415 | 7.34 / 14.13 / 17.44 | 2.61 / 4.86 / 5.11 | 0.0418 .. 1.7401 rad | 2.510 / 5.916 deg | 101.65 deg |
| laptop13 | j_0_1 | hinge | 453 | 17.73 / 32.94 / 34.11 | 4.61 / 8.14 / 8.66 | 0.1297 .. 1.9453 rad | 4.530 / 12.352 deg | 104.29 deg |
| microwave04 | j_0_1 | hinge | 1800 | 11.00 / 20.93 / 39.48 | 0.52 / 0.86 / 1.17 | 0.0595 .. 1.1425 rad | 1.421 / 5.410 deg | 68.23 deg |
| treasurebox07 | j_0_1 | hinge | 685 | 2.80 / 5.90 / 7.50 | 0.87 / 1.47 / 1.62 | 0.0552 .. 1.5202 rad | 1.829 / 5.509 deg | 52.00 deg |
| treasurebox08 | j_0_1 | hinge | 796 | 3.10 / 7.62 / 9.31 | 1.25 / 2.02 / 2.36 | -0.0238 .. 1.2535 rad | 2.073 / 6.855 deg | 33.18 deg |
| treasurebox26 | j_0_1 | hinge | 1337 | 5.66 / 9.73 / 12.38 | 3.03 / 4.95 / 6.56 | -0.0271 .. 1.604 rad | 4.043 / 8.923 deg | 100.09 deg |

## Shards (float32, 20 Hz, as an evaluator reads them)

{"joint_frames": 1599, "hinge_frames": 1189, "prismatic_frames": 410, "anchor_gap_mm": {"mean": 6.2338, "median": 4.1781, "p95": 16.6286, "max": 33.5731}, "anchor_gap_mm_hinge": {"mean": 7.6636, "median": 5.9245, "p95": 18.2233, "max": 33.5731}, "anchor_gap_mm_prismatic": {"mean": 2.0874, "median": 1.9333, "p95": 3.522, "max": 16.648}, "off_axis_deg_hinge": {"mean": 1.7066, "median": 1.1436, "p95": 4.7229, "max": 8.5274}, "off_axis_deg_prismatic": {"mean": 0.8103, "median": 0.5768, "p95": 2.3792, "max": 3.6554}, "limit_excess_max": 0.0, "joint_pos_vs_joint_gnn_max": 3.5762786865234375e-07}

## Free-motion segments

Kinematic candidates (hand presence unknown): 23; verified free of contact (usable): **0**; recordings with F/T data: 2.

| recording | joint | pattern | start s | duration s | peak speed | travel | max F/T deviation N | max stick speed m/s | verified |
|---|---|---|---|---|---|---|---|---|---|
| cabinet03 | 0 (prismatic) | decelerating_tail | 2.8 | 0.37 | 0.0962 | 0.0237 | None | None | False |
| ikea08 | 0 (prismatic) | decelerating_tail | 0.75 | 0.61 | 0.456 | 0.1712 | None | None | False |
| ikea09 | 1 (hinge) | accelerating_abrupt_stop | 0.33 | 0.79 | 2.4488 | 1.1611 | None | None | False |
| laptop04 | 0 (hinge) | decelerating_tail | 4.37 | 0.93 | 2.4748 | 1.4106 | None | None | False |
| laptop07 | 0 (hinge) | decelerating_tail | 2.55 | 0.41 | 2.3135 | 0.5599 | None | None | False |
| laptop13 | 0 (hinge) | decelerating_tail | 1.73 | 1.1 | 1.9281 | 1.1226 | None | None | False |
| microwave04 | 0 (hinge) | accelerating_abrupt_stop | 3.11 | 1.21 | 0.4942 | 0.1992 | 1.717 | 0.085 | False |
| microwave04 | 0 (hinge) | decelerating_tail | 10.44 | 0.45 | 0.4461 | 0.1153 | 2.806 | 0.082 | False |
| microwave04 | 0 (hinge) | decelerating_tail | 11.79 | 0.74 | 0.3542 | 0.1578 | 2.693 | 0.096 | False |
| microwave04 | 0 (hinge) | accelerating_abrupt_stop | 12.7 | 0.37 | 0.3563 | 0.0824 | 2.357 | 0.082 | False |
| microwave04 | 0 (hinge) | decelerating_tail | 15.641 | 0.6 | 0.3474 | 0.12 | 0.454 | 0.075 | False |
| microwave04 | 0 (hinge) | accelerating_abrupt_stop | 16.261 | 0.46 | 0.7159 | 0.1134 | 0.502 | 0.072 | False |
| treasurebox07 | 0 (hinge) | decelerating_tail | 2.74 | 1.04 | 1.9703 | 0.8543 | None | None | False |
| treasurebox07 | 0 (hinge) | decelerating_tail | 4.92 | 0.53 | 1.8055 | 0.423 | None | None | False |
| treasurebox08 | 0 (hinge) | decelerating_tail | 4.23 | 0.88 | 1.4285 | 0.7208 | None | None | False |
| treasurebox08 | 0 (hinge) | decelerating_tail | 6.15 | 0.3 | 2.2806 | 0.4901 | None | None | False |
| treasurebox26 | 0 (hinge) | decelerating_tail | 0.8 | 0.42 | 0.8244 | 0.2336 | 8.458 | 0.071 | False |
| treasurebox26 | 0 (hinge) | decelerating_tail | 1.7 | 0.66 | 0.8901 | 0.2462 | 6.772 | 0.06 | False |
| treasurebox26 | 0 (hinge) | decelerating_tail | 3.47 | 1.16 | 1.5283 | 0.7266 | 16.764 | 0.082 | False |
| treasurebox26 | 0 (hinge) | decelerating_tail | 5.38 | 0.6 | 1.8186 | 0.5327 | 11.844 | 0.051 | False |
| treasurebox26 | 0 (hinge) | decelerating_tail | 6.52 | 1.02 | 2.1709 | 1.1881 | 14.474 | 0.082 | False |
| treasurebox26 | 0 (hinge) | decelerating_tail | 8.52 | 0.46 | 0.4681 | 0.1273 | 8.54 | 0.018 | False |
| treasurebox26 | 0 (hinge) | decelerating_tail | 11.99 | 0.3 | 0.9221 | 0.1592 | 3.423 | 0.051 | False |

## Skipped

- microwave22: ValueError: kinematic model configuration_2017-08-04 does not fit this recording (median anchor gap 754 mm, median axis error 178 deg)
