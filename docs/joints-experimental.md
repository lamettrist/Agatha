# Articulated bodies (joints) - EXPERIMENTAL

> **EXPERIMENTAL.** This is an extension track, not the shipped model. The shipped world model is the box
> model `models/contact_gnn.pt`; nothing here changes it. The joint model is **not wired into the app or the
> API** (`agatha/main.py`, `agatha/world_model.py` and `static/` do not reference it): it exists as a
> checkpoint, an offline evaluation and tests. Synthetic PyBullet data only.

All numbers below were re-measured by an independent review run on 2026-11-15 from the saved checkpoint
(`models/joint_gnn.pt`, sha256 `83c83fd3...`, identical to `models/runs/joint_v1/best.pt`, epoch 27). They
match the trainer's own report (`models/joint_eval.md`) to 3 significant figures; the residual differences
are GPU float noise.

## What it is

`agatha/joint_gnn.py`: `JointGNN` = the shipped `ContactGNN` plus joint edges (hinge, ball, prismatic,
fixed) between box links, and a non-learned projection `project_joints` (contact projection + joint solver:
anchors, axes, limits with `LIMIT_SLOP` = 0.05 rad / 5 mm).

- 1,548,300 parameters. The 711,692 inherited from `contact_gnn.pt` are **frozen** and verified bit-identical
  to the shipped checkpoint; 836,608 new ones (joint encoder, 2 extra gated rounds) were trained.
- With zero joints the model reproduces `contact_gnn` (one-step max difference 1.9e-6 on GPU).

## Data

- `data/joints_v1` (PyBullet, `scripts/generate_joint_data.py`, seed 4041): 15,948 train / 599 val / 1,996
  test episodes, 4 s each (80 steps of 0.05 s), up to 12 box links and 12 joints. Kinds: laptop, box_lid,
  door, drawer, pendulum, chain, robot_arm, ragdoll, and about 15% jointless `plain_*` box scenes.
- Training used `data/joints_v2` = joints_v1 train/val plus `joints_v1b` (4,179 extra laptop / robot_arm /
  ragdoll train episodes, seed 5051); 20,127 train episodes. 5% box windows from `data/v3` were mixed in.
- Test set = `data/joints_v1/test`. Checked: no test episode shares its initial state with any train or val
  episode (0 of 1,996). It is still the **same generator** as training: in-distribution only.

## Training

One run, one seed. 4 GPUs (DDP), 75.6 minutes wall clock (budget stop in epoch 37 of 60; 7,170 updates),
fp32, no skipped updates. Best validation checkpoint is epoch 27; validation was flat from about epoch 21.

## Measured results

`data/joints_v1` test, 1,996 episodes (1,300 chaotic), 5,271 joints. Every engine is rolled out
**autoregressively for 4 s from the same true t=0 state** of the same episodes; nothing is teacher-forced.

- `joint_raw`: JointGNN alone. `joint_projected`: JointGNN + `project_joints` each step.
- `contact_projected`: the shipped box model on the same bodies, joints ignored.
- `hold_initial`: trivial baseline, nothing moves. `physics_reference`: PyBullet itself (metric floor).

| engine | pos 0.5s (m) | pos 1s | pos 2s | pos 4s | pos 2s articulated | hinge 2s (rad) | gap mean / max (mm) | limit viol strict / > slop (%) | pen mean (mm) | fell acc | rest drift (mm/s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| physics_reference | 0 | 0 | 0 | 0 | 0 | 0 | 0.00 / 0.0 | 1.95 / 1.29 | 0.05 | 1.000 | 0.32 |
| joint_raw | 0.041 | 0.089 | 0.116 | 0.118 | 0.154 | 0.343 | 1.37 / 86.2 | 0.93 / 0.24 | 0.43 | 0.964 | 9.1 |
| joint_projected | 0.042 | 0.090 | 0.114 | 0.116 | 0.149 | 0.391 | 0.00 / 0.0 | 4.87 / 0.79 | 0.51 | 0.964 | 8.2 |
| contact_projected | 0.135 | 0.216 | 0.255 | 0.256 | 0.371 | 0.941 | 304 / 5096 | 22.4 / 21.4 | 0.02 | 0.880 | 10.7 |
| hold_initial | 0.254 | 0.372 | 0.409 | 0.410 | 0.556 | 0.623 | 0.00 / 0.0 | 0 / 0 | 0.17 | 0.767 | 0.0 |

The means are dominated by a heavy tail. Distribution at 2 s (same rollouts):

| engine | hinge err median (rad) | hinges within 0.1 rad | within 0.3 rad | off by > 1 rad | episode pos err median (m) | episodes < 5 cm | episodes > 30 cm |
|---|---|---|---|---|---|---|---|
| joint_raw | 0.074 | 54% | 72% | 11.5% | 0.024 | 63% | 7.4% |
| joint_projected | 0.084 | 53% | 68% | 13.2% | 0.025 | 62% | 7.4% |
| contact_projected | 0.575 | 21% | 39% | 39.5% | 0.127 | 31% | 27.6% |
| hold_initial | 0.401 | 22% | 44% | 23.8% | 0.156 | 25% | 34.1% |

On the 1,715 episodes that have joints, `joint_projected` has lower position error at 2 s than
`contact_projected` in 89% of episodes and than `hold_initial` in 92%.

Per kind, all articulated kinds (position error 2 s in m / hinge error 2 s in rad); jointless `plain_*`
kinds (6-44 episodes each) are in `models/joint_eval.md` and match the box model to within 0.006 m:

| kind | n | joint_raw | joint_projected | contact_projected | hold_initial | fell acc (joint_projected) |
|---|---|---|---|---|---|---|
| pendulum | 253 | 0.035 / 0.124 | 0.036 / 0.125 | 0.304 / 1.434 | 0.165 / 0.870 | 0.991 |
| robot_arm | 240 | 0.031 / 0.244 | 0.034 / 0.291 | 0.142 / 0.827 | 0.159 / 0.736 | 0.990 |
| laptop | 238 | 0.068 / 0.138 | 0.071 / 0.173 | 0.112 / 0.629 | 0.288 / 0.580 | 0.975 |
| box_lid | 134 | 0.075 / 0.243 | 0.071 / 0.323 | 0.131 / 0.719 | 0.297 / 0.643 | 0.977 |
| door | 153 | 0.071 / 0.518 | 0.071 / 0.509 | 0.132 / 0.893 | 0.147 / 0.946 | 0.971 |
| drawer (prismatic) | 162 | 0.083 / - | 0.081 / - | 0.172 / - | 0.284 / - | 0.983 |
| chain | 265 | 0.125 / 0.405 | 0.126 / 0.469 | 0.317 / 0.766 | 0.461 / 0.528 | 0.948 |
| ragdoll | 270 | 0.220 / 0.407 | 0.210 / 0.460 | 0.426 / 1.052 | 0.741 / 0.557 | 0.942 |

### Box regression (the shipped task)

`data/v2` test, 4,998 jointless episodes, JointGNN with zero joints against the shipped
`models/contact_gnn.pt`:

| engine | pos 2s (m) | chaotic pos 2s | pos 4s | fell acc |
|---|---|---|---|---|
| contact_raw (shipped, no projection) | 0.05723 | 0.11247 | 0.06122 | 0.9795 |
| joint_raw | 0.05724 | 0.11247 | 0.06123 | 0.9795 |
| contact_projected (shipped) | 0.06053 | 0.11885 | 0.06410 | 0.9786 |
| joint_projected | 0.06054 | 0.11886 | 0.06420 | 0.9786 |

Relative change in position error at 2 s: +0.01% (projected, paired bootstrap 95% CI
[-0.0004, +0.0004] m) and +0.007% (raw). **No measurable box regression**; far inside a 3% tolerance. This is
by construction (frozen trunk, gated joint rounds), not luck.

### Three rollouts, joint angle true vs predicted

Chosen by a fixed rule (first test episode by index that matches the description), not by result.
Angles in rad at 0 / 0.5 / 1 / 2 / 4 s.

- **Laptop lid falling shut** (episode 14, lid hinge): PyBullet 0.926 / 1.088 / 0.000 / 0.000 / 0.000;
  joint_raw 0.926 / 0.956 / 0.067 / 0.003 / -0.004; joint_projected 0.926 / 1.029 / 0.019 / -0.050 / -0.044.
  The lid closes at the right time. **But** in the same rollout the model then slides the closed laptop off
  the table (both laptop bodies end on the floor; in PyBullet they stay on the table): body position error
  0.38 m at 2 s, a false "fell".
- **Robot arm falling** (episode 7, 4 hinges): joint_raw follows PyBullet on all four hinges within 0.06 rad
  at every horizon (e.g. joint 1: 1.304 -> 2.415 true, 2.442 predicted; joint 3: 2.605 -> 1.931 true, 1.886
  predicted). joint_projected is correct at 0.5 s and then **keeps creeping after PyBullet has come to
  rest** (joint 3: 2.04 / 2.09 / 2.39 / 2.79 against a constant 1.93).
- **Ragdoll off the table** (episode 3, 11 bodies, 9 joints): the gross motion is right (the same 10 bodies
  end on the floor, the same one stays up; body position error 0.18-0.22 m at 2 s against 0.52 m without
  joints and 1.28 m for hold_initial). The individual limb angles do **not** follow PyBullet (mean hinge
  error 0.32 rad raw / 0.56 projected at 2 s; hold_initial 0.35).

## Limits (read before quoting any number)

- Mean hinge error at 2 s is 0.34-0.39 rad (about 20 deg). The median is 0.07-0.08 rad, but 11-13% of hinges
  are wrong by more than 1 rad. Door, chain and ragdoll are near 0.5 rad mean. For ragdolls the limb angles
  are not better than "nothing moves"; only the body positions are.
- Chaotic scenes: 0.15 m mean position error at 2 s. Plausible motion, not precise prediction. 7% of
  episodes are off by more than 30 cm, including false falls like the laptop above.
- The projection removes the anchor gap (1.4 mm mean, 86 mm worst case raw; 0 projected) but makes hinge
  angles slightly worse (0.391 vs 0.343 rad) and can make resting articulations creep. Raw is the more
  accurate engine; projected is the one that never comes apart.
- Rest drift is 8-9 mm/s (PyBullet 0.3). Penetration on articulated scenes is 0.5 mm mean with 1.9% of
  frames deeper than 5 mm (box model: 0.02 mm / 0.05%).
- Strict limit violations (> 0.02 rad) are 4.9% of joint-frames for joint_projected against 1.95% for
  PyBullet, because the projection only clamps beyond its 0.05 rad slop; beyond the slop it is 0.8%.
- One run, one seed, no ablations. Test data comes from the same synthetic generator as training (box links
  only, eight articulation templates). No real-world or out-of-distribution articulated data was evaluated.
- Latency at batch 1 on a shared CPU: about 8 ms/step raw, 20 ms/step projected (indicative only).

## Round 2 (2026-11-15): diagnosis of the round-1 error tail, more data, training changes

Round-1 files are kept as `models/joint_gnn_r1.pt`, `models/joint_eval_r1.json`, `models/joint_eval_r1.md`.
Everything in this section was measured on `models/joint_gnn_r1.pt`; **round-2 results are not in this section**
(see the top of `models/joint_eval.md` once the round-2 run has been evaluated on the same test set).

### Diagnosis (raw model, `data/joints_v1`; scratch script, numbers reproducible from the checkpoint)

- **Not a generalisation gap.** 4 s rollouts on 2,000 TRAIN episodes are as wrong as on test: position error at 2 s
  0.096 m train / 0.095 val / 0.116 test; hinge error at 2 s 0.337 / 0.365 / 0.343 rad; hinges off by > 1 rad 11.4% /
  12.8% / 11.5%; episodes off by > 30 cm 6.0% / 5.5% / 7.4%. The model is under-fitted / chaos-limited, not
  data-starved: more data of the same mix cannot close a gap that is not there. Round 2 therefore also adds capacity
  (two more joint-only rounds) and re-weights the data toward the failures.
- **Where the > 30 cm episodes are** (147 of 1,996): ragdoll 56 (20.7% of ragdolls), chain 29 (10.9%), laptop 15,
  box_lid 11, drawer 10, door 8; 93-100% are chaotic episodes. The first > 10 cm error appears early (median
  0.40 s, 75% by 0.55 s, i.e. inside the 8-step training window). In 95 of 147 the worst body is an articulated
  body that is not in contact itself 0.2 s before (falling / swinging link). False falls and missed falls are
  balanced (bodies on the floor at 4 s: 2,265 true, 2,354 predicted; 216 false, 127 missed; 39 / 31 of the 147
  episodes): edge decisions go wrong in both directions, there is no one-sided "falls off the table" bias.
- **Hinges off by > 1 rad** (439 of 3,814): limited 11.6%, unlimited 11.0%; only 151 have the true angle at a limit
  and 39 sit on the opposite stop. 66-96% of them (by kind) are in episodes that are NOT among the > 30 cm ones: a limb or
  lid settled on the other side. Limits are not the dominant cause.
- **The frozen contact trunk is not the cause of rest drift.** Teacher-forced one-step velocity predicted for
  bodies truly at rest: free boxes 1.2 mm/s; articulated bodies 2.7 mm/s (0.014 rad/s) with the joint rounds, and
  12.7 mm/s with the box model alone (joints ignored). The joint rounds already remove most of the trunk's error;
  thin links (< 13 mm) are not worse (2.4 mm/s). What is missing is supervision: the dataset `rest` flag requires
  a contact, so a lid on its stop or an arm link held by its joints was never asked to stand still.
- **Projection.** On TRUE transitions `solve_joints` without limits is the identity, the limit clamp changes 0.4% of
  hinge-frames by > 0.02 rad (Bullet's soft limits) and the contact projection edits velocities by 0.8 mm/s. On
  rollouts of 600 jointed test episodes, hinge error at 2 s is 0.340 rad raw and 0.352-0.372 for every projection
  variant tried (limits off, no articulation sleep, slop 0 / 5 mm, joints only, no snap, translation-only gap
  closing, no velocity pass, 1-2 mm dead band); hinges > 1 rad 10.3% raw, 10.4-12.5% projected. No single component
  is responsible: every per-step edit of a chaotic rollout tried flips up to 2% more hinges (600 episodes, about
  1,150 hinges: differences between variants are within noise). Resting-hinge creep (hinges
  truly still from 2 s; change between 3 s and 4 s) is the same raw (0.0141 rad) and projected (0.0142): the creep
  is the model's, not the projection's. `gap_turn=False` (new option) lowers it to 0.0105 rad but worsens the
  median hinge error (0.079 -> 0.095 rad), so the default projection is unchanged.

### Data added

`data/joints_v1c` (seed 6061, 28 workers; composition in its `meta.json`): no jointless scenes (the trainer drops
them), weighted toward the measured failures (ragdoll 22%, chain 16%, door 8%, laptop 8%, box_lid 7%, drawer 6%,
robot_arm 5%, pendulum 3%) plus 25% new "calm" kinds that mostly stay on the table: `lid_calm` (laptops / lidded
boxes resting, nudged 0.05-0.6 m/s, at an edge with the centre of mass on the table, lids on a stop or closing
slowly), `ragdoll_calm` (sitting / lying in the middle, standing and barely pushed, let go a few cm above the
table), `chain_calm` (lying, slightly draped, folded and let go: link-link contacts), `arm_calm` (stiff or damped
arms starting on a limit or near rest). The default `KIND_WEIGHTS` never draw the new kinds: `data/joints_v1` is
reproduced bit-for-bit (checked on 12 episodes). `data/joints_v3` = symlinks: train = v1 + v1b + v1c, val = all
three, test = `data/joints_v1/test` only (unchanged).

### Code changes (all defaults reproduce round 1)

- `scripts/train_joint_ddp.py`: `--joint-rest 1` (rest / slide penalties also for articulated bodies that are truly
  slow on both frames of a step without a contact), `--grow-rounds N` (with `--init`: append N identity-initialised
  rounds; they are gated to scenes with a joint, so joint-free scenes keep the box computation exactly).
- `agatha/joint_gnn.py`: `solve_joints` / `project_joints` options `gap_turn`, `velocity`, `gap_free` (ablation
  switches, defaults unchanged).
- `scripts/generate_joint_data.py`: the four calm kinds and the `nudge` / `edge_safe` staging modes.

### Round-2 training command (warm start from round 1; result not yet in this document)

```
CUDA_VISIBLE_DEVICES=0,1,2,3 .venv-train/bin/torchrun --nproc_per_node=4 scripts/train_joint_ddp.py \
    --data data/joints_v3 --init models/joint_gnn_r1.pt --grow-rounds 2 --joint-rest 1 \
    --teacher models/runs/joint_v1/init_contact.pt --mix-box-data data/v3 --mix-box-frac 0.05 \
    --inherit-lr-scale 0 --drop-jointless-train 1 --box-guard 0.01 --w-move 3 --w-impulse 3 --w-speed 1 \
    --event-frac 0.3 --w-jcoord 1 --w-jrel 0.5 --w-jmove 3 --w-rest 3 --w-slide 2 --w-pen 3 --w-anchor 4 \
    --batch 704 --epochs 40 --windows-per-epoch 600000 --start-horizon 8 --lr 5e-4 --warmup 300 --amp off \
    --val-rollout-episodes 599 --max-gpu-gb 22 --out models/runs/joint_v2 --max-minutes 100
```

The contact trunk stays frozen (the diagnosis does not implicate it, and frozen + gated is what guarantees box
parity). Note: `models/contact_gnn.pt` was replaced by another experiment on 2026-11-15 11:10 UTC; the joint model
descends from the PREVIOUS box checkpoint, pinned as `models/runs/joint_v1/init_contact.pt`. Box parity of the
joint model is by construction against that file, so pass `--contact models/runs/joint_v1/init_contact.pt` to
`scripts/eval_joints.py`; against the new `models/contact_gnn.pt` the zero-joint outputs differ.

## How to run

```
# tests (CPU)
.venv/bin/python -m pytest tests/test_joint_gnn.py tests/test_joint_data.py -q

# evaluation (about 2 minutes on one GPU); writes <out>.json and <out>.md
CUDA_VISIBLE_DEVICES=0 .venv-train/bin/python scripts/eval_joints.py --model models/joint_gnn.pt \
    --contact models/contact_gnn.pt --data data/joints_v1 --split test --box-data data/v2 \
    --out models/joint_eval.json

# data
.venv/bin/python scripts/generate_joint_data.py --out data/joints_v1 \
    --train-episodes 16000 --val-episodes 600 --test-episodes 2000 --workers 40

# training as run (4 GPUs, 75 minutes)
CUDA_VISIBLE_DEVICES=0,1,2,3 .venv-train/bin/torchrun --nproc_per_node=4 scripts/train_joint_ddp.py \
    --data data/joints_v2 --mix-box-data data/v3 --mix-box-frac 0.05 --gate-extra-rounds 1 \
    --inherit-lr-scale 0 --drop-jointless-train 1 --box-guard 0.03 --w-move 3 --w-impulse 3 --w-speed 1 \
    --event-frac 0.3 --w-jcoord 1 --w-jrel 0.5 --w-jmove 3 --batch 1024 --epochs 60 \
    --windows-per-epoch 800000 --amp off --out models/runs/joint_v1 --max-minutes 75
```

In Python: `model = joint_gnn.build_from_checkpoint(torch.load('models/joint_gnn.pt'))`, then
`next = model(state, props, context, mask, joints, joint_mask)` and optionally
`joint_gnn.project_joints(state, next, props, context, mask, joints, joint_mask)`. Array formats are in
`data/joints_v1/meta.json`.

## Round 2 (2026-11-15, stopped early at 45 of 100 minutes)

More data and a partial unfreeze: about 45,000 extra PyBullet episodes aimed at the round-1 failures (`data/joints_v3`), warm start from round 1, two extra message-passing rounds, box scenes still distilled against the box model. Training was stopped at epoch 12 of 40 to free the GPUs; the checkpoint below is `models/joint_gnn.pt` (round 1 kept as `models/joint_gnn_r1.pt`). Same fixed test set (`data/joints_v1` test, 1,996 episodes). Single evaluation run, not independently re-reviewed.

| Model only (no projection) | Round 1 | Round 2 |
|---|---|---|
| Position error at 2 s, all bodies | 0.116 m | 0.108 m |
| Position error at 2 s, jointed bodies | 0.154 m | 0.141 m |
| Hinge angle error at 2 s | 0.343 rad | 0.338 rad |
| Mean joint gap | 1.37 mm | 0.95 mm |
| Mean penetration | 0.43 mm | 0.34 mm |
| Rest drift | 9.2 mm/s | 6.9 mm/s |
| Fell accuracy | 0.964 | 0.964 |

With projection: position error at 2 s 0.114 to 0.105 m, hinge error 0.391 to 0.358 rad, rest drift 8.2 to 5.9 mm/s. Box-only scenes are unchanged against the box model the trunk came from (position error at 2 s 0.0603 vs 0.0607 m projected, identical without projection).

What did not improve: hinge-angle error barely moved, so the error tail (hinges off by more than 1 rad, false falls, ragdoll limb angles) was not re-measured and should be assumed still present. Full tables: `models/joint_eval.md`.
