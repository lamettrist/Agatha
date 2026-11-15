# Soft bodies and cloth (EXPERIMENTAL)

**Status: EXPERIMENTAL research prototype. Not part of the shipped box model (`models/contact_gnn.pt`),
not wired into the default app path, one training run, one seed.** Numbers below were measured on
2026-11-15 and re-measured by an independent review run from the saved checkpoint (same values to
within 0.1 mm).

## What it is

`agatha/soft_gnn.py` (`SoftGNN`) is a learned GNS-style simulator for ONE deformable body (a soft cube,
a soft ball or a cloth sheet, at most 128 vertices) on a table / floor, optionally next to rigid boxes.
Given the vertex positions of the last two frames it predicts the next frame (0.05 s step) and is rolled
out autoregressively. `project_soft` is an optional non-learned step that pushes vertices out of the
table, the floor and the boxes.

**It is not a coupled soft-rigid simulator.** Rigid boxes in a soft scene are read from the ground-truth
trajectory at every step (teacher-forced obstacles), in training and in every evaluation number below.
Only the soft vertices are predicted; nothing pushes back on the boxes.

## Data

- `data/soft_v1` + `data/soft_v1b` (union `data/soft_v2`): PyBullet soft-body episodes from
  `scripts/generate_soft_data.py`, 61 frames of 0.05 s. 8,524 training / 369 validation episodes;
  the benchmark is the 752-episode `data/soft_v1/test` split (verified: no episode is shared between
  test, validation and training).
- `data/external_v2/meshgraphnets_soft`: 80 / 10 / 10 trajectories converted from the DeepMind
  MeshGraphNets `sphere_simple` cloth set, mixed in at 5% of each training batch. **This data has no
  explicit licence**; it is used by team decision under the repository's Apache-2.0 licence for this
  non-released prototype only (`DATA_LICENSES.md`). Revisit before any public release of
  `models/soft_gnn.pt`.

## Model and training

- 1,141,507 parameters (width 128, 6 message-passing rounds); mesh edges plus contact edges to boxes,
  table and floor.
- One run: 4 GPUs, 75.5 minutes (wall-clock budget; stopped in epoch 17 of 30), about 17,500 updates of
  384 windows, fp16, 11 skipped updates (0.06%), no diverged rollouts. Validation error flattened from
  epoch 10 on (52 -> 49 mm at 2 s). Checkpoint = epoch 16, selected on the validation split.
- `models/soft_gnn.pt`, sha256 `638df09e...`.

## Measured results (`data/soft_v1` test, 752 episodes)

Every engine is rolled out autoregressively for the whole episode from the same two true frames (0, 1).
Vertex error = mean distance between predicted and true vertices. Baselines: `hold_initial` (the body
does not move) and `rigid_freefall` (the undeformed shape falls ballistically and stops on the table / floor).

| engine | vert 0.5 s (mm) | 1 s | 2 s | last frame (3 s) | shape 2 s (mm) | vertices > 5 mm inside a surface (%) | deepest (mm) | strain err | CPU ms/step |
|---|---|---|---|---|---|---|---|---|---|
| hold_initial | 235.2 | 286.4 | 300.1 | 308.2 | 16.0 | 0.16 | 57.3 | 0.0094 | 0.0 |
| rigid_freefall | 146.0 | 170.2 | 172.3 | 177.4 | 16.0 | 3.17 | 78.7 | 0.0094 | 0.3 |
| soft_raw (model only) | 24.6 | 38.4 | 47.2 | 55.1 | 9.9 | 0.09 | 51.8 | 0.0121 | 13-14 |
| soft_projected (model + projection) | 24.5 | 38.4 | 47.4 | 54.9 | 9.9 | 0.00 | 2.2 | 0.0125 | 17-19 |

All nine scenario kinds, vertex error at 2 s in mm. "diff" = model minus the better baseline, paired over
episodes, with a 95% bootstrap interval; "wins" = share of episodes where the model beats that baseline.

| kind (episodes) | hold_initial | rigid_freefall | soft_raw | soft_projected | diff vs better baseline [95%] | wins | verdict |
|---|---|---|---|---|---|---|---|
| soft_rest (50) | 7.1 | 7.1 | 6.8 | 6.8 | -0.4 [-1.2, +0.2] | 36% | **tie** |
| box_on_soft (123) | 41.9 | 41.9 | 42.3 | 41.9 | +0.3 [-6.1, +8.2] | 59% | **tie** |
| soft_drop (176) | 239.8 | 51.9 | 24.6 | 24.3 | -27.3 [-39.3, -18.6] | 74% | better |
| soft_launch (126) | 470.6 | 400.9 | 44.2 | 44.0 | -356.7 [-431.7, -289.1] | 95% | better |
| soft_edge_fall (77) | 877.1 | 487.6 | 90.9 | 91.4 | -396.7 [-516.5, -282.0] | 88% | better |
| soft_on_boxes (86) | 378.5 | 183.6 | 110.0 | 112.5 | -73.6 [-102.6, -47.4] | 83% | better, but 110 mm is large |
| cloth_drop (32) | 215.6 | 96.9 | 11.4 | 12.5 | -85.6 [-150.9, -45.2] | 97% | better |
| cloth_drape_box (50) | 127.2 | 91.2 | 24.8 | 24.8 | -66.4 [-79.2, -54.4] | 98% | better |
| cloth_drape_edge (32) | 166.9 | 106.8 | 62.4 | 62.4 | -44.4 [-131.5, +32.5] | 91% | better in most episodes; mean not significant |

MeshGraphNets `sphere_simple` test (10 trajectories, weak evidence): soft_raw 9.1 mm at 2 s
(19.0 mm at the last frame) against 269.6 mm for hold_initial and 726.8 mm for rigid_freefall.

Full tables (every metric, shape subsets, the `soft_projected_margin` and `soft_projected_sleep`
variants): `models/soft_eval.md`, `models/soft_eval_meshgraphnets.md`.

## Limits (read before quoting any number)

- **Resting and loaded bodies: no gain.** On soft_rest and box_on_soft the model only ties "do nothing".
  In the typical resting episode it drifts slightly MORE than holding still (median 0.8 mm against
  0.5 mm at 2 s; it is ahead in only 36% of soft_rest episodes); the equal means come from the few
  episodes where the true body moves.
- **Edge lengths are worse than not deforming at all.** Strain error 0.0121 against 0.0094 for a rigidly
  held shape; the model's mean absolute strain is 0.0149 where the ground truth has 0.0105, i.e. it
  over-stretches the mesh by about 40%. (Shape error after removing the centre of mass is better: 9.9
  against 16.0 mm.)
- **Error grows with time and never settles:** 24.6 mm at 0.5 s, 47.2 at 2 s, 55.1 at 3 s.
- **Large absolute errors remain** on soft_on_boxes (110 mm) and soft_edge_fall (91 mm) at 2 s, and a
  quarter of the soft_drop episodes are still worse than rigid free-fall.
- **cloth_drape_edge depends on the contact margin** (62.4 mm with margin 0, 31.6 mm with a 5 mm margin)
  and reaches 100.7 mm at the last frame.
- **Rigid boxes are teacher-forced** (see above), so box_on_soft / soft_on_boxes numbers are with the
  true box motion given.
- **Same generator for training and test**; nothing here measures generalisation to other meshes,
  materials or simulators. The MeshGraphNets result is 10 trajectories from the same upstream file as
  its 80 training trajectories, with placeholder material properties and an assumed gravity scale.
- **One run, one seed.** Kinds with 32-50 episodes have wide intervals.
- **Latency** 13-14 ms/step (model) and 17-19 ms/step (with projection): CPU, batch 1, 8 threads, one
  64-vertex episode, measured on a loaded machine. Not measured: fp32 against fp16 training, more
  training time.
- **MeshGraphNets data licence is unresolved** (see Data).

## How to run

```
# evaluate the saved checkpoint (about 1 minute on one GPU; --device cpu also works)
CUDA_VISIBLE_DEVICES=4 .venv-train/bin/python scripts/eval_soft.py --data data/soft_v1 --split test \
    --ckpt models/soft_gnn.pt --out models/soft_eval.json --example docs/soft-example.json --max-gpu-gb 3
CUDA_VISIBLE_DEVICES=4 .venv-train/bin/python scripts/eval_soft.py --data data/external_v2/meshgraphnets_soft \
    --split test --ckpt models/soft_gnn.pt --out models/soft_eval_meshgraphnets.json --example '' --max-gpu-gb 3

# the training run that produced the checkpoint (4 GPUs, 75 minutes)
CUDA_VISIBLE_DEVICES=4,5,6,7 .venv-train/bin/torchrun --nproc_per_node=4 scripts/train_soft_ddp.py \
    --data data/soft_v2 --extra-data data/external_v2/meshgraphnets_soft --extra-frac 0.05 --batch 96 \
    --epochs 30 --windows-per-epoch 400000 --out models/runs/soft_v1 --max-minutes 75

# tests (CPU, 15 s)
.venv/bin/python -m pytest -q tests/test_soft_gnn.py tests/test_soft_data.py tests/test_external_soft.py
```

`docs/soft-example.json` (284 kB) is one test rollout for a viewer: episode 206, a soft ball on boxes,
chosen as the MEDIAN last-frame error of `soft_projected` (20.4 mm), with true and predicted vertices,
faces and the prescribed box poses.
