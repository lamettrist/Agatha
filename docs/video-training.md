# Training on video (experimental)

Honest framing, in one paragraph: the box world model is **state based** (3D box poses). "Using video for
training" therefore means: convert video to **approximate object trajectories** (segmentation + tracking +
geometry) and mix those in. The trajectories are much noisier than simulator states, carry **no orientation**,
and cover only the frames in which an object could be located. They are meant to be mixed in with a **small
weight** and a **position-only, noise-tolerant loss**, into a **separate experimental checkpoint**, and the result
is kept only if the simulator metrics do not get worse. The shipped `contact_gnn` is not trained on any of this.

Outcome in one line: three fine-tune arms were run and evaluated (sections 4-5); **none met the keep rule, no
video-trained checkpoint was kept, and the shipped model is unchanged.** Sections 1-3 describe the pipeline, the
data and its measured accuracy; an independent review pass (section 6) lists what was re-checked and corrected.

## 1. Pipeline: video -> trajectories (`scripts/video_to_trajectories.py`)

```
CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/video_to_trajectories.py \
    --sources kubric clevrer deskspace --out data/video_v1 --workers 12        # ~10 min, CPU only
CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest tests/test_video_trajectories.py -q
```

| Step | What is done |
|---|---|
| Background | Per-pixel median of one frame from each of 150 (Kubric) / 80 (CLEVRER) videos. Pixels only. |
| Segmentation | Colour segmentation against the 8-colour CLEVR palette (both datasets use it). A pixel is "coloured" if its chromaticity differs from the background; it gets the palette colour with the most similar hue, restricted to the colours in the video's attribute list. Gray = low-chroma pixels much darker / brighter than the background. |
| Tracking | Connected components per colour; a colour with k objects keeps its k largest blobs, linked by nearest neighbour. |
| Lifting, Kubric | Known camera (pose + focal length per episode). Ray through the mask centroid intersected with the plane at the object's half height, i.e. the object is **assumed to sit on the floor**. A frame counts as observed only if the blob's apparent size matches the size class at that depth (ratio within 0.90-1.06 of a constant that is self-calibrated from the last frames of all videos), for at least 4 consecutive frames. Airborne objects look too big for the floor hit point and are rejected; occluded ones look too small. |
| Lifting, CLEVRER | The camera is fixed across videos: ONE pixel-to-ground homography (planar motion). |
| Episode | Unobserved frames hold / linearly interpolate the neighbouring observed positions and are flagged `obs_mask = False`. 20 Hz resampling exactly as `scripts/prepare_external_data.py`. Velocity = least-squares slope over +-2 frames. Orientation = identity, `has_orientation = False` everywhere, angular velocity 0. Ground plane -> support slab top at 0.75 m, Y-up metres. |

What does **not** come from pixels (stated so nobody over-reads "trained on video"):

- **The static attribute list of each video** (colour, shape, size class, material) is read from the dataset
  metadata. It says which objects to look for and which box / mass / friction they get. Shape is not recognised
  from pixels. No annotated position, velocity or mask is read for a converted video.
- **Kubric camera** pose and focal length come from the dataset.
- **Calibration videos (in no split).** CLEVRER: the annotated positions of the last 20 videos (11580-11599) fit
  the homography (1,097 point pairs, 86 % RANSAC inliers, median residual 0.8 cm). Kubric: the size-ratio gate
  and the run length were chosen by looking at the error against ground truth on a trial run over the first 200
  episodes. Only the first 40 of those are kept out of every split; of the other 160, the 112 with at most 8
  objects are in the splits (95 train, 7 val, **10 of the 91 test videos**). So two thresholds and one run length
  were chosen with sight of the ground truth of 10 test videos. This affects which Kubric frames count as observed
  (the Kubric rows of section 3 and the video-derived Kubric test rows of section 5); it does not touch the
  ground-truth Kubric held-out evaluation, whose states come from the dataset.
- **Pixels of all videos, including val / test** are used without labels for the background median and, for
  Kubric, the apparent-size constant per shape (median over the late frames of every video).
- **Physics parameters** are those of the ground-truth converter (Kubric K1/K3: length x0.36, time x0.6, gravity
  10, ground friction 0.3 / restitution 0.5, object friction / restitution from the material label; CLEVRER
  C1/C3: 25 fps, length x0.5, every object a 0.2 m box of 0.5 kg, friction 0.133). Kubric sizes are the size
  class (0.252 / 0.504 m), masses are density x volume of the size class x 0.36; these agree with the metadata
  masses to within 0.2-0.6 %. So the only difference to `data/external_v2/<dataset>/` is that states come from
  pixels.

Objects that are deliberately dropped (they are then absent from `mask`, and `n_untracked` counts them, so the
scene is incomplete in those episodes):

- a colour group whose members cannot be told apart from pixels (same colour, different shape, same size);
  different sizes are separated by apparent size. Kubric: 1,983 objects, CLEVRER: 891 objects;
- **gray objects in Kubric** (gray on a gray floor; in the first trial their p90 error was 1.7 m): 893 objects;
- Kubric episodes with more than 8 objects (the shard format has 8 slots): 281 of 1,137 episodes.

Spheres and cylinders are **kept** as tracked obstacles with box stand-ins (as in the converter, A1);
`box_like` marks the cubes, which were the intended training targets. **The trainer as run does not filter on
`box_like`**: its video loss covers every tracked dynamic object, so 72 % of the supervised Kubric object-frames
(10,509 of 14,540) and 71 % of the CLEVRER ones (58,227 of 81,715) are spheres and cylinders treated as boxes
(a rolling sphere is supervised as a sliding box), and the desk-clip bottle is included too.

## 2. Data produced (`data/video_v1/`, git-ignored, 4 MB)

Layout: `data/video_v1/<source>/{train,val,test}/shard_*.npz`, `test_gt/`, `meta.json`. Keys are those of
`data/external_v2` (a superset of `data/v2`), N = 8 slots, T+1 = 81 frames (`n_frames` = valid length; Kubric
episodes have 24 frames = 1.15 s), plus:

| Key | Meaning |
|---|---|
| `obs_mask` bool [E,81,8] | object located in that frame. Positions elsewhere are held / interpolated: do not put a loss on them. |
| `pos_sigma` float32 [E,8] | position noise estimate in metres: 68th percentile horizontal error of that source and shape on the calibration videos. A per-source constant, not a per-track estimate. |
| `has_orientation` bool [E,8] | False everywhere (quaternions are identity). |
| `source` str [E], `video_derived` bool [E] | `kubric_movi_a_video`, `clevrer_video`, `team_clip_video`; True. |
| `n_untracked` int [E] | objects of the video missing from `mask`. |

| Source | Videos used | Train episodes | Train cube tracks | Train cube transitions (observed at t and t+1) | Train episodes with every object tracked | Val / test episodes |
|---|---|---|---|---|---|---|
| Kubric MOVi-A | 856 of 1,137 (281 have > 8 objects); 31 of them calibration (the first 40 episodes minus those with > 8 objects) | 644 | 415 | 3,579 | 34 | 90 / 91 |
| CLEVRER | 600 validation videos 11000-11599, 20 of them calibration | 464 | 439 | 22,905 | 134 | 58 / 58 |
| Team clip `deskspace.mp4` | 1 | 1 (rest only) | 0 | 0 | 1 | 0 / 0 |

Splits are by video. Kubric: `sha1(source_id)` 80 / 10 / 10. CLEVRER: by video index, 80 / 10 / 10.

Leakage:

- `test_gt/` holds the **dataset ground truth** (converter output, 8 slots) of exactly the held-out test videos,
  slot-aligned with `test/`. Use it for "trained on video-derived train, evaluated on dataset ground truth".
- `data/external_v2/kubric_movi_a/test` contains **all** 1,137 Kubric episodes, including these training videos.
  After training on the Kubric video split, evaluate on `data/video_v1/kubric_movi_a/test_gt` or on the
  `test_source_ids` listed in `meta.json`, not on the whole external set.
- `data/external_v2/clevrer/test` is built from annotations 10000-10999. All CLEVRER videos used here are
  11000-11599, so that whole set stays held out.

Team clip, stated exactly: `scripts/track_video.py` runs on `tests/images/deskspace.mp4` (re-run for this work as
a check: it works, reports 4 bottle tracks and warns that the camera moves by 54 % of the image width; the episode
itself is built from the script's earlier output in `data/external_v2/track_video_test/deskspace/`). It is a hand-held
walk-around of a static desk. **No object moves, so there is no dynamic trajectory.** What was written is one
80-step "at rest stays at rest" episode with one object (a bottle, the only track present in the first frames,
at its first-frame table-plane position), detector-prior size and mass, an assumed 1.2 m table and an assumed
`pos_sigma` of 0.10 m. `box_like` is False, so a cubes-only trainer would ignore it; the trainer as run is not cubes-only, so this one
episode (81 object-frames out of about 96,000) was in the video training set. No accuracy can be measured.
This is the only real-world video in the set.

## 3. Measured accuracy of the pipeline (the noise floor of "training on video")

3D position error of video-derived tracks against dataset ground truth, on observed object-frames, in
centimetres of the rescaled Agatha scene. Cubes only, except the last column.

| Source, split | Videos | Observed cube object-frames | Median | p90 | Mean | Rest-position error, median / p90 (n) | Recall, all GT cube object-frames | Recall, cubes visible in the image | Cubes tracked at all | Identity switches (tracks in same-colour groups) | All shapes, median / p90 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Kubric, train | 644 | 4,031 | 2.3 | 16.0 | 6.1 | 1.2 / 2.5 (97) | 14.5 % | 15.5 % | 415 of 1,157 | 0 (95) | 2.4 / 13.8 |
| Kubric, val | 90 | 577 | 2.6 | 17.5 | 7.1 | 0.9 / 1.6 (12) | 14.0 % | 14.8 % | 63 of 172 | 0 (11) | 2.3 / 13.7 |
| Kubric, test | 91 | 586 | 2.5 | 13.3 | 5.3 | 1.2 / 2.1 (8) | 15.2 % | 16.1 % | 55 of 161 | 0 (13) | 2.3 / 12.2 |
| CLEVRER, train | 464 | 23,488 | 0.6 | 2.6 | 1.5 | 0.6 / 2.0 (93) | 43.2 % | 50.4 % | 439 of 672 | 15 (132) | 0.7 / 2.7 |
| CLEVRER, val | 58 | 3,197 | 0.6 | 2.0 | 1.3 | 0.6 / 1.7 (12) | 43.9 % | 49.1 % | 61 of 90 | 0 (21) | 0.7 / 3.4 |
| CLEVRER, test | 58 | 2,779 | 0.6 | 3.0 | 1.4 | 0.6 / 2.1 (13) | 36.1 % | 40.4 % | 57 of 95 | 0 (10) | 0.6 / 2.4 |
| CLEVRER, calibration (20 videos) | 20 | 737 | 1.1 | 11.1 | 4.2 | 0.9 / 1.2 (2) | 31.4 % | 32.5 % | 12 of 29 | 2 (6) | 0.8 / 8.9 |

Definitions: recall = observed object-frames / ground-truth object-frames (second recall column: only frames
where the dataset says the object is in view: Kubric visibility > 80 px, CLEVRER `inside_camera_view`). Rest
error = mean of the last 3 frames against the converter's `rest_pos`, for objects that are settled in the ground
truth and observed in the last 3 frames. Identity switch = the nearest ground-truth object of the same colour
group changes along a track. Objects of one colour with identical attributes are interchangeable and are matched
to the ground truth by the lowest-error permutation (evaluation only).

How to read it:

- **CLEVRER is clean.** Median 0.6 cm on 20 cm objects, p90 2-3 cm, about half of the visible cube frames.
  Planar sliding and collisions with a fixed camera are the easy case. The physics in it is also the least
  informative for the demo (no toppling, no falls, friction 0.018 effective).
- **Kubric is sparse and has a heavy tail.** Median 2.4 cm (objects are 25 / 50 cm), but p90 13-18 cm and a mean
  twice the median: some airborne or half-occluded frames pass the size gate. Only 15 % of cube object-frames
  survive, because objects are thrown and are airborne or tumbling for most of the 1.15 s; what survives is
  mostly the late, on-floor, slow phase. The tosses and bounces themselves, which are the interesting part of
  this dataset, are exactly what the pipeline discards. Before the gate was tightened (0.80-1.18), the first
  200 episodes gave median 7 cm / p90 41 cm at 24 % recall.
- For comparison, the shipped model's own 2 s position error on chaotic simulator scenes is 12 cm. Kubric video
  targets have a p90 error of the same size, so they can only be a weak signal; CLEVRER targets are well below it.
- `pos_sigma` stored in the shards: Kubric cube 2.9 cm, cylinder 5.0 cm, sphere 2.7 cm; CLEVRER cube 5.5 cm,
  cylinder 1.2 cm, sphere 1.2 cm. The CLEVRER cube value comes from only 12 tracked cubes in the 20 calibration
  videos, which contain a few bad tracks; it overstates the noise seen on the train / val / test splits (p68
  there is about 1 cm). It is left as measured rather than tuned on the splits.

## Limitations

- Orientation is never estimated. Tipping and tumbling cannot be learned from this data, and velocities come
  from smoothed finite differences of noisy positions.
- Kubric heights are an assumption (object on the floor), not a measurement; the vertical velocity is zero by
  construction.
- Incomplete scenes: only 34 of 644 Kubric and 134 of 464 CLEVRER training episodes have every object tracked.
  In the others an untracked object can still push the tracked ones, which the model would see as unexplained
  motion.
- Attribute lists and the Kubric camera come from dataset metadata; only the motion comes from pixels.
- Both dataset sources are **rendered simulator video**, not real footage. The only real video is the team
  desk clip, which contributes one rest episode with no cube in it. No additional openly licensed real video of
  falling / sliding boxes was searched for in this time budget (the optional Exa search was not run); the
  I.AM. reference videos are CC BY-NC-SA and were not used.
- Kubric's data licence is the repository's Apache-2.0 only (no dedicated data licence statement); CLEVRER is
  explicit CC0. See `DATA_LICENSES.md`.
- The CLEVRER homography and all `pos_sigma` values come from calibration videos that are in no split. The Kubric
  gate thresholds were chosen on a 200-episode trial that included 10 test and 7 val videos (section 1); nothing
  else was tuned on val / test, and the pipeline was not optimised either (no shape recognition, no occlusion reasoning beyond a box-overlap flag).

## 4. Fine-tune

APPROXIMATE, experimental. Nothing here changed the shipped model (`models/contact_gnn.pt` = `models/world_model.pt`,
sha256 `e67e8871...`, unchanged before and after).

Three arms, each a single process on its own GPU (shared with other trainings, 9 GB cap), warm-started from the
shipped `models/contact_gnn.pt`, same seed, same recipe, 34 minutes wall-clock each (10:05-10:40 UTC):

```
COMMON="scripts/train_contact_ddp.py --data data/v3 --init models/contact_gnn.pt --start-horizon 8 --horizon 8 --lr 1e-4 \
  --w-move 3 --w-impulse 3 --w-speed 1 --event-frac 0.3 --score-fall-weight 0.3 --val-rollout-episodes 1000 \
  --batch 1536 --windows-per-epoch 600000 --epochs 30 --max-minutes 34 --load-threads 6"
VID="--video-data data/video_v1/kubric_movi_a data/video_v1/clevrer data/video_v1/team_deskspace --video-weight 0.2"
# capped.py = 6-line launcher that calls torch.cuda.set_per_process_memory_fraction (GPU_CAP_GB=9) and then runs the script
CUDA_VISIBLE_DEVICES=7 .venv-train/bin/python capped.py $COMMON --out models/runs/video_ft_control                          # A control (no video)
CUDA_VISIBLE_DEVICES=6 .venv-train/bin/python capped.py $COMMON $VID --video-frac 0.1  --out models/runs/video_ft_video    # B video 10%
CUDA_VISIBLE_DEVICES=5 .venv-train/bin/python capped.py $COMMON $VID --video-frac 0.25 --out models/runs/video_ft_video25  # C video 25%
```

The control arm separates "34 more minutes of simulator fine-tuning at lr 1e-4" from "video".

| Arm | Video windows per update (of 1536) | Updates done in 34 min | `best.pt` = epoch (sim val score) | Last epoch |
|---|---|---|---|---|
| control | 0 | 5,320 | 13 (0.1500) | 14 |
| video (frac 0.1) | 154 | 3,950 | 1 (0.1493) | 11 |
| video25 (frac 0.25) | 384 | 4,100 | 2 (0.1507) | 11 |

- The arms are matched by wall-clock, not by updates: the video arms are slower per update, so the control arm got
  about 30% more updates (5,320 vs 3,950 / 4,100), and each video-arm update also has fewer simulator windows
  (1,382 / 1,152 instead of 1,536): in total the control saw 8.2 M simulator windows, the video arms 5.5 M and
  4.7 M. An update-matched comparison (epoch 10 = 3,900 updates in every arm) is in section 6.
- `best.pt` is chosen by the **simulator** validation score only (1,000 val episodes). That score moved between
  0.149 and 0.160 from epoch to epoch in all three arms with no trend, so which epoch is "best" is mostly noise.
  The video arms' `best.pt` are from epoch 1 (`video`) and epoch 2 (`video25`), i.e. after 390 and 780 updates: they contain
  very little video training. The last-epoch checkpoints are therefore reported too (supplementary; not part
  of the decision rule).
- Video training set as loaded by the trainer: 1,109 episodes (Kubric 644, CLEVRER 464, desk clip 1), 15.5% of
  frames are valid window starts, median `pos_sigma` 2.7 cm, no orientation anywhere.
- Video validation (video-derived val split, video-derived targets, 1,173 windows, never scored), arm B:
  position error after 8 steps 2.04 cm at epoch 1 -> 1.86 cm at epoch 11; at 1 s 7.97 cm -> 6.43 cm ("hold at the
  start position" reference: 9.79 cm and 24.48 cm). Arm C: the same within 0.02 cm. Training video error 1.74 ->
  1.46 cm. So the model does fit the video-derived tracks a little better; the gain flattens after about 6 epochs.

## 5. Results

All evaluations ran after training, on the same episodes for every arm. Commands:

```
# sim ship gate (per arm; shipped = models/contact_gnn.pt)
CUDA_VISIBLE_DEVICES=7 .venv-train/bin/python scripts/eval_contact.py --data data/v2 --split test --new <ckpt> \
    --out models/runs/video_ft_sim_<arm>.json --device cuda:0 --no-latency --no-old-projected
# external ground truth, standard report per arm (CPU)
.venv-train/bin/python scripts/eval_external.py --new <ckpt> --no-physics --threads 8 \
    --datasets clevrer kubric_movi_a physion real_iam_box_toss real_contactnets_cube --out models/runs/video_ft_external_<arm>.json
# all arms on the same episodes with paired bootstraps, Kubric restricted to the held-out videos, and the video-derived test split
CUDA_VISIBLE_DEVICES=7 .venv-train/bin/python scripts/eval_video_ft.py --arm shipped=models/contact_gnn.pt \
    --arm control=models/runs/video_ft_control/best.pt --arm video=models/runs/video_ft_video/best.pt \
    --arm video25=models/runs/video_ft_video25/best.pt --arm control_last=models/runs/video_ft_control/epoch_014.pt \
    --arm video_last=models/runs/video_ft_video/epoch_011.pt --arm video25_last=models/runs/video_ft_video25/epoch_011.pt \
    --out models/runs/video_ft_compare.json --device cuda:0
.venv/bin/python scripts/video_ft_report.py shipped control video video25 > models/runs/video_ft_report.md
```

Reading notes:

- `models/runs/video_ft_external_<arm>.json` evaluates Kubric on all 1,137 episodes, most of which overlap the
  video-derived training set. The Kubric rows below use only the 91 held-out episodes (`kubric_movi_a_heldout`,
  full ground-truth scenes). CLEVRER (videos 10000-10999) is disjoint from the training videos (11000-11599) but
  comes from the same generator, so it measures in-domain generalisation, not transfer.
- "slab" keeps the dataset's ground as a large support slab; "floor" maps it to Agatha's floor (see
  `scripts/eval_external.py`). Both are reported; neither was picked after seeing the results.
- Rest position errors on CLEVRER / Kubric are tens of centimetres for every arm (objects travel metres in 4 s
  and the converted scenes are outside the training distribution). These are approximate conversions, not a
  validation of the product.
- The sim evaluation is not bit-reproducible on GPU: two runs of the shipped checkpoint in this session gave
  chaotic 0.1214 / 0.1209 m and gentle 0.00294 / 0.00297 m. Differences of about 0.5-1% are within that noise.
  The gentle error is about 3 mm, so the 2% tolerance is 0.06 mm.
- I.AM. and ContactNets are REAL box-toss tracks and were never trained on.

#### Sim ship gate (data/v2 test, live_physics, 2468 episodes)

| Arm | Chaotic pos err @2s proj / raw (m) | Gentle pos err @2s proj / raw (m) | Penetration proj / raw (mm) | Rest drift proj (m/s) | Fall precision / recall (proj) | Chaotic ori err @2s proj (deg) |
|---|---|---|---|---|---|---|
| shipped | 0.1214 / 0.1141 | 0.00294 / 0.00240 | 0.056 / 0.392 | 0.0055 | 0.915 / 0.943 | 28.5 |
| control | 0.1237 / 0.1147 | 0.00251 / 0.00287 | 0.055 / 0.415 | 0.0057 | 0.912 / 0.935 | 28.6 |
| video | 0.1222 / 0.1156 | 0.00274 / 0.00326 | 0.054 / 0.391 | 0.0058 | 0.911 / 0.940 | 28.9 |
| video25 | 0.1216 / 0.1160 | 0.00252 / 0.00294 | 0.057 / 0.393 | 0.0057 | 0.913 / 0.940 | 28.7 |

#### External ground truth: rest (final-frame) position error, m, mean [95% episode bootstrap CI], model + projection

| Dataset | Mode | Subset | n episodes / objects | hold still | shipped | control | video | video25 |
|---|---|---|---|---|---|---|---|---|
| clevrer | slab | all_objects | 1000 / 4955 | 1.359 | 0.676 [0.662, 0.689] | 0.659 [0.646, 0.672] | 0.603 [0.590, 0.615] | 0.550 [0.537, 0.563] |
| clevrer | slab | box_like_objects | 832 / 1505 | 1.269 | 0.559 [0.538, 0.581] | 0.548 [0.527, 0.569] | 0.514 [0.494, 0.533] | 0.488 [0.467, 0.508] |
| clevrer | floor | all_objects | 1000 / 4955 | 1.359 | 1.022 [1.002, 1.040] | 1.021 [1.002, 1.040] | 1.009 [0.990, 1.027] | 1.010 [0.991, 1.028] |
| clevrer | floor | box_like_objects | 832 / 1505 | 1.269 | 0.918 [0.882, 0.951] | 0.919 [0.884, 0.952] | 0.902 [0.867, 0.934] | 0.902 [0.866, 0.934] |
| kubric_movi_a_heldout | slab | all_objects | 91 / 514 | 2.178 | 0.786 [0.724, 0.844] | 0.773 [0.715, 0.831] | 0.772 [0.713, 0.829] | 0.765 [0.703, 0.826] |
| kubric_movi_a_heldout | slab | box_like_objects | 80 / 161 | 2.136 | 0.680 [0.592, 0.772] | 0.665 [0.581, 0.756] | 0.659 [0.577, 0.747] | 0.669 [0.585, 0.753] |
| kubric_movi_a_heldout | floor | all_objects | 91 / 514 | 2.178 | 0.823 [0.771, 0.877] | 0.830 [0.775, 0.883] | 0.821 [0.766, 0.878] | 0.811 [0.758, 0.864] |
| kubric_movi_a_heldout | floor | box_like_objects | 80 / 161 | 2.136 | 0.694 [0.615, 0.778] | 0.694 [0.614, 0.782] | 0.707 [0.622, 0.790] | 0.686 [0.610, 0.765] |
| physion | slab | all_objects | 10 / 47 | 0.376 | 0.908 [0.071, 1.869] | 0.864 [0.071, 1.758] | 0.819 [0.066, 1.668] | 0.833 [0.079, 1.697] |
| physion | slab | box_like_objects | 10 / 43 | 0.251 | 0.550 [0.071, 1.340] | 0.511 [0.071, 1.226] | 0.397 [0.066, 0.918] | 0.429 [0.079, 0.998] |
| physion | floor | all_objects | 10 / 47 | 0.376 | 0.896 [0.070, 1.834] | 0.848 [0.061, 1.718] | 0.864 [0.081, 1.733] | 0.848 [0.063, 1.711] |
| physion | floor | box_like_objects | 10 / 43 | 0.251 | 0.529 [0.070, 1.264] | 0.486 [0.061, 1.159] | 0.452 [0.081, 1.018] | 0.406 [0.063, 0.899] |
| real_iam_box_toss | slab | all_objects | 49 / 49 | 1.028 | 0.515 [0.449, 0.588] | 0.528 [0.462, 0.602] | 0.520 [0.454, 0.594] | 0.636 [0.566, 0.713] |
| real_iam_box_toss | floor | all_objects | 49 / 49 | 1.028 | 0.288 [0.271, 0.306] | 0.296 [0.280, 0.314] | 0.299 [0.282, 0.318] | 0.302 [0.284, 0.320] |
| real_contactnets_cube | slab | all_objects | 570 / 570 | 0.400 | 0.082 [0.079, 0.085] | 0.078 [0.075, 0.081] | 0.082 [0.078, 0.085] | 0.076 [0.073, 0.079] |
| real_contactnets_cube | floor | all_objects | 570 / 570 | 0.400 | 0.110 [0.107, 0.113] | 0.106 [0.103, 0.109] | 0.109 [0.106, 0.112] | 0.108 [0.105, 0.110] |

Paired differences in rest position error (B - A, m; negative = B better; 95% paired episode bootstrap; `*` = CI excludes 0):

| Dataset | Mode | Subset | control - shipped | video - shipped | video25 - shipped | video - control | video25 - control | video25 - video |
|---|---|---|---|---|---|---|---|---|
| clevrer | slab | all_objects | -0.017 [-0.020, -0.014] * | -0.073 [-0.080, -0.067] * | -0.125 [-0.137, -0.114] * | -0.056 [-0.063, -0.050] * | -0.109 [-0.120, -0.097] * | -0.052 [-0.061, -0.044] * |
| clevrer | slab | box_like_objects | -0.012 [-0.016, -0.008] * | -0.046 [-0.057, -0.033] * | -0.071 [-0.090, -0.050] * | -0.034 [-0.046, -0.021] * | -0.060 [-0.078, -0.039] * | -0.026 [-0.039, -0.012] * |
| clevrer | floor | all_objects | -0.000 [-0.000, -0.000] * | -0.013 [-0.013, -0.012] * | -0.012 [-0.012, -0.011] * | -0.013 [-0.013, -0.012] * | -0.012 [-0.012, -0.011] * | +0.001 [+0.000, +0.001] * |
| clevrer | floor | box_like_objects | +0.001 [+0.001, +0.002] * | -0.016 [-0.017, -0.015] * | -0.016 [-0.017, -0.015] * | -0.017 [-0.019, -0.016] * | -0.018 [-0.019, -0.016] * | -0.000 [-0.001, +0.001] |
| kubric_movi_a_heldout | slab | all_objects | -0.013 [-0.033, +0.007] | -0.014 [-0.032, +0.004] | -0.021 [-0.042, -0.003] * | -0.001 [-0.021, +0.019] | -0.008 [-0.030, +0.012] | -0.007 [-0.026, +0.012] |
| kubric_movi_a_heldout | slab | box_like_objects | -0.014 [-0.046, +0.016] | -0.021 [-0.050, +0.007] | -0.011 [-0.047, +0.027] | -0.006 [-0.033, +0.020] | +0.003 [-0.032, +0.043] | +0.009 [-0.020, +0.039] |
| kubric_movi_a_heldout | floor | all_objects | +0.007 [-0.013, +0.026] | -0.003 [-0.024, +0.018] | -0.012 [-0.029, +0.006] | -0.009 [-0.033, +0.013] | -0.018 [-0.036, -0.002] * | -0.009 [-0.026, +0.007] |
| kubric_movi_a_heldout | floor | box_like_objects | -0.000 [-0.029, +0.028] | +0.013 [-0.020, +0.047] | -0.008 [-0.033, +0.016] | +0.013 [-0.018, +0.045] | -0.008 [-0.035, +0.019] | -0.021 [-0.054, +0.011] |
| physion | slab | all_objects | -0.044 [-0.118, +0.001] | -0.089 [-0.227, -0.004] * | -0.075 [-0.173, +0.013] | -0.046 [-0.110, -0.002] * | -0.031 [-0.081, +0.013] | +0.015 [-0.042, +0.073] |
| physion | slab | box_like_objects | -0.039 [-0.116, +0.002] | -0.152 [-0.425, -0.004] * | -0.121 [-0.343, +0.013] | -0.113 [-0.310, -0.002] * | -0.082 [-0.229, +0.013] | +0.031 [-0.005, +0.085] |
| physion | floor | all_objects | -0.048 [-0.108, +0.006] | -0.032 [-0.096, +0.023] | -0.048 [-0.144, +0.016] | +0.017 [+0.001, +0.035] * | +0.000 [-0.048, +0.055] | -0.016 [-0.060, +0.034] |
| physion | floor | box_like_objects | -0.043 [-0.119, +0.009] | -0.077 [-0.257, +0.027] | -0.124 [-0.388, +0.022] | -0.034 [-0.146, +0.034] | -0.080 [-0.272, +0.036] | -0.046 [-0.136, +0.021] |
| real_iam_box_toss | slab | all_objects | +0.013 [-0.001, +0.031] | +0.006 [-0.003, +0.013] | +0.121 [+0.088, +0.157] * | -0.008 [-0.026, +0.007] | +0.108 [+0.076, +0.143] * | +0.116 [+0.082, +0.154] * |
| real_iam_box_toss | floor | all_objects | +0.008 [+0.007, +0.009] * | +0.011 [+0.010, +0.012] * | +0.013 [+0.012, +0.015] * | +0.003 [+0.002, +0.004] * | +0.005 [+0.004, +0.007] * | +0.002 [+0.002, +0.003] * |
| real_contactnets_cube | slab | all_objects | -0.003 [-0.004, -0.002] * | -0.000 [-0.001, +0.001] | -0.006 [-0.007, -0.005] * | +0.003 [+0.002, +0.004] * | -0.002 [-0.003, -0.001] * | -0.006 [-0.007, -0.005] * |
| real_contactnets_cube | floor | all_objects | -0.004 [-0.005, -0.003] * | -0.001 [-0.002, -0.000] * | -0.003 [-0.004, -0.002] * | +0.003 [+0.002, +0.004] * | +0.001 [+0.000, +0.003] * | -0.002 [-0.003, -0.000] * |

#### Held-out video-derived test split: rollout from the video-derived start state, error vs dataset ground truth at the last frame, m

| Dataset | Mode | Subset | n episodes / objects | hold still | shipped | control | video | video25 |
|---|---|---|---|---|---|---|---|---|
| kubric_movi_a | slab | all_objects | 60 / 129 | 0.658 | 0.339 [0.272, 0.419] | 0.339 [0.271, 0.418] | 0.335 [0.268, 0.414] | 0.324 [0.259, 0.403] |
| kubric_movi_a | slab | box_like_objects | 30 / 38 | 0.542 | 0.236 [0.179, 0.297] | 0.235 [0.179, 0.296] | 0.232 [0.175, 0.291] | 0.224 [0.168, 0.281] |
| kubric_movi_a | floor | all_objects | 60 / 129 | 0.658 | 0.371 [0.298, 0.456] | 0.372 [0.299, 0.457] | 0.369 [0.297, 0.454] | 0.368 [0.296, 0.452] |
| kubric_movi_a | floor | box_like_objects | 30 / 38 | 0.542 | 0.261 [0.198, 0.327] | 0.261 [0.198, 0.327] | 0.259 [0.196, 0.325] | 0.257 [0.194, 0.323] |
| clevrer | slab | all_objects | 46 / 128 | 0.606 | 0.439 [0.359, 0.522] | 0.438 [0.358, 0.524] | 0.412 [0.331, 0.500] | 0.397 [0.309, 0.500] |
| clevrer | slab | box_like_objects | 28 / 38 | 0.509 | 0.304 [0.183, 0.426] | 0.309 [0.185, 0.435] | 0.271 [0.152, 0.414] | 0.261 [0.132, 0.437] |
| clevrer | floor | all_objects | 46 / 128 | 0.606 | 0.515 [0.429, 0.603] | 0.515 [0.429, 0.604] | 0.509 [0.423, 0.597] | 0.509 [0.424, 0.597] |
| clevrer | floor | box_like_objects | 28 / 38 | 0.509 | 0.405 [0.258, 0.557] | 0.405 [0.258, 0.558] | 0.397 [0.252, 0.546] | 0.398 [0.253, 0.548] |

Paired differences in rest position error (B - A, m; negative = B better; 95% paired episode bootstrap; `*` = CI excludes 0):

| Dataset | Mode | Subset | control - shipped | video - shipped | video25 - shipped | video - control | video25 - control | video25 - video |
|---|---|---|---|---|---|---|---|---|
| kubric_movi_a | slab | all_objects | -0.001 [-0.001, +0.000] | -0.005 [-0.006, -0.003] * | -0.016 [-0.020, -0.011] * | -0.004 [-0.006, -0.002] * | -0.015 [-0.019, -0.010] * | -0.011 [-0.014, -0.008] * |
| kubric_movi_a | slab | box_like_objects | -0.001 [-0.002, -0.000] * | -0.005 [-0.007, -0.002] * | -0.013 [-0.019, -0.006] * | -0.003 [-0.006, -0.001] * | -0.011 [-0.018, -0.005] * | -0.008 [-0.013, -0.004] * |
| kubric_movi_a | floor | all_objects | +0.001 [+0.000, +0.001] * | -0.002 [-0.003, -0.001] * | -0.003 [-0.004, -0.002] * | -0.003 [-0.004, -0.002] * | -0.004 [-0.005, -0.003] * | -0.001 [-0.002, -0.001] * |
| kubric_movi_a | floor | box_like_objects | +0.000 [-0.000, +0.001] | -0.002 [-0.003, -0.001] * | -0.004 [-0.005, -0.003] * | -0.002 [-0.003, -0.001] * | -0.004 [-0.006, -0.003] * | -0.002 [-0.003, -0.001] * |
| clevrer | slab | all_objects | -0.001 [-0.005, +0.004] | -0.028 [-0.056, +0.004] | -0.043 [-0.089, +0.012] | -0.027 [-0.054, +0.002] | -0.042 [-0.088, +0.011] | -0.015 [-0.039, +0.013] |
| clevrer | slab | box_like_objects | +0.005 [-0.005, +0.016] | -0.033 [-0.092, +0.034] | -0.044 [-0.134, +0.063] | -0.038 [-0.092, +0.022] | -0.048 [-0.131, +0.048] | -0.011 [-0.050, +0.033] |
| clevrer | floor | all_objects | +0.000 [-0.000, +0.000] | -0.006 [-0.008, -0.005] * | -0.006 [-0.008, -0.004] * | -0.007 [-0.008, -0.005] * | -0.006 [-0.008, -0.004] * | +0.000 [-0.000, +0.001] |
| clevrer | floor | box_like_objects | +0.000 [-0.000, +0.001] | -0.009 [-0.013, -0.004] * | -0.007 [-0.011, -0.003] * | -0.009 [-0.013, -0.004] * | -0.007 [-0.011, -0.003] * | +0.001 [+0.001, +0.002] * |

#### Real box-toss sets: more metrics (model + projection)

| Dataset | Mode | Arm | pos err 0.5 s (m) | pos err 1 s (m) | rest pos err (m) | final ori err (deg) | rest face accuracy |
|---|---|---|---|---|---|---|---|
| real_iam_box_toss | slab | hold_initial | 0.918 | 1.029 | 1.028 | 35.7 | 1.000 |
| real_iam_box_toss | slab | shipped | 0.108 | 0.378 | 0.515 | 72.4 | 0.551 |
| real_iam_box_toss | slab | control | 0.108 | 0.381 | 0.528 | 78.8 | 0.490 |
| real_iam_box_toss | slab | video | 0.108 | 0.380 | 0.520 | 77.2 | 0.510 |
| real_iam_box_toss | slab | video25 | 0.113 | 0.408 | 0.636 | 102.9 | 0.306 |
| real_iam_box_toss | floor | hold_initial | 0.918 | 1.029 | 1.028 | 35.7 | 1.000 |
| real_iam_box_toss | floor | shipped | 0.065 | 0.286 | 0.288 | 31.4 | 1.000 |
| real_iam_box_toss | floor | control | 0.068 | 0.294 | 0.296 | 31.1 | 1.000 |
| real_iam_box_toss | floor | video | 0.068 | 0.297 | 0.299 | 31.5 | 1.000 |
| real_iam_box_toss | floor | video25 | 0.067 | 0.299 | 0.302 | 31.5 | 1.000 |
| real_contactnets_cube | slab | hold_initial | 0.382 | 0.499 | 0.400 | 61.0 | 0.433 |
| real_contactnets_cube | slab | shipped | 0.064 | 0.103 | 0.082 | 30.6 | 0.812 |
| real_contactnets_cube | slab | control | 0.061 | 0.099 | 0.078 | 29.9 | 0.819 |
| real_contactnets_cube | slab | video | 0.064 | 0.106 | 0.082 | 30.2 | 0.816 |
| real_contactnets_cube | slab | video25 | 0.060 | 0.092 | 0.076 | 30.4 | 0.812 |
| real_contactnets_cube | floor | hold_initial | 0.382 | 0.499 | 0.400 | 61.0 | 0.433 |
| real_contactnets_cube | floor | shipped | 0.092 | 0.137 | 0.110 | 36.8 | 0.735 |
| real_contactnets_cube | floor | control | 0.088 | 0.130 | 0.106 | 36.0 | 0.740 |
| real_contactnets_cube | floor | video | 0.091 | 0.132 | 0.109 | 36.6 | 0.735 |
| real_contactnets_cube | floor | video25 | 0.089 | 0.133 | 0.108 | 36.3 | 0.742 |

#### Decision rule, applied literally

**Arm `video`**

| Check | vs shipped | vs control |
|---|---|---|
| sim chaotic pos err @2s (<= +2%) | +0.63% PASS | -1.20% PASS |
| sim gentle pos err @2s (<= +2%) | -6.70% PASS | +9.15% FAIL |
| penetration (<= +0.02 mm) | -0.0022 mm PASS | -0.0008 mm PASS |
| fall precision (>= -0.01) | -0.0041 PASS | -0.0012 PASS |
| fall recall (>= -0.01) | -0.0025 PASS | +0.0050 PASS |
| clevrer rest pos err improves, slab mode, all objects (paired CI below 0) | -0.073 [-0.080, -0.067] * PASS | -0.056 [-0.063, -0.050] * PASS |
| clevrer rest pos err improves, floor mode, all objects (paired CI below 0) | -0.013 [-0.013, -0.012] * PASS | -0.013 [-0.013, -0.012] * PASS |
| kubric_movi_a_heldout rest pos err improves, slab mode, all objects (paired CI below 0) | -0.014 [-0.032, +0.004] FAIL | -0.001 [-0.021, +0.019] FAIL |
| kubric_movi_a_heldout rest pos err improves, floor mode, all objects (paired CI below 0) | -0.003 [-0.024, +0.018] FAIL | -0.009 [-0.033, +0.013] FAIL |

Sim conditions: FAIL. External improvement vs both references (in at least one floor mode): clevrer yes, kubric_movi_a_heldout no; in both floor modes: clevrer yes, kubric_movi_a_heldout no. **Kept: NO**

**Arm `video25`**

| Check | vs shipped | vs control |
|---|---|---|
| sim chaotic pos err @2s (<= +2%) | +0.18% PASS | -1.64% PASS |
| sim gentle pos err @2s (<= +2%) | -14.22% PASS | +0.34% PASS |
| penetration (<= +0.02 mm) | +0.0008 mm PASS | +0.0022 mm PASS |
| fall precision (>= -0.01) | -0.0019 PASS | +0.0010 PASS |
| fall recall (>= -0.01) | -0.0025 PASS | +0.0050 PASS |
| clevrer rest pos err improves, slab mode, all objects (paired CI below 0) | -0.125 [-0.137, -0.114] * PASS | -0.109 [-0.120, -0.097] * PASS |
| clevrer rest pos err improves, floor mode, all objects (paired CI below 0) | -0.012 [-0.012, -0.011] * PASS | -0.012 [-0.012, -0.011] * PASS |
| kubric_movi_a_heldout rest pos err improves, slab mode, all objects (paired CI below 0) | -0.021 [-0.042, -0.003] * PASS | -0.008 [-0.030, +0.012] FAIL |
| kubric_movi_a_heldout rest pos err improves, floor mode, all objects (paired CI below 0) | -0.012 [-0.029, +0.006] FAIL | -0.018 [-0.036, -0.002] * PASS |

Sim conditions: all PASS. External improvement vs both references (in at least one floor mode): clevrer yes, kubric_movi_a_heldout no; in both floor modes: clevrer yes, kubric_movi_a_heldout no. **Kept: NO**


Supplementary, last-epoch checkpoints (full table: `models/runs/video_ft_report_with_last.md`, model + projection):

| Arm (last epoch) | Sim chaotic @2s (m) | Sim gentle @2s (m) | CLEVRER slab rest err (m) | Kubric held-out slab / floor rest err (m) | I.AM. slab / floor rest err (m) | ContactNets slab rest err (m) |
|---|---|---|---|---|---|---|
| shipped | 0.1214 | 0.00294 | 0.676 | 0.786 / 0.823 | 0.515 / 0.288 | 0.082 |
| control_last (epoch 14) | 0.1232 | 0.00252 | 0.658 | 0.776 / 0.830 | 0.521 / 0.294 | 0.079 |
| video_last (epoch 11) | 0.1223 | 0.00260 | 0.528 | 0.761 / 0.830 | 0.547 / 0.289 | 0.079 |
| video25_last (epoch 11) | 0.1223 | 0.00289 | 0.524 | 0.742 / 0.832 | 0.541 / 0.288 | 0.079 |

Against control_last, video25_last improves Kubric held-out in slab mode (-0.033 m, paired CI [-0.062, -0.004])
but not in floor mode (+0.002 [-0.016, +0.021]), is not significantly different on I.AM. slab
(+0.020 [-0.001, +0.041]; video_last is worse there, +0.026 [+0.003, +0.049]), and is about 15% (0.37 mm) worse
on the gentle sim error, outside the 2% tolerance.

### Decision

**Not kept. `models/contact_gnn_video.pt` was not created and the default model is unchanged.**

- Arm B (`video`, the arm the rule names): fails the sim tolerance on gentle position error against the control
  (+9.15%, i.e. 2.74 mm vs 2.51 mm; it is better than shipped), and shows no significant improvement on the
  held-out Kubric ground truth against either reference.
- Arm C (`video25`): passes every sim condition and improves CLEVRER clearly against both references in both
  floor modes, but on held-out Kubric it beats shipped only in slab mode and control only in floor mode; in
  neither mode does it beat both. The rule asks for improvement against the control AND the shipped model on the
  CLEVRER and Kubric sets, so it is not met. It is a near miss on a 91-episode set, and its `best.pt` is the
  epoch-2 checkpoint (780 updates), picked by a noisy score. It is also significantly WORSE on the real I.AM. box-toss set in
  slab mode (0.636 m vs 0.515 m shipped; rest-face accuracy 0.31 vs 0.55), which the rule does not test but
  which matters more than CLEVRER for a product about real boxes.

### Interpretation

- **What video-derived training changed.** It measurably moved the model towards CLEVRER-style sliding scenes:
  rest error on 1,000 unseen CLEVRER ground-truth episodes fell from 0.676 m (shipped) and 0.659 m (control) to
  0.603 m (10% video, epoch 1), 0.550 m (25% video, epoch 2) and about 0.53 m (last epochs), all with paired CIs
  far from 0. The control shows that only 0.017 m of that comes from more simulator fine-tuning. On the
  held-out video-derived test split (start state from pixels, target from ground truth) the video arms are also
  slightly better (Kubric -0.5 to -1.6 cm, significant; CLEVRER -3 to -4 cm in slab mode, not significant on 46
  episodes).
- **What it did not change.** Simulator metrics stayed within about 1-2% on chaotic error, penetration and fall
  precision / recall; nothing improved there. Kubric held-out improved only inconsistently. The two real
  box-toss sets did not improve: ContactNets moved by a few millimetres in both directions, I.AM. got slightly
  worse in every `best.pt` arm including the control (+0.8 to +1.3 cm in floor mode; the last-epoch video checkpoints are level with shipped there) and much worse for
  `video25` in slab mode.
- **Noise floor.** The CLEVRER tracks are accurate (0.6 cm median, section 3) and plentiful (22,905 cube
  transitions), and that is where the effect is. The Kubric tracks are sparse (3,579 cube transitions, about 15%
  of object-frames, mostly the late on-floor phase) with a 13-18 cm p90 error, which is as large as the shipped
  model's own 2 s chaotic error; they carry little usable signal. At the first fully observed frame the
  video-derived start positions of the test split are off by 8.6 cm (Kubric) and 5.1 cm (CLEVRER) on average
  against ground truth, the tail of the same noise. No orientation is ever supervised from video.
- **Why the default was not changed.** The pre-agreed rule was not met, the gain is confined to one rendered
  dataset from the same generator as the training videos, there is no gain on real data and one significant
  loss, and the checkpoints the selection picked contain almost no video training. The shipped simulator-trained
  model remains the default; the video path stays an opt-in experiment (`--video-data`, `--video-frac`).

### Limits of this experiment

- One run per arm, one seed, 34 minutes, GPUs shared with other trainings; arms matched by wall-clock, not updates.
- Checkpoint selection by a noisy sim validation score; a fixed-epoch comparison (the supplementary table) tells
  a slightly different story for Kubric slab mode.
- Held-out Kubric is 91 episodes; Physion is 10 episodes (its CIs are too wide to read).
- Both video sources are rendered simulator video. No real video of moving boxes was used for training.
- Static attributes (shape, size class, material, colour) and the Kubric camera come from dataset metadata, and
  the CLEVRER homography was calibrated on ground-truth annotations (section 1 and Limitations above).
