# Part C - external physics data (approximate evaluation)

> **Everything on this page is APPROXIMATE.** The external data was converted with the approximations
> listed in section 4. The three sources of sections 1-10 are *simulators* from other projects, not
> real-world video. Section 12 adds the first **real-world** data for the box model (motion-capture /
> AprilTag pose tracks of one tossed box, found with Exa); it is still approximate (assumed materials, one
> box, no video). Section 11 (articulated RBO recordings, kinematics only) is a separate track.
> This is an out-of-distribution sanity check of the learned world model, **not a validation of the product**.
> No model was trained or fine-tuned on any of this data.

## 1. Summary

- Three openly licensed datasets were verified, downloaded (subsets) and converted to Agatha episode
  shards: Kubric MOVi-A (1137 episodes), CLEVRER (1000 episodes), Physion (10 trials). Licences are in
  `DATA_LICENSES.md`. No scraped internet video is used. No team phone clips have been received.
- **None of the three contains what Part C wanted most**: an object leaving a table. All three have an
  unbounded ground plane, so "fell off" never happens in the ground truth; it can only be scored as
  "no false alarm". "Toppled" is degenerate in CLEVRER (1 of 4955 objects tips) and near-trivial in
  Kubric (objects are thrown; 84 % of cubes tip). Only Physion Dominoes has a meaningful
  toppled-or-not outcome, and only 8 such trials fit in the download budget.
- Measured (section 6): the new ContactGNN is clearly closer to the external ground truth than the old
  MLP on Kubric and CLEVRER (rest-position error on true cuboids, each model at its better ground
  mapping: Kubric 1.14 m -> 0.77 m, CLEVRER 1.27 m -> 0.72 m), produces no spurious falls or tips on
  CLEVRER where the old model in slab mode produces thousands, and is 0.25-0.35 m further from the
  source data than Agatha's own PyBullet box engine started from the same inputs (cuboids, slab). On the 8 Physion domino trials the new model **misses most
  topples** (recall 0.25 versus 0.91 for Agatha's PyBullet engine on identical inputs).
- **Fine-tuning on this data: not recommended** (section 8).
- **Real-world addition (section 12, measured on a newer checkpoint than sections 5-8):** 619 real tosses of
  a single box (49 I.AM. motion-capture, 570 ContactNets AprilTag). With the product's default materials
  Agatha's PyBullet engine ends 0.36 m (I.AM., ~1.0 m of travel) and 0.10 m (ContactNets, ~0.39 m of travel)
  from where the real box stops; one friction value measured from the data brings that to 0.07 m on both. With the
  friction input corrected, the new ContactGNN scores within 0.01 m of its teacher simulator in rest-position
  error on these real initial states in the shipped (slab) configuration (with the default friction on I.AM. it
  ends 0.10 m closer to the real rest position than its teacher, an imitation error and not a virtue); it is worse than the teacher at predicting which face a tumbling cube
  lands on (80 % versus 85 %) and is 0.05-0.43 m closer to the real rest position than the old MLP.

## 2. Data and licences

| Dataset | Simulator | Subset on disk | Licence (see `DATA_LICENSES.md` for caveats) | Used for |
|---|---|---|---|---|
| Kubric MOVi-A 128x128 | PyBullet | 1137 episodes (train shards 0-59 of 512), 0.54 GB | Apache-2.0 (repository-level; no dedicated data licence) | eval only |
| CLEVRER validation annotations | Blender / Bullet | 5000 annotation files (first 1000 converted), 0.72 GB | CC0 (explicit, official README) | eval only |
| Physion (NeurIPS 2021) | TDW / PhysX | 10 trials (8 Dominoes, 2 Support), 1.8 GB | MIT (repository-level; no dedicated data licence) | eval only (anecdotal) |
| I.AM. Archive 021, Box009 tosses (section 12) | **real** (OptiTrack mocap, 360 Hz) | 50 recordings, 0.39 GB | CC BY-NC-SA 4.0 (explicit) | eval + PyBullet parameter search only |
| ContactNets cube tosses (section 12) | **real** (AprilTags, 148 Hz) | 570 tosses, 9.9 MB | BSD-3-Clause (repository-level; no dedicated data licence) | eval + PyBullet parameter search only |
| Team phone clips | real | none received (`data/external/team_clips/` absent) | team-owned | nothing yet |

Raw downloads: `data/external/<dataset>/` (git-ignored; 3.22 GB = 2.99 GiB apparent size, `du -sh` reports
3.1G, i.e. over a 3 GB target in decimal units. `data/external/clevrer/annotation_validation.zip`, 78 MB,
duplicates the extracted directory and can be deleted, which leaves 3.14 GB). Converted shards: `data/external_v2/<dataset>/test/shard_*.npz` (16 MB).

## 3. Pipeline and commands

```
# 1. convert (9 s, CPU)
CUDA_VISIBLE_DEVICES="" .venv-train/bin/python -I scripts/prepare_external_data.py --workers 16
# 2. evaluate (about 7 min on 16 CPU threads) -> models/external_eval.json / .md
CUDA_VISIBLE_DEVICES="" .venv-train/bin/python -I scripts/eval_external.py
# 2b. real-world box tosses (section 12): convert (10 s), then evaluate + PyBullet grid search (about 4 min on 16 CPU
#     processes); --merge keeps the entries of the other datasets in models/external_eval.json untouched
CUDA_VISIBLE_DEVICES="" .venv-train/bin/python -I scripts/prepare_external_data.py --datasets real_iam_box_toss real_contactnets_cube
CUDA_VISIBLE_DEVICES="" .venv-train/bin/python -I scripts/eval_external.py --merge --calibrate \
    --datasets real_iam_box_toss real_iam_box_toss_calib real_contactnets_cube real_contactnets_cube_calib
# 3. unit tests (2 s, no network, no downloaded data needed)
.venv/bin/python -m pytest tests/test_external_data.py -q
# 4. team clips, when there are any (use .venv: YOLO / torchvision only import cleanly there)
CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 .venv/bin/python scripts/track_video.py CLIP.mp4
```

- `scripts/prepare_external_data.py`: parses the downloads with numpy / json / h5py only, converts to
  Y-up metres, resamples to 20 Hz (linear positions, slerp orientations), builds the data/v2 shard keys
  (`states, props, context, mask, contact_*, rest, fell, tipped, kind, chaotic, live_physics,
  episode_index, init_penetration`) plus `n_frames, shape, box_like, moved_far, settled, rest_pos,
  source_id`, and writes `meta.json` with measured statistics.
- `scripts/eval_external.py`: batched autoregressive rollouts (never teacher-forced) from the first
  converted frame over the whole available horizon, for six engines plus the ground truth:
  `hold_initial` (nothing moves), `agatha_physics` (Agatha's own PyBullet box engine, same settings as
  `agatha.physics.World`, from the same converted initial state), `old_mlp`, `old_mlp_projected`,
  `new_raw`, `new_projected`. Metric primitives are imported from `scripts/eval_contact.py`.
  `AGATHA_CODE_ROOT=<dir>` pins the `agatha` package to another checkout.
- Both ground mappings are evaluated: **slab** (ground = top of a large support slab at 0.75 m, as
  stored) and **floor** (ground = Agatha's y = 0 floor, default 1.2 x 0.8 m support with collisions
  disabled, i.e. the "noclip" situation of data/v2).

## 4. Approximations

The authoritative list is the docstring of `scripts/prepare_external_data.py` (A1-A9, K1-K5, C1-C5,
P1-P6). The ones that matter most for reading the numbers:

- **Boxes only.** Spheres, cylinders and cones are replaced by their bounding boxes (spheres with a
  non-rotating box). Two thirds of the Kubric and CLEVRER objects are not boxes. The box stand-ins
  overlap each other and the ground: the converted *ground truth itself* has a mean penetration of
  13.2 mm on Kubric (4.6 mm on its cubes, mostly from non-box neighbours), 1.2 mm on CLEVRER.
- **Ground plane instead of a table.** Unbounded ground mapped to an 8-15 m slab (training saw
  0.9-1.5 m tables) or to the floor with the support disabled. `fell` is all False.
- **`tipped` is relative**: the initially-up body axis rotates more than 35 degrees (objects start in
  arbitrary orientations), not `agatha.physics.summarize`'s absolute rule. For cylinders it means "rolled".
- **Kubric**: similarity rescale (length x0.36, time x0.6) so that 12 fps becomes exactly 0.05 s, gravity
  stays 10 m/s^2 and objects are 0.25 / 0.50 m (the larger is still above the 0.35 m training maximum);
  dataset velocities; restitution up to 0.7 (training: 0-0.6); ground friction 0.3 and restitution 0.5
  (generator defaults, consistent with the measured sliding deceleration and bounces; slab mode only,
  Agatha's y = 0 floor is fixed at 0.5 / 0); episodes last 1.15 s and only 22 % of
  cubes have come to rest by then, so "rest position" is mostly "position at 1.15 s".
- **CLEVRER**: units, sizes, masses and friction are undocumented (the README only gives 128 frames in
  5 s). Assumed 25 fps, 0.2 m
  cubes, 0.5 kg; friction **calibrated from the data** (median sliding deceleration -> object and
  support friction 0.133); finite-difference velocities; 4 s horizon.
- **Physion**: Froude rescale (length x0.5, time x sqrt 0.5; gravity stays 9.81); left-handed to
  right-handed mirror; rollouts start 2 source frames after the scripted push; distractor meshes and the
  static zone mat are dropped; PhysX, not Bullet.

## 5. Converted data (measured, `data/external_v2/*/meta.json`)

| | Kubric MOVi-A | CLEVRER | Physion |
|---|---|---|---|
| episodes | 1137 | 1000 | 10 (8 Dominoes + 2 Support) |
| dynamic objects | 7414 | 4955 | 47 |
| box-like objects | 2457 (33 %) | 1505 (30 %) | 43 (91 %) |
| box-like-only episodes | 6 | 4 | 8 |
| horizon | 1.15 s | 4 s | 0.8-4 s |
| object size | 0.25-0.50 m | 0.20 m (assumed) | 0.05-0.38 m |
| ground truth tipped (all / box-like) | 59.7 % / 83.6 % | 0.02 % / 0.07 % | 85 % / 91 % |
| at rest at the last frame (all / box-like) | 8.9 % / 22.4 % | 25.9 % / 29.4 % | 96 % / 100 % |
| objects that fall off anything | 0 | 0 | 0 |
| initial box overlap > 5 mm (episodes) | 36 % | 0 % | 0 % |

## 6. Measured results

Checkpoints: new = `models/contact_gnn.pt` (sha256 `7c53e6b60107...`, file written 2026-11-15 05:27, epoch
115 - the file may be an interim export of the still-running training job; re-run step 2 for a later
one), old = `models/mlp_multistep.pt`. Model / projection code: a snapshot of the **uncommitted working
tree** `agatha/contact_gnn.py` taken at 05:52 (sha256 `7f4ce7fcf5f5...`, i.e. `d7f6650` plus the other
session's pending `project` changes). Evaluation run 2026-11-15 05:52, after the Kubric ground
parameters were corrected in review (support friction 0.5 -> 0.3, restitution 0 -> 0.5). Full tables, all subsets, both ground mappings, confusion
matrices and paired bootstraps: `models/external_eval.md` / `.json`.

Position errors in metres. "rest pos" = position error at the last frame; "settled" = the same, only
for objects that are at rest in the ground truth. "tipped acc" must be read against the majority-class
accuracy in the caption. `agatha_physics` is not a learned model: it shows how far Agatha's own box
physics is from the source data when given the same converted inputs.

**Kubric MOVi-A, true cuboids** (2457 objects in 1001 episodes, 551 settled; almost all share the scene
with non-box shapes; always answering "tipped" scores 0.836)

| engine | ground | pos 0.5 s | pos 1 s | rest pos (1.15 s) | settled | tipped acc | tipped P / R | pen mean (mm) |
|---|---|---|---|---|---|---|---|---|
| hold_initial | either | 1.312 | 1.882 | 2.001 | 1.150 | 0.164 | - / 0.00 | 3.29 |
| agatha_physics | slab | 0.171 | 0.451 | 0.518 | 0.350 | 0.891 | 0.96 / 0.91 | 0.33 |
| agatha_physics | floor | 0.197 | 0.569 | 0.668 | 0.323 | 0.796 | 0.96 / 0.79 | 0.63 |
| old_mlp | slab | 0.919 | 1.440 | 1.554 | 0.737 | 0.593 | 0.83 / 0.65 | 32.48 |
| old_mlp | floor | 0.455 | 1.012 | 1.140 | 0.544 | 0.723 | 0.84 / 0.83 | 65.68 |
| old_mlp_projected | floor | 0.408 | 0.951 | 1.078 | 0.485 | 0.740 | 0.84 / 0.85 | 0.41 |
| new_raw | slab | 0.302 | 0.739 | 0.840 | 0.409 | 0.741 | 0.95 / 0.73 | 6.04 |
| new_raw | floor | 0.302 | 0.758 | 0.867 | 0.407 | 0.731 | 0.95 / 0.71 | 5.19 |
| **new_projected** | slab | 0.273 | 0.674 | 0.767 | 0.394 | 0.760 | 0.94 / 0.76 | 0.11 |
| **new_projected** | floor | 0.275 | 0.703 | 0.808 | 0.385 | 0.753 | 0.95 / 0.74 | 0.11 |

All 7414 Kubric objects (majority-class tipped accuracy 0.597), slab / floor: rest pos hold 2.032,
agatha_physics 0.649 / 0.779, old_mlp 1.589 / 1.177, new_raw 0.911 / 0.940, new_projected 0.857 / 0.899;
tipped accuracy agatha_physics 0.780 / 0.734, old_mlp 0.484 / 0.614, new_projected 0.780 / 0.765.

**CLEVRER, true cuboids** (1505 objects in 832 episodes, 443 settled; 1 object tips in the ground truth)

| engine | ground | pos 0.5 s | pos 1 s | rest pos (4 s) | settled | spurious tips | spurious falls | moved-far acc | pen mean (mm) |
|---|---|---|---|---|---|---|---|---|---|
| hold_initial | either | 0.252 | 0.476 | 1.269 | 0.293 | 0 | 0 | 0.282 | 0.11 |
| agatha_physics | slab | 0.006 | 0.025 | 0.369 | 0.067 | 0 | 0 | 0.894 | 0.04 |
| agatha_physics | floor | 0.037 | 0.141 | 0.923 | 0.165 | 0 | n/a | 0.815 | 0.02 |
| old_mlp | slab | 0.379 | 0.853 | 7.297 | 6.738 | 1504 | 103 | 0.718 | 1403.60 |
| old_mlp | floor | 0.185 | 0.417 | 1.270 | 0.379 | 4 | n/a | 0.474 | 4.10 |
| old_mlp_projected | slab | 0.318 | 0.743 | 3.962 | 2.643 | 358 | 341 | 0.718 | 0.00 |
| old_mlp_projected | floor | 0.183 | 0.412 | 1.253 | 0.338 | 0 | n/a | 0.476 | 0.00 |
| new_raw | slab | 0.019 | 0.071 | 0.728 | 0.120 | 0 | 0 | 0.817 | 0.19 |
| **new_projected** | slab | 0.018 | 0.070 | 0.720 | 0.117 | 0 | 0 | 0.817 | 0.01 |
| **new_projected** | floor | 0.042 | 0.159 | 0.956 | 0.177 | 0 | n/a | 0.815 | 0.01 |

In `floor` mode the calibrated support friction cannot be applied (Agatha's floor friction is fixed at
0.5), which is why every engine, including PyBullet, is worse there. All 4955 CLEVRER objects, slab:
rest pos hold 1.359, agatha_physics 0.455, old_mlp 7.345, old_mlp_projected 3.950, new_projected 0.869;
spurious falls old_mlp 379, old_mlp_projected 1110, new 0.

**Physion Dominoes** (8 trials, 34 dominoes, all settled; 32 topple, so always answering "tipped"
scores 0.941). **Anecdotal: n = 8 trials.**

| engine | ground | pos 0.5 s | rest pos | tipped acc | tipped P / R | moved-far acc | pen mean (mm) |
|---|---|---|---|---|---|---|---|
| hold_initial | either | 0.090 | 0.161 | 0.059 | - / 0.00 | 1.000 | 0.00 |
| agatha_physics | slab | 0.061 | 0.120 | 0.912 | 1.00 / 0.91 | 0.912 | 0.28 |
| old_mlp | slab | 0.318 | 1.011 | 0.853 | 0.94 / 0.91 | 0.000 | 41.41 |
| old_mlp | floor | 0.087 | 0.183 | 0.088 | 1.00 / 0.03 | 1.000 | 5.09 |
| new_raw | slab | 0.053 | 0.124 | 0.294 | 1.00 / 0.25 | 0.971 | 0.34 |
| **new_projected** | slab | 0.051 | 0.120 | 0.294 | 1.00 / 0.25 | 1.000 | 0.00 |
| **new_projected** | floor | 0.055 | 0.124 | 0.294 | 1.00 / 0.25 | 0.971 | 0.08 |

All 10 Physion trials (47 objects; the 2 Support towers are hit by a ball at up to 7.9 m/s after
rescaling): rest pos hold 0.376, agatha_physics 0.856 / 0.873 (slab / floor), old_mlp 2.080 / 0.836,
new_projected 0.578 / 0.532; tipped accuracy (majority 0.851) agatha_physics 0.851 / 0.872, new_projected
0.383 / 0.404. Nothing beats "nothing moves" on rest position there.

**Paired episode bootstrap, rest-position error on true cuboids, new_projected minus the other engine
(m; negative = new_projected is closer to the ground truth; 1000 resamples)**

| dataset | ground | vs old_mlp | vs old_mlp_projected | vs agatha_physics |
|---|---|---|---|---|
| Kubric | slab | -0.786 [-0.824, -0.752] | -0.785 [-0.825, -0.750] | +0.250 [+0.228, +0.269] |
| Kubric | floor | -0.332 [-0.356, -0.310] | -0.270 [-0.295, -0.247] | +0.139 [+0.120, +0.157] |
| CLEVRER | slab | -6.577 [-6.762, -6.380] | -3.243 [-3.416, -3.072] | +0.351 [+0.332, +0.369] |
| CLEVRER | floor | -0.314 [-0.328, -0.301] | -0.297 [-0.311, -0.284] | +0.033 [+0.031, +0.035] |
| Physion (10 trials) | slab | -1.411 [-2.117, -0.805] | -1.496 [-2.554, -0.775] | -0.074 [-0.330, +0.089] |
| Physion (10 trials) | floor | +0.151 [-0.063, +0.511] | +0.137 [-0.045, +0.416] | -0.114 [-0.476, +0.111] |

**Ground mapping.** For new_projected the slab mapping gives the lower rest-position error on the two large
sets (Kubric 0.857 vs 0.899 m, CLEVRER 0.869 vs 1.068 m, all objects) and floor is slightly lower on the 10
Physion trials (0.532 vs 0.578 m): the shards keep the slab mapping. The old MLP breaks down with the large
slab (on CLEVRER it tips every object and throws them 7 m) and must be read in floor mode, its better one.

## 7. Interpretation

- **New versus old.** On Kubric and CLEVRER the ContactGNN is closer to the external ground truth than
  the old MLP under both ground mappings, with bootstrap intervals well away from zero, and the gain is
  already there without the projection (`new_raw`). The old model is fragile to an unfamiliar support
  size; the new one is not. This agrees in direction with the in-distribution evaluation but is
  measured on other simulators' data through a lossy conversion, so the *sizes* of the gaps are not
  transferable.
- **How much of the remaining error is the model?** `agatha_physics` receives exactly the same converted
  inputs and is still 0.52 m (Kubric cubes, 1.15 s, slab) and 0.37 m (CLEVRER cubes, 4 s) away from the source
  data: that part is the conversion plus simulator differences (box stand-ins, restitution and damping
  conventions, unknown CLEVRER parameters) and chaotic sensitivity. The learned model adds 0.25 m
  (Kubric, slab) and 0.35 m (CLEVRER, slab) on top of that; the CLEVRER gap is long low-friction slides that
  the model stops too early (friction 0.133 x 0.133 is at the very edge of the training range).
- **Toppled / fell.** There is no evidence here about falling off a table: no dataset contains it. The
  only thing measured is the absence of false alarms (0 spurious falls and 0 spurious tips for the new
  model on 4955 CLEVRER objects, against 379 and 4954 for the old model in slab mode). On Kubric cubes
  (slab) Agatha's PyBullet engine reaches 0.891 tipped accuracy against 0.836 for always answering
  "tipped", while the learned models stay below that trivial baseline (new_projected 0.760, recall 0.76
  versus 0.91 for PyBullet): the new model under-predicts tossed-cube tipping as well. On Physion Dominoes the new model predicts only 8 of 32 topples while Agatha's PyBullet engine
  predicts 29 of 32 from the same inputs: that is a real weakness of the learned model on
  thin-tall-domino chain reactions (n = 8 trials), not a conversion artefact.
- **Penetration.** With the projection the new model's mean penetration is at most 0.11 mm, below the
  converted ground truth itself (1.2-13 mm, caused by the box stand-ins).
- **Not shown by this data:** anything about real objects, real friction, real cameras, or falls from a
  table edge.

## 8. Should this data be used for (low-weight) fine-tuning? No.

1. **It lacks the target events.** No table edge and no falls in any of the three; toppling labels are
   degenerate (CLEVRER) or near-constant (Kubric). Fine-tuning cannot improve "fell / toppled" from
   data that does not contain the distinction; the one informative set (Physion Dominoes) has 8 trials.
2. **The labels are inconsistent with Agatha's own physics.** Agatha's PyBullet engine, the reference the
   product compares the learned model against, disagrees with the converted trajectories by 0.37-0.86 m (rest position; 0.37 CLEVRER cubes, 0.86 Physion).
   Training towards them would pull the model away from the live physics engine.
3. **The box approximation is baked into the targets.** Two thirds of the objects are spheres and
   cylinders; their trajectories (rolling) are impossible for the boxes the model sees, and the targets
   contain 1-13 mm of mean interpenetration that the contact losses and the projection are designed to
   remove.
4. **Distribution gap in the inputs**: 8-15 m slab instead of a 0.9-1.5 m table, objects up to 0.5 m,
   restitution 0.7, guessed or calibrated friction and mass (CLEVRER), a 12 fps source with episodes
   that end in mid-air (Kubric).
5. **It is not real-world evidence anyway.** All three are simulators; Kubric is PyBullet, the same
   engine Agatha's generator uses. Anything they could teach can be generated in-house with exact
   parameters and labels (`scripts/generate_contact_data.py`).

What the data is good for: a fixed, external, out-of-distribution regression check (`eval_external.py`),
and one concrete lead - **add domino / chain-topple scenes of thin tall boxes (about 0.05 x 0.25 x 0.125 m,
friction down to 0.01) to our own generator**, then re-check the Physion recall. Real fine-tuning signal
for "fell / did not fall" has to come from team-recorded clips with a table edge (section 9).

## 9. Team phone clips and `scripts/track_video.py`

No clips have been received. The tool is ready and was smoke-tested on `tests/images/deskspace.mp4` (a
16 s walk-around clip of a desk with two bottles; nothing moves or falls in it):

- What works: frame reading, YOLO detection through `agatha.reconstruction.detect` (159 bottle detections
  in 164 sampled frames), IoU / centroid tracking (18 raw tracks, 4 kept), the repo's relative-depth +
  RANSAC plane code on the first frame, shard + `tracks.json` output; 19 s on CPU.
- What does not: the clip is filmed while walking, so the static-camera assumption is violated (global
  image shift up to 54 % of the width). The tool detects this and marks `labels_reliable: false`. The
  two real bottles were fragmented into 4 tracks; without compensation one track was labelled "fell"
  (false); with the crude global-shift compensation no track is labelled fallen, which is correct for
  this clip. 3D positions from this clip are meaningless. There is no ground truth, so no accuracy is
  claimed.
- Needed from the team for usable labels: a **tripod / static camera**, the whole table and the floor
  landing zone in view, objects from the YOLO classes in `agatha.reconstruction.PRIORS`, 3-10 s per
  clip, files under `data/external/team_clips/`.

## 10. Limitations

- Sections 1-9: simulator data only; conversions are lossy; CLEVRER physical parameters are assumed or calibrated.
  Section 12 (real-world data) lists its own limitations.
- Physion: 10 trials, intervals are wide; treat as anecdotes.
- Horizons are short on Kubric (1.15 s); "rest position" there is mostly a mid-motion position.
- `new_projected` / `old_mlp_projected` depend on the version of `contact_gnn.project`; the numbers here
  are for an uncommitted working-tree snapshot (sha256 `7f4ce7fcf5f5...`) because another session is still
  editing `agatha/contact_gnn.py`; step 2 was run against a copy of the package (`AGATHA_CODE_ROOT`).
  Against the committed `d7f6650` version the `*_projected` rows differ by at most 0.08 m (old model) and
  0.002 m (new model) on the unchanged CLEVRER / Physion shards. Re-run when that edit is committed.
- Kubric ground friction / restitution are the generator's defaults (checked against the data with small
  samples for the bounces); the released episodes were assumed to use them.
- The checkpoint may not be the final one (see section 6).

## 11. Exa-discovered articulated data (RBO, joints track; approximate, kinematics only)

Source: the **RBO Dataset of Articulated Objects and Interactions** (real mocap of people opening laptops,
drawers, doors, lids; CC BY 4.0, see `DATA_LICENSES.md`), one of the candidates returned by the Exa dataset
search (`scripts/find_physics_datasets.py` -> `data/external/exa_dataset_search.json`). It is the only
external source so far with real jointed rigid bodies and a kinematic model, and it is used for the
EXPERIMENTAL joint track (`agatha/joint_gnn.py`), not for the shipped box model.

**The motion is hand-driven.** Every joint movement is caused by a person (hand, or a stick with a
force/torque sensor in 78 of the 358 recordings). Those forces are not in the state, so a passive rollout from
t = 0 cannot be scored against these trajectories, and no dynamics result is reported. Every episode carries
`driven = True`.

### Commands

```bash
# optional: fetch more (streams the Zenodo archive, keeps only the .csv members; a recording archive is 40-800 MB)
.venv/bin/python -I scripts/prepare_external_joints.py --download treasurebox treasurebox07
# convert + kinematic report (CPU, ~5 s) -> data/external_v2/rbo_articulated_joints/, models/external_joints_kinematics.{json,md}
CUDA_VISIBLE_DEVICES="" .venv/bin/python -I scripts/prepare_external_joints.py
# tests (synthetic recording, no network / downloads needed, ~4 s)
CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest tests/test_external_joints.py -q
```

### What is converted

10 recordings -> **33 episodes** of 40 steps (2 s, non-overlapping windows; `--steps 80` gives 4 s
windows but drops the recordings shorter than 4 s) in the `data/joints_v1` layout (12 body slots, 12 joint
slots, `joints`, `joint_mask`, `joint_pos`, `group`), readable by `scripts/eval_joints.py::load_split`.
Kinds: rbo_cabinet 4, rbo_ikea 2, rbo_laptop 7, rbo_microwave 8, rbo_treasurebox 12. Skipped: microwave22 (kinematic model configuration_2017-08-04 does not fit this recording (median anchor gap 754 mm, median axis error 178 deg)).

| recording | object (marker configuration) | duration s | episodes | box sizes m (root / children; x, y up, z) | root displacement m | F/T | free-motion candidates | verified free |
|---|---|---|---|---|---|---|---|---|
| cabinet03 | cabinet (2017-08-21) | 8.23 | 4 | 0.84x0.49x0.43 / 0.64x0.17x0.43 / 0.64x0.06x0.43 | 0.011 | no | 1 | 0 |
| ikea08 | ikea (2017-06-01) | 2.69 | 1 | 0.78x0.80x0.38 / 0.33x0.17x0.38 / 0.37x0.38x0.17 | 0.019 | no | 1 | 0 |
| ikea09 | ikea (2017-06-01) | 2.93 | 1 | 0.78x0.80x0.38 / 0.33x0.17x0.38 / 0.37x0.38x0.17 | 0.011 | no | 1 | 0 |
| laptop04 | laptop (2017-06-01) | 6.42 | 3 | 0.31x0.04x0.24 / 0.30x0.03x0.24 | 0.089 | no | 1 | 0 |
| laptop07 | laptop (2017-06-01) | 4.14 | 2 | 0.24x0.04x0.31 / 0.24x0.03x0.30 | 0.006 | no | 1 | 0 |
| laptop13 | laptop (2017-06-01) | 4.52 | 2 | 0.31x0.05x0.24 / 0.30x0.04x0.24 | 0.008 | no | 1 | 0 |
| microwave04 | microwave (2017-08-03) | 17.99 | 8 | 0.33x0.29x0.52 / 0.03x0.27x0.41 | 0.004 | yes | 6 | 0 |
| treasurebox07 | treasurebox (2017-06-01) | 6.84 | 3 | 0.14x0.10x0.18 / 0.14x0.06x0.18 | 0.017 | no | 2 | 0 |
| treasurebox08 | treasurebox (2017-06-01) | 7.95 | 3 | 0.14x0.10x0.18 / 0.14x0.06x0.18 | 0.150 | no | 2 | 0 |
| treasurebox26 | treasurebox (2017-08-02) | 13.36 | 6 | 0.18x0.10x0.14 / 0.18x0.06x0.14 | 0.015 | yes | 7 | 0 |

Approximations (R1-R9 in the docstring of `scripts/prepare_external_joints.py`): one bounding box per rigid
part from the scan mesh (hollow furniture becomes solid, jointed boxes overlap); the box frame of a hinged
child is the parent's frame at the most compact pose (lid / door closed), so the row's coordinate is the URDF
value minus `joint_zero` (the laptop's URDF zero is ~120 deg open); `/map` is assumed Z-up; the resting surface
is the median lowest corner of the root box and becomes a support slab at 0.75 m of assumed size; velocities
are 20 Hz finite differences; masses (box volume x 300 kg/m^3, clipped to 0.05-3 kg), friction and damping are
placeholders; joint limits are not known (URDF placeholders) and stored as "no limit".

### (a) Kinematic consistency (measured)

`agatha.joint_gnn.joint_violation` / `joint_coordinates` on the measured part poses, source rate (100 Hz),
all valid frames:

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

On the float32 20 Hz shards, as an evaluator reads them (1599 joint-frames: 1189 hinge, 410 prismatic;
mean / median / p95 / max):

| | hinge | prismatic |
|---|---|---|
| anchor gap mm | 7.66 / 5.92 / 18.22 / 33.57 | 2.09 / 1.93 / 3.52 / 16.65 (off-axis part only) |
| axis / orientation error deg | 1.71 / 1.14 / 4.72 / 8.53 | 0.81 / 0.58 / 2.38 / 3.66 |

Reading:

- **Conventions hold on real data.** The joint coordinate computed by `joint_gnn` from the poses equals the
  dataset's own `joint_states` for prismatic joints (0.000 mm) and is within 0.3-4.5 deg (mean) for hinges; the
  sign-flipped comparison is 33-106 deg off wherever the joint actually moves (ikea08's door barely moves, so
  it cannot tell). Stored `joint_pos` equals `joint_coordinates` on the shard states to 3.6e-07.
- **Real joints are not exact.** Anchor gaps are millimetres to centimetres (2-6 mm for drawers and the
  treasure-box lid, 7-18 mm mean and up to 34-39 mm for the laptop and the microwave door) and hinge axes are
  0.2-4.6 deg (mean) off, against the 0.5 mm / 0.002 rad that `joint_gnn.joint_losses` leaves unpenalised
  (tuned for exact simulator joints). This is model-fit residual plus
  mocap noise (the URDF joints were themselves estimated by the dataset authors from mocap of the same marker
  configuration), not an independent ground truth. Consequence: a real-data joint evaluation needs tolerances
  around 2 cm / 5 deg, and a hard projection (`joint_gnn.solve_joints`) would move real states by that much.
- One recording (microwave22) is 0.75 m / 178 deg off with the URDF of its own marker configuration
  (2017-08-04) and also does not fit the other two microwave configurations (0.62 m); it is excluded by the
  fit gate (median gap > 5 cm or axis error > 15 deg). The cause was not investigated.

### (b) Free-motion segments: 0 usable

Searched at 100 Hz for two patterns per joint: a decelerating tail (>= 0.3 s after the speed peak, ending at
rest) and an acceleration that ends in an abrupt stop (a lid falling shut). Found: **23 kinematic candidates** in
10 recordings. **Verified free of external contact: 0.**

- Without F/T data (8 recordings, 10 candidates) the hand is simply not observed in what we keep (and RGB has no
  hand labels), and a hand slowing a lid down looks exactly like a lid coasting. None can be used.
- With F/T data (microwave04, treasurebox26; 13 candidates) a segment counts only if the force stays within
  0.5 N of its idle level AND the mocap-tracked F/T stick is at rest (< 0.02 m/s). The treasure-box candidates
  have 3-17 N on the stick. Two microwave04 candidates have only 0.45 / 0.50 N, but the stick travels with the
  door at 0.07 m/s, i.e. a slow push of a light door, which is why the stick criterion exists. None passes.
- So this subset contains no segment on which a passive joint model could be scored. The list of candidates
  with times is in `models/external_joints_kinematics.md`.

### Recommendation

- **Evaluation only, kinematics only.** Use the shards to check joint conventions and constraint metrics on
  real poses (done above), and as a sanity input for the joint projection. Do not report rollout errors on
  them as dynamics accuracy.
- **Do not train or fine-tune on it.** The transitions are driven by unobserved forces (the model would learn
  that lids move by themselves), masses / friction / limits are placeholders, and there are 33 windows from 5
  objects.
- If real free articulated dynamics is wanted, the cheapest route is a team recording (release a lid / door /
  drawer with a static camera or markers), not more of this dataset: its index describes the remaining recordings as the
  same kind of human interaction (not checked beyond the 11 downloaded).

### Limitations

- 10 of 358 recordings, 5 of 14 objects, chosen partly by archive size; not a representative sample.
- Box geometry is a coarse stand-in (R1); `rest` / `contact_support` flags are geometric guesses; lid boxes dip
  up to 1.5 cm below the estimated surface in some laptop frames (mesh / joint residual).
- Z-up and "the object stands upright at the first frame" are inferred from the data, not documented.
- The F/T idle level is the median force while no joint moves, which can include light contact; this is one
  reason the stick-at-rest condition is required as well. Both thresholds are heuristics.
- Nothing here was evaluated against a trained `joint_gnn` checkpoint; no model accuracy is claimed.

## 12. Exa-discovered real-world box data (APPROXIMATE)

> **First real-world comparison of the box model, and still approximate.** Two public datasets of ONE rigid
> box tossed onto a flat surface, with measured 6-D pose (no video is used). Friction and restitution are
> not given by either dataset and are *assumed*; velocities are finite differences; the 20 Hz model step
> is coarser than the impacts. Nothing here covers box-box contact, stacks, table edges or falls.
> No model was trained or fine-tuned on this data. Checkpoint: `models/contact_gnn.pt` sha256 `e4b5e4a3a905`
> (epoch 246, written 2026-11-15 06:35) - **newer than the checkpoint behind sections 5-8** (sha256
> `7c53e6b60107`, epoch 115), whose entries were kept unchanged in `models/external_eval.json` by `--merge`.

### 12.1 Discovery, licences, data

Both datasets were found with the Exa search in `scripts/find_physics_datasets.py` (raw results:
`data/external/exa_dataset_search.json`); the licences were then read on the official pages (`DATA_LICENSES.md`).

| | I.AM. Archive 021 (Box009) | ContactNets cube tosses |
|---|---|---|
| Source | 4TU.ResearchData, DOI 10.4121/21815994 (TU Eindhoven) | DAIR Lab, UPenn, github.com/DAIRLab/contact-nets |
| Licence | CC BY-NC-SA 4.0 (explicit; non-commercial, share-alike) | BSD-3-Clause (repository licence; no separate data licence) |
| Sensor | OptiTrack motion capture, 360 Hz | AprilTag pose tracking, 148 Hz |
| Object | foam-filled carton box 0.1644 x 0.1253 x 0.1202 m, 0.285 kg | acrylic cube 0.1048 m, 0.37 kg |
| Surface | idle rubber-belt conveyor | flat wooden surface |
| Motion | hand toss from beside the conveyor, lands flat and slides (authors removed tumbles) | hand toss, bounces / tumbles / slides to rest |
| Trajectories used / available | 49 / 50 (1 has no free-flight frame clear of the belt plane) | 570 / 570 |
| 20 Hz transitions | 1294 | 7434 |
| Episode length (s) min / median / max | 1.15 / 1.30 / 1.85 | 0.50 / 0.60 / 2.95 |
| Speed at the first frame (m/s) min / median / max | 2.33 / 2.63 / 2.98 | 0.92 / 1.15 / 1.43 |
| Spin at the first frame (rad/s) min / median / max | 0.16 / 1.02 / 3.55 | 0.46 / 3.95 / 10.87 |
| Horizontal travel (m) min / median / max | 0.74 / 1.04 / 1.26 | 0.02 / 0.39 / 0.58 |
| Rests on the face that was down at release | 49 / 49 (always the same face: the label is trivial here) | 43.3 % (6 faces, most frequent one 24.9 %) |
| Table (R6) | 48 of 49 need a table wider than the 1.5 m training maximum (1.5-2.0 m x 1.0 m) | all fit the 1.5 x 1.0 m table |
| Effective friction from sliding deceleration (R7) | 0.344 (IQR 0.331-0.373, 37 tracks) | 0.218 (IQR 0.208-0.226, 566 tracks) |

Total: **619 trajectories, 8728 transitions at 20 Hz** (data/v2 has 24 990 training episodes of 80 steps, i.e.
1 999 200 transitions; the real data is 0.4 % of that). Converted shards: `data/external_v2/real_*/test/` (1.3 MB).

### 12.2 Conversion (R1-R9 in `scripts/prepare_external_data.py`)

- Frames: I.AM. box pose relative to the conveyor pose; ContactNets positions x0.0524. Z-up -> Y-up by the
  proper rotation (x, y, z) -> (x, z, -y), quaternions to xyzw.
- Landing plane: one constant per dataset (the median lowest-corner height at the last frame: -2.4 mm
  I.AM., -1.9 mm ContactNets) so that a resting box touches the surface; then +0.75 m onto the support slab.
  The converted ground truth still penetrates the plane by 1.48 mm (I.AM., the carton rocks and gives) and
  0.25 mm (ContactNets) on average: that is the measurement / rigidity floor of the penetration metric.
- Velocities: slope of a local least-squares line over 9 frames at 360 Hz (25 ms) / 5 frames at 148 Hz
  (34 ms) (Savitzky-Golay first derivative), world-frame angular velocity from the same window on the
  relative rotation; poses are not smoothed. ContactNets ships velocities; ours differ from its `linvel`
  column by 0.033 m/s on average at the first kept frame. Then 20 Hz: positions linear, orientations slerp.
- Segment: I.AM. recordings start in the thrower's hand. The episode starts at the first frame of the
  detected free flight (acceleration within 2 m/s^2 of gravity) at which the lowest corner is >= 5 mm above
  the belt plane, and ends 0.5 s after the pose stops changing (3 mm / 1.5 deg). The flight left before the
  impact is short: 0.06-0.20 s (median 0.16 s). ContactNets tosses already start after release.
- The episode is rotated about the vertical (travel along +x) and centred on a 1.5 x 1.0 m table (the
  largest of the training range), enlarged only where the real trajectory would come closer than 0.25 m
  to an edge (48 I.AM. episodes). `floor` mode (Agatha's unbounded y = 0 floor) is the cross-check.
- Size and mass from the dataset metadata. Inertia is not used (uniform box assumed everywhere in Agatha).
- Materials, three variants. **default**: object 0.5 x support 0.5 (effective 0.25), restitution 0 - what
  the product would assume. **`_calib`**: one friction value per dataset from the measured sliding
  deceleration (above), restitution 0. **`_gridfit`**: the best pair of the PyBullet grid search of 12.4
  (chosen on half of the trajectories, in-memory variant of the default shards).
- Labels: `rest_face` = which of the 6 body faces is down at the last frame (the real "landed on which
  side" label). The relative `tipped` rule of the other datasets is not meaningful for a toss and is not reported.

### 12.3 Measured results (`models/external_eval.json` / `.md`)

Engines as in section 3. **`agatha_physics` is the simulator baseline**: PyBullet from the same measured
initial state with the same assumed parameters. The learned models were trained to imitate exactly this
simulator, so its row is what a perfect student would score. Position errors in metres; "rest" = last
frame of the episode; `pos 1 s (n)` = error at 1 s over the n trajectories that last that long.

**I.AM. box (49 real trajectories), ground = slab, default materials**

| engine | pos 0.25 s | pos 0.5 s | pos 1 s (n) | rest pos mean | rest pos median | ori final (deg) | rest-face acc | fell false alarms | pen mean (mm) |
|---|---|---|---|---|---|---|---|---|---|
| hold_initial | 0.583 | 0.918 | 1.029 (49) | 1.028 | 1.037 | 35.7 | 1.000 | 0 | 0.00 |
| agatha_physics | 0.021 | 0.104 | 0.360 (49) | 0.476 | 0.352 | 63.8 | 0.633 | 13 | 0.47 |
| old_mlp | 0.055 | 0.209 | 0.640 (49) | 0.651 | 0.891 | 86.8 | 0.224 | 28 | 33.49 |
| old_mlp_projected | 0.055 | 0.134 | 0.347 (49) | 0.439 | 0.280 | 51.0 | 0.796 | 10 | 0.00 |
| new_raw | 0.014 | 0.078 | 0.309 (49) | 0.372 | 0.294 | 48.1 | 0.837 | 8 | 0.16 |
| new_projected | 0.014 | 0.078 | 0.308 (49) | 0.371 | 0.296 | 47.9 | 0.837 | 8 | 0.00 |

**I.AM. box, ground = slab, friction from sliding deceleration (`real_iam_box_toss_calib`)**

| engine | pos 0.25 s | pos 0.5 s | pos 1 s (n) | rest pos mean | rest pos median | ori final (deg) | rest-face acc | fell false alarms | pen mean (mm) |
|---|---|---|---|---|---|---|---|---|---|
| hold_initial | 0.583 | 0.918 | 1.029 (49) | 1.028 | 1.037 | 35.7 | 1.000 | 0 | 0.00 |
| agatha_physics | 0.016 | 0.034 | 0.066 (49) | 0.066 | 0.055 | 29.0 | 1.000 | 0 | 0.39 |
| old_mlp | 0.065 | 0.192 | 0.475 (49) | 0.490 | 0.317 | 71.9 | 0.449 | 4 | 35.44 |
| old_mlp_projected | 0.064 | 0.137 | 0.220 (49) | 0.239 | 0.185 | 40.6 | 0.959 | 2 | 0.00 |
| new_raw | 0.019 | 0.031 | 0.062 (49) | 0.062 | 0.054 | 30.8 | 1.000 | 0 | 0.07 |
| new_projected | 0.019 | 0.031 | 0.062 (49) | 0.062 | 0.054 | 30.7 | 1.000 | 0 | 0.00 |

**I.AM. box, ground = slab, PyBullet grid-search optimum (`real_iam_box_toss_gridfit`)**

| engine | pos 0.25 s | pos 0.5 s | pos 1 s (n) | rest pos mean | rest pos median | ori final (deg) | rest-face acc | fell false alarms | pen mean (mm) |
|---|---|---|---|---|---|---|---|---|---|
| hold_initial | 0.583 | 0.918 | 1.029 (49) | 1.028 | 1.037 | 35.7 | 1.000 | 0 | 0.00 |
| agatha_physics | 0.015 | 0.032 | 0.057 (49) | 0.057 | 0.045 | 28.3 | 1.000 | 0 | 0.31 |
| old_mlp | 0.065 | 0.192 | 0.456 (49) | 0.473 | 0.317 | 70.7 | 0.469 | 2 | 35.47 |
| old_mlp_projected | 0.065 | 0.139 | 0.220 (49) | 0.237 | 0.173 | 40.6 | 0.959 | 2 | 0.01 |
| new_raw | 0.019 | 0.031 | 0.059 (49) | 0.059 | 0.050 | 30.8 | 1.000 | 0 | 0.07 |
| new_projected | 0.019 | 0.031 | 0.058 (49) | 0.058 | 0.050 | 30.8 | 1.000 | 0 | 0.00 |

**ContactNets cube (570 real trajectories), ground = slab, default materials**

| engine | pos 0.25 s | pos 0.5 s | pos 1 s (n) | rest pos mean | rest pos median | ori final (deg) | rest-face acc | fell false alarms | pen mean (mm) |
|---|---|---|---|---|---|---|---|---|---|
| hold_initial | 0.256 | 0.382 | 0.499 (10) | 0.400 | 0.396 | 61.0 | 0.433 | 0 | 0.00 |
| agatha_physics | 0.025 | 0.085 | 0.114 (10) | 0.103 | 0.100 | 24.6 | 0.853 | 0 | 0.44 |
| old_mlp | 0.057 | 0.138 | 0.201 (10) | 0.153 | 0.147 | 53.1 | 0.640 | 0 | 23.85 |
| old_mlp_projected | 0.051 | 0.133 | 0.193 (10) | 0.149 | 0.145 | 51.1 | 0.607 | 0 | 0.00 |
| new_raw | 0.025 | 0.085 | 0.115 (10) | 0.105 | 0.102 | 30.6 | 0.804 | 0 | 0.32 |
| new_projected | 0.025 | 0.083 | 0.112 (10) | 0.101 | 0.099 | 31.4 | 0.800 | 0 | 0.00 |

**ContactNets cube, ground = slab, friction from sliding deceleration (`real_contactnets_cube_calib`)**

| engine | pos 0.25 s | pos 0.5 s | pos 1 s (n) | rest pos mean | rest pos median | ori final (deg) | rest-face acc | fell false alarms | pen mean (mm) |
|---|---|---|---|---|---|---|---|---|---|
| hold_initial | 0.256 | 0.382 | 0.499 (10) | 0.400 | 0.396 | 61.0 | 0.433 | 0 | 0.00 |
| agatha_physics | 0.020 | 0.057 | 0.077 (10) | 0.073 | 0.070 | 24.1 | 0.856 | 0 | 0.47 |
| old_mlp | 0.056 | 0.135 | 0.197 (10) | 0.150 | 0.144 | 53.1 | 0.646 | 0 | 25.19 |
| old_mlp_projected | 0.049 | 0.127 | 0.187 (10) | 0.142 | 0.138 | 50.9 | 0.609 | 0 | 0.00 |
| new_raw | 0.021 | 0.060 | 0.084 (10) | 0.077 | 0.074 | 31.1 | 0.791 | 0 | 0.31 |
| new_projected | 0.021 | 0.058 | 0.082 (10) | 0.075 | 0.073 | 31.4 | 0.788 | 0 | 0.00 |

**ContactNets cube, ground = slab, PyBullet grid-search optimum (`real_contactnets_cube_gridfit`)**

| engine | pos 0.25 s | pos 0.5 s | pos 1 s (n) | rest pos mean | rest pos median | ori final (deg) | rest-face acc | fell false alarms | pen mean (mm) |
|---|---|---|---|---|---|---|---|---|---|
| hold_initial | 0.256 | 0.382 | 0.499 (10) | 0.400 | 0.396 | 61.0 | 0.433 | 0 | 0.00 |
| agatha_physics | 0.013 | 0.026 | 0.084 (10) | 0.043 | 0.041 | 30.5 | 0.779 | 0 | 0.55 |
| old_mlp | 0.053 | 0.129 | 0.247 (10) | 0.147 | 0.136 | 53.2 | 0.630 | 0 | 27.91 |
| old_mlp_projected | 0.043 | 0.112 | 0.175 (10) | 0.127 | 0.121 | 50.5 | 0.604 | 0 | 0.00 |
| new_raw | 0.015 | 0.033 | 0.084 (10) | 0.048 | 0.045 | 38.4 | 0.696 | 0 | 0.39 |
| new_projected | 0.016 | 0.034 | 0.075 (10) | 0.049 | 0.047 | 38.4 | 0.700 | 0 | 0.00 |

**Ground = floor (unbounded y = 0 floor, support collisions off; object friction rescaled to keep the pair product)**

I.AM. box:

| materials | engine | pos 0.5 s | rest pos mean | ori final (deg) | rest-face acc |
|---|---|---|---|---|---|
| default (0.5 x 0.5 = 0.25, e 0) | agatha_physics | 0.104 | 0.359 | 29.0 | 1.000 |
| default (0.5 x 0.5 = 0.25, e 0) | old_mlp | 0.243 | 0.297 | 46.3 | 0.980 |
| default (0.5 x 0.5 = 0.25, e 0) | new_projected | 0.060 | 0.295 | 31.2 | 1.000 |
| friction from sliding deceleration | agatha_physics | 0.034 | 0.066 | 29.0 | 1.000 |
| friction from sliding deceleration | old_mlp | 0.348 | 0.437 | 43.6 | 1.000 |
| friction from sliding deceleration | new_projected | 0.064 | 0.078 | 31.4 | 1.000 |
| PyBullet grid-search optimum | agatha_physics | 0.032 | 0.057 | 29.0 | 1.000 |
| PyBullet grid-search optimum | old_mlp | 0.354 | 0.444 | 43.4 | 1.000 |
| PyBullet grid-search optimum | new_projected | 0.069 | 0.088 | 31.5 | 1.000 |

ContactNets cube:

| materials | engine | pos 0.5 s | rest pos mean | ori final (deg) | rest-face acc |
|---|---|---|---|---|---|
| default (0.5 x 0.5 = 0.25, e 0) | agatha_physics | 0.085 | 0.103 | 24.6 | 0.853 |
| default (0.5 x 0.5 = 0.25, e 0) | old_mlp | 0.200 | 0.215 | 53.3 | 0.574 |
| default (0.5 x 0.5 = 0.25, e 0) | new_projected | 0.130 | 0.149 | 46.3 | 0.612 |
| friction from sliding deceleration | agatha_physics | 0.057 | 0.073 | 24.1 | 0.856 |
| friction from sliding deceleration | old_mlp | 0.192 | 0.208 | 53.3 | 0.579 |
| friction from sliding deceleration | new_projected | 0.098 | 0.117 | 49.1 | 0.575 |
| PyBullet grid-search optimum | agatha_physics | 0.026 | 0.043 | 30.5 | 0.779 |
| PyBullet grid-search optimum | old_mlp | 0.175 | 0.188 | 53.2 | 0.582 |
| PyBullet grid-search optimum | new_projected | 0.034 | 0.044 | 54.0 | 0.505 |

**Paired episode bootstrap (1000 resamples), B - A, negative = B closer to the measurement.**
`rest_face_error_rate` = fraction of trajectories with the wrong face down.

I.AM. box:

| materials | ground | metric | A -> B | A | B | B - A | 95 % CI | n |
|---|---|---|---|---|---|---|---|---|
| default (0.5 x 0.5 = 0.25, e 0) | slab | rest_pos_err_m | old_mlp -> new_projected | 0.651 | 0.371 | -0.280 | [-0.388, -0.169] | 49 |
| default (0.5 x 0.5 = 0.25, e 0) | slab | rest_face_error_rate | old_mlp -> new_projected | 0.776 | 0.163 | -0.612 | [-0.755, -0.469] | 49 |
| default (0.5 x 0.5 = 0.25, e 0) | slab | rest_pos_err_m | agatha_physics -> new_projected | 0.476 | 0.371 | -0.105 | [-0.144, -0.072] | 49 |
| default (0.5 x 0.5 = 0.25, e 0) | slab | rest_face_error_rate | agatha_physics -> new_projected | 0.367 | 0.163 | -0.204 | [-0.306, -0.102] | 49 |
| default (0.5 x 0.5 = 0.25, e 0) | floor | rest_pos_err_m | old_mlp -> new_projected | 0.297 | 0.295 | -0.002 | [-0.041, 0.032] | 49 |
| default (0.5 x 0.5 = 0.25, e 0) | floor | rest_face_error_rate | old_mlp -> new_projected | 0.020 | 0.000 | -0.020 | [-0.061, 0.000] | 49 |
| default (0.5 x 0.5 = 0.25, e 0) | floor | rest_pos_err_m | agatha_physics -> new_projected | 0.359 | 0.295 | -0.064 | [-0.070, -0.058] | 49 |
| default (0.5 x 0.5 = 0.25, e 0) | floor | rest_face_error_rate | agatha_physics -> new_projected | 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 49 |
| friction from sliding deceleration | slab | rest_pos_err_m | old_mlp -> new_projected | 0.490 | 0.062 | -0.428 | [-0.509, -0.342] | 49 |
| friction from sliding deceleration | slab | rest_face_error_rate | old_mlp -> new_projected | 0.551 | 0.000 | -0.551 | [-0.673, -0.408] | 49 |
| friction from sliding deceleration | slab | rest_pos_err_m | agatha_physics -> new_projected | 0.066 | 0.062 | -0.004 | [-0.011, 0.004] | 49 |
| friction from sliding deceleration | slab | rest_face_error_rate | agatha_physics -> new_projected | 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 49 |
| friction from sliding deceleration | floor | rest_pos_err_m | old_mlp -> new_projected | 0.437 | 0.078 | -0.360 | [-0.380, -0.340] | 49 |
| friction from sliding deceleration | floor | rest_face_error_rate | old_mlp -> new_projected | 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 49 |
| friction from sliding deceleration | floor | rest_pos_err_m | agatha_physics -> new_projected | 0.066 | 0.078 | 0.012 | [-0.006, 0.030] | 49 |
| friction from sliding deceleration | floor | rest_face_error_rate | agatha_physics -> new_projected | 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 49 |
| PyBullet grid-search optimum | slab | rest_pos_err_m | old_mlp -> new_projected | 0.473 | 0.058 | -0.415 | [-0.493, -0.332] | 49 |
| PyBullet grid-search optimum | slab | rest_face_error_rate | old_mlp -> new_projected | 0.531 | 0.000 | -0.531 | [-0.653, -0.388] | 49 |
| PyBullet grid-search optimum | slab | rest_pos_err_m | agatha_physics -> new_projected | 0.057 | 0.058 | 0.001 | [-0.005, 0.008] | 49 |
| PyBullet grid-search optimum | slab | rest_face_error_rate | agatha_physics -> new_projected | 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 49 |
| PyBullet grid-search optimum | floor | rest_pos_err_m | old_mlp -> new_projected | 0.444 | 0.088 | -0.356 | [-0.375, -0.336] | 49 |
| PyBullet grid-search optimum | floor | rest_face_error_rate | old_mlp -> new_projected | 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 49 |
| PyBullet grid-search optimum | floor | rest_pos_err_m | agatha_physics -> new_projected | 0.057 | 0.088 | 0.032 | [0.014, 0.049] | 49 |
| PyBullet grid-search optimum | floor | rest_face_error_rate | agatha_physics -> new_projected | 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 49 |

ContactNets cube:

| materials | ground | metric | A -> B | A | B | B - A | 95 % CI | n |
|---|---|---|---|---|---|---|---|---|
| default (0.5 x 0.5 = 0.25, e 0) | slab | rest_pos_err_m | old_mlp -> new_projected | 0.153 | 0.101 | -0.051 | [-0.056, -0.047] | 570 |
| default (0.5 x 0.5 = 0.25, e 0) | slab | rest_face_error_rate | old_mlp -> new_projected | 0.360 | 0.200 | -0.160 | [-0.200, -0.116] | 570 |
| default (0.5 x 0.5 = 0.25, e 0) | slab | rest_pos_err_m | agatha_physics -> new_projected | 0.103 | 0.101 | -0.002 | [-0.004, 0.001] | 570 |
| default (0.5 x 0.5 = 0.25, e 0) | slab | rest_face_error_rate | agatha_physics -> new_projected | 0.147 | 0.200 | 0.053 | [0.025, 0.082] | 570 |
| default (0.5 x 0.5 = 0.25, e 0) | floor | rest_pos_err_m | old_mlp -> new_projected | 0.215 | 0.149 | -0.067 | [-0.070, -0.063] | 570 |
| default (0.5 x 0.5 = 0.25, e 0) | floor | rest_face_error_rate | old_mlp -> new_projected | 0.426 | 0.388 | -0.039 | [-0.065, -0.011] | 570 |
| default (0.5 x 0.5 = 0.25, e 0) | floor | rest_pos_err_m | agatha_physics -> new_projected | 0.103 | 0.149 | 0.045 | [0.043, 0.048] | 570 |
| default (0.5 x 0.5 = 0.25, e 0) | floor | rest_face_error_rate | agatha_physics -> new_projected | 0.147 | 0.388 | 0.240 | [0.195, 0.288] | 570 |
| friction from sliding deceleration | slab | rest_pos_err_m | old_mlp -> new_projected | 0.150 | 0.075 | -0.075 | [-0.079, -0.070] | 570 |
| friction from sliding deceleration | slab | rest_face_error_rate | old_mlp -> new_projected | 0.354 | 0.212 | -0.142 | [-0.184, -0.102] | 570 |
| friction from sliding deceleration | slab | rest_pos_err_m | agatha_physics -> new_projected | 0.073 | 0.075 | 0.002 | [0.000, 0.005] | 570 |
| friction from sliding deceleration | slab | rest_face_error_rate | agatha_physics -> new_projected | 0.144 | 0.212 | 0.068 | [0.042, 0.095] | 570 |
| friction from sliding deceleration | floor | rest_pos_err_m | old_mlp -> new_projected | 0.208 | 0.117 | -0.091 | [-0.095, -0.087] | 570 |
| friction from sliding deceleration | floor | rest_face_error_rate | old_mlp -> new_projected | 0.421 | 0.425 | 0.004 | [-0.023, 0.030] | 570 |
| friction from sliding deceleration | floor | rest_pos_err_m | agatha_physics -> new_projected | 0.073 | 0.117 | 0.044 | [0.041, 0.047] | 570 |
| friction from sliding deceleration | floor | rest_face_error_rate | agatha_physics -> new_projected | 0.144 | 0.425 | 0.281 | [0.235, 0.323] | 570 |
| PyBullet grid-search optimum | slab | rest_pos_err_m | old_mlp -> new_projected | 0.147 | 0.049 | -0.098 | [-0.104, -0.092] | 570 |
| PyBullet grid-search optimum | slab | rest_face_error_rate | old_mlp -> new_projected | 0.370 | 0.300 | -0.070 | [-0.109, -0.026] | 570 |
| PyBullet grid-search optimum | slab | rest_pos_err_m | agatha_physics -> new_projected | 0.043 | 0.049 | 0.006 | [0.004, 0.008] | 570 |
| PyBullet grid-search optimum | slab | rest_face_error_rate | agatha_physics -> new_projected | 0.221 | 0.300 | 0.079 | [0.053, 0.109] | 570 |
| PyBullet grid-search optimum | floor | rest_pos_err_m | old_mlp -> new_projected | 0.188 | 0.044 | -0.144 | [-0.149, -0.140] | 570 |
| PyBullet grid-search optimum | floor | rest_face_error_rate | old_mlp -> new_projected | 0.418 | 0.495 | 0.077 | [0.049, 0.105] | 570 |
| PyBullet grid-search optimum | floor | rest_pos_err_m | agatha_physics -> new_projected | 0.043 | 0.044 | 0.001 | [-0.001, 0.003] | 570 |
| PyBullet grid-search optimum | floor | rest_face_error_rate | agatha_physics -> new_projected | 0.221 | 0.495 | 0.274 | [0.233, 0.312] | 570 |

### 12.4 PyBullet friction x restitution grid search (simulator versus reality, no learned model)

`eval_external.py --calibrate`: 12 effective friction values x 7 effective restitution values (pair
products; object = support = sqrt(value)), Agatha's PyBullet box engine from the measured initial state of
every trajectory on an 8 m slab (no reachable edge, so this is contact physics only). The pair is chosen
on the even-numbered trajectories by the mean position error over the rollout and reported on the
odd-numbered ones.

I.AM. box (fit 25, holdout 24):

| parameters | mu_eff | e_eff | fit rollout pos (m) | holdout rollout pos (m) | holdout rest pos mean (m) | holdout rest pos median (m) | holdout ori final (deg) | holdout rest-face acc |
|---|---|---|---|---|---|---|---|---|
| grid best (rollout objective) | 0.350 | 0.100 | 0.041 | 0.039 | 0.056 | 0.047 | 23.0 | 1.000 |
| grid best by rest position | 0.350 | 0.100 | 0.041 | 0.039 | 0.056 | 0.047 | 23.0 | 1.000 |
| product default | 0.250 | 0.000 | 0.205 | 0.190 | 0.346 | 0.342 | 23.4 | 1.000 |
| friction from sliding deceleration, e = 0 | 0.344 | 0.000 | 0.047 | 0.042 | 0.062 | 0.058 | 23.5 | 1.000 |

Grid best minus default on the 24 holdout trajectories (paired bootstrap): rest position -0.290 m [-0.309, -0.265], rollout-mean position -0.151 m [-0.161, -0.138], final orientation -0.4 deg [-2.0, 1.0]. Fit on 25 trajectories.

Holdout rest-position error (m), rows mu_eff, columns e_eff:

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

ContactNets cube (fit 285, holdout 285):

| parameters | mu_eff | e_eff | fit rollout pos (m) | holdout rollout pos (m) | holdout rest pos mean (m) | holdout rest pos median (m) | holdout ori final (deg) | holdout rest-face acc |
|---|---|---|---|---|---|---|---|---|
| grid best (rollout objective) | 0.150 | 0.000 | 0.019 | 0.018 | 0.043 | 0.040 | 30.9 | 0.782 |
| grid best by rest position | 0.200 | 0.300 | 0.022 | 0.022 | 0.042 | 0.038 | 30.0 | 0.842 |
| product default | 0.250 | 0.000 | 0.049 | 0.051 | 0.105 | 0.103 | 24.5 | 0.856 |
| friction from sliding deceleration, e = 0 | 0.218 | 0.000 | 0.034 | 0.036 | 0.075 | 0.071 | 23.2 | 0.874 |
| dataset authors' simulation values (not on the grid) | 0.180 | 0.125 | 0.017 | 0.017 | 0.029 | 0.023 | 27.7 | 0.828 |

Grid best minus default on the 285 holdout trajectories (paired bootstrap): rest position -0.062 m [-0.066, -0.057], rollout-mean position -0.032 m [-0.034, -0.031], final orientation 6.5 deg [2.9, 10.1]. Fit on 285 trajectories.

Holdout rest-position error (m), rows mu_eff, columns e_eff:

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

### 12.5 Interpretation

**(a) Simulator versus reality.**
- With the product's default materials the simulator is far off on the I.AM. box: it ends 0.36 m (floor) /
  0.48 m (table, including 13 of 49 predicted slides off the table edge that did not happen) from the real
  rest position after about 1.0 m of travel. On the ContactNets cube it is 0.10 m off after about 0.39 m
  of travel. For comparison, "nothing moves" is 1.03 m and 0.40 m off.
- **Most of that gap is one number, friction.** The real effective friction is 0.34 (carton on rubber
  belt) and 0.22 (acrylic on wood) by the sliding-deceleration rule; the default 0.25 lies between them.
  With that single measured value PyBullet ends 0.066 m (I.AM.) and 0.073 m (ContactNets) from the real
  rest position. The grid search gives 0.35 / 0.1 for I.AM. (holdout 0.056 m, median 0.047 m) and
  0.15 / 0 for ContactNets (holdout 0.043 m); restitution is practically unidentifiable for the sliding
  I.AM. box (the error is flat along that axis) and matters for the bouncing cube.
- **The grid is coarse and the residual is not zero.** The ContactNets authors' own simulation values
  (0.18 / 0.125, between our grid points) do better than our grid optimum on the same holdout: 0.029 m.
  After calibration the remaining error is about 3-6 cm of rest position (5 % of the travel for I.AM.,
  7-11 % for ContactNets), 23-31 deg of final orientation (for I.AM. this is rotation about the vertical,
  since the face is right in every toss; not decomposed for the cube), and the simulator puts the cube on
  the measured face in only 78-87 % of the holdout tosses for every parameter set in the table above (no
  grid point exceeds 86 %; badly wrong friction goes as low as 20 %). A plausible reading, not tested
  here, is that tumbling outcomes depend on details a uniform rigid box with one friction value lacks.
  Lower position error does not imply better face agreement (grid optimum 78.2 % versus default 85.6 %).

**(b) Learned model versus its teacher simulator.**
- In the shipped configuration (slab, with projection) the ContactGNN is not distinguishable
  from PyBullet in rest position on the I.AM. box (n = 49) with calibrated or grid-fit materials (difference
  -0.004 m [-0.011, 0.004] and +0.001 m [-0.005, 0.008]) and within 0.002-0.006 m of it on the ContactNets
  cube (n = 570) in all three material variants. On these two objects, correcting the friction *input*
  moves the model about as far as it moves the simulator (I.AM.: 0.371 m ->
  0.062 m for the model, 0.476 m -> 0.066 m for PyBullet).
- With the default (wrong) friction on I.AM. the model is *closer* to reality than its teacher (-0.105 m
  [-0.144, -0.072] slab, -0.064 m [-0.070, -0.058] floor). That is not a virtue: it is consistent with the
  model stopping long, fast slides somewhat earlier than PyBullet does (8 versus 13 predicted slides off
  the table edge), an imitation error that happens to point towards the real box here.
- Where the model is measurably worse than its teacher: **which face a tumbling cube lands on**
  (ContactNets, slab: 80.0 % versus 85.3 %, error rate +5.3 points [2.5, 8.2]; final orientation 31 deg
  versus 25 deg) and **the floor / support-collisions-off configuration** (ContactNets floor: rest
  position 0.149 m versus 0.103 m, face accuracy 61 % versus 85 %). The second is a known weaker regime of
  the model (the "noclip" scenes are 3 % of training), not something specific to real data.
- New versus old model: the ContactGNN is closer to the real rest position than the old MLP in every slab
  comparison (I.AM. -0.28 to -0.43 m, ContactNets -0.05 to -0.10 m, all intervals away from zero) and has the
  right face down far more often (I.AM. 84-100 % versus 22-47 %; ContactNets 70-80 % versus 63-65 %). The
  one tie is I.AM. in floor mode with default materials (-0.002 m [-0.041, 0.032]). The old model
  without projection also interpenetrates the surface by 24-35 mm on average; the new one by 0.1-0.4 mm
  before projection.
- Sample sizes: 49 trajectories (I.AM.; one box, one surface, one thrower, flat landings only) and 570
  (ContactNets; one cube, one surface). The 1 s horizon on ContactNets has only 10 trajectories.

### 12.6 Would these trajectories be useful for low-weight fine-tuning? Measured assessment

- **Size.** 619 trajectories / 8728 transitions, two objects, two surfaces, single body only. That is 0.4 %
  of the simulated training transitions and contains none of the multi-object events the product is about.
- **What the data disagrees with the simulator about.** After fixing one friction value per surface the
  simulator is within 3-7 cm of the real rest position, and the learned model is already within 1 cm of
  the simulator there. The dominant real-versus-sim difference is therefore a *parameter the model takes
  as an input*, not dynamics the model gets wrong. Fine-tuning on the real tracks with the default
  materials as inputs would teach the model that "friction 0.5 x 0.5" sometimes means 0.34 and sometimes
  0.22-0.15, i.e. it would blur the friction conditioning that section 12.5 (b) shows is working.
- **With calibrated materials as inputs** the residual signal is small (the 3-6 cm and the face
  disagreements above), is partly measurement and resampling error (20 Hz steps, finite-difference
  velocities, 1-3 mm plane uncertainty, a carton that is not rigid), and comes from 2 objects: a
  low-weight fine-tune could at best nudge single-box toss behaviour and risks moving the model away from
  the live PyBullet engine it is compared against in the product. Not measured: no fine-tuning run was done.
- **Licence.** I.AM. is CC BY-NC-SA 4.0 (non-commercial, share-alike); training a shipped model on it
  would need a licence decision. ContactNets is BSD-3 at repository level.
- **Recommendation.** (1) Keep both sets as a fixed **real-world regression check** (`eval_external.py`,
  about 4 minutes on CPU) with the simulator-baseline row next to every model row. (2) Use them for
  **parameter calibration, not for gradient updates**: the measured friction values (0.34 carton / rubber,
  0.15-0.22 acrylic / wood) are evidence about the material defaults, and a material prior per surface
  would remove most of the 0.10-0.36 m default-material error at no training cost. (3) **Do not fine-tune
  now.** If real-data fine-tuning is wanted later, the larger I.AM. archive (DOI 10.4121/17122553, 241
  recordings, several box types, 2.65 GB, same licence; not downloaded because of the 2 GB cap) and
  multi-object recordings would be needed first, with calibrated materials as inputs and a held-out split.
- **A lead for the simulator / generator rather than the model:** the cube's resting face. PyBullet agrees
  with the measurement in 78-87 % of the tosses and the model in 70-80 %; the model-versus-teacher part
  (5-8 points in slab mode) can be attacked with more tumbling single-box tosses in our own generator.

### 12.7 Limitations of the real-world comparison

- One box per scene on a flat surface: no box-box contact, no stacking, no edges, no falls.
- Friction / restitution are assumed or fitted, never measured independently; the `_calib` and `_gridfit`
  variants use the evaluation data itself to choose 1-2 scalars (`_gridfit`: half of the trajectories).
- The I.AM. start state is taken 0.06-0.20 s before the impact; 48 of 49 tables are wider than the training
  range; the rest-face label is constant there. The carton is not perfectly rigid.
- Position errors at rest include yaw-induced and resampling effects; orientation errors are not reduced
  modulo the box symmetry.
- Results depend on the working-tree version of `agatha/contact_gnn.py` (sha256 `7f4ce7fcf5f5`) for the
  `*_projected` rows, and on the checkpoint named at the top of this section. Training was still running
  when this was written; a later `models/contact_gnn.pt` needs a rerun (about 4 minutes) before these
  numbers are quoted for it.
- The old MLP takes neither support friction nor restitution as input (`props[..., :6]`, `context[:, :8]`,
  as in `scripts/eval_contact.py`). In slab mode with the `_calib` / `_gridfit` materials it therefore
  only sees the changed *object* friction; `floor` mode, where the object friction is rescaled against the
  fixed 0.5 floor, is the comparison in which it receives the full effective friction. The new-versus-old
  ordering is the same in both modes except for the I.AM. default-material tie.
- Gravity is 9.81 m/s^2 in every engine. The free-flight vertical acceleration measured in the raw tracks
  is lower: median 9.59 m/s^2 (I.AM., 50 recordings) and 9.61 m/s^2 (ContactNets, 560 tosses with enough
  flight frames), i.e. about 2 % (air drag on light boxes and / or scale error; not investigated).
- The measured I.AM. box dips up to 11 mm below the fitted belt plane during a trajectory (median over
  trajectories of the deepest dip 5.6 mm; ContactNets median 1.0 mm, deepest 3.5 mm): carton and belt give
  at impact, which no rigid engine reproduces. At rest the lowest corner is within 0.7 mm (I.AM.) and
  1.3 mm (ContactNets) of the plane in every trajectory.
- I.AM. is licensed CC BY-NC-SA 4.0: evaluation and parameter fitting only, nothing redistributed.
- **Independent review (same day).** Re-checked without the converter's code: raw I.AM. box vertices rest
  on the conveyor frame with the 0.1202 m axis vertical in 50 of 50 recordings (so the size permutation is
  right); converted first steps are ballistic (p1 - p0 - v0 dt = -12.0 / -12.1 mm vertically against
  -12.3 mm expected, horizontal residual below 1 mm); the stored angular velocity is world-frame
  (0.16 deg one-step orientation residual, against 2.5 / 14 deg if it were body-frame); every I.AM.
  episode starts in free flight (acceleration within 3.6 m/s^2 of gravity from the first frame to the
  impact on a 13-frame fit, against a median 18 m/s^2 in the hand phase before it). The 137 table rows
  of 12.3-12.4 were compared with `models/external_eval.json` by script, and rerunning the two I.AM.
  default / `_calib` evaluations reproduced the stored values exactly.

## 13. MeshGraphNets cloth data for the EXPERIMENTAL soft-body track (training + evaluation)

Scope: only the experimental soft-body model (`agatha/soft_gnn.py`); the shipped box model does not use it.
**Licence status: no explicit data licence; used by team decision (project owner, 2026-11-15) under the
repository Apache-2.0 licence; model not publicly released** (row in `DATA_LICENSES.md`; revisit before any
release). Section 8 (no fine-tuning on external data) is about the box model and stays as it is.

### Commands

```bash
# convert (7 s, CPU; hand-written TFRecord reader, nothing is executed) -> data/external_v2/meshgraphnets_soft/{train,val,test}
.venv/bin/python scripts/prepare_external_soft.py
# tests (synthetic trajectories, no download needed, 2 s)
.venv/bin/python -m pytest tests/test_external_soft.py -q
# mix into training: 5% of every batch;  evaluate on the converted test split
... scripts/train_soft_ddp.py --data data/soft_v2 --extra-data data/external_v2/meshgraphnets_soft --extra-frac 0.05 ...
.venv-train/bin/python scripts/eval_soft.py --data data/external_v2/meshgraphnets_soft --split test --ckpt CKPT --out models/soft_eval_meshgraphnets.json --example ''
```

### What is converted (measured, `data/external_v2/meshgraphnets_soft/meta.json`)

| set | local files | converted | reason / notes |
|---|---|---|---|
| `sphere_simple` | `valid.tfrecord` only (100 trajectories, 501 frames at 100 Hz, 1731 nodes = 1087 cloth + 2 handles + 642 sphere) | **100 of 100** -> 80 train / 10 val / 10 test (our own partition by trajectory index; the upstream train / test files were not downloaded) | cloth pinned at two handles, kinematic sphere moves through it |
| `flag_minimal` | train / valid / test, byte-identical (11 trajectories) | **0** | wind-driven: the driving force is not observable from the state; no held-out data. Mix fraction 0. |
| `flag_simple`, `sphere_dynamic`, `deforming_plate` | `meta.json` only | 0 | not downloaded (`deforming_plate` would also be unsuitable: quasi-static, actuator-driven) |

Conversion factors: axes (x, y, z) Z-up -> (x, z, -y) Y-up; lengths x 0.36 (1 m cloth -> 0.36 m, sphere
radius 0.1 -> 0.036 m), times x 0.6 = sqrt(0.36) so gravity keeps its value (taken as ArcSim's 9.8 m/s^2
from the published configs: **not verifiable from the data**, there is no free-falling body); 20 Hz frames
by linear interpolation every 8.333 source frames (61 frames = 60 steps); scene lifted so the handles are at
y = 1.0 m (lowest cloth vertex 0.62 m: clear of the model's analytic floor). Mesh: farthest-point sampling
to 128 vertices in material space (handles always kept), coarse edges / faces from the adjacency of the
fine-mesh clusters (342 edges, 215 faces; rest-length error at frame 0: 7e-8). Two **pinned** vertices per
episode (`pinned` bool [E,V] in the shards; prescribed from the ground truth, like the rigid obstacles).
The sphere is **not** approximated by a box (the surface would be off by 19-40% of the radius): rigid slot
with a 10th `props` column = rounding radius, which `soft_gnn` evaluates as an exact sphere SDF. Check: the
ground truth scores 0.0 mm vertex penetration against that sphere (minimum vertex-surface gap over all
converted frames -0.18 mm). Masses and material parameters do not exist in the source: `soft_props` are
placeholders, identical for every episode.

Per-step statistics of the free vertices (20 Hz):

| data | step mean (mm) | median | p95 | max | steps < 0.5 mm | accel mean (m/s^2) | accel p95 |
|---|---|---|---|---|---|---|---|
| MeshGraphNets `sphere_simple`, converted train (80 episodes, 128 vertices) | 15.7 | 8.1 | 57.3 | 206 | 2.5% | 3.9 | 12.4 |
| `data/soft_v1` train (first 4 shards) | 6.4 | 0.25 | 40.9 | 239 | 59% | 1.25 | 9.7 |

Same order of magnitude in speed and acceleration, but the hanging cloth almost never rests (2.5% against
59% still steps), so it adds nothing for the rest-drift problem; what it adds is pinned cloth, a curved
moving obstacle and large swinging motion.

### Measured (4.3-minute single-GPU smokes on a GPU shared with a production run; NOT the final model)

Converted test split, 10 episodes, rollouts seeded with the true frames (0, 1), sphere and handles
prescribed, mean vertex error (mm):

| engine | 0.5 s | 1 s | 2 s | final (3 s) | strain err |
|---|---|---|---|---|---|
| hold_initial | 282.8 | 256.9 | 269.6 | 288.9 | 0.025 |
| rigid_freefall (meaningless here: no table, the cloth hangs) | 739.9 | 722.0 | 726.8 | 719.0 | 0.025 |
| soft_raw, smoke trained on soft_v1 only (800 updates) | - | - | 298.1 | - | 0.049 |
| soft_raw, smoke trained on soft_v2 + 5% MeshGraphNets windows (about 1100 updates) | 176.3 | 137.2 | 156.9 | 172.8 | 0.144 |

- Without MeshGraphNets windows the smoke model is **worse than holding the first frame** on this data
  (298 against 270 mm); with a 5% batch fraction it is 42% better than hold_initial at 2 s, but 157 mm on a
  0.36 m cloth is still a large error and the strain error is 14%.
- The same smoke model on `data/soft_v1` test: 117.3 mm at 2 s (soft_v1-only smoke: 118.7 mm) - no measurable
  harm from the mix at this training length; the two runs differ in update count and training set, so this
  is not a controlled comparison.
- 10 test episodes from the same upstream file as the training episodes: this is a weak generalisation
  estimate.

**Recommended mix fraction: 0.05** for `sphere_simple` (80 training episodes are about 4,700 windows against
about 500,000 in soft_v2, so 5% already repeats every window many times; more would overfit them);
**0 for the flag sets**.

### Limitations

- 80 training trajectories of ONE scene type from ONE upstream file; material parameters unknown.
- A 1087-node cloth is represented by 128 point-sampled vertices: wrinkles below about 3 cm are gone and the
  coarse edges can cut through the sphere (contact is vertex-only). Cloth self-collision in the source is
  not modelled.
- Gravity magnitude assumed, not verified; linear time interpolation lowers accelerations by about 0.2%.
