# Part C - external physics data (approximate evaluation)

> **Everything on this page is APPROXIMATE.** The external data was converted with the approximations
> listed in section 4. All three sources are *simulators* from other projects, not real-world video.
> This is an out-of-distribution sanity check of the learned world model, **not a real-world validation**.
> No model was trained or fine-tuned on this data.

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

## 2. Data and licences

| Dataset | Simulator | Subset on disk | Licence (see `DATA_LICENSES.md` for caveats) | Used for |
|---|---|---|---|---|
| Kubric MOVi-A 128x128 | PyBullet | 1137 episodes (train shards 0-59 of 512), 0.54 GB | Apache-2.0 (repository-level; no dedicated data licence) | eval only |
| CLEVRER validation annotations | Blender / Bullet | 5000 annotation files (first 1000 converted), 0.72 GB | CC0 (explicit, official README) | eval only |
| Physion (NeurIPS 2021) | TDW / PhysX | 10 trials (8 Dominoes, 2 Support), 1.8 GB | MIT (repository-level; no dedicated data licence) | eval only (anecdotal) |
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

- Simulator data only; conversions are lossy; CLEVRER physical parameters are assumed or calibrated.
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
