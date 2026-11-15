# Metric reconstruction progress

## Milestone 1: geometry and physical bodies

- Added additive collision/provenance schema with legacy box defaults.
- Added metric backprojection, EXIF equivalent focal length, heuristic depth confidence,
  boundary removal, robust reference scale with outlier/disagreement reporting,
  RANSAC planes and image-region growing, gravity-aligned OBBs and cylinder RANSAC.
- Added explicit Y-up/Bullet conversion for primitive and fixed-link compound bodies.
- Added collision-bottom snapping, damped settling, actual closest-distance penetration
  diagnostics, and maximum drift across all 1,200 steps of a 5-second simulation.
- Validation: 15 geometry/physics tests passed, including a .21 m cylinder resting
  stably and tipping when pushed. Existing bundled desk JSON is unchanged.
- Concurrent contact-learning edits in physics.py are preserved and excluded from
  this milestone commit.
- Acceptance fixture absent at both requested paths as of initial inspection;
  no fixture measurements claimed.

## Milestone 2: working photo API and viewer, accuracy still failing

- Real acceptance photo: `tests/images/acceptance.jpg` (4000×2252), resized to 1280×721.
- Depth Anything V2 Metric Indoor Small + Grounding DINO Tiny + SAM 2.1 Small + CLIP.
- Photo API now saves per-body collision/provenance and real masks, metric depth array,
  point PLY, textured GLBs, original image and calibrated triangle reprojection bundle.
- Viewer supports original/reprojection/3D switching, inspect-or-push interaction,
  uncertain-object coloring, rigid-approximation labels and persistent mass/shape editing.
- Before saving: logged minimal horizontal overlap correction, collision-bottom snap,
  damped settle, and explicit inferred support extension for cropped edges. No invented stacks.
- Real fixture test: **11.7624 seconds warm**, **mean silhouette IoU 0.35278**,
  **0 floating**, **0 penetrations >5 mm**, **5 s max drift 0.00020617 m**,
  **left black bottle tips with 0.55 Ns push**.
- **Acceptance fails**: one laptop and gloves missing; IoU below .75. Metric dimensions
  have not been independently measured; class references are not treated as ground truth.
- Exact report: `docs/desk-reconstruction-report.json`. Run the opt-in acceptance test with
  `RUN_RECONSTRUCTION_FIXTURE=1 .venv-reconstruction/bin/python -m pytest tests/test_reconstruction_fixture.py -q -s`.
- Focused geometry/physics/legacy/API suite: **23 passed**. The fixture test fails its
  actual acceptance assertion; no mocks or fabricated pass metrics are substituted.


## Video backend repair: real HTTP validation

- Both `/api/scenes` and `/api/scenes/video` accept multipart `file`, `image`, or `video`; filename routing handles octet-stream MP4 uploads.
- Default CORS permits `https://precognition.base44.app`; `/api/health` documents the response field `scene.objects`. Empty video results return a useful error/report.
- Removed VGGT padding with matching intrinsics, retained fused world positions, separated table regions, and preserved floor/secondary-table supports.
- SAM 2 propagation runs on seven GPU workers. Checkpoint initialization is serialized to avoid meta tensors; object-ID lists are copied before the processor consumes pending inputs.
- Empty or degraded tracking runs cannot populate a healthy cache. Completed runs save reloadable scene/report/debug assets; cached request time is separate from original reconstruction time.
- Real `tests/images/deskspace.mp4`: **29 candidate bodies, 30 sharp frames, 23 tracks, 690 observations, no tracking fallbacks**. Actual multipart HTTP response: **92.51 s**; processing **92.18 s**. Repeat upload **0.45 s** with a cache hit.
- Scene readback matched the uploaded result. Actual bundled desk load + physics simulation returned HTTP 200 and six frames. `scenes/demo_desk.json` is unchanged.
- Final full suite: **238 passed, 1 opt-in fixture skipped**. Exact output: `docs/video-tests.txt`. Focused tests also cover world-to-body mesh conversion, independent track identities, sequential sharp-frame sampling, and table-bound preservation.
- **Accuracy still fails**: silhouette IoU **0.08731**, **7 declared-support gaps**, **0 penetrations >5 mm**, five-second maximum drift **0.02583 m**. The API explicitly returns `quality_status=needs_review`; no accuracy pass is claimed.
- Mesh fallback: the detected backpack (`object_005`) kept its completed box because the generated mesh had a worse projected silhouette. Other class-template completions and sparse-depth fallback reasons are listed per object in the full report.
- Main remaining accuracy problems: incorrect/fragmented instance labels; support assignment for objects on secondary tables; incomplete or noisy geometry from sparse/occluded observations. Metric sizes have not been independently measured.
- Reports: `docs/video-reconstruction-report.json`, `docs/video-http-validation.json`. The fixture profile ran while other GPU training work was active.


## 2026-11-15 Video reconstruction repair (round 1)

Input: `tests/images/deskspace.mp4` (primary) and `tests/images/acceptance.jpg` (must not regress).
All numbers below were measured on this machine on 2026-11-15; "before" is a refit of the same cached
neural outputs with the code as it stood at the start of the day (23 bodies; the earlier report in this
file, 24 bodies / IoU 0.087, was from an older build).

### What was wrong
- **Support**: the table slab was the camera-0 axis-aligned 1-99 % box of frame-0 table pixels. The table is
  yawed about 59 deg to the first view and `Support` has no yaw, so the slab was 1.41 x 1.23 m, overhung the
  stools and missed the far end. Gravity, table height and scale were *not* wrong (checked independently).
- **Fusion**: `fuse_track_points` kept every frame of a SAM 2 track inside a loose box (margin in VGGT units,
  about 0.5 m metric). Tracks that drifted onto table legs, the next stool or a person's head, plus 0.1-0.2 m
  ghosting between VGGT chunks, turned stools into 1-2 m slabs and doubled bottle diameters.
- **No evidence gate**: every frame-0 detection became a body, including 13 objects 4-7 m away whose pixels
  sit at VGGT's confidence floor, three automatic SAM masks on the pillar/baseboard, a "can" on a wall.
- **Supports**: a body outside every footprint kept `supported_by='support'` and was teleported to table
  height by settle (the "floaters"). Secondary tables were built from a handful of far points.
- **Settle**: overlapping oversized chairs were ejected by the solver (2.5 m "drift") and shoved bottles
  by up to 0.5 m; the saved poses were settle artefacts, not measurements.
- **GPU assumptions**: default device list `0..6`, VGGT on `device(4)`, 7 tracking workers. VGGT-1B needed
  more than 9 GB and the pipeline silently fell back to COLMAP + monocular depth (still `status=complete`).
- `physics.summarize` reported every floor-standing body as `fell` at t=0.

### What changed
- `structure.table_rectangle`: minimum-area rectangle of the table top from all frames (occupancy grid of
  near-plane points seen in >= 5 frames, joined with the frame-0 table mask). The video world frame is rotated
  about gravity so world X is the table's long axis; the slab is that rectangle. Photo path untouched.
- `structure.fuse_views`: per-track view gate (a view must agree with the consensus position/extent of the
  first views) followed by voxel voting across kept views (1.5 cm for small objects, 4 cm for furniture),
  done in metric world coordinates. Each body reports `multiview: views_observed / views_kept / points`.
- `video.select_trackable`: automatic SAM masks and detections with median VGGT confidence < 1.1
  (`RECON_VIDEO_MIN_CONF`) are not tracked. Applied before SAM 2 (saves tracking time: 22 -> 6 tracks) and
  again in `finalize_video`. Dropped items are listed in `video.stages.reference_selection.dropped`.
- `pipeline.lift_multiview`: class-aware bodies. Bottles/cups/cans are upright cylinders from robust extents
  (diameter clamped per class, base on the support). Chairs/stools are upright, on the floor, footprint
  clamped to 0.35-0.70 m, built as base disc + stem + seat + narrow back, yaw chosen from 8 candidates by
  frame-0 silhouette. Bodies outside `VIDEO_LIMITS` are **rejected, not clamped**; also bodies with fewer
  than 3 consistent views or 150 points, or beyond 3 m (`RECON_VIDEO_MAX_RANGE_M`) of the table centre.
  Reasons are in `stages.lift.dropped`. Video bodies carry the canonical class as their label.
- `assign_video_supports`: a body is on a table only if it is inside the footprint with its bottom near the
  top; otherwise on the floor if its bottom is below 0.25 m; otherwise dropped (`no_support_under_body`).
  Secondary table slabs need >= 0.3 m2. Wall slabs need one dominant plane clear of the table.
- `bodies.settle(measured_poses=True)` (video): floor bodies that actually penetrate static geometry are slid
  out horizontally (recorded in `static_overlap_corrections_m`); after the damped settle any body moved more
  than 3 cm or 5 deg is restored to its measured, snapped pose (`reverted_to_measured_pose`).
- Stools/chairs are settled and drift-tested as dynamic bodies, then **published static**
  (`RECON_VIDEO_STATIC_FURNITURE=1`, default). Reason: the learned world model drifted a dynamic 1.1 m stool
  2-5 cm with no action (outside its 0.02-0.35 m training range); with static stools it shows 0.00.
- Scale: when no plastic-bottle reference is found on VGGT geometry the scale now comes from a 1.45 m
  hand-held camera height above the fitted floor (confidence `low`, warning) instead of factor 1.0. On this
  video that gives 2.453 against 2.469 from the bottle (checked by disabling the bottle reference).
- GPUs: `RECON_GPU_IDS` defaults to the visible devices and ignores ids that do not exist;
  `RECON_TRACK_WORKERS` defaults to that count; `RECON_GPU_MEMORY_GB` caps the process. VGGT loads without
  its unused point/track heads and with an fp16 aggregator (`RECON_VGGT_HALF=1`; depth changes by a median
  0.03 % because the aggregator already ran under fp16 autocast): 10-frame chunks peak at 5.65 GB instead of
  failing at 9 GB. A GPU OOM now retries with 8/6/4-frame chunks and warns, instead of silently using
  COLMAP. With one GPU the VGGT weights are released before the other models load.
- `physics.summarize`: bodies declared `supported_by='floor'` are never counted as `fell`.
- `multiview_silhouette` metric: IoU of final bodies against SAM 2 track masks in up to 8 frames with each
  frame's pose and the slab as occluder, only over frames the fusion kept.
- `.venv` now has scipy + trimesh, and they are in the `test` extra in `pyproject.toml` (the reconstruction
  unit tests import them).

### Measured, video (live run, one shared GPU, 9 GB cap)
| | before | now |
|---|---|---|
| bodies (dynamic) | 23 (19) | 5 (2 bottles dynamic, 3 stools static) |
| support | 1.41 x 1.23 m at 1.022 m | 1.65 x 0.90 m at 1.022 m, aligned to the table |
| black bottle | 0.12 dia x 0.21 m, IoU f0 0.61 | 0.06 x 0.23 m, IoU f0 0.84, 7-frame mean 0.74 |
| water bottle | 0.15 dia x 0.17 m, IoU f0 0.00, 16 cm from the other | 0.07 x 0.18 m, IoU f0 0.78, 5-frame mean 0.77, 0.37 m from the other |
| stool o002 | 0.60 x 1.09 x 0.58, ejected 2.5 m | 0.56 x 1.16 x 0.53, IoU f0 0.59 |
| stool o003 | 1.16 x 1.17 x 0.67 slab | 0.55 x 1.18 x 0.47, IoU f0 0.55 |
| stool o004 | 1.04 x 1.08 x 0.60 | 0.50 x 1.11 x 0.52, IoU f0 0.13 |
| largest dynamic extent | 2.16 m | 0.23 m (largest body 1.18 m, a stool) |
| floating / penetrations | 6 / 5 | 0 / 0 |
| 5 s max drift | 2553 mm | 0.2 mm |
| mean silhouette IoU, frame 0 | 0.064 | 0.580 |
| quality_status | needs_review | needs_review (reprojection gate 0.75 not met; contacts, stability, tracking pass) |
| wall time | 92 s (7 GPUs, earlier session) | 90-92 s on one busy GPU, peak 5.65 GB |

Push check on the live scene (PyBullet): no action -> nothing falls or tips; 0.3 N s on a bottle tips it;
1.0 N s towards the nearest edge puts it on the floor (`fell`); no other body moves.

Photo (`acceptance.jpg`, live): unchanged from before: 19 bodies, IoU 0.435, 0 floating, 0 penetrations,
8.3 mm drift, 40 s. Reports: `docs/video-reconstruction-report.json`, `docs/desk-reconstruction-report.json`.

Thresholds: no quality gate was loosened. The reprojection gate stays at mean frame-0 IoU >= 0.75 and still
fails (0.58); stool masks fill only 12-43 % of their bounding box, so a primitive stool cannot reach it.

### Still wrong
- Stools are class completions inside a measured envelope: seat/back proportions are generic, o004 (seen
  mostly through the table legs) has IoU 0.13-0.24 and its yaw is a guess.
- In the overlays the slab visibly overshoots one end of the real table in some views (not measured); VGGT geometry is only
  consistent with frame 0 to a few cm / degrees in later frames and nothing corrects that drift.
- Everything visible only after frame 0 (the standing person, the second room side) is not reconstructed;
  background tables, chairs and laptops are deliberately dropped, not modelled.
- Thresholds (confidence 1.1, 3 views / 150 points, class limits, 3 m range) were tuned on this one video.
- Scale still rests on one bottle silhouette (or the camera-height prior); no cross-check gate.
- The COLMAP fallback path was not re-validated with the new filters (it has no confidence channel).
- Photo path defects are unchanged: EXIF focal length is never found (always 60 deg FOV), a false "can",
  a table leg labelled "tumbler", backpack o012 placed away from its mask.
- The learned world model treats the static stools as bounding boxes and disagrees with PyBullet on where a
  pushed bottle ends up; that is outside the reconstruction code.
- Open3D needs `libGL.so.1` on the loader path (`LD_LIBRARY_PATH`), otherwise TSDF fusion and soft-body
  meshes fall back silently.

### Tools
`scripts/recon_refit.py` replays every deterministic stage from cached neural outputs in about 10 s and writes
per-frame overlays (`scripts/recon_overlay.py`). Opt-in real-data regression:
`RECON_REAL_RUN=<captured run> .venv/bin/python -m pytest tests/test_video_accuracy.py -k real_video`.

## 2026-11-15 Video reconstruction repair (round 2, after two independent reviews)

Same inputs. "Round 1" is a refit of the round 1 live run with round 1 code; "now" is the live pipeline
(`reconstruct_video`, one GPU, `RECON_GPU_MEMORY_GB=9`), which a refit of its own bundle reproduces exactly.

### What was wrong (review findings, each confirmed before fixing)
- The support slab (1.65 x 0.90 m) was about 10 % larger than the real top: it was the union of every view's
  table points, and VGGT poses disagree by centimetres between views, so the union overhangs on all sides.
- Stool yaw was chosen on frame 0 only. The stool behind the table is seen there through the table legs, so
  its yaw was a guess (best candidate IoU 0.12) and looked 45-90 degrees off in later frames.
- Fixed-camera clips (`tests/images/Example_*.mp4`, a hand moving objects in front of a tripod) returned
  HTTP 422 with no bodies: VGGT's confidence is at its floor everywhere (0-3 % of pixels above 1.1, against
  26-49 % for the walk-around), so the per-object confidence gate dropped every detection.
- On a warm one-GPU server every video after the first ran out of memory at 10-frame VGGT chunks and
  silently degraded to 4 or 6 (scale 2.30 instead of 2.47), because the other models stayed on the GPU.
- A degraded run was cached as complete; several settings were missing from the cache key.
- CPU-only did not fail (no result in 500 s). Static stools ignored pushes without saying so. Photo reports
  had no `quality_status`, and a photo with no table produced a scene with no warning.

### What changed
- `structure.trim_rectangle` (called by `table_rectangle`): keep cells seen at least half as often as the
  typical interior cell, then put each edge where half of the cross-section is still occupied.
- `pipeline.orient_chairs`: 8 yaws (and, with 3+ frames, a +-8 cm shift that must beat the measured position
  by 0.01 IoU) are ranked by mean silhouette IoU over up to 10 frames in which the track is still on the
  body (`consistent_frames`, 0.2 m), with the table top and the other tracked objects as occluders.
  `multiview_silhouettes` uses the same frame selection. `pipeline.frame_view` is the shared per-frame pose.
- `video.geometry_consistency` + `video.photo_fallback`: when under 10 % of pixels have consistent multi-view
  geometry (`RECON_VIDEO_MIN_CONSISTENT`), or the multi-view lift yields no movable body, the first sharp
  frame is reconstructed as a photo, bodies below the floor or metres from the table are removed, and the
  report says so (`pipeline=video_representative_photo_fallback`, `video_fallback`, `quality_status=needs_review`).
- `video.make_room_for_geometry` + `Registry.park`: on one GPU the tracker and TripoSR are dropped and the
  detector/SAM/depth/CLIP weights are parked on the CPU while VGGT runs, then moved back.
- Cache: the key now includes `RECON_VIDEO_MIN_CONF`, `RECON_VIDEO_MAX_RANGE_M`, `RECON_VIDEO_STATIC_FURNITURE`,
  `RECON_VGGT_HALF`, `RECON_REFINE`; a cached run whose VGGT chunk differs from the configured one, or that
  used COLMAP, is redone.
- `reconstruct_video` raises immediately without CUDA unless `RECON_ALLOW_CPU=1`.
- A warning names the furniture that is published static. Photo reports carry `quality_checks`,
  `quality_status`, `status`, and warn when no table was detected or mean IoU is under 0.2.
- No quality threshold was changed. The 0.75 frame-0 reprojection gate still fails (0.565), so the video is
  `needs_review`.

### Measured, video
| | start of day | round 1 | now (live) |
|---|---|---|---|
| Bodies (dynamic) | 23 (19) | 5 (2) | 5 (2 bottles; 3 stools published static) |
| Support | 1.41 x 1.23 m at 1.022 m, misaligned | 1.65 x 0.90 m | 1.55 x 0.82 m at 1.022 m |
| Slab vs wood-colour mask, 30 frames (reviewer's script; its best fit scores 0.68) | not measured | 0.621 | 0.670 |
| Black bottle o000: size, IoU frame 0 / all frames | 0.12 x 0.21 m, 0.61 / - | 0.064 x 0.234, 0.84 / 0.73 | 0.064 x 0.234, 0.81 / 0.71 |
| Water bottle o001 | 0.15 x 0.17 m, 0.00 / - | 0.066 x 0.180, 0.78 / 0.62 | 0.069 x 0.177, 0.76 / 0.64 |
| Stool o002 | 0.60 x 1.09 x 0.58, thrown 2.5 m | 0.56 x 1.16 x 0.53, 0.59 / 0.54 | 0.56 x 1.16 x 0.53, 0.59 / 0.55 |
| Stool o003 | 1.16 x 1.17 x 0.67 slab | 0.55 x 1.18 x 0.47, 0.55 / 0.21 | 0.54 x 1.18 x 0.47, 0.54 / 0.22 |
| Stool o004 | 1.04 x 1.08 x 0.60 | 0.50 x 1.11 x 0.52, 0.13 / 0.40 | 0.52 x 1.11 x 0.48, 0.12 / 0.37 |
| o004 best yaw score (frames scored) | - | 0.12 (1) | 0.37 (10) |
| Mean IoU frame 0 / 8-frame metric in the report | 0.064 / - | 0.580 / - | 0.565 / 0.533 |
| Floating / penetrations | 6 / 5 | 0 / 0 | 0 / 0 |
| Largest body / largest dynamic | 2.16 m | 1.18 / 0.23 m | 1.18 / 0.23 m |
| 5 s max drift | 2553 mm | 0.2 mm | 0.1 mm |
| Runtime | 92 s (7 GPUs) | 90 s | 56-61 s, peak 5.76 GB |

"All frames" is the harness mean over every frame with a track mask, including frames where SAM 2 has left
the object, which is why o003 stays low. The o004 all-frame mean went down slightly while its yaw now agrees
with the views in which it is clearly seen (frames 12 and 23 in the overlays); the per-frame number is
dominated by occlusion by the table and by the person, neither of which is modelled.

- Warm server (one process, one GPU, 9 GB cap): photo, video, Example_4, video again. Both video runs used
  10-frame chunks, gave identical scenes, and took 61 s and 58 s.
- PyBullet on the live scene: nothing moves with no action; 0.3 N s tips the black bottle; 1.0 N s across the
  table puts the black bottle on the floor in both directions and the water bottle in one (it tips and stays on
  in the other); 1.0 N s along the table tips a bottle and it stays on. No other body moves.
- Learned `default` model: zero displacement with no action; it still keeps a bottle on the table in a case
  where PyBullet has it fall off the edge (world-model issue, not changed here).
- Example_1, Example_3, Example_4 now return a scene through the photo fallback in 43-51 s. Those scenes are
  photo-path quality and contain false bodies (see below).

### Measured, photo (`acceptance.jpg`, live)
Unchanged from the baseline: 19 bodies, IoU 0.435, 0 floating, 0 penetrations, 8.3 mm drift, 35 s. Body positions
and sizes are identical to the baseline refit (max difference 0.0). New: `quality_status=needs_review`.

### Still wrong
- Fixed-camera clips get a single-frame scene only, with the photo path's defects: on Example_1 the one
  real bottle comes out 0.14 x 0.56 m beside false "can", "chair" and "backpack" bodies; Example_4 has both
  bottles at plausible sizes plus false "monitor", "keyboard", "backpack" and a 1.3 m "chair". Nothing in
  those clips is tracked over time.
- Stools remain class completions; o003 and o004 are right in position and envelope but only loosely match
  their silhouettes, and are static by default.
- Late frames show the bodies a few centimetres off their masks (VGGT pose drift, uncorrected), so overlays
  on frames 17-29 look worse than frames 0-12.
- The slab is still a few centimetres wide in some views; the comparison uses a colour mask that is
  itself approximate.
- Both bottles get the class mass (0.55 kg). Scale still rests on one bottle silhouette or the camera prior.
- Thresholds (0.10 consistent fraction, 0.2 m frame gate, half-maximum trim) come from this one walk-around.
- Photo path: EXIF focal length, false "can", table leg as "tumbler", backpack o012 off its mask and
  reported "tipped" at rest, a small false wall slab. None of these was touched.
- The COLMAP path is still unvalidated. Open3D still needs `libGL.so.1` on `LD_LIBRARY_PATH`.
- The HTTP 422 for an empty video still carries the whole report as `detail`, which the frontend prints raw.
