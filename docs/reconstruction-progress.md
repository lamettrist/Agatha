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
