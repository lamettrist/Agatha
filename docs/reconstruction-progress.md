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
