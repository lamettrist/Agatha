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
