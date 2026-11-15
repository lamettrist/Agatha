# Milestone evidence

## M1 — physics core
- `python -m pytest tests/test_physics.py -q`: **6 passed in 0.36s**.
- Offline demo push: mug fell and tipped; chaos score 2.642423355.
- Fixed 240 Hz timestep, isolated DIRECT clients, deterministic repeat test.

## M2 — photo reconstruction
- `python -m pytest tests -q`: **10 passed** (physics + reconstruction/API/persistence).
- Real photograph: YOLO detected laptop, potted plant, chair, keyboard; kept 2 supported desk objects.
- Real Depth Anything V2 Small + RANSAC ran on CPU in a 4.02 second reconstruction; support inlier ratio 0.5745.
- Depth/YOLO failure paths explicitly report fallback/unavailable status; no fabricated detections.
- Server needs headless OpenCV (GUI OpenCV failed because libGL is absent).
