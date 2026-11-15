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

## M3 — genuine learned dynamics
- Generated 30,000 scene transitions / 114,840 object transitions from 500 randomized scenes in 39.56 seconds on CPU.
- Trained 79,181 parameter MLP in 17.65 seconds; checkpoint `models/world_model.pt`.
- 50 independent held-out scenes, 196 objects: 0.5s **0.07144 m**, 1s **0.12062 m**, 2s **0.23266 m** mean position error; fall agreement **93.37%** (18 physics falls, 21 learned falls).
- Free-flight position/orientation residual; next linear/angular velocity learned directly. No physics stepping in learned rollout (tested).
- `python -m pytest tests -q`: **13 passed in 2.27s**.
- Offline desk push: both predict mug fall; final mean position error across objects at 3s **0.12037 m**.

## M4 — OpenAI + Exa agent
- `python -m pytest tests -q`: **16 passed in 2.43s**.
- Exactly four callable tools. Integration test drives actual physics + learned tools through mocked OpenAI tool selections; final prose ignores unsupported LLM outcome claims.
- Exa extraction test checks source membership, exact source evidence, numerical mass/unit agreement, and disk cache; fabricated URL and mismatched mass rejected.
- Without keys: assistant HTTP 200 `status=unavailable`; both-engine simulation HTTP 200 and working.
- **Live authenticated OpenAI/Exa calls unverified: no keys supplied.** SDK interfaces inspected against installed packages and official docs. No fake network results are shipped.
