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

## M5 — API and live demo polish
- Final `.venv/bin/python -m pytest tests -q`: **20 passed in 2.91s**. Exact output: `docs/test-output.txt`.
- Actual HTTP `bash scripts/demo.sh` on port **8017**: 91 frames per engine, both mug fall/tip, physics chaos **2.642423355**, final position error **0.12037 m**. Exact output: `docs/demo-output.txt`.
- Chromium WebGL QA passed: initial scene, actual raycast click-to-push, playback, physics/learned toggle, leaderboard submit, missing-key assistant response, real YOLO/depth photo upload + detection overlay + both-engine simulation, mobile width 390px with no horizontal overflow, no browser external/CDN requests, no JavaScript errors.
- Real photo upload: **4 detections, 2 kept, relative_depth_ransac**, medium scale confidence. Both engines ran on the reconstructed scene.
- Actual endpoint request/response excerpts captured in `docs/api-examples.json`; rendered screenshot in `docs/demo.png`.
- Additional agent test confirms a verified mass changes the physical outcome and does not mutate the saved scene.
- Editable package installation, Python compilation, shell syntax, and `git diff --check` passed.
- Default port is 8017 because another service already occupies port 8000 in this workspace.
- Runtime stubs: **none**. Explicit fallback paths: depth → flat table; unavailable detector → zero fabricated detections; missing learned checkpoint → physics with warning; unavailable OpenAI/Exa → unavailable/estimated response.
- Remaining verification gap: live authenticated OpenAI/Exa calls require the user's `.env` keys. Base44 itself is external; its integration contract and exact endpoints are documented, but the Base44 app was not edited or deployed.

## Expanded training and graph/MLP A/B (user-requested follow-up)
- Attached `origin` to `https://github.com/lamettrist/agatha.git`; read access verified. No push requested or performed.
- Read the user-provided `NOTES.md` proposal; implemented an actual three-message-round graph impulse network and a matched multi-step MLP.
- Generated 5,000 base scenes / 300,000 scene transitions / 1,172,100 object transitions in 401.73 seconds. Added 1,000 launch/noclip episodes in 38.98 seconds; shared A/B dataset totals 6,000 episodes / 360,000 scene transitions (5,400 train / 600 validation).
- Both A/B models trained from scratch on identical data splits, sampled windows, three-step objective, input noise and optimizer/schedule. MLP: 79,373 parameters, 90.33 seconds. Graph: 40,646 parameters, 341.31 seconds. Different capacities; not a parameter-matched study.
- Same 200 ordinary held-out scenes / 763 objects / seed 20261115: original 500-scene MLP 2s error **0.26646 m**; expanded one-step MLP **0.16992 m**; multi-step MLP **0.15039 m**; graph **0.46338 m**. MLP fall agreement **94.76%**; graph **84.14%**. Selected multi-step MLP reduces 2s error **43.6%** relative to the original.
- Additional 250-case paired stress benchmark: layouts, bottle tosses, disabled table collisions, toss+disabled table, stacks and nine objects. MLP lower 2s position error in all six suites. Full precision/recall, clipping diagnostics, stability and latency saved in `models/ab_results.json`.
- **Clipping is not solved.** Solid-layout diagnostic flags MLP 8 crossings, graph 60, PyBullet 1; geometric metric is approximate. Intentional table disabling is separately labeled and excluded from penetration-error statistics.
- Both candidates finite/bounded by the benchmark criterion on 250/250 cases. Graph raw pair impulses satisfy antisymmetry; permutation equivariance and no physics stepping tested. Full angular momentum conservation is not claimed.
- Added backwards-compatible 3D pushes, explicit `set_support_collisions`, bottle demo, selectable learned architecture, and visible ghost table. Four agent tools remain unchanged.
- Chromium: actual bottle peak physics **1.311 m**, selected MLP **1.302 m**; intentionally disabled-table toss predicts bottle falling in both engines. Original desk/photo flows, architecture selector, overlay, playback, leaderboard and mobile checks passed. Exact output `docs/browser-output.txt`.
- Final tests: **29 passed in 3.44s** (`docs/test-output.txt`). Actual HTTP desk demo: both mug fall/tip, final mean position error **0.07805 m** (`docs/demo-output.txt`). Python compilation and `git diff --check` passed.
- Report and plots: `docs/world-model-ab.md`, `docs/world-model-ab.png`, `docs/world-model-ab.svg`; screenshot `docs/bottle-noclip.png`. Both trained candidate checkpoints and the selected default are shipped.
- User-provided `NOTES.md` remains unchanged. Live authenticated OpenAI/Exa verification still requires keys.
- Documented retraining command smoke-tested with one epoch in `/tmp/agatha-trainer-smoke/`: loadable checkpoint plus fresh 200-scene `metrics.json` generated successfully. Production weights were not touched by this smoke run.
