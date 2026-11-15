# Agatha / Foresight

Photo → approximate 3D boxes → poke the scene → compare PyBullet with a genuine small learned world model → ask an OpenAI agent grounded in Exa product specs.

Built for a 3-minute Hack the Nest demo. No auth, database, scene history, or cloud service is needed for the offline desk. All dimensions, positions and unsourced masses are **estimates**; predictions are **not guaranteed real-world outcomes**.

## Run in three commands

Python 3.11–3.12 recommended, with a C++ compiler if PyBullet has no wheel for your Python. Run from this checkout:

```bash
bash scripts/setup.sh
.venv/bin/python scripts/prepare_vision.py --depth
.venv/bin/uvicorn agatha.main:app --host 0.0.0.0 --port 8017
```

Open **http://127.0.0.1:8017/**. Interactive API docs: **http://127.0.0.1:8017/docs**. Skip the second command for the entirely offline desk. Setup creates `.env` from `.env.example` only when absent, installs CPU PyTorch, and replaces GUI OpenCV with headless OpenCV (no libGL required).

For the agent, put `OPENAI_API_KEY` and `EXA_API_KEY` in `.env`, then restart. Never put keys in Base44/browser code. `OPENAI_MODEL` defaults to `gpt-4.1-mini` and is configurable. Missing credentials or failed services produce explicit unavailable/estimated results; both simulation engines remain independent.

Run the required terminal demo in another shell:

```bash
bash scripts/demo.sh
```

It calls the running API, loads the desk, pushes the mug by **0.55 Ns toward +X**, runs both engines, and prints actual fall/tip events, scores and errors. Override the port/host with `API_BASE=http://127.0.0.1:8017 bash scripts/demo.sh`.

## Three-minute demo

1. Load the offline desk. Explain that these are approximate boxes with estimated masses.
2. Click the mug, or press **Run both engines**. Both predict the mug falling. Scrub/replay, show side-by-side frames and position error. The learned model really predicts each next state without running physics.
3. Upload a prepared desk photo with a table width reference. Show real YOLO detections and the reconstruction report. Depth unavailable is visibly labeled as flat-table fallback.
4. With API keys configured, ask: **“What if I push the mug? Look up its mass and compare both engines.”** Show the representative product mass, Exa URL, actual actions, and tool outcomes. Published specs are representative: a single photo does not establish product identity.
5. Submit a physics chaos score if time permits.

The UI bundles three.js and OrbitControls locally, including their license. No CDN is used. Use WebGL-capable Chrome/Firefox. Photo models should be downloaded/warmed before presenting; only the desk and trained world model are shipped for offline use.

## What runs

- Deterministic PyBullet DIRECT simulation, fixed **240 Hz**, sampled at 10/20/30 fps, up to 5 seconds. One isolated client per simulation. Actions execute **once at t=0 in list order**; every run starts from the saved scene.
- `push`, `remove`, `set_mass`, `tilt_support`, `set_gravity`, `set_support_collisions`. Push directions may be horizontal `[x,z]` or full Y-up `[x,y,z]`, so `[0,1,0]` tosses upward. Table collision disabling is an explicit counterfactual, shown as a ghost table. Static objects ignore pushes/mass dynamics. `direction` is normalized, so impulse magnitude is independent of direction-vector length. A push is a center-of-mass impulse, not an arbitrary hit point.
- Scene objects are boxes with ID, label, dimensions, center position, XYZ Euler rotation, kg mass, `mass_source`, friction, static flag and detection confidence. Support `size` is `[width, depth]`; `height` is its top surface; `collisions_enabled` defaults to true. A floor at Y=0 catches fallen objects. The demo's static `desk` box visually represents the support.
- API uses **meters, kg, seconds, radians, right-handed Y-up**, quaternions **[x,y,z,w]**. Bullet conversion is the proper rotation `(x,y,z) → (x,-z,y)`; quaternion vector part maps identically. Box sizes map to Bullet half extents `[w/2,d/2,h/2]`. Unit tests check vectors, quaternions, handedness, and rotated vectors.
- Events: first sampled contact per pair, COM outside support bounds and below its top by 3 cm = “fell”; local Y axis tilted >35° from world up = “tipped.” Removed objects do not count as falls. With table collisions deliberately disabled, a COM dropping below the support top also counts as falling even while within its footprint. For tilted support, falls use support-local coordinates. Contacts are sampled at every 240 Hz step, timestamps rounded to the output interval.
- Chaos score = **count of physics objects that fell + sum of final center displacements (m)**, excluding static/removed objects. Leaderboard accepts only the latest server-computed physics score for that scene.
- In-memory scene cache + atomic JSON files in `DATA_DIR` (default `data/`), persistent leaderboard and property cache. **Run one API worker** for this demo.

## Photo reconstruction and honesty

YOLOv8n returns real label/bbox/confidence detections. Supported reconstructed classes: laptop, cell phone, cup, bottle, book, bowl, keyboard, mouse (box prior also available). Unsupported labels remain visible in the detection report and are not silently turned into known objects.

Depth Anything V2 Small through Transformers produces **relative inverse depth**. The code maps that to approximate distances, back-projects with an assumed 60° pinhole camera, masks detected objects, and RANSAC-fits a dominant support plane. It projects detections into that plane, scales bbox extents with relative depth, clamps sizes to class ranges, and puts box bottoms on the support. Overlapping footprints are stacked conservatively; this can introduce incorrect stacks. Thin-object thicknesses use class priors. Support depth/height and all rotations are estimates. No metric-depth or camera-calibration claim is made.

Optional `table_width_m` anchors the scale. Otherwise a detected phone (~0.15 m long) or laptop (~0.33 m wide) supplies an assumed reference; absent those, support width defaults to 1.2 m with **low** scale confidence. A supplied reference is **medium**, since perspective and boxes remain approximate. No EXIF intrinsics are currently used. Images are EXIF-oriented and resized to at most 1280 px; detection bboxes are in `report.image_size` coordinates.

If depth or plane fitting fails, the report says **`flat_table_fallback`** and uses bbox-only placement. If YOLO itself is unavailable, it returns zero fabricated objects and an explicit warning; load the offline demo instead. Network weights are optional, not bundled. `VISION_OFFLINE=1` forces this unavailable-model path.

## Learned model: actual training and A/B results

The selected [checkpoint](models/world_model.pt) is our **79,373-parameter PyTorch MLP**, trained from scratch. It sees object state, mass/friction/size, support geometry/gravity, three nearby objects, and an explicit table-collision bit. It predicts pose corrections and next velocities at **20 Hz**, autoregressing on its own states without stepping PyBullet. Position/orientation use a free-flight residual baseline; output at 10/30 fps is interpolated.

We expanded from 500 scenes to **5,000 randomized layouts / 300,000 scene transitions / 1,172,100 object transitions**, then added **1,000 bottle-toss and disabled-table episodes**. The shared final dataset contains **6,000 episodes / 360,000 scene transitions**, split into 5,400 training and 600 validation scenes. The final MLP uses three-step autoregressive training and input noise; training took **90.33 seconds** on four CPU threads. No pretrained dynamics weights are used.

We also implemented and trained the `NOTES.md` proposal: a **40,646-parameter, three-message-round graph impulse model**, including explicit table/floor nodes and antisymmetric pair impulses. Both candidates were trained from scratch with identical data, splits, sampled windows, loss, noise and optimizer/schedule. The graph took **341.31 seconds**. This graph candidate did **not** beat the MLP, so the MLP remains default. Both are available through the existing simulation endpoint and UI for inspection.

Paired ordinary-scene evaluation on **200 held-out scenes / 763 dynamic objects**, seed 20261115:

| Model | 0.5 s error | 1 s error | 2 s error | Fall agreement |
| --- | ---: | ---: | ---: | ---: |
| Original 500-scene MLP | 0.073 m | 0.157 m | 0.266 m | 90.7% |
| Larger 5,000-scene one-step MLP | 0.059 m | 0.109 m | 0.170 m | 93.8% |
| Selected 6,000-scene multi-step MLP | 0.063 m | 0.101 m | 0.150 m | 94.8% |
| Matched graph impulse candidate | 0.113 m | 0.248 m | 0.463 m | 84.1% |

The selected model cuts two-second error by **43.6%** versus the original on the same test. The older 50-scene evaluation is retained under `legacy_50_scene_evaluation` in [models/metrics.json](models/metrics.json). These are synthetic development-held-out measurements, not blind real-world validation.

A separate **250-case stress benchmark** covers layouts, upward bottle tosses, deliberately disabled table collisions, combined toss/noclip, six-object stacks and nine-object stacks. The MLP wins the position-error comparison in every suite. **Clipping is not solved**: learned objects can still pass through a solid table. Disabling the table intentionally is separately labeled and not counted as penetration error. Fall precision/recall, geometric penetration diagnostics, full crossings, stability and latency are reported in the [A/B report](docs/world-model-ab.md), [raw results](models/ab_results.json), and [plot](docs/world-model-ab.png).

To see the requested counterfactual: **Load bottle experiment**, choose **A / Multi-step MLP** or **B / Graph impulses**, then run **Toss upward**. Uncheck **Solid table collisions** and run again. Bottles remain box approximations; there is no water/fluid simulation. Every run restarts from the original scene.

Regenerate the enlarged data and selected model (no API keys):

```bash
.venv/bin/python scripts/generate_data.py --scenes 5000 --seed 2026
.venv/bin/python scripts/prepare_ab_data.py --extra 1000
.venv/bin/python scripts/train_ab.py --kind mlp_multistep --output models/world_model.pt --epochs 30 --samples-per-scene 2
```

To reproduce the full matched graph/MLP experiment, use the commands in the [A/B report](docs/world-model-ab.md). The older `scripts/train_world_model.py` remains available for the one-step baseline. Generated arrays/episodes are gitignored; model weights and recorded metrics are shipped. Restart the API after training because loaded checkpoints are cached.

Both learned models normalize quaternions; the graph and legacy paths also bound velocities. Neither uses analytic table/floor snapping or a collision solver. Learned contacts are unavailable (`contacts_available=false`). The graph enforces internal pairwise linear impulse antisymmetry, with gravity/drag/static supports as external influences; full angular-momentum conservation is not claimed. Long rollouts and unusual geometry remain approximate.

## Agent: four tools, no invented outcomes

`POST /api/assistant/query` uses OpenAI function calling over exactly:

| Tool | Execution |
| --- | --- |
| `get_scene` | Read current scene; forced first call. |
| `lookup_object_properties(label)` | Exa search of product/spec pages, OpenAI Pydantic structured extraction, source/evidence/unit checks, disk cache. |
| `simulate(actions)` | Actual 3-second PyBullet run. |
| `world_model_rollout(actions)` | Actual 3-second learned autoregressive run. |

A verified Exa mass updates matching objects in the agent's **working copy** before subsequent tools. It does not overwrite the saved scene. Product identity is explicitly unverified, so `mass_source=exa` means a published representative product mass, not a measured mass of the photographed object. Optional sizes are returned only when exact source evidence verifies the units and values; they are not used to rescale an existing reconstructed scene.

OpenAI decides what tools/actions to run and parses product specs; these are load-bearing uses. Numerical answer prose is assembled by the server from completed tool outputs, rather than trusting arbitrary model outcome prose. Responses contain `actions_ran`, `simulation_results`, `sources`, and `tool_trace`. Mismatched actions between a physics and subsequent learned comparison are rejected. Lookups without verified evidence remain `estimated`. No successful Exa results are prefilled.

Missing/down OpenAI returns an explicit `unavailable` or `partial` status and only completed tool facts. Missing/down Exa retains estimated masses with a warning. **Authenticated OpenAI/Exa calls have not been verified in this checkout because no keys were supplied.** Integration tests use mocked external SDK responses and execute real local physics/model tools; mocks are not runtime fallbacks.

Official integration references: [OpenAI function calling](https://developers.openai.com/api/docs/guides/function-calling), [structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs), [Exa SDK](https://exa.ai/docs/sdks/quickstart), [Depth Anything model card](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf).

## API and Base44

Local base URL: **`http://127.0.0.1:8017`**. Base44 on a remote/HTTPS origin needs a reachable **HTTPS** API URL, e.g. `https://YOUR-API-HOST`. Localhost refers to the viewer's machine. Set `CORS_ORIGINS` in `.env` to the exact Base44 preview/production origins, comma-separated, and restart. No API key is required for local simulation endpoints.

| Method | Path | Request / response |
| --- | --- | --- |
| POST | `/api/scenes` | multipart `image`, optional `table_width_m` → `{scene_id, scene, report}` |
| POST | `/api/scenes/demo` | `{}`; optional `?scenario=desk` or `?scenario=bottle` → `{scene_id, scene, report}` |
| GET | `/api/scenes/{id}` | → same scene record |
| POST | `/api/simulate` | `{scene_id, actions, engine, learned_architecture?, duration?, fps?}` → `{physics, learned, error, warnings, scene_id}` |
| POST | `/api/assistant/query` | `{scene_id, question}` → `{status, answer, actions_ran, simulation_results, sources, tool_trace, warnings?}` |
| GET | `/api/leaderboard` | → `{entries:[{name,scene_id,chaos_score}]}` |
| POST | `/api/leaderboard` | `{name,scene_id,chaos_score}` → saved entry (latest physics score required) |

`engine` is `physics`, `learned` or `both`. `learned_architecture` is `default`, `mlp_multistep` or `graph_impulse`; `default` selects the measured better MLP. The engine not requested is null. On `both`, a missing checkpoint returns physics plus a warning and null learned/error; learned-only returns 503. Unknown scene is 404; invalid inputs/actions are 422. Images are limited to 10 MB / 20 megapixels. Frame positions are centers, quaternions xyzw; frames additionally include velocities. Error includes per-frame, overall mean and final mean position error, plus fall-set agreement. API comparison averages over all retained objects, including static objects; held-out training metrics above exclude static objects.

Example Base44 client:

```javascript
const API = 'https://YOUR-API-HOST';
const post = async (path, body) => {
  const r = await fetch(API + path, {
    method: 'POST', headers: {'Content-Type':'application/json'},
    body: JSON.stringify(body)
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
};
const {scene_id, scene} = await post('/api/scenes/demo', {});
const result = await post('/api/simulate', {
  scene_id, actions: [{type:'push', id:'mug', direction:[1,0], impulse_Ns:0.55}],
  engine:'both', duration:3, fps:30
});
// Three.js is Y-up already; mesh.position.fromArray(frame.position),
// mesh.quaternion.fromArray(frame.quaternion). No Bullet conversion in UI.
const reply = await post('/api/assistant/query', {
  scene_id, question:'Look up the mug mass, then push it and compare both engines.'
});
const form = new FormData(); form.append('image', file); form.append('table_width_m','1.2');
const uploaded = await fetch(API+'/api/scenes',{method:'POST',body:form}).then(r=>r.json());
```

All action request shapes (each applies at t=0; radians only in scene rotations, tilt action explicitly degrees):

```json
[
  {"type":"push","id":"mug","direction":[1,0],"impulse_Ns":0.55},
  {"type":"remove","id":"phone"},
  {"type":"set_mass","id":"book1","kg":0.8},
  {"type":"tilt_support","angle_deg":15,"axis":"z"},
  {"type":"set_gravity","g":3},
  {"type":"set_support_collisions","enabled":false}
]
```

Actual recorded JSON request/response examples, including the demo scene, both-engine simulation, missing-key assistant and leaderboard: [docs/api-examples.json](docs/api-examples.json). These are captured from real endpoint executions, not invented results. Frame arrays in this document are explicitly excerpted to first/last frames. A photo response also returns detector bboxes/confidences, kept count, scale confidence, support-plane fit and estimated-value report.

## Validation and limitations

```bash
.venv/bin/python -m pytest tests -q
bash scripts/demo.sh
```

Final test output is saved in [docs/test-output.txt](docs/test-output.txt). The real HTTP desk demo and Chromium photo-to-simulation flow also passed.

See [MILESTONES.md](MILESTONES.md) for actual milestone checks and [docs/demo.png](docs/demo.png) for the captured local demo. Tests cover coordinate handedness, rest/push/static/removal/determinism, reconstruction/RANSAC/fallbacks, persistence, learned rollout without physics stepping, API comparison/errors, four-tool agent grounding, property provenance/cache, score verification and CORS.

This is a demo, not an engineering tool: monocular scale is approximate, masses are estimated unless representative Exa-sourced or user-set, shapes are boxes, occlusions/perspective can produce wrong sizes/stacks, and learned trajectories accumulate error. No contact output is claimed for the learned model. No temporal tracking, identity matching, SfM/COLMAP, optimizer, authentication or history is implemented. Nothing returns fabricated detections, fabricated API citations or disguised physics as a learned prediction.

Optional browser regression check (requires Playwright plus Chromium):

```bash
.venv/bin/pip install playwright
.venv/bin/playwright install chromium
.venv/bin/python scripts/browser_smoke.py
```

Add `--photo /path/to/real-desk.jpg` to exercise real photo upload/detections and both-engine simulation. `--chromium /path/to/headless-shell` supports a preinstalled browser. QA records the desk view to `docs/demo.png`.
