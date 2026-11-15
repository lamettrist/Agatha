# Agatha / Foresight

The video backend now accepts multipart video through both `POST /api/scenes` and
`POST /api/scenes/video` (`file`, `image`, or `video` fields). Bodies are returned in
`scene.objects`; `GET /api/health` reports supported endpoints and model status.
See [video integration](docs/video-upload-integration.md) and the
[measured video report](docs/video-reconstruction-report.json). The real fixture
returns nonempty scenes within the three-minute budget, but **geometry accuracy
and stability still fail the acceptance targets**. The report flags those failures.


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

## Contact GNN world model (v2, current default)

The default learned model is now a **contact graph network** (`models/contact_gnn.pt`, also `models/world_model.pt`; 712k parameters). Objects, the table and the floor are nodes; nearby pairs are edges carrying explicit box-contact geometry. It predicts velocity and pose changes in a normalised space at 20 Hz and integrates them. It was trained with multi-step rollouts, input noise, and penalties for penetration, motion at rest, and sliding at rest. A **non-learned projection** runs after each step at inference (push boxes out along the minimum-overlap axis, clamp to the support, put slow supported objects to sleep); set `AGATHA_CONTACT_PROJECTION=0` to turn it off. The previous MLP stays available as `models/mlp_multistep.pt` ("A / Multi-step MLP" in the UI).

### Measured results

Held-out **synthetic PyBullet box scenes**: 4,998 test episodes, 4 s autoregressive rollouts. The table uses the 2,468 episodes at the live simulator's default physics. "Model only" is the network alone; "+ projection" is what ships. Full tables, per-scenario breakdowns and bootstrap intervals are in [models/contact_eval.md](models/contact_eval.md); the UI shows the same numbers from `/api/metrics`.

| Metric | Old MLP | New, model only | New + projection |
|---|---|---|---|
| Gentle position error at 0.5 s | 0.031 m | 0.002 m | 0.002 m |
| Gentle position error at 1 s | 0.072 m | 0.002 m | 0.002 m |
| Gentle position error at 2 s | 0.143 m | 0.003 m | 0.003 m |
| Chaotic position error at 0.5 s | 0.145 m | 0.045 m | 0.047 m |
| Chaotic position error at 1 s | 0.263 m | 0.083 m | 0.089 m |
| Chaotic position error at 2 s | 0.403 m | 0.115 m | 0.122 m |
| Orientation error at 2 s (all) | 34.9° | 14.6° | 15.0° |
| Fell / didn't-fall accuracy | 0.853 | 0.979 | 0.978 |
| Fall precision | 0.508 | 0.923 | 0.916 |
| Fall recall | 0.797 | 0.938 | 0.943 |
| Mean penetration depth | 47.82 mm | 0.39 mm | 0.05 mm |
| Rest drift, all scenes | 121.4 mm/s | 6.0 mm/s | 5.5 mm/s |
| Rest drift, gentle scenes | 77.7 mm/s | 0.7 mm/s | 0.3 mm/s |

Ship rule (all met): chaotic error and penetration improve and gentle error is not worse, both against the old model and against the old model with the same projection added. PyBullet itself scores 0.12 mm penetration and 0.26 mm/s rest drift on these metrics.

### Video-derived training

The default checkpoint's last training stage is a 34-minute fine-tune in which **25% of each batch are trajectories extracted from video** (position-only loss, weight 0.2): 1,109 clips tracked from pixels, from rendered Kubric MOVi-A and CLEVRER videos plus one team-recorded desk clip. Median tracking noise is about 2.7 cm. Stated plainly: these are **rendered (synthetic) videos**; the one real clip is a static desk and contributes only objects at rest. No real-world video of moving objects was used.

Measured effect against the simulation-only checkpoint (`models/contact_gnn_sim_only.pt`, a matched no-video control is in the report): simulation metrics are unchanged within noise; rest-position error improves on CLEVRER ground truth (0.676 to 0.524 m) and held-out Kubric (0.786 to 0.742 m); on real motion-capture box tosses it is slightly worse for I.AM. in table mode (0.515 to 0.541 m), unchanged in floor mode, and slightly better for ContactNets (0.082 to 0.079 m). Selection caveat: this is the last epoch of that fine-tune, chosen after its test results were known rather than by the pre-set keep rule (which no checkpoint passed); the video arm ran fewer updates than the no-video control in the same wall-clock time. Kubric-derived data carries a repository-level licence only, so these weights should not be released publicly without re-checking it. Pipeline, licences, commands and full tables: [docs/video-training.md](docs/video-training.md).

### What improved, what did not, known limits

- **Improved:** clipping (48 mm to under 0.1 mm), drift at rest, chaotic-case error (about 70% lower at 2 s), and fall recall (0.80 to 0.94).
- **Fall precision is lower than an earlier checkpoint** (0.92 versus 0.94): the changes that raised recall cost about two points of precision.
- **The projection slightly hurts chaotic position error** (0.114 m without, 0.121 m with) while removing the remaining penetration. One projection constant was chosen using the test set; its effect is 0.003 m.
- **Still wrong in places:** orientation in chaotic scenes (about 28° at 2 s), pushing the base of a stack, objects larger than 0.5 m, gravity above 15 m/s², and scenes far from the training range. Non-box shapes are approximated by their bounding box.
- **Latency:** about 0.9 s for a 3 s rollout of the demo desk on CPU with projection (the MLP took 0.14 s).
- **Hardware note:** training used fp16 mixed precision; the Quadro RTX 6000 cards have no native bf16.
- **External checks (approximate):** on two real motion-capture box-toss datasets the simulation-only checkpoint matches PyBullet's rest-position error and is well ahead of the old MLP; results depend strongly on the assumed friction. See [docs/part-c-external-data.md](docs/part-c-external-data.md) and [DATA_LICENSES.md](DATA_LICENSES.md). The only external data used in training are the video-derived trajectories described above.

### Reproduce

Training needs a CUDA build that matches the driver; here that is a separate env (`.venv-train`, torch 2.14.1+cu126).

```bash
# 1. Data: 31k episodes (2.48M transitions), then 13.9k hard-case episodes; data/v3 = both for training, data/v2/test for testing
.venv/bin/python scripts/generate_contact_data.py --out data/v2 --train-episodes 25000 --val-episodes 1000 --test-episodes 5000 --workers 56
.venv/bin/python scripts/generate_contact_data.py --out data/v2_hard --train-episodes 12000 --val-episodes 400 --test-episodes 1500 --workers 40 --seed 7331 --mix hard
# data/v3: symlink data/v2 and data/v2_hard train/val shards together (hard shards renamed shard_hNNNNN.npz); test = data/v2/test only

# 2. Stage 1, 7 GPUs, 90 minutes
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 .venv-train/bin/torchrun --nproc_per_node=7 scripts/train_contact_ddp.py \
  --data data/v2 --out models/runs/main_a --epochs 400 --batch 4096 --max-minutes 90

# 3. Stage 2, 4 GPUs, motion-onset fixes for fall recall (stopped after 137 of 165 minutes)
CUDA_VISIBLE_DEVICES=0,1,2,3 .venv-train/bin/torchrun --nproc_per_node=4 scripts/train_contact_ddp.py \
  --data data/v3 --init models/runs/main_a/best.pt --start-horizon 8 --horizon 8 --epochs 2000 --lr 5e-4 \
  --w-move 3 --w-impulse 3 --w-speed 1 --event-frac 0.3 --score-fall-weight 0.3 --val-rollout-episodes 1000 \
  --batch 4096 --max-minutes 165 --out models/runs/final_pen1

# 3b. Stage 3, 1 GPU, 34 minutes: fine-tune with 25% video-derived windows (build them first: scripts/video_to_trajectories.py, see docs/video-training.md)
CUDA_VISIBLE_DEVICES=0 .venv-train/bin/python scripts/train_contact_ddp.py --data data/v3 --init models/contact_gnn_sim_only.pt \\
  --start-horizon 8 --horizon 8 --lr 1e-4 --w-move 3 --w-impulse 3 --w-speed 1 --event-frac 0.3 --score-fall-weight 0.3 \\
  --val-rollout-episodes 1000 --batch 1536 --windows-per-epoch 600000 --epochs 30 --max-minutes 34 \\
  --video-data data/video_v1/kubric_movi_a data/video_v1/clevrer data/video_v1/team_deskspace --video-frac 0.25 --video-weight 0.2 \\
  --out models/runs/video_ft_video25   # shipped checkpoint = epoch_011.pt (last epoch)

# 4. Evaluate old versus new, with and without projection
CUDA_VISIBLE_DEVICES=0 .venv-train/bin/python scripts/eval_contact.py --data data/v2 --split test \
  --old models/mlp_multistep.pt --old-graph models/graph_impulse.pt --new models/contact_gnn.pt --out models/contact_eval.json
```

The trainer checkpoints every epoch (`last.pt`, `best.pt`, `epoch_NNN.pt`), resumes with `--resume`, and logs GPU utilisation to `gpu_util.jsonl` (about 78-88% during these runs). Restart the API after changing checkpoints; loaded models are cached.

## Learned model: earlier MLP and graph experiments (history): actual training and A/B results

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

The frontend is a separate app that uses this server only through HTTP. Full reference: [docs/api.md](docs/api.md).
Frontend guide with copy-paste code: [docs/base44-integration.md](docs/base44-integration.md).
Recorded real exchanges for every route: [docs/api-examples.json](docs/api-examples.json)
(`scripts/capture_api_examples.py --base <running server>`).

Local base URL: **`http://127.0.0.1:8017`**. Base44 on a remote/HTTPS origin needs a reachable **HTTPS** API URL; localhost refers to the viewer's machine. `https://precognition.base44.app` is allowed by default; set `CORS_ORIGINS` in `.env` (exact origins, comma-separated) for others and restart. No API key is required.

| Method | Path | Request → response |
| --- | --- | --- |
| POST | `/api/scenes/demo?scenario=` | `desk` (default), `bottle`, `laptop`, `soft` → `{scene_id, scene, report}` |
| GET | `/api/scenes/{id}` | → same scene record |
| POST | `/api/scenes` | multipart `image` (JPEG/PNG/WebP, 10 MB), optional `table_width_m`, `distance_reference` → scene record |
| POST | `/api/scenes/video` | multipart `file` (video, 100 MB), optional `table_width_m` → scene record; 422/503 with a report when nothing is reconstructed |
| POST | `/api/scenes/scan` | multipart `file` (`.ply`, `.obj`, `.usdz`) → scene record |
| POST | `/api/scenes/{id}/objects/{object_id}` | `{mass_kg?, shape?, size?}` → a new scene record (mass only for scenes with joints or soft bodies) |
| POST | `/api/simulate` | `{scene_id, actions, engine?, learned_architecture?, duration?, fps?}` → `{scene_id, physics, learned, error, warnings}` |
| POST | `/api/assistant/query` | `{scene_id, question}` → `{status, answer, actions_ran, simulation_results, sources, tool_trace, warnings?}` |
| GET | `/api/leaderboard` | → `{entries:[{name, scene_id, chaos_score}]}` |
| POST | `/api/leaderboard` | `{name, scene_id, chaos_score}` → saved entry (must match the latest physics score) |
| GET | `/api/metrics` | → held-out evaluation numbers (`contact_v2` block for the current default) |
| GET | `/api/reconstructions/{run_id}/{filename}` | → debug image / mesh of an upload |
| GET | `/api/health` | → supported scenarios, architectures, experimental flags, loaded vision models |

A scene is `{objects[], support, joints[], soft_bodies[]}`. Units are metres/kg/seconds/radians, right-handed Y-up, positions are centres, frame quaternions are xyzw, `support.height` is the table top.

**Engines.** `engine` is `physics`, `learned` or `both`. `learned_architecture` is `default` (or `auto`): the server routes by scene content. Other values: `contact_gnn`, `joint_gnn`, `soft_gnn`, and the earlier baselines `mlp_multistep`, `graph_impulse`.

| Scene contains | Learned model (`model_architecture`) | PyBullet reference |
| --- | --- | --- |
| boxes only (`desk`, `bottle`, uploads) | `contact_gnn` (contact graph network + non-learned contact projection) | yes |
| joints (`laptop`) — experimental | `joint_gnn` | no: `physics` and `error` are `null`, `warnings` explains |
| soft bodies (`soft`) — experimental | `contact_gnn+soft_gnn` (one-way rigid → soft coupling) | no: same |

Unknown scene is 404; invalid inputs are 422 (`detail` is a string or a validation array); a learned-only request with a missing checkpoint is 503. Joints, soft bodies and video-upload accuracy are experimental.

```javascript
const API = 'https://YOUR-API-HOST';
const post = async (path, body) => {
  const r = await fetch(API + path, {
    method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(body)
  });
  const data = await r.json();
  if (!r.ok) throw new Error(typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail));
  return data;
};
const {scene_id, scene} = await post('/api/scenes/demo?scenario=desk', {});   // or bottle | laptop | soft
const {physics, learned, error, warnings} = await post('/api/simulate', {
  scene_id, actions: [{type:'push', id:'mug', direction:[1,0], impulse_Ns:0.55}],
  engine:'both', duration:3, fps:30
});
const run = physics ?? learned;        // physics is null for jointed / soft scenes: show `warnings`
for (const o of run.frames[0].objects) {
  // Three.js is Y-up already: mesh.position.fromArray(o.position); mesh.quaternion.fromArray(o.quaternion)
}
// Soft bodies (learned only): learned.soft_meshes[id].faces (flat triangle indices) +
// frame.soft_bodies[i].vertices (flat xyz) -> THREE.BufferGeometry. See docs/base44-integration.md.
const form = new FormData(); form.append('image', file); form.append('table_width_m','1.2');
const uploaded = await fetch(API+'/api/scenes',{method:'POST',body:form}).then(r=>r.json());
```

Actions (each applies at t=0; tilt is in degrees):

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
