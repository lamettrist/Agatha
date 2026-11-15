# Agatha HTTP API reference

Complete reference for the FastAPI server in `agatha/main.py` (13 API routes, enumerated from the running
server's `/openapi.json`). The practical frontend guide is [base44-integration.md](base44-integration.md);
recorded exchanges for every route are in [api-examples.json](api-examples.json) (regenerate with
`scripts/capture_api_examples.py` against a running server).

Everything below was observed on a running server built from this tree, except the items explicitly marked
**(from source, not exercised)**.

## Status of features

| Feature | Status |
|---|---|
| Box scenes (demo `desk`, `bottle`), PyBullet reference, learned `contact_gnn`, leaderboard, metrics | Supported |
| Photo upload (`POST /api/scenes`) | Supported; the result is an estimate, check `report.quality_status` |
| **Joints** (`scene.joints`, `joint_gnn`, demo `laptop`) | **EXPERIMENTAL**: learned only, no PyBullet reference in the API |
| **Soft bodies** (`scene.soft_bodies`, `soft_gnn`, demo `soft`) | **EXPERIMENTAL**: learned only, one-way coupling, no PyBullet reference |
| **Video upload** (`POST /api/scenes/video`) | **EXPERIMENTAL accuracy**: scenes usually come back `needs_review` |
| Scan upload (`POST /api/scenes/scan`) | Present; **(from source, not exercised)** in this documentation pass |

`GET /api/health` reports the same flags in `experimental_features`.

## Conventions

- Units: metres, kilograms, seconds, radians (the only degrees are `tilt_support.angle_deg`).
- Axes: right-handed, **Y up**. Three.js uses the same convention, no conversion is needed.
- `position` is always the centre of a body. `size` is the full extent `[x, y, z]` of its bounding box.
- Scene rotations are `rotation_euler` (fixed XYZ, radians). Simulation frames use `quaternion` in **xyzw** order.
- `support.height` is the height of the **top** surface of the table; `support.size` is `[width (x), depth (z)]`;
  the slab is centred on the origin in x/z.
- All actions are applied once at t = 0, in list order.
- Request bodies are strict: an unknown JSON field is a 422.
- No authentication. CORS allows `GET` and `POST` from the origins in `CORS_ORIGINS` (default includes
  `https://precognition.base44.app` and localhost 8000/8017).
- Run **one** server worker (scenes are cached in memory and stored as JSON files in `DATA_DIR`).

## Errors

`detail` has three shapes; a client must handle all three:

| Case | Status | `detail` |
|---|---|---|
| Raised by a handler (unknown scene, bad scenario, bad action target, ...) | 404 / 413 / 422 / 503 | string |
| Request body failed validation (wrong type, out of range, unknown field) | 422 | array of `{type, loc, msg, input, ctx?}` (FastAPI) |
| Video upload reconstructed zero bodies | 422, or 503 when the detector is unavailable | object `{message, report}` **(from source, not exercised)** |

Observed strings: `"Scene not found"` (404), `"scenario must be desk, bottle, laptop or soft"` (422),
`"Unknown object: nope"` (422), `"learned_architecture joint_gnn needs a scene with joints"` (422),
`"Provide a multipart image, file, or video field"` (422), `"Object not found"` (404), `"Asset not found"` (404).

## Routes

| # | Method | Path | Purpose |
|---|---|---|---|
| 1 | POST | `/api/scenes/demo` | Create a bundled demo scene |
| 2 | GET | `/api/scenes/{scene_id}` | Read a scene record |
| 3 | POST | `/api/scenes` | Photo upload (also accepts video / scan files and forwards them) |
| 4 | POST | `/api/scenes/video` | Video upload |
| 5 | POST | `/api/scenes/scan` | 3D scan upload (PLY / OBJ / USDZ) |
| 6 | POST | `/api/scenes/{scene_id}/objects/{object_id}` | Edit one object; returns a NEW scene |
| 7 | POST | `/api/simulate` | Run PyBullet and/or the learned model |
| 8 | POST | `/api/assistant/query` | Tool-using assistant (needs `OPENAI_API_KEY`) |
| 9 | GET | `/api/leaderboard` | Top 20 chaos scores |
| 10 | POST | `/api/leaderboard` | Submit a score |
| 11 | GET | `/api/metrics` | Held-out evaluation numbers of the learned models |
| 12 | GET | `/api/reconstructions/{run_id}/{filename}` | Reconstruction debug images and meshes |
| 13 | GET | `/api/health` | Capabilities, loaded vision models |

Not in the OpenAPI schema: `GET /` and `/static/*` serve the local test viewer only.

---

## The scene record

Every scene-producing route (1-6) answers with the same record:

```
{ scene_id: string (32 hex chars),
  scene: { objects: SceneObject[], support: Support, joints: Joint[], soft_bodies: SoftBody[] },
  report: object | null }
```

Scene records are immutable. Simulations never change them; the object edit creates a new record.

### SceneObject

| Field | Type | Default | Meaning |
|---|---|---|---|
| `id` | string 1-80 | required | Unique in the scene; `support` and `floor` are reserved |
| `label` | string 1-120 | required | Human-readable name |
| `shape` | `box` \| `cylinder` \| `compound` \| `convex_hull` | `box` | Collision shape family used by PyBullet |
| `collision` | `{parts: CollisionPart[1..64]}` \| null | null | null means a single box of `size`. See below |
| `shape_method` | string | `legacy_box` | How the shape was obtained (free text, e.g. `silhouette_cylinder_metric_reference`, `user_edited`) |
| `visual_mesh_url` | string \| null | null | Server-relative URL of a GLB mesh for display only (prefix with the API base). null for demo scenes |
| `scale` | [x,y,z] | [1,1,1] | Visual scale |
| `supported_by` | string \| null | null | `"support"`, `"floor"`, another object id, or null (unknown / demo). Floor-standing objects never count as `fell` |
| `soft_approx` | bool | false | The real object is soft but is simulated rigid |
| `restitution` | number 0-1 | 0 | |
| `physical_properties_estimated` | bool | true | |
| `linear_damping`, `angular_damping` | number 0-1 | 0.04 | |
| `size` | [x,y,z], each >0 and <=100 m | required | Bounding-box extents. The learned models only see this box |
| `position` | [x,y,z] m | required | Centre |
| `rotation_euler` | [x,y,z] rad | [0,0,0] | Fixed XYZ |
| `mass_kg` | number >0, <=1000 | 0.5 | |
| `mass_source` | `estimated` \| `exa` \| `user` | `estimated` | |
| `friction` | number 0-2 | 0.5 | |
| `is_static` | bool | false | Fixed in place (the `desk` slab, furniture). Static objects still appear in every frame |
| `confidence` | number 0-1 | 1 | Detection confidence (uploads) |

`CollisionPart`: `{type: box|cylinder|convex_hull, size?: [x,y,z], radius?, height?, vertices?: [[x,y,z],...] (<=4096), position: [x,y,z], rotation_euler: [x,y,z]}`
in the object's local frame. `box` needs `size`, `cylinder` needs `radius` and `height` (axis = local Y),
`convex_hull` needs at least 4 vertices.

### Support

| Field | Type | Default | Meaning |
|---|---|---|---|
| `size` | [width, depth] m | [1.2, 0.8] | x and z extents |
| `height` | number 0-5 m | 0.75 | Top surface |
| `thickness` | number >0, <=0.5 m | 0.08 | |
| `collisions_enabled` | bool | true | false = objects pass through the table |

The demo scenes also contain a static object `desk` that duplicates the slab.

### Joint (EXPERIMENTAL)

| Field | Type | Default | Meaning |
|---|---|---|---|
| `id` | string 1-80 | required | Unique among joints |
| `type` | `hinge` \| `ball` \| `prismatic` \| `fixed` | required | |
| `parent` | object id \| null | null | null = attached to the world |
| `child` | object id | required | An object can be the child of at most one joint; no cycles |
| `anchor_parent` | [x,y,z] m | required | In the parent's body frame (world coordinates when `parent` is null) |
| `anchor_child` | [x,y,z] m | required | In the child's body frame |
| `axis` | [x,y,z] | [0,0,1] | Hinge / prismatic axis in the parent's frame |
| `lower`, `upper` | number (rad for hinge, m for prismatic) | -3.2, 3.2 | `lower <= upper` |
| `damping` | number 0-10 | 0 | |

At joint coordinate 0 the child has the parent's orientation. The object poses in the scene must already
satisfy the joints: `POST /api/simulate` answers 422 when an anchor is more than 5 mm open or a limit is
exceeded by more than 0.1 **(from source, not exercised)**.

### SoftBody (EXPERIMENTAL)

| Field | Type | Default | Meaning |
|---|---|---|---|
| `id` | string 1-80 | required | Unique, distinct from object ids |
| `label` | string 1-120 | required | |
| `kind` | `cube` \| `ball` \| `cloth` | required | Procedural mesh family |
| `size` | [x,y,z] m | required | Extents; cloth uses `[width, ignored, depth]` |
| `position` | [x,y,z] m | required | Centre |
| `rotation_euler` | [x,y,z] rad | [0,0,0] | |
| `mass_kg` | number >0, <=100 | 0.3 | |
| `stiffness` | number 0-1 | 0.5 | 0 softest, 1 stiffest of the trained range |
| `friction` | number 0-2 | 0.5 | |
| `velocity` | [x,y,z] m/s | [0,0,0] | Initial velocity |

A soft body has no vertices in the scene record. The mesh (vertex positions per frame and the triangle
index list) only exists in a simulation response. To draw a soft body before the first simulation, draw a
placeholder from `kind`, `size`, `position`.

At most 8 soft bodies, 64 joints and 256 objects per scene.

### report

`report` is free-form and depends on the source. Demo scenes:
`{source: "bundled_demo", scenario, estimated: ["all masses and dimensions"]}`.

For uploads, the fields a frontend should use (observed on photo and video uploads):

| Field | Type | Meaning |
|---|---|---|
| `quality_status` | string | `"needs_review"` was observed on both fixtures. It means a measured check failed, not that the request failed |
| `quality_checks` | object of bool | Photo: `table_detected, contacts, stability, reprojection`. Video: `tracking, contacts, stability, reprojection` |
| `warnings` | string[] | Show these to the user |
| `body_count`, `dynamic_body_count` | int | Bodies in `scene.objects` and how many are movable |
| `detected_count`, `kept_count` | int | Detector output before / after filtering |
| `scale_confidence`, `scale_basis` | string | How metric scale was obtained (observed `"medium"`, `"manual"`) |
| `estimated` | string[] | What was assumed rather than measured |
| `input`, `pipeline` | string | `photo` / `video`; pipeline version (observed `metric-photo-v17`, `video_multiview_v8`) |
| `run_id` | 64 hex chars | Reconstruction run |
| `debug_url` | string | **URL prefix**, `/api/reconstructions/{run_id}`. It is not a page: requesting it bare is a 404. Append a filename (route 12) |
| `cache_hit` | bool | true when this input was reconstructed before |
| `duration_s` | number | Wall time of the original reconstruction |
| `request_duration_s` | number, only on a cache hit | Time of this request |
| `acceptance_pass` | bool | false unless an annotated fixture passed; false after any manual edit |
| `status` | string | observed `"complete"` |

Also present (diagnostics; do not build UI on them): `stages`, `metrics`, `detections`, `objects`, `models`,
`camera`, `camera_pose`, `scale`, `image_size`, `detector`, `depth`, `acceptance_note`, `photo_latency_pass` /
`video_latency_pass`, `video`, `photo_run_id`. After an object edit: `edited_from` (source scene id).

---

## 1. POST /api/scenes/demo

Query parameter `scenario`: `desk` (default), `bottle`, `laptop`, `soft`. The JSON body is ignored (send `{}`
or nothing). Response: scene record. Unknown scenario: 422 string. A new `scene_id` is created on every call.

| scenario | Contents | Learned route |
|---|---|---|
| `desk` | laptop, mug, two books, box, phone on a desk (all boxes) | `contact_gnn` |
| `bottle` | bottle scene (boxes) | `contact_gnn` |
| `laptop` | laptop base + lid joined by a hinge (open at 100 degrees, limits 0-125 degrees), and a mug | `joint_gnn` |
| `soft` | two boxes, a soft ball above the table, a cloth above a box | `contact_gnn+soft_gnn` |

## 2. GET /api/scenes/{scene_id}

Response: the scene record. 404 `"Scene not found"` for an unknown or malformed id.

## 3. POST /api/scenes (photo)

`multipart/form-data`:

| Field | Type | Meaning |
|---|---|---|
| `image` (also accepted: `file`, `video`) | file | JPEG / PNG / WebP, at most 10 MB and 20 megapixels |
| `table_width_m` | number, optional | Known table width, sets metric scale |
| `distance_reference` | string, optional | JSON `[[x1,y1],[x2,y2],metres]`: two points in the image resized so its longer side is at most 1280 px, and their real distance |

If the file is a video (`video/*` content type or `.mp4 .mov .m4v .webm .avi .mkv`) the request is handled as
route 4; `.ply .obj .usdz` as route 5.

Response: scene record with the upload report. Errors: 422 string (no file, not an image, bad reference),
413 string (too large). Needs the vision models (GPU server, `.venv-reconstruction`).

Caching: identical input and settings return the cached reconstruction (`report.cache_hit: true`) under a new
`scene_id`.

## 4. POST /api/scenes/video (EXPERIMENTAL accuracy)

`multipart/form-data`: `file` (also accepted: `image`, `video`), at most 100 MB; optional `table_width_m`.

Response: scene record. A 200 always has at least one object. Errors:

- 422 string: unreadable video or a pipeline error. **(from source, not exercised)**
- 422 object `{message, report}`: no bodies reconstructed. 503 with the same object when the detector is
  unavailable. **(from source, not exercised)**
- 413 string: over 100 MB. **(from source, not exercised)**

Caching: completed runs with valid tracks are cached by content and pipeline configuration. Furniture is
returned `is_static: true` (stated in `report.warnings`).

## 5. POST /api/scenes/scan

**(from source, not exercised.)** `multipart/form-data` field `file`: `.ply`, `.obj` or `.usdz`, at most
100 MB. Response: scene record. 422 string for another extension or a parse failure, 413 over 100 MB.

## 6. POST /api/scenes/{scene_id}/objects/{object_id}

JSON body, all optional:

| Field | Type | Meaning |
|---|---|---|
| `mass_kg` | number >0, <=1000 | New mass; sets `mass_source: "user"` |
| `shape` | `"box"` \| `"cylinder"` | Replace the collision shape |
| `size` | [x,y,z], each >0 and <=100 | New extents. Cylinder: diameter = `size[0]`, height = `size[1]` |

Response: a **new** scene record (new `scene_id`); the original is unchanged. `report.edited_from` is the
original id, `report.acceptance_pass` is false and a warning is appended.

- Box-only scenes: the scene is re-settled in PyBullet after the edit, so poses change by small amounts
  (observed: sub-millimetre for untouched objects) and `report.metrics` is replaced.
- **Scenes with joints or soft bodies:** only `mass_kg` may be edited. The scene is **not** re-settled; every
  pose, the joints and the soft bodies are kept exactly. A request containing `shape` or `size` is rejected
  with 422 `"Shape/size edits are not supported for scenes with joints or soft bodies; only mass_kg can be edited"`.
  (The settle step knows neither joints nor soft bodies and would leave a scene the simulator rejects.)

Errors: 404 `"Scene not found"`, 404 `"Object not found"` (checked before the rule above), 422 string for a
shape other than box/cylinder, 422 array for out-of-range values. Soft bodies cannot be edited (they are not
in `scene.objects`).

## 7. POST /api/simulate

### Request

| Field | Type | Default | Meaning |
|---|---|---|---|
| `scene_id` | string | required | |
| `actions` | Action[], at most 20 | [] | Applied at t=0 in order |
| `engine` | `physics` \| `learned` \| `both` | `both` | |
| `learned_architecture` | see below | `default` | |
| `duration` | number 0.1-5 s | 3 | |
| `fps` | 10 \| 20 \| 30 | 30 | Output frame rate. The learned models step at 20 Hz; 30 fps frames are interpolated |

Actions:

| `type` | Fields | Range |
|---|---|---|
| `push` | `id`, `direction` `[x,z]` (horizontal) or `[x,y,z]`, `impulse_Ns` | direction non-zero (normalised by the server); impulse >0, <=20 |
| `remove` | `id` | |
| `set_mass` | `id`, `kg` | >0, <=1000 |
| `tilt_support` | `angle_deg`, `axis` (`x` \| `z`) | -60 to 60 degrees |
| `set_gravity` | `g` | 0 to 30 m/s2 |
| `set_support_collisions` | `enabled` | bool; false lets objects fall through the table |

`id` may be a rigid object id or a soft body id. On a soft body, `push` adds impulse/mass to every vertex
velocity, `remove` drops it, `set_mass` changes its mass. An id that does not exist: 422 `"Unknown object: <id>"`.

`learned_architecture`:

| Value | What runs |
|---|---|
| `default`, `auto` | The router decides from the scene (identical behaviour for both values): no joints -> `contact_gnn` (`models/world_model.pt`); joints -> `joint_gnn` for all rigid bodies; soft bodies -> an additional `soft_gnn` pass |
| `contact_gnn` | The contact graph network explicitly (`models/contact_gnn.pt`). Joints in the scene are ignored, and this is stated in `routing.reasons` and `approximations` |
| `mlp_multistep`, `graph_impulse` | Earlier baseline models, kept for comparison. Joints ignored |
| `joint_gnn` | Force the joint model; 422 if the scene has no joints |
| `soft_gnn` | Force the soft pass; 422 if the scene has no soft bodies |

With an explicit rigid model the soft pass still runs when the scene has soft bodies.

### Response

```
{ scene_id, physics: EngineResult | null, learned: EngineResult | null, error: Error | null, warnings: string[] }
```

**When a field is null**

| Situation | `physics` | `learned` | `error` | `warnings` |
|---|---|---|---|---|
| Box scene, `engine: both` | result | result | result | `[]` |
| `engine: physics` | result | null | null | |
| `engine: learned` | null | result | null | |
| **Scene with joints or soft bodies**, `engine: both` | **null** | result | **null** | `"PyBullet reference is not wired for jointed/soft scenes in the API (this scene has joints); no PyBullet result is returned."` (or `soft bodies`) |
| Scene with joints or soft bodies, `engine: physics` | null | null | null | same warning. The 200 response carries no result at all |
| Learned checkpoint missing, `engine: both` **(from source, not exercised)** | result | null | null | `"Learned model unavailable: ..."` |
| Learned checkpoint missing, `engine: learned` **(from source, not exercised)** | | | | 503 string instead of a response |

Other warnings: `"Learned dynamics uses bounding-box dimensions; it was not trained on these collision shapes."`
when the learned model runs on a scene with a non-box object **(from source, not exercised)**.

No chaos score is recorded for the leaderboard when `physics` is null.

### EngineResult (both engines)

| Field | Type | Meaning |
|---|---|---|
| `engine` | `"physics"` \| `"learned"` | |
| `frames` | Frame[] | `round(duration*fps) + 1` frames (91 for 3 s at 30 fps), first at t = 0 |
| `events.contacts` | `[{ids: [a, b], t}]` | First contact time per pair; ids may be `support` / `floor`. **Always `[]` for learned** |
| `events.fell` | string[] (sorted ids) | Left the support: centre outside the table footprint (or table collisions off) and more than 3 cm below the top surface, at any frame. Never static or floor-standing objects |
| `events.tipped` | string[] (sorted ids) | Local up axis more than 35 degrees from vertical at any frame |
| `final_state` | Frame | Same object as the last frame |
| `total_displacement_m` | number | Sum over non-static objects of the distance between initial and final position |
| `chaos_score` | number | `len(fell) + total_displacement_m` |
| `support_rotation_euler` | [x,y,z] rad | Table tilt after `tilt_support` (observed `[0,0,0.2618]` for 15 degrees about z). Rotate the table mesh by it |
| `support_collisions_enabled` | bool | |
| `estimated` | bool | Always true: parameters are estimates |
| `disclaimer` | string | Show it next to the result |

Frame:

```
{ t: seconds,
  objects: [{ id, position: [x,y,z], quaternion: [x,y,z,w], velocity: [x,y,z] m/s, angular_velocity: [x,y,z] rad/s }],
  soft_bodies?: [{ id, vertices: [x0,y0,z0, x1,y1,z1, ...] }] }
```

`objects` contains every object that was not removed, static ones included, in scene order.
`soft_bodies` is present only in learned frames of a scene that has soft bodies: world-space vertex positions,
flat, 4 decimals, `3 * soft_meshes[id].vertex_count` numbers.

### Learned-only fields

| Field | Type | Meaning |
|---|---|---|
| `model_architecture` | string | What ran: `contact_gnn`, `joint_gnn`, `contact_gnn+soft_gnn`, `joint_gnn+soft_gnn`, or `mlp_multistep` / `graph_impulse` (optionally `+soft_gnn`) when requested |
| `contacts_available` | false | The learned models do not report contacts |
| `projection_applied` | bool | A non-learned correction ran after each step (contact push-out; for joints also anchors / axes / limits). **Absent** for `mlp_multistep` |
| `approximations` | string[] | Everything outside the training range or not learned, e.g. `"joints are ignored by the selected rigid model"`, `"first 0.05 s of airborne soft bodies (ball, cloth) is a non-learned ballistic step that seeds the model"`. **Absent** for `mlp_multistep` |
| `routing.rigid` | string | Rigid model that ran |
| `routing.soft` | bool | Whether the soft pass ran |
| `routing.reasons` | string[] | Why, in plain words |
| `routing.requested` | string | The `learned_architecture` of the request |
| `routing.checkpoints` | `{model: path}` | e.g. `{"contact_gnn": "models/world_model.pt", "soft_gnn": "models/soft_gnn.pt"}` |
| `routing.coupling` | `"none"` \| `"one_way_rigid_to_soft"` | |
| `joints` | Joint[] | The scene joints that survived the actions (`[]` for box scenes) |
| `soft_meshes` | `{id: {faces: int[], vertex_count: int, kind}}` | Triangle indices, flat (`[a0,b0,c0, a1,b1,c1, ...]`), constant over the rollout (`{}` without soft bodies). Observed in the soft demo: ball 64 vertices / 108 triangles, cloth 81 vertices / 128 triangles |

One-way coupling: rigid bodies are predicted first and never feel a soft body; each soft body then treats the
predicted boxes (the 4 nearest) as moving obstacles. Soft bodies do not collide with each other.

### Error (only when both engines returned a result)

| Field | Type | Meaning |
|---|---|---|
| `mean_position_error_m` | number | Mean over frames of the per-frame value |
| `final_position_error_m` | number | Per-frame value of the last frame |
| `per_frame` | `[{t, mean_position_error_m}]` | One entry per frame: mean distance between physics and learned positions over all retained objects, static ones included |
| `fell_agreement` | bool | true when both engines report exactly the same `fell` list |

### Errors

404 unknown scene. 422 string: unknown object id, `joint_gnn` / `soft_gnn` on a scene without joints / soft
bodies, joint pose not satisfied, diverged model. 422 array: validation. 503 string: learned-only request with
a missing checkpoint.

## 8. POST /api/assistant/query

Request: `{scene_id, question}` (question 1-2000 chars). 404 for an unknown scene.

Response:

| Field | Type | Meaning |
|---|---|---|
| `status` | `ok` \| `partial` \| `unavailable` | `unavailable`: no `OPENAI_API_KEY`, or no tool completed |
| `answer` | string | Rendered from tool results only |
| `actions_ran` | `[{engine, actions}]` | |
| `simulation_results` | summary[] | `{engine, support_collisions_enabled, fell, tipped, total_displacement_m, chaos_score, final_state, disclaimer}`; for jointed / soft scenes also `model_architecture`, `approximations` |
| `sources` | `[{url, label, mass_kg, reference_kind}]` | Product lookups |
| `tool_trace` | `[{tool, arguments, result}]` | |
| `warnings` | string[] | **Absent** in the no-key response |

Only the no-key response (`status: "unavailable"`, empty arrays) was observed on the running server; the keyed
path is covered by `tests/test_assistant.py` and `tests/test_api_gaps.py` with a stubbed OpenAI client.
The assistant's simulations go through the same router as route 7: on a scene with joints or soft bodies its
`simulate` tool reports the "PyBullet reference is not wired" message as a tool error (no physics outcome is
claimed) and its learned rollout uses `joint_gnn` / `soft_gnn`. Rollouts are fixed at 3 s.
With a key, a query makes up to 8 OpenAI calls with a 25 s timeout each **(from source)**.

## 9. GET /api/leaderboard

`{entries: [{name, scene_id, chaos_score}]}`, at most 20, highest first.

## 10. POST /api/leaderboard

Request: `{name (1-40 chars), scene_id, chaos_score (>= 0)}`. `chaos_score` must equal the latest
`physics.chaos_score` the server computed for that scene (each physics run overwrites it). Response: the saved
entry. Errors: 404 unknown scene; 422 `"Run the physics engine for this scene before submitting a score"`;
422 `"chaos_score must match the latest server-computed physics score"`.

Jointed / soft scenes never get a physics score, so they cannot be submitted.

## 11. GET /api/metrics

Returns `models/metrics.json` as written by the evaluation scripts (NaN / Infinity become null). 404 when the
file is missing.

- Top level: evaluation of the **previous** default (`kind: "mlp_multistep"`), kept for history, plus
  `selected_kind: "contact_gnn"` and `selection_note`.
- `contact_v2`: the current default.
  - `dataset`: `{path, split, n_episodes, rollout_seconds, subset}`.
  - `old`, `old_projected`, `new_raw`, `new_projected`, `physics_reference`: each
    `{all, gentle, chaotic, latency_ms_per_step_cpu}`. `old` = previous MLP, `new_projected` = what is served
    (contact_gnn + non-learned projection), `new_raw` = the network alone.
  - Each of `all` / `gentle` / `chaotic`: `{n_episodes, pos_err_m: {"0.5s","1s","2s"}, pos_err_active_2s_m,
    out_of_bounds_episodes, ori_err_deg_2s, fell_accuracy, fell_precision, fell_recall, mean_penetration_m,
    mean_penetration_capped_50mm_m, frac_frames_penetrating_over_5mm, rest_drift_mps, diverged_episodes}`.
  - `ship_decision`: `{rule, variants, ship, model_alone_passes, beats_old_projected, statement, ...}`.
  - `shipped` (bool), `checkpoint` `{file, sha256_12, kind, parameters, training}`, `video_training`,
    `notes`, `notes_extra`, `live_subset_kinds`, `full_eval`, `previous_default`, `simulation_only_checkpoint`.

For a before / after card use `contact_v2.old.all` against `contact_v2.new_projected.all`. These numbers are for
synthetic box scenes only; there is no entry for joints or soft bodies in this file.

## 12. GET /api/reconstructions/{run_id}/{filename}

Files of a reconstruction run. `run_id` is 64 lowercase hex chars; `filename` must match
`[a-zA-Z0-9_]+\.(jpg|png|json|ply|glb|npy|npz)`. 404 `"Run not found"` / `"Asset not found"` otherwise.

Files observed in photo and video runs: `original.jpg`, `masks_overlay.png`, `depth.png`, `render.png`,
`reprojection.png`, `points.ply`, `scene.json`, `reconstruction_report.json`, `object_NNN.glb` (the targets of
`visual_mesh_url`). Not every object has a GLB.

## 13. GET /api/health

| Field | Type | Meaning |
|---|---|---|
| `status` | `"ok"` | |
| `reconstruction_api_version` | int | 2 |
| `media_types` | string[] | `photo, video, ply, obj, usdz` |
| `photo_endpoint`, `video_endpoint`, `scan_endpoint`, `demo_endpoint` | string | Paths |
| `objects_field` | string | `scene.objects` |
| `loaded_models` | string[] | Vision models in memory (`depth, detector, sam, clip` on the GPU server; `[]` on a CPU-only server) |
| `model_load_failures` | object | name -> error |
| `cuda_available` | bool | false means uploads will be slow or unavailable |
| `demo_scenarios` | string[] | `desk, bottle, laptop, soft` |
| `learned_architectures` | string[] | Accepted `learned_architecture` values |
| `default_learned_architecture` | string | `default` |
| `experimental_features` | `{joints, soft_bodies, video_upload}` | true = experimental |
| `physics_reference` | `{box_scenes, joints, soft_bodies}` | false = `physics` is null for such scenes |
| `docs` | `{api_reference, frontend_guide, examples, router}` | Repository paths of the written spec |

---

## Latency

Measured once on the development machine by `scripts/capture_api_examples.py` (client-side wall time of
single requests, server idle; `elapsed_s` of each entry in `api-examples.json`). Simulation: CPU only, 4 torch
threads, 3 s at 30 fps. Uploads: one GPU (24 GB card, `RECON_GPU_MEMORY_GB=9`), vision models preloaded.

| Request | Seconds |
|---|---|
| Demo scene, get scene, health, metrics, leaderboard | under 0.01 |
| Simulate box scene, `both` (PyBullet + contact_gnn) | 0.9 warm; 1.1 on the first call (checkpoint load) |
| Simulate box scene, `physics` only | 0.16 |
| Simulate box scene, `learned` only (contact_gnn) | 0.8 |
| Simulate box scene, `both` with `mlp_multistep` | 0.26 |
| Simulate laptop (joint_gnn; physics null) | 1.2 |
| Simulate soft demo (contact_gnn + soft_gnn, 2 soft bodies; physics null) | 2.5-2.6 |
| Object edit, box scene (re-settle) | 0.27 |
| Object edit, jointed scene (mass only) | under 0.01 |
| Photo upload, 4000x2252 JPEG, 19 bodies | 23.8 first time; 0.27 cached |
| Video upload, 35 MB MP4, 5 bodies | 55.7 first time; 0.43 cached |
| Reconstruction asset (0.9 MB PNG) | 0.02 |

Earlier video runs on a busier GPU took 56-92 s (`docs/reconstruction-progress.md`). Other videos can take
longer; use a client timeout of at least 3 minutes for video and 1 minute for photos. The assistant with an
OpenAI key was not timed.

## Caching

- Scene records: kept in memory and in `DATA_DIR`; they survive a restart. Never evicted.
- Reconstructions: keyed by input content and pipeline configuration; a repeat returns `report.cache_hit: true`
  with a new `scene_id` (observed for photo and video). Empty or degraded video runs are recomputed.
- Simulations are not cached: every `POST /api/simulate` recomputes. Learned checkpoints are loaded on first use
  and stay in memory.
- `GET /api/metrics` reads the file on every call.

## Known inconsistencies

- `detail` has three shapes (string, validation array, `{message, report}`), see Errors.
- `events.tipped` tests orientation against world vertical, so a body that is tilted by design is reported as
  tipped with no action: the laptop demo returns `tipped: ["laptop_lid"]` at rest.
- `engine: "physics"` on a jointed / soft scene is a 200 with every result null and only a warning.
- `report.debug_url` is a prefix, not a fetchable URL.
- The no-key assistant response has no `warnings` key; the other statuses do.
- `mlp_multistep` results lack `projection_applied` and `approximations`.
- `default` and `auto` are two names for the same routing.
- `POST /api/scenes/demo` takes `scenario` as a query parameter and ignores its body.
- A soft body has no drawable geometry until a simulation has run.
- Physics frames carry full float precision; soft vertices are rounded to 4 decimals.
