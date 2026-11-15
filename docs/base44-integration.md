# Frontend integration guide (Base44)

Practical guide for the separate frontend (`https://precognition.base44.app`). The full field-by-field
reference is [api.md](api.md); real recorded exchanges are in [api-examples.json](api-examples.json).

## Connection

- Local base URL: `http://127.0.0.1:8017`. A frontend served over HTTPS needs a reachable **HTTPS** API URL;
  `localhost` in the browser is the viewer's machine, not the server.
- CORS: `https://precognition.base44.app` is allowed by default. For other origins (previews) set
  `CORS_ORIGINS` in `.env` to the exact origins, comma-separated, and restart. Only `GET` and `POST` are allowed.
- No API key or auth header. Do not send credentials.
- `GET /api/health` tells you what the server supports (`demo_scenarios`, `learned_architectures`,
  `experimental_features`, `loaded_models`, `cuda_available`).

## Nothing changes for existing calls

Existing requests keep working and existing response fields keep their meaning; joints, soft bodies and the
routing fields are additions. The measured compatibility check is in
[api-compat-report.md](api-compat-report.md) (produced separately; see that file for the numbers).

The one behaviour change to be aware of: `learned_architecture: "default"` now runs the contact graph network
(`model_architecture: "contact_gnn"`) instead of the earlier MLP.

## Helpers

```javascript
const API = 'https://YOUR-API-HOST';

// `detail` is a string, a validation array, or {message, report} (video). Turn any of them into text.
function errorText(detail) {
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) return detail.map(d => `${(d.loc || []).slice(1).join('.')}: ${d.msg}`).join('; ');
  if (detail && detail.message) return detail.message;
  return JSON.stringify(detail);
}

async function call(path, { method = 'GET', json, form, timeoutMs = 30000 } = {}) {
  const response = await fetch(API + path, {
    method,
    headers: json ? { 'Content-Type': 'application/json' } : undefined,
    body: json ? JSON.stringify(json) : form,
    signal: AbortSignal.timeout(timeoutMs),
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    const error = new Error(errorText(body.detail ?? body));
    error.status = response.status;
    error.report = body.detail?.report;   // video: reconstruction report of the failed run
    throw error;
  }
  return body;
}
```

## Load a demo scene

`scenario` is a query parameter. All four return the same record `{scene_id, scene, report}`.

```javascript
const loadDemo = scenario => call(`/api/scenes/demo?scenario=${scenario}`, { method: 'POST', json: {} });

const desk   = await loadDemo('desk');     // boxes
const bottle = await loadDemo('bottle');   // boxes
const laptop = await loadDemo('laptop');   // EXPERIMENTAL: hinge between laptop_base and laptop_lid
const soft   = await loadDemo('soft');     // EXPERIMENTAL: soft ball and cloth

const { scene_id, scene } = laptop;
// scene.objects[], scene.support, scene.joints[], scene.soft_bodies[]
```

## Upload a photo or a video

Uploads are slow (measured: photo 24 s, video 56 s, more on a busy GPU). Use long timeouts and a progress state.

```javascript
async function uploadPhoto(file, tableWidthM) {
  const form = new FormData();
  form.append('image', file);
  if (tableWidthM) form.append('table_width_m', String(tableWidthM));
  return call('/api/scenes', { method: 'POST', form, timeoutMs: 90000 });
}

async function uploadVideo(file) {
  const form = new FormData();
  form.append('file', file);
  return call('/api/scenes/video', { method: 'POST', form, timeoutMs: 240000 });
}

try {
  const record = await uploadVideo(file);
  const { quality_status, warnings = [], body_count, dynamic_body_count } = record.report;
  // A 200 always has objects. needs_review = a measured check failed: show the scene AND the warnings.
  showScene(record.scene);
  showBadge(quality_status, `${body_count} bodies, ${dynamic_body_count} movable`);
  warnings.forEach(showWarning);
} catch (error) {
  if (error.status === 422) showError('Nothing could be reconstructed: ' + error.message);
  else if (error.status === 503) showError('Reconstruction models are unavailable on the server.');
  else if (error.status === 413) showError('File too large (photo 10 MB, video 100 MB).');
  else if (error.name === 'TimeoutError') showError('The server is still working; try again in a minute.');
  else showError(error.message);
}
```

Never show an HTTP error or a timeout as "0 objects detected". `object.visual_mesh_url` and
`report.debug_url + '/masks_overlay.png'` are server-relative: prefix them with `API`. `report.debug_url` alone
is not a page (404).

## Run a simulation

```javascript
const result = await call('/api/simulate', {
  method: 'POST',
  timeoutMs: 30000,
  json: {
    scene_id,
    actions: [{ type: 'push', id: 'mug', direction: [1, 0], impulse_Ns: 0.55 }],
    engine: 'both',        // 'physics' | 'learned' | 'both'
    duration: 3,           // 0.1 .. 5 s
    fps: 30,               // 10 | 20 | 30
    // learned_architecture: 'default'  -> the server picks the model from the scene
  },
});
const { physics, learned, error, warnings } = result;
```

All action shapes:

```json
[
  {"type":"push","id":"mug","direction":[1,0],"impulse_Ns":0.55},
  {"type":"push","id":"mug","direction":[0,1,0],"impulse_Ns":0.3},
  {"type":"remove","id":"phone"},
  {"type":"set_mass","id":"book1","kg":0.8},
  {"type":"tilt_support","angle_deg":15,"axis":"z"},
  {"type":"set_gravity","g":3},
  {"type":"set_support_collisions","enabled":false}
]
```

### `physics` can be null

For scenes with joints or soft bodies there is no PyBullet reference: `physics` and `error` are `null`, and
`warnings` says why. Pick whichever result exists and always show the warnings.

```javascript
const run = physics ?? learned;          // jointed / soft scenes: only `learned`
if (!run) { showWarning(warnings.join(' ') || 'No result for this engine.'); return; }
warnings.forEach(showWarning);
if (!physics) hideComparisonPanel();     // no physics, so no error curve and no leaderboard score
else if (error) showErrorCurve(error.per_frame, error.mean_position_error_m, error.fell_agreement);
```

Asking `engine: 'physics'` for such a scene returns 200 with `physics: null` and `learned: null`. Use `both`
or `learned`.

### Animate rigid bodies

Positions are centres in metres, Y up, quaternions are xyzw: exactly what three.js expects.

```javascript
// meshes: Map id -> THREE.Mesh, built from scene.objects (BoxGeometry(...o.size), or the GLB at visual_mesh_url)
function applyFrame(frame, meshes) {
  for (const o of frame.objects) {
    const mesh = meshes.get(o.id);
    if (!mesh) continue;
    mesh.position.fromArray(o.position);
    mesh.quaternion.fromArray(o.quaternion);
  }
  // objects removed by a `remove` action are absent from every frame: hide their meshes
  const present = new Set(frame.objects.map(o => o.id));
  meshes.forEach((mesh, id) => { mesh.visible = present.has(id); });
}

// Table: centre at y = height - thickness/2; tilt comes from the result
table.rotation.set(...run.support_rotation_euler);
table.visible = run.support_collisions_enabled;   // or draw it translucent

const start = performance.now();
function tick() {
  const t = (performance.now() - start) / 1000;
  const index = Math.min(run.frames.length - 1, Math.round(t * 30));   // same fps as the request
  applyFrame(run.frames[index], meshes);
  applySoftFrame(run, run.frames[index], softMeshes);                  // see below; no-op without soft bodies
  if (index < run.frames.length - 1) requestAnimationFrame(tick);
}
tick();
```

### Render soft bodies (EXPERIMENTAL)

Soft bodies exist only in `learned`. Topology is in `learned.soft_meshes[id].faces` (flat triangle indices,
constant); positions are in every frame as `frame.soft_bodies[i].vertices` (flat xyz in world space).

```javascript
function buildSoftMeshes(learned, threeScene) {
  const softMeshes = new Map();
  const first = learned.frames[0].soft_bodies ?? [];
  for (const body of first) {
    const info = learned.soft_meshes[body.id];            // {faces, vertex_count, kind}
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(body.vertices, 3));
    geometry.setIndex(info.faces);
    geometry.computeVertexNormals();
    const material = new THREE.MeshStandardMaterial({
      color: info.kind === 'cloth' ? 0x4f8fd9 : 0xd9774f,
      side: info.kind === 'cloth' ? THREE.DoubleSide : THREE.FrontSide,
      flatShading: false,
    });
    const mesh = new THREE.Mesh(geometry, material);       // leave position/rotation at identity: vertices are world space
    mesh.frustumCulled = false;
    threeScene.add(mesh);
    softMeshes.set(body.id, mesh);
  }
  return softMeshes;
}

function applySoftFrame(run, frame, softMeshes) {
  for (const body of frame.soft_bodies ?? []) {
    const mesh = softMeshes.get(body.id);
    if (!mesh) continue;
    const position = mesh.geometry.attributes.position;
    position.array.set(body.vertices);
    position.needsUpdate = true;
    mesh.geometry.computeVertexNormals();
  }
}
```

Before the first simulation there are no vertices. Draw a placeholder from `scene.soft_bodies[i]`
(`kind`, `size`, `position`): a sphere for `ball`, a box for `cube`, a flat plane of `size[0]` x `size[2]` for
`cloth`.

### Draw joints (EXPERIMENTAL)

`scene.joints` (and `learned.joints` in a result) describe the joints. The anchor is given in the parent's
body frame; compute its world position from the parent's pose in the current frame.

```javascript
function jointWorld(joint, frame) {
  const pose = id => frame.objects.find(o => o.id === id);
  const anchor = new THREE.Vector3().fromArray(joint.anchor_parent);
  const axis = new THREE.Vector3().fromArray(joint.axis).normalize();
  if (joint.parent !== null) {                             // null parent = world frame already
    const parent = pose(joint.parent);
    if (!parent) return null;                              // parent was removed
    const q = new THREE.Quaternion().fromArray(parent.quaternion);
    anchor.applyQuaternion(q).add(new THREE.Vector3().fromArray(parent.position));
    axis.applyQuaternion(q);
  }
  return { anchor, axis };
}

// hinge / prismatic: a short line along the axis through the anchor; ball / fixed: a small sphere at the anchor
function drawJoint(joint, frame, marker) {
  const world = jointWorld(joint, frame);
  marker.visible = !!world;
  if (!world) return;
  marker.position.copy(world.anchor);
  if (joint.type === 'hinge' || joint.type === 'prismatic')
    marker.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), world.axis);   // marker = thin cylinder along Y
}
```

The bodies themselves need no special handling: jointed bodies are ordinary entries in `frame.objects`.

### Model label and disclaimer

```javascript
if (learned) {
  const label = {
    'contact_gnn': 'Contact graph network',
    'joint_gnn': 'Joint graph network (experimental)',
    'contact_gnn+soft_gnn': 'Contact + soft-body networks (experimental)',
    'joint_gnn+soft_gnn': 'Joint + soft-body networks (experimental)',
  }[learned.model_architecture] ?? learned.model_architecture;
  showModelLabel(label);
  showTooltip(learned.routing.reasons.join(' '));
  const experimental = learned.routing.rigid === 'joint_gnn' || learned.routing.soft;
  showDisclaimer(learned.disclaimer, { experimental });       // always show the server's disclaimer text
  (learned.approximations ?? []).forEach(showNote);           // absent for mlp_multistep
}
```

`events.tipped` lists any body whose up axis is more than 35 degrees from vertical, so the open laptop lid is
always in it. Do not present `tipped` as "knocked over" for jointed scenes.

## Before / after card from /api/metrics

```javascript
const metrics = await call('/api/metrics');
const before = metrics.contact_v2.old.all;             // previous MLP default
const after  = metrics.contact_v2.new_projected.all;   // what the server runs now
card({
  positionErrorAt2s: [before.pos_err_m['2s'], after.pos_err_m['2s']],   // metres
  meanPenetration:   [before.mean_penetration_m, after.mean_penetration_m],
  fellAccuracy:      [before.fell_accuracy, after.fell_accuracy],
  episodes: after.n_episodes,
  caption: 'Held-out synthetic box scenes. Includes a non-learned contact projection. No joints or soft bodies.',
});
```

Values can be `null` (the server converts NaN). The file has no numbers for joints or soft bodies.

## Leaderboard

```javascript
if (physics) {   // only box scenes get a score
  await call('/api/leaderboard', { method: 'POST',
    json: { name: teamName, scene_id, chaos_score: physics.chaos_score } });   // must be the latest physics run
}
const { entries } = await call('/api/leaderboard');   // [{name, scene_id, chaos_score}], top 20
```

A 422 here means the score does not match the last physics run of that scene, or physics was never run.

## Assistant

```javascript
const reply = await call('/api/assistant/query', { method: 'POST', timeoutMs: 120000,
  json: { scene_id, question: 'Look up the mug mass, then push it and compare both engines.' } });
if (reply.status === 'unavailable') showNote(reply.answer);       // no OPENAI_API_KEY on the server
else { showAnswer(reply.answer); (reply.warnings ?? []).forEach(showWarning); }
// reply.simulation_results: summaries (no frames). To animate, re-run reply.actions_ran[i].actions via /api/simulate.
```

On a jointed or soft scene the assistant reports that no PyBullet reference exists and gives the learned
result only.

## Editing an object

```javascript
const edited = await call(`/api/scenes/${scene_id}/objects/${objectId}`, { method: 'POST',
  json: { mass_kg: 0.4 /*, shape: 'cylinder', size: [0.09, 0.1, 0.09] */ } });
sceneId = edited.scene_id;   // a NEW scene; switch to it
```

For scenes with joints or soft bodies only `mass_kg` is accepted; `shape` / `size` return 422. Disable those
inputs when `scene.joints.length || scene.soft_bodies.length`.

## Checklist

### To support joints and soft bodies, change this

- [ ] Stop assuming `result.physics` exists. Use `physics ?? learned`; hide the comparison and the score
      submission when `physics` is null.
- [ ] Show `result.warnings` (they explain the missing physics).
- [ ] Offer the two new demo scenarios: `laptop`, `soft`.
- [ ] Soft bodies: build one `BufferGeometry` per soft body from `learned.soft_meshes[id].faces` and update
      positions from `frame.soft_bodies[].vertices` each frame. Draw a placeholder before the first run.
- [ ] Joints: optional marker from `scene.joints` (anchor + axis). The bodies animate without it.
- [ ] Show `learned.model_architecture`, and an "experimental" tag when `routing.rigid === 'joint_gnn'` or
      `routing.soft`. Show `learned.disclaimer`.
- [ ] Do not treat `events.tipped` as an alarm for jointed scenes.
- [ ] In the object editor, allow only mass for jointed / soft scenes.
- [ ] Handle `detail` being a string, an array or an object.

### You can ignore

- `scene.joints` and `scene.soft_bodies` on box scenes: they are empty arrays.
- `learned.routing`, `joints`, `soft_meshes`, `approximations`, `projection_applied`, `contacts_available` if you
  only show box scenes.
- `learned_architecture`: leave it out; `default` routes correctly. `mlp_multistep` and `graph_impulse` are
  earlier baselines.
- Scene object fields `collision`, `shape_method`, `scale`, `restitution`, damping, `soft_approx`: needed by the
  simulator, not for drawing a box of `size` (or the GLB at `visual_mesh_url`).
- Report diagnostics (`stages`, `metrics`, `detections`, `camera`, ...).
- `POST /api/scenes/scan`, unless you add LiDAR scan upload.
