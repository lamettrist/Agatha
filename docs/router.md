# Model router (box / joints / soft)

`agatha/router.py` is the single learned-rollout entry point. `POST /api/simulate` calls
`router.rollout` for the learned side and `router.reference` for the PyBullet side.

## How routing decides

`router.route(scene)` looks only at the scene:

| scene has | rigid bodies go through | soft pass |
|---|---|---|
| no joints, no soft bodies | `contact_gnn` (`models/world_model.pt`), exactly the previous code path | no |
| `joints` | `joint_gnn` (`models/joint_gnn.pt`) for ALL rigid bodies | no |
| `soft_bodies` | as above | yes, `soft_gnn` (`models/soft_gnn.pt`) |

`learned_architecture`:

- `default` / `auto`: the router decides.
- `mlp_multistep`, `graph_impulse`, `contact_gnn`: that rigid model, as before. Joints are then ignored
  (said in `routing.reasons` and `approximations`); a soft pass still runs if the scene has soft bodies.
- `joint_gnn` / `soft_gnn`: force that path; 422 if the scene has no joints / no soft bodies.

A missing checkpoint raises `FileNotFoundError` (API: 503 for `engine=learned`, a warning for `both`).

## Scene additions (`agatha/schema.py`, all optional)

- `joints: [{id, type: hinge|ball|prismatic|fixed, parent (object id or null = world), child, anchor_parent,
  anchor_child, axis, lower, upper, damping}]`. Anchors/axis are in the body frames (world when `parent` is
  null); at joint coordinate 0 the child has the parent's orientation (`agatha/joint_gnn.py`).
  The object poses in the scene must already satisfy the joint.
- `soft_bodies: [{id, label, kind: cube|ball|cloth, size, position, rotation_euler, mass_kg, stiffness (0-1),
  friction, velocity}]`. Meshes are the procedural training meshes (cube: 3-5 nodes per axis, ball: 4^3
  grid mapped to an ellipsoid, cloth: 9x9 grid in the local xz plane; `size[1]` is ignored for cloth).

## Response additions

Same dict as `world_model.rollout`, plus:

```
model_architecture: "contact_gnn" | "joint_gnn" | "contact_gnn+soft_gnn" | "joint_gnn+soft_gnn"
routing: {rigid, soft, reasons: [...], requested, checkpoints: {model: path}, coupling: "none" | "one_way_rigid_to_soft"}
joints: [...]                       # the scene joints that survived the actions
soft_meshes: {id: {faces: [i0,i1,i2,...], vertex_count, kind}}
frames[i].soft_bodies: [{id, vertices: [x0,y0,z0,...]}]   # only when the scene has soft bodies; 4 decimals
projection_applied, approximations  # also for joint_gnn
```

A box-only scene returns the same frames, events and disclaimer as before, bit for bit; only `routing`,
`joints: []` and `soft_meshes: {}` are added.

## Actions

`push` / `remove` / `set_mass` on a rigid id: as before. On a soft id: `push` adds impulse/mass to the
velocity of every vertex, `remove` drops the body, `set_mass` changes its mass input. `set_gravity` goes to
both models. `tilt_support` and `set_support_collisions` go to the rigid model; the soft model was not
trained on them, which is noted in `approximations`.

## Limits

- **One-way coupling.** Rigid bodies are predicted first and never feel a soft body. Each soft body then
  sees the predicted rigid boxes as prescribed obstacles (the 4 nearest; dropping farther ones is noted).
  Soft bodies do not collide with each other. A soft ball will not push a box; a box will dent the ball.
- **Soft seed.** The soft model needs two frames. A body that starts motionless in mid-air made the model
  hover (measured: about 0.6 s before it fell), so when one ballistic 0.05 s step touches nothing, that
  non-learned step is used as frame 1. Noted in `approximations`.
- **Joints.** Default is model + `project_joints`; `AGATHA_JOINT_PROJECTION=0` serves the raw model.
  At 30 fps the frames between two 20 Hz steps are linear blends; with the projection on, each jointed child
  is re-seated on its anchor in those frames. Joints to an object that duplicates the support slab are not handled.
- **Joint validation.** The schema rejects `lower > upper`, two joints on one child and joint cycles (joint_gnn
  was trained on trees). `router.rollout` raises `ValueError` (422) when the scene pose leaves a joint anchor more
  than 5 mm open or a limit exceeded by more than 0.1, instead of letting the model teleport the body.
  Tilt, no-clip, out-of-range gravity and pushes on jointed bodies (applied as a centre-of-mass velocity only)
  are listed in `approximations`.
- **Assistant and object edit.** `agatha/assistant.py` now calls `router.rollout` / `router.reference`, so on a
  jointed / soft scene its PyBullet tool reports the reference warning as a tool error and its learned tool uses
  the routed models. The object-edit endpoint (`POST /api/scenes/{id}/objects/{id}`) accepts only `mass_kg` for
  jointed / soft scenes and does not re-settle them; `shape` / `size` are a 422 (`docs/api.md`, route 6).
- All three models keep their own limits (`docs/joints-experimental.md`, `docs/soft-bodies-experimental.md`).
  Nothing here measures the accuracy of the composite; the tests only check sanity (finite, joints closed,
  no sinking into the table, a resting body stays put).

## PyBullet reference

`router.reference(scene, actions, duration, fps)` returns `(physics, warning)`.

- Box-only scenes: `agatha.physics.simulate`, unchanged.
- Scenes with joints or soft bodies: `physics` is `None`, with the warning "PyBullet reference is not wired
  for jointed/soft scenes in the API". `agatha.physics` does not model joints or deformables, and the
  generators' worlds (`scripts/generate_joint_data.py`, `scripts/generate_soft_data.py`) are not wired in.
  `error` is then `null` and no chaos score is recorded.

## Demo scenes

`POST /api/scenes/demo?scenario=laptop` (base + lid hinged at 100 degrees, limit 125, plus a mug) and
`?scenario=soft` (two boxes, a soft ball dropped from 8 cm, a cloth patch 5 cm above a box).

## Latency (CPU, 4 torch threads, 3 s at 30 fps, measured once on the dev machine)

| scene | architecture | seconds |
|---|---|---|
| desk | contact_gnn | 0.8-0.9 |
| laptop | joint_gnn | 1.4-1.5 |
| soft (2 soft bodies) | contact_gnn+soft_gnn | 2.9 |
