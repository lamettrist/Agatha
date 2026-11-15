"""One learned-rollout entry point for the three trained models (box / joints / soft).

route(scene)    -> which rigid model runs and whether a soft pass follows.
rollout(...)    -> world_model.rollout-shaped dict, plus routing / joints / soft_meshes and per-frame soft_bodies.
reference(...)  -> PyBullet reference for box-only scenes; (None, warning) for jointed / soft scenes.

Composite rollout is ONE-WAY coupled: rigid bodies are rolled out first (contact_gnn, or joint_gnn when the
scene has joints) and ignore soft bodies; each soft body is then rolled out by soft_gnn with those rigid
trajectories as prescribed obstacles.  Soft bodies do not push rigid bodies or each other.
See docs/router.md.
"""
from functools import lru_cache
import itertools
import math
import os
from pathlib import Path
from typing import Any
import numpy as np
import pybullet as p  # quaternion math only
import torch
from . import world_model as wm
from .schema import Scene, Action, Joint, SoftBody
from .storage import ROOT

STEP = wm.STEP
JOINT_PROJECTION_ENV = 'AGATHA_JOINT_PROJECTION'
CHECKPOINTS = dict(joint_gnn=ROOT/'models/joint_gnn.pt', soft_gnn=ROOT/'models/soft_gnn.pt')
RIGID_ARCHITECTURES = ('mlp_multistep', 'graph_impulse', 'contact_gnn')
SOFT_RIGID_SLOTS = 4      # soft_gnn was trained with at most four rigid boxes next to the soft body
JOINT_SLOTS = 12          # joint_gnn training scenes: at most 12 bodies / 12 joints
COUPLING = ('One-way coupling: rigid bodies are predicted first and ignore soft bodies; each soft body then '
            'sees the predicted rigid boxes as prescribed obstacles. Soft bodies do not push rigid bodies or each other.')

def joint_projection_default() -> bool:
    """project_joints runs after each joint_gnn step unless AGATHA_JOINT_PROJECTION is 0/false/off/no."""
    return os.environ.get(JOINT_PROJECTION_ENV, '1').strip().lower() not in ('0', 'false', 'off', 'no')

# ------------------------------------------------------------------- routing
def route(scene: Scene) -> dict[str, Any]:
    """Joints present -> joint_gnn for all rigid bodies, else contact_gnn; soft bodies present -> soft pass."""
    reasons = []
    if scene.joints:
        rigid = 'joint_gnn'; reasons.append(f'{len(scene.joints)} joint(s): all rigid bodies go through joint_gnn')
    else:
        rigid = 'contact_gnn'; reasons.append('no joints: rigid bodies go through the default box model (contact_gnn)')
    soft = bool(scene.soft_bodies)
    if soft: reasons.append(f'{len(scene.soft_bodies)} soft body(ies): soft_gnn pass with the rigid trajectories as prescribed obstacles')
    return dict(rigid=rigid, soft=soft, reasons=reasons)

def _load(kind: str, path: str):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'Learned checkpoint missing ({kind}: {path.name}). Regenerate it with the README training commands; physics remains available.')
    torch.set_num_threads(4)
    try:
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        if kind == 'joint_gnn':
            from . import joint_gnn
            return joint_gnn.build_from_checkpoint(checkpoint)
        from . import soft_gnn
        return soft_gnn.build_from_checkpoint(checkpoint)
    except Exception as exc:
        raise RuntimeError(f'Learned checkpoint could not load ({kind}: {type(exc).__name__}); physics remains available') from exc

_load_cached = lru_cache(maxsize=4)(_load)

def load(kind: str):
    """joint_gnn / soft_gnn module from CHECKPOINTS[kind]; FileNotFoundError when the file is absent."""
    return _load_cached(kind, str(CHECKPOINTS[kind]))

# -------------------------------------------------------------------- joints
def joint_rows(joints: list[Joint], ids: list[str]) -> np.ndarray:
    """[J,18] rows in the agatha/joint_gnn.py layout for the joints whose bodies are in ids."""
    from .joint_gnn import JOINT_TYPES
    slot = {name: i for i, name in enumerate(ids)}; rows = []
    for j in joints:
        if j.child not in slot or (j.parent is not None and j.parent not in slot): continue
        axis = np.zeros(3); lower = upper = 0.
        if j.type in ('hinge', 'prismatic'):
            axis = np.asarray(j.axis, float); axis = axis/max(np.linalg.norm(axis), 1e-9); lower, upper = j.lower, j.upper
        rows.append([-1 if j.parent is None else slot[j.parent], slot[j.child], *[float(j.type == t) for t in JOINT_TYPES],
                     *j.anchor_parent, *j.anchor_child, *axis, lower, upper, j.damping])
    return np.asarray(rows, np.float32).reshape(-1, 18)

class JointStepper:
    """joint_gnn behind the LearnedModel.step interface (mirrors world_model.ContactGNNAdapter)."""
    kind = 'joint_gnn'
    def __init__(self, model, joints: list[Joint], projection: bool | None = None):
        from . import contact_gnn, joint_gnn
        self.cg, self.jg, self.model, self.joints = contact_gnn, joint_gnn, model, joints
        self.projection_applied = joint_projection_default() if projection is None else bool(projection)
        self._rows: tuple[tuple, torch.Tensor, torch.Tensor] | None = None
    def tensors(self, scene: Scene):
        ids = tuple(o.id for o in scene.objects)
        if self._rows is None or self._rows[0] != ids:
            rows = joint_rows(self.joints, list(ids))
            self._rows = (ids, torch.from_numpy(rows)[None], torch.ones(1, len(rows), dtype=torch.bool))
        return self._rows[1], self._rows[2]
    def step(self, scene: Scene, states: np.ndarray, gravity: float, tilt: list[float]) -> np.ndarray:
        n = len(states)
        if not n: return states.copy()
        state, props, context, mask = self.cg.scene_tensors(scene, states, gravity, tilt)
        for i, o in enumerate(scene.objects[:n]):
            if o.restitution: props[0, i, 6] = min(o.restitution, .6)
            props[0, i, 4] = min(max(o.friction, .1), 1.)
        joints, joint_mask = self.tensors(scene)
        with torch.inference_mode():
            output = self.model(state, props, context, mask, joints, joint_mask)
            if self.projection_applied:
                output = self.jg.project_joints(state, output, props, context, mask, joints, joint_mask)
        output = output[0, :n].numpy().astype(np.float32)
        if not np.isfinite(output).all(): raise ValueError('Learned model diverged')
        return output

class _Recorder:
    """Pass-through around a rigid model that keeps the 20 Hz states for the soft pass."""
    def __init__(self, model):
        self.model = model; self.kind = model.kind; self.states: list[np.ndarray] = []
    @property
    def projection_applied(self): return self.model.projection_applied
    def step(self, scene, states, gravity, tilt):
        output = self.model.step(scene, states, gravity, tilt); self.states.append(output)
        return output

# --------------------------------------------------- soft meshes (procedural)
# Copied from scripts/generate_soft_data.py (tet_grid, cube_to_ball, tet_edges, tet_surface, cloth_grid, tri_edges):
# the soft model was trained on exactly these meshes, and scripts/ is not importable from the served package.
def _tet_grid(nx, ny, nz):
    idx = lambda i, j, k: (i*ny+j)*nz+k
    pts = np.asarray([[i/(nx-1), j/(ny-1), k/(nz-1)] for i in range(nx) for j in range(ny) for k in range(nz)])-.5
    tets = []
    for i, j, k in itertools.product(range(nx-1), range(ny-1), range(nz-1)):
        corner = lambda v: idx(i+v[0], j+v[1], k+v[2])
        for perm in itertools.permutations(range(3)):
            v = [0, 0, 0]; path = [corner(v)]
            for axis in perm:
                v[axis] = 1; path.append(corner(v))
            tets.append(path)
    tets = np.asarray(tets, np.int64)
    a, b, c, d = (pts[tets[:, i]] for i in range(4))
    negative = np.einsum('ij,ij->i', np.cross(b-a, c-a), d-a) < 0
    tets[negative] = tets[negative][:, [0, 2, 1, 3]]
    return pts, tets

def _cube_to_ball(pts):
    q = 2*pts; x, y, z = q[:, 0], q[:, 1], q[:, 2]
    return np.stack([x*np.sqrt(1-y*y/2-z*z/2+y*y*z*z/3), y*np.sqrt(1-z*z/2-x*x/2+z*z*x*x/3),
                     z*np.sqrt(1-x*x/2-y*y/2+x*x*y*y/3)], 1)/2

def _tet_surface(pts, tets):
    count = {}
    for tet in tets:
        for skip in range(4):
            face = [tet[i] for i in range(4) if i != skip]
            count.setdefault(tuple(sorted(face)), []).append((face, tet[skip]))
    faces = []
    for key in sorted(count):
        if len(count[key]) != 1: continue
        (a, b, c), opposite = count[key][0]
        if np.dot(np.cross(pts[b]-pts[a], pts[c]-pts[a]), pts[opposite]-pts[a]) > 0: b, c = c, b
        faces.append((a, b, c))
    return np.asarray(faces, np.int64)

def _cloth_grid(nx, nz):
    pts = np.asarray([[i/(nx-1)-.5, 0., k/(nz-1)-.5] for i in range(nx) for k in range(nz)])
    tris = []
    for i in range(nx-1):
        for k in range(nz-1):
            a = i*nz+k; b = a+1; c = a+nz; d = c+1
            tris += [(a, b, c), (b, d, c)] if (i+k) % 2 else [(a, b, d), (a, d, c)]
    return pts, np.asarray(tris, np.int64)

@lru_cache(maxsize=32)
def soft_mesh(kind: str, size: tuple[float, float, float]) -> dict[str, Any]:
    """Local vertices, unique edges, surface faces and rest volume (cloth: area) of a procedural soft body."""
    extent = np.asarray(size, float)
    if kind == 'cloth':
        extent = np.asarray([size[0], 0., size[2]]); pts, tris = _cloth_grid(9, 9); pts = pts*extent
        a, b, c = (pts[tris[:, i]] for i in range(3))
        pairs = np.concatenate([tris[:, [i, j]] for i, j in ((0, 1), (1, 2), (0, 2))])
        return dict(pts=pts, edges=np.unique(np.sort(pairs, 1), axis=0), faces=tris,
                    measure=float(np.linalg.norm(np.cross(b-a, c-a), axis=1).sum()/2))
    res = (4, 4, 4) if kind == 'ball' else tuple(int(np.clip(round(s/(extent.max()/3))+1, 3, 5)) for s in extent)
    pts, tets = _tet_grid(*res)
    if kind == 'ball': pts = _cube_to_ball(pts)
    pts = pts*extent
    pairs = np.concatenate([tets[:, [i, j]] for i, j in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))])
    a, b, c, d = (pts[tets[:, i]] for i in range(4))
    return dict(pts=pts, edges=np.unique(np.sort(pairs, 1), axis=0), faces=_tet_surface(pts, tets),
                measure=float(np.abs(np.einsum('ij,ij->i', np.cross(b-a, c-a), d-a)).sum()/6))

def soft_props(body: SoftBody, measure: float) -> list[float]:
    """soft_props row of scripts/generate_soft_data.py; stiffness 0..1 maps onto the trained normalised range -1..1."""
    stiffness = 2*body.stiffness-1
    if body.kind == 'cloth':
        damping = float(np.clip(10**(1.48+stiffness)*2e-3, .015, .3))   # spring k = 10**(1.48+stiffness), mid damping ratio
        return [body.mass_kg, stiffness, damping, body.friction, 1., measure]
    return [body.mass_kg, stiffness, .02, body.friction, 0., measure]   # NeoHookean damping: middle of the trained 0.005-0.1

def soft_approximations(body: SoftBody) -> list[str]:
    notes = []; size = body.size
    trained = dict(cube=(.06, .2), ball=(.056, .24), cloth=(.2, .45))[body.kind]
    sides = (size[0], size[2]) if body.kind == 'cloth' else size
    if any(not trained[0]-1e-9 <= s <= trained[1]+1e-9 for s in sides):
        notes.append(f'soft body {body.id}: size outside the trained range ({trained[0]:g}-{trained[1]:g} m)')
    limit = .3 if body.kind == 'cloth' else 2.
    if not .03 <= body.mass_kg <= limit: notes.append(f'soft body {body.id}: mass outside the trained range (0.03-{limit:g} kg)')
    if not .3 <= body.friction <= 1: notes.append(f'soft body {body.id}: friction outside the trained range (0.3-1.0)')
    return notes

def _matrix(euler) -> np.ndarray:
    return np.asarray(p.getMatrixFromQuaternion(p.getQuaternionFromEuler(euler))).reshape(3, 3)

def _split_actions(scene: Scene, actions: list[Action]):
    """(rigid actions, surviving soft bodies, extra velocity per soft id); soft ids are handled here, the rest by world_model.prepare."""
    soft = {s.id: s.model_copy(deep=True) for s in scene.soft_bodies}; names = set(soft)
    push = {name: np.zeros(3) for name in soft}; rigid = []
    for a in actions:
        name = getattr(a, 'id', None)
        if name is None or name not in names: rigid.append(a); continue
        if name not in soft: raise ValueError(f'Unknown or removed object: {name}')
        if a.type == 'remove': del soft[name]
        elif a.type == 'set_mass': soft[name].mass_kg = a.kg
        elif a.type == 'push':
            direction = a.direction if len(a.direction) == 3 else (a.direction[0], 0, a.direction[1])
            push[name] += np.asarray(direction, float)*a.impulse_Ns/math.hypot(*a.direction)/soft[name].mass_kg
    return rigid, list(soft.values()), push

def _soft_pass(model, prepared: Scene, bodies: list[SoftBody], push: dict, rigid: np.ndarray, gravity: float, tilt: list[float]):
    """Vertex trajectories {id: [T+1,V,3]} at 20 Hz, mesh info and notes.  rigid: [T+1,n,13] predicted states."""
    from . import contact_gnn, soft_gnn
    count = rigid.shape[0]-1; n = rigid.shape[1]; notes = []; out = {}; meshes = {}; seeded = []
    _, props, context, mask = contact_gnn.scene_tensors(prepared, rigid[0], gravity, tilt)
    usable = np.flatnonzero(mask[0, :n].numpy()) if n else np.zeros(0, int)   # the desk that duplicates the support is analytic
    if max(abs(t) for t in tilt) > 1e-6: notes.append('soft_gnn was trained on a level support only; tilt is outside its training range')
    if not prepared.support.collisions_enabled: notes.append('soft_gnn was trained with support collisions on; the no-clip counterfactual is outside its training range')
    if not 2 <= gravity <= 15: notes.append(f'gravity {gravity:g} m/s2 is outside the soft_gnn training range (2-15)')
    for body in bodies:
        mesh = soft_mesh(body.kind, tuple(body.size)); pts = mesh['pts']
        verts = pts@_matrix(body.rotation_euler).T+np.asarray(body.position, float)
        velocity = np.asarray(body.velocity, float)+push[body.id]
        centre = verts.mean(0)
        order = sorted(usable, key=lambda i: float(np.linalg.norm(rigid[0, i, :3]-centre)))
        if len(order) > SOFT_RIGID_SLOTS:
            notes.append(f'soft body {body.id}: only the {SOFT_RIGID_SLOTS} nearest rigid boxes are obstacles ({len(order)-SOFT_RIGID_SLOTS} farther ones dropped)')
        order = order[:SOFT_RIGID_SLOTS]
        states = np.zeros((1, count+1, SOFT_RIGID_SLOTS, 13), np.float32); states[..., 6] = 1
        rigid_props = np.zeros((1, SOFT_RIGID_SLOTS, props.shape[-1]), np.float32); rigid_props[..., 3] = 1
        rigid_mask = np.zeros((1, SOFT_RIGID_SLOTS), bool)
        for k, i in enumerate(order):
            states[0, :, k] = rigid[:, i]; rigid_props[0, k] = props[0, i].numpy(); rigid_mask[0, k] = True
        v = torch.from_numpy(verts.astype(np.float32))[None]
        prev = soft_gnn.prev_from_velocity(v, torch.from_numpy(np.broadcast_to(velocity, verts.shape).astype(np.float32).copy())[None])
        edges = mesh['edges']
        rest = np.linalg.norm(pts[edges[:, 0]]-pts[edges[:, 1]], axis=1).astype(np.float32)
        args = (torch.ones(1, len(pts), dtype=torch.bool), torch.from_numpy(edges)[None], torch.from_numpy(rest)[None],
                torch.tensor([soft_props(body, mesh['measure'])], dtype=torch.float32), context)
        rigid_states, rigid_props, rigid_mask = torch.from_numpy(states), torch.from_numpy(rigid_props), torch.from_numpy(rigid_mask)
        with torch.inference_mode():
            # NON-LEARNED seed: the model needs two frames and was only trained from a body already in motion or
            # supported (from an exactly motionless mid-air start it hovers).  If one ballistic 0.05 s step touches
            # nothing, that step is frame 1 and the model takes over from there; otherwise the model starts at frame 0.
            ballistic = v+(v-prev)-torch.tensor([0., gravity*STEP*(STEP+1/240)/2, 0.])
            pushed = soft_gnn.project_soft(v, ballistic, args[0], context, rigid_states[:, min(1, count)], rigid_props, rigid_mask, margin=.005)
            if count > 1 and float((pushed-ballistic).abs().max()) < 1e-6:
                trajectory = torch.cat((v[:, None], soft_gnn.rollout(model, ballistic, v, count-1, *args, rigid_states[:, 1:], rigid_props, rigid_mask, project=True)), 1)
                seeded.append(body.id)
            else:
                trajectory = soft_gnn.rollout(model, v, prev, count, *args, rigid_states, rigid_props, rigid_mask, project=True)
        trajectory = trajectory[0].numpy().astype(np.float64)
        if not np.isfinite(trajectory).all(): raise ValueError('Learned soft model diverged')
        out[body.id] = trajectory
        meshes[body.id] = dict(faces=mesh['faces'].reshape(-1).tolist(), vertex_count=len(pts), kind=body.kind)
        notes += soft_approximations(body)
    if seeded: notes.append('first 0.05 s of airborne soft bodies ('+', '.join(seeded)+') is a non-learned ballistic step that seeds the model')
    return out, meshes, notes

def _close_interpolated_joints(frames: list[dict], joints: list[Joint]) -> None:
    """Frames between two 20 Hz model steps are linear blends, which opens hinge / ball / fixed anchors by a
    millimetre or two while a link rotates.  Re-seat each child on its anchor (orientation untouched), in joint order."""
    rot = lambda o: np.asarray(p.getMatrixFromQuaternion(o['quaternion'])).reshape(3, 3)
    for frame in frames:
        objects = {o['id']: o for o in frame['objects']}
        for j in joints:
            if j.type == 'prismatic': continue
            child = objects[j.child]
            anchor = np.asarray(j.anchor_parent, float) if j.parent is None else np.asarray(objects[j.parent]['position'])+rot(objects[j.parent])@np.asarray(j.anchor_parent, float)
            child['position'] = (anchor-rot(child)@np.asarray(j.anchor_child, float)).tolist()

def _check_joint_pose(stepper: JointStepper, prepared: Scene, initial: np.ndarray, gravity: float, tilt: list[float]) -> None:
    """ValueError (API: 422) when the scene poses do not satisfy the joints: the model would teleport the bodies in step 1."""
    if not len(initial): return
    state = stepper.cg.scene_tensors(prepared, initial, gravity, tilt)[0]
    joints, joint_mask = stepper.tensors(prepared)
    if not joints.shape[1]: return
    violation = stepper.jg.joint_violation(state, joints, joint_mask)
    gap, excess = float(violation['anchor_gap'].max()), float(violation['limit_excess'].max())
    if gap > .005 or excess > .1:
        raise ValueError(f'Scene pose does not satisfy its joints (anchor gap {gap*1000:.1f} mm, limit excess {excess:.2f}); '
                         'place the bodies so the joint anchors coincide and the joint is within its limits')

# ------------------------------------------------------------------- rollout
def rollout(scene: Scene, actions: list[Action], steps: int = 60, fps: int = 20, architecture: str = 'auto') -> dict[str, Any]:
    """Routed learned rollout; same dict as world_model.rollout plus routing / joints / soft_meshes (see module docstring)."""
    if fps not in (10, 20, 30) or not 1 <= steps <= 150: raise ValueError('Invalid rollout length/fps')
    decision = route(scene); reasons = decision['reasons']
    if architecture == 'joint_gnn' and not scene.joints: raise ValueError('learned_architecture joint_gnn needs a scene with joints')
    if architecture == 'soft_gnn' and not scene.soft_bodies: raise ValueError('learned_architecture soft_gnn needs a scene with soft bodies')
    if architecture in RIGID_ARCHITECTURES:
        if scene.joints and architecture != decision['rigid']: reasons.append(f'{architecture} was requested explicitly: joints are IGNORED by this model')
        decision['rigid'] = architecture
    elif architecture not in ('default', 'auto', 'joint_gnn', 'soft_gnn'): raise ValueError('Unknown learned architecture')
    rigid_actions, soft_bodies, push = _split_actions(scene, actions)
    if len(soft_bodies) != len(scene.soft_bodies):   # removed by the actions: restate the soft reason
        reasons[:] = [r for r in reasons if 'soft_gnn pass' not in r]+[
            f'{len(soft_bodies)} soft body(ies) left after the actions: soft_gnn pass with the rigid trajectories as prescribed obstacles'
            if soft_bodies else 'all soft bodies were removed by the actions: no soft_gnn pass']
    checkpoints = {}
    if decision['rigid'] == 'joint_gnn':
        model = JointStepper(load('joint_gnn'), scene.joints); checkpoints['joint_gnn'] = 'models/'+CHECKPOINTS['joint_gnn'].name
    else:
        name = 'default' if architecture not in RIGID_ARCHITECTURES else architecture
        model = wm.load_model(name)
        checkpoints[model.kind] = 'models/'+('world_model.pt' if name == 'default' else name+'.pt')
        decision['rigid'] = model.kind
    soft_model = None
    if soft_bodies:
        soft_model = load('soft_gnn'); checkpoints['soft_gnn'] = 'models/'+CHECKPOINTS['soft_gnn'].name
        model = _Recorder(model)
    # (a) rigid bodies; without soft bodies and joints this is exactly today's world_model.rollout call
    result = wm.rollout(scene, rigid_actions, steps, fps, model=model)
    prepared, initial, gravity, tilt = wm.prepare(scene, rigid_actions)
    ids = {o.id for o in prepared.objects}
    joints = [j for j in scene.joints if j.child in ids and (j.parent is None or j.parent in ids)]
    notes = list(result.get('approximations', []))
    if decision['rigid'] == 'joint_gnn':
        applied = model.model.projection_applied if soft_model else model.projection_applied
        _check_joint_pose(model.model if soft_model else model, prepared, initial, gravity, tilt)
        if len(joints) < len(scene.joints): notes.append('joints of removed objects were dropped')
        # joint_gnn shares contact_gnn's body inputs, so the same out-of-training notes apply (shapes, sizes, gravity 2-15, ...)
        notes += [note for note in wm.contact_approximations(prepared, gravity) if note not in notes]
        if max(abs(t) for t in tilt) > 1e-6: notes.append('joint_gnn was trained on a level support only; tilt is outside its training range')
        if not prepared.support.collisions_enabled: notes.append('joint_gnn was not trained on the no-clip counterfactual (support collisions off)')
        jointed = {name for j in joints for name in (j.parent, j.child)}
        if any(getattr(a, 'id', None) in jointed and a.type in ('push', 'toss') for a in rigid_actions):
            notes.append('a push on a jointed body is applied as a centre-of-mass velocity only (no joint-consistent spin), so it starts slower than in rigid-body physics')
        if len(prepared.objects) > JOINT_SLOTS or len(joints) > JOINT_SLOTS: notes.append(f'more than {JOINT_SLOTS} bodies/joints is outside the joint_gnn training range')
        result['projection_applied'] = applied
        if applied and fps == 30: _close_interpolated_joints(result['frames'], joints)
        disclaimer = ('Experimental learned joint graph network trained on synthetic PyBullet articulated boxes; ' +
                      ('a non-learned projection (contact push-out, then joint anchors/axes/limits closed) is applied after each step.' if applied
                       else 'no projection applied, so joints may open and objects may interpenetrate.') +
                      ' Contacts are not reported. Autoregressive error grows with time.')
    else:
        disclaimer = result['disclaimer']
        if scene.joints: notes.append('joints are ignored by the selected rigid model')
    # (b) soft bodies, one-way coupled
    meshes = {}
    if soft_model is not None:
        rigid = np.stack([initial]+model.states) if len(initial) else np.zeros((len(model.states)+1, 0, 13), np.float32)
        trajectories, meshes, soft_notes = _soft_pass(soft_model, prepared, soft_bodies, push, rigid, gravity, tilt)
        notes += soft_notes; count = rigid.shape[0]-1
        for i, frame in enumerate(result['frames']):
            u = i/fps/STEP; low = min(int(u), count); high = min(low+1, count); weight = u-low
            frame['soft_bodies'] = [dict(id=name, vertices=np.round((1-weight)*t[low]+weight*t[high], 4).reshape(-1).tolist())
                                    for name, t in trajectories.items()]
        soft_text = ' Soft bodies: experimental learned soft_gnn (vertices only, non-learned push-out of the table, floor and boxes). '+COUPLING
    else:
        soft_text = ''
        if scene.soft_bodies:
            for frame in result['frames']: frame['soft_bodies'] = []
    if soft_model is not None or decision['rigid'] == 'joint_gnn' or len(notes) != len(result.get('approximations', [])):
        # anything beyond today's box path: restate the disclaimer with everything that was approximated
        disclaimer = disclaimer.split(' Approximations: ')[0]+soft_text+(' Approximations: '+'; '.join(notes)+'.' if notes else '')
        result['approximations'] = notes
    result['model_architecture'] = decision['rigid']+('+soft_gnn' if soft_model is not None else '')
    result['disclaimer'] = disclaimer
    result['routing'] = dict(decision, soft=soft_model is not None, requested=architecture, checkpoints=checkpoints,
                             coupling='one_way_rigid_to_soft' if soft_model is not None else 'none')
    result['joints'] = [j.model_dump(mode='json') for j in joints]
    result['soft_meshes'] = meshes
    return result

# ----------------------------------------------------------------- reference
REFERENCE_WARNING = 'PyBullet reference is not wired for jointed/soft scenes in the API'

def reference(scene: Scene, actions: list[Action], duration: float, fps: int) -> tuple[dict | None, str | None]:
    """(physics result, warning).  Box-only scenes: agatha.physics.simulate, unchanged.  Scenes with joints or
    soft bodies: (None, warning), because agatha.physics does not model them and nothing is faked here."""
    if scene.joints or scene.soft_bodies:
        what = ' and '.join(k for k, v in (('joints', scene.joints), ('soft bodies', scene.soft_bodies)) if v)
        return None, f'{REFERENCE_WARNING} (this scene has {what}); no PyBullet result is returned.'
    from .physics import simulate
    return simulate(scene, actions, duration, fps), None
