"""Landing benchmark: rotate a box to an arbitrary 3D orientation, release it, and score how it comes to rest.

PyBullet (agatha.physics.World, the live physics path) is the ground truth; every learned engine is
rolled out autoregressively for 4 s (80 steps of 0.05 s) from the same initial state.  Groups:

  single_default   one box, random orientation, lowest corner 0-0.3 m above the table, friction 0.5
  single_friction  same, object friction 0.2-1.0
  multi            a tilted box released next to / onto 1-3 boxes resting on the table
  edge_rest        zero lift, the box starts at rest on one bottom edge, tilted 3-80 deg about it
  stuck_examples   the four cases from the bug report (size / euler / lift), repeated at 8 yaw angles

Engines: model_only (learned model, no projection), projection_legacy (contact_gnn.project(landing=False):
the projection as shipped before the landing fix), projection (contact_gnn.project defaults) and the
projection variants in VARIANTS.  With --project-file an extra engine uses project() from another copy of
agatha/contact_gnn.py (e.g. a snapshot of the old file).

Metrics are for the released box (slot 0) at the end of the rollout unless stated otherwise:
  tilt            angle (deg) between its nearest face normal and the support normal (0 = a face is flat)
  same_face       the same face (axis and sign) is down as in PyBullet
  pos_err         distance to PyBullet's final position (m)
  t_rest          first time after which it never again moves > 1 mm or > 0.5 deg in a step
  moving          still moving in the last 0.25 s
  frozen          at rest, tilt > 8 deg, not touching another box, and its centre of mass is not above the
                  convex hull of its lowest corners (within 3 mm of the lowest): frozen mid-topple
  penetration     contact_gnn.penetration_depths, max over support / floor / objects (mm)

Usage:
  CUDA_VISIBLE_DEVICES=0 .venv-train/bin/python scripts/eval_landing.py --model models/contact_gnn.pt \
      --cache <dir> --out models/runs/landing_eval.json [--device cuda:0] [--variants all|none|name,name]
"""
import argparse, hashlib, importlib.util, inspect, json, math, sys, time
from multiprocessing import Pool
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

STEP = .05
STEPS = 80
SLOTS = 4
MOVE_M, MOVE_DEG = .001, .5          # per 0.05 s step: 0.02 m/s, ~0.17 rad/s
TABLE_H = .75
GROUPS = ('single_default', 'single_friction', 'multi', 'edge_rest', 'stuck_examples')
STUCK = [((.2, .04, .15), (-96, -23, 171), 0.), ((.07, .24, .07), (-94, 148, 138), .1),
         ((.15, .1, .06), (30, -106, 171), 0.), ((.3, .03, .2), (-94, -123, 20), .03)]
# Projection variants: name -> keyword arguments of contact_gnn.project (skipped when project() lacks a keyword).
VARIANTS = {
    'projection_legacy': dict(landing=False),
    'no_false_sleep': dict(landing=True, settle_deg=0., tip=0.),
    'settle3': dict(landing=True, settle_deg=3., tip=0.),
    'settle6': dict(landing=True, settle_deg=6., tip=0.),
    'tip': dict(landing=True, settle_deg=0., tip=1., tip_ang=.25),
    'settle3+tip': dict(landing=True, settle_deg=3., tip=1., tip_ang=.25),
    'settle6+tip': dict(landing=True, settle_deg=6., tip=1., tip_ang=.25),
    'settle10+tip': dict(landing=True, settle_deg=10., tip=1., tip_ang=.25),
    'settle6+tip_ang1': dict(landing=True, settle_deg=6., tip=1., tip_ang=1.),
    'settle6+tip_half': dict(landing=True, settle_deg=6., tip=.5, tip_ang=.25),
}

# ----------------------------------------------------------------------------- cases

def _quat_mul(a, b):
    ax, ay, az, aw = a; bx, by, bz, bw = b
    return np.array([aw*bx+ax*bw+ay*bz-az*by, aw*by-ax*bz+ay*bw+az*bx, aw*bz+ax*by-ay*bx+az*bw, aw*bw-ax*bx-ay*by-az*bz])

def _axis_quat(axis, angle):
    axis = np.asarray(axis, float); axis = axis/np.linalg.norm(axis)
    return np.array([*(axis*math.sin(angle/2)), math.cos(angle/2)])

def _rotmat(q):
    x, y, z, w = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])

def _low(size, q):
    """Height of the box centre above its lowest corner."""
    return float((np.abs(_rotmat(q)[1])*np.asarray(size)/2).sum())

def _size(rng, style):
    j = lambda v: float(v*rng.uniform(.85, 1.15))
    if style == 'cube': s = rng.uniform(.04, .15); return (s, s, s)
    if style == 'tall': b = rng.uniform(.04, .1); return (b, float(rng.uniform(.15, .3)), float(b*rng.uniform(.8, 1.25)))
    if style == 'slab': return (float(rng.uniform(.15, .3)), float(rng.uniform(.02, .04)), float(rng.uniform(.12, .25)))
    if style == 'bottle': return (j(.07), j(.24), j(.07))
    if style == 'book': return (j(.2), j(.035), j(.14))
    return tuple(float(v) for v in rng.uniform(.03, .3, 3))

STYLES = ('cube', 'tall', 'slab', 'bottle', 'book', 'random')

def _obj(name, size, pos, q, mass, friction):
    return dict(id=name, size=[float(v) for v in size], position=[float(v) for v in pos], quaternion=[float(v) for v in q],
                mass_kg=float(mass), friction=float(friction))

def _random_quat(rng):
    q = rng.normal(size=4); return q/np.linalg.norm(q)

def make_cases(seed: int, singles: int, friction: int, multi: int, edge: int) -> list[dict]:
    from pybullet import getQuaternionFromEuler
    rng = np.random.default_rng(seed); cases = []
    def single(group, fr):
        style = STYLES[len(cases) % len(STYLES)]; size = _size(rng, style); q = _random_quat(rng)
        lift = 0. if rng.random() < .15 else float(rng.uniform(0, .3))
        pos = (rng.uniform(-.35, .35), TABLE_H+lift+_low(size, q), rng.uniform(-.2, .2))
        mass = float(np.exp(rng.uniform(np.log(.05), np.log(2.))))
        cases.append(dict(group=group, style=style, lift=lift, objects=[_obj('box', size, pos, q, mass, fr)]))
    for _ in range(singles): single('single_default', .5)
    for _ in range(friction): single('single_friction', float(rng.uniform(.2, 1.)))
    for _ in range(multi):
        style = STYLES[len(cases) % len(STYLES)]; size = _size(rng, style); q = _random_quat(rng)
        others = []
        for k in range(int(rng.integers(1, SLOTS))):
            for _try in range(30):
                osize = _size(rng, STYLES[int(rng.integers(len(STYLES)))])
                osize = tuple(sorted(osize, reverse=True)[i] for i in ((0, 2, 1) if rng.random() < .7 else (2, 0, 1)))   # flat-ish or upright
                opos = np.array([rng.uniform(-.3, .3), TABLE_H+osize[1]/2, rng.uniform(-.18, .18)])
                radius = math.hypot(osize[0], osize[2])/2
                if all(np.hypot(*(opos-o[1])[[0, 2]]) > radius+o[2]+.01 for o in others):
                    others.append((osize, opos, radius, _axis_quat((0, 1, 0), rng.uniform(0, 2*math.pi)))); break
        target = others[int(rng.integers(len(others)))]
        angle = rng.uniform(0, 2*math.pi); reach = rng.uniform(0, target[2]+.12)
        xz = target[1][[0, 2]]+reach*np.array([math.cos(angle), math.sin(angle)])
        mine = float(np.linalg.norm(size))/2
        base = max([TABLE_H]+[o[1][1]+o[0][1]/2 for o in others if np.hypot(*(xz-o[1][[0, 2]])) < mine+o[2]+.005])
        lift = 0. if rng.random() < .15 else float(rng.uniform(0, .2))
        mass = float(np.exp(rng.uniform(np.log(.05), np.log(2.))))
        objects = [_obj('box', size, (xz[0], base+.001+lift+_low(size, q), xz[1]), q, mass, .5)]
        objects += [_obj(f'rest{i}', o[0], o[1], o[3], float(np.exp(rng.uniform(np.log(.1), np.log(2.)))), .5) for i, o in enumerate(others)]
        cases.append(dict(group='multi', style=style, lift=lift, objects=objects))
    turns = [np.array([0, 0, 0, 1.]), _axis_quat((1, 0, 0), math.pi/2), _axis_quat((0, 0, 1), math.pi/2)]
    for _ in range(edge):
        style = STYLES[len(cases) % len(STYLES)]; size = _size(rng, style)
        q = _quat_mul(_axis_quat((1, 0, 0), math.radians(rng.uniform(3, 80))*(1 if rng.random() < .5 else -1)), turns[int(rng.integers(3))])
        q = _quat_mul(_axis_quat((0, 1, 0), rng.uniform(0, 2*math.pi)), q)
        pos = (rng.uniform(-.3, .3), TABLE_H+_low(size, q), rng.uniform(-.15, .15))
        cases.append(dict(group='edge_rest', style=style, lift=0., objects=[_obj('box', size, pos, q, float(np.exp(rng.uniform(np.log(.05), np.log(2.)))), .5)]))
    for size, euler, lift in STUCK:
        for k in range(8):
            q = _quat_mul(_axis_quat((0, 1, 0), k*math.pi/4), np.array(getQuaternionFromEuler([math.radians(v) for v in euler])))
            cases.append(dict(group='stuck_examples', style='report', lift=lift,
                              objects=[_obj('box', size, (0., TABLE_H+lift+_low(size, q), 0.), q, .5, .5)]))
    return cases

def case_scene(case: dict):
    from pybullet import getEulerFromQuaternion
    from agatha.schema import Scene
    return Scene.model_validate(dict(objects=[dict(id=o['id'], label=o['id'], size=o['size'], position=o['position'],
        rotation_euler=list(getEulerFromQuaternion(o['quaternion'])), mass_kg=o['mass_kg'], friction=o['friction']) for o in case['objects']]))

def simulate_truth(case: dict) -> np.ndarray:
    """[STEPS+1, n, 13] PyBullet states at 20 Hz (240 Hz substeps), exactly the live physics World."""
    from agatha.physics import World
    from agatha.world_model import state_array
    scene = case_scene(case); ids = [o.id for o in scene.objects]
    with World(scene) as world:
        frames = [state_array(world.snapshot(0), ids)]
        for i in range(STEPS):
            world.step(12); frames.append(state_array(world.snapshot((i+1)*STEP), ids))
    return np.stack(frames)

def load_truth(cases: list[dict], cache: Path | None, workers: int) -> np.ndarray:
    key = hashlib.sha1(json.dumps(cases, sort_keys=True).encode()).hexdigest()[:16]
    path = cache/f'landing_truth_{key}.npz' if cache else None
    if path and path.exists(): return np.load(path)['states']
    with Pool(workers) as pool: parts = pool.map(simulate_truth, cases, chunksize=8)
    states = np.zeros((len(cases), STEPS+1, SLOTS, 13), np.float32); states[..., 6] = 1
    for i, part in enumerate(parts): states[i, :, :part.shape[1]] = part
    if path: path.parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(path, states=states)
    return states

def case_tensors(cases: list[dict]):
    """props / context / mask exactly as world_model.ContactGNNAdapter.step builds them (padded to SLOTS)."""
    import torch
    from agatha import contact_gnn
    props = np.zeros((len(cases), SLOTS, contact_gnn.PROPS_DIM), np.float32); props[..., 3] = 1
    mask = np.zeros((len(cases), SLOTS), bool)
    context = np.tile(np.array([[1.2, .8, TABLE_H, 9.81, 0, 0, 0, 1., .5, 0.]], np.float32), (len(cases), 1))
    for i, case in enumerate(cases):
        for j, o in enumerate(case['objects']):
            props[i, j, :6] = [*o['size'], o['mass_kg'], min(max(o['friction'], .1), 1.), 0.]; mask[i, j] = True
    return torch.from_numpy(props), torch.from_numpy(context), torch.from_numpy(mask)

# ----------------------------------------------------------------------------- metrics

def _rot_rows(quat: np.ndarray) -> np.ndarray:
    """World-up row of the rotation matrix for xyzw quaternions [...,4] -> [...,3]."""
    x, y, z, w = np.moveaxis(quat/np.linalg.norm(quat, axis=-1, keepdims=True), -1, 0)
    return np.stack((2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)), -1)

def tilt_and_face(quat: np.ndarray):
    """Tilt (deg) of the nearest face from horizontal and the id (axis*2 + (normal points up)) of the face that is down."""
    row = _rot_rows(quat); axis = np.abs(row).argmax(-1)
    value = np.take_along_axis(row, axis[..., None], -1)[..., 0]
    return np.degrees(np.arccos(np.clip(np.abs(value), 0, 1))), axis*2+(value > 0)

def _rotation(quat: np.ndarray) -> np.ndarray:
    x, y, z, w = np.moveaxis(quat/np.linalg.norm(quat, axis=-1, keepdims=True), -1, 0)
    return np.stack((np.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)), -1), np.stack((2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)), -1),
                     np.stack((2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), -1)), -2)

SIGNS = np.array([[x, y, z] for x in (-1., 1.) for y in (-1., 1.) for z in (-1., 1.)])
RING = np.stack((np.cos(np.arange(32)*math.pi/16), np.sin(np.arange(32)*math.pi/16)), -1)

def com_supported(quat: np.ndarray, size: np.ndarray, band: float = .003) -> np.ndarray:
    """[C] the centre of mass is above the convex hull of the lowest corners (those within `band` of the lowest)."""
    offsets = np.einsum('kj,cij->cki', SIGNS, _rotation(quat)*(size/2)[:, None, :])        # [C,8,3] relative to the COM
    contact = offsets[..., 1] <= offsets[..., 1].min(-1, keepdims=True)+band
    reach = offsets[..., [0, 2]]@RING.T                                                    # [C,8,32]
    return (np.where(contact[..., None], reach, np.inf).min(1) <= 1e-4).all(-1)

def step_motion(traj: np.ndarray):
    """Per-step displacement (m) and rotation (deg) of slot 0: [C, T]."""
    move = np.linalg.norm(np.diff(traj[:, :, 0, :3], axis=1), axis=-1)
    q = traj[:, :, 0, 3:7]/np.linalg.norm(traj[:, :, 0, 3:7], axis=-1, keepdims=True)
    turn = np.degrees(2*np.arccos(np.clip(np.abs((q[:, 1:]*q[:, :-1]).sum(-1)), 0, 1)))
    return move, turn

def near_other(state: np.ndarray, sizes: np.ndarray, mask: np.ndarray, gap: float = .005) -> np.ndarray:
    """[C] slot 0 touches (within gap) another box, tested with corners both ways plus bounding-sphere prefilter."""
    out = np.zeros(len(state), bool)
    rot = _rotation(state[..., 3:7])
    corners = state[:, :, None, :3]+np.einsum('kj,cnij->cnki', SIGNS, rot*(sizes/2)[:, :, None, :])
    def inside(points, j):
        local = np.einsum('cki,cij->ckj', points-state[:, j, None, :3], rot[:, j])
        return (np.abs(local) <= (sizes[:, j]/2)[:, None]+gap).all(-1).any(-1)
    for j in range(1, state.shape[1]):
        # corners, edge midpoints and face centres of each against the other
        grid = np.array([[x, y, z] for x in (-1., 0., 1.) for y in (-1., 0., 1.) for z in (-1., 0., 1.)])
        pa = state[:, 0, None, :3]+np.einsum('kj,cij->cki', grid, rot[:, 0]*(sizes[:, 0]/2)[:, None, :])
        pb = state[:, j, None, :3]+np.einsum('kj,cij->cki', grid, rot[:, j]*(sizes[:, j]/2)[:, None, :])
        out |= mask[:, j] & (inside(pa, j) | inside(pb, 0))
    return out

def summarize(values: np.ndarray) -> dict:
    if not len(values): return dict(n=0)
    return dict(median=float(np.median(values)), mean=float(values.mean()), p90=float(np.percentile(values, 90)), max=float(values.max()))

def score(traj: np.ndarray, truth: np.ndarray, sizes: np.ndarray, mask: np.ndarray, pen: np.ndarray, select: np.ndarray) -> dict:
    """Metrics of slot 0 over the cases in `select` (bool [C])."""
    t, g, sz, mk, pn = traj[select], truth[select], sizes[select], mask[select], pen[select]
    count = len(t)
    if not count: return dict(cases=0)
    tilt, face = tilt_and_face(t[:, :, 0, 3:7]); gtilt, gface = tilt_and_face(g[:, :, 0, 3:7])
    move, turn = step_motion(t); gmove, gturn = step_motion(g)
    quiet = (move < MOVE_M) & (turn < MOVE_DEG); gquiet = (gmove < MOVE_M) & (gturn < MOVE_DEG)
    def rest_time(q):   # first step index after which every step is quiet
        busy = ~q; last = np.where(busy.any(1), q.shape[1]-busy[:, ::-1].argmax(1), 0)
        return last*STEP
    t_rest, g_rest = rest_time(quiet), rest_time(gquiet)
    moving = ~quiet[:, -5:].all(1); gmoving = ~gquiet[:, -5:].all(1)
    final = tilt[:, -1]
    touching = near_other(t[:, -1], sz, mk) if mk[:, 1:].any() else np.zeros(count, bool)
    frozen = ~moving & (final > 8) & ~touching & ~com_supported(t[:, -1, 0, 3:7], sz[:, 0])
    gflat = gtilt[:, -1] < 1.
    err = np.linalg.norm(t[:, -1, 0, :3]-g[:, -1, 0, :3], axis=-1)
    series = {f'{k*STEP:g}s': dict(tilt_median_deg=float(np.median(tilt[:, k])), tilt_over_2deg=float((tilt[:, k] > 2).mean()),
                                    tilt_over_8deg=float((tilt[:, k] > 8).mean()), truth_tilt_over_2deg=float((gtilt[:, k] > 2).mean()),
                                    pos_err_mean_m=float(np.linalg.norm(t[:, k, 0, :3]-g[:, k, 0, :3], axis=-1).mean()))
              for k in range(0, traj.shape[1], 5)}
    return dict(
        cases=int(count), truth_flat_fraction=float(gflat.mean()),
        tilt_deg=summarize(final), tilt_over_2deg=float((final > 2).mean()), tilt_over_8deg=float((final > 8).mean()),
        tilt_over_2deg_where_truth_flat=float((final[gflat] > 2).mean()) if gflat.any() else None,
        tilt_over_8deg_where_truth_flat=float((final[gflat] > 8).mean()) if gflat.any() else None,
        same_face=float((face[:, -1] == gface[:, -1]).mean()),
        pos_err_m=summarize(err), pos_err_same_face_m=summarize(err[face[:, -1] == gface[:, -1]]),
        t_rest_s=summarize(t_rest), truth_t_rest_s=summarize(g_rest),
        t_rest_minus_truth_s=dict(median=float(np.median(t_rest-g_rest)), mean=float((t_rest-g_rest).mean()), mean_abs=float(np.abs(t_rest-g_rest).mean())),
        moving_at_end=float(moving.mean()), truth_moving_at_end=float(gmoving.mean()),
        frozen_mid_topple=int(frozen.sum()), frozen_mid_topple_fraction=float(frozen.mean()),
        penetration_mm=dict(mean_over_time=float(pn.mean()*1000), final_mean=float(pn[:, -1].mean()*1000), max=float(pn.max()*1000),
                            frames_over_5mm=float((pn > .005).mean())),
        series=series)

# ----------------------------------------------------------------------------- engines

def rollout(step_fn, state0, props, context, mask, steps: int, batch: int = 2048):
    import torch
    out, pens = [], []
    from agatha.contact_gnn import penetration_depths
    with torch.inference_mode():
        for a in range(0, len(state0), batch):
            sl = slice(a, a+batch); state = state0[sl]; frames = [state]
            for _ in range(steps):
                nxt = step_fn(state, props[sl], context[sl], mask[sl])
                state = torch.where(torch.isfinite(nxt).all(-1, keepdim=True), nxt, state); frames.append(state)
            traj = torch.stack(frames, 1)
            flat = traj.flatten(0, 1); rep = lambda v: v[sl].repeat_interleave(traj.shape[1], 0)
            d = penetration_depths(flat, rep(props), rep(context), rep(mask))
            pens.append(torch.stack([d[k][:, 0] for k in ('support', 'floor', 'objects')], -1).amax(-1).reshape(traj.shape[:2]).cpu())
            out.append(traj.cpu())
    return torch.cat(out).numpy(), torch.cat(pens).numpy()

def build_engines(model, project, variants: dict, extra_project=None) -> dict:
    accepted = set(inspect.signature(project).parameters)
    raw = lambda s, p, c, m: model(s, p, c, m)
    engines = {'model_only': raw, 'projection': lambda s, p, c, m: project(s, raw(s, p, c, m), p, c, m)}
    if extra_project is not None:
        engines['projection_file'] = lambda s, p, c, m: extra_project(s, raw(s, p, c, m), p, c, m)
    for name, kwargs in variants.items():
        if set(kwargs) <= accepted:
            engines[name] = (lambda kw: lambda s, p, c, m: project(s, raw(s, p, c, m), p, c, m, **kw))(kwargs)
    return engines

def markdown(result: dict) -> str:
    lines = ['# Landing benchmark', '', f"Model `{result['model']}`; {result['cases']} cases; seed {result['seed']}; 4 s rollouts (80 steps).",
             'Tilt = angle between the nearest face and the support plane at 4 s. PyBullet row = ground truth scored the same way.', '']
    head = ('| engine | tilt median | mean | p90 | >2 deg | >8 deg | same face | pos err median (m) | t_rest - truth mean (s) | moving @4s | frozen mid-topple | pen mean (mm) |',
            '|---|---|---|---|---|---|---|---|---|---|---|---|')
    for group, engines in result['groups'].items():
        n = next(iter(engines.values()))['cases']
        if not n: continue
        lines += [f'## {group} ({n} cases)', '', *head]
        for name, m in engines.items():
            lines.append(f"| {name} | {m['tilt_deg']['median']:.2f} | {m['tilt_deg']['mean']:.2f} | {m['tilt_deg']['p90']:.2f} | {100*m['tilt_over_2deg']:.1f}% | "
                         f"{100*m['tilt_over_8deg']:.1f}% | {100*m['same_face']:.1f}% | {m['pos_err_m']['median']:.4f} | {m['t_rest_minus_truth_s']['mean']:+.2f} | "
                         f"{100*m['moving_at_end']:.1f}% | {m['frozen_mid_topple']} | {m['penetration_mm']['mean_over_time']:.3f} |")
        lines.append('')
    lines += ['## Tilt > 2 deg over time (all single-box groups)', '', '| engine | ' + ' | '.join(result['series_times']) + ' |',
              '|---|' + '---|'*len(result['series_times'])]
    for name, m in result['groups']['single_all'].items():
        lines.append(f'| {name} | ' + ' | '.join(f"{100*m['series'][t]['tilt_over_2deg']:.0f}%" for t in result['series_times']) + ' |')
    return '\n'.join(lines)+'\n'

def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', type=Path, default=Path('models/contact_gnn.pt'))
    parser.add_argument('--out', type=Path, default=Path('models/runs/landing_eval.json'))
    parser.add_argument('--cache', type=Path, help='directory for the cached PyBullet ground truth')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--singles', type=int, default=1200); parser.add_argument('--friction', type=int, default=600)
    parser.add_argument('--multi', type=int, default=400); parser.add_argument('--edge', type=int, default=200)
    parser.add_argument('--workers', type=int, default=32)
    parser.add_argument('--device', default=None)
    parser.add_argument('--variants', default='all', help="'all', 'none' or comma-separated names from VARIANTS")
    parser.add_argument('--project-file', type=Path, help='another copy of agatha/contact_gnn.py whose project() is scored as projection_file')
    parser.add_argument('--dump', type=Path, help='write every engine trajectory of slot 0 to this .npz (diagnosis)')
    args = parser.parse_args(argv)
    began = time.perf_counter()
    cases = make_cases(args.seed, args.singles, args.friction, args.multi, args.edge)
    truth = load_truth(cases, args.cache, args.workers)
    print(f'{len(cases)} cases, ground truth ready after {time.perf_counter()-began:.0f}s', flush=True)
    import torch
    from agatha import contact_gnn
    device = torch.device(args.device or ('cuda:0' if torch.cuda.is_available() else 'cpu'))
    model = contact_gnn.build_from_checkpoint(torch.load(args.model, map_location='cpu', weights_only=True)).to(device).eval()
    extra = None
    if args.project_file:
        spec = importlib.util.spec_from_file_location('agatha.contact_gnn_reference', args.project_file)
        module = importlib.util.module_from_spec(spec); module.__package__ = 'agatha'; spec.loader.exec_module(module); extra = module.project
    chosen = {} if args.variants == 'none' else VARIANTS if args.variants == 'all' else {k: VARIANTS[k] for k in args.variants.split(',')}
    engines = build_engines(model, contact_gnn.project, chosen, extra)
    props, context, mask = (v.to(device) for v in case_tensors(cases))
    state0 = torch.from_numpy(truth[:, 0]).to(device)
    truth_pen = rollout_pen(truth, props, context, mask, device)
    sizes = props[..., :3].cpu().numpy(); mk = mask.cpu().numpy()
    group = np.array([c['group'] for c in cases])
    selections = {name: group == name for name in GROUPS}
    selections['single_all'] = np.isin(group, ('single_default', 'single_friction', 'edge_rest', 'stuck_examples'))
    selections['single_lift0'] = selections['single_all'] & (np.array([c['lift'] for c in cases]) == 0)
    selections['all'] = np.ones(len(cases), bool)
    result = dict(schema_version=1, model=str(args.model), seed=args.seed, cases=len(cases), steps=STEPS, step_s=STEP,
                  args={k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
                  variants={k: v for k, v in chosen.items() if k in engines},
                  series_times=[f'{k*STEP:g}s' for k in range(0, STEPS+1, 10)],
                  groups={name: {'pybullet': score(truth, truth, sizes, mk, truth_pen, sel)} for name, sel in selections.items()})
    dump = dict(truth=truth[:, :, 0], sizes=sizes[:, 0], group=group)
    for name, step_fn in engines.items():
        start = time.perf_counter()
        traj, pen = rollout(step_fn, state0, props, context, mask, STEPS)
        for key, sel in selections.items(): result['groups'][key][name] = score(traj, truth, sizes, mk, pen, sel)
        dump[name] = traj[:, :, 0]
        m = result['groups']['single_all'][name]
        print(f"{name:32s} single_all: tilt median {m['tilt_deg']['median']:.2f} mean {m['tilt_deg']['mean']:.2f} >2deg {100*m['tilt_over_2deg']:.1f}% "
              f">8deg {100*m['tilt_over_8deg']:.1f}% same face {100*m['same_face']:.1f}% frozen {m['frozen_mid_topple']} ({time.perf_counter()-start:.0f}s)", flush=True)
    result['seconds'] = time.perf_counter()-began
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1)); args.out.with_suffix('.md').write_text(markdown(result))
    if args.dump: np.savez_compressed(args.dump, **dump)
    print(f'wrote {args.out} and {args.out.with_suffix(".md")}')
    return result

def rollout_pen(truth: np.ndarray, props, context, mask, device) -> np.ndarray:
    import torch
    from agatha.contact_gnn import penetration_depths
    out = []
    with torch.inference_mode():
        for a in range(0, len(truth), 1024):
            traj = torch.from_numpy(truth[a:a+1024]).to(device); sl = slice(a, a+1024)
            rep = lambda v: v[sl].repeat_interleave(traj.shape[1], 0)
            d = penetration_depths(traj.flatten(0, 1), rep(props), rep(context), rep(mask))
            out.append(torch.stack([d[k][:, 0] for k in ('support', 'floor', 'objects')], -1).amax(-1).reshape(traj.shape[:2]).cpu())
    return torch.cat(out).numpy()

if __name__ == '__main__':
    main()
