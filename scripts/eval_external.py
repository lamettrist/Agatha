"""APPROXIMATE evaluation of the learned world models on converted external datasets.

    CUDA_VISIBLE_DEVICES="" .venv-train/bin/python -I scripts/eval_external.py \
        --data data/external_v2 [--new models/contact_gnn.pt] [--old models/mlp_multistep.pt] \
        --out models/external_eval.json

Every engine is rolled out autoregressively (never teacher-forced) from the first frame of each
converted episode (the first frame whose velocities are defined, see prepare_external_data.py)
over the whole available horizon.  Engines:

  reference          the converted ground truth itself (floor of the penetration metric; shows the
                     penetration that the box approximation alone creates)
  hold_initial       trivial baseline: nothing ever moves
  agatha_physics     Agatha's own PyBullet BOX engine (same body / solver settings as
                     agatha.physics.World) started from the same converted initial state.  It
                     separates 'the conversion / the other simulator disagrees with Agatha's box
                     physics' from 'the learned model is wrong'.  --no-physics skips it.
  old_mlp            models/mlp_multistep.pt
  old_mlp_projected  old model + contact_gnn.project after every step (attribution only)
  new_raw            ContactGNN alone
  new_projected      ContactGNN + contact_gnn.project after every step (the shipped configuration)

Ground mappings (`--floor-modes`): `slab` = the shards as written (ground = top of a large support
slab at 0.75 m); `floor` = heights shifted so the ground is Agatha's y = 0 floor, with a default
1.2 x 0.8 m support at 0.75 m whose collisions are disabled (the 'noclip' situation of data/v2).

THE RESULTS ARE APPROXIMATE: external simulator data converted with the approximations documented
at the top of scripts/prepare_external_data.py.  They are not a real-world validation.
"""
import argparse
import glob
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(Path(__file__).resolve().parent)):
    if _p not in sys.path: sys.path.insert(0, _p)
import numpy as np
import torch
import eval_contact as ev                       # metric primitives are reused, the file is not modified
from prepare_external_data import MOVED_FAR_M, NOTE, STEP, TIP_COS

HORIZONS_S = (.5, 1.)
PEN_CAP_M = ev.PEN_CAP_M
BANNER = ('APPROXIMATE: external data converted with documented approximations (scripts/prepare_external_data.py); '
          'simulator output from other generators, NOT a real-world validation.')
KEYS = ('states', 'props', 'context', 'mask', 'fell', 'tipped', 'kind', 'chaotic', 'n_frames', 'shape', 'box_like', 'moved_far', 'settled', 'source_id')
ENGINE_INFO = dict(
    reference='converted ground truth scored with the same metrics',
    hold_initial='trivial baseline: every object stays at its initial pose',
    agatha_physics="Agatha's PyBullet box engine from the same converted initial state (how far the box approximation + a different simulator are from the source, without any learned model)",
    old_mlp='old shipped model alone (mlp_multistep)',
    old_mlp_projected='old mlp_multistep + the non-learned contact projection (attribution only)',
    new_raw='ContactGNN alone: learned model only, no projection',
    new_projected='ContactGNN + non-learned contact projection after every step (shipped configuration)',
)
SUBSETS = dict(all_objects='all dynamic objects', box_like_objects='true cuboids, in any episode (neighbours may be non-box shapes)',
               box_like_only_episodes='all dynamic objects of episodes that contain only true cuboids')


# ----------------------------------------------------------------------------- data
def load_dataset(path: Path, max_episodes: int | None = None) -> dict:
    files = sorted(glob.glob(str(Path(path) / 'test' / 'shard_*.npz')))
    if not files: raise FileNotFoundError(f'No shards in {Path(path) / "test"}')
    parts = {k: [] for k in KEYS}
    for name in files:
        with np.load(name, allow_pickle=False) as shard:
            for k in KEYS: parts[k].append(shard[k])
    out = {k: np.concatenate(v) for k, v in parts.items()}
    if max_episodes and max_episodes < len(out['states']):
        keep = np.unique(np.linspace(0, len(out['states']) - 1, max_episodes).round().astype(int))
        out = {k: v[keep] for k, v in out.items()}
    return out


def to_floor_mode(data: dict) -> dict:
    """Ground plane -> Agatha's y = 0 floor; default-size support at 0.75 m with collisions disabled."""
    out = dict(data); states = data['states'].copy(); context = data['context'].copy()
    states[..., 1] -= context[:, None, None, 2]
    context[:, 0], context[:, 1], context[:, 2], context[:, 7] = 1.2, .8, .75, 0.
    out['states'], out['context'] = states, context
    return out


# ----------------------------------------------------------------------------- metrics
def tipped_relative(traj: torch.Tensor) -> torch.Tensor:
    """traj [B,F,N,13] -> bool [B,N]: the initial up axis is more than 35 deg from world up in any frame."""
    q = traj[..., 3:7].double(); q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    q0 = q[:, :1] * q.new_tensor([-1., -1., -1., 1.])
    ax, ay, az, aw = q.unbind(-1); bx, by, bz, bw = q0.unbind(-1)
    x = aw*bx+ax*bw+ay*bz-az*by; z = aw*bz+ax*by-ay*bx+az*bw
    return ((1 - 2 * (x * x + z * z)) < TIP_COS).any(1)


def freeze_after(traj: torch.Tensor, n_frames: torch.Tensor) -> torch.Tensor:
    """Repeat frame n_frames-1 afterwards, so padded frames never contribute to any-over-time rules."""
    index = torch.minimum(torch.arange(traj.shape[1], device=traj.device)[None], (n_frames - 1)[:, None])
    return torch.gather(traj, 1, index[:, :, None, None].expand(-1, -1, *traj.shape[2:]))


def object_stats(traj: torch.Tensor, batch: dict, diverged, pen_fn, fell_meaningful: bool) -> dict:
    """Per-object arrays [B,N] (and per-episode [B]) for one engine. traj must already be frozen after n_frames."""
    true, props, context, mask, n = batch['states'], batch['props'], batch['context'], batch['mask'], batch['n_frames']
    dyn = ev.dynamic_mask(props, mask)
    error = (traj[..., :3] - true[..., :3]).norm(dim=-1).double()                                # [B,F,N]
    out = dict(dyn=dyn, pos_final=error[:, -1], ori_final=ev.orientation_error_deg(traj[:, -1, :, 3:7], true[:, -1, :, 3:7]).double())
    for seconds in HORIZONS_S:
        index = round(seconds / STEP)
        out[f'pos_{seconds:g}s'] = error[:, min(index, error.shape[1] - 1)]
        out[f'valid_{seconds:g}s'] = dyn & (n > index)[:, None]
    frames = torch.arange(true.shape[1], device=true.device)[None]
    valid = ((frames >= 1) & (frames < n[:, None])).double()                                     # [B,F]
    out['pos_rollout_sum'] = (error * valid[:, :, None]).sum(1); out['frames'] = valid.sum(1)[:, None].expand_as(out['pos_final'])
    out['tipped_pred'] = tipped_relative(traj) & dyn
    out['moved_far_pred'] = ((traj[:, -1, :, :3] - traj[:, 0, :, :3]).norm(dim=-1) > MOVED_FAR_M) & dyn
    out['fell_pred'] = (ev.fell_tipped(traj, props, context, mask)[0] if fell_meaningful else torch.zeros_like(dyn)) & dyn
    pen_sum = torch.zeros_like(out['pos_final']); pen_cap = torch.zeros_like(pen_sum); pen_max = torch.zeros_like(pen_sum)
    if pen_fn is not None:
        for t in range(1, true.shape[1]):
            depth = pen_fn(traj[:, t], props, context, mask)
            worst = torch.stack([torch.nan_to_num(depth[k].double(), nan=0., posinf=1e3).clamp_min(0) for k in ('support', 'floor', 'objects')]).amax(0)
            worst = worst * valid[:, t, None]
            pen_sum += worst; pen_cap += worst.clamp_max(PEN_CAP_M); pen_max = torch.maximum(pen_max, worst)
    out.update(pen_sum=pen_sum, pen_cap_sum=pen_cap, pen_max=pen_max)
    out['diverged'] = diverged if diverged is not None else torch.zeros_like(n, dtype=torch.bool)
    return out


def hold_step(state, props, context, mask):
    out = state.clone(); out[..., 7:] = 0
    return out


def _physics_episode(args):
    """One episode in a private PyBullet DIRECT client, mirroring agatha.physics.World (boxes, 240 Hz, 50 iterations, damping 0.04)."""
    state0, props, context, mask, steps = args
    import pybullet as p
    from agatha.coordinates import to_bullet, quat_to_bullet, from_bullet, quat_from_bullet
    c = p.connect(p.DIRECT)
    try:
        p.setPhysicsEngineParameter(fixedTimeStep=1 / 240, numSolverIterations=50, deterministicOverlappingPairs=1, physicsClientId=c)
        p.setGravity(0, 0, -float(context[3]), physicsClientId=c)
        def box(size, pos, quat, mass, friction, restitution, rolling=0., spinning=0.):
            shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=(size[0] / 2, size[2] / 2, size[1] / 2), physicsClientId=c)
            body = p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=shape, basePosition=to_bullet(pos), baseOrientation=quat_to_bullet(quat), physicsClientId=c)
            p.changeDynamics(body, -1, lateralFriction=friction, restitution=restitution, linearDamping=.04, angularDamping=.04,
                             rollingFriction=rolling, spinningFriction=spinning, physicsClientId=c)
            return body
        box((20, .1, 20), (0, -.05, 0), (0, 0, 0, 1), 0, .5, 0)
        if context[7] > .5:
            box((float(context[0]), .08, float(context[1])), (0, float(context[2]) - .04, 0), (0, 0, 0, 1), 0, float(context[8]), float(context[9]))
        bodies = {}
        for i in np.flatnonzero(mask):
            s, q = state0[i].astype(float), props[i].astype(float)
            bodies[i] = box(q[:3], s[:3], s[3:7] / max(np.linalg.norm(s[3:7]), 1e-9), 0 if q[5] > .5 else q[3], q[4], q[6], q[7], q[8])
            if q[5] < .5: p.resetBaseVelocity(bodies[i], to_bullet(s[7:10]), to_bullet(s[10:13]), physicsClientId=c)
        traj = np.repeat(state0[None], steps + 1, 0).astype(np.float32)
        for t in range(1, steps + 1):
            for _ in range(12): p.stepSimulation(physicsClientId=c)
            for i, body in bodies.items():
                pos, quat = p.getBasePositionAndOrientation(body, physicsClientId=c); vel, ang = p.getBaseVelocity(body, physicsClientId=c)
                traj[t, i] = (*from_bullet(pos), *quat_from_bullet(quat), *from_bullet(vel), *from_bullet(ang))
        return traj
    finally:
        p.disconnect(c)


def physics_rollouts(data: dict, workers: int = 16) -> np.ndarray:
    """[E,F,N,13] rollouts of Agatha's PyBullet box engine from frame 0 of every episode."""
    steps = data['states'].shape[1] - 1
    jobs = [(data['states'][e, 0], data['props'][e], data['context'][e], data['mask'][e], steps) for e in range(len(data['states']))]
    if workers <= 1: return np.stack([_physics_episode(j) for j in jobs])
    import multiprocessing as mp
    with mp.get_context('spawn').Pool(workers) as pool: return np.stack(pool.map(_physics_episode, jobs, chunksize=8))


def evaluate(engines: dict, data: dict, pen_fn, fell_meaningful: bool, batch_size: int = 256) -> dict:
    count, steps = len(data['states']), data['states'].shape[1] - 1
    collected = {name: [] for name in engines}
    with torch.inference_mode():
        for start in range(0, count, batch_size):
            sl = slice(start, min(start + batch_size, count))
            batch = dict(states=torch.from_numpy(data['states'][sl]).float(), props=torch.from_numpy(data['props'][sl]).float(),
                         context=torch.from_numpy(data['context'][sl]).float(), mask=torch.from_numpy(data['mask'][sl]).bool(),
                         n_frames=torch.from_numpy(data['n_frames'][sl]).long())
            for name, step_fn in engines.items():
                if step_fn is None: traj, diverged = batch['states'], None
                elif isinstance(step_fn, np.ndarray): traj, diverged = torch.from_numpy(step_fn[sl]).float(), None
                else: traj, diverged = ev.rollout(step_fn, batch['states'][:, 0], batch['props'], batch['context'], batch['mask'], steps)
                traj = freeze_after(traj, batch['n_frames'])
                collected[name].append({k: v.cpu().numpy() for k, v in object_stats(traj, batch, diverged, pen_fn, fell_meaningful).items()})
    return {name: {k: np.concatenate([p[k] for p in parts]) for k in parts[0]} for name, parts in collected.items()}


def _mean(values, select):
    return float(values[select].mean()) if select.any() else None


def _confusion(pred, true, select) -> dict:
    p, t = pred[select], true[select]
    tp, fp, fn, tn = int((p & t).sum()), int((p & ~t).sum()), int((~p & t).sum()), int((~p & ~t).sum())
    total = tp + fp + fn + tn
    return dict(accuracy=(tp + tn) / total if total else None, precision=tp / (tp + fp) if tp + fp else None,
                recall=tp / (tp + fn) if tp + fn else None, tp=tp, fp=fp, fn=fn, tn=tn,
                majority_class_accuracy=max(tp + fn, fp + tn) / total if total else None)


def subset_masks(data: dict) -> dict:
    dyn = data['mask'] & (data['props'][..., 5] < .5); box = data['box_like'].astype(bool)
    box_only = ((dyn & ~box).sum(1) == 0) & dyn.any(1)
    return dict(all_objects=dyn, box_like_objects=dyn & box, box_like_only_episodes=dyn & box_only[:, None])


def summarize(s: dict, data: dict, select: np.ndarray, fell_meaningful: bool) -> dict:
    """Pool one engine's per-object statistics over the object mask `select` [E,N]."""
    out = dict(n_objects=int(select.sum()), n_episodes=int(select.any(1).sum()))
    for seconds in HORIZONS_S:
        valid = select & s[f'valid_{seconds:g}s']
        out[f'pos_err_{seconds:g}s_m'] = _mean(s[f'pos_{seconds:g}s'], valid); out[f'n_objects_{seconds:g}s'] = int(valid.sum())
    settled = select & data['settled'].astype(bool)
    frames = float(s['frames'][select].sum())
    out.update(rest_pos_err_m=_mean(s['pos_final'], select), rest_pos_err_median_m=float(np.median(s['pos_final'][select])) if select.any() else None,
               rest_pos_err_settled_m=_mean(s['pos_final'], settled), n_settled_objects=int(settled.sum()),
               pos_err_rollout_mean_m=float(s['pos_rollout_sum'][select].sum() / frames) if frames else None,
               ori_err_final_deg=_mean(s['ori_final'], select),
               tipped=_confusion(s['tipped_pred'], data['tipped'].astype(bool), select),
               moved_far=_confusion(s['moved_far_pred'], data['moved_far'].astype(bool), select),
               fell=_confusion(s['fell_pred'], data['fell'].astype(bool), select) if fell_meaningful else None,
               penetration=dict(mean_m=float(s['pen_sum'][select].sum() / frames) if frames else None,
                                mean_capped_50mm_m=float(s['pen_cap_sum'][select].sum() / frames) if frames else None,
                                max_m=float(s['pen_max'][select].max()) if select.any() else None),
               diverged_episodes=int(s['diverged'][select.any(1)].sum()))
    return out


def paired_bootstrap(old: dict, new: dict, select: np.ndarray, key: str = 'pos_final', resamples: int = 1000, seed: int = 0) -> dict | None:
    """Episode-level paired bootstrap of the pooled mean difference new - old (negative = new is better)."""
    episodes = np.flatnonzero(select.any(1))
    if len(episodes) < 2: return None
    sums = [np.where(select, d[key], 0.)[episodes].sum(1) for d in (old, new)]; counts = select[episodes].sum(1)
    rng = np.random.default_rng(seed); pick = rng.integers(0, len(episodes), (resamples, len(episodes)))
    deltas = (sums[1][pick].sum(1) - sums[0][pick].sum(1)) / counts[pick].sum(1)
    return dict(old=float(sums[0].sum() / counts.sum()), new=float(sums[1].sum() / counts.sum()), delta=float((sums[1].sum() - sums[0].sum()) / counts.sum()),
                ci95=[float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))], n_episodes=int(len(episodes)))


# ----------------------------------------------------------------------------- report
def _f(value, scale=1., digits=3):
    return '-' if value is None else f'{value * scale:.{digits}f}'


def _conf(c):
    if c is None: return '-'
    return f"{_f(c['accuracy'])} ({c['tp']}/{c['fp']}/{c['fn']}/{c['tn']})"


def markdown(result: dict) -> str:
    lines = ['# External-data evaluation (APPROXIMATE)', '', f'> **{BANNER}**', '',
             f"New model: `{result['checkpoints']['new']}` (epoch {result['new_model'].get('epoch')}, file written {result['checkpoints']['new_mtime']}, sha256 {result['checkpoints']['new_sha256'][:12]}); old model: `{result['checkpoints']['old']}`. "
             'Autoregressive rollouts from the first converted frame, never teacher-forced. Licences: `DATA_LICENSES.md`.', '',
             'Engines:'] + [f'- `{k}`: {v}' for k, v in ENGINE_INFO.items()] + ['', 'Subsets:'] + [f'- `{k}`: {v}' for k, v in SUBSETS.items()] + ['',
             'Columns: position error (m) at 0.5 s / 1 s / the last frame ("rest pos", all objects) and at the last frame for the objects that are '
             'at rest in the ground truth ("settled"); tipped / moved-far accuracy with (TP/FP/FN/TN), `maj` = accuracy of always predicting the majority class; '
             '`fell` false alarms (the ground truth never falls: there is no edge); penetration in mm.', '']
    for name, block in result['datasets'].items():
        d = block['data']
        lines += [f"## {name}", '', f"{d['n_episodes']} episodes, {d['n_dynamic_objects']} dynamic objects ({d['n_box_like_objects']} box-like, "
                  f"{d['n_box_like_only_episodes']} box-like-only episodes); horizon {d['horizon_s'][0]:g}-{d['horizon_s'][1]:g} s; "
                  f"ground truth: tipped {_f(d['tipped_fraction'])}, moved far {_f(d['moved_far_fraction'])}, settled at the end {_f(d['settled_fraction'])}. "
                  f"Preferred ground mapping for new_projected (lower rest-position error, all objects): **{block['preferred_floor_mode']}**.", '']
        for mode, engines in block['floor_modes'].items():
            for subset in SUBSETS:
                first = next(iter(engines.values()))[subset]
                if not first['n_objects']: continue
                lines += [f"### {name} / ground = {mode} / {subset} ({first['n_objects']} objects in {first['n_episodes']} episodes)", '',
                          '| engine | pos 0.5s | pos 1s | rest pos | rest pos median | rest pos settled (n) | ori final (deg) | tipped acc (TP/FP/FN/TN) | tipped maj | moved-far acc (TP/FP/FN/TN) | fell false alarms | pen mean (mm) | pen capped (mm) | diverged |',
                          '|' + '---|' * 14]
                for engine, subsets in engines.items():
                    m = subsets[subset]
                    lines.append(f"| {engine} | {_f(m['pos_err_0.5s_m'])} | {_f(m['pos_err_1s_m'])} | {_f(m['rest_pos_err_m'])} | {_f(m['rest_pos_err_median_m'])} | "
                                 f"{_f(m['rest_pos_err_settled_m'])} ({m['n_settled_objects']}) | {_f(m['ori_err_final_deg'], digits=1)} | {_conf(m['tipped'])} | "
                                 f"{_f(m['tipped']['majority_class_accuracy'])} | {_conf(m['moved_far'])} | {'-' if m['fell'] is None else m['fell']['fp']} | "
                                 f"{_f(m['penetration']['mean_m'], 1000, 2)} | {_f(m['penetration']['mean_capped_50mm_m'], 1000, 2)} | {m['diverged_episodes']} |")
                lines.append('')
        if block.get('paired_bootstrap'):
            lines += [f'### {name} / paired episode bootstrap of the rest-position error (new - old, m; negative = new better)', '',
                      '| ground | subset | comparison | old | new | delta | 95% CI | episodes |', '|' + '---|' * 8]
            for row in block['paired_bootstrap']:
                b = row['result']
                lines.append(f"| {row['floor_mode']} | {row['subset']} | {row['new']} vs {row['old']} | {_f(b['old'])} | {_f(b['new'])} | {_f(b['delta'])} | "
                             f"[{_f(b['ci95'][0])}, {_f(b['ci95'][1])}] | {b['n_episodes']} |")
            lines.append('')
    lines += ['## Notes', ''] + [f'- {n}' for n in result['notes']] + ['']
    return '\n'.join(lines)


NOTES = [
    BANNER,
    'All three sources are simulators (PyBullet, Blender/Bullet, TDW/PhysX); no real video is evaluated here.',
    'Non-box shapes are replaced by their bounding boxes; the `reference` row shows how much penetration that alone creates.',
    "`tipped` is the relative rule (initial up axis rotated > 35 deg), not agatha.physics.summarize's absolute rule.",
    "`rest pos` is the position error at the last available frame; many Kubric / CLEVRER objects are still moving then, see the 'settled' column for the objects that are at rest in the ground truth.",
    'Physion: 10 trials only; every number there is anecdotal.',
    'CLEVRER friction was calibrated from the data set itself (median sliding deceleration), sizes / masses / time base are assumed.',
    'Chaotic rollouts (tosses, collapses) are sensitive to initial conditions; a position error of several object sizes is expected even for a perfect model of a slightly different simulator.',
]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--data', type=Path, default=ROOT / 'data' / 'external_v2')
    parser.add_argument('--datasets', nargs='+', default=None)
    parser.add_argument('--new', type=Path, default=None, help='contact_gnn checkpoint (default models/contact_gnn.pt, else models/runs/prelim.pt)')
    parser.add_argument('--old', type=Path, default=ROOT / 'models' / 'mlp_multistep.pt')
    parser.add_argument('--out', type=Path, default=ROOT / 'models' / 'external_eval.json')
    parser.add_argument('--floor-modes', nargs='+', default=['slab', 'floor'], choices=['slab', 'floor'])
    parser.add_argument('--max-episodes', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--threads', type=int, default=16)
    parser.add_argument('--no-physics', action='store_true', help="skip the agatha_physics (PyBullet) engine")
    args = parser.parse_args(argv)
    torch.set_num_threads(min(args.threads, 24))
    from agatha import contact_gnn
    new_path = args.new
    if new_path is None:
        new_path = next((p for p in (ROOT / 'models' / 'contact_gnn.pt', ROOT / 'models' / 'runs' / 'prelim.pt') if p.exists()), None)
        if new_path is None: raise SystemExit('no contact_gnn checkpoint found (models/contact_gnn.pt or models/runs/prelim.pt)')
    checkpoint = torch.load(new_path, map_location='cpu', weights_only=True)
    new = contact_gnn.build_from_checkpoint(checkpoint).eval()
    old = ev.load_old(args.old, 'cpu')
    raw = lambda state, props, context, mask: new(state, props, context, mask)
    engines = dict(reference=None, hold_initial=hold_step, old_mlp=ev.old_step(old),
                   old_mlp_projected=ev.projected_step(ev.old_step(old), contact_gnn.project),
                   new_raw=raw, new_projected=ev.projected_step(raw, contact_gnn.project))
    names = args.datasets or sorted(p.parent.name for p in args.data.glob('*/test'))
    result = dict(approximate=True, note=BANNER, conversion_note=NOTE, generated=time.strftime('%Y-%m-%d %H:%M:%S'),
                  checkpoints=dict(new=str(new_path), old=str(args.old), new_sha256=hashlib.sha256(Path(new_path).read_bytes()).hexdigest(),
                                   new_mtime=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(Path(new_path).stat().st_mtime))),
                  new_model=dict(config=checkpoint.get('config'), epoch=checkpoint.get('epoch')),
                  engines=ENGINE_INFO, subsets=SUBSETS, datasets={}, notes=NOTES)
    for name in names:
        base = load_dataset(args.data / name, args.max_episodes)
        dyn = base['mask'] & (base['props'][..., 5] < .5); masks = subset_masks(base)
        block = dict(data=dict(path=str(args.data / name), n_episodes=int(len(base['states'])), n_dynamic_objects=int(dyn.sum()),
                               n_box_like_objects=int(masks['box_like_objects'].sum()),
                               n_box_like_only_episodes=int(masks['box_like_only_episodes'].any(1).sum()),
                               horizon_s=[float((base['n_frames'].min() - 1) * STEP), float((base['n_frames'].max() - 1) * STEP)],
                               tipped_fraction=float(base['tipped'][dyn].mean()), moved_far_fraction=float(base['moved_far'][dyn].mean()),
                               settled_fraction=float(base['settled'][dyn].mean()), kinds=sorted(set(base['kind'].tolist()))),
                     floor_modes={}, paired_bootstrap=[])
        for mode in args.floor_modes:
            began = time.time()
            data = base if mode == 'slab' else to_floor_mode(base)
            run = dict(engines)
            if not args.no_physics:
                run = dict(reference=None, hold_initial=hold_step, agatha_physics=physics_rollouts(data, min(args.threads, 24)),
                           **{k: v for k, v in engines.items() if k not in ('reference', 'hold_initial')})
            stats = evaluate(run, data, contact_gnn.penetration_depths, mode == 'slab', args.batch_size)
            block['floor_modes'][mode] = {engine: {subset: summarize(s, data, select, mode == 'slab') for subset, select in masks.items()}
                                          for engine, s in stats.items()}
            for subset in ('all_objects', 'box_like_objects', 'box_like_only_episodes'):
                for a, b in (('old_mlp', 'new_projected'), ('old_mlp', 'new_raw'), ('old_mlp_projected', 'new_projected'), ('hold_initial', 'new_projected'),
                             ('agatha_physics', 'new_projected')):
                    if a not in stats: continue
                    boot = paired_bootstrap(stats[a], stats[b], masks[subset])
                    if boot: block['paired_bootstrap'].append(dict(floor_mode=mode, subset=subset, old=a, new=b, metric='rest_pos_err_m', result=boot))
            print(f'{name} / {mode}: {len(base["states"])} episodes [{time.time() - began:.1f}s]', flush=True)
        score = {mode: block['floor_modes'][mode]['new_projected']['all_objects']['rest_pos_err_m'] for mode in args.floor_modes}
        block['preferred_floor_mode'] = min(score, key=score.get)
        result['datasets'][name] = block
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1))
    args.out.with_suffix('.md').write_text(markdown(result))
    print(f'wrote {args.out} and {args.out.with_suffix(".md")}')


if __name__ == '__main__':
    main()
