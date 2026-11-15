"""Full-rollout evaluation of the EXPERIMENTAL soft-body model (agatha.soft_gnn.SoftGNN) on data/soft_v1.

Every engine is rolled out autoregressively to the end of the episode (never teacher-forced on the
soft body).  By default a rollout is seeded with the TRUE frames (0, 1) and frames 2..T are predicted
(the model's state is two position frames); --start velocity seeds with frame 0 and its recorded
vertex velocity instead (prev = verts_0 - vert_vel_0*STEP) and predicts frames 1..T.
Rigid boxes in a soft scene are PRESCRIBED from the ground-truth trajectory in every engine
(teacher-forced obstacles): only the soft vertices are predicted, nothing pushes back on the boxes.

Engines:
  physics_reference      ground truth scored with the same metrics (floor for penetration / strain)
  hold_initial           trivial baseline: the last seeded frame is held for the whole episode
  rigid_freefall         trivial baseline: the seeded shape translates rigidly with its COM velocity under
                         gravity until its lowest vertex reaches the table top (COM over the footprint) or
                         the floor, then stops
  soft_raw               SoftGNN alone (learned model only)
  soft_projected         SoftGNN + the non-learned soft_gnn.project_soft after every step (--project-margin)
  soft_projected_margin  the same with --margin-engine metres of clearance (PyBullet soft nodes rest about one
                         collision margin above a surface); omitted when --margin-engine is 0
  soft_projected_sleep   soft_projected + the NON-LEARNED sleep rule of project_soft (--sleep-move metres per step:
                         a supported body whose vertices all move less than that is held); omitted when 0
Pinned (kinematic) vertices of a shard ('pinned' [E,V], e.g. the handles of the converted MeshGraphNets cloth)
are PRESCRIBED from the ground truth in every model engine, like the rigid obstacles.  --data accepts any data
set in the soft shard format, e.g. data/external_v2/meshgraphnets_soft.

Metrics per scenario kind (means over episodes; COM = unweighted mean of the vertices):
  vertex position error at 0.5 / 1 / 2 s (absolute episode time) and at the last frame, COM error,
  shape error (vertex error after removing each trajectory's COM translation), vertex penetration into
  support / floor / boxes over the predicted frames, edge-strain error, final COM error and final COM
  error restricted to episodes whose ground truth is at rest at the last frame, diverged rollouts,
  CPU ms per step (batch 1).

Usage:
  CUDA_VISIBLE_DEVICES=7 .venv-train/bin/python scripts/eval_soft.py --data data/soft_v1 --split test \
      --ckpt models/runs/soft_NAME/best.pt --out models/soft_eval.json --example docs/soft-example.json
Writes <out> (JSON), <out stem>.md and the example rollout JSON (true vs predicted vertices + faces).
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch

from agatha import soft_gnn as sg
import train_soft_ddp as ts

STEP = ts.STEP
HORIZONS = ts.HORIZON_FRAMES
PEN_THRESHOLD_M = .005
ENGINE_INFO = {
    'physics_reference': 'PyBullet ground truth scored with the same metrics (floor for penetration / strain)',
    'hold_initial': 'trivial baseline: the last seeded frame is held',
    'rigid_freefall': 'trivial baseline: rigid free-fall of the seeded shape (COM velocity + gravity) until the table top / floor '
                      '(+ --baseline-clearance, the collision margin), then stop',
    'soft_raw': 'SoftGNN alone: learned model only, no projection',
    'soft_projected': 'SoftGNN + non-learned project_soft after every step',
    'soft_projected_margin': 'SoftGNN + project_soft keeping vertices one collision margin outside surfaces',
    'soft_projected_sleep': 'SoftGNN + project_soft + NON-LEARNED sleep rule (a supported body whose vertices all move less than '
                            '--sleep-move per step is held in place)',
}


# ----------------------------------------------------------------------------- engines
def hold_step(cur, prev, batch, frame):
    return cur


def freefall_step(cur, prev, batch, frame, clearance=0.):
    """Rigid translation of the current shape: ballistic COM, stops on the table top (COM over the footprint,
    tilt ignored) or on the floor.  Stateless: after the stop cur == prev, so the next step lands on the same spot.
    ``clearance``: the lowest vertex stops this far above the surface (Bullet's soft collision margin), so the
    baseline is not handicapped by sinking one margin into a surface the ground truth hovers above."""
    vm = batch['vert_mask']
    w = vm[..., None].float()
    n = vm.sum(1).clamp_min(1).float()[:, None]
    context = batch['context']
    com = (cur * w).sum(1) / n
    shift = ((cur - prev) * w).sum(1) / n                       # COM displacement of the last step [B,3]
    shift = shift.clone()
    shift[:, 1] -= context[:, 3] * STEP ** 2
    over = (context[:, 7] > .5) & (com[:, 0].abs() <= context[:, 0] / 2) & (com[:, 2].abs() <= context[:, 1] / 2)
    lowest = torch.where(vm, cur[..., 1], torch.full_like(cur[..., 1], float('inf'))).min(1).values
    surface = torch.where(over & (lowest >= context[:, 2] - .02), context[:, 2], torch.zeros_like(lowest)) + clearance
    landed = lowest + shift[:, 1] <= surface
    shift = torch.where(landed[:, None], torch.stack((torch.zeros_like(lowest), surface - lowest, torch.zeros_like(lowest)), -1), shift)
    return torch.where(vm[..., None], cur + shift[:, None], cur)


def build_engines(model, args, amp_dtype=None):
    limits = dict(max_strain=args.max_strain) if args.max_strain > 0 else {}
    raw = lambda cur, prev, batch, t: ts.model_step(model, cur, prev, batch, t, amp_dtype)
    engines = dict(hold_initial=hold_step,
                   rigid_freefall=lambda cur, prev, batch, t: freefall_step(cur, prev, batch, t, args.baseline_clearance))
    if model is not None:
        engines['soft_raw'] = raw
        engines['soft_projected'] = lambda cur, prev, batch, t: ts.project_step(cur, raw(cur, prev, batch, t), batch, t, margin=args.project_margin, **limits)
        if args.margin_engine > 0:
            engines['soft_projected_margin'] = lambda cur, prev, batch, t: ts.project_step(cur, raw(cur, prev, batch, t), batch, t, margin=args.margin_engine, **limits)
        if args.sleep_move > 0:
            engines['soft_projected_sleep'] = lambda cur, prev, batch, t: ts.project_step(cur, raw(cur, prev, batch, t), batch, t, margin=args.project_margin,
                                                                                         sleep_move=args.sleep_move, **limits)
    return engines


def seed(batch, start_mode, vel0=None):
    """(start frame, prev0 override) of a rollout."""
    if start_mode == 'velocity':
        return 0, sg.prev_from_velocity(batch['verts'][:, 0].float(), vel0.float())
    return ts.START_FRAME, None


# ----------------------------------------------------------------------------- evaluation
@torch.no_grad()
def evaluate(engines, data, device, args, log=print):
    """Per-episode numpy stats {engine: {metric: [E,...]}} (physics_reference included)."""
    count = len(data['verts'])
    collected = {name: [] for name in ('physics_reference', *engines)}
    for i in range(0, count, args.batch):
        sl = slice(i, min(i + args.batch, count))
        batch = ts.episodes(data, sl, device)
        start, prev0 = seed(batch, args.start, data['vel0'][sl].to(device))
        stats = ts.trajectory_stats(batch['verts'].float(), batch, start)
        stats['diverged'] = torch.zeros(len(batch['verts']), device=device)
        collected['physics_reference'].append({k: v.cpu().numpy() for k, v in stats.items()})
        for name, step_fn in engines.items():
            traj, diverged = ts.rollout(step_fn, batch, start, prev0)
            stats = ts.trajectory_stats(traj, batch, start)
            stats['diverged'] = diverged.float()
            collected[name].append({k: v.cpu().numpy() for k, v in stats.items()})
        log(f'  evaluated {sl.stop}/{count} episodes')
    return {name: {k: np.concatenate([q[k] for q in parts]) for k in parts[0]} for name, parts in collected.items()}


def _mean(values, select):
    values = np.asarray(values, np.float64)[select]
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else None


def aggregate(stats, select, rest_end):
    """Scalar report of one engine over the episodes in `select` (bool [E])."""
    frames = stats['vert_err'].shape[1]
    out = dict(n_episodes=int(select.sum()), n_rest_at_end=int((select & rest_end).sum()))
    for name in ('vert_err', 'com_err', 'shape_err'):
        for frame, label in HORIZONS:
            out[f'{name}_{label}_m'] = _mean(stats[name][:, frame], select) if frame < frames else None
        out[f'{name}_final_m'] = _mean(stats[name][:, -1], select)
    out['final_rest_com_err_m'] = _mean(stats['com_err'][:, -1], select & rest_end)
    out['penetration_mean_m'] = _mean(stats['pen_mean'], select)
    out['penetration_capped_mean_m'] = _mean(stats['pen_capped'], select)
    out['penetration_over_5mm_frac'] = _mean(stats['pen_deep_frac'], select)
    out['penetration_max_m'] = float(stats['pen_max'][select].max()) if select.any() else None
    out['strain_err'] = _mean(stats['strain_err'], select)
    out['strain_abs'] = _mean(stats['strain_abs'], select)
    out['diverged'] = int(stats['diverged'][select].sum())
    return out


def groups(data):
    count = len(data['kind'])
    out = {'all': np.ones(count, bool)}
    for kind in sorted(set(data['kind'].tolist())):
        out[f'kind:{kind}'] = data['kind'] == kind
    shapes = sorted(set(data['soft_shape'].tolist()))
    if shapes != ['unknown']:
        for shape in shapes:
            out[f'shape:{shape}'] = data['soft_shape'] == shape
    boxes = np.asarray(data['mask'].any(1).cpu()) if data['mask'].shape[1] else np.zeros(count, bool)
    out['with_rigid_box'], out['no_rigid_box'] = boxes, ~boxes
    return {k: v for k, v in out.items() if v.any()}


@torch.no_grad()
def latency_ms(step_fn, data, start_mode, steps=40, warmup=5):
    """Median wall ms per step: batch 1, CPU, the first episode, autoregressive from the seed."""
    batch = ts.episodes(data, slice(0, 1), torch.device('cpu'))
    start, prev0 = seed(batch, start_mode, data['vel0'][:1])
    cur = batch['verts'][:, start].float()
    prev = prev0 if prev0 is not None else batch['verts'][:, start - 1].float()
    last = batch['verts'].shape[1] - 2
    times = []
    for i in range(warmup + steps):
        began = time.perf_counter()
        nxt = step_fn(cur, prev, batch, min(start + i, last)).float()
        times.append(time.perf_counter() - began)
        prev, cur = cur, (nxt if torch.isfinite(nxt).all() else cur)
    return float(np.median(times[warmup:])) * 1000              # median: the machine is shared with training jobs


# ----------------------------------------------------------------------------- example export
def _round(array, digits=4):
    return np.round(np.asarray(array, np.float64), digits).tolist()


@torch.no_grad()
def export_example(engines, data, index, device, args, split, stats):
    batch = ts.episodes(data, slice(index, index + 1), device)
    start, prev0 = seed(batch, args.start, data['vel0'][index:index + 1].to(device))
    n = int(data['vert_mask'][index].sum())
    faces = data['faces'][index].numpy() if torch.is_tensor(data['faces']) else data['faces'][index]
    faces = faces[(faces >= 0).all(-1) & (faces < n).all(-1)]
    edges = data['rest_edges'][index].cpu().numpy()
    edges = edges[(edges >= 0).all(-1) & (edges < n).all(-1)]
    out = dict(
        note='EXPERIMENTAL soft-body rollout (SoftGNN). ' + ts.RIGID_NOTE + '. Y-up metres; frames are 0.05 s apart.',
        experimental=True, rigid_boxes_prescribed=True, split=split, episode=int(index), kind=str(data['kind'][index]),
        soft_shape=str(data['soft_shape'][index]), selection=args.example_rule, step_s=STEP, frames=int(batch['verts'].shape[1]),
        seeded_frames=start + 1, start=args.start, n_verts=n, faces=faces.astype(int).tolist(), edges=edges.astype(int).tolist(),
        soft_props=_round(data['soft_props'][index].cpu().numpy(), 5),
        context=dict(zip(('support_w', 'support_d', 'support_height', 'gravity', 'tilt_x', 'tilt_y', 'tilt_z', 'support_collisions_enabled',
                          'support_friction', 'support_restitution'), _round(data['context'][index].cpu().numpy(), 5))),
        true=_round(batch['verts'][0, :, :n].cpu().numpy()))
    mask = data['mask'][index].cpu().numpy().astype(bool)
    out['rigid_boxes'] = [dict(size=_round(data['props'][index, i, :3].cpu().numpy()), is_static=bool(data['props'][index, i, 5] > .5),
                               poses=_round(data['states'][index, :, i, :7].cpu().numpy()))      # pos3 + quat xyzw per frame (ground truth)
                          for i in np.nonzero(mask)[0]]
    out['predicted'], out['vert_err_mm'] = {}, {}
    for name in [x for x in args.example_engines.split(',') if x]:
        if name in engines:
            traj, _ = ts.rollout(engines[name], batch, start, prev0)
            out['predicted'][name] = _round(traj[0, :, :n].cpu().numpy())
            out['vert_err_mm'][name] = _round(stats[name]['vert_err'][index] * 1000, 2)
    return out


def pick_example(stats, data, args):
    if args.example_index >= 0:
        return min(args.example_index, len(data['kind']) - 1)
    engine = 'soft_projected' if 'soft_projected' in stats else 'rigid_freefall'
    error = np.nan_to_num(stats[engine]['vert_err'][:, -1], nan=np.inf)
    select = np.ones(len(error), bool)
    if args.example_kind:
        select = data['kind'] == args.example_kind
        if not select.any():
            raise ValueError(f'no episode of kind {args.example_kind}')
    candidates = np.nonzero(select)[0]
    return int(candidates[np.argsort(error[candidates])[len(candidates) // 2]])        # the median-error episode: representative, not the best


# ----------------------------------------------------------------------------- report
def _fmt(value, scale=1., digits=1):
    return '-' if value is None or (isinstance(value, float) and not math.isfinite(value)) else f'{value * scale:.{digits}f}'


def markdown(result):
    lines = ['# Soft-body evaluation (EXPERIMENTAL)', '',
             f"**{result['note']}**", '',
             f"Checkpoint `{result['checkpoint']}` ({result.get('checkpoint_info')}); data `{result['data']}` split `{result['split']}`, "
             f"{result['episodes']} episodes, {result['frames'] - 1} steps of {STEP} s; {result['seed_note']}", '',
             'Engines:'] + [f"- `{name}`: {e['description']}" for name, e in result['engines'].items()] + ['']
    header = ('| engine | vert 0.5s (mm) | vert 1s | vert 2s | vert final | COM 1s (mm) | COM 2s | shape 2s (mm) | final-rest COM (mm) | '
              'pen capped mean (mm) | pen >5mm (%) | pen max (mm) | strain err | diverged | CPU ms/step |')
    rule = '|' + '---|' * (header.count('|') - 1)

    def table(group):
        rows = [header, rule]
        for name, e in result['engines'].items():
            r = e['groups'].get(group)
            if r is None:
                continue
            rows.append(f"| {name} | {_fmt(r['vert_err_0.5s_m'], 1e3)} | {_fmt(r['vert_err_1s_m'], 1e3)} | {_fmt(r['vert_err_2s_m'], 1e3)} | "
                        f"{_fmt(r['vert_err_final_m'], 1e3)} | {_fmt(r['com_err_1s_m'], 1e3)} | {_fmt(r['com_err_2s_m'], 1e3)} | "
                        f"{_fmt(r['shape_err_2s_m'], 1e3)} | {_fmt(r['final_rest_com_err_m'], 1e3)} | {_fmt(r['penetration_capped_mean_m'], 1e3, 2)} | "
                        f"{_fmt(r['penetration_over_5mm_frac'], 100, 2)} | {_fmt(r['penetration_max_m'], 1e3)} | {_fmt(r['strain_err'], 1, 4)} | "
                        f"{r['diverged']} | {_fmt(e.get('latency_ms_per_step'), 1, 2)} |")
        return rows
    for group, count in result['group_counts'].items():
        lines += [f'## {group} ({count} episodes)', ''] + table(group) + ['']
    lines += ['Notes:'] + [f'- {n}' for n in result['notes']] + ['']
    return '\n'.join(lines)


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return str(value) if isinstance(value, Path) else value


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', type=Path, default=Path('data/soft_v1'))
    p.add_argument('--split', default='test')
    p.add_argument('--ckpt', type=Path, default=None, help='shipped-format soft_gnn checkpoint; omit to score only the baselines')
    p.add_argument('--out', type=Path, default=Path('models/soft_eval.json'))
    p.add_argument('--example', type=Path, default=Path('docs/soft-example.json'), help="example rollout JSON ('' = skip)")
    p.add_argument('--example-index', type=int, default=-1, help='episode to export (-1 = the median final-frame error of soft_projected)')
    p.add_argument('--example-engines', default='soft_raw,soft_projected', help='comma separated engines whose trajectory goes into the example')
    p.add_argument('--example-kind', default='', help='restrict the automatic example choice to one scenario kind')
    p.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--max-episodes', type=int, default=0, help='evenly spaced subset (0 = all)')
    p.add_argument('--batch', type=int, default=256)
    p.add_argument('--start', choices=('frames', 'velocity'), default='frames')
    p.add_argument('--project-margin', type=float, default=0., help='clearance (m) of the soft_projected engine')
    p.add_argument('--margin-engine', type=float, default=.005, help='clearance (m) of the extra soft_projected_margin engine (0 = omit)')
    p.add_argument('--sleep-move', type=float, default=.0015, help='per-step vertex motion (m) below which the soft_projected_sleep engine holds a supported body (0 = omit)')
    p.add_argument('--baseline-clearance', type=float, default=.005,
                   help='height (m) above the table / floor at which the rigid_freefall baseline stops (the data\'s soft collision margin)')
    p.add_argument('--max-strain', type=float, default=0., help='strain limit passed to project_soft in the projected engines (0 = off)')
    p.add_argument('--amp', choices=('off', 'fp16'), default='off')
    p.add_argument('--latency-steps', type=int, default=40)
    p.add_argument('--max-gpu-gb', type=float, default=0, help='hard cap on the CUDA memory of this process (0 = none)')
    p.add_argument('--threads', type=int, default=8, help='torch CPU threads')
    args = p.parse_args(argv)
    torch.set_num_threads(max(1, args.threads))
    device = torch.device(args.device)
    if device.type == 'cuda' and args.max_gpu_gb > 0:
        torch.cuda.set_per_process_memory_fraction(min(1., args.max_gpu_gb * 2**30 / torch.cuda.get_device_properties(device).total_memory), device)
    began = time.perf_counter()
    notes = ['EXPERIMENTAL research prototype. ' + ts.RIGID_NOTE + ' in every engine.',
             'COM is the unweighted mean of the soft vertices; shape error removes each trajectory\'s COM translation (no rotation alignment).',
             'Penetration and strain error are averaged over the predicted frames only; physics_reference shows what the ground truth itself scores '
             '(PyBullet soft nodes hover about one collision margin above surfaces, so its penetration is near zero).',
             f'The penetration column is the mean vertex depth with each vertex capped at {ts.PEN_CAP_M * 1000:.0f} mm (one fall-through rollout cannot dominate); '
             'the JSON also holds the uncapped mean.',
             'final-rest COM error is the last-frame COM error over episodes whose ground truth is at rest at the last frame.',
             'Pinned (kinematic) vertices, if the data has any, are prescribed from the ground truth in the model engines and count in the vertex means '
             '(with zero error there); the trivial baselines do not move them either way.']

    raw = ts.load_split(args.data, args.split, args.max_episodes, spread=True)
    data = ts.to_device(raw, 'cpu', ts.TENSOR_KEYS + ('vel0',))
    if bool(data['pinned'].any()):
        notes.append(f'{float(data["pinned"].float().sum(1).mean()):.1f} pinned vertices per episode on average.')
    data.update(kind=raw['kind'], soft_shape=raw['soft_shape'], faces=raw['faces'])
    rest_end = raw['rest_end'].astype(bool)
    count, frames = data['verts'].shape[:2]
    print(f'[data] {args.data}/{args.split}: {count} episodes, {frames - 1} steps, {raw["files"]} shards; device {device}', flush=True)
    for frame, label in HORIZONS:
        if frame >= frames:
            notes.append(f'episodes are shorter than {label}: that horizon is empty')

    model, info = None, None
    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=True)
        model = sg.build_from_checkpoint(ckpt).to(device)
        info = dict(kind=ckpt.get('kind'), config=ckpt.get('config'), epoch=ckpt.get('epoch'),
                    params=sum(q.numel() for q in model.parameters()))
    else:
        notes.append('no checkpoint given: only the baselines were scored')
    amp_dtype = torch.float16 if args.amp == 'fp16' and device.type == 'cuda' else None
    engines = build_engines(model, args, amp_dtype)
    stats = evaluate(engines, data, device, args, log=lambda m: print(m, flush=True))
    masks = groups(data)
    report = {name: dict(description=ENGINE_INFO[name], groups={g: aggregate(s, m, rest_end) for g, m in masks.items()})
              for name, s in stats.items()}
    if model is not None:
        cpu_engines = build_engines(sg.build_from_checkpoint(ckpt), args)
        for name, step_fn in cpu_engines.items():
            report[name]['latency_ms_per_step'] = latency_ms(step_fn, data, args.start, args.latency_steps)
        notes.append(f'CPU ms/step: batch 1, first episode of the split ({int(data["vert_mask"][0].sum())} vertices), '
                     f'{torch.get_num_threads()} torch threads, float32, median over {args.latency_steps} autoregressive steps.')
    seed_note = ('rollouts are seeded with the true frames (0, 1) and predict frames 2..T' if args.start == 'frames' else
                 'rollouts are seeded with frame 0 and its recorded vertex velocity and predict frames 1..T')
    result = dict(
        experimental=True, rigid_boxes_prescribed=True,
        note='EXPERIMENTAL; ' + ts.RIGID_NOTE + '.',
        checkpoint=str(args.ckpt), checkpoint_info=info, data=str(args.data), split=args.split, episodes=count, frames=frames,
        step_s=STEP, start=args.start, seed_note=seed_note, horizons={label: frame for frame, label in HORIZONS},
        args={k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        group_counts={g: int(m.sum()) for g, m in masks.items()}, rest_at_end_episodes=int(rest_end.sum()),
        engines=report, notes=notes)
    if str(args.example) not in ('', '.'):          # Path('') is '.'
        index = pick_example(stats, data, args)
        args.example_rule = (f'episode {index} requested with --example-index' if args.example_index >= 0 else
                             'median final-frame vertex error of soft_projected' + (f' within kind {args.example_kind}' if args.example_kind else ''))
        example = export_example(engines, data, index, device, args, args.split, stats)
        args.example.parent.mkdir(parents=True, exist_ok=True)
        args.example.write_text(json.dumps(clean(example), separators=(',', ':')))
        result['example'] = dict(path=str(args.example), episode=index, kind=example['kind'], selection=args.example_rule,
                                 bytes=args.example.stat().st_size)
        print(f'[example] episode {index} ({example["kind"]}) -> {args.example} ({args.example.stat().st_size / 1024:.0f} kB)', flush=True)
    result['seconds'] = time.perf_counter() - began
    result = clean(result)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1))
    text = markdown(result)
    args.out.with_suffix('.md').write_text(text)
    print(text.split('## kind:')[0], flush=True)
    print(f'[done] {args.out} and {args.out.with_suffix(".md")} in {result["seconds"]:.0f}s', flush=True)
    return result


if __name__ == '__main__':
    main()
