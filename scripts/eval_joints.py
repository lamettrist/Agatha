"""Evaluate the EXPERIMENTAL articulated-body world model (agatha.joint_gnn.JointGNN) on data/joints_v1.

Every engine is rolled out autoregressively from the true t=0 state for all T steps (never teacher-forced):

  physics_reference   (e) PyBullet ground truth scored with the same metrics (floor of every constraint metric)
  joint_raw           (a) JointGNN alone (learned model only)
  joint_projected     (b) JointGNN + joint_gnn.project_joints after every step (contact + joint projection, non-learned)
  contact_projected   (c) baseline: box model models/contact_gnn.pt + contact_gnn.project, joints ignored
  hold_initial        (d) trivial baseline: every body keeps its initial pose

Metrics per scenario kind and overall: body position error at 0.5 / 1 / 2 / 4 s (all bodies, articulated bodies,
free bodies), orientation error, joint coordinate error (hinge rad, prismatic m), mean / max anchor gap (mm),
off-axis error, limit violations, penetration (pairs connected by a joint excluded), fell accuracy, rest drift,
CPU ms/step at batch 1.

The same JointGNN checkpoint is also rolled out on a box-only dataset (default data/v2 test, zero joints) next to
contact_gnn, to quantify any regression on plain box dynamics.

Usage:
  CUDA_VISIBLE_DEVICES=7 .venv-train/bin/python scripts/eval_joints.py --model models/runs/NAME/best.pt \
      [--data data/joints_v1] [--split test] [--contact models/contact_gnn.pt] [--box-data data/v2] \
      [--out models/joint_eval.json] [--device cuda:0] [--max-episodes N] [--max-box-episodes N]
Writes <out> (JSON) and <out>.md (markdown).
"""
import argparse, copy, glob, json, math, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch
from agatha import contact_gnn as cg
import eval_contact as ec                      # proven box metrics: rollout, episode_stats, aggregate, bootstrap
import train_joint_ddp as tj                   # jointed_pairs, joint-aware penetration, joint_metrics, the joint_gnn module

jg = tj.jg
STEP = ec.STEP
HORIZONS_S = ec.HORIZONS_S
JOINT_DIM = tj.JOINT_DIM
HINGE_LIMIT_TOL_RAD = .02
PRISMATIC_LIMIT_TOL_M = .002
SLOP_HINGE_RAD, SLOP_PRISMATIC_M = (float(x) for x in getattr(jg, 'LIMIT_SLOP', (.05, .005)))   # what the projection itself tolerates
EXPERIMENTAL_NOTE = ('EXPERIMENTAL: articulated-body (joint) extension track. Not the shipped model; the shipped box model is '
                     'models/contact_gnn.pt. Synthetic PyBullet box-link articulations only.')
ENGINE_INFO = {
    'physics_reference': ('PyBullet ground truth scored with the same metrics (floor for constraint / penetration / rest metrics)', False),
    'joint_raw': ('JointGNN alone: learned model only, no projection', False),
    'joint_projected': ('JointGNN + joint_gnn.project_joints after every step (non-learned contact + joint projection)', True),
    'contact_projected': ('baseline: box model contact_gnn + contact_gnn.project, joints ignored', True),
    'contact_raw': ('box model contact_gnn alone', False),
    'hold_initial': ('trivial baseline: every body keeps its initial pose (zero velocity)', False),
}
ARRAY_KEYS = ec.ARRAY_KEYS + ('joints', 'joint_mask', 'joint_pos', 'group')


# ----------------------------------------------------------------------------- data
def load_split(data: Path, split: str, max_episodes: int | None = None, joint_slots: int = 12) -> dict:
    """Shards of a joint dataset, or of a box dataset (then joints are empty: zero rows, joint_mask False)."""
    files = sorted(glob.glob(str(Path(data) / split / 'shard_*.npz')))
    if not files: raise FileNotFoundError(f'No shards in {Path(data) / split}')
    parts = {k: [] for k in ARRAY_KEYS}
    for name in files:
        with np.load(name) as shard:
            for k in ARRAY_KEYS:
                if k in shard.files: parts[k].append(shard[k])
    out = {k: np.concatenate(v) for k, v in parts.items() if len(v) == len(files)}
    missing = [k for k in ('states', 'props', 'context', 'mask', 'rest', 'fell', 'tipped') if k not in out]
    if missing: raise KeyError(f'Shards are missing required arrays: {missing}')
    count, frames, slots = out['states'].shape[:3]
    out.setdefault('kind', np.array(['unknown'] * count)); out.setdefault('live_physics', np.zeros(count, bool))
    out.setdefault('chaotic', np.zeros(count, bool))
    out['kind'] = out['kind'].astype(str)
    out['has_joint_arrays'] = 'joints' in out
    if 'joints' not in out:
        out['joints'] = np.zeros((count, joint_slots, JOINT_DIM), np.float32); out['joint_mask'] = np.zeros((count, joint_slots), bool)
    out['joints'] = np.where(out['joint_mask'][..., None].astype(bool), out['joints'], 0).astype(np.float32)
    if 'group' not in out:           # derive: every body that takes part in a joint is 'articulated' (one group per episode)
        group = np.full((count, slots), -1, np.int8)
        for column in (0, 1):
            index = out['joints'][..., column].round().astype(int)
            e, j = np.nonzero(out['joint_mask'].astype(bool) & (index >= 0) & (index < slots))
            group[e, index[e, j]] = 0
        out['group'] = group
    if max_episodes and max_episodes < count:   # evenly spaced, keeps the generator's scenario mix
        keep = np.unique(np.linspace(0, count - 1, max_episodes).round().astype(int))
        out = {k: (v[keep] if isinstance(v, np.ndarray) else v) for k, v in out.items()}
    out['files'] = len(files)
    return out


# ----------------------------------------------------------------------------- joint statistics
def joint_stats(traj: torch.Tensor, batch: dict) -> dict:
    """Per-episode sufficient statistics of the joint constraints / coordinates for one engine (float64, [B] or [B,H])."""
    true, joints, joint_mask = batch['states'], batch['joints'], batch['joint_mask']
    count, frames = traj.shape[:2]; steps = frames - 1; slots = joints.shape[1]
    zero = torch.zeros(count, dtype=torch.float64, device=traj.device)
    names = ('gap_sum', 'gap_max', 'off_sum', 'off_max', 'hinge_err_sum', 'prismatic_err_sum', 'limit_over', 'limit_over_slop', 'limit_excess_sum', 'limit_max',
             'joint_frames', 'axial_frames', 'oriented_frames', 'hinge_frames', 'prismatic_frames', 'n_joints', 'n_hinge', 'n_prismatic')
    out = {name: zero.clone() for name in names}
    horizon = {name: torch.zeros(count, len(HORIZONS_S), dtype=torch.float64, device=traj.device)
               for name in ('gap_h', 'off_h', 'hinge_err_h', 'prismatic_err_h', 'joint_h_n', 'oriented_h_n', 'hinge_h_n', 'prismatic_h_n')}
    out.update(horizon)
    if not bool(joint_mask.any()): return out
    spread = lambda x: x[:, None].expand(count, frames, *x.shape[1:]).reshape(count * frames, *x.shape[1:])
    m = tj.joint_metrics(traj.reshape(count * frames, *traj.shape[2:]), true.reshape(count * frames, *true.shape[2:]), spread(joints), spread(joint_mask))
    m = {k: v.reshape(count, frames, slots) for k, v in m.items()}
    valid, axial, hinge, prismatic, oriented = (m[k].double() for k in ('valid', 'axial', 'hinge', 'prismatic', 'oriented'))
    gap, off, hinge_err, prismatic_err, excess = (m[k].double() for k in ('anchor_gap', 'off_axis', 'hinge_err', 'prismatic_err', 'limit_excess'))
    later = slice(1, None)
    out['gap_sum'] = (gap * valid)[:, later].sum((1, 2)); out['gap_max'] = (gap * valid)[:, later].amax((1, 2))
    out['off_sum'] = (off * oriented)[:, later].sum((1, 2)); out['off_max'] = (off * oriented)[:, later].amax((1, 2))
    out['hinge_err_sum'] = (hinge_err * hinge)[:, later].sum((1, 2)); out['prismatic_err_sum'] = (prismatic_err * prismatic)[:, later].sum((1, 2))
    over = ((excess > HINGE_LIMIT_TOL_RAD) & m['hinge']) | ((excess > PRISMATIC_LIMIT_TOL_M) & m['prismatic'])
    over_slop = ((excess > SLOP_HINGE_RAD) & m['hinge']) | ((excess > SLOP_PRISMATIC_M) & m['prismatic'])
    out['limit_over_slop'] = over_slop[:, later].double().sum((1, 2))
    out['limit_over'] = over[:, later].double().sum((1, 2)); out['limit_excess_sum'] = (excess * axial)[:, later].sum((1, 2))
    out['limit_max'] = (excess * axial)[:, later].amax((1, 2))
    for name, select in (('joint', valid), ('axial', axial), ('oriented', oriented), ('hinge', hinge), ('prismatic', prismatic)):
        out[f'{name}_frames'] = select[:, later].sum((1, 2))
    out['n_joints'], out['n_hinge'], out['n_prismatic'] = valid[:, 0].sum(-1), hinge[:, 0].sum(-1), prismatic[:, 0].sum(-1)
    for column, seconds in enumerate(HORIZONS_S):
        index = round(seconds / STEP)
        if index > steps: continue
        for name, value, select in (('gap_h', gap, valid), ('off_h', off, oriented), ('hinge_err_h', hinge_err, hinge), ('prismatic_err_h', prismatic_err, prismatic)):
            out[name][:, column] = (value * select)[:, index].sum(-1)
        for name, select in (('joint_h_n', valid), ('oriented_h_n', oriented), ('hinge_h_n', hinge), ('prismatic_h_n', prismatic)):
            out[name][:, column] = select[:, index].sum(-1)
    return out


def body_group_stats(traj: torch.Tensor, batch: dict) -> dict:
    """Position / orientation error sums at the horizons, split into articulated (group >= 0) and free bodies."""
    true = batch['states']; steps = true.shape[1] - 1
    dyn = ec.dynamic_mask(batch['props'], batch['mask'])
    select = dict(art=(dyn & (batch['group'] >= 0)).double(), free=(dyn & (batch['group'] < 0)).double())
    out = {}
    for name, weight in select.items():
        pos, ori, n = [], [], []
        for seconds in HORIZONS_S:
            index = round(seconds / STEP)
            if index <= steps:
                pos.append(((traj[:, index, :, :3] - true[:, index, :, :3]).norm(dim=-1).double() * weight).sum(-1))
                ori.append((ec.orientation_error_deg(traj[:, index, :, 3:7], true[:, index, :, 3:7]).double() * weight).sum(-1))
                n.append(weight.sum(-1))
            else:
                zero = torch.zeros_like(weight[:, 0]); pos.append(zero); ori.append(zero); n.append(zero)
        out[f'pos_{name}_sum'], out[f'ori_{name}_sum'], out[f'{name}_n'] = torch.stack(pos, -1), torch.stack(ori, -1), torch.stack(n, -1)
    return out


_ratio = ec._ratio


def joint_aggregate(s: dict, index: np.ndarray) -> dict:
    """Pool the joint statistics over the selected episodes."""
    pick = lambda name: s[name][index]
    horizon = lambda num, den, scale=1.: {ec.horizon_key(h): (None if (v := _ratio(pick(num)[:, i].sum(), pick(den)[:, i].sum())) is None else v * scale)
                                          for i, h in enumerate(HORIZONS_S)}
    scaled = lambda value, scale: None if value is None else value * scale
    count = int(pick('n_joints').sum())
    return dict(
        n_joints=count, n_hinge=int(pick('n_hinge').sum()), n_prismatic=int(pick('n_prismatic').sum()),
        anchor_gap_mm=dict(mean=scaled(_ratio(pick('gap_sum').sum(), pick('joint_frames').sum()), 1000.),
                           max=float(pick('gap_max').max()) * 1000 if count else None, at=horizon('gap_h', 'joint_h_n', 1000.)),
        off_axis_deg=dict(mean=scaled(_ratio(pick('off_sum').sum(), pick('oriented_frames').sum()), 180 / math.pi),
                          max=float(pick('off_max').max()) * 180 / math.pi if count else None, at=horizon('off_h', 'oriented_h_n', 180 / math.pi)),
        hinge_coord_err_rad=dict(rollout_mean=_ratio(pick('hinge_err_sum').sum(), pick('hinge_frames').sum()), at=horizon('hinge_err_h', 'hinge_h_n')),
        prismatic_coord_err_m=dict(rollout_mean=_ratio(pick('prismatic_err_sum').sum(), pick('prismatic_frames').sum()), at=horizon('prismatic_err_h', 'prismatic_h_n')),
        limit=dict(frac_joint_frames_over=_ratio(pick('limit_over').sum(), pick('axial_frames').sum()),
                   frac_joint_frames_over_slop=_ratio(pick('limit_over_slop').sum(), pick('axial_frames').sum()),
                   slop_tolerance=dict(hinge_rad=SLOP_HINGE_RAD, prismatic_m=SLOP_PRISMATIC_M),
                   mean_excess=_ratio(pick('limit_excess_sum').sum(), pick('axial_frames').sum()),
                   max_excess=float(pick('limit_max').max()) if count else None,
                   tolerance=dict(hinge_rad=HINGE_LIMIT_TOL_RAD, prismatic_m=PRISMATIC_LIMIT_TOL_M)),
        pos_err_articulated_m=horizon('pos_art_sum', 'art_n'), pos_err_free_m=horizon('pos_free_sum', 'free_n'),
        ori_err_articulated_deg=horizon('ori_art_sum', 'art_n'), ori_err_free_deg=horizon('ori_free_sum', 'free_n'),
        n_articulated_bodies=int(pick('art_n').max(1).sum()) if len(pick('art_n')) else 0,
        n_free_bodies=int(pick('free_n').max(1).sum()) if len(pick('free_n')) else 0)


def engine_report(s: dict, kinds: np.ndarray, chaotic: np.ndarray) -> dict:
    def block(index):
        row = ec.aggregate(s, index); row['joints'] = joint_aggregate(s, index); return row
    everything = np.ones(len(kinds), bool)
    return dict(overall=block(everything), gentle=block(~chaotic), chaotic=block(chaotic),
                per_kind={kind: block(kinds == kind) for kind in sorted(set(kinds.tolist()))})


# ----------------------------------------------------------------------------- engines
def joint_step(model):
    return lambda state, props, context, mask, joints, joint_mask: model(state, props, context, mask, joints, joint_mask)


def joint_projected_step(model, iters: int, **options):
    def step(state, props, context, mask, joints, joint_mask):
        return jg.project_joints(state, model(state, props, context, mask, joints, joint_mask), props, context, mask, joints, joint_mask, iters=iters, **options)
    return step


def contact_step(model, project: bool):
    def step(state, props, context, mask, joints, joint_mask):
        nxt = model(state, props, context, mask)
        return cg.project(state, nxt, props, context, mask) if project else nxt
    return step


def hold_step(state, props, context, mask, joints, joint_mask):
    return torch.cat((state[..., :7], torch.zeros_like(state[..., 7:])), -1)


def evaluate(engines: dict, data: dict, device, batch_size: int, log=print) -> dict:
    """engines: {name: 6-argument step_fn or None (ground truth)} -> {name: per-episode numpy statistics}."""
    count = len(data['states']); steps = data['states'].shape[1] - 1
    collected = {name: [] for name in engines}
    with torch.inference_mode():
        for start in range(0, count, batch_size):
            stop = min(start + batch_size, count)
            batch = {k: torch.from_numpy(np.ascontiguousarray(data[k][start:stop])).to(device)
                     for k in ('states', 'props', 'context', 'mask', 'rest', 'fell', 'tipped', 'joints', 'joint_mask', 'group')}
            for k in ('states', 'props', 'context', 'joints'): batch[k] = batch[k].float()
            for k in ('mask', 'rest', 'fell', 'tipped', 'joint_mask'): batch[k] = batch[k].bool()
            joints, joint_mask = batch['joints'], batch['joint_mask']
            adjacent = tj.jointed_pairs(joints, joint_mask, batch['states'].shape[2])
            pen_fn = lambda state, props, context, mask: tj.penetration_depths(state, props, context, mask, adjacent)
            for name, step_fn in engines.items():
                if step_fn is None: traj, diverged = batch['states'], None
                else:
                    traj, diverged = ec.rollout(lambda s, p, c, m: step_fn(s, p, c, m, joints, joint_mask),
                                                batch['states'][:, 0], batch['props'], batch['context'], batch['mask'], steps)
                stats = ec.episode_stats(traj, batch, diverged, pen_fn)
                stats.update(joint_stats(traj, batch)); stats.update(body_group_stats(traj, batch))
                collected[name].append({k: v.cpu().numpy() for k, v in stats.items()})
            log(f'  episodes {stop}/{count}')
    return {name: {k: np.concatenate([part[k] for part in parts]) for k in parts[0]} for name, parts in collected.items()}


def latency_ms(step_fn, data: dict, steps: int = 60, warmup: int = 5) -> float:
    """Mean wall ms per step: batch 1, CPU, autoregressive, on the first episode that has a joint (else the first episode)."""
    has = np.flatnonzero(data['joint_mask'].any(1)); e = int(has[0]) if len(has) else 0
    joints = torch.as_tensor(data['joints'][e:e + 1]).float(); joint_mask = torch.as_tensor(data['joint_mask'][e:e + 1]).bool()
    one = {k: data[k][e:e + 1] for k in ('states', 'props', 'context', 'mask')}
    return ec.latency_ms(lambda s, p, c, m: step_fn(s, p.float(), c.float(), m.bool(), joints, joint_mask), one, steps, warmup)


def data_checks(data: dict, device, batch_size: int) -> dict:
    """Ground-truth consistency: joint_gnn.joint_coordinates(true states) against the dataset's joint_pos."""
    if 'joint_pos' not in data or not data['joint_mask'].any(): return dict(joint_pos_available='joint_pos' in data)
    sums = np.zeros(4)
    worst = np.zeros(2)
    with torch.inference_mode():
        for start in range(0, len(data['states']), batch_size):
            sl = slice(start, start + batch_size)
            states = torch.from_numpy(data['states'][sl]).float().to(device); joints = torch.from_numpy(data['joints'][sl]).float().to(device)
            joint_mask = torch.from_numpy(data['joint_mask'][sl]).bool().to(device); truth = torch.from_numpy(data['joint_pos'][sl]).float().to(device)
            count, frames = states.shape[:2]
            spread = lambda x: x[:, None].expand(count, frames, *x.shape[1:]).reshape(count * frames, *x.shape[1:])
            q = jg.joint_coordinates(states.reshape(count * frames, *states.shape[2:]), spread(joints), spread(joint_mask)).reshape(count, frames, -1)
            hinge = (joint_mask & (joints[..., 2] > .5))[:, None].expand_as(q); prismatic = (joint_mask & (joints[..., 4] > .5))[:, None].expand_as(q)
            dh, dp = tj.wrap_angle(q - truth).abs(), (q - truth).abs()
            sums += [float((dh * hinge).sum()), float(hinge.sum()), float((dp * prismatic).sum()), float(prismatic.sum())]
            worst = np.maximum(worst, [float((dh * hinge).max()), float((dp * prismatic).max())])
    return dict(joint_pos_available=True, description='|joint_gnn.joint_coordinates(true states) - dataset joint_pos| over all frames (hinge wrapped to +-pi)',
                hinge_mean_abs_diff_rad=_ratio(sums[0], sums[1]), hinge_max_abs_diff_rad=float(worst[0]) if sums[1] else None,
                prismatic_mean_abs_diff_m=_ratio(sums[2], sums[3]), prismatic_max_abs_diff_m=float(worst[1]) if sums[3] else None)


def zero_joint_equivalence(joint_model, contact_model, data: dict, device, n: int = 256) -> dict:
    """One-step |JointGNN(zero joints) - ContactGNN| on box states (exactly 0 only for an un-finetuned warm start)."""
    with torch.inference_mode():
        index = np.unique(np.linspace(0, len(data['states']) - 1, min(n, len(data['states']))).round().astype(int))
        frame = data['states'].shape[1] // 4
        state = torch.from_numpy(data['states'][index, frame]).float().to(device); props = torch.from_numpy(data['props'][index]).float().to(device)
        context = torch.from_numpy(data['context'][index]).float().to(device); mask = torch.from_numpy(data['mask'][index]).bool().to(device)
        joints = torch.zeros(len(index), 12, JOINT_DIM, device=device); joint_mask = torch.zeros(len(index), 12, dtype=torch.bool, device=device)
        diff = (joint_model(state, props, context, mask, joints, joint_mask) - contact_model(state, props, context, mask)).abs()
        return dict(n_states=len(index), max_abs_state_diff=float(diff.max()), mean_abs_state_diff=float(diff.mean()))


# ----------------------------------------------------------------------------- reporting
def _fmt(value, scale=1., digits=3):
    return '-' if value is None else f'{value * scale:.{digits}f}'


def _row(name: str, e: dict, block: dict) -> str:
    pos, j, pen = block['pos_err_m'], block['joints'], block['penetration']
    return (f"| {name} | {_fmt(pos['0.5s'], digits=4)} | {_fmt(pos['1s'], digits=4)} | {_fmt(pos['2s'], digits=4)} | {_fmt(pos['4s'], digits=4)} | "
            f"{_fmt(j['pos_err_articulated_m']['2s'], digits=4)} | {_fmt(j['pos_err_free_m']['2s'], digits=4)} | {_fmt(block['ori_err_deg']['2s'], digits=2)} | "
            f"{_fmt(j['hinge_coord_err_rad']['at']['1s'])} | {_fmt(j['hinge_coord_err_rad']['at']['2s'])} | {_fmt(j['prismatic_coord_err_m']['at']['2s'], digits=4)} | "
            f"{_fmt(j['anchor_gap_mm']['mean'], digits=2)} | {_fmt(j['anchor_gap_mm']['max'], digits=1)} | {_fmt(j['off_axis_deg']['mean'], digits=2)} | "
            f"{_fmt(j['limit']['frac_joint_frames_over'], 100, 2)} | {_fmt(j['limit']['frac_joint_frames_over_slop'], 100, 2)} | {_fmt(pen['mean_m'], 1000, 2)} | {_fmt(pen['frac_over_5mm'], 100, 2)} | "
            f"{_fmt(block['fell']['accuracy'])} | {_fmt(block['rest_drift']['fd_speed_mps'], 1000, 2)} | {block['diverged_episodes']} | "
            f"{_fmt(e.get('latency_ms_per_step'), digits=2)} |")


HEADER = ('| engine | pos 0.5s (m) | pos 1s | pos 2s | pos 4s | pos 2s articulated | pos 2s free | ori 2s (deg) | hinge err 1s (rad) | hinge err 2s | '
          'slide err 2s (m) | gap mean (mm) | gap max (mm) | off-axis mean (deg) | limit viol (%) | limit viol > slop (%) | pen mean (mm) | pen >5mm (%) | fell acc | '
          'rest drift (mm/s) | diverged | ms/step |')


def markdown(result: dict) -> str:
    lines = ['# Joint (articulated body) evaluation - EXPERIMENTAL', '', f"> {result['experimental_note']}", '']
    d = result['data']
    lines += [f"Model `{result['model']['checkpoint']}` (config {result['model']['config']}, epoch {result['model']['epoch']}, "
              f"{result['model']['parameters']:,} parameters).", '',
              f"Data `{d['path']}` split `{d['split']}`: {d['n_episodes']} episodes, {d['rollout_seconds']:g} s autoregressive rollouts, "
              f"{d['slots']} body slots, {d['n_joints']} joints ({d['n_hinge']} hinge, {d['n_ball']} ball, {d['n_prismatic']} prismatic, {d['n_fixed']} fixed).", '']
    engines = result['engines']
    lines += ['Engines:'] + [f"- `{name}`: {e['description']}" for name, e in engines.items()] + ['']
    for subset in ('overall', 'gentle', 'chaotic'):
        first = next(iter(engines.values()))[subset]
        if not first['n_episodes']: continue
        lines += [f"## {subset} ({first['n_episodes']} episodes)", '', HEADER, '|' + '---|' * 22]
        lines += [_row(name, e, e[subset]) for name, e in engines.items()] + ['']
    kinds = sorted(next(iter(engines.values()))['per_kind'])
    if kinds:
        lines += ['## Per scenario kind: pos err 2s (m) / hinge err 2s (rad) / mean anchor gap (mm) / mean penetration (mm)', '',
                  '| kind | n | ' + ' | '.join(engines) + ' |', '|' + '---|' * (len(engines) + 2)]
        for kind in kinds:
            cells = []
            for e in engines.values():
                b = e['per_kind'][kind]
                cells.append(f"{_fmt(b['pos_err_m']['2s'], digits=4)} / {_fmt(b['joints']['hinge_coord_err_rad']['at']['2s'])} / "
                             f"{_fmt(b['joints']['anchor_gap_mm']['mean'], digits=2)} / {_fmt(b['penetration']['mean_m'], 1000, 2)}")
            lines.append(f"| {kind} | {next(iter(engines.values()))['per_kind'][kind]['n_episodes']} | " + ' | '.join(cells) + ' |')
        lines.append('')
    box = result.get('box_regression')
    if box:
        b = box['data']
        lines += [f"## Box-only regression check (`{b['path']}` split `{b['split']}`, {b['n_episodes']} episodes, zero joints)", '',
                  '| engine | subset | pos 1s (m) | pos 2s | pos 4s | ori 2s (deg) | fell acc | fell P | fell R | pen mean (mm) | pen >5mm (%) | rest drift (mm/s) | diverged |',
                  '|' + '---|' * 13]
        for name, e in box['engines'].items():
            for subset in ('overall', 'gentle', 'chaotic'):
                m = e[subset]
                lines.append(f"| {name} | {subset} | {_fmt(m['pos_err_m']['1s'], digits=4)} | {_fmt(m['pos_err_m']['2s'], digits=4)} | {_fmt(m['pos_err_m']['4s'], digits=4)} | "
                             f"{_fmt(m['ori_err_deg']['2s'], digits=2)} | {_fmt(m['fell']['accuracy'])} | {_fmt(m['fell']['precision'])} | {_fmt(m['fell']['recall'])} | "
                             f"{_fmt(m['penetration']['mean_m'], 1000, 2)} | {_fmt(m['penetration']['frac_over_5mm'], 100, 2)} | "
                             f"{_fmt(m['rest_drift']['fd_speed_mps'], 1000, 2)} | {m['diverged_episodes']} |")
        lines += ['', 'Paired bootstrap, JointGNN (zero joints) minus contact_gnn (negative = JointGNN better):', '',
                  '| pair | metric | contact_gnn | joint_gnn | delta | 95% CI |', '|---|---|---|---|---|---|']
        for pair, block in box['deltas'].items():
            for metric, v in block.items():
                ci = v.get('ci95')
                lines.append(f"| {pair} | {metric} | {_fmt(v['old'], digits=5)} | {_fmt(v['new'], digits=5)} | {_fmt(v['delta'], digits=5)} | "
                             f"{'-' if not ci else f'[{ci[0]:.5f}, {ci[1]:.5f}]'} |")
        eq = box.get('zero_joint_one_step')
        if eq: lines += ['', f"One-step |JointGNN(zero joints) - ContactGNN| on {eq['n_states']} box states: max {eq['max_abs_state_diff']:.3e}, mean {eq['mean_abs_state_diff']:.3e} "
                             '(exactly zero is only expected for a warm start that has not been trained further).']
        lines.append('')
    lines += ['Notes:'] + [f'- {note}' for note in result['notes']] + ['']
    return '\n'.join(lines)


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', type=Path, required=True, help='joint_gnn checkpoint (shipped format, e.g. <run>/best.pt)')
    parser.add_argument('--data', type=Path, default=Path('data/joints_v1')); parser.add_argument('--split', default='test')
    parser.add_argument('--contact', type=Path, default=Path('models/contact_gnn.pt'), help='box baseline checkpoint')
    parser.add_argument('--box-data', type=Path, default=Path('data/v2'), help='box-only dataset for the regression check')
    parser.add_argument('--box-split', default='test')
    parser.add_argument('--out', type=Path, default=Path('models/joint_eval.json'))
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--max-episodes', type=int); parser.add_argument('--max-box-episodes', type=int)
    parser.add_argument('--batch-size', type=int, default=512); parser.add_argument('--project-iters', type=int, default=6)
    parser.add_argument('--bootstrap', type=int, default=1000); parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--no-box', action='store_true', help='skip the box-only regression check')
    parser.add_argument('--no-latency', action='store_true')
    parser.add_argument('--joint-slop', type=float, default=None, help='project_joints joint_slop (m); default: joint_gnn.JOINT_CONTACT_SLOP; 0 = previous behaviour')
    parser.add_argument('--project-outer', type=int, default=None, help='project_joints outer (contact + joint passes per step; default 1)')
    parser.add_argument('--slop-scope', choices=('scene', 'articulation'), default=None, help='project_joints slop_scope')
    parser.add_argument('--no-articulation-sleep', action='store_true', help='project_joints articulation_sleep=False (previous behaviour)')
    parser.add_argument('--max-gpu-gb', type=float, default=0, help='hard cap on the CUDA memory of this process (0 = none); for shared GPUs')
    args = parser.parse_args(argv)
    device = torch.device(args.device); began = time.perf_counter()
    if device.type == 'cuda' and args.max_gpu_gb > 0:
        torch.cuda.set_per_process_memory_fraction(min(1., args.max_gpu_gb * 2**30 / torch.cuda.get_device_properties(device).total_memory), device)
    say = lambda text: print(text, flush=True)
    notes = [EXPERIMENTAL_NOTE,
             'Autoregressive rollouts from the true t=0 state over the full episode; never teacher-forced.',
             'joint_projected applies joint_gnn.project_joints (non-learned contact + joint projection) after every step; joint_raw is the learned model alone.',
             'contact_projected ignores the joints entirely (the shipped box model on the same bodies): the gap to it is what the joint edges buy.',
             'Joint coordinate error compares joint_gnn.joint_coordinates of the predicted and of the TRUE states (hinge error wrapped to +-pi); '
             'ball and fixed joints have no coordinate. data_checks reports how well that function reproduces the dataset joint_pos.',
             'Anchor gap / off-axis / limit metrics come from joint_gnn.joint_violation; physics_reference shows PyBullet\'s own residual.',
             f'Limit violation = joint-frames with limit excess > {HINGE_LIMIT_TOL_RAD} rad (hinge) or {PRISMATIC_LIMIT_TOL_M} m (prismatic); '
             f'"> slop" counts only excess beyond joint_gnn.LIMIT_SLOP = {SLOP_HINGE_RAD} rad / {SLOP_PRISMATIC_M} m, the tolerance project_joints itself '
             'clamps to (Bullet limits are soft, so physics_reference is non-zero in the strict column).',
             'Penetration = contact_gnn corner-in-box depth (support, floor, other boxes) with pairs directly connected by a joint excluded.',
             'fell / rest drift follow scripts/eval_contact.py (dataset fell flag and rest frames).',
             'Latency: batch 1, CPU (4 threads), 60 autoregressive steps on the first episode with a joint; shared machine, indicative only.']

    checkpoint = torch.load(args.model, map_location='cpu', weights_only=True)
    model = jg.build_from_checkpoint(checkpoint).to(device).eval()
    model_info = dict(checkpoint=str(args.model), kind=checkpoint.get('kind'), config=checkpoint.get('config'), epoch=checkpoint.get('epoch'),
                      train_args=checkpoint.get('train_args'), train_metrics=checkpoint.get('metrics'), parameters=sum(p.numel() for p in model.parameters()))
    contact = None
    if args.contact and Path(args.contact).exists():
        contact = cg.build_from_checkpoint(torch.load(args.contact, map_location='cpu', weights_only=True)).to(device).eval()
    else:
        notes.append(f'{args.contact} not found: the contact_gnn baseline and the box regression comparison are omitted.')

    project_options = {}
    if args.joint_slop is not None: project_options['joint_slop'] = args.joint_slop
    if args.slop_scope is not None: project_options['slop_scope'] = args.slop_scope
    if args.project_outer is not None: project_options['outer'] = args.project_outer
    if args.no_articulation_sleep: project_options['articulation_sleep'] = False
    if project_options: notes.append(f'project_joints options: {project_options}')

    def build_engines(joint_data: bool):
        engines = {'physics_reference': None, 'joint_raw': joint_step(model), 'joint_projected': joint_projected_step(model, args.project_iters, **project_options)}
        cpu = {}
        if not args.no_latency:
            cpu_model = copy.deepcopy(model).cpu()
            cpu.update(joint_raw=joint_step(cpu_model), joint_projected=joint_projected_step(cpu_model, args.project_iters, **project_options))
        if contact is not None:
            engines['contact_projected'] = contact_step(contact, True)
            if not joint_data: engines['contact_raw'] = contact_step(contact, False)
            if not args.no_latency: cpu['contact_projected'] = contact_step(copy.deepcopy(contact).cpu(), True)
        if joint_data:
            engines['hold_initial'] = hold_step
            if not args.no_latency: cpu['hold_initial'] = hold_step
        return engines, cpu

    # ---- joint data
    data = load_split(args.data, args.split, args.max_episodes)
    count, frames, slots = data['states'].shape[:3]
    say(f'{count} episodes, {frames - 1} steps, {slots} slots, {int(data["joint_mask"].sum())} joints from {data["files"]} shard(s); device {device}')
    if not data['has_joint_arrays']: notes.append(f'{args.data} has no joints array: evaluated as a box dataset.')
    engines, cpu_engines = build_engines(True)
    stats = evaluate(engines, data, device, args.batch_size, say)
    kinds, chaotic = data['kind'], data['chaotic'].astype(bool)
    report = {name: dict(description=ENGINE_INFO[name][0], includes_non_learned_projection=ENGINE_INFO[name][1], latency_ms_per_step=None,
                         **engine_report(stats[name], kinds, chaotic)) for name in engines}
    if cpu_engines:
        threads = torch.get_num_threads(); torch.set_num_threads(4)
        for name, step_fn in cpu_engines.items(): report[name]['latency_ms_per_step'] = latency_ms(step_fn, data)
        torch.set_num_threads(threads)
    types = data['joints'][..., 2:6][data['joint_mask'].astype(bool)]
    meta_path = Path(args.data) / 'meta.json'
    result = dict(
        schema_version=1, experimental=True, experimental_note=EXPERIMENTAL_NOTE, created_unix=time.time(), seconds=None,
        args={k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}, model=model_info,
        data=dict(path=str(args.data), split=args.split, n_episodes=count, steps=frames - 1, step_s=STEP, rollout_seconds=(frames - 1) * STEP, slots=int(slots),
                  shards=data['files'], n_joints=int(data['joint_mask'].sum()), n_hinge=int((types[:, 0] > .5).sum()), n_ball=int((types[:, 1] > .5).sum()),
                  n_prismatic=int((types[:, 2] > .5).sum()), n_fixed=int((types[:, 3] > .5).sum()), n_chaotic=int(chaotic.sum()),
                  kinds={k: int((kinds == k).sum()) for k in sorted(set(kinds.tolist()))},
                  meta=json.loads(meta_path.read_text()) if meta_path.exists() else None),
        data_checks=data_checks(data, device, args.batch_size),
        metric_definitions=dict(
            pos_err_m='mean |p_pred - p_true| over dynamic real bodies at the horizon; joints.pos_err_articulated_m / pos_err_free_m split by the dataset group id (>= 0 = articulated)',
            ori_err_deg='geodesic quaternion angle, same pooling',
            hinge_coord_err_rad='mean |wrap(q_pred - q_true)| over hinge joints, q from joint_gnn.joint_coordinates', prismatic_coord_err_m='mean |q_pred - q_true| over prismatic joints',
            anchor_gap_mm='joint_gnn.joint_violation anchor_gap over all joint-frames t>=1 (mean, max) and at the horizons',
            off_axis_deg='joint_gnn.joint_violation off_axis over hinge (axis misalignment) and prismatic/fixed (relative rotation) joint-frames', limit='limit_excess from joint_gnn.joint_violation over hinge/prismatic joint-frames',
            penetration='as scripts/eval_contact.py, jointed pairs excluded', fell='as scripts/eval_contact.py', rest_drift='as scripts/eval_contact.py',
            latency_ms_per_step='batch 1, CPU (4 threads), mean over 60 autoregressive steps'),
        engines=report, notes=notes)

    # ---- box-only regression
    if not args.no_box:
        try:
            box = load_split(args.box_data, args.box_split, args.max_box_episodes)
        except FileNotFoundError as error:
            box = None; notes.append(f'Box regression check skipped: {error}')
        if box is not None:
            say(f'box regression: {len(box["states"])} episodes from {args.box_data}/{args.box_split}')
            box_engines, _ = build_engines(False)
            box_stats = evaluate(box_engines, box, device, args.batch_size, say)
            box_kinds, box_chaotic = box['kind'], box['chaotic'].astype(bool)
            box_report = {}
            for name in box_engines:
                full = engine_report(box_stats[name], box_kinds, box_chaotic)
                for subset in ('overall', 'gentle', 'chaotic'): full[subset].pop('joints', None)
                full['per_kind'] = {k: {m: v[m] for m in ('n_episodes', 'pos_err_m', 'fell', 'penetration')} for k, v in full['per_kind'].items()}
                box_report[name] = dict(description=ENGINE_INFO[name][0] + (' (zero joints)' if name.startswith('joint') else ''), **full)
            deltas = {}
            if contact is not None:
                everything = np.ones(len(box_kinds), bool)
                metrics = dict(pos_err_2s_m=(everything, 'pos_sum', 'horizon_n', HORIZONS_S.index(2.)),
                               chaotic_pos_err_2s_m=(box_chaotic, 'pos_sum', 'horizon_n', HORIZONS_S.index(2.)),
                               gentle_pos_err_2s_m=(~box_chaotic, 'pos_sum', 'horizon_n', HORIZONS_S.index(2.)),
                               mean_penetration_m=(everything, 'pen_sum', 'pen_n', None),
                               rest_drift_mps=(everything, 'rest_fd_sum', 'rest_n', None))
                for pair, old, new in (('projected', 'contact_projected', 'joint_projected'), ('raw', 'contact_raw', 'joint_raw')):
                    deltas[pair] = {name: ec.bootstrap_delta(box_stats[old], box_stats[new], select, num, den, column, args.bootstrap, args.seed)
                                    for name, (select, num, den, column) in metrics.items()}
            result['box_regression'] = dict(
                data=dict(path=str(args.box_data), split=args.box_split, n_episodes=len(box['states']), n_chaotic=int(box_chaotic.sum())),
                engines=box_report, deltas=deltas,
                zero_joint_one_step=zero_joint_equivalence(model, contact, box, device) if contact is not None else None,
                note='Same JointGNN checkpoint with zero joints against contact_gnn; deltas are joint_gnn minus contact_gnn (paired episode bootstrap).')

    o = report['joint_projected']['overall']; r = report['joint_raw']['overall']
    pick = lambda block: dict(pos_err_m={k: block['pos_err_m'][k] for k in ('0.5s', '1s', '2s')}, ori_err_deg_2s=block['ori_err_deg']['2s'],
                              hinge_coord_err_rad_2s=block['joints']['hinge_coord_err_rad']['at']['2s'], anchor_gap_mm_mean=block['joints']['anchor_gap_mm']['mean'],
                              anchor_gap_mm_max=block['joints']['anchor_gap_mm']['max'], limit_violation_frac=block['joints']['limit']['frac_joint_frames_over'],
                              limit_violation_frac_over_slop=block['joints']['limit']['frac_joint_frames_over_slop'],
                              mean_penetration_m=block['penetration']['mean_m'], fell_accuracy=block['fell']['accuracy'],
                              rest_drift_mps=block['rest_drift']['fd_speed_mps'], diverged_episodes=block['diverged_episodes'])
    result['summary'] = dict(experimental=True, **{name: pick(e['overall']) | dict(latency_ms_per_step=e.get('latency_ms_per_step')) for name, e in report.items()})
    del o, r
    result['seconds'] = time.perf_counter() - began
    result = ec.clean(result)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2)); args.out.with_suffix('.md').write_text(markdown(result))
    print(markdown(result)); print(f'wrote {args.out} and {args.out.with_suffix(".md")} in {result["seconds"]:.1f}s', flush=True)
    return result


if __name__ == '__main__':
    main()
