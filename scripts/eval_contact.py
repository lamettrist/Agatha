"""Evaluate learned world models on the contact-v2 dataset (held-out split).

Every engine is rolled out autoregressively from the true t=0 state for all T
steps (never teacher-forced). Engines:

  physics_reference  ground truth itself (metric floor: PyBullet's own penetration etc.)
  old_mlp            models/mlp_multistep.pt (the currently shipped model)
  old_graph          models/graph_impulse.pt (optional)
  old_mlp_projected  old MLP + contact_gnn.project after every step (attribution)
  new_raw            ContactGNN alone (learned model only)
  new_projected      ContactGNN + contact_gnn.project after every step (would ship)

Usage:
  CUDA_VISIBLE_DEVICES=0 .venv-train/bin/python scripts/eval_contact.py --data data/v2 --split test \
      --old models/mlp_multistep.pt [--old-graph models/graph_impulse.pt] [--new ckpt.pt] \
      --out models/contact_eval.json [--device cuda:0] [--max-episodes N]
"""
import argparse, copy, glob, json, math, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from agatha.dynamics import build_model, euler_quaternion, rotation

STEP = .05
HORIZONS_S = (.5, 1., 2., 4.)
TIP_COS = math.cos(math.radians(35))
PEN_THRESHOLD_M = .005
FLOOR_HALF_M = 10.
PEN_CAP_M = .05          # robust mean: per-frame depth capped so a single fall-through-the-floor rollout cannot dominate
DIVERGE_LIMIT = 1e3   # |any state component| above this (or non-finite) => rollout counted as diverged
GENTLE_TOLERANCE = 1.02
ACTIVE_DISPLACEMENT_M = .02   # ground-truth max displacement above this (or fell / tipped) => 'active' object
PEN_MATCH_M = .001            # new_projected may exceed old_mlp_projected's capped penetration by at most this
PHYSICS_GROUPS = ('all_physics', 'live_physics')
OUTCOMES = ('all', 'gentle', 'chaotic')
ENGINE_INFO = {
    'physics_reference': ('PyBullet ground truth scored with the same metrics (floor for penetration / rest drift)', False),
    'old_mlp': ('old shipped model alone (mlp_multistep)', False),
    'old_graph': ('old graph_impulse model alone', False),
    'old_mlp_projected': ('old mlp_multistep + the non-learned contact projection (attribution only)', True),
    'new_raw': ('ContactGNN alone: learned model only, no projection', False),
    'new_projected': ('ContactGNN + non-learned contact projection after every step (configuration that would ship)', True),
}


def horizon_key(seconds: float) -> str:
    return f'{seconds:g}s'


# ----------------------------------------------------------------------------- metric primitives

def dynamic_mask(props: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Real, non-static object slots [.., N]."""
    return mask & (props[..., 5] < .5)


def orientation_error_deg(q_pred: torch.Tensor, q_true: torch.Tensor) -> torch.Tensor:
    """Geodesic angle (degrees) between xyzw quaternions; sign-invariant."""
    a = q_pred.double(); b = q_true.double()
    a = a / a.norm(dim=-1, keepdim=True).clamp_min(1e-12); b = b / b.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    dot = (a * b).sum(-1).abs().clamp(max=1.)
    # atan2 form stays accurate near 0 where acos loses precision.
    return torch.rad2deg(2 * torch.atan2((1 - dot * dot).clamp_min(0).sqrt(), dot)).float()


def support_local(pos: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
    """Positions [B, ..., 3] in the (tilted) support frame centred at (0, height-.04, 0)."""
    rot = rotation(euler_quaternion(context[:, 4:7]))                      # [B,3,3]
    center = torch.zeros(context.shape[0], 3, dtype=pos.dtype, device=pos.device); center[:, 1] = context[:, 2] - .04
    extra = (1,) * (pos.dim() - 2)
    rel = pos - center.view(-1, *extra, 3)
    return torch.einsum('bji,b...j->b...i', rot, rel)                      # R^T @ rel


def fell_tipped(traj: torch.Tensor, props: torch.Tensor, context: torch.Tensor, mask: torch.Tensor):
    """Batched agatha.physics.summarize rule. traj [B,F,N,13] -> (fell[B,N], tipped[B,N]) bool."""
    local = support_local(traj[..., :3], context)                           # [B,F,N,3]
    half_w = (context[:, 0] / 2).view(-1, 1, 1); half_d = (context[:, 1] / 2).view(-1, 1, 1)
    disabled = (context[:, 7] < .5).view(-1, 1, 1)
    off = disabled | (local[..., 0].abs() > half_w) | (local[..., 2].abs() > half_d)
    dyn = dynamic_mask(props, mask)
    fell = (off & (local[..., 1] < .01)).any(1) & dyn
    x, _, z, _ = traj[..., 3:7].unbind(-1)
    tipped = ((1 - 2 * (x * x + z * z)) < TIP_COS).any(1) & dyn             # rotation matrix element [1][1]
    return fell, tipped


def rest_drift(traj: torch.Tensor, rest: torch.Tensor, dyn: torch.Tensor):
    """Per-episode sums over object-frames (t>=1) flagged at rest in the dataset.
    Returns (finite-difference speed sum, |v| state sum, count), each [B]."""
    fd = (traj[:, 1:, :, :3] - traj[:, :-1, :, :3]).norm(dim=-1) / STEP
    speed = traj[:, 1:, :, 7:10].norm(dim=-1)
    sel = (rest[:, 1:] & dyn[:, None, :]).double()
    return (fd.double() * sel).sum((1, 2)), (speed.double() * sel).sum((1, 2)), sel.sum((1, 2))


def confusion(pred: torch.Tensor, true: torch.Tensor, dyn: torch.Tensor) -> dict:
    p, t = pred & dyn, true & dyn
    f = lambda v: v.sum(-1).double()
    return dict(tp=f(p & t), fp=f(p & ~t & dyn), fn=f(~p & t & dyn), tn=f(~p & ~t & dyn))


def rollout(step_fn, state0: torch.Tensor, props, context, mask, steps: int):
    """Autoregressive rollout. A non-finite / exploding episode is frozen at its last good state
    and reported in `diverged` (never dropped). Returns traj [B,steps+1,N,13], diverged [B]."""
    traj = [state0]; state = state0
    diverged = torch.zeros(state0.shape[0], dtype=torch.bool, device=state0.device)
    for _ in range(steps):
        nxt = step_fn(state, props, context, mask)
        bad = (~torch.isfinite(nxt) | (nxt.abs() > DIVERGE_LIMIT)).flatten(1).any(1)
        diverged = diverged | bad
        state = torch.where(diverged[:, None, None], state, nxt)
        traj.append(state)
    return torch.stack(traj, 1), diverged


def episode_stats(traj: torch.Tensor, batch: dict, diverged: torch.Tensor | None = None, pen_fn=None) -> dict:
    """Per-episode sufficient statistics (all [B] or [B,H] float64 tensors) for one engine."""
    true, props, context, mask = batch['states'], batch['props'], batch['context'], batch['mask']
    steps = true.shape[1] - 1
    dyn = dynamic_mask(props, mask); dynf = dyn.double()
    out = dict(n_dyn=dynf.sum(-1), n_frames=dynf.sum(-1) * steps)
    pos_err = (traj[..., :3] - true[..., :3]).norm(dim=-1).double()         # [B,F,N]
    out['pos_rollout_sum'] = (pos_err[:, 1:] * dynf[:, None]).sum((1, 2))
    pos_sum, ori_sum, valid = [], [], []
    for seconds in HORIZONS_S:
        index = round(seconds / STEP)
        if index <= steps:
            pos_sum.append((pos_err[:, index] * dynf).sum(-1))
            ori_sum.append((orientation_error_deg(traj[:, index, :, 3:7], true[:, index, :, 3:7]).double() * dynf).sum(-1))
            valid.append(out['n_dyn'])
        else:
            zero = torch.zeros_like(out['n_dyn']); pos_sum.append(zero); ori_sum.append(zero); valid.append(zero)
    out['pos_sum'] = torch.stack(pos_sum, -1); out['ori_sum'] = torch.stack(ori_sum, -1); out['horizon_n'] = torch.stack(valid, -1)
    # 'active' objects: the ones that really move in the ground truth (bystanders at rest dilute the chaotic mean)
    moved = (true[..., :3] - true[:, :1, :, :3]).norm(dim=-1).amax(1) > ACTIVE_DISPLACEMENT_M
    active = (dyn & (moved | batch['fell'] | batch['tipped'])).double()
    index = [round(seconds / STEP) for seconds in HORIZONS_S]
    out['pos_active_sum'] = torch.stack([(pos_err[:, i] * active).sum(-1) if i <= steps else torch.zeros_like(out['n_dyn']) for i in index], -1)
    out['active_n'] = torch.stack([active.sum(-1) if i <= steps else torch.zeros_like(out['n_dyn']) for i in index], -1)
    # the generator's own validity rule: such a rollout left the world even if no value is non-finite
    position = traj[..., :3]
    outside = ~torch.isfinite(position).all(-1) | (position[..., 1] < -.5) | (position[..., 0].abs() > FLOOR_HALF_M) | (position[..., 2].abs() > FLOOR_HALF_M) | (position.norm(dim=-1) > 50)
    out['out_of_bounds'] = (outside & dyn[:, None]).flatten(1).any(1).double()
    fell, tipped = fell_tipped(traj, props, context, mask)
    for name, pred, key in (('fell', fell, 'fell'), ('tip', tipped, 'tipped')):
        for k, v in confusion(pred, batch[key], dyn).items(): out[f'{name}_{k}'] = v
    out['rest_fd_sum'], out['rest_v_sum'], out['rest_n'] = rest_drift(traj, batch['rest'], dyn)
    out['diverged'] = (diverged if diverged is not None else torch.zeros_like(dyn[:, 0])).double()
    zero = torch.zeros_like(out['n_dyn'])
    pen = dict(pen_sum=zero.clone(), pen_cap_sum=zero.clone(), pen_over=zero.clone(), pen_max=zero.clone(), pen_support_sum=zero.clone(),
               pen_floor_sum=zero.clone(), pen_objects_sum=zero.clone(), pen_n=zero.clone())
    if pen_fn is not None:
        for t in range(1, steps + 1):
            depth = pen_fn(traj[:, t], props, context, mask)
            parts = {k: torch.nan_to_num(depth[k].double(), nan=0., posinf=1e3).clamp_min(0) * dynf for k in ('support', 'floor', 'objects')}
            # The floor is a finite 20 m x 20 m box: an object that slid past its edge (in truth or prediction) is not penetrating it.
            parts['floor'] = parts['floor'] * ((traj[:, t, :, 0].abs() <= FLOOR_HALF_M) & (traj[:, t, :, 2].abs() <= FLOOR_HALF_M))
            worst = torch.stack(list(parts.values()), 0).amax(0)
            pen['pen_sum'] += worst.sum(-1); pen['pen_cap_sum'] += worst.clamp_max(PEN_CAP_M).sum(-1); pen['pen_over'] += ((worst > PEN_THRESHOLD_M).double() * dynf).sum(-1)
            pen['pen_max'] = torch.maximum(pen['pen_max'], worst.amax(-1))
            for k, v in parts.items(): pen[f'pen_{k}_sum'] += v.sum(-1)
        pen['pen_n'] = out['n_frames'].clone()
    out.update(pen)
    return out


# ----------------------------------------------------------------------------- aggregation

def _ratio(num: float, den: float):
    return float(num / den) if den > 0 else None


def _prf(s: dict, name: str, index) -> dict:
    tp, fp, fn, tn = (float(s[f'{name}_{k}'][index].sum()) for k in ('tp', 'fp', 'fn', 'tn'))
    return dict(accuracy=_ratio(tp + tn, tp + fp + fn + tn), precision=_ratio(tp, tp + fp), recall=_ratio(tp, tp + fn),
                confusion=dict(tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn)))


def aggregate(s: dict, index: np.ndarray) -> dict:
    """Pool per-episode statistics (numpy) over the selected episodes (bool mask or indices)."""
    index = np.asarray(index)
    count = int(index.sum()) if index.dtype == bool else len(index)
    n = s['horizon_n'][index].sum(0)
    out = dict(n_episodes=count, n_dynamic_objects=int(s['n_dyn'][index].sum()),
               pos_err_m={horizon_key(h): _ratio(s['pos_sum'][index][:, i].sum(), n[i]) for i, h in enumerate(HORIZONS_S)},
               pos_err_rollout_mean_m=_ratio(s['pos_rollout_sum'][index].sum(), s['n_frames'][index].sum()),
               ori_err_deg={horizon_key(h): _ratio(s['ori_sum'][index][:, i].sum(), n[i]) for i, h in enumerate(HORIZONS_S)},
               fell=_prf(s, 'fell', index), tipped=_prf(s, 'tip', index))
    active_n = s['active_n'][index].sum(0)
    out['pos_err_active_m'] = {horizon_key(h): _ratio(s['pos_active_sum'][index][:, i].sum(), active_n[i]) for i, h in enumerate(HORIZONS_S)}
    out['n_active_objects'] = int(s['active_n'][index].max(1).sum()) if count else 0
    pen_n = s['pen_n'][index].sum()
    out['penetration'] = dict(mean_m=_ratio(s['pen_sum'][index].sum(), pen_n), mean_capped_50mm_m=_ratio(s['pen_cap_sum'][index].sum(), pen_n), frac_over_5mm=_ratio(s['pen_over'][index].sum(), pen_n),
                              max_m=float(s['pen_max'][index].max()) if pen_n > 0 else None,
                              mean_support_m=_ratio(s['pen_support_sum'][index].sum(), pen_n),
                              mean_floor_m=_ratio(s['pen_floor_sum'][index].sum(), pen_n),
                              mean_objects_m=_ratio(s['pen_objects_sum'][index].sum(), pen_n), n_object_frames=int(pen_n))
    rest_n = s['rest_n'][index].sum()
    out['rest_drift'] = dict(fd_speed_mps=_ratio(s['rest_fd_sum'][index].sum(), rest_n),
                             state_speed_mps=_ratio(s['rest_v_sum'][index].sum(), rest_n), n_object_frames=int(rest_n))
    out['diverged_episodes'] = int(s['diverged'][index].sum())
    out['out_of_bounds_episodes'] = int(s['out_of_bounds'][index].sum())
    return out


def subset_masks(chaotic: np.ndarray, live: np.ndarray) -> dict:
    """{physics group: {outcome: bool[E]}}; gentle = not chaotic."""
    chaotic = np.asarray(chaotic, bool); live = np.asarray(live, bool); everything = np.ones_like(chaotic)
    return {group: dict(all=base.copy(), gentle=base & ~chaotic, chaotic=base & chaotic)
            for group, base in (('all_physics', everything), ('live_physics', live))}


def engine_report(s: dict, chaotic, live, kinds) -> dict:
    masks = subset_masks(chaotic, live)
    report = dict(subsets={g: {o: aggregate(s, m) for o, m in group.items()} for g, group in masks.items()}, per_kind={})
    for kind in sorted(set(kinds.tolist())):
        select = kinds == kind
        report['per_kind'][kind] = dict(all_physics=aggregate(s, select), live_physics=aggregate(s, select & np.asarray(live, bool)))
    return report


HEADLINES = dict(chaotic_pos_err_2s_m=('chaotic', 'pos_sum', 'horizon_n', HORIZONS_S.index(2.)),
                 mean_penetration_m=('all', 'pen_sum', 'pen_n', None),
                 gentle_pos_err_2s_m=('gentle', 'pos_sum', 'horizon_n', HORIZONS_S.index(2.)),
                 mean_penetration_capped_50mm_m=('all', 'pen_cap_sum', 'pen_n', None),   # gates only vs old_mlp_projected
                 chaotic_active_pos_err_2s_m=('chaotic', 'pos_active_sum', 'active_n', HORIZONS_S.index(2.)))   # informational


def _column(values: np.ndarray, column):
    return values if column is None else values[:, column]


def bootstrap_delta(old: dict, new: dict, select: np.ndarray, num: str, den: str, column=None,
                    resamples: int = 1000, seed: int = 0) -> dict:
    """Paired episode-level bootstrap of (new - old) for a pooled ratio-of-sums metric."""
    index = np.flatnonzero(select)
    arrays = [_column(d[k], column)[index] for d in (old, new) for k in (num, den)]
    if not len(index) or arrays[1].sum() <= 0 or arrays[3].sum() <= 0:
        return dict(old=None, new=None, delta=None, ci95=None, n_episodes=int(len(index)))
    point_old, point_new = arrays[0].sum() / arrays[1].sum(), arrays[2].sum() / arrays[3].sum()
    rng = np.random.default_rng(seed); deltas = []
    for start in range(0, resamples, 50):
        pick = rng.integers(0, len(index), (min(50, resamples - start), len(index)))
        sums = [a[pick].sum(1) for a in arrays]
        with np.errstate(invalid='ignore', divide='ignore'):
            deltas.append(sums[2] / sums[3] - sums[0] / sums[1])
    deltas = np.concatenate(deltas); deltas = deltas[np.isfinite(deltas)]
    return dict(old=float(point_old), new=float(point_new), delta=float(point_new - point_old),
                ci95=[float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))] if len(deltas) else None,
                n_episodes=int(len(index)))


def ship_decision(stats: dict, chaotic, live, group: str = 'live_physics', resamples: int = 1000, seed: int = 0) -> dict:
    masks = subset_masks(chaotic, live)[group]
    decision = dict(subset=group, baseline_engine='old_mlp', primary_variant='new_projected', gentle_tolerance=GENTLE_TOLERANCE,
                    rule='ship iff, vs old_mlp: chaotic pos err @2s improves AND mean penetration improves AND gentle pos err @2s <= old * 1.02; '
                         'AND (when old_mlp_projected was evaluated) vs old_mlp_projected: chaotic pos err @2s improves AND gentle pos err @2s <= '
                         f'old_projected * 1.02 AND capped mean penetration <= old_projected + {PEN_MATCH_M * 1000:g} mm. Gates use point estimates; '
                         "'significant' reports whether the paired-bootstrap 95% CI agrees",
                    bootstrap=dict(resamples=resamples, seed=seed, unit='episode', paired=True, delta='new - old (negative = better)'),
                    variants={})
    for variant in ('new_projected', 'new_raw'):
        if 'old_mlp' not in stats or variant not in stats: continue
        block = dict(description=ENGINE_INFO[variant][0], includes_non_learned_projection=ENGINE_INFO[variant][1])
        for name, (outcome, num, den, column) in HEADLINES.items():
            block[name] = bootstrap_delta(stats['old_mlp'], stats[variant], masks[outcome], num, den, column, resamples, seed)
        c, pen, g = block['chaotic_pos_err_2s_m'], block['mean_penetration_m'], block['gentle_pos_err_2s_m']
        c['improves'] = None if c['delta'] is None else bool(c['new'] < c['old'])
        pen['improves'] = None if pen['delta'] is None else bool(pen['new'] < pen['old'])
        cap = block['mean_penetration_capped_50mm_m']; cap['improves'] = None if cap['delta'] is None else bool(cap['new'] < cap['old']); cap['gating'] = False
        g['limit'] = None if g['delta'] is None else g['old'] * GENTLE_TOLERANCE
        g['not_worse'] = None if g['delta'] is None else bool(g['new'] <= g['limit'])
        active = block['chaotic_active_pos_err_2s_m']; active['improves'] = None if active['delta'] is None else bool(active['new'] < active['old']); active['gating'] = False
        for v in (c, pen, cap, active): v['significant'] = None if not v.get('ci95') else bool(v['ci95'][1] < 0)
        g['significant'] = None if not g.get('ci95') else bool(g['ci95'][1] < g['old'] * (GENTLE_TOLERANCE - 1))   # non-inferiority
        flags = (c['improves'], pen['improves'], g['not_worse'])
        block['ship_vs_old_mlp'] = None if any(f is None for f in flags) else bool(all(flags))
        block['beats_old_projected'] = None
        if 'old_mlp_projected' in stats:
            # The projection alone removes most penetration and helps gentle scenes, so the learned model
            # must also beat 'old model + the same projection' before the gain is attributed to it.
            versus = {name: bootstrap_delta(stats['old_mlp_projected'], stats[variant], masks[outcome], num, den, column, resamples, seed)
                      for name, (outcome, num, den, column) in HEADLINES.items()}
            vc, vg, vp = versus['chaotic_pos_err_2s_m'], versus['gentle_pos_err_2s_m'], versus['mean_penetration_capped_50mm_m']
            vc['improves'] = None if vc['delta'] is None else bool(vc['new'] < vc['old'])
            vg['limit'] = None if vg['delta'] is None else vg['old'] * GENTLE_TOLERANCE
            vg['not_worse'] = None if vg['delta'] is None else bool(vg['new'] <= vg['limit'])
            vp['limit'] = None if vp['delta'] is None else vp['old'] + PEN_MATCH_M
            vp['not_worse'] = None if vp['delta'] is None else bool(vp['new'] <= vp['limit'])
            for v in versus.values(): v['significant'] = None if not v.get('ci95') else bool(v['ci95'][1] < 0)
            block['vs_old_projected'] = versus
            extra = (vc['improves'], vg['not_worse'], vp['not_worse'])
            block['beats_old_projected'] = None if any(f is None for f in extra) else bool(all(extra))
            flags = flags + extra
        block['ship'] = None if any(f is None for f in flags) else bool(all(flags))
        decision['variants'][variant] = block
    primary = decision['variants'].get('new_projected')
    decision['ship'] = primary['ship'] if primary else None
    raw = decision['variants'].get('new_raw')
    decision['model_alone_passes'] = raw['ship'] if raw else None
    decision['beats_old_projected'] = primary['beats_old_projected'] if primary else None
    decision['statement'] = ('new_projected = learned ContactGNN + NON-LEARNED contact projection (what would ship); new_raw = the learned model '
                             'ALONE. The penetration gate against raw old_mlp can be passed by the projection alone, so the gain is only '
                             'attributed to the learned model when beats_old_projected is true (same projection on the old model).')
    return decision


# ----------------------------------------------------------------------------- data / engines

ARRAY_KEYS = ('states', 'props', 'context', 'mask', 'contact_support', 'contact_floor', 'contact_object', 'rest', 'fell', 'tipped',
              'kind', 'chaotic', 'live_physics')


def load_split(data: Path, split: str, max_episodes: int | None = None) -> dict:
    files = sorted(glob.glob(str(Path(data) / split / 'shard_*.npz')))
    if not files: raise FileNotFoundError(f'No shards in {Path(data) / split}')
    parts = {k: [] for k in ARRAY_KEYS}
    for name in files:
        with np.load(name) as shard:
            for k in ARRAY_KEYS:
                if k in shard.files: parts[k].append(shard[k])
    out = {k: np.concatenate(v) for k, v in parts.items() if len(v) == len(files)}
    missing = [k for k in ('states', 'props', 'context', 'mask', 'rest', 'fell', 'tipped', 'chaotic') if k not in out]
    if missing: raise KeyError(f'Shards are missing required arrays: {missing}')
    count = len(out['states'])
    out.setdefault('kind', np.array(['unknown'] * count)); out.setdefault('live_physics', np.zeros(count, bool))
    out['kind'] = out['kind'].astype(str)
    if max_episodes and max_episodes < count:   # evenly spaced, keeps the generator's scenario mix
        keep = np.unique(np.linspace(0, count - 1, max_episodes).round().astype(int))
        out = {k: v[keep] for k, v in out.items()}
    out['files'] = len(files)
    return out


def load_old(path: Path, device) -> torch.nn.Module:
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    model = build_model(checkpoint['kind']); model.load_state_dict(checkpoint['state_dict'])
    return model.eval().to(device)


def old_step(model):
    return lambda state, props, context, mask: model(state, props[..., :6], context[:, :8], mask)


def projected_step(step_fn, project):
    return lambda state, props, context, mask: project(state, step_fn(state, props, context, mask), props, context, mask)


def latency_ms(step_fn, batch: dict, steps: int = 60, warmup: int = 5) -> float:
    """Mean wall ms per model step: batch 1, CPU, the first episode's N slots, autoregressive."""
    state, props, context, mask = (torch.as_tensor(batch[k][:1]).cpu() for k in ('states', 'props', 'context', 'mask'))
    state = state[:, 0].float()
    with torch.inference_mode():
        for i in range(warmup + steps):
            if i == warmup: start = time.perf_counter()
            nxt = step_fn(state, props, context, mask)
            state = nxt if torch.isfinite(nxt).all() else state
    return (time.perf_counter() - start) / steps * 1000


def evaluate(engines: dict, data: dict, device, batch_size: int = 1024, pen_fn=None, log=print) -> dict:
    """engines: {name: step_fn or None (ground truth)} -> {name: per-episode numpy statistics}."""
    count = len(data['states']); steps = data['states'].shape[1] - 1
    collected = {name: [] for name in engines}
    with torch.inference_mode():
        for start in range(0, count, batch_size):
            stop = min(start + batch_size, count)
            batch = {k: torch.from_numpy(np.ascontiguousarray(data[k][start:stop])).to(device)
                     for k in ('states', 'props', 'context', 'mask', 'rest', 'fell', 'tipped')}
            batch['states'] = batch['states'].float(); batch['props'] = batch['props'].float(); batch['context'] = batch['context'].float()
            for k in ('mask', 'rest', 'fell', 'tipped'): batch[k] = batch[k].bool()
            for name, step_fn in engines.items():
                if step_fn is None: traj, diverged = batch['states'], None
                else: traj, diverged = rollout(step_fn, batch['states'][:, 0], batch['props'], batch['context'], batch['mask'], steps)
                stats = episode_stats(traj, batch, diverged, pen_fn)
                collected[name].append({k: v.cpu().numpy() for k, v in stats.items()})
            log(f'  episodes {stop}/{count}')
    return {name: {k: np.concatenate([part[k] for part in parts]) for k in parts[0]} for name, parts in collected.items()}


# ----------------------------------------------------------------------------- reporting

def _get(tree, *path):
    for key in path:
        if not isinstance(tree, dict) or key not in tree: return None
        tree = tree[key]
    return tree


def summarize_for_ui(eval_json: dict) -> dict:
    """Small before/after dict for models/metrics.json['contact_v2'] (the lead merges it)."""
    engines = eval_json.get('engines', {})
    group = 'live_physics' if (_get(engines, 'old_mlp', 'subsets', 'live_physics', 'all', 'n_episodes') or 0) > 0 else 'all_physics'
    def headline(name: str) -> dict | None:
        if name not in engines: return None
        row = {}
        for outcome in OUTCOMES:
            m = engines[name]['subsets'][group][outcome]
            row[outcome] = dict(n_episodes=m['n_episodes'],
                                pos_err_m={k: m['pos_err_m'].get(k) for k in ('0.5s', '1s', '2s')},
                                pos_err_active_2s_m=m['pos_err_active_m'].get('2s'), out_of_bounds_episodes=m['out_of_bounds_episodes'],
                                ori_err_deg_2s=m['ori_err_deg'].get('2s'), fell_accuracy=m['fell']['accuracy'],
                                fell_precision=m['fell']['precision'], fell_recall=m['fell']['recall'],
                                mean_penetration_m=m['penetration']['mean_m'], mean_penetration_capped_50mm_m=m['penetration']['mean_capped_50mm_m'], frac_frames_penetrating_over_5mm=m['penetration']['frac_over_5mm'],
                                rest_drift_mps=m['rest_drift']['fd_speed_mps'], diverged_episodes=m['diverged_episodes'])
        row['latency_ms_per_step_cpu'] = engines[name].get('latency_ms_per_step')
        return row
    labels = dict(old='old_mlp', new_raw='new_raw', new_projected='new_projected', physics_reference='physics_reference',
                  old_projected='old_mlp_projected')
    decision = eval_json.get('ship_decision') or {}
    return dict(
        dataset=dict(path=_get(eval_json, 'data', 'path'), split=_get(eval_json, 'data', 'split'), n_episodes=_get(eval_json, 'data', 'n_episodes'),
                     rollout_seconds=_get(eval_json, 'data', 'rollout_seconds'), subset=group),
        **{label: headline(name) for label, name in labels.items() if name in engines},
        ship_decision=dict(ship=decision.get('ship'), model_alone_passes=decision.get('model_alone_passes'),
                           beats_old_projected=decision.get('beats_old_projected'),
                           variants={k: {m: {f: v[m].get(f) for f in ('old', 'new', 'delta', 'ci95', 'significant')} for m in HEADLINES}
                                     | dict(ship=v.get('ship'), ship_vs_old_mlp=v.get('ship_vs_old_mlp'), beats_old_projected=v.get('beats_old_projected'))
                                     for k, v in (decision.get('variants') or {}).items()}),
        live_subset_kinds={kind: _get(row, 'live_physics', 'n_episodes') for kind, row in (_get(engines, 'physics_reference', 'per_kind') or {}).items()},
        notes=['Autoregressive rollouts from the true t=0 state over the full episode; never teacher-forced.',
               'old = previously shipped mlp_multistep; new_raw = learned ContactGNN alone; new_projected = ContactGNN + a NON-LEARNED '
               'contact projection applied after every step (the shipped configuration).',
               f'Headline subset is {group}: episodes whose hidden physics parameters equal the live API defaults; '
               'gentle/chaotic is the dataset outcome flag (fell, tipped, >0.30 m displacement or >0.5 m/s object collision).',
               'The live subset is not a random half of every scenario: legacy_layout (the old model\'s own training distribution) is always '
               'live and so is over-represented in it (see live_subset_kinds), which favours the old model; live and all-physics numbers are not '
               'comparable kind by kind.',
               f'pos_err_active = position error over the objects that actually move in the ground truth (> {ACTIVE_DISPLACEMENT_M * 100:g} cm, fell or '
               'tipped); the plain chaotic mean also averages resting bystanders. The ship gates are point estimates; "significant" marks a paired-bootstrap 95% CI that excludes zero.',
               'out_of_bounds_episodes = rollouts where a dynamic object left the world (y < -0.5 m, |x| or |z| > 10 m); diverged_episodes only counts non-finite / >1e3 states.',
               'Penetration = mean over dynamic object-frames of the deepest overlap with support, floor or another box '
               '(corner-in-box test, so edge-edge overlaps are under-counted); physics_reference is the floor of this metric.',
               'Rest drift = mean finite-difference speed on object-frames where ground truth is at rest in contact.',
               'Synthetic PyBullet boxes only; errors grow with horizon and chaotic outcomes remain sensitive to initial conditions.'])


def _fmt(value, scale=1., digits=3):
    return '-' if value is None else f'{value * scale:.{digits}f}'


def markdown(result: dict) -> str:
    engines = result['engines']; lines = ['# Contact-v2 evaluation', '']
    d = result['data']
    lines += [f"Data `{d['path']}` split `{d['split']}`: {d['n_episodes']} episodes, {d['rollout_seconds']:g} s autoregressive rollouts "
              f"(live_physics {d['counts']['live_physics']['all']}, chaotic {d['counts']['all_physics']['chaotic']}, gentle {d['counts']['all_physics']['gentle']}).", '']
    lines += ['Engines:'] + [f"- `{name}`: {e['description']}" for name, e in engines.items()] + ['']
    header = '| engine | pos 0.5s (m) | pos 1s | pos 2s | pos 4s | pos 2s active | ori 2s (deg) | fell acc | fell P | fell R | pen mean (mm) | pen capped (mm) | pen >5mm (%) | pen max (mm) | rest drift (mm/s) | diverged | out of bounds | ms/step |'
    for group in reversed(PHYSICS_GROUPS):
        for outcome in OUTCOMES:
            first = next(iter(engines.values()))['subsets'][group][outcome]
            lines += [f"## {group} / {outcome} ({first['n_episodes']} episodes)", '', header, '|' + '---|' * 18]
            for name, e in engines.items():
                m = e['subsets'][group][outcome]; pos = m['pos_err_m']; pen = m['penetration']
                lines.append(f"| {name} | {_fmt(pos['0.5s'], digits=4)} | {_fmt(pos['1s'], digits=4)} | {_fmt(pos['2s'], digits=4)} | {_fmt(pos['4s'], digits=4)} | {_fmt(m['pos_err_active_m']['2s'], digits=4)} | "
                             f"{_fmt(m['ori_err_deg']['2s'], digits=2)} | {_fmt(m['fell']['accuracy'])} | {_fmt(m['fell']['precision'])} | {_fmt(m['fell']['recall'])} | "
                             f"{_fmt(pen['mean_m'], 1000, 2)} | {_fmt(pen['mean_capped_50mm_m'], 1000, 2)} | {_fmt(pen['frac_over_5mm'], 100, 2)} | {_fmt(pen['max_m'], 1000, 1)} | "
                             f"{_fmt(m['rest_drift']['fd_speed_mps'], 1000, 2)} | {m['diverged_episodes']} | {m['out_of_bounds_episodes']} | {_fmt(e.get('latency_ms_per_step'), digits=2)} |")
            lines.append('')
    kinds = sorted(next(iter(engines.values()))['per_kind'])
    if kinds:
        lines += ['## Per scenario kind (all physics): pos err @2s (m) / mean penetration (mm)', '',
                  '| kind | n | ' + ' | '.join(engines) + ' |', '|' + '---|' * (len(engines) + 2)]
        for kind in kinds:
            cells = [f"{_fmt(e['per_kind'][kind]['all_physics']['pos_err_m']['2s'], digits=4)} / {_fmt(e['per_kind'][kind]['all_physics']['penetration']['mean_m'], 1000, 2)}"
                     for e in engines.values()]
            lines.append(f"| {kind} | {next(iter(engines.values()))['per_kind'][kind]['all_physics']['n_episodes']} | " + ' | '.join(cells) + ' |')
        lines.append('')
    for key in ('ship_decision', 'ship_decision_all_physics'):
        decision = result.get(key)
        if not decision or not decision['variants']: continue
        lines += [f"## {key} (subset {decision['subset']}, baseline old_mlp)", '', decision['rule'] + '. ' + decision['statement'], '',
                  '| variant | includes projection | baseline | metric | old | new | delta | 95% CI | pass | CI significant |', '|---|---|---|---|---|---|---|---|---|---|']
        for variant, block in decision['variants'].items():
            rows = [('old_mlp', block, metric, flag) for metric, flag in (('chaotic_pos_err_2s_m', 'improves'), ('mean_penetration_m', 'improves'), ('gentle_pos_err_2s_m', 'not_worse'),
                                                                         ('mean_penetration_capped_50mm_m', 'improves (not gating)'), ('chaotic_active_pos_err_2s_m', 'improves (not gating)'))]
            if block.get('vs_old_projected'):
                rows += [('old_mlp_projected', block['vs_old_projected'], metric, flag) for metric, flag in (('chaotic_pos_err_2s_m', 'improves'), ('gentle_pos_err_2s_m', 'not_worse'),
                                                                                                         ('mean_penetration_capped_50mm_m', 'not_worse'))]
            for baseline, source, metric, flag in rows:
                v = source[metric]; ci = v['ci95']; key = flag.split(' ')[0]
                lines.append(f"| {variant} | {block['includes_non_learned_projection']} | {baseline} | {metric} | {_fmt(v['old'], digits=5)} | {_fmt(v['new'], digits=5)} | "
                             f"{_fmt(v['delta'], digits=5)} | {'-' if not ci else f'[{ci[0]:.5f}, {ci[1]:.5f}]'} | {v.get(key)}{' (not gating)' if 'not gating' in flag else ''} | {v.get('significant')} |")
            lines.append(f"| {variant} | {block['includes_non_learned_projection']} | | **ship** (vs old_mlp only: {block.get('ship_vs_old_mlp')}; beats old_mlp_projected: {block.get('beats_old_projected')}) | | | | | **{block['ship']}** | |")
        lines.append('')
    lines += ['Notes:'] + [f'- {note}' for note in result['notes']] + ['']
    return '\n'.join(lines)


def clean(value):
    """JSON-safe: numpy scalars -> python, NaN/inf -> None."""
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [clean(v) for v in value]
    if isinstance(value, (np.bool_,)): return bool(value)
    if isinstance(value, np.integer): return int(value)
    if isinstance(value, (float, np.floating)): return float(value) if math.isfinite(value) else None
    if isinstance(value, Path): return str(value)
    return value


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', type=Path, default=Path('data/v2')); parser.add_argument('--split', default='test')
    parser.add_argument('--old', type=Path, default=Path('models/mlp_multistep.pt')); parser.add_argument('--old-graph', type=Path)
    parser.add_argument('--new', type=Path); parser.add_argument('--out', type=Path, default=Path('models/contact_eval.json'))
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--max-episodes', type=int); parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--bootstrap', type=int, default=1000); parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--no-old-projected', action='store_true', help='skip the old MLP + projection attribution engine')
    parser.add_argument('--no-latency', action='store_true')
    args = parser.parse_args(argv)
    device = torch.device(args.device); began = time.perf_counter(); notes = []
    data = load_split(args.data, args.split, args.max_episodes)
    count, frames = data['states'].shape[:2]
    print(f'{count} episodes, {frames - 1} steps, {data["states"].shape[2]} slots from {data["files"]} shard(s); device {device}', flush=True)

    try:
        from agatha import contact_gnn
    except ImportError as exc:
        if args.new: raise
        contact_gnn = None
        notes.append(f'agatha.contact_gnn unavailable ({exc}); penetration metrics and projection engines omitted.')
    pen_fn = contact_gnn.penetration_depths if contact_gnn else None
    project = contact_gnn.project if contact_gnn else None

    engines, cpu_engines, checkpoints = {'physics_reference': None}, {}, {}
    def add(name, model, make_step, path):
        engines[name] = make_step(model); checkpoints[name] = str(path)
        if not args.no_latency: cpu_engines[name] = make_step(copy.deepcopy(model).cpu())
    old = load_old(args.old, device); add('old_mlp', old, old_step, args.old)
    if args.old_graph: add('old_graph', load_old(args.old_graph, device), old_step, args.old_graph)
    if project and not args.no_old_projected:
        add('old_mlp_projected', old, lambda m: projected_step(old_step(m), project), args.old)
    new_info = None
    if args.new:
        checkpoint = torch.load(args.new, map_location='cpu', weights_only=True)
        new = contact_gnn.build_from_checkpoint(checkpoint).to(device).eval()
        new_info = dict(config=checkpoint.get('config'), epoch=checkpoint.get('epoch'), train_args=checkpoint.get('train_args'),
                        parameters=sum(p.numel() for p in new.parameters()))
        raw_step = lambda m: (lambda state, props, context, mask: m(state, props, context, mask))
        add('new_raw', new, raw_step, args.new); add('new_projected', new, lambda m: projected_step(raw_step(m), project), args.new)

    stats = evaluate(engines, data, device, args.batch_size, pen_fn, log=lambda text: print(text, flush=True))
    chaotic, live, kinds = data['chaotic'].astype(bool), data['live_physics'].astype(bool), data['kind']
    report = {}
    for name in engines:
        report[name] = dict(description=ENGINE_INFO[name][0], includes_non_learned_projection=ENGINE_INFO[name][1],
                            checkpoint=checkpoints.get(name), latency_ms_per_step=None, **engine_report(stats[name], chaotic, live, kinds))
    if cpu_engines:
        threads = torch.get_num_threads(); torch.set_num_threads(4)   # the live API uses 4 CPU threads
        for name, step_fn in cpu_engines.items(): report[name]['latency_ms_per_step'] = latency_ms(step_fn, data)
        torch.set_num_threads(threads)

    masks = subset_masks(chaotic, live)
    meta_path = Path(args.data) / 'meta.json'
    result = dict(
        schema_version=1, created_unix=time.time(), seconds=None,
        args={k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        data=dict(path=str(args.data), split=args.split, n_episodes=count, steps=frames - 1, step_s=STEP, rollout_seconds=(frames - 1) * STEP,
                  slots=int(data['states'].shape[2]), shards=data['files'],
                  counts={g: {o: int(m.sum()) for o, m in group.items()} for g, group in masks.items()},
                  kinds={k: int((kinds == k).sum()) for k in sorted(set(kinds.tolist()))},
                  meta=json.loads(meta_path.read_text()) if meta_path.exists() else None),
        metric_definitions=dict(
            pos_err_m='mean |p_pred - p_true| over dynamic real objects at the horizon (pooled over objects of the subset)',
            ori_err_deg='geodesic quaternion angle, same pooling', pos_err_rollout_mean_m='same, averaged over every step 1..T',
            fell='object-level confusion vs dataset fell; predicted with the agatha.physics.summarize rule on all 20 Hz frames', tipped='same for tipped',
            penetration=f'max(support, floor, objects) depth from contact_gnn.penetration_depths per dynamic object-frame (t>=1; floor depth ignored where the object centre is beyond the finite 20 m floor): mean, mean with per-frame depth capped at {PEN_CAP_M * 1000:g} mm (robust to fall-through outliers), fraction > {PEN_THRESHOLD_M * 1000:g} mm, max',
            rest_drift='over object-frames where dataset rest is True (t>=1): fd_speed = |p_t - p_{t-1}|/0.05 of the PREDICTION, state_speed = predicted |v|',
            diverged_episodes=f'rollouts with a non-finite or |x|>{DIVERGE_LIMIT:g} state; frozen at last good state and still scored',
            out_of_bounds_episodes=f'rollouts where a dynamic object reaches y < -0.5 m, |x| or |z| > {FLOOR_HALF_M:g} m or |p| > 50 m at any frame (the generator rejects such ground truth); still scored',
            pos_err_active_m=f'pos_err_m restricted to dynamic objects whose ground-truth displacement exceeds {ACTIVE_DISPLACEMENT_M:g} m at some frame or that fell / tipped',
            latency_ms_per_step='batch 1, CPU (4 threads), first episode, mean over 60 autoregressive steps (shared machine: indicative)'),
        new_model=new_info, engines=report,
        ship_decision=ship_decision(stats, chaotic, live, 'live_physics', args.bootstrap, args.seed),
        ship_decision_all_physics=ship_decision(stats, chaotic, live, 'all_physics', args.bootstrap, args.seed),
        notes=notes)
    if not args.new: notes.append('Baseline-only run (no --new): ship_decision has no variants.')
    if masks['live_physics']['all'].sum() == 0: notes.append('No live_physics episodes in this split: the primary ship_decision is undefined (null).')
    reference = report['physics_reference']['subsets']['all_physics']['all']['fell']['accuracy']
    if reference is not None and reference < .999:
        notes.append(f'Fell rule applied to ground-truth 20 Hz frames agrees with the dataset fell flag on only {reference:.4f} of objects '
                     '(generator may evaluate the rule at different frames); this bounds achievable fell accuracy.')
    result['ui_summary'] = summarize_for_ui(result); result['seconds'] = time.perf_counter() - began
    result = clean(result)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2)); args.out.with_suffix('.md').write_text(markdown(result))
    print(markdown(result)); print(f'wrote {args.out} and {args.out.with_suffix(".md")} in {result["seconds"]:.1f}s', flush=True)
    return result


if __name__ == '__main__':
    main()
