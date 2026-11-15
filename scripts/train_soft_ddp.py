"""DDP trainer for agatha.soft_gnn.SoftGNN (TRACK S: basic soft bodies; EXPERIMENTAL research prototype).

Modelled on scripts/train_contact_ddp.py (same DDP / fp16 AMP / EMA / atomic per-epoch checkpoints /
--resume / --max-minutes / GPU-util logging; the sampler, schedule and io helpers are imported from it).

Multi-GPU:
  CUDA_VISIBLE_DEVICES=0,1,2,3 .venv-train/bin/torchrun --nproc_per_node=4 \
      scripts/train_soft_ddp.py --data data/soft_v1 --out models/runs/soft_NAME
Single process (one GPU or CPU):
  CUDA_VISIBLE_DEVICES=7 .venv-train/bin/python scripts/train_soft_ddp.py --data data/soft_v1 --out models/runs/soft_NAME

What is trained
- Only the soft-body vertices are predicted.  Rigid boxes in a soft scene are PRESCRIBED from the
  ground-truth trajectory (teacher-forced obstacles): the model reads the boxes at step t, the
  penetration penalty / projection use them at step t+1, nothing pushes back on the boxes.
- A window is (episode, t0 >= 1): input frames (t0-1, t0), targets t0+1 .. t0+k.  k follows the
  curriculum 1 -> 2 -> 4 (-> --horizon) at 0/5/15/35% of the schedule.
- GNS noise: the newest input frame gets per-vertex Gaussian noise of --noise metres (one random-walk
  step of velocity noise = position noise of the last frame) and both input frames share an extra
  per-vertex offset of --noise-pos metres (accumulated drift of earlier steps).  The acceleration
  target is always computed from the (noisy / drifted) model input to the TRUE next positions.
  Noise is skipped on --noise-clean of the windows, on --noise-rest-clean of the windows whose body is
  fully at rest in the input (no vertex moves faster than --rest-speed), and always on pinned vertices.
- Loss per unroll step j: --w-acc * acceleration (steps j < --acc-steps, default the first only)
  + --w-pos * position (steps j >= 1; also step 0 with --pos-first 1) + --w-strain * edge-strain
  + --w-pen * vertex penetration into support / floor / boxes + --w-rest * rest penalty (unroll steps
  j >= 1 only, also step 0 with --rest-first 1: for vertices whose TRUE speed into the target frame is
  below --rest-speed the predicted step displacement must reproduce the true (about zero) one;
  Smooth-L1 in units of --rest-scale m/s); steps weighted by --step-decay^j.
  (Measured in 4.3-minute smokes: the rest term on the noisy step 0 with weight 0.5 together with
  --w-strain 0.3 --strain-scale 0.02 slowed learning of the dynamics, so those are not the defaults.)  Pinned (kinematic) vertices are prescribed and carry no loss.
- Sampler: --rest-frac of every batch are windows that contain a frame at rest (settling tails); the
  remainder is half uniform, half active as before.
- Round 2 (all off by default): --box-frac F draws a fraction F of every batch from windows in which a soft
  vertex is within --box-gap of a rigid box (measured on the round-1 checkpoint: 10-step rollouts from the
  truth are 2-3x worse in box-contact windows); --late-horizon K (--late-from, --late-batch) unrolls K steps
  in the last part of the schedule (a 6-minute k=8 probe was NOT better than k=4, so it is not in the
  round-2 command); --w-strain 1-3 instead of 0.2 lowered the rollout strain error by 13-21% in 6-minute
  probes without a measurable change of the position error.
- --extra-data DIR --extra-frac F: a fraction F of every batch are windows from a second data set in the
  same shard format (e.g. data/external_v2/meshgraphnets_soft); its val split is rolled out every epoch and
  logged as extra_* (not part of the model-selection score).

Validation (EMA weights, rank 0, every epoch)
- one-step loss terms on fixed noise-free windows (val_*),
- full rollouts from the true frames (0, 1) to the end of the episode (frames 2..T are predicted,
  boxes prescribed), raw model and model + project_soft: mean vertex position error at 0.5 / 1 / 2 s
  (absolute episode time), COM error (mean of the vertices), mean vertex penetration, strain error.
- score (lower is better, selects best.pt) = vert_err_2s + 0.5*vert_err_1s + 10*mean_penetration of the RAW model.

Notes
- --batch is PER GPU; --lr is not scaled by the number of ranks; an epoch is --windows-per-epoch
  windows summed over ranks; with --max-minutes the cosine and the curriculum are compressed into
  the wall-clock budget of THIS invocation (same rule as the contact trainer).
- All shards live in GPU memory (about 0.1 MB per episode); windows are gathered on-device.
- GPU memory is dominated by the unroll activations (batch * k * rounds * directed mesh edges * width); the
  per-epoch log prints rank 0's peak.  --max-gpu-gb caps each process on a shared GPU.
"""
import argparse
import copy
import glob
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F

from agatha import soft_gnn as sg
from train_contact_ddp import (GpuMonitor, WindowSampler, atomic_save, choose_amp, clean, horizon_at, jsonable, log,
                               lr_at)

STEP = sg.STEP
TERMS = ('loss', 'acc', 'pos', 'strain', 'pen', 'rest')
TENSOR_KEYS = ('verts', 'vert_mask', 'rest_edges', 'rest_len', 'soft_props', 'states', 'props', 'context', 'mask', 'pinned')
STATIC_KEYS = ('vert_mask', 'rest_edges', 'rest_len', 'soft_props', 'props', 'context', 'mask', 'pinned')   # per-episode (no time axis)
HORIZON_FRAMES = ((10, '0.5s'), (20, '1s'), (40, '2s'))
START_FRAME = 1                # rollouts are seeded with the true frames (0, 1)
DIVERGE_M = 100.
PEN_CAP_M = .05                # robust penetration mean: per-vertex depth capped at this
RIGID_NOTE = ('rigid boxes are PRESCRIBED from the ground-truth trajectory (teacher-forced obstacles); '
              'only the soft-body vertices are predicted')


# ----------------------------------------------------------------------------- args
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--resume', action='store_true', help='continue from <out>/last.pt if present')
    p.add_argument('--init', type=Path, default=None,
                   help='warm-start model + EMA weights, normalizer and model config from a shipped-format soft_gnn checkpoint '
                        '(fresh optimizer and schedule); ignored when --resume finds <out>/last.pt')
    p.add_argument('--start-horizon', type=int, default=0, help='lower bound on the curriculum horizon k (0 = normal curriculum)')
    p.add_argument('--late-horizon', type=int, default=0, help='round 2: unroll horizon k used from --late-from of the schedule on (0 = off; e.g. 8)')
    p.add_argument('--late-from', type=float, default=.6, help='fraction of the schedule (updates or wall-clock budget) at which --late-horizon starts')
    p.add_argument('--late-batch', type=int, default=0, help='windows per GPU per update while k > --horizon (0 = --batch); memory grows with batch * k')
    p.add_argument('--box-frac', type=float, default=0., help='round 2: fraction of every batch drawn from windows in which a soft vertex is within '
                                                              '--box-gap of a rigid box (box-contact windows have 2-3x the rollout error)')
    p.add_argument('--box-gap', type=float, default=.02, help='m: distance to a box that makes a frame a box-contact frame')
    # model
    p.add_argument('--width', type=int, default=128)
    p.add_argument('--rounds', type=int, default=6)
    p.add_argument('--box-radius', type=float, default=.08)
    p.add_argument('--centre', type=int, default=1, help='1 = virtual centre node per body (global information in two hops)')
    # extra data
    p.add_argument('--extra-data', type=Path, default=None, help='second data set in the soft shard format (train/ and val/), mixed into every batch')
    p.add_argument('--extra-frac', type=float, default=0., help='fraction of every batch drawn from --extra-data')
    p.add_argument('--rest-frac', type=float, default=.15, help='fraction of every batch drawn from windows that contain a frame at rest')
    # schedule
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--windows-per-epoch', type=int, default=200_000)
    p.add_argument('--batch', type=int, default=64, help='windows per GPU per update (memory grows with batch * k * rounds * mesh edges)')
    p.add_argument('--max-minutes', type=float, default=0, help='wall-clock budget; 0 = none')
    p.add_argument('--horizon', type=int, default=4)
    p.add_argument('--bptt', type=int, default=1, help='1 = gradients through the unroll, 0 = pushforward (detach)')
    p.add_argument('--step-decay', type=float, default=.9)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--lr-min', type=float, default=1e-5)
    p.add_argument('--wd', type=float, default=1e-5)
    p.add_argument('--warmup', type=int, default=500)
    p.add_argument('--clip', type=float, default=1.0)
    p.add_argument('--ema', type=float, default=.999)
    # noise
    p.add_argument('--noise', type=float, default=.0007, help='std (m) of the per-vertex noise on the newest input frame (velocity noise of one step)')
    p.add_argument('--noise-pos', type=float, default=.0003, help='std (m) of the per-vertex offset shared by both input frames (drift)')
    p.add_argument('--noise-clean', type=float, default=.1, help='fraction of windows that are seen without noise')
    p.add_argument('--noise-rest-clean', type=float, default=.5, help='fraction of the windows whose body is fully at rest in the input that are seen without noise')
    # loss weights
    p.add_argument('--w-acc', type=float, default=1.0)
    p.add_argument('--w-pos', type=float, default=1.0)
    p.add_argument('--w-strain', type=float, default=.2)
    p.add_argument('--w-pen', type=float, default=1.0)
    p.add_argument('--w-rest', type=float, default=.2)
    p.add_argument('--rest-first', type=int, default=0, help='1 = rest penalty also on unroll step 0 (there: land on the true position despite the input noise)')
    p.add_argument('--rest-speed', type=float, default=.01, help='m/s: a vertex slower than this (true motion) counts as resting')
    p.add_argument('--rest-scale', type=float, default=.005, help='m/s: unit of the rest penalty')
    p.add_argument('--acc-steps', type=int, default=1, help='acceleration loss on the first n unroll steps (0 = all steps)')
    p.add_argument('--pos-first', type=int, default=0, help='1 = position loss also on unroll step 0')
    p.add_argument('--pos-scale', type=float, default=.01, help='scale (m) of the position Smooth-L1 term')
    p.add_argument('--strain-scale', type=float, default=.03)
    # system
    p.add_argument('--amp', choices=('auto', 'bf16', 'fp16', 'off'), default='auto')
    p.add_argument('--compile', type=int, default=0)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--max-train-episodes', type=int, default=0)
    p.add_argument('--max-val-episodes', type=int, default=4096)
    p.add_argument('--val-rollout-episodes', type=int, default=512)
    p.add_argument('--val-windows', type=int, default=8192)
    p.add_argument('--val-batch', type=int, default=128)
    p.add_argument('--log-every', type=int, default=50, help='updates between log lines')
    p.add_argument('--data-device', choices=('auto', 'cuda', 'cpu'), default='auto')
    p.add_argument('--load-threads', type=int, default=8)
    p.add_argument('--dist-backend', choices=('auto', 'nccl', 'gloo'), default='auto')
    p.add_argument('--nccl-p2p', type=int, default=0, help='1 = leave NCCL P2P transport enabled (hangs on this server)')
    p.add_argument('--max-gpu-gb', type=float, default=0, help='hard cap on the CUDA memory of EACH process (0 = none); for shared GPUs')
    p.add_argument('--find-unused', type=int, default=0, help='DDP find_unused_parameters (not needed: every parameter is tied into the loss)')
    return p.parse_args()


# ----------------------------------------------------------------------------- data
def _load_shard(path):
    with np.load(path) as z:
        verts = z['verts'].astype(np.float32, copy=False)
        count = len(verts)
        out = dict(verts=verts, vert_mask=z['vert_mask'].astype(bool), rest_edges=z['rest_edges'].astype(np.int16),
                   rest_len=z['rest_len'].astype(np.float32, copy=False), soft_props=z['soft_props'].astype(np.float32, copy=False),
                   context=z['context'].astype(np.float32, copy=False))
        if 'states' in z.files:
            out.update(states=z['states'].astype(np.float32, copy=False), props=z['props'].astype(np.float32, copy=False), mask=z['mask'].astype(bool))
        else:                                             # scene without a rigid block: zero boxes
            out.update(states=np.zeros((count, verts.shape[1], 0, 13), np.float32), props=np.zeros((count, 0, 9), np.float32),
                       mask=np.zeros((count, 0), bool))
        if out['props'].shape[-1] < sg.RIGID_DIM:               # 10th column: rounding radius of the obstacle (0 = box)
            out['props'] = np.concatenate((out['props'], np.zeros((*out['props'].shape[:-1], sg.RIGID_DIM - out['props'].shape[-1]), np.float32)), -1)
        out['pinned'] = z['pinned'].astype(bool) if 'pinned' in z.files else np.zeros_like(out['vert_mask'])   # kinematic vertices
        out['kind'] = z['kind'].astype(str) if 'kind' in z.files else np.array(['unknown'] * count)
        out['faces'] = z['faces'].astype(np.int16) if 'faces' in z.files else -np.ones((count, 0, 3), np.int16)
        # eval-only extras: analytic / substep velocity of frame 0, at-rest flag of the last frame, mesh family
        out['vel0'] = z['vert_vel'][:, 0].astype(np.float32) if 'vert_vel' in z.files else np.zeros_like(verts[:, 0])
        out['rest_end'] = z['rest'][:, -1].astype(bool) if 'rest' in z.files else _still(verts, out['vert_mask'])
        out['soft_shape'] = z['soft_shape'].astype(str) if 'soft_shape' in z.files else np.array(['unknown'] * count)
    return out


def _still(verts, vert_mask, frames=5, limit=.002):
    """Fallback at-rest flag [E]: no real vertex moved more than `limit` m over the last `frames` frames."""
    move = np.linalg.norm(verts[:, -1] - verts[:, -1 - min(frames, verts.shape[1] - 1)], axis=-1)
    return (move * vert_mask).max(-1) < limit


def _pad_axis(array, size, axis, value):
    if array.shape[axis] >= size:
        return array
    widths = [(0, 0)] * array.ndim
    widths[axis] = (0, size - array.shape[axis])
    return np.pad(array, widths, constant_values=value)


def load_split(root: Path, split: str, limit: int = 0, threads: int = 8, spread: bool = False):
    """All shards of a split as numpy arrays (TENSOR_KEYS + kind, faces, vel0, rest_end, soft_shape + active [E,T+1]).

    limit > 0 keeps the first `limit` episodes (spread=True: evenly spaced instead, keeps the scenario mix).
    """
    files = sorted(glob.glob(str(Path(root) / split / 'shard_*.npz')))
    if not files:
        raise FileNotFoundError(f'no shards in {Path(root) / split}')
    if limit and not spread:
        kept, total = [], 0
        for f in files:
            kept.append(f)
            with np.load(f) as z:
                total += len(z['vert_mask'])
            if total >= limit:
                break
        files = kept
    with ThreadPoolExecutor(max(1, threads)) as pool:
        parts = list(pool.map(_load_shard, files))
    pads = dict(verts=(2, 0.), vel0=(1, 0.), vert_mask=(1, False), pinned=(1, False), rest_edges=(1, -1), rest_len=(1, 0.), faces=(1, -1), states=(2, 0.), props=(1, 0.), mask=(1, False))
    for key, (axis, value) in pads.items():               # shards may disagree on the padded sizes
        size = max(q[key].shape[axis] for q in parts)
        for q in parts:
            q[key] = _pad_axis(q[key], size, axis, value)
    data = {k: np.concatenate([q[k] for q in parts]) for k in parts[0]}
    count = len(data['verts'])
    if limit and limit < count:
        keep = np.unique(np.linspace(0, count - 1, limit).round().astype(int)) if spread else np.arange(limit)
        data = {k: v[keep] for k, v in data.items()}
    # activity index for the sampler: the soft body moves faster than 3 cm/s (mean vertex speed) into frame t
    vm = data['vert_mask'][:, None, :]
    speed = (np.linalg.norm(np.diff(data['verts'], axis=1), axis=-1) * vm).sum(-1) / np.maximum(vm.sum(-1), 1) / STEP    # [E,T]
    data['active'] = np.concatenate((speed[:, :1], speed), 1) > .03
    data['files'] = len(files)
    return data


def pad_like(data, sizes):
    """Pad the vertex / edge / box axes of a load_split dict to `sizes` = (V, M, N) so two data sets can share a batch."""
    v, m, n = sizes
    for key, (axis, size, value) in dict(verts=(2, v, 0.), vel0=(1, v, 0.), vert_mask=(1, v, False), pinned=(1, v, False), rest_edges=(1, m, -1),
                                         rest_len=(1, m, 0.), states=(2, n, 0.), props=(1, n, 0.), mask=(1, n, False)).items():
        data[key] = _pad_axis(data[key], size, axis, value)
    return data


def sizes_of(*datasets):
    return tuple(max(d[key].shape[1] for d in datasets) for key in ('vert_mask', 'rest_len', 'mask'))


def to_device(data, device, keys=TENSOR_KEYS + ('active',)):
    return {k: torch.from_numpy(np.ascontiguousarray(data[k])).to(device) for k in keys if k in data}


def gather(data, e, t0, k, device):
    """Window dict for episodes e: frames t0-1 .. t0+k of verts / rigid states ([B,k+2,...]) plus the static tensors."""
    t = t0[:, None] - 1 + torch.arange(k + 2, device=t0.device)
    out = dict(verts=data['verts'][e[:, None], t], states=data['states'][e[:, None], t])
    out.update({key: data[key][e] for key in STATIC_KEYS})
    return {key: value.to(device, non_blocking=True) for key, value in out.items()}


def episodes(data, sl, device):
    """Whole-episode batch dict (all frames) for the slice / index `sl`."""
    return {key: data[key][sl].to(device) for key in TENSOR_KEYS}


@torch.no_grad()
def box_contact_frames(data, gap=.02, chunk=64):
    """bool [E,T+1]: some real soft vertex is within `gap` metres of a rigid obstacle (true trajectory)."""
    verts, states = data['verts'], data['states']
    E, T1 = verts.shape[:2]
    out = torch.zeros(E, T1, dtype=torch.bool, device=verts.device)
    if states.shape[2] == 0:
        return out
    for e in data['mask'].any(1).nonzero().squeeze(1).split(chunk):
        n = len(e)
        v = verts[e].float().flatten(0, 1)
        state = states[e].float().flatten(0, 1)
        rep = lambda x: x[e][:, None].expand(-1, T1, *x.shape[1:]).flatten(0, 1)
        props, mask, context, vm = rep(data['props']), rep(data['mask']), rep(data['context']), rep(data['vert_mask'])
        state, props, mask = sg._rigid(len(v), v.device, state, props, mask)
        sdf = sg._box_point_sdf(v, state, props)[0]
        ok = sg._obstacles(state, props, mask, sg._context(context))[:, None, :] & vm[:, :, None]
        out[e] = ((sdf < gap) & ok).flatten(1).any(1).reshape(n, T1)
    return out


def soft_horizon(step, total, horizon, progress, args):
    """Curriculum of the contact trainer, plus --late-horizon from --late-from of the schedule on."""
    k = horizon_at(step, total, horizon, progress, args.start_horizon)
    if args.late_horizon > 0 and max(step / max(total, 1), progress) >= args.late_from:
        k = max(k, args.late_horizon)
    return k


def draw_active(sampler, n):
    """n windows from a WindowSampler's event pool only (its `active` list), with its own cursor."""
    flat = sampler.active[(sampler.ca + torch.arange(n, device=sampler.frames.device)) % len(sampler.active)]
    sampler.ca += n
    return flat // sampler.per, flat % sampler.per


# ----------------------------------------------------------------------------- model calls
def rigid_at(batch, frame):
    """(rigid_state, rigid_props, rigid_mask) with the PRESCRIBED boxes at a window / episode frame index."""
    states = batch['states']
    return states[:, min(frame, states.shape[1] - 1)], batch['props'], batch['mask']


def predict_raw(model, cur, prev, batch, frame, amp_dtype=None):
    use_amp = amp_dtype is not None and cur.is_cuda
    with torch.autocast('cuda', dtype=amp_dtype or torch.float16, enabled=use_amp):
        raw = model.predict_raw(cur, prev, batch['vert_mask'], batch['rest_edges'], batch['rest_len'], batch['soft_props'],
                                batch['context'], *rigid_at(batch, frame), pinned=batch['pinned'])
    return raw.float()


def model_step(model, cur, prev, batch, frame, amp_dtype=None):
    """Next vertex positions from the model alone (boxes read at `frame`, the frame of `cur`)."""
    raw = predict_raw(model, cur, prev, batch, frame, amp_dtype)
    verts = batch['verts']
    return model.integrate(cur, prev, raw * model.out_std.float() + model.out_mean.float(), batch['context'], batch['vert_mask'], clamp=True,
                           pinned=batch['pinned'], pinned_next=verts[:, min(frame + 1, verts.shape[1] - 1)])   # pinned vertices are prescribed


def project_step(cur, nxt, batch, frame, **project_args):
    """agatha.soft_gnn.project_soft with the boxes at frame+1 (the frame of `nxt`)."""
    return sg.project_soft(cur, nxt, batch['vert_mask'], batch['context'], *rigid_at(batch, frame + 1),
                           edges=batch['rest_edges'], rest_len=batch['rest_len'], pinned=batch['pinned'], **project_args)


# ----------------------------------------------------------------------------- losses
def add_noise(prev, cur, vert_mask, args, gen):
    """vert_mask: the vertices that may get noise (real and not pinned)."""
    shape, device = cur.shape, cur.device
    on = (torch.rand(shape[0], 1, 1, device=device, generator=gen) >= args.noise_clean) & vert_mask[..., None]
    resting = (((cur - prev).norm(dim=-1) * vert_mask).amax(1) < args.rest_speed * STEP)[:, None, None]
    on = on & ~(resting & (torch.rand(shape[0], 1, 1, device=device, generator=gen) < args.noise_rest_clean))
    drift = torch.randn(shape, device=device, generator=gen) * args.noise_pos
    step = torch.randn(shape, device=device, generator=gen) * args.noise
    return torch.where(on, prev + drift, prev), torch.where(on, cur + drift + step, cur)


def unroll(model, batch, k, args, amp_dtype, gen=None):
    """k-step autoregressive loss on a window dict. Returns (loss, stats[len(TERMS)], bad flag tensor)."""
    verts, vm, context = batch['verts'], batch['vert_mask'], batch['context']
    edges, rest_len = batch['rest_edges'], batch['rest_len']
    pinned = batch['pinned']
    free = vm & ~pinned
    count = free.sum().clamp_min(1)
    prev, cur = verts[:, 0], verts[:, 1]
    if gen is not None and (args.noise > 0 or args.noise_pos > 0):
        prev, cur = add_noise(prev, cur, free, args, gen)
    mean, std = model.out_mean.float(), model.out_std.float()
    weights = [args.step_decay ** j for j in range(k)]
    total = verts.new_zeros(())
    stats = verts.new_zeros(len(TERMS))
    bad = torch.zeros((), dtype=torch.bool, device=verts.device)
    for j in range(k):
        true = verts[:, j + 2]
        raw = predict_raw(model, cur, prev, batch, j + 1, amp_dtype)
        with torch.no_grad():
            # pushforward-style target: from the model's CURRENT (noisy / drifted) input to the TRUE next positions
            target = model.normalize(model.raw_targets(cur.detach(), prev.detach(), true, context, vm))
        nxt = model.integrate(cur, prev, raw * std + mean, context, vm, pinned=pinned, pinned_next=true)
        acc = (F.smooth_l1_loss(raw, target, reduction='none').sum(-1) * free).sum() / count
        true_cur = verts[:, j + 1]
        terms = dict(acc=acc,
                     pos=sg.position_loss(nxt, true, free, scale=args.pos_scale),
                     strain=sg.strain_loss(nxt, true, vm, edges, rest_len, scale=args.strain_scale),
                     pen=sg.penetration_loss(nxt, true, free, context, *rigid_at(batch, j + 2)),
                     # step 0 starts from a noisy input: land on the true resting position; later steps: reproduce the (zero) true step
                     rest=sg.rest_loss(nxt, true_cur if j == 0 else cur.detach(), true, true_cur, free, args.rest_speed, args.rest_scale))
        w_acc = args.w_acc if not args.acc_steps or j < args.acc_steps else 0.
        w_pos = args.w_pos if j >= 1 or args.pos_first else 0.
        w_rest = args.w_rest if j >= 1 or args.rest_first else 0.
        step = w_acc * terms['acc'] + w_pos * terms['pos'] + args.w_strain * terms['strain'] + args.w_pen * terms['pen'] + w_rest * terms['rest']
        total = total + weights[j] * step
        stats = stats + weights[j] * torch.stack([step.detach()] + [terms[n].detach().float() for n in TERMS[1:]])
        bad = bad | ~torch.isfinite(nxt).all() | (nxt.abs() > DIVERGE_M).any()
        prev, cur = (cur, nxt) if args.bptt else (cur.detach(), nxt.detach())
    norm = sum(weights)
    return total / norm, stats / norm, bad | ~torch.isfinite(total)


class Trainee(nn.Module):
    """Whole unroll inside forward() so DistributedDataParallel sees one forward per update."""

    def __init__(self, model, args, amp_dtype):
        super().__init__()
        self.model, self.args, self.amp_dtype = model, args, amp_dtype

    def forward(self, batch, k, gen):
        loss, stats, bad = unroll(self.model, batch, k, self.args, self.amp_dtype, gen)
        # A batch without rigid boxes near the soft body never calls the box / contact encoders; tie every
        # parameter into the graph (zero gradient) so DDP's reduction always sees all of them.
        return loss + sum(p.sum() for p in self.model.parameters()) * 0., stats, bad


# ----------------------------------------------------------------------------- rollouts and their metrics
def edge_tensors(batch):
    """Dense valid-edge mask [B,M] and clamped endpoint indices [B,M] x2 (padded edges point at vertex 0)."""
    edges = batch['rest_edges'].long()
    count = batch['vert_mask'].shape[1]
    i, j = edges[..., 0].clamp(0, count - 1), edges[..., 1].clamp(0, count - 1)
    ok = (edges >= 0).all(-1) & (edges < count).all(-1) & (i != j) & (batch['rest_len'] > 1e-6)
    ok = ok & batch['vert_mask'].gather(1, i) & batch['vert_mask'].gather(1, j)
    return ok, i, j


def strains(verts, batch, edges=None):
    """Edge strain len/rest-1 [B,M] of one frame [B,V,3] (0 for padded edges) and the valid mask."""
    ok, i, j = edges or edge_tensors(batch)
    index = lambda x: x[..., None].expand(-1, -1, 3)
    length = (verts.gather(1, index(i)) - verts.gather(1, index(j))).norm(dim=-1)
    return (length / batch['rest_len'].clamp_min(1e-6) - 1) * ok, ok


def penetration(verts, batch, frame):
    """Deepest penetration [B,V] (m) of each vertex into the support slab, the floor or a box at `frame`."""
    depth = sg.vertex_penetration(verts, batch['vert_mask'], batch['context'], *rigid_at(batch, frame))
    return torch.maximum(torch.maximum(depth['support'], depth['floor']), depth['objects'])


@torch.no_grad()
def rollout(step_fn, batch, start=START_FRAME, prev0=None):
    """Autoregressive trajectory [B,T+1,V,3]: frames <= start are the truth, step_fn(cur, prev, batch, frame) -> frame+1.

    prev0 overrides the frame before `start` (start=0 needs it).  Returns (trajectory, diverged [B]).
    """
    verts = batch['verts']
    frames = [verts[:, t].float() for t in range(start + 1)]
    prev = prev0.float() if prev0 is not None else frames[-2]
    cur = frames[-1]
    diverged = torch.zeros(verts.shape[0], dtype=torch.bool, device=verts.device)
    for t in range(start, verts.shape[1] - 1):
        nxt = step_fn(cur, prev, batch, t).float()
        diverged |= (~torch.isfinite(nxt) | (nxt.abs() > DIVERGE_M)).flatten(1).any(1)
        nxt = torch.nan_to_num(nxt, nan=1e3, posinf=1e3, neginf=-1e3).clamp(-1e3, 1e3)
        frames.append(nxt)
        prev, cur = cur, nxt
    return torch.stack(frames, 1), diverged


@torch.no_grad()
def trajectory_stats(traj, batch, start=START_FRAME, pen_cap=PEN_CAP_M):
    """Per-episode metrics of a predicted trajectory [B,T+1,V,3] against batch['verts'].

    vert_err / com_err / shape_err are [B,T+1] (mean over real vertices; COM = unweighted vertex mean;
    shape error = vertex error after removing each trajectory's own COM).  pen_* and strain_err are
    means over the PREDICTED frames (start+1 .. T) only.  pen_capped caps a vertex depth at pen_cap.
    """
    true = batch['verts'].float()
    vm = batch['vert_mask']
    w = vm[:, None, :, None].float()
    n = vm.sum(1).clamp_min(1).float()
    diff = (traj - true) * w
    com_pred, com_true = (traj * w).sum(2) / n[:, None, None], (true * w).sum(2) / n[:, None, None]
    out = dict(vert_err=diff.norm(dim=-1).sum(-1) / n[:, None], com_err=(com_pred - com_true).norm(dim=-1),
               shape_err=((diff - (com_pred - com_true)[:, :, None]) * w).norm(dim=-1).sum(-1) / n[:, None])
    edges = edge_tensors(batch)
    edge_count = edges[0].sum(1).clamp_min(1).float()
    frames = range(start + 1, traj.shape[1])
    zero = torch.zeros(len(traj), device=traj.device)
    pen, capped, deep, worst, strain_err, strain_abs = (zero.clone() for _ in range(6))
    for t in frames:
        depth = torch.nan_to_num(penetration(traj[:, t], batch, t), nan=1.)
        pen += depth.sum(1) / n
        capped += depth.clamp_max(pen_cap).sum(1) / n
        deep += (depth > .005).float().sum(1) / n
        worst = torch.maximum(worst, depth.max(1).values)
        predicted, _ = strains(traj[:, t], batch, edges)
        reference, _ = strains(true[:, t], batch, edges)
        strain_err += (predicted - reference).abs().sum(1) / edge_count
        strain_abs += predicted.abs().sum(1) / edge_count
    count = max(len(frames), 1)
    out.update(pen_mean=pen / count, pen_capped=capped / count, pen_deep_frac=deep / count, pen_max=worst,
               strain_err=strain_err / count, strain_abs=strain_abs / count)
    return out


@torch.no_grad()
def fit_normalizer(model, data, seed, cap=4096):
    """Acceleration / speed / soft_props statistics from (a random subset of) the training episodes."""
    count = data['verts'].shape[0]
    pick = torch.randperm(count, generator=torch.Generator().manual_seed(seed + 7919))[:cap].sort().values.to(data['verts'].device)
    stats = model.fit_normalizer(data['verts'][pick], data['vert_mask'][pick], data['context'][pick], data['soft_props'][pick])
    return stats, len(pick)


def _nanmean(x):
    x = x[torch.isfinite(x)]
    return float(x.mean()) if len(x) else float('nan')


@torch.no_grad()
def validate(model, val, device, args, amp_dtype):
    model.eval()
    E, T = val['verts'].shape[0], val['verts'].shape[1] - 1
    # (i) one-step loss on a fixed, noise-free set of windows
    g = torch.Generator().manual_seed(args.seed + 104729)
    per = T - 1                                                 # t0 in [1, T-1]
    n = min(args.val_windows, E * per)
    flat = torch.randperm(E * per, generator=g)[:n].to(val['verts'].device)
    acc, done = torch.zeros(len(TERMS), device=device), 0
    for i in range(0, n, args.val_batch * 4):
        f = flat[i:i + args.val_batch * 4]
        _, stats, _ = unroll(model, gather(val, f // per, f % per + 1, 1, device), 1, args, amp_dtype, None)
        acc += torch.nan_to_num(stats, nan=1e6, posinf=1e6, neginf=1e6) * len(f)
        done += len(f)
    out = {f'val_{name}': float(v) for name, v in zip(TERMS, acc / max(done, 1))}
    # (ii) full rollouts from the true frames (0, 1); boxes prescribed
    m = min(args.val_rollout_episodes, E)
    engines = dict(raw=lambda cur, prev, batch, t: model_step(model, cur, prev, batch, t, amp_dtype),
                   proj=lambda cur, prev, batch, t: project_step(cur, model_step(model, cur, prev, batch, t, amp_dtype), batch, t))
    for name, step_fn in engines.items():
        parts = []
        for i in range(0, m, args.val_batch):
            batch = episodes(val, slice(i, min(i + args.val_batch, m)), device)
            traj, diverged = rollout(step_fn, batch)
            stats = trajectory_stats(traj, batch)
            stats['diverged'] = diverged.float()
            parts.append(stats)
        stats = {key: torch.cat([q[key] for q in parts]) for key in parts[0]}
        tag = '' if name == 'raw' else 'proj_'
        for frame, label in HORIZON_FRAMES:
            if frame <= T:
                out[f'{tag}vert_err_{label}'] = _nanmean(stats['vert_err'][:, frame])
                out[f'{tag}com_err_{label}'] = _nanmean(stats['com_err'][:, frame])
        out[f'{tag}vert_err_final'] = _nanmean(stats['vert_err'][:, -1])
        out[f'{tag}mean_penetration'] = _nanmean(stats['pen_capped'])
        out[f'{tag}strain_err'] = _nanmean(stats['strain_err'])
        out[f'{tag}diverged'] = int(stats['diverged'].sum())
    out['rollout_episodes'] = m

    def z(key):
        v = out.get(key, float('nan'))
        return 1. if v != v else v                              # a missing / NaN horizon is penalised, not rewarded
    late = 'vert_err_2s' if 'vert_err_2s' in out else 'vert_err_final'
    out['score'] = z(late) + .5 * z('vert_err_1s' if 'vert_err_1s' in out else late) + 10 * z('mean_penetration')
    return out


def concat_windows(a, b):
    return {key: torch.cat((a[key], b[key])) for key in a}


# ----------------------------------------------------------------------------- io
def shipped(model, config, epoch, args, metrics):
    return {'kind': sg.SoftGNN.kind, 'config': dict(config),
            'state_dict': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            'model_step_s': STEP, 'epoch': int(epoch), 'train_args': jsonable(vars(args)), 'metrics': clean(metrics),
            'experimental': True, 'note': RIGID_NOTE}


# ----------------------------------------------------------------------------- main
def main():
    args = parse_args()
    distributed = 'LOCAL_RANK' in os.environ and int(os.environ.get('WORLD_SIZE', '1')) >= 1 and 'RANK' in os.environ
    rank = int(os.environ.get('RANK', 0)) if distributed else 0
    world = int(os.environ.get('WORLD_SIZE', 1)) if distributed else 1
    local = int(os.environ.get('LOCAL_RANK', 0)) if distributed else 0
    cuda = torch.cuda.is_available()
    gpu = local % torch.cuda.device_count() if cuda else 0
    device = torch.device(f'cuda:{gpu}' if cuda else 'cpu')
    torch.set_num_threads(max(1, min(4, (os.cpu_count() or 4) // max(world, 1))))
    if cuda:
        torch.cuda.set_device(device)
        if args.max_gpu_gb > 0:
            torch.cuda.set_per_process_memory_fraction(min(1., args.max_gpu_gb * 2**30 / torch.cuda.get_device_properties(device).total_memory), device)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    backend = 'single'
    if distributed:
        backend = args.dist_backend
        if backend == 'auto':   # NCCL cannot place two ranks on one GPU
            backend = 'nccl' if cuda and world <= torch.cuda.device_count() else 'gloo'
        if not args.nccl_p2p:
            # On this server NCCL's default P2P/CUMEM transport hangs in the first collective; shared memory works.
            os.environ.setdefault('NCCL_P2P_DISABLE', '1')
        dist.init_process_group(backend)
    main_rank = rank == 0
    torch.manual_seed(args.seed * 1000 + rank)
    np.random.seed(args.seed * 1000 + rank)
    start_wall = time.time()
    if main_rank:
        args.out.mkdir(parents=True, exist_ok=True)
        log(f'[setup] world={world} backend={backend} gpus={torch.cuda.device_count() if cuda else 0} device={device} torch={torch.__version__} out={args.out}')
        log(f'[scope] EXPERIMENTAL soft-body track: {RIGID_NOTE}')

    amp_dtype, use_scaler, amp_note = choose_amp(args.amp, device)
    if main_rank:
        log(f'[amp] {amp_note}')

    # ---- data
    t = time.time()
    train_np = load_split(args.data, 'train', args.max_train_episodes, args.load_threads)
    use_extra = args.extra_data is not None and args.extra_frac > 0
    extra = extra_sampler = extra_val = None
    if use_extra:
        extra_np = load_split(args.extra_data, 'train', 0, args.load_threads)
        sizes = sizes_of(train_np, extra_np)
        pad_like(train_np, sizes), pad_like(extra_np, sizes)
        extra_kinds = extra_np['kind']
    nbytes = sum(train_np[k].nbytes for k in TENSOR_KEYS)
    data_device = device if args.data_device in ('auto', 'cuda') and cuda else torch.device('cpu')
    if args.data_device == 'auto' and cuda and nbytes > .5 * torch.cuda.get_device_properties(device).total_memory:
        data_device = torch.device('cpu')
    train = to_device(train_np, data_device)
    kinds = train_np['kind']
    del train_np
    if use_extra:
        extra = to_device(extra_np, data_device)
        del extra_np
        if extra['verts'].shape[1] < 4:
            raise ValueError('--extra-data episodes need at least 4 frames')
        if main_rank:
            names, counts = np.unique(extra_kinds, return_counts=True)
            log(f'[data] extra {args.extra_data}: episodes={extra["verts"].shape[0]} T={extra["verts"].shape[1] - 1} batch fraction {args.extra_frac}; '
                f'pinned vertices/episode {float(extra["pinned"].float().sum(1).mean()):.1f}; kinds ' + ' '.join(f'{a}:{b}' for a, b in zip(names, counts)))
    E, T, V = train['verts'].shape[0], train['verts'].shape[1] - 1, train['verts'].shape[2]
    if T < 2:
        raise ValueError('episodes need at least 3 frames')
    horizon = max(1, min(args.horizon, T - 1))
    if main_rank:
        names, counts = np.unique(kinds, return_counts=True)
        log(f'[data] train episodes={E} T={T} V={V} M={train["rest_edges"].shape[1]} boxes={train["states"].shape[2]} {nbytes / 2**30:.2f} GiB on {data_device} '
            f'loaded in {time.time() - t:.1f}s; active frames {float(train["active"].float().mean()):.3f}; '
            f'mean vertices {float(train["vert_mask"].float().sum(1).mean()):.1f}; episodes with a rigid box {float(train["mask"].any(1).float().mean()) if train["mask"].shape[1] else 0.:.3f}')
        log('[data] kinds ' + ' '.join(f'{a}:{b}' for a, b in zip(names, counts)))
        val = to_device(load_split(args.data, 'val', args.max_val_episodes, args.load_threads, spread=True), device)
        log(f'[data] val episodes={val["verts"].shape[0]}')
        if use_extra and (Path(args.extra_data) / 'val').is_dir() and glob.glob(str(Path(args.extra_data) / 'val' / 'shard_*.npz')):
            extra_val = to_device(load_split(args.extra_data, 'val', args.max_val_episodes, args.load_threads, spread=True), device)
            log(f'[data] extra val episodes={extra_val["verts"].shape[0]}')
    # the sampler sees frames 1..T, so its t0' = t0-1 and every window has the previous frame t0-1 >= 0
    active = train.pop('active')[:, 1:]
    sampler = WindowSampler(active, args.seed, rank, world, event_frames=~active if args.rest_frac > 0 else None, event_frac=args.rest_frac)
    box_sampler = None
    if args.box_frac > 0:
        near = box_contact_frames(train, args.box_gap)
        if bool(near.any()):
            box_sampler = WindowSampler(near[:, 1:], args.seed + 53, rank, world)
        if main_rank:
            log(f'[data] box-contact frames (a vertex within {args.box_gap} m of a box): {float(near.float().mean()):.3f} of all frames, '
                f'{float(near.any(1).float().mean()):.3f} of the episodes; batch fraction {args.box_frac if box_sampler else 0}')
        del near
    args.late_horizon = max(0, min(args.late_horizon, T - 1, extra['verts'].shape[1] - 2 if use_extra else T - 1))
    if use_extra:
        horizon = max(1, min(horizon, extra['verts'].shape[1] - 2))
        extra_sampler = WindowSampler(extra.pop('active')[:, 1:], args.seed + 31, rank, world)
    def mix(batch_size):                                   # (extra-data windows, box-contact windows) of one batch
        n_extra = min(batch_size - 1, max(1, int(round(batch_size * args.extra_frac)))) if use_extra else 0
        n_box = min(batch_size - n_extra - 1, int(round(batch_size * args.box_frac))) if box_sampler is not None else 0
        return n_extra, max(n_box, 0)

    # ---- model
    last_path = args.out / 'last.pt'
    init = None
    if args.init is not None and not (args.resume and last_path.exists()):
        init = torch.load(args.init, map_location='cpu', weights_only=True)
        if init.get('kind') != sg.SoftGNN.kind:
            raise ValueError(f'--init is not a soft_gnn checkpoint: {init.get("kind")}')
        for key, value in (init.get('config') or {}).items():      # architecture follows the checkpoint
            if key in ('width', 'rounds', 'box_radius'):
                setattr(args, key, value)
        args.centre = int(bool((init.get('config') or {}).get('centre', any(k.startswith('centre_encoder.') for k in init['state_dict']))))
    model = sg.SoftGNN(width=args.width, rounds=args.rounds, box_radius=args.box_radius, centre=bool(args.centre)).to(device)
    config = dict(model.config)
    updates_per_epoch = max(1, args.windows_per_epoch // (args.batch * world))
    total_updates = updates_per_epoch * args.epochs
    requested_warmup = args.warmup
    args.warmup = max(1, min(args.warmup, int(.05 * total_updates)))
    resume = None
    if args.resume and last_path.exists():
        resume = torch.load(last_path, map_location='cpu', weights_only=False)
        model.load_state_dict(resume['model'])
    elif init is not None:
        sg.load_weights(model, init['state_dict'])   # includes the normalizer buffers; zero-pads input features added since
        if main_rank:
            log(f'[init] warm start from {args.init} (epoch {init.get("epoch")}, config {init.get("config")}); normalizer from the checkpoint; '
                f'fresh optimizer/schedule; start horizon {max(args.start_horizon, 1) if args.start_horizon else "curriculum"}')
        del init
    elif main_rank:
        stats, n = fit_normalizer(model, train, args.seed)
        log(f'[normalizer] {n} episodes: acc mean={[round(float(x), 3) for x in stats["mean"]]} std={[round(float(x), 3) for x in stats["std"]]} m/s^2 '
            f'(gravity removed); vel_scale={float(stats["vel_scale"]):.3f} m/s; --noise {args.noise} m is '
            f'{2 * args.noise / STEP ** 2 / float(stats["std"].mean()):.2f} sigma of target acceleration noise')
    ema = copy.deepcopy(model).eval().requires_grad_(False) if main_rank else None
    if resume is not None and main_rank:
        ema.load_state_dict(resume['ema'])
    trainee = Trainee(model, args, amp_dtype)
    if args.compile:
        trainee = torch.compile(trainee)
    if distributed:
        # rank 0's parameters and buffers (the normalizers) go to every rank explicitly
        with torch.no_grad():
            for tensor in model.state_dict().values():
                dist.broadcast(tensor, 0)
        net = nn.parallel.DistributedDataParallel(trainee, device_ids=[gpu] if cuda else None, broadcast_buffers=False,
                                                  find_unused_parameters=bool(args.find_unused))
        check = torch.cat([b.float().flatten() for b in model.buffers()] + [torch.stack([p.float().sum() for p in model.parameters()]).sum()[None]])
        low, high = check.clone(), check.clone()
        dist.all_reduce(low, dist.ReduceOp.MIN)
        dist.all_reduce(high, dist.ReduceOp.MAX)
        assert torch.equal(low, high), 'ranks disagree on initial weights/normalizer'
    else:
        net = trainee
    if main_rank and ema is not None and resume is None:
        ema.load_state_dict(model.state_dict())

    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (no_decay if p.ndim < 2 else decay).append(p)
    optimizer = torch.optim.AdamW([dict(params=decay, weight_decay=args.wd), dict(params=no_decay, weight_decay=0.)],
                                  lr=args.lr, fused=cuda)
    scaler = torch.amp.GradScaler('cuda', enabled=use_scaler)
    start_epoch = global_step = 0
    best = float('inf')
    dropped_total = skipped_total = 0
    if resume is not None:
        optimizer.load_state_dict(resume['optimizer'])
        scaler.load_state_dict(resume['scaler'])
        start_epoch, global_step, best = resume['epoch'], resume['global_step'], resume.get('best', best)
        dropped_total, skipped_total = resume.get('dropped', 0), resume.get('skipped', 0)
        if main_rank:
            log(f'[resume] {last_path} epoch={start_epoch} step={global_step} best={best:.5f}')
    del resume
    params = sum(p.numel() for p in model.parameters())
    monitor = log_file = None
    if main_rank:
        log(f'[model] SoftGNN {config} params={params:,}; updates/epoch={updates_per_epoch} total={total_updates} '
            f'batch={args.batch}x{world} lr={args.lr} (not scaled by ranks) bptt={args.bptt} horizon={horizon}')
        log(f'[schedule] warmup {args.warmup} updates ({100 * args.warmup / max(total_updates, 1):.1f}% of total'
            f'{f"; capped from {requested_warmup}" if args.warmup != requested_warmup else ""}); cosine to lr_min={args.lr_min}; '
            f'horizon k=1/2/4/{horizon} (capped at {horizon}) from 0/5/15/35% of the schedule'
            + (f', k={args.late_horizon} from {100 * args.late_from:.0f}% on (batch {args.late_batch or args.batch}/GPU there)' if args.late_horizon > horizon else '')
            + (f'; schedule is also driven by the {args.max_minutes:g} min wall-clock budget' if args.max_minutes > 0 else ''))
        if total_updates < 2000:
            log(f'[schedule] WARNING only {total_updates} optimizer updates in total: raise --epochs/--windows-per-epoch or lower --batch')
        monitor = GpuMonitor(args.out / 'gpu_util.jsonl') if cuda else None
        if monitor:
            monitor.start()
        log_file = open(args.out / 'train_log.jsonl', 'a')
    gen = torch.Generator(device=device)
    budget = args.max_minutes * 60 if args.max_minutes > 0 else None
    stop = False
    width = len(TERMS) + 4                                 # + dropped, skipped, stop, wall-clock progress
    progress = 0.

    for epoch in range(start_epoch, args.epochs):
        net.train()
        gen.manual_seed(args.seed * 1_000_003 + epoch * 7919 + rank * 104_729 + 17)
        accum = torch.zeros(width, device=device)
        interval_windows = 0
        epoch_stats = torch.zeros(len(TERMS), device=device)
        epoch_logs = 0
        epoch_start = interval_start = time.time()
        interval_updates = 0
        k = soft_horizon(global_step, total_updates, horizon, progress, args)
        lr = args.lr
        for update in range(updates_per_epoch):
            k = soft_horizon(global_step, total_updates, horizon, progress, args)
            sampler.set(epoch, k)
            lr = lr_at(global_step, total_updates, args, progress)
            for group in optimizer.param_groups:
                group['lr'] = lr
            batch_size = args.late_batch if args.late_batch > 0 and k > horizon else args.batch
            n_extra, n_box = mix(batch_size)
            interval_windows += batch_size
            e, t0 = sampler.next(batch_size - n_extra - n_box)
            if n_box:
                box_sampler.set(epoch, k)
                eb, tb = draw_active(box_sampler, n_box)
                e, t0 = torch.cat((e, eb)), torch.cat((t0, tb))
            batch = gather(train, e, t0 + 1, k, device)
            if n_extra:
                extra_sampler.set(epoch, k)
                e, t0 = extra_sampler.next(n_extra)
                batch = concat_windows(batch, gather(extra, e, t0 + 1, k, device))
            loss, stats, bad = net(batch, k, gen)
            optimizer.zero_grad(set_to_none=True)
            if bool(bad):
                # keep the DDP reduction in lock-step: this rank contributes zero gradients
                loss = sum(p.sum() for p in model.parameters()) * 0.
                accum[len(TERMS)] += 1
            else:
                accum[:len(TERMS)] += stats
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            norm = nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            if bool(torch.isfinite(norm)):
                scaler.step(optimizer)
                if ema is not None:
                    d = min(args.ema, (1 + global_step) / (10 + global_step))
                    for pe, pm in zip(ema.parameters(), model.parameters()):
                        pe.lerp_(pm.detach(), 1 - d)
            else:
                optimizer.zero_grad(set_to_none=True)
                accum[len(TERMS) + 1] += 1
            scaler.update()
            global_step += 1
            interval_updates += 1

            if interval_updates == args.log_every or update == updates_per_epoch - 1:
                if main_rank and budget is not None:
                    accum[len(TERMS) + 3] = (time.time() - start_wall) / budget
                    if time.time() - start_wall > budget:
                        accum[len(TERMS) + 2] = 1
                if distributed:
                    dist.all_reduce(accum)
                if cuda:
                    torch.cuda.synchronize()
                values = accum.tolist()
                good = max(interval_updates * world - values[len(TERMS)], 1)
                stop = values[len(TERMS) + 2] > 0
                progress = min(1., values[len(TERMS) + 3])
                if main_rank:
                    elapsed = time.time() - interval_start
                    wps = interval_windows * world / max(elapsed, 1e-9)
                    dropped_total += int(values[len(TERMS)])
                    skipped_total += int(values[len(TERMS) + 1] / world)
                    row = dict(step=global_step, epoch=epoch + 1, k=k, lr=lr, windows_per_sec=wps, windows_per_sec_per_gpu=wps / world,
                               dropped_batches=dropped_total, skipped_updates=skipped_total, minutes=(time.time() - start_wall) / 60,
                               **{name: values[i] / good for i, name in enumerate(TERMS)})
                    log_file.write(json.dumps(clean(row)) + '\n')
                    log_file.flush()
                    epoch_stats += accum[:len(TERMS)] / good
                    epoch_logs += 1
                    if epoch_logs % 5 == 1:
                        log(f'  e{epoch + 1} u{update + 1}/{updates_per_epoch} k={k} loss={row["loss"]:.4f} acc={row["acc"]:.4f} '
                            f'pos={row["pos"]:.4f} strain={row["strain"]:.4f} pen={row["pen"]:.4f} rest={row["rest"]:.4f} lr={lr:.2e} '
                            f'{wps:,.0f} win/s ({wps / world:,.0f}/gpu) dropped={dropped_total}')
                accum.zero_()
                interval_updates = interval_windows = 0
                interval_start = time.time()
                if stop:
                    break

        if main_rank:
            train_time = time.time() - epoch_start
            for be, bm in zip(ema.buffers(), model.buffers()):   # buffers (normalizers) follow the live model
                be.copy_(bm)
            t = time.time()
            metrics = validate(ema, val, device, args, amp_dtype)
            if extra_val is not None:
                metrics.update({f'extra_{key}': value for key, value in validate(ema, extra_val, device, args, amp_dtype).items()})
            metrics.update(epoch=epoch + 1, step=global_step, k=k, train_seconds=train_time, val_seconds=time.time() - t,
                           **{f'train_{name}': float(v) / max(epoch_logs, 1) for name, v in zip(TERMS, epoch_stats)})
            improved = metrics['score'] < best
            best = min(best, metrics['score'])
            ship = shipped(ema, config, epoch + 1, args, metrics)
            atomic_save(ship, args.out / f'epoch_{epoch + 1:03d}.pt')
            if improved:
                atomic_save(ship, args.out / 'best.pt')
            atomic_save(dict(model=model.state_dict(), ema=ema.state_dict(), optimizer=optimizer.state_dict(),
                             scheduler=dict(global_step=global_step, total_updates=total_updates), scaler=scaler.state_dict(),
                             epoch=epoch + 1, global_step=global_step, best=best, dropped=dropped_total, skipped=skipped_total,
                             config=config, args=jsonable(vars(args)), rng=dict(torch=torch.get_rng_state(), noise=gen.get_state().cpu())), last_path)
            log_file.write(json.dumps(clean(dict(event='epoch', improved=improved, **metrics))) + '\n')
            log_file.flush()
            nan = float('nan')
            mm = lambda key: metrics.get(key, nan) * 1000
            log(f'[epoch {epoch + 1}/{args.epochs}] k={k} train={metrics["train_loss"]:.4f} val1 loss/acc={metrics["val_loss"]:.4f}/{metrics["val_acc"]:.4f} | '
                f'rollout raw vert .5/1/2s={mm("vert_err_0.5s"):.1f}/{mm("vert_err_1s"):.1f}/{mm("vert_err_2s"):.1f}mm '
                f'com2s={mm("com_err_2s"):.1f}mm pen={mm("mean_penetration"):.2f}mm strain={metrics["strain_err"]:.4f} div={metrics["diverged"]} | '
                f'proj vert2s={mm("proj_vert_err_2s"):.1f}mm pen={mm("proj_mean_penetration"):.2f}mm | score={metrics["score"]:.4f}'
                f'{" *best*" if improved else ""} train {train_time:.0f}s val {metrics["val_seconds"]:.0f}s'
                + (f' | extra val raw vert 1/2s={mm("extra_vert_err_1s"):.1f}/{mm("extra_vert_err_2s"):.1f}mm proj 2s={mm("extra_proj_vert_err_2s"):.1f}mm' if extra_val is not None else ''))
            if monitor:
                log(f'[epoch {epoch + 1}] {monitor.summary()}; rank 0 peak allocated {torch.cuda.max_memory_allocated(device) / 2**30:.2f} GiB '
                    f'(reserved {torch.cuda.max_memory_reserved(device) / 2**30:.2f} GiB)')
        if stop:
            if main_rank:
                log(f'[stop] wall-clock budget of {args.max_minutes} min reached in epoch {epoch + 1} (step {global_step} of {total_updates}, '
                    f'{100 * global_step / max(total_updates, 1):.0f}% of the updates; k={k}, lr={lr:.2e}); checkpoint saved. '
                    'The LR/curriculum schedule was compressed into the budget; --resume restarts the clock and continues from the update count.')
            break

    if main_rank:
        log(f'[done] best score {best:.5f}; {(time.time() - start_wall) / 60:.1f} min; dropped batches {dropped_total}; skipped updates {skipped_total}')
        log_file.close()
        if monitor:
            monitor.stop_flag.set()
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
