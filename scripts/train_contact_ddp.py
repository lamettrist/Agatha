"""DDP trainer for agatha.contact_gnn.ContactGNN (multi-step, noise-injected, physics-loss).

Multi-GPU:
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 .venv-train/bin/torchrun --nproc_per_node=7 \
      scripts/train_contact_ddp.py --data data/v2 --out models/runs/NAME
Single process (one GPU or CPU; used for the per-GPU hyperparameter sweep):
  CUDA_VISIBLE_DEVICES=3 .venv-train/bin/python scripts/train_contact_ddp.py --data data/v2 --out models/runs/NAME

Notes
- --batch is PER GPU and --lr is used as given (no automatic scaling with the number of
  ranks). DDP averages gradients over ranks, so the effective batch is batch*ranks.
- An epoch is --windows-per-epoch windows summed over all ranks.
- The LR schedule and the horizon curriculum are functions of the global update count over
  --epochs epochs. With --max-minutes they follow max(update fraction, elapsed/budget) instead, so
  the cosine and the curriculum always complete inside the wall-clock budget of THIS invocation
  (the stop lands on a log interval; the partial epoch is validated and saved as usual).
- --warmup is capped at 5% of the total updates (large --batch runs have few updates).
- Everything a rank needs lives on its GPU: no DataLoader, windows are gathered on-device.
- OPT-IN video-derived data (--video-data DIR ... --video-frac F): the model is state-based (3D box poses), so video
  is only usable after it has been converted to approximate object trajectories (tracking + geometry; shards in the
  data/v2 format plus obs_mask / pos_sigma / has_orientation). Those are much noisier than simulator states, so a
  fraction F of every batch is drawn from video windows and trained with a position-only, noise-tolerant loss times
  the small --video-weight; the checkpoint selection score is the simulator score only. With --video-frac 0 (default)
  sampling, loss and updates are identical to a run without the flags.
"""
import argparse
import copy
import glob
import json
import math
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F

from agatha import contact_gnn as cg
from agatha.dynamics import euler_quaternion, rotation

TERMS = ('loss', 'one', 'pos', 'vel', 'ang', 'rot', 'pen', 'rest', 'slide')
# video windows (only with --video-frac > 0): unweighted video loss, its three parts, mean position error in metres
VIDEO_TERMS = ('video_loss', 'video_pos', 'video_pen', 'video_rot', 'video_err_m')


# ----------------------------------------------------------------------------- args
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--resume', action='store_true', help='continue from <out>/last.pt if present')
    p.add_argument('--init', type=Path, default=None,
                   help='warm-start model + EMA weights, normalizer and model config from a shipped-format checkpoint '
                        '(fresh optimizer and schedule); ignored when --resume finds <out>/last.pt')
    p.add_argument('--start-horizon', type=int, default=0,
                   help='lower bound on the curriculum horizon k (0 = normal 1/2/4/H curriculum); set to --horizon when warm-starting')
    # model
    p.add_argument('--width', type=int, default=128)
    p.add_argument('--rounds', type=int, default=4)
    p.add_argument('--edge-radius', type=float, default=0.12)
    # schedule
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--windows-per-epoch', type=int, default=2_000_000)
    p.add_argument('--batch', type=int, default=1024, help='windows per GPU per update')
    p.add_argument('--max-minutes', type=float, default=0, help='wall-clock budget; 0 = none')
    p.add_argument('--horizon', type=int, default=8)
    p.add_argument('--bptt', type=int, default=1, help='1 = gradients through the unroll, 0 = pushforward (detach)')
    p.add_argument('--step-decay', type=float, default=0.9)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--lr-min', type=float, default=1e-5)
    p.add_argument('--wd', type=float, default=1e-5)
    p.add_argument('--warmup', type=int, default=500)
    p.add_argument('--clip', type=float, default=1.0)
    p.add_argument('--ema', type=float, default=0.999)
    # noise
    p.add_argument('--noise-pos', type=float, default=0.002)
    p.add_argument('--noise-vel', type=float, default=0.02)
    p.add_argument('--noise-ang', type=float, default=0.05)
    p.add_argument('--noise-rot', type=float, default=0.005)
    # loss weights
    p.add_argument('--w-one', type=float, default=1.0)
    p.add_argument('--w-pos', type=float, default=1.0)
    p.add_argument('--w-vel', type=float, default=0.5)
    p.add_argument('--w-ang', type=float, default=0.2)
    p.add_argument('--w-rot', type=float, default=0.5)
    p.add_argument('--w-pen', type=float, default=1.0)
    p.add_argument('--w-rest', type=float, default=1.0)
    p.add_argument('--w-slide', type=float, default=1.0)
    p.add_argument('--one-steps', type=int, default=0, help='apply the one-step raw-target term only to the first n unroll steps (0 = all steps)')
    # fall-recall fixes; every default reproduces the original objective/sampling exactly. Loss changes are TRAIN-ONLY
    # (the val_* loss terms always use the original objective, so they stay comparable across runs).
    p.add_argument('--w-move', type=float, default=1.0,
                   help='per-object loss weight for objects that are not at rest on every frame of the window (1 = off)')
    p.add_argument('--w-impulse', type=float, default=0.0,
                   help='extra per-object-step loss weight w*min(1, max(|dv_true|/0.25 m/s, |dw_true|/2.5 rad/s)) (0 = off)')
    p.add_argument('--w-speed', type=float, default=0.0,
                   help='weight of a speed term on objects truly moving > 0.03 m/s: huber((v_pred . dir_true - |v_true|)/--speed-scale) (0 = off)')
    p.add_argument('--speed-scale', type=float, default=0.05)
    p.add_argument('--speed-asym', type=float, default=1.0, help='multiplier on the UNDER-shoot side of the --w-speed term')
    p.add_argument('--vel-scale', type=float, default=0.2, help='scale (m/s) of the velocity Huber term')
    p.add_argument('--one-delta', type=float, default=1.0, help='Huber delta (in target sigmas) of the one-step term')
    p.add_argument('--rest-lookahead', type=int, default=0,
                   help='apply the rest/slide penalties only where the object is truly at rest for this many FURTHER frames (0 = off)')
    p.add_argument('--event-frac', type=float, default=0.0,
                   help='fraction of every batch drawn from windows containing a motion onset or the 1 s approach of a true fall '
                        '(0 = off; the remainder stays half uniform / half active)')
    p.add_argument('--score-fall-weight', type=float, default=0.0,
                   help='selection score += w*((1 - fall recall) + 0.5*(1 - fall precision)), 4 s val rollout, collisions-enabled episodes (0 = unchanged)')
    # video-derived trajectories (opt-in; nothing changes while --video-frac is 0)
    p.add_argument('--video-data', type=str, nargs='+', default=None,
                   help='roots of video-derived shards (<root>/{train,val}/shard_*.npz: data/v2 keys + obs_mask [E,T+1,N], '
                        'pos_sigma [E,N] m, has_orientation [E,N]); val splits, when present, are always evaluated')
    p.add_argument('--video-frac', type=float, default=0.0, help='fraction of every batch drawn from video windows (0 = off)')
    p.add_argument('--video-weight', type=float, default=0.2, help='multiplier on the video loss: total = sim loss + W * video loss')
    p.add_argument('--video-horizon', type=int, default=0,
                   help='unroll length on video windows (0 = the current sim curriculum horizon k, i.e. --horizon once the curriculum is done)')
    p.add_argument('--video-w-rot', type=float, default=0.1,
                   help='weight of the weak orientation term on video objects whose has_orientation flag is set')
    # system
    p.add_argument('--amp', choices=('auto', 'bf16', 'fp16', 'off'), default='auto')
    p.add_argument('--compile', type=int, default=0)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--max-train-episodes', type=int, default=0)
    p.add_argument('--max-val-episodes', type=int, default=4096)
    p.add_argument('--val-rollout-episodes', type=int, default=512)
    p.add_argument('--val-windows', type=int, default=8192)
    p.add_argument('--log-every', type=int, default=50, help='updates between log lines')
    p.add_argument('--data-device', choices=('auto', 'cuda', 'cpu'), default='auto')
    p.add_argument('--load-threads', type=int, default=8)
    p.add_argument('--dist-backend', choices=('auto', 'nccl', 'gloo'), default='auto')
    p.add_argument('--nccl-p2p', type=int, default=0, help='1 = leave NCCL P2P transport enabled (hangs on this server)')
    p.add_argument('--find-unused', type=int, default=0, help='DDP find_unused_parameters')
    return p.parse_args()


def log(*a):
    print(*a, flush=True)


# ----------------------------------------------------------------------------- data
def _load_shard(path, need_meta):
    with np.load(path) as z:
        states = z['states'].astype(np.float32, copy=False)
        props = z['props'].astype(np.float32, copy=False)
        context = z['context'].astype(np.float32, copy=False)
        mask = z['mask'].astype(bool)
        rest = z['rest'].astype(bool)
        # one byte per object-frame holding the three contact bits; only used for the activity index
        contact = (z['contact_support'].astype(np.uint8) | (z['contact_floor'].astype(np.uint8) << 1)
                   | (z['contact_object'].astype(np.uint8) << 2))
        chaotic = z['chaotic'].astype(bool) if need_meta and 'chaotic' in z.files else np.zeros(len(states), bool)
    dyn = mask & (props[..., 5] < .5)                                        # [E,N]
    speed = np.linalg.norm(states[..., 7:10], axis=-1)                       # [E,T+1,N]
    moving = ((speed > .05) & dyn[:, None]).any(-1)                          # [E,T+1]
    change = np.zeros_like(moving)
    change[:, 1:] = ((contact[:, 1:] != contact[:, :-1]) & dyn[:, None]).any(-1)
    return dict(states=states, props=props, context=context, mask=mask, rest=rest,
                active=moving | change, chaotic=chaotic)


def load_split(root: Path, split: str, limit: int, threads: int, need_meta: bool = False):
    files = sorted(glob.glob(str(root / split / 'shard_*.npz')))
    if not files:
        raise FileNotFoundError(f'no shards in {root / split}')
    if limit:
        # shards are loaded in order until the cap is reached (smoke runs)
        kept, total = [], 0
        for f in files:
            kept.append(f)
            with np.load(f) as z:
                total += len(z['mask'])
            if total >= limit:
                break
        files = kept
    with ThreadPoolExecutor(max(1, threads)) as pool:
        parts = list(pool.map(lambda f: _load_shard(f, need_meta), files))
    data = {k: np.concatenate([q[k] for q in parts]) for k in parts[0]}
    if limit:
        data = {k: v[:limit] for k, v in data.items()}
    return data


def to_device(data, device):
    return {k: torch.from_numpy(v).to(device) for k, v in data.items()}


class WindowSampler:
    """Disjoint per-rank slices of a shared (seed, epoch, k) permutation; half uniform, half active."""

    def __init__(self, active_frames: torch.Tensor, seed: int, rank: int, world: int, event_frames=None, event_frac: float = 0.):
        self.frames = active_frames                       # [E,T+1] bool on the data device
        self.event_frac = float(event_frac) if event_frames is not None else 0.
        self.event_cs = F.pad(event_frames.long().cumsum(1), (1, 0)) if self.event_frac > 0 else None
        self.event = None
        self.E, self.T = active_frames.shape[0], active_frames.shape[1] - 1
        self.cs = F.pad(active_frames.long().cumsum(1), (1, 0))  # [E,T+2]
        self.seed, self.rank, self.world = seed, rank, world
        self.key = None

    def set(self, epoch: int, k: int):
        if self.key == (epoch, k):
            return
        self.key, self.k = (epoch, k), k
        self.per = self.T - k + 1                         # valid t0 in [0, T-k]
        device = self.frames.device
        g = torch.Generator().manual_seed(self.seed * 1_000_003 + epoch * 1009 + k)
        n = self.E * self.per
        self.uniform = torch.randperm(n, generator=g)[self.rank::self.world].to(device)
        t0 = torch.arange(self.per, device=device)
        win = (self.cs[:, t0 + k + 1] - self.cs[:, t0]) > 0   # any active frame in [t0, t0+k]
        idx = win.reshape(-1).nonzero().squeeze(1)
        self.active_fraction = float(len(idx)) / max(n, 1)
        if len(idx) >= self.world:
            self.active = idx[torch.randperm(len(idx), generator=g)[self.rank::self.world].to(device)]
        else:
            self.active = self.uniform
        if len(self.uniform) == 0:                        # fewer windows than ranks (tiny smoke data)
            self.uniform = torch.randperm(n, generator=g).to(device)
        self.cu = self.ca = 0
        if self.event_cs is not None:                     # drawn AFTER the original pools: their permutations are unchanged
            idx = ((self.event_cs[:, t0 + k + 1] - self.event_cs[:, t0]) > 0).reshape(-1).nonzero().squeeze(1)
            self.event_fraction = float(len(idx)) / max(n, 1)
            self.event = idx[torch.randperm(len(idx), generator=g)[self.rank::self.world].to(device)] if len(idx) >= self.world else self.active
            self.ce = 0

    def next(self, batch: int):
        device = self.frames.device
        ne = int(round(batch * self.event_frac)) if self.event is not None else 0
        nu = (batch - ne) // 2
        na = batch - ne - nu
        u = self.uniform[(self.cu + torch.arange(nu, device=device)) % len(self.uniform)]
        a = self.active[(self.ca + torch.arange(na, device=device)) % len(self.active)]
        self.cu += nu
        self.ca += na
        flat = torch.cat((u, a))
        if ne:
            flat = torch.cat((flat, self.event[(self.ce + torch.arange(ne, device=device)) % len(self.event)]))
            self.ce += ne
        return flat // self.per, flat % self.per


def gather(data, e, t0, k, device, lookahead=0):
    """Windows [B,k+1,...] for episodes e starting at t0 (rest gets `lookahead` extra frames, clipped at the episode end)."""
    t = t0[:, None] + torch.arange(k + 1, device=t0.device)
    tr = t if not lookahead else (t0[:, None] + torch.arange(k + 1 + lookahead, device=t0.device)).clamp(max=data['rest'].shape[1] - 1)
    out = (data['states'][e[:, None], t], data['rest'][e[:, None], tr], data['props'][e], data['context'][e], data['mask'][e])
    return tuple(x.to(device, non_blocking=True) for x in out)


# ----------------------------------------------------------------------------- video-derived data
def _load_video_shard(path):
    """data/v2 shard keys + obs_mask [E,T+1,N], pos_sigma [E,N], has_orientation [E,N] (+ optional vel_valid [E,T+1,N]).
    Missing keys fall back to the conservative choice: every frame observed, sigma 0, no orientation."""
    with np.load(path) as z:
        states = np.array(z['states'], dtype=np.float32)
        props = z['props'].astype(np.float32, copy=False)
        context = z['context'].astype(np.float32, copy=False)
        mask = z['mask'].astype(bool)
        E, frames, N = states.shape[:3]
        obs = z['obs_mask'].astype(bool) if 'obs_mask' in z.files else np.ones((E, frames, N), bool)
        sigma = z['pos_sigma'].astype(np.float32) if 'pos_sigma' in z.files else np.zeros((E, N), np.float32)
        ori = z['has_orientation'].astype(bool) if 'has_orientation' in z.files else np.zeros((E, N), bool)
        vel_valid = z['vel_valid'].astype(bool) if 'vel_valid' in z.files else None
    finite = np.isfinite(states).all(-1)
    obs = obs & finite & mask[:, None]
    states = np.nan_to_num(states, nan=0., posinf=0., neginf=0.)
    norm = np.linalg.norm(states[..., 3:7], axis=-1, keepdims=True)
    states[..., 3:7] = np.where(norm > 1e-6, states[..., 3:7] / np.maximum(norm, 1e-6), np.array([0, 0, 0, 1], np.float32))
    if vel_valid is None:
        # a finite-difference velocity needs a neighbouring observation of the same object
        neighbour = np.zeros_like(obs)
        neighbour[:, 1:] |= obs[:, :-1]
        neighbour[:, :-1] |= obs[:, 1:]
        vel_valid = neighbour
    sigma = np.where(np.isfinite(sigma), sigma, 0.).astype(np.float32).clip(min=0.)
    return dict(states=states, props=props, context=context, mask=mask, obs=obs, vel_ok=obs & vel_valid, sigma=sigma, ori=ori & mask)


def _pad_video(part, frames, slots):
    E, f, n = part['states'].shape[:3]
    if (f, n) == (frames, slots):
        return part

    def grow(x, shape, fill=0):
        out = np.full(shape + x.shape[len(shape):], fill, x.dtype)
        out[tuple(slice(0, s) for s in x.shape[:len(shape)])] = x
        return out
    states = grow(part['states'], (E, frames, slots))
    pad = np.ones((E, frames, slots), bool)
    pad[:, :f, :n] = False
    states[pad, 6] = 1.                                    # identity quaternion in padded frames / slots
    return dict(states=states, props=grow(part['props'], (E, slots)), context=part['context'], mask=grow(part['mask'], (E, slots)),
                obs=grow(part['obs'], (E, frames, slots)), vel_ok=grow(part['vel_ok'], (E, frames, slots)),
                sigma=grow(part['sigma'], (E, slots)), ori=grow(part['ori'], (E, slots)))


def load_video_split(roots, split: str, threads: int = 8, slots: int = 0):
    """All video sources concatenated (frames / slots padded to the largest; padded entries are unobserved and masked).
    Returns (data dict of numpy arrays or None when no root has this split, [source names])."""
    parts, names = [], []
    for root in roots:
        files = sorted(glob.glob(str(Path(root) / split / 'shard_*.npz')))
        if not files:
            continue
        with ThreadPoolExecutor(max(1, threads)) as pool:
            loaded = list(pool.map(_load_video_shard, files))
        for q in loaded:
            q['source'] = np.full(len(q['states']), len(names), np.int64)
        parts += loaded
        names.append(Path(root).name)
    if not parts:
        return None, names
    for key in ('props', 'context'):
        dims = {q[key].shape[-1] for q in parts}
        if len(dims) > 1:
            raise ValueError(f'video shards disagree on the {key} width: {sorted(dims)}')
    frames, slots = max(q['states'].shape[1] for q in parts), max(slots, max(q['states'].shape[2] for q in parts))
    parts = [dict(_pad_video(q, frames, slots), source=q['source']) for q in parts]
    data = {k: np.concatenate([q[k] for q in parts]) for k in parts[0]}
    dyn = data['mask'] & (data['props'][..., 5] < .5)                                   # [E,N]
    # a window may START at a frame where every real dynamic object is observed with a valid velocity estimate
    data['start_ok'] = (data['vel_ok'] | ~dyn[:, None]).all(-1) & dyn.any(-1)[:, None]  # [E,T+1]
    data['obs_count'] = (data['obs'] & dyn[:, None]).sum(-1).astype(np.int64)           # observed dynamic objects per frame
    del data['vel_ok']
    return data, names


def video_window_index(data, k: int):
    """Flat indices e*(T-k+1)+t0 of the windows that start at a valid frame and have an observed target in t0+1..t0+k."""
    T = data['states'].shape[1] - 1
    per = T - k + 1
    t0 = torch.arange(per, device=data['states'].device)
    cs = F.pad(data['obs_count'].cumsum(1), (1, 0))
    win = data['start_ok'][:, :per] & ((cs[:, t0 + k + 1] - cs[:, t0 + 1]) > 0)
    return win.reshape(-1).nonzero().squeeze(1), per


class VideoSampler:
    """Per-rank slices of a (seed, epoch, k) permutation of the valid video windows; independent of WindowSampler's generator."""

    def __init__(self, data, seed: int, rank: int, world: int):
        self.data, self.seed, self.rank, self.world, self.key = data, seed, rank, world, None

    def set(self, epoch: int, k: int):
        if self.key == (epoch, k):
            return
        self.key = (epoch, k)
        idx, self.per = video_window_index(self.data, k)
        if len(idx) == 0:
            raise RuntimeError(f'no valid video window of horizon {k}: lower --video-horizon or check obs_mask / velocities')
        g = torch.Generator().manual_seed(self.seed * 1_000_003 + epoch * 1009 + k + 500_009)
        perm = torch.randperm(len(idx), generator=g)
        self.pool = idx[(perm[self.rank::self.world] if len(idx) >= self.world else perm).to(idx.device)]
        self.windows, self.cursor = len(idx), 0

    def next(self, batch: int):
        flat = self.pool[(self.cursor + torch.arange(batch, device=self.pool.device)) % len(self.pool)]
        self.cursor += batch
        return flat // self.per, flat % self.per


def gather_video(data, e, t0, k, device):
    t = t0[:, None] + torch.arange(k + 1, device=t0.device)
    out = (data['states'][e[:, None], t], data['obs'][e[:, None], t], data['props'][e], data['context'][e], data['mask'][e],
           data['sigma'][e], data['ori'][e])
    return tuple(x.to(device, non_blocking=True) for x in out)


# ----------------------------------------------------------------------------- losses
def dynamic_mask(props, context, mask):
    static = props[..., 5] > .5
    alias = static & (props[..., 0] >= .95 * context[:, 0, None]) & (props[..., 2] >= .95 * context[:, 1, None])
    return mask & ~static & ~alias


def quat_mul(a, b):
    ax, ay, az, aw = a.unbind(-1)
    bx, by, bz, bw = b.unbind(-1)
    return torch.stack((aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                        aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz), -1)


def geodesic(q, target):
    """Rotation angle (rad) between unit quaternions, sign-invariant, finite gradient at 0."""
    conj = torch.cat((-target[..., :3], target[..., 3:4]), -1)
    rel = quat_mul(q, conj)
    sin = (rel[..., :3].square().sum(-1) + 1e-12).sqrt()
    return 2 * torch.atan2(sin, rel[..., 3].abs())


def add_noise(state, rest0, dyn, args, gen):
    shape = state.shape[:2]
    device = state.device

    def rnd(*s):
        return torch.randn(*s, device=device, generator=gen)
    clean = rest0 & (torch.rand(shape, device=device, generator=gen) < .5)   # true rest seen clean half the time
    on = (dyn & ~clean)[..., None].float()
    half = rnd(*shape, 3) * (args.noise_rot / 2)
    dq = torch.cat((half, torch.ones(*shape, 1, device=device)), -1)
    q = quat_mul(dq / dq.norm(dim=-1, keepdim=True), state[..., 3:7])
    noisy = torch.cat((state[..., :3] + rnd(*shape, 3) * args.noise_pos, q,
                       state[..., 7:10] + rnd(*shape, 3) * args.noise_vel,
                       state[..., 10:13] + rnd(*shape, 3) * args.noise_ang), -1)
    return torch.where(on > 0, noisy, state)


def huber(x, delta=1.0):
    return F.huber_loss(x, torch.zeros_like(x), reduction='none', delta=delta)


def support_local(pos, context):
    """Positions [B,...,3] in the tilted support frame centred at (0, height-.04, 0) (same as scripts/eval_contact.py)."""
    rot = rotation(euler_quaternion(context[:, 4:7]))
    center = torch.zeros(context.shape[0], 3, dtype=pos.dtype, device=pos.device)
    center[:, 1] = context[:, 2] - .04
    return torch.einsum('bji,b...j->b...i', rot, pos - center.view(-1, *((1,) * (pos.dim() - 2)), 3))


def fell_frames(traj, props, context, mask):
    """scripts/eval_contact.fell_tipped rule per frame. traj [B,F,N,13] -> (off footprint [B,F,N], fell frame [B,F,N], dyn [B,N])."""
    local = support_local(traj[..., :3].float(), context.float())
    half_w, half_d = (context[:, 0] / 2).view(-1, 1, 1), (context[:, 1] / 2).view(-1, 1, 1)
    off = (context[:, 7] < .5).view(-1, 1, 1) | (local[..., 0].abs() > half_w) | (local[..., 2].abs() > half_d)
    return off, off & (local[..., 1] < .01), mask & (props[..., 5] < .5)


@torch.no_grad()
def event_frames(data, lead=20, chunk=2048):
    """[E,T+1] bool: frames with a motion onset (dynamic object slower than 0.05 m/s that reaches 0.1 m/s within 4 frames)
    or inside the `lead` frames before (to 2 after) a true faller first leaves the footprint (collisions-enabled supports)."""
    out = []
    for i in range(0, data['states'].shape[0], chunk):
        sl = slice(i, i + chunk)
        states, props, context, mask = data['states'][sl], data['props'][sl], data['context'][sl], data['mask'][sl]
        off, fell, dyn = fell_frames(states, props, context, mask)
        faller = fell.any(1) & dyn & (context[:, 7] > .5)[:, None]                       # [e,N]
        frames = states.shape[1]
        tc = off.float().argmax(1)                                                       # first off-footprint frame [e,N]
        t = torch.arange(frames, device=states.device).view(1, -1, 1)
        approach = (faller[:, None] & (t >= tc[:, None] - lead) & (t <= tc[:, None] + 2)).any(-1)
        speed = states[..., 7:10].norm(dim=-1).transpose(1, 2)                           # [e,N,T+1]
        ahead = F.max_pool1d(F.pad(speed[..., 1:], (0, 4)), 4, 1)                        # max speed over t+1..t+4
        onset = ((speed < .05) & (ahead >= .1) & dyn[..., None]).any(1)
        out.append(approach | onset)
    return torch.cat(out)


def unroll(model, states, rest, props, context, mask, k, args, amp_dtype, gen=None, video=None):
    """k-step autoregressive loss. Returns (loss, stats[len(TERMS)], bad flag tensor).
    video: optional (states, obs, props, context, mask, sigma, ori) windows of the same k and slot count. They ride in the
    same model calls (the model is dispatch-bound, a second unroll would halve the throughput) but are scored only by
    VideoLoss; every sim term below is computed on the sim rows alone. Then returns
    (sim loss + --video-weight * video loss, sim stats, bad, video stats[len(VIDEO_TERMS)])."""
    dyn = dynamic_mask(props, context, mask)
    count = dyn.sum().clamp_min(1)

    def avg(x, w=None):
        if w is None:
            return (x * dyn).sum() / count
        return (x * w).sum() / w.sum().clamp_min(1e-6)
    train = gen is not None                               # the fall-recall loss options never touch validation
    rest_all, rest = rest, rest[:, :k + 1]                # training windows may carry --rest-lookahead extra rest frames
    lookahead = rest_all.shape[1] - (k + 1) if train else 0
    base = None
    if train and (args.w_move != 1 or args.w_impulse > 0):
        mover = (~rest).any(1) & dyn
        base = 1 + (args.w_move - 1) * mover.float()
    delta = args.one_delta if train else 1.0
    cur = states[:, 0]
    if gen is not None:
        cur = add_noise(cur, rest[:, 0], dyn, args, gen)
    mean, std = model.out_mean.float(), model.out_std.float()
    weights = [args.step_decay ** j for j in range(k)]
    total = states.new_zeros(())
    stats = states.new_zeros(len(TERMS))
    bad = torch.zeros((), dtype=torch.bool, device=states.device)
    use_amp = amp_dtype is not None and states.is_cuda
    if video is not None:
        rows = states.shape[0]
        vloss = VideoLoss(*video, args)
        vcur = video[0][:, 0]
        props_all, context_all, mask_all = torch.cat((props, video[2])), torch.cat((context, video[3])), torch.cat((mask, video[4]))
    for j in range(k):
        true = states[:, j + 1]
        if video is None:
            with torch.autocast('cuda', dtype=amp_dtype or torch.float16, enabled=use_amp):
                raw = model.predict_raw(cur, props, context, mask)
            raw = raw.float()
        else:
            cur_all = torch.cat((cur, vcur))
            with torch.autocast('cuda', dtype=amp_dtype or torch.float16, enabled=use_amp):
                raw_all = model.predict_raw(cur_all, props_all, context_all, mask_all)
            raw_all = raw_all.float()
            raw = raw_all[:rows]
        with torch.no_grad():
            # pushforward-style target: from the model's CURRENT (possibly drifted) input to the TRUE next state
            target = (model.raw_targets(cur.detach(), true, props, context, mask).float() - mean) / std
        if video is None:
            nxt = model.integrate(cur, raw * std + mean, props, context, mask)
        else:
            nxt_all = model.integrate(cur_all, raw_all * std + mean, props_all, context_all, mask_all)
            nxt, vnxt = nxt_all[:rows], nxt_all[rows:]
            vloss.step(j, vnxt)
            vcur = vnxt if args.bptt else vnxt.detach()
        wj = None
        if base is not None:
            wj = base
            if args.w_impulse > 0:
                jump = torch.maximum((true[..., 7:10] - states[:, j, :, 7:10]).norm(dim=-1) / .25,
                                     (true[..., 10:13] - states[:, j, :, 10:13]).norm(dim=-1) / 2.5).clamp(max=1)
                wj = wj + args.w_impulse * jump
            wj = wj * dyn
        terms = dict(
            one=avg(huber(raw - target, delta).mean(-1), wj),
            pos=avg(huber((nxt[..., :3] - true[..., :3]) / .02).mean(-1), wj),
            vel=avg(huber((nxt[..., 7:10] - true[..., 7:10]) / (args.vel_scale if train else .2)).mean(-1), wj),
            ang=avg(huber((nxt[..., 10:13] - true[..., 10:13]) / 2).mean(-1), wj),
            rot=avg(huber(geodesic(nxt[..., 3:7], true[..., 3:7]) / .1), wj))
        # Slide anchor for objects truly at rest: while the input is within 1 cm of the truth (always the
        # case for the noised j=0 input) the object must end on the TRUE resting pose, so the ground truth
        # scores zero and correcting noise/drift is never punished; a further-drifted object is only asked
        # not to move relative to its own previous pose (bounded, as before).
        near = (cur.detach()[..., :3] - states[:, j, :, :3]).norm(dim=-1, keepdim=True) < .01
        rest_next = rest_all[:, j + 1:j + 2 + lookahead].all(1) if lookahead > 0 else rest[:, j + 1]
        phys = cg.physics_losses(nxt, true, torch.where(near, true, cur), props, context, mask, rest_next)
        terms.update(pen=phys['penetration'], rest=phys['rest'], slide=phys['slide'])
        w_one = args.w_one if not args.one_steps or j < args.one_steps else 0.
        step = (w_one * terms['one'] + args.w_pos * terms['pos'] + args.w_vel * terms['vel'] + args.w_ang * terms['ang']
                + args.w_rot * terms['rot'] + args.w_pen * terms['pen'] + args.w_rest * terms['rest'] + args.w_slide * terms['slide'])
        if train and args.w_speed > 0:
            v_true = true[..., 7:10]
            s_true = v_true.norm(dim=-1)
            err = ((nxt[..., 7:10] * v_true).sum(-1) / s_true.clamp_min(1e-6) - s_true) / args.speed_scale
            speed = huber(err) * torch.where(err < 0, args.speed_asym, 1.) * (s_true > .03)
            step = step + args.w_speed * avg(speed, wj)   # reported inside 'loss' only (TERMS is unchanged)
        total = total + weights[j] * step
        stats = stats + weights[j] * torch.stack([step.detach()] + [terms[n].detach().float() for n in TERMS[1:]])
        bad = bad | ~torch.isfinite(nxt).all() | (nxt[..., :3].abs() > 100).any()
        cur = nxt if args.bptt else nxt.detach()
    norm = sum(weights)
    if video is not None:
        video_loss, video_stats, video_bad = vloss.result()
        return total / norm + args.video_weight * video_loss, stats / norm, bad | ~torch.isfinite(total) | video_bad, video_stats
    return total / norm, stats / norm, bad | ~torch.isfinite(total)


class VideoLoss:
    """Loss on video-derived windows, accumulated over the unroll steps.
    Position-only Huber on (p_pred - p_video)/max(0.02, 2*pos_sigma) at observed frames, a penetration penalty on the
    PREDICTED state alone (depth beyond 2 mm; there is no trustworthy reference state) and a weak orientation term on
    objects flagged has_orientation. No velocity / one-step / rest / slide terms: those targets are not trustworthy."""

    def __init__(self, states, obs, props, context, mask, sigma, ori, args):
        self.states, self.obs, self.props, self.context, self.mask, self.ori, self.args = states, obs, props, context, mask, ori, args
        self.dyn = dynamic_mask(props, context, mask)
        self.count = self.dyn.sum().clamp_min(1)
        self.scale = (2 * sigma).clamp_min(.02)[..., None]
        zero = states.new_zeros(())
        self.pos_sum = self.pos_w = self.rot_sum = self.rot_w = self.pen_sum = self.err_sum = self.err_n = zero
        self.weight_sum = 0.
        self.bad = torch.zeros((), dtype=torch.bool, device=states.device)

    def step(self, j, nxt):
        """nxt: the model's prediction for frame j+1 of the window."""
        weight = self.args.step_decay ** j
        target = self.states[:, j + 1]
        seen = (self.obs[:, j + 1] & self.dyn).float()
        diff = nxt[..., :3] - target[..., :3]
        self.pos_sum = self.pos_sum + weight * (huber(diff / self.scale).mean(-1) * seen).sum()
        self.pos_w = self.pos_w + weight * seen.sum()
        oriented = seen * self.ori
        self.rot_sum = self.rot_sum + weight * (huber(geodesic(nxt[..., 3:7], target[..., 3:7]) / .3) * oriented).sum()
        self.rot_w = self.rot_w + weight * oriented.sum()
        depth = cg.penetration_depths(nxt, self.props, self.context, self.mask)
        for key in ('support', 'floor', 'objects'):
            self.pen_sum = self.pen_sum + weight * (huber((depth[key] - .002).clamp_min(0) / .005) * self.dyn).sum() / self.count
        self.weight_sum += weight
        self.err_sum = self.err_sum + (diff.detach().norm(dim=-1) * seen).sum()
        self.err_n = self.err_n + seen.sum()
        self.bad = self.bad | ~torch.isfinite(nxt).all() | (nxt[..., :3].abs() > 100).any()

    def result(self):
        """(loss, stats[len(VIDEO_TERMS)], bad flag tensor)"""
        pos = self.pos_sum / self.pos_w.clamp_min(1e-6)
        rot = self.rot_sum / self.rot_w.clamp_min(1e-6)
        pen = self.pen_sum / max(self.weight_sum, 1e-9)
        loss = self.args.w_pos * pos + self.args.w_pen * pen + self.args.video_w_rot * rot
        stats = torch.stack([loss.detach(), pos.detach(), pen.detach(), rot.detach(), self.err_sum / self.err_n.clamp_min(1)]).float()
        return loss, stats, self.bad | ~torch.isfinite(loss)


def unroll_video(model, states, obs, props, context, mask, sigma, ori, k, args, amp_dtype):
    """k-step VideoLoss on video-derived windows alone, unrolled exactly like a sim window from the (already noisy)
    video state. Used when the video windows cannot share the sim forward (different k, slot count or feature widths).
    Returns (loss, stats[len(VIDEO_TERMS)], bad flag tensor)."""
    acc = VideoLoss(states, obs, props, context, mask, sigma, ori, args)
    mean, std = model.out_mean.float(), model.out_std.float()
    cur = states[:, 0]
    use_amp = amp_dtype is not None and states.is_cuda
    for j in range(k):
        with torch.autocast('cuda', dtype=amp_dtype or torch.float16, enabled=use_amp):
            raw = model.predict_raw(cur, props, context, mask)
        nxt = model.integrate(cur, raw.float() * std + mean, props, context, mask)
        acc.step(j, nxt)
        cur = nxt if args.bptt else nxt.detach()
    return acc.result()


class Trainee(nn.Module):
    """Whole unroll inside forward() so DistributedDataParallel sees one forward per update."""

    def __init__(self, model, args, amp_dtype):
        super().__init__()
        self.model, self.args, self.amp_dtype = model, args, amp_dtype

    def forward(self, states, rest, props, context, mask, k, gen, video=None):
        if video is None:
            return unroll(self.model, states, rest, props, context, mask, k, self.args, self.amp_dtype, gen)
        # video windows share the forward (one DDP forward per update); the sim loss and its statistics stay separate
        *windows, kv = video
        if kv == k and all(a.shape[1:] == b.shape[1:] for a, b in zip(windows[2:5], (props, context, mask))):
            return unroll(self.model, states, rest, props, context, mask, k, self.args, self.amp_dtype, gen, windows)
        out = unroll(self.model, states, rest, props, context, mask, k, self.args, self.amp_dtype, gen)
        loss, stats, bad = unroll_video(self.model, *windows, kv, self.args, self.amp_dtype)
        return out[0] + self.args.video_weight * loss, out[1], out[2] | bad, stats


# ----------------------------------------------------------------------------- schedule
def horizon_at(step, total, horizon, progress=0., start=0):
    f = max(step / max(total, 1), progress)
    k = 1 if f < .05 else 2 if f < .15 else 4 if f < .35 else horizon
    return min(max(k, start), horizon)


def lr_at(step, total, args, progress=0.):
    """progress: wall-clock fraction of --max-minutes used (0 without a budget)."""
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    f = min(1., max((step - args.warmup) / max(total - args.warmup, 1), progress))
    return args.lr_min + (args.lr - args.lr_min) * .5 * (1 + math.cos(math.pi * f))


def choose_amp(mode, device):
    if device.type != 'cuda' or mode == 'off':
        return None, False, 'off (' + ('requested' if mode == 'off' else 'no CUDA device') + ')'
    if mode == 'auto':
        try:
            native = torch.cuda.is_bf16_supported(including_emulation=False)
        except TypeError:
            native = False
        name = torch.cuda.get_device_name(device)
        if native:
            return torch.bfloat16, False, f'bf16 (auto: {name} has native bf16)'
        return torch.float16, True, f'fp16 + GradScaler (auto: {name} has no native bf16)'
    if mode == 'bf16':
        return torch.bfloat16, False, 'bf16 (requested; emulated and slow on GPUs without native support)'
    return torch.float16, True, 'fp16 + GradScaler (requested)'


# ----------------------------------------------------------------------------- validation
@torch.no_grad()
def fit_normalizer(model, data, device, seed, samples=200_000):
    E, T = data['states'].shape[0], data['states'].shape[1] - 1
    g = torch.Generator().manual_seed(seed + 7919)
    e = torch.randint(E, (samples,), generator=g).to(data['states'].device)
    t = torch.randint(T, (samples,), generator=g).to(data['states'].device)
    rows = []
    for i in range(0, samples, 16384):
        ei, ti = e[i:i + 16384], t[i:i + 16384]
        s0, s1 = data['states'][ei, ti].to(device), data['states'][ei, ti + 1].to(device)
        props, context, mask = data['props'][ei].to(device), data['context'][ei].to(device), data['mask'][ei].to(device)
        raw = model.raw_targets(s0, s1, props, context, mask).float()
        rows.append(raw[dynamic_mask(props, context, mask)])
    raw = torch.cat(rows)
    raw = raw[torch.isfinite(raw).all(-1)]
    mean, std = raw.mean(0), raw.std(0).clamp_min(1e-4)
    model.set_normalizer(mean, std)
    return mean, std, len(raw)


def _mean(x, m):
    m = m.float()
    return float((x * m).sum() / m.sum()) if float(m.sum()) > 0 else float('nan')


@torch.no_grad()
def validate(model, val, device, args, amp_dtype):
    model.eval()
    E, T = val['states'].shape[0], val['states'].shape[1] - 1
    k = min(args.horizon, T)
    # (i) k-step loss on a fixed, noise-free set of windows
    g = torch.Generator().manual_seed(args.seed + 104729)
    n = min(args.val_windows, E * (T - k + 1))
    flat = torch.randperm(E * (T - k + 1), generator=g)[:n].to(val['states'].device)
    acc, done = torch.zeros(len(TERMS), device=device), 0
    for i in range(0, n, 2048):
        f = flat[i:i + 2048]
        batch = gather(val, f // (T - k + 1), f % (T - k + 1), k, device)
        _, stats, _ = unroll(model, *batch, k, args, amp_dtype, None)
        acc += torch.nan_to_num(stats, nan=1e6, posinf=1e6, neginf=1e6) * len(f)
        done += len(f)
    out = {f'val_{name}': float(v) for name, v in zip(TERMS, acc / max(done, 1))}
    # (ii) full rollouts from t=0, no projection
    m = min(args.val_rollout_episodes, E)
    steps = min(40, T)
    sl = slice(0, m)
    states, rest = val['states'][sl].to(device), val['rest'][sl].to(device)
    props, context, mask = val['props'][sl].to(device), val['context'][sl].to(device), val['mask'][sl].to(device)
    chaotic = val['chaotic'][sl].to(device)
    dyn = dynamic_mask(props, context, mask)
    cur = states[:, 0]
    pen_sum = drift_sum = drift_n = 0.
    errors = {}
    fall_steps = min(80, T)                               # the 4 s rollout continues past `steps` for the fall metrics only
    traj = [cur]
    for j in range(1, fall_steps + 1):
        with torch.autocast('cuda', dtype=amp_dtype or torch.float16, enabled=amp_dtype is not None and device.type == 'cuda'):
            cur = model(cur, props, context, mask)
        cur = torch.nan_to_num(cur.float(), nan=1e3, posinf=1e3, neginf=-1e3).clamp(-1e3, 1e3)
        traj.append(cur)
        if j > steps:
            continue
        depth = cg.penetration_depths(cur, props, context, mask)
        worst = torch.maximum(torch.maximum(depth['support'], depth['floor']), depth['objects'])
        pen_sum += float((torch.nan_to_num(worst, nan=1.) * dyn).sum())
        at_rest = rest[:, j] & dyn
        drift_sum += float((cur[..., 7:10].norm(dim=-1) * at_rest).sum())
        drift_n += float(at_rest.sum())
        if j in (20, 40):
            errors[j] = (cur[..., :3] - states[:, j, :, :3]).norm(dim=-1)
    for j, name in ((20, '1s'), (40, '2s')):
        if j in errors:
            out[f'pos_err_{name}_chaotic'] = _mean(errors[j], dyn & chaotic[:, None])
            out[f'pos_err_{name}_gentle'] = _mean(errors[j], dyn & ~chaotic[:, None])
    out['mean_penetration'] = pen_sum / max(float(dyn.sum()) * steps, 1)
    out['rest_drift'] = drift_sum / max(drift_n, 1)
    out['rollout_episodes'], out['rollout_chaotic'] = m, int(chaotic.sum())

    def z(key):
        v = out.get(key, float('nan'))
        return 0. if v != v else v
    out['score'] = z('pos_err_2s_chaotic') + z('pos_err_2s_gentle') + 10 * out['mean_penetration'] + .5 * out['rest_drift']
    # fall precision / recall with the scripts/eval_contact.py rule on every frame of the 4 s rollout (object level)
    _, pred_fell, fdyn = fell_frames(torch.stack(traj, 1), props, context, mask)
    _, true_fell, _ = fell_frames(states[:, :fall_steps + 1], props, context, mask)
    pred_fell, true_fell = pred_fell.any(1) & fdyn, true_fell.any(1) & fdyn
    for tag, sel in (('', fdyn), ('_on', fdyn & (context[:, 7] > .5)[:, None])):   # _on: collisions-enabled supports only
        tp, fp, fn = (float((a & sel).sum()) for a in (pred_fell & true_fell, pred_fell & ~true_fell, ~pred_fell & true_fell))
        out[f'fall_precision{tag}'] = tp / (tp + fp) if tp + fp else float('nan')
        out[f'fall_recall{tag}'] = tp / (tp + fn) if tp + fn else float('nan')
        out[f'fall_true{tag}'] = int(tp + fn)
    if args.score_fall_weight > 0:
        r, q = out['fall_recall_on'], out['fall_precision_on']
        out['score'] += args.score_fall_weight * ((1 - (0. if r != r else r)) + .5 * (1 - (0. if q != q else q)))
    return out


@torch.no_grad()
def validate_video(model, data, names, device, args, amp_dtype):
    """Position error (m) of raw rollouts started from video-derived states, against the video-derived positions at
    observed frames. Reported only: never part of the selection score. `hold` = error of leaving every object at its start."""
    model.eval()
    T = data['states'].shape[1] - 1
    k = min(20, T)                                        # up to 1 s
    short = min(args.video_horizon or args.horizon, k)
    idx, per = video_window_index(data, k)
    if len(idx) == 0:
        return {'video_val_windows': 0}
    g = torch.Generator().manual_seed(args.seed + 15485863)
    idx = idx[torch.randperm(len(idx), generator=g)[:args.val_windows].to(idx.device)]
    err = torch.zeros(k + 1, device=device)
    hold = torch.zeros(k + 1, device=device)
    seen_n = torch.zeros(k + 1, device=device)
    src_err = torch.zeros(len(names), device=device)
    src_n = torch.zeros(len(names), device=device)
    pen_sum = pen_n = 0.
    for i in range(0, len(idx), 2048):
        f = idx[i:i + 2048]
        e = f // per
        states, obs, props, context, mask, _, _ = gather_video(data, e, f % per, k, device)
        source = data['source'][e].to(device)
        dyn = dynamic_mask(props, context, mask)
        cur = states[:, 0]
        for j in range(1, k + 1):
            with torch.autocast('cuda', dtype=amp_dtype or torch.float16, enabled=amp_dtype is not None and device.type == 'cuda'):
                cur = model(cur, props, context, mask)
            cur = torch.nan_to_num(cur.float(), nan=1e3, posinf=1e3, neginf=-1e3).clamp(-1e3, 1e3)
            seen = (obs[:, j] & dyn).float()
            d = (cur[..., :3] - states[:, j, :, :3]).norm(dim=-1) * seen
            err[j] += d.sum()
            hold[j] += ((states[:, 0, :, :3] - states[:, j, :, :3]).norm(dim=-1) * seen).sum()
            seen_n[j] += seen.sum()
            if j <= short:
                src_err.index_add_(0, source, d.sum(-1))
                src_n.index_add_(0, source, seen.sum(-1))
                depth = cg.penetration_depths(cur, props, context, mask)
                worst = torch.maximum(torch.maximum(depth['support'], depth['floor']), depth['objects'])
                pen_sum += float((torch.nan_to_num(worst, nan=1.) * dyn).sum())
                pen_n += float(dyn.sum())

    def ratio(a, b):
        return float(a / b) if float(b) > 0 else float('nan')
    out = {'video_val_windows': int(len(idx)), 'video_val_horizon': short,
           'video_val_pos_err_k': ratio(err[1:short + 1].sum(), seen_n[1:short + 1].sum()),
           'video_val_hold_err_k': ratio(hold[1:short + 1].sum(), seen_n[1:short + 1].sum()),
           'video_val_penetration': pen_sum / max(pen_n, 1)}
    for j, name in ((10, '0.5s'), (20, '1s')):
        if j <= k:
            out[f'video_val_pos_err_{name}'] = ratio(err[j], seen_n[j])
            out[f'video_val_hold_err_{name}'] = ratio(hold[j], seen_n[j])
    for i, name in enumerate(names):
        out[f'video_val_pos_err_k/{name}'] = ratio(src_err[i], src_n[i])
    return out


# ----------------------------------------------------------------------------- io
def atomic_save(obj, path: Path):
    tmp = path.with_name(path.name + f'.tmp{os.getpid()}')
    torch.save(obj, tmp)
    os.replace(tmp, path)


def jsonable(d):
    return {k: (str(v) if isinstance(v, Path) else v) for k, v in d.items()}


def clean(d):
    return {k: (None if isinstance(v, float) and not math.isfinite(v) else v) for k, v in d.items()}


def shipped(model, config, epoch, args, metrics):
    return {'kind': 'contact_gnn', 'config': dict(config),
            'state_dict': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            'model_step_s': .05, 'epoch': int(epoch), 'train_args': jsonable(vars(args)), 'metrics': clean(metrics)}


class GpuMonitor(threading.Thread):
    def __init__(self, path: Path, interval=30.):
        super().__init__(daemon=True)
        self.path, self.interval, self.stop_flag, self.samples = path, interval, threading.Event(), []
        visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        self.visible = {int(x) for x in visible.split(',') if x.strip().isdigit()} or None

    def run(self):
        while not self.stop_flag.wait(self.interval):
            try:
                text = subprocess.run(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used', '--format=csv,noheader,nounits'],
                                      capture_output=True, text=True, timeout=20).stdout
                rows = [[int(float(x)) for x in line.split(',')] for line in text.strip().splitlines() if line.strip()]
                rows = [r for r in rows if self.visible is None or r[0] in self.visible]
                self.samples.append(rows)
                with open(self.path, 'a') as f:
                    f.write(json.dumps(dict(time=time.time(), gpus=[dict(index=r[0], util=r[1], mem_mb=r[2]) for r in rows])) + '\n')
            except Exception as error:  # monitoring must never kill training
                log(f'[gpu-monitor] {error}')

    def summary(self):
        samples, self.samples = self.samples, []
        if not samples:
            return 'gpu util: no samples yet'
        util, mem = {}, {}
        for rows in samples:
            for i, u, m in rows:
                util.setdefault(i, []).append(u)
                mem[i] = max(mem.get(i, 0), m)
        per = ' '.join(f'{i}:{sum(v) / len(v):.0f}%/{mem[i] / 1024:.1f}G' for i, v in sorted(util.items()))
        mean = sum(sum(v) / len(v) for v in util.values()) / len(util)
        return f'gpu util mean {mean:.0f}% over {len(samples)} samples [{per}]'


# ----------------------------------------------------------------------------- main
def main():
    args = parse_args()
    distributed = 'LOCAL_RANK' in os.environ and int(os.environ.get('WORLD_SIZE', '1')) >= 1 and 'RANK' in os.environ
    rank = int(os.environ.get('RANK', 0)) if distributed else 0
    world = int(os.environ.get('WORLD_SIZE', 1)) if distributed else 1
    local = int(os.environ.get('LOCAL_RANK', 0)) if distributed else 0
    cuda = torch.cuda.is_available()
    # more ranks than GPUs is allowed (ranks share GPUs round-robin): the model is dispatch-bound at
    # batch 1024, so e.g. --nproc_per_node=14 on 7 GPUs raises utilization and total windows/sec
    gpu = local % torch.cuda.device_count() if cuda else 0
    device = torch.device(f'cuda:{gpu}' if cuda else 'cpu')
    torch.set_num_threads(max(1, min(4, (os.cpu_count() or 4) // max(world, 1))))
    if cuda:
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    backend = 'single'
    if distributed:
        backend = args.dist_backend
        if backend == 'auto':   # NCCL cannot place two ranks on one GPU; gloo is fine for a <1M-parameter model
            backend = 'nccl' if cuda and world <= torch.cuda.device_count() else 'gloo'
        if not args.nccl_p2p:
            # On this server NCCL's default P2P/CUMEM transport hangs in the first collective
            # (verified with a 4-float all_reduce); the shared-memory transport works.
            os.environ.setdefault('NCCL_P2P_DISABLE', '1')
        dist.init_process_group(backend)
    main_rank = rank == 0
    torch.manual_seed(args.seed * 1000 + rank)
    np.random.seed(args.seed * 1000 + rank)
    start_wall = time.time()
    if main_rank:
        args.out.mkdir(parents=True, exist_ok=True)
        log(f'[setup] world={world} backend={backend} gpus={torch.cuda.device_count() if cuda else 0} device={device} torch={torch.__version__} out={args.out}')

    amp_dtype, use_scaler, amp_note = choose_amp(args.amp, device)
    if main_rank:
        log(f'[amp] {amp_note}')

    # ---- data
    t = time.time()
    train_np = load_split(args.data, 'train', args.max_train_episodes, args.load_threads)
    nbytes = sum(v.nbytes for v in train_np.values())
    data_device = device if args.data_device in ('auto', 'cuda') and cuda else torch.device('cpu')
    if args.data_device == 'auto' and cuda and nbytes > .5 * torch.cuda.get_device_properties(device).total_memory:
        data_device = torch.device('cpu')
    train = to_device(train_np, data_device)
    del train_np
    E, T = train['states'].shape[0], train['states'].shape[1] - 1
    horizon = min(args.horizon, T)
    if main_rank:
        log(f'[data] train episodes={E} T={T} slots={train["states"].shape[2]} {nbytes / 2**30:.2f} GiB on {data_device} '
            f'loaded in {time.time() - t:.1f}s; active frames {float(train["active"].float().mean()):.3f}')
        val = to_device(load_split(args.data, 'val', args.max_val_episodes, args.load_threads, need_meta=True), device)
        log(f'[data] val episodes={val["states"].shape[0]} chaotic={int(val["chaotic"].sum())}')
    # ---- video-derived data (opt-in)
    if not 0 <= args.video_frac < 1:
        raise SystemExit('--video-frac must be in [0, 1)')
    if args.video_frac > 0 and not args.video_data:
        raise SystemExit('--video-frac > 0 needs --video-data')
    video_batch = int(round(args.batch * args.video_frac))
    if args.video_frac > 0 and not 0 < video_batch < args.batch:
        raise SystemExit(f'--video-frac {args.video_frac} of --batch {args.batch} gives {video_batch} video windows; need 1..batch-1')
    vtrain = vsampler = vval = None
    vnames = []
    if video_batch:
        vtrain_np, vnames_train = load_video_split(args.video_data, 'train', args.load_threads, train['states'].shape[2])  # sim slot count: shared model calls
        if vtrain_np is None:
            raise SystemExit(f'no train shards under {args.video_data}')
        vtrain = to_device(vtrain_np, data_device)
        del vtrain_np
        vsampler = VideoSampler(vtrain, args.seed, rank, world)
        if main_rank:
            vT = vtrain['states'].shape[1] - 1
            log(f'[video] train sources {vnames_train}: episodes={vtrain["states"].shape[0]} T={vT} slots={vtrain["states"].shape[2]}; '
                f'valid start frames {float(vtrain["start_ok"].float().mean()):.3f}; median pos_sigma {float(vtrain["sigma"][vtrain["mask"]].median()):.4f} m; '
                f'objects with orientation {float(vtrain["ori"][vtrain["mask"]].float().mean()):.2f}')
            log(f'[video] {video_batch} of {args.batch} windows per rank per update are video windows; total loss = sim + {args.video_weight} * video '
                f'(position-only Huber at observed frames + predicted-state penetration + {args.video_w_rot} * orientation where flagged); '
                f'video horizon {args.video_horizon or "follows the sim k"}; selection score is the SIM score only')
            if vtrain['states'].shape[2] != train['states'].shape[2] or any(vtrain[key].shape[-1] != train[key].shape[-1] for key in ('props', 'context')):
                log('[video] NOTE slot count or props/context width differs from the sim data: video windows get their own unroll (about half the throughput)')
            elif args.video_horizon:
                log('[video] NOTE with --video-horizon different from the sim k the video windows get their own unroll (about half the throughput)')
    if args.video_data and main_rank:
        vval_np, vnames = load_video_split(args.video_data, 'val', args.load_threads)
        if vval_np is not None:
            vval = to_device(vval_np, device)
            log(f'[video] val sources {vnames}: episodes={vval["states"].shape[0]} (reported only, never scored)')
            del vval_np
        else:
            log('[video] no val split under --video-data: no video validation')
    events = event_frames(train) if args.event_frac > 0 else None
    sampler = WindowSampler(train.pop('active'), args.seed, rank, world, events, args.event_frac)
    if main_rank and events is not None:
        log(f'[data] event frames (motion onset / fall approach) {float(events.float().mean()):.3f}; --event-frac {args.event_frac}')

    # ---- model
    last_path = args.out / 'last.pt'
    init = None
    if args.init is not None and not (args.resume and last_path.exists()):
        init = torch.load(args.init, map_location='cpu', weights_only=True)
        for key, value in (init.get('config') or {}).items():      # architecture follows the checkpoint
            if key in ('width', 'rounds', 'edge_radius'):
                setattr(args, key, value)
    model = cg.ContactGNN(width=args.width, rounds=args.rounds, edge_radius=args.edge_radius).to(device)
    config = dict(getattr(model, 'config', None) or dict(width=args.width, rounds=args.rounds, edge_radius=args.edge_radius))
    updates_per_epoch = max(1, args.windows_per_epoch // (args.batch * world))
    total_updates = updates_per_epoch * args.epochs
    requested_warmup = args.warmup
    args.warmup = max(1, min(args.warmup, int(.05 * total_updates)))
    resume = None
    if args.resume and last_path.exists():
        resume = torch.load(last_path, map_location='cpu', weights_only=False)
        model.load_state_dict(resume['model'])
    elif init is not None:
        model.load_state_dict(init['state_dict'])   # includes the normalizer buffers; the EMA copy starts from the same weights
        if main_rank:
            log(f'[init] warm start from {args.init} (epoch {init.get("epoch")}, config {init.get("config")}); normalizer from the checkpoint; '
                f'fresh optimizer/schedule; start horizon {max(args.start_horizon, 1) if args.start_horizon else "curriculum"}')
        del init
    elif main_rank:
        mean, std, n = fit_normalizer(model, train, device, args.seed)
        log(f'[normalizer] {n} dynamic transitions\n  mean={[round(float(x), 4) for x in mean]}\n  std ={[round(float(x), 4) for x in std]}')
    ema = copy.deepcopy(model).eval().requires_grad_(False) if main_rank else None
    if resume is not None and main_rank:
        ema.load_state_dict(resume['ema'])
    trainee = Trainee(model, args, amp_dtype)
    if args.compile:
        trainee = torch.compile(trainee)
    if distributed:
        # rank 0's parameters and buffers (the normalizer) go to every rank explicitly: DDP with
        # broadcast_buffers=False does not sync buffers at construction
        with torch.no_grad():
            for tensor in model.state_dict().values():
                dist.broadcast(tensor, 0)
        net = nn.parallel.DistributedDataParallel(trainee, device_ids=[gpu] if cuda else None, broadcast_buffers=False,  # buffers are still synced once at construction
                                                 
                                                  find_unused_parameters=bool(args.find_unused))
        check = torch.cat((model.out_mean.float(), model.out_std.float(), torch.stack([p.float().sum() for p in model.parameters()]).sum()[None]))
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
    if main_rank:
        log(f'[model] ContactGNN {config} params={params:,}; updates/epoch={updates_per_epoch} total={total_updates} '
            f'batch={args.batch}x{world} lr={args.lr} (not scaled by ranks) bptt={args.bptt} horizon={horizon}')
        log(f'[schedule] warmup {args.warmup} updates ({100 * args.warmup / max(total_updates, 1):.1f}% of total'
            f'{f"; capped from {requested_warmup}" if args.warmup != requested_warmup else ""}); cosine to lr_min={args.lr_min}; '
            f'horizon k=1/2/4/{horizon} from 0/5/15/35% of the schedule'
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
    vslot = width                                          # video statistics are appended only when video windows are in use
    if video_batch:
        width += len(VIDEO_TERMS)
    progress = 0.                                          # elapsed/budget, shared by all ranks at log intervals

    for epoch in range(start_epoch, args.epochs):
        net.train()
        gen.manual_seed(args.seed * 1_000_003 + epoch * 7919 + rank * 104_729 + 17)
        accum = torch.zeros(width, device=device)
        epoch_stats = torch.zeros(len(TERMS), device=device)
        epoch_vstats = torch.zeros(len(VIDEO_TERMS), device=device)
        epoch_logs = 0
        epoch_start = interval_start = time.time()
        interval_updates = 0
        k = horizon_at(global_step, total_updates, horizon, progress, args.start_horizon)
        for update in range(updates_per_epoch):
            k = horizon_at(global_step, total_updates, horizon, progress, args.start_horizon)
            sampler.set(epoch, k)
            lr = lr_at(global_step, total_updates, args, progress)
            for group in optimizer.param_groups:
                group['lr'] = lr
            e, t0 = sampler.next(args.batch - video_batch)
            batch = gather(train, e, t0, k, device, args.rest_lookahead)
            if video_batch:
                kv = min(args.video_horizon or k, vtrain['states'].shape[1] - 1)
                vsampler.set(epoch, kv)
                ve, vt = vsampler.next(video_batch)
                loss, stats, bad, vstats = net(*batch, k, gen, gather_video(vtrain, ve, vt, kv, device) + (kv, ))
            else:
                loss, stats, bad = net(*batch, k, gen)
            optimizer.zero_grad(set_to_none=True)
            if bool(bad):
                # keep the DDP reduction in lock-step: this rank contributes zero gradients
                loss = sum(p.sum() for p in model.parameters()) * 0.
                accum[len(TERMS)] += 1
            else:
                accum[:len(TERMS)] += stats
                if video_batch:
                    accum[vslot:] += vstats
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
                    wps = interval_updates * args.batch * world / max(elapsed, 1e-9)
                    dropped_total += int(values[len(TERMS)])
                    skipped_total += int(values[len(TERMS) + 1] / world)
                    row = dict(step=global_step, epoch=epoch + 1, k=k, lr=lr, windows_per_sec=wps, windows_per_sec_per_gpu=wps / world,
                               dropped_batches=dropped_total, skipped_updates=skipped_total, minutes=(time.time() - start_wall) / 60,
                               **{name: values[i] / good for i, name in enumerate(TERMS)})
                    if video_batch:
                        row.update({name: values[vslot + i] / good for i, name in enumerate(VIDEO_TERMS)}, video_k=kv)
                        epoch_vstats += accum[vslot:] / good
                    log_file.write(json.dumps(clean(row)) + '\n')
                    log_file.flush()
                    epoch_stats += accum[:len(TERMS)] / good
                    epoch_logs += 1
                    if epoch_logs % 5 == 1:
                        log(f'  e{epoch + 1} u{update + 1}/{updates_per_epoch} k={k} loss={row["loss"]:.4f} one={row["one"]:.4f} '
                            f'pos={row["pos"]:.4f} pen={row["pen"]:.4f} rest={row["rest"]:.4f} slide={row["slide"]:.4f} lr={lr:.2e} '
                            f'{wps:,.0f} win/s ({wps / world:,.0f}/gpu) dropped={dropped_total}'
                            + (f' | video k={kv} loss={row["video_loss"]:.4f} pos={row["video_pos"]:.4f} pen={row["video_pen"]:.4f} '
                               f'rot={row["video_rot"]:.4f} err={row["video_err_m"] * 100:.2f}cm' if video_batch else ''))
                accum.zero_()
                interval_updates = 0
                interval_start = time.time()
                if stop:
                    break

        if main_rank:
            train_time = time.time() - epoch_start
            if ema is not None:      # buffers (normalizer) follow the live model
                for be, bm in zip(ema.buffers(), model.buffers()):
                    be.copy_(bm)
            t = time.time()
            metrics = validate(ema, val, device, args, amp_dtype)
            metrics.update(epoch=epoch + 1, step=global_step, k=k, train_seconds=train_time, val_seconds=time.time() - t,
                           **{f'train_{name}': float(v) / max(epoch_logs, 1) for name, v in zip(TERMS, epoch_stats)})
            if video_batch:
                metrics.update({f'train_{name}': float(v) / max(epoch_logs, 1) for name, v in zip(VIDEO_TERMS, epoch_vstats)})
            if vval is not None:     # reported only; metrics['score'] (the selection score) is the simulator score, untouched
                metrics.update(validate_video(ema, vval, vnames, device, args, amp_dtype))
            improved = metrics['score'] < best
            best = min(best, metrics['score'])
            ship = shipped(ema, config, epoch + 1, args, metrics)
            atomic_save(ship, args.out / f'epoch_{epoch + 1:03d}.pt')
            if improved:
                atomic_save(ship, args.out / 'best.pt')
            atomic_save(dict(model=model.state_dict(), ema=ema.state_dict(), optimizer=optimizer.state_dict(), scheduler=dict(global_step=global_step, total_updates=total_updates),
                             scaler=scaler.state_dict(), epoch=epoch + 1, global_step=global_step, best=best, dropped=dropped_total, skipped=skipped_total,
                             config=config, args=jsonable(vars(args)), rng=dict(torch=torch.get_rng_state(), noise=gen.get_state().cpu())), last_path)
            log_file.write(json.dumps(clean(dict(event='epoch', improved=improved, **metrics))) + '\n')
            log_file.flush()
            log(f'[epoch {epoch + 1}/{args.epochs}] k={k} train={metrics["train_loss"]:.4f} val_k{horizon}={metrics["val_loss"]:.4f} '
                f'pos1s c/g={metrics.get("pos_err_1s_chaotic", float("nan")):.4f}/{metrics.get("pos_err_1s_gentle", float("nan")):.4f} '
                f'pos2s c/g={metrics.get("pos_err_2s_chaotic", float("nan")):.4f}/{metrics.get("pos_err_2s_gentle", float("nan")):.4f} '
                f'pen={metrics["mean_penetration"] * 1000:.2f}mm drift={metrics["rest_drift"]:.4f} '
                f'fall P/R={metrics["fall_precision"]:.3f}/{metrics["fall_recall"]:.3f} (on {metrics["fall_precision_on"]:.3f}/{metrics["fall_recall_on"]:.3f}) '
                f'score={metrics["score"]:.4f}'
                f'{" *best*" if improved else ""} train {train_time:.0f}s val {metrics["val_seconds"]:.0f}s')
            if video_batch or vval is not None:
                def cm(key):
                    v = metrics.get(key)
                    return 'n/a' if v is None or v != v else f'{v * 100:.2f}cm'
                log(f'[epoch {epoch + 1}] video (not scored): train loss={metrics.get("train_video_loss", float("nan")):.4f} err={cm("train_video_err_m")}; '
                    f'val windows={metrics.get("video_val_windows", 0)} pos err k{metrics.get("video_val_horizon", 0)}={cm("video_val_pos_err_k")} '
                    f'(hold {cm("video_val_hold_err_k")}) 0.5s={cm("video_val_pos_err_0.5s")} 1s={cm("video_val_pos_err_1s")} (hold {cm("video_val_hold_err_1s")}) '
                    f'pen={metrics.get("video_val_penetration", float("nan")) * 1000:.2f}mm')
            if monitor:
                log(f'[epoch {epoch + 1}] {monitor.summary()}')
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
