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

TERMS = ('loss', 'one', 'pos', 'vel', 'ang', 'rot', 'pen', 'rest', 'slide')


# ----------------------------------------------------------------------------- args
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--resume', action='store_true', help='continue from <out>/last.pt if present')
    p.add_argument('--init', type=Path, default=None, help='warm-start weights from a shipped-format checkpoint')
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

    def __init__(self, active_frames: torch.Tensor, seed: int, rank: int, world: int):
        self.frames = active_frames                       # [E,T+1] bool on the data device
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

    def next(self, batch: int):
        device = self.frames.device
        nu = batch // 2
        na = batch - nu
        u = self.uniform[(self.cu + torch.arange(nu, device=device)) % len(self.uniform)]
        a = self.active[(self.ca + torch.arange(na, device=device)) % len(self.active)]
        self.cu += nu
        self.ca += na
        flat = torch.cat((u, a))
        return flat // self.per, flat % self.per


def gather(data, e, t0, k, device):
    """Windows [B,k+1,...] for episodes e starting at t0."""
    t = t0[:, None] + torch.arange(k + 1, device=t0.device)
    out = (data['states'][e[:, None], t], data['rest'][e[:, None], t], data['props'][e], data['context'][e], data['mask'][e])
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


def huber(x):
    return F.huber_loss(x, torch.zeros_like(x), reduction='none')


def unroll(model, states, rest, props, context, mask, k, args, amp_dtype, gen=None):
    """k-step autoregressive loss. Returns (loss, stats[len(TERMS)], bad flag tensor)."""
    dyn = dynamic_mask(props, context, mask)
    count = dyn.sum().clamp_min(1)

    def avg(x):
        return (x * dyn).sum() / count
    cur = states[:, 0]
    if gen is not None:
        cur = add_noise(cur, rest[:, 0], dyn, args, gen)
    mean, std = model.out_mean.float(), model.out_std.float()
    weights = [args.step_decay ** j for j in range(k)]
    total = states.new_zeros(())
    stats = states.new_zeros(len(TERMS))
    bad = torch.zeros((), dtype=torch.bool, device=states.device)
    use_amp = amp_dtype is not None and states.is_cuda
    for j in range(k):
        true = states[:, j + 1]
        with torch.autocast('cuda', dtype=amp_dtype or torch.float16, enabled=use_amp):
            raw = model.predict_raw(cur, props, context, mask)
        raw = raw.float()
        with torch.no_grad():
            # pushforward-style target: from the model's CURRENT (possibly drifted) input to the TRUE next state
            target = (model.raw_targets(cur.detach(), true, props, context, mask).float() - mean) / std
        nxt = model.integrate(cur, raw * std + mean, props, context, mask)
        terms = dict(
            one=avg(huber(raw - target).mean(-1)),
            pos=avg(huber((nxt[..., :3] - true[..., :3]) / .02).mean(-1)),
            vel=avg(huber((nxt[..., 7:10] - true[..., 7:10]) / .2).mean(-1)),
            ang=avg(huber((nxt[..., 10:13] - true[..., 10:13]) / 2).mean(-1)),
            rot=avg(huber(geodesic(nxt[..., 3:7], true[..., 3:7]) / .1)))
        # Slide anchor for objects truly at rest: while the input is within 1 cm of the truth (always the
        # case for the noised j=0 input) the object must end on the TRUE resting pose, so the ground truth
        # scores zero and correcting noise/drift is never punished; a further-drifted object is only asked
        # not to move relative to its own previous pose (bounded, as before).
        near = (cur.detach()[..., :3] - states[:, j, :, :3]).norm(dim=-1, keepdim=True) < .01
        phys = cg.physics_losses(nxt, true, torch.where(near, true, cur), props, context, mask, rest[:, j + 1])
        terms.update(pen=phys['penetration'], rest=phys['rest'], slide=phys['slide'])
        w_one = args.w_one if not args.one_steps or j < args.one_steps else 0.
        step = (w_one * terms['one'] + args.w_pos * terms['pos'] + args.w_vel * terms['vel'] + args.w_ang * terms['ang']
                + args.w_rot * terms['rot'] + args.w_pen * terms['pen'] + args.w_rest * terms['rest'] + args.w_slide * terms['slide'])
        total = total + weights[j] * step
        stats = stats + weights[j] * torch.stack([step.detach()] + [terms[n].detach().float() for n in TERMS[1:]])
        bad = bad | ~torch.isfinite(nxt).all() | (nxt[..., :3].abs() > 100).any()
        cur = nxt if args.bptt else nxt.detach()
    norm = sum(weights)
    return total / norm, stats / norm, bad | ~torch.isfinite(total)


class Trainee(nn.Module):
    """Whole unroll inside forward() so DistributedDataParallel sees one forward per update."""

    def __init__(self, model, args, amp_dtype):
        super().__init__()
        self.model, self.args, self.amp_dtype = model, args, amp_dtype

    def forward(self, states, rest, props, context, mask, k, gen):
        return unroll(self.model, states, rest, props, context, mask, k, self.args, self.amp_dtype, gen)


# ----------------------------------------------------------------------------- schedule
def horizon_at(step, total, horizon, progress=0.):
    f = max(step / max(total, 1), progress)
    k = 1 if f < .05 else 2 if f < .15 else 4 if f < .35 else horizon
    return min(k, horizon)


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
    for j in range(1, steps + 1):
        with torch.autocast('cuda', dtype=amp_dtype or torch.float16, enabled=amp_dtype is not None and device.type == 'cuda'):
            cur = model(cur, props, context, mask)
        cur = torch.nan_to_num(cur.float(), nan=1e3, posinf=1e3, neginf=-1e3).clamp(-1e3, 1e3)
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
    sampler = WindowSampler(train.pop('active'), args.seed, rank, world)

    # ---- model
    model = cg.ContactGNN(width=args.width, rounds=args.rounds, edge_radius=args.edge_radius).to(device)
    config = dict(getattr(model, 'config', None) or dict(width=args.width, rounds=args.rounds, edge_radius=args.edge_radius))
    updates_per_epoch = max(1, args.windows_per_epoch // (args.batch * world))
    total_updates = updates_per_epoch * args.epochs
    requested_warmup = args.warmup
    args.warmup = max(1, min(args.warmup, int(.05 * total_updates)))
    last_path = args.out / 'last.pt'
    resume = None
    if args.resume and last_path.exists():
        resume = torch.load(last_path, map_location='cpu', weights_only=False)
        model.load_state_dict(resume['model'])
    elif args.init is not None:
        model.load_state_dict(torch.load(args.init, map_location='cpu', weights_only=True)['state_dict'])
        if main_rank:
            log(f'[init] warm start from {args.init}')
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
    progress = 0.                                          # elapsed/budget, shared by all ranks at log intervals

    for epoch in range(start_epoch, args.epochs):
        net.train()
        gen.manual_seed(args.seed * 1_000_003 + epoch * 7919 + rank * 104_729 + 17)
        accum = torch.zeros(width, device=device)
        epoch_stats = torch.zeros(len(TERMS), device=device)
        epoch_logs = 0
        epoch_start = interval_start = time.time()
        interval_updates = 0
        k = horizon_at(global_step, total_updates, horizon, progress)
        for update in range(updates_per_epoch):
            k = horizon_at(global_step, total_updates, horizon, progress)
            sampler.set(epoch, k)
            lr = lr_at(global_step, total_updates, args, progress)
            for group in optimizer.param_groups:
                group['lr'] = lr
            e, t0 = sampler.next(args.batch)
            batch = gather(train, e, t0, k, device)
            loss, stats, bad = net(*batch, k, gen)
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
                    wps = interval_updates * args.batch * world / max(elapsed, 1e-9)
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
                        log(f'  e{epoch + 1} u{update + 1}/{updates_per_epoch} k={k} loss={row["loss"]:.4f} one={row["one"]:.4f} '
                            f'pos={row["pos"]:.4f} pen={row["pen"]:.4f} rest={row["rest"]:.4f} slide={row["slide"]:.4f} lr={lr:.2e} '
                            f'{wps:,.0f} win/s ({wps / world:,.0f}/gpu) dropped={dropped_total}')
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
                f'pen={metrics["mean_penetration"] * 1000:.2f}mm drift={metrics["rest_drift"]:.4f} score={metrics["score"]:.4f}'
                f'{" *best*" if improved else ""} train {train_time:.0f}s val {metrics["val_seconds"]:.0f}s')
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
