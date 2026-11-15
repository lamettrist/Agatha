"""DDP trainer for agatha.joint_gnn.JointGNN (EXPERIMENTAL articulated-body track).

A copy of scripts/train_contact_ddp.py (same DDP / AMP / EMA / checkpoint / logging / --max-minutes /
--init machinery and the same fall-recall loss flags) adapted to the joint data format
(data/joints_v1: box-format keys with N=12 slots plus joints [E,J,18], joint_mask, joint_pos, group).

Multi-GPU:
  CUDA_VISIBLE_DEVICES=0,1,2,3 .venv-train/bin/torchrun --nproc_per_node=4 \
      scripts/train_joint_ddp.py --data data/joints_v1 --out models/runs/NAME --mix-box-data data/v3
Single process (one GPU or CPU):
  CUDA_VISIBLE_DEVICES=7 .venv-train/bin/python scripts/train_joint_ddp.py --data data/joints_v1 --out models/runs/NAME

Differences from the box trainer
- joints / joint_mask are plumbed through window sampling, the k-step unroll and validation.
- Warm start: --init-contact <contact_gnn checkpoint> (default models/contact_gnn.pt when it exists) through
  joint_gnn.init_from_contact_gnn (weights + normalizer of the box model; a zero-joint scene then reproduces
  ContactGNN). --init <joint_gnn checkpoint> and --resume take precedence. --init-contact none = from scratch.
- Extra loss terms from joint_gnn.joint_losses: --w-anchor 2.0 --w-limit 1.0 --w-axis 1.0 (TERMS janchor/jlimit/jaxis).
- The penetration penalty ignores pairs of bodies that are directly connected by a joint.
- --mix-box-data DIR --mix-box-frac 0.3: that fraction of every batch is plain box windows (slots padded to the
  joint data's N, no joints) so the model does not forget box dynamics; box validation metrics are logged as box_*.
- Validation rollouts additionally report the mean joint anchor gap (mm), hinge/prismatic off-axis error (deg)
  and joint coordinate error (rad for hinges, m for prismatic) at 1 s / 2 s; the selection score adds
  2*anchor_gap_2s(m) + 0.05*hinge coordinate error 2s (rad) (+ 0.5 * the box score when box data is mixed).
- Box-regression guards (all off by default; see the flags):
  --inherit-lr-scale / --inherit-freeze-updates: separate LR for the weights inherited from contact_gnn (its 4 rounds,
  encoders, decoder) against the new modules (joint encoder / updates, world node, extra rounds);
  --w-distill: on windows without joints the raw output is also pulled to the frozen contact_gnn teacher's raw output
  for the same input (TERM distill); --box-guard: a checkpoint whose box validation score is more than this fraction
  above the teacher's own score on the same rollouts gets +1000 on its selection score (never best.pt while any
  other checkpoint passes).
- Joint accuracy (off by default): --w-jcoord / --w-jrel (joint_gnn.joint_tracking_losses: hinge angle / slider
  position and child-in-parent pose against the true next state), --w-jmove (weight of joints that move in the
  window in those two terms), --joint-one-delta (Huber delta of the one-step term for bodies of an articulation:
  their targets are 2.5x the box normaliser's sigma, see the report in the docstring of that flag).
- The contact_gnn checkpoint of the warm start is copied to <out>/init_contact.pt (models/contact_gnn.pt may be replaced
  later): evaluate the box regression against THAT file (scripts/eval_joints.py --contact <out>/init_contact.pt).
- Checkpoints: epoch_XXX.pt / best.pt in the shipped format {'kind': 'joint_gnn', 'config', 'state_dict', ...}
  (load with joint_gnn.build_from_checkpoint), last.pt for --resume.

Notes (unchanged from the box trainer)
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
import importlib
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
from agatha.dynamics import euler_quaternion, normalize_q, rotation

# AGATHA_JOINT_GNN_MODULE is a development hook only (interface stub before agatha/joint_gnn.py existed)
jg = importlib.import_module(os.environ.get('AGATHA_JOINT_GNN_MODULE', 'agatha.joint_gnn'))

TERMS = ('loss', 'one', 'pos', 'vel', 'ang', 'rot', 'pen', 'rest', 'slide', 'janchor', 'jlimit', 'jaxis', 'jcoord', 'jrel', 'distill')
JOINT_DIM = 18
DEFAULT_CONTACT = Path(__file__).resolve().parents[1] / 'models' / 'contact_gnn.pt'


# ----------------------------------------------------------------------------- args
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--resume', action='store_true', help='continue from <out>/last.pt if present')
    p.add_argument('--init', type=Path, default=None,
                   help='warm-start model + EMA weights, normalizer and model config from a shipped-format joint_gnn checkpoint '
                        '(fresh optimizer and schedule); ignored when --resume finds <out>/last.pt')
    p.add_argument('--init-contact', type=str, default='auto',
                   help="contact_gnn checkpoint for joint_gnn.init_from_contact_gnn (width / edge radius / normalizer follow it); "
                        "'auto' = models/contact_gnn.pt when it exists, 'none' = train from scratch; ignored with --init or a resumed run")
    p.add_argument('--start-horizon', type=int, default=0,
                   help='lower bound on the curriculum horizon k (0 = normal 1/2/4/H curriculum); set to --horizon when warm-starting')
    # model
    p.add_argument('--width', type=int, default=128)
    p.add_argument('--rounds', type=int, default=6, help='message passing rounds (JointGNN default 6 for chains)')
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
    p.add_argument('--w-anchor', type=float, default=2.0, help='weight of joint_losses()["anchor"]')
    p.add_argument('--w-limit', type=float, default=1.0, help='weight of joint_losses()["limit"]')
    p.add_argument('--w-axis', type=float, default=1.0, help='weight of joint_losses()["axis"]')
    p.add_argument('--w-jcoord', type=float, default=0.0, help='weight of joint_tracking_losses()["coord"]: hinge angle (0.05 rad scale) / slider (5 mm) error (0 = off)')
    p.add_argument('--w-jrel', type=float, default=0.0, help='weight of joint_tracking_losses()["relative"]: child pose in the parent frame (10 mm / 0.1 rad) (0 = off)')
    p.add_argument('--w-jmove', type=float, default=1.0,
                   help='TRAIN-ONLY weight in the jcoord / jrel terms of joints that truly move within the window (relative rotation > 0.01 rad or slide > 1 mm); 1 = off')
    p.add_argument('--joint-one-delta', type=float, default=0.0,
                   help='TRAIN-ONLY Huber delta (target sigmas) of the one-step term for bodies that belong to an articulation (0 = --one-delta). '
                        'Measured on data/joints_v1 with the inherited box normaliser: articulated bodies have rms normalised targets of 2.5-2.7 '
                        '(boxes 1.0-1.1), 14%% / 9%% of their dv / dw components beyond 1 sigma (boxes 3%% / 1%%), p99 10 sigma; '
                        'chain and ragdoll are the extreme (rms 2.8-3.9).')
    # round 2 (defaults reproduce round 1 exactly)
    p.add_argument('--joint-rest', type=int, default=0,
                   help='TRAIN-ONLY, 1 = the rest / slide penalties also cover bodies of an articulation that are truly slow (|v| < 0.01 m/s, '
                        '|w| < 0.05 rad/s) on BOTH frames of a step WITHOUT being in contact (the dataset rest flag needs a contact, so a lid '
                        'on its stop or an arm link held by its joints was never asked to stand still: measured resting-hinge creep)')
    p.add_argument('--grow-rounds', type=int, default=0,
                   help='with --init: append this many message-passing rounds to the checkpoint model. They start as the identity (final '
                        'LayerNorm zeroed) and, with shared_rounds > 0, are gated to scenes with a joint like the other extra rounds, so the '
                        'start is exactly the checkpoint and joint-free scenes keep the box computation')
    p.add_argument('--inherit-lr-scale', type=float, default=1.0,
                   help='LR multiplier for the weights inherited from the contact_gnn warm start (new joint modules and extra rounds use --lr); 0 = frozen')
    p.add_argument('--inherit-freeze-updates', type=int, default=0, help='inherited weights get LR 0 for this many updates (then --inherit-lr-scale)')
    p.add_argument('--inherited-rounds', type=int, default=-1,
                   help='rounds of the contact_gnn the model descends from (-1 = from the warm start / checkpoint; 0 = nothing is inherited)')
    p.add_argument('--gate-extra-rounds', type=int, default=0,
                   help='1 = with a contact_gnn warm start, the rounds beyond the inherited ones only update scenes that contain a joint '
                        '(JointGNN shared_rounds): joint-free scenes are computed by the inherited rounds alone; with --inherit-lr-scale 0 '
                        'the box model is then reproduced exactly')
    p.add_argument('--drop-jointless-train', type=int, default=0,
                   help='1 = remove the joint-free (plain_*) episodes of --data from the TRAINING set (validation keeps them). With '
                        '--gate-extra-rounds 1 --inherit-lr-scale 0 such windows reach no trainable weight')
    p.add_argument('--w-distill', type=float, default=0.0,
                   help='TRAIN-ONLY weight of huber(raw - teacher raw) on windows without joints (teacher = frozen --teacher on the same input) (0 = off)')
    p.add_argument('--teacher', type=str, default='auto',
                   help="contact_gnn checkpoint for --w-distill and --box-guard; 'auto' = the --init-contact checkpoint, else models/contact_gnn.pt")
    p.add_argument('--box-guard', type=float, default=0.0,
                   help='with --mix-box-data: a checkpoint whose box validation score exceeds (1 + this) x the teacher score on the same rollouts '
                        'gets +1000 on the selection score (0 = off)')
    p.add_argument('--box-score-weight', type=float, default=0.5, help='selection score += this * box validation score (with --mix-box-data)')
    p.add_argument('--mix-box-data', type=Path, default=None,
                   help='box-format dataset (e.g. data/v3) mixed into every batch with no joints, slots padded to the joint data')
    p.add_argument('--mix-box-frac', type=float, default=0.3, help='fraction of every batch drawn from --mix-box-data')
    p.add_argument('--max-box-episodes', type=int, default=0, help='cap on --mix-box-data train episodes (0 = all)')
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
    p.add_argument('--max-gpu-gb', type=float, default=0, help='hard cap on the CUDA memory of EACH process (0 = none); for shared GPUs')
    return p.parse_args()


def log(*a):
    print(*a, flush=True)


# ----------------------------------------------------------------------------- data
def _pad(array, axis, size):
    if array.shape[axis] >= size:
        return array
    widths = [(0, 0)] * array.ndim
    widths[axis] = (0, size - array.shape[axis])
    return np.pad(array, widths)


def _load_shard(path, need_meta, slots=0, jslots=0):
    """One shard as numpy arrays; object slots are padded to `slots` and joint slots to `jslots` (box shards have no joints)."""
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
        if 'joints' in z.files:
            joints, joint_mask = z['joints'].astype(np.float32, copy=False), z['joint_mask'].astype(bool)
        else:
            joints, joint_mask = np.zeros((len(states), 0, JOINT_DIM), np.float32), np.zeros((len(states), 0), bool)
    dyn = mask & (props[..., 5] < .5)                                        # [E,N]
    speed = np.linalg.norm(states[..., 7:10], axis=-1)                       # [E,T+1,N]
    moving = ((speed > .05) & dyn[:, None]).any(-1)                          # [E,T+1]
    change = np.zeros_like(moving)
    change[:, 1:] = ((contact[:, 1:] != contact[:, :-1]) & dyn[:, None]).any(-1)
    count = states.shape[2]
    if slots > count:                                                        # empty slots as the generators write them
        states, props, mask, rest = _pad(states, 2, slots), _pad(props, 1, slots), _pad(mask, 1, slots), _pad(rest, 2, slots)
        states[:, :, count:, 6] = 1
        props[:, count:, 3] = 1
    joints = np.where(joint_mask[..., None], joints, 0)                      # unused joint rows are all-zero
    joints, joint_mask = _pad(joints, 1, jslots), _pad(joint_mask, 1, jslots)
    return dict(states=states, props=props, context=context, mask=mask, rest=rest,
                active=moving | change, chaotic=chaotic, joints=np.ascontiguousarray(joints, np.float32), joint_mask=joint_mask)


def load_split(root: Path, split: str, limit: int, threads: int, need_meta: bool = False, slots: int = 0, jslots: int = 0):
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
        parts = list(pool.map(lambda f: _load_shard(f, need_meta, slots, jslots), files))
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
    out = (data['states'][e[:, None], t], data['rest'][e[:, None], tr], data['props'][e], data['context'][e], data['mask'][e],
           data['joints'][e], data['joint_mask'][e])
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
    if torch.is_tensor(delta):                            # per-element delta (broadcast)
        a = x.abs()
        return torch.where(a <= delta, .5 * x * x, delta * (a - .5 * delta))
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


# ----------------------------------------------------------------------------- joints
def jointed_pairs(joints, joint_mask, count):
    """[B,N,N] bool: object slots directly connected by a joint (they do not collide, as PyBullet parent/child links)."""
    if hasattr(jg, 'jointed_pairs'):                      # the model's own rule (also used by its contact-edge filter / projection)
        return jg.jointed_pairs(joints, joint_mask, count)
    parent, child = joints[..., 0].round().long(), joints[..., 1].round().long()
    ok = joint_mask & (parent >= 0) & (child >= 0) & (parent < count) & (child < count)
    adjacent = torch.zeros(joints.shape[0], count, count, dtype=torch.bool, device=joints.device)
    b, j = ok.nonzero(as_tuple=True)
    adjacent[b, parent[b, j], child[b, j]] = True
    adjacent[b, child[b, j], parent[b, j]] = True
    return adjacent


def penetration_depths(state, props, context, mask, adjacent=None):
    """contact_gnn.penetration_depths with the object-object depth ignoring `adjacent` (jointed) pairs.
    Identical to the box function when adjacent is None or empty."""
    if adjacent is None or not bool(adjacent.any()):
        return cg.penetration_depths(state, props, context, mask)
    with cg._no_autocast(state):
        props, context = cg._pad_inputs(props, context)
        state = state.float()
        valid = mask & ~cg.alias_mask(props, context, state)
        corners = cg.box_corners(state, props)
        support = cg._support_corner_depth(corners, context)[0].max(-1).values * valid
        floor = (-corners[..., 1]).clamp_min(0).max(-1).values * valid
        count = state.shape[1]
        pos, rot, half = state[..., :3], rotation(state[..., 3:7]), props[..., :3] / 2
        movable = props[..., 5] < .5
        pair = valid[:, :, None] & valid[:, None, :] & (movable[:, :, None] | movable[:, None, :]) & ~adjacent
        pair = pair & ~torch.eye(count, dtype=torch.bool, device=state.device)
        with torch.no_grad():
            radius = half.norm(dim=-1)
            near = (pos[:, :, None] - pos[:, None]).norm(dim=-1) <= radius[:, :, None] + radius[:, None, :]
        b, i, j = (pair & near).nonzero(as_tuple=True)
        points = pos[b, i][:, None] + cg._mm(cg._samples(state.device) * half[b, i][:, None], rot[b, i].transpose(-1, -2))
        depth = (-cg._sdf(cg._mm(points - pos[b, j][:, None], rot[b, j]), half[b, j][:, None])).clamp_min(0).max(-1).values
        objects = floor.new_zeros(state.shape[0] * count)
        objects = objects.scatter_reduce(0, b * count + i, depth, 'amax').scatter_reduce(0, b * count + j, depth, 'amax')
        return dict(support=support, floor=floor, objects=objects.reshape(state.shape[0], count))


def physics_losses(pred_next, true_next, prev_state, props, context, mask, rest_next, adjacent=None):
    """contact_gnn.physics_losses with jointed pairs excluded from the penetration penalty."""
    if adjacent is None or not bool(adjacent.any()):
        return cg.physics_losses(pred_next, true_next, prev_state, props, context, mask, rest_next)
    with cg._no_autocast(pred_next):
        props, context = cg._pad_inputs(props, context)
        pred_next, true_next, prev_state = pred_next.float(), true_next.float(), prev_state.float()
        dynamic = cg.dynamic_mask(props, context, mask)
        count = dynamic.sum().clamp_min(1)
        predicted = penetration_depths(pred_next, props, context, mask, adjacent)
        with torch.no_grad():
            allowed = penetration_depths(true_next, props, context, mask, adjacent)
        penetration = pred_next.new_zeros(())
        for key in ('support', 'floor', 'objects'):
            excess = (predicted[key] - allowed[key] - .001).clamp_min(0) / .005
            penetration = penetration + (F.smooth_l1_loss(excess, torch.zeros_like(excess), reduction='none') * dynamic).sum() / count
        resting = dynamic & rest_next.bool()
        rest_count = resting.sum().clamp_min(1)
        zero = torch.zeros_like(pred_next[..., :3])
        rest = (F.smooth_l1_loss(pred_next[..., 7:10] / .05, zero, reduction='none').sum(-1)
                + F.smooth_l1_loss(pred_next[..., 10:13] / .25, zero, reduction='none').sum(-1))
        slide = F.smooth_l1_loss((pred_next[..., :3] - prev_state[..., :3]) / cg.STEP / .05, zero, reduction='none').sum(-1)
        turn = 2 * cg.quat_mul(normalize_q(pred_next[..., 3:7]), cg.quat_conj(normalize_q(prev_state[..., 3:7])))[..., :3]
        slide = slide + .1 * F.smooth_l1_loss(turn / cg.STEP / .25, zero, reduction='none').sum(-1)
        return dict(penetration=penetration, rest=(rest * resting).sum() / rest_count, slide=(slide * resting).sum() / rest_count)


def wrap_angle(x):
    return torch.atan2(torch.sin(x), torch.cos(x))


@torch.no_grad()
def joint_metrics(pred, true, joints, joint_mask):
    """Per-joint [B,J] constraint and coordinate errors of predicted states [B,N,13] against true states.
    anchor_gap (m), off_axis (rad; hinge axis misalignment, full relative rotation for prismatic / fixed), limit_excess (rad or m),
    hinge_err (rad, wrapped), prismatic_err (m) plus the bool selectors valid / hinge / prismatic / axial (hinge or prismatic) /
    oriented (every type but ball: the joints for which off_axis is defined)."""
    pred, true = pred.float(), true.float()
    violation = jg.joint_violation(pred, joints, joint_mask)
    q_pred, q_true = jg.joint_coordinates(pred, joints, joint_mask), jg.joint_coordinates(true, joints, joint_mask)
    hinge, prismatic = joint_mask & (joints[..., 2] > .5), joint_mask & (joints[..., 4] > .5)
    fix = lambda x: torch.nan_to_num(x.float(), nan=10., posinf=10., neginf=10.).clamp(0, 10.)
    return dict(anchor_gap=fix(violation['anchor_gap']), off_axis=fix(violation['off_axis']), limit_excess=fix(violation['limit_excess']),
                hinge_err=fix(wrap_angle(q_pred - q_true).abs()), prismatic_err=fix((q_pred - q_true).abs()),
                valid=joint_mask, hinge=hinge, prismatic=prismatic, axial=hinge | prismatic,
                oriented=joint_mask & ~(joints[..., 3] > .5))


def unroll(model, states, rest, props, context, mask, joints, joint_mask, k, args, amp_dtype, gen=None, teacher=None):
    """k-step autoregressive loss. Returns (loss, stats[len(TERMS)], bad flag tensor)."""
    dyn = dynamic_mask(props, context, mask)
    has_joints = bool(joint_mask.any())
    adjacent = jointed_pairs(joints, joint_mask, states.shape[2]) if has_joints else None
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
    if train and has_joints and getattr(args, 'joint_one_delta', 0) > 0:      # per-body Huber delta [B,N,1]
        articulated = jg.articulations(joints, joint_mask, states.shape[2]) >= 0
        delta = torch.where(articulated, float(args.joint_one_delta), float(args.one_delta))[..., None]
    track = has_joints and (getattr(args, 'w_jcoord', 0) > 0 or getattr(args, 'w_jrel', 0) > 0 or not train)
    joint_weight = None
    if train and track and getattr(args, 'w_jmove', 1.) != 1:
        joint_weight = 1 + (args.w_jmove - 1) * jg.joint_motion(states[:, :k + 1], joints, joint_mask).float()
    held = None                                           # --joint-rest: articulated bodies that are truly slow, contact or not
    if train and has_joints and getattr(args, 'joint_rest', 0):
        slow = (states[..., 7:10].norm(dim=-1) < .01) & (states[..., 10:13].norm(dim=-1) < .05)          # [B,k+1,N]
        held = slow[:, :-1] & slow[:, 1:] & (dyn & (jg.articulations(joints, joint_mask, states.shape[2]) >= 0))[:, None]
    plain = ~joint_mask.any(1)                             # windows without joints (the mixed box data, plain_* episodes)
    distill = train and teacher is not None and getattr(args, 'w_distill', 0) > 0 and bool(plain.any())
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
            raw = model.predict_raw(cur, props, context, mask, joints, joint_mask)
        raw = raw.float()
        with torch.no_grad():
            # pushforward-style target: from the model's CURRENT (possibly drifted) input to the TRUE next state
            target = (model.raw_targets(cur.detach(), true, props, context, mask).float() - mean) / std
        nxt = model.integrate(cur, raw * std + mean, props, context, mask)
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
        if held is not None:
            rest_next = rest_next | held[:, j]
        phys = physics_losses(nxt, true, torch.where(near, true, cur), props, context, mask, rest_next, adjacent)
        terms.update(pen=phys['penetration'], rest=phys['rest'], slide=phys['slide'])
        terms.update(jcoord=total.new_zeros(()), jrel=total.new_zeros(()), distill=total.new_zeros(()))
        if has_joints:
            with cg._no_autocast(nxt):
                joint = jg.joint_losses(nxt.float(), true.float(), joints, joint_mask)
                if track:
                    tracking = jg.joint_tracking_losses(nxt.float(), true.float(), joints, joint_mask, joint_weight)
                    terms.update(jcoord=tracking['coord'].float(), jrel=tracking['relative'].float())
            terms.update(janchor=joint['anchor'].float(), jlimit=joint['limit'].float(), jaxis=joint['axis'].float())
        else:                                             # pure box batch: the joint terms are exactly zero
            terms.update(janchor=total.new_zeros(()), jlimit=total.new_zeros(()), jaxis=total.new_zeros(()))
            if train and j == 0:                          # keep the joint-only parameters in the autograd graph (DDP needs every parameter used)
                total = total + sum(p.sum() for p in model.parameters()) * 0.
        if distill:                                       # joint-free windows: stay on the frozen box model's raw output
            with torch.no_grad():
                with torch.autocast('cuda', dtype=amp_dtype or torch.float16, enabled=use_amp):
                    t_raw = teacher.predict_raw(cur[plain].detach(), props[plain], context[plain], mask[plain])
                t_raw = (t_raw.float() * teacher.out_std.float() + teacher.out_mean.float() - mean) / std
            gap = huber(raw[plain] - t_raw).mean(-1)
            terms['distill'] = (gap * dyn[plain]).sum() / dyn[plain].sum().clamp_min(1)
        w_one = args.w_one if not args.one_steps or j < args.one_steps else 0.
        step = (w_one * terms['one'] + args.w_pos * terms['pos'] + args.w_vel * terms['vel'] + args.w_ang * terms['ang']
                + args.w_rot * terms['rot'] + args.w_pen * terms['pen'] + args.w_rest * terms['rest'] + args.w_slide * terms['slide']
                + args.w_anchor * terms['janchor'] + args.w_limit * terms['jlimit'] + args.w_axis * terms['jaxis']
                + getattr(args, 'w_jcoord', 0.) * terms['jcoord'] + getattr(args, 'w_jrel', 0.) * terms['jrel']
                + (getattr(args, 'w_distill', 0.) * terms['distill'] if distill else 0.))
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
    return total / norm, stats / norm, bad | ~torch.isfinite(total)


def grow_rounds(model, extra):
    """A JointGNN with `extra` more rounds that computes exactly what `model` computes (new rounds = identity)."""
    config = dict(model.config); old = int(config['rounds']); config['rounds'] = old + int(extra)
    if not 0 < int(config.get('shared_rounds', 0)) < old:
        raise ValueError('--grow-rounds needs a checkpoint with gated extra rounds (0 < shared_rounds < rounds); otherwise box scenes would change')
    grown = jg.JointGNN(**config)
    missing = grown.load_state_dict(model.state_dict(), strict=False)
    assert not missing.unexpected_keys and all(key.split('.')[0] in ('edge_updates', 'node_updates', 'joint_updates') for key in missing.missing_keys)
    with torch.no_grad():
        for index in range(old, config['rounds']):
            for update in (grown.edge_updates[index], grown.node_updates[index], grown.joint_updates[index]):
                update[-1].weight.zero_(); update[-1].bias.zero_()
    grown.edge_radius = model.edge_radius
    return grown.train(model.training)


class Trainee(nn.Module):
    """Whole unroll inside forward() so DistributedDataParallel sees one forward per update."""

    def __init__(self, model, args, amp_dtype, teacher=None):
        super().__init__()
        self.model, self.args, self.amp_dtype = model, args, amp_dtype
        self.teacher = [teacher]                          # frozen; kept out of the module tree (not a DDP / optimizer parameter)

    def forward(self, states, rest, props, context, mask, joints, joint_mask, k, gen):
        return unroll(self.model, states, rest, props, context, mask, joints, joint_mask, k, self.args, self.amp_dtype, gen, self.teacher[0])


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
    sl = torch.linspace(0, E - 1, m, device=val['states'].device).round().long()   # evenly spaced: keeps the scenario mix
    states, rest = val['states'][sl].to(device), val['rest'][sl].to(device)
    props, context, mask = val['props'][sl].to(device), val['context'][sl].to(device), val['mask'][sl].to(device)
    joints, joint_mask = val['joints'][sl].to(device), val['joint_mask'][sl].to(device)
    chaotic = val['chaotic'][sl].to(device)
    dyn = dynamic_mask(props, context, mask)
    has_joints = bool(joint_mask.any())
    adjacent = jointed_pairs(joints, joint_mask, states.shape[2]) if has_joints else None
    gap_sum = 0.
    cur = states[:, 0]
    pen_sum = drift_sum = drift_n = 0.
    errors = {}
    fall_steps = min(80, T)                               # the 4 s rollout continues past `steps` for the fall metrics only
    traj = [cur]
    for j in range(1, fall_steps + 1):
        with torch.autocast('cuda', dtype=amp_dtype or torch.float16, enabled=amp_dtype is not None and device.type == 'cuda'):
            cur = model(cur, props, context, mask, joints, joint_mask)
        cur = torch.nan_to_num(cur.float(), nan=1e3, posinf=1e3, neginf=-1e3).clamp(-1e3, 1e3)
        traj.append(cur)
        if j > steps:
            continue
        if has_joints:
            jm = joint_metrics(cur, states[:, j], joints, joint_mask)
            gap_sum += float((jm['anchor_gap'] * jm['valid']).sum())
            if j in (20, 40):
                name = {20: '1s', 40: '2s'}[j]
                out[f'joint_gap_mm_{name}'] = 1000 * _mean(jm['anchor_gap'], jm['valid'])
                out[f'joint_off_axis_deg_{name}'] = math.degrees(1) * _mean(jm['off_axis'], jm['oriented'])
                out[f'joint_coord_err_rad_{name}'] = _mean(jm['hinge_err'], jm['hinge'])
                out[f'joint_slide_err_m_{name}'] = _mean(jm['prismatic_err'], jm['prismatic'])
                out[f'joint_limit_excess_{name}'] = _mean(jm['limit_excess'], jm['axial'])
        depth = penetration_depths(cur, props, context, mask, adjacent)
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
    out['rollout_joints'] = int(joint_mask.sum())
    if has_joints:
        out['joint_gap_mm_mean'] = 1000 * gap_sum / max(float(joint_mask.sum()) * steps, 1)
        out['score'] += 2 * z('joint_gap_mm_2s') / 1000 + .05 * z('joint_coord_err_rad_2s')
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
    return {'kind': getattr(jg.JointGNN, 'kind', 'joint_gnn'), 'config': dict(config), 'experimental': True,
            'inherited_rounds': int(getattr(args, 'inherited_rounds', 0)),
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
        if args.max_gpu_gb > 0:
            torch.cuda.set_per_process_memory_fraction(min(1., args.max_gpu_gb * 2**30 / torch.cuda.get_device_properties(device).total_memory), device)
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
    if args.drop_jointless_train:
        keep = train_np['joint_mask'].any(1)
        if main_rank:
            log(f'[data] --drop-jointless-train: {int((~keep).sum())} of {len(keep)} training episodes without joints removed')
        train_np = {k: v[keep] for k, v in train_np.items()}
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
        log(f'[data] joints: slots={train["joints"].shape[1]} mean joints/episode={float(train["joint_mask"].float().sum(1).mean()):.2f} '
            f'episodes with joints={float(train["joint_mask"].any(1).float().mean()):.3f}')
        val = to_device(load_split(args.data, 'val', args.max_val_episodes, args.load_threads, need_meta=True), device)
        log(f'[data] val episodes={val["states"].shape[0]} chaotic={int(val["chaotic"].sum())}')
    events = event_frames(train) if args.event_frac > 0 else None
    sampler = WindowSampler(train.pop('active'), args.seed, rank, world, events, args.event_frac)
    if main_rank and events is not None:
        log(f'[data] event frames (motion onset / fall approach) {float(events.float().mean()):.3f}; --event-frac {args.event_frac}')
    # ---- optional plain box windows (no joints) so box dynamics are not forgotten
    box = box_sampler = box_val = None
    box_batch = 0
    if args.mix_box_data is not None and args.mix_box_frac > 0:
        slots, jslots = train['states'].shape[2], train['joints'].shape[1]
        box_np = load_split(args.mix_box_data, 'train', args.max_box_episodes, args.load_threads, slots=slots, jslots=jslots)
        if box_np['states'].shape[2] != slots:
            raise ValueError(f'--mix-box-data has {box_np["states"].shape[2]} object slots, more than the joint data ({slots})')
        if box_np['states'].shape[1] - 1 < horizon:
            raise ValueError('--mix-box-data episodes are shorter than --horizon')
        box_bytes = sum(v.nbytes for v in box_np.values())
        box_device = data_device
        if args.data_device == 'auto' and cuda and nbytes + box_bytes > .5 * torch.cuda.get_device_properties(device).total_memory:
            box_device = torch.device('cpu')
        box = to_device(box_np, box_device)
        del box_np
        box_events = event_frames(box) if args.event_frac > 0 else None
        box_sampler = WindowSampler(box.pop('active'), args.seed + 1, rank, world, box_events, args.event_frac)
        box_batch = min(args.batch - 1, max(1, int(round(args.batch * args.mix_box_frac))))
        if main_rank:
            log(f'[data] box mix {args.mix_box_data}: episodes={box["states"].shape[0]} padded to {slots} slots, {box_bytes / 2**30:.2f} GiB on {box_device}; '
                f'{box_batch}/{args.batch} windows per batch ({box_batch / args.batch:.2f})')
            box_val = to_device(load_split(args.mix_box_data, 'val', min(args.max_val_episodes, 2048), args.load_threads, need_meta=True,
                                           slots=slots, jslots=jslots), device)

    # ---- model
    last_path = args.out / 'last.pt'
    resuming = args.resume and last_path.exists()
    init = contact = resume = None
    contact_path = None
    if resuming:
        resume = torch.load(last_path, map_location='cpu', weights_only=False)
        model = jg.JointGNN(**resume['config'])
        model.load_state_dict(resume['model'])
        if args.inherited_rounds < 0:
            args.inherited_rounds = int(resume.get('inherited_rounds', 0))
    elif args.init is not None:
        init = torch.load(args.init, map_location='cpu', weights_only=True)
        model = jg.build_from_checkpoint(init)            # architecture, weights and normalizer follow the checkpoint
        if args.grow_rounds > 0:
            model = grow_rounds(model, args.grow_rounds)
        if args.inherited_rounds < 0:
            args.inherited_rounds = int(init.get('inherited_rounds', 0))
    else:
        if args.init_contact.lower() == 'auto':
            contact_path = DEFAULT_CONTACT if DEFAULT_CONTACT.exists() else None
        elif args.init_contact.lower() not in ('', 'none', 'off'):
            contact_path = Path(args.init_contact)
        if contact_path is not None:
            contact = torch.load(contact_path, map_location='cpu', weights_only=True)
            for key, value in (contact.get('config') or {}).items():   # trunk shape follows the box checkpoint; --rounds stays ours
                if key in ('width', 'edge_radius'):
                    setattr(args, key, value)
        shared = 0
        if contact is not None and args.gate_extra_rounds:
            shared = len({key.split('.')[1] for key in contact['state_dict'] if key.startswith('edge_updates.')})
            shared = shared if shared < args.rounds else 0
        model = jg.JointGNN(width=args.width, rounds=args.rounds, edge_radius=args.edge_radius, shared_rounds=shared)
        if contact is not None:
            jg.init_from_contact_gnn(model, contact)
            if main_rank:                                 # pin the exact box checkpoint this run descends from (eval --contact)
                args.out.mkdir(parents=True, exist_ok=True)
                atomic_save(contact, args.out / 'init_contact.pt')
                args.init_contact_epoch = contact.get('epoch')
            if args.inherited_rounds < 0:
                args.inherited_rounds = len({key.split('.')[1] for key in contact['state_dict'] if key.startswith('edge_updates.')})
    args.inherited_rounds = max(args.inherited_rounds, 0)
    model = model.to(device).train()
    # frozen box teacher (distillation target on joint-free windows, reference of the box guard)
    teacher = None
    if args.w_distill > 0 or args.box_guard > 0:
        teacher_path = args.teacher
        if teacher_path.lower() == 'auto':
            pinned = args.out / 'init_contact.pt'         # a resumed / --init run keeps the teacher of the original warm start
            teacher_path = contact_path if contact_path is not None else pinned if pinned.exists() else DEFAULT_CONTACT
        # the very bytes of the warm start when it is the same file (models/contact_gnn.pt may be replaced while we run)
        teacher_ckpt = contact if contact is not None and Path(teacher_path) == Path(contact_path) else torch.load(teacher_path, map_location='cpu', weights_only=True)
        if teacher_ckpt.get('kind') != cg.ContactGNN.kind:
            raise ValueError(f'--teacher must be a contact_gnn checkpoint, got {teacher_ckpt.get("kind")}')
        teacher = jg.build_from_checkpoint(teacher_ckpt).to(device).eval().requires_grad_(False)   # exact on joint-free scenes
        if main_rank:
            log(f'[teacher] frozen contact_gnn {teacher_path} (epoch {teacher_ckpt.get("epoch")}): --w-distill {args.w_distill} --box-guard {args.box_guard}')
        del teacher_ckpt
    config = dict(model.config)
    args.width, args.rounds, args.edge_radius = (config.get(key, getattr(args, key)) for key in ('width', 'rounds', 'edge_radius'))
    updates_per_epoch = max(1, args.windows_per_epoch // (args.batch * world))
    total_updates = updates_per_epoch * args.epochs
    requested_warmup = args.warmup
    args.warmup = max(1, min(args.warmup, int(.05 * total_updates)))
    if main_rank:
        horizon_note = max(args.start_horizon, 1) if args.start_horizon else 'curriculum'
        if init is not None:
            log(f'[init] warm start from joint_gnn checkpoint {args.init} (epoch {init.get("epoch")}, config {init.get("config")}); '
                f'normalizer from the checkpoint; fresh optimizer/schedule; start horizon {horizon_note}')
        elif contact is not None:
            log(f'[init] warm start from contact_gnn checkpoint {contact_path} (epoch {contact.get("epoch")}, config {contact.get("config")}) via '
                f'joint_gnn.init_from_contact_gnn; normalizer from the box checkpoint; fresh optimizer/schedule; start horizon {horizon_note}')
        elif resume is None:
            mean, std, n = fit_normalizer(model, train, device, args.seed)
            log(f'[normalizer] from scratch; {n} dynamic transitions\n  mean={[round(float(x), 4) for x in mean]}\n  std ={[round(float(x), 4) for x in std]}')
    del init, contact
    ema = copy.deepcopy(model).eval().requires_grad_(False) if main_rank else None
    if resume is not None and main_rank:
        ema.load_state_dict(resume['ema'])
    trainee = Trainee(model, args, amp_dtype, teacher)
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

    # parameter groups: (inherited from contact_gnn | new) x (decay | no decay); the groups keep this order so that
    # the optimizer state of a resumed run lines up
    inherited_names = set()
    if args.inherited_rounds > 0:
        inherited_names = {name for name, _ in cg.ContactGNN(args.width, args.inherited_rounds).named_parameters()}
    groups = {(old, wd): [] for old in (True, False) for wd in (True, False)}
    for name, p in model.named_parameters():
        groups[name in inherited_names, p.ndim >= 2].append(p)
    optimizer = torch.optim.AdamW([dict(params=params, weight_decay=args.wd if wd else 0., inherited=old)
                                   for (old, wd), params in groups.items() if params], lr=args.lr, fused=cuda)
    if main_rank:
        n_old = sum(p.numel() for (old, _), ps in groups.items() if old for p in ps)
        n_new = sum(p.numel() for (old, _), ps in groups.items() if not old for p in ps)
        log(f'[optimizer] inherited from contact_gnn ({args.inherited_rounds} rounds): {n_old:,} parameters at LR x{args.inherit_lr_scale:g}'
            f'{f" (frozen for the first {args.inherit_freeze_updates} updates)" if args.inherit_freeze_updates else ""}; new: {n_new:,} parameters at LR x1')

    def inherited_scale(step):
        return 0. if step < args.inherit_freeze_updates else args.inherit_lr_scale
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
        log(f'[model] JointGNN {config} params={params:,}; updates/epoch={updates_per_epoch} total={total_updates} '
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
    box_reference = None
    if main_rank and box_val is not None and teacher is not None and args.box_guard > 0:
        box_reference = validate(teacher, box_val, device, args, amp_dtype)
        log(f'[box-guard] teacher on the box val rollouts: score={box_reference["score"]:.5f} '
            f'pos2s c/g={box_reference.get("pos_err_2s_chaotic", float("nan")):.4f}/{box_reference.get("pos_err_2s_gentle", float("nan")):.4f}; '
            f'limit {1 + args.box_guard:.3f}x')
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
        k = horizon_at(global_step, total_updates, horizon, progress, args.start_horizon)
        for update in range(updates_per_epoch):
            k = horizon_at(global_step, total_updates, horizon, progress, args.start_horizon)
            sampler.set(epoch, k)
            lr = lr_at(global_step, total_updates, args, progress)
            old_lr = lr * inherited_scale(global_step)
            for group in optimizer.param_groups:
                group['lr'] = old_lr if group.get('inherited') else lr
            e, t0 = sampler.next(args.batch - box_batch)
            batch = gather(train, e, t0, k, device, args.rest_lookahead)
            if box_batch:
                box_sampler.set(epoch, k)
                e, t0 = box_sampler.next(box_batch)
                batch = tuple(torch.cat(pair) for pair in zip(batch, gather(box, e, t0, k, device, args.rest_lookahead)))
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
                            f'pos={row["pos"]:.4f} pen={row["pen"]:.4f} rest={row["rest"]:.4f} slide={row["slide"]:.4f} '
                            f'janchor={row["janchor"]:.4f} jlimit={row["jlimit"]:.4f} jaxis={row["jaxis"]:.4f} '
                            f'jcoord={row["jcoord"]:.4f} jrel={row["jrel"]:.4f} distill={row["distill"]:.4f} lr={lr:.2e} '
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
            if box_val is not None:                       # box-only regression check on the mixed dataset's val split
                box_metrics = validate(ema, box_val, device, args, amp_dtype)
                metrics.update({f'box_{name}': value for name, value in box_metrics.items()})
                metrics['score'] += args.box_score_weight * box_metrics['score']
                if box_reference is not None:
                    ratio = box_metrics['score'] / max(box_reference['score'], 1e-9)
                    metrics.update(box_score_ratio=ratio, box_reference_score=box_reference['score'], box_guard_violated=bool(ratio > 1 + args.box_guard))
                    if ratio > 1 + args.box_guard:        # never best.pt while any checkpoint passes the guard
                        metrics['score'] += 1000.
            metrics.update(epoch=epoch + 1, step=global_step, k=k, train_seconds=train_time, val_seconds=time.time() - t,
                           **{f'train_{name}': float(v) / max(epoch_logs, 1) for name, v in zip(TERMS, epoch_stats)})
            improved = metrics['score'] < best
            best = min(best, metrics['score'])
            ship = shipped(ema, config, epoch + 1, args, metrics)
            atomic_save(ship, args.out / f'epoch_{epoch + 1:03d}.pt')
            if improved:
                atomic_save(ship, args.out / 'best.pt')
            atomic_save(dict(model=model.state_dict(), ema=ema.state_dict(), optimizer=optimizer.state_dict(), scheduler=dict(global_step=global_step, total_updates=total_updates),
                             inherited_rounds=args.inherited_rounds,
                             scaler=scaler.state_dict(), epoch=epoch + 1, global_step=global_step, best=best, dropped=dropped_total, skipped=skipped_total,
                             config=config, args=jsonable(vars(args)), rng=dict(torch=torch.get_rng_state(), noise=gen.get_state().cpu())), last_path)
            log_file.write(json.dumps(clean(dict(event='epoch', improved=improved, **metrics))) + '\n')
            log_file.flush()
            log(f'[epoch {epoch + 1}/{args.epochs}] k={k} train={metrics["train_loss"]:.4f} val_k{horizon}={metrics["val_loss"]:.4f} '
                f'pos1s c/g={metrics.get("pos_err_1s_chaotic", float("nan")):.4f}/{metrics.get("pos_err_1s_gentle", float("nan")):.4f} '
                f'pos2s c/g={metrics.get("pos_err_2s_chaotic", float("nan")):.4f}/{metrics.get("pos_err_2s_gentle", float("nan")):.4f} '
                f'pen={metrics["mean_penetration"] * 1000:.2f}mm drift={metrics["rest_drift"]:.4f} '
                f'fall P/R={metrics["fall_precision"]:.3f}/{metrics["fall_recall"]:.3f} (on {metrics["fall_precision_on"]:.3f}/{metrics["fall_recall_on"]:.3f}) '
                f'joints gap 1s/2s={metrics.get("joint_gap_mm_1s", float("nan")):.2f}/{metrics.get("joint_gap_mm_2s", float("nan")):.2f}mm '
                f'off-axis 2s={metrics.get("joint_off_axis_deg_2s", float("nan")):.2f}deg '
                f'coord 1s/2s={metrics.get("joint_coord_err_rad_1s", float("nan")):.3f}/{metrics.get("joint_coord_err_rad_2s", float("nan")):.3f}rad '
                + (f'box pos2s c/g={metrics.get("box_pos_err_2s_chaotic", float("nan")):.4f}/{metrics.get("box_pos_err_2s_gentle", float("nan")):.4f} ' if box_val is not None else '')
                + (f'box/teacher={metrics["box_score_ratio"]:.3f}{" GUARD" if metrics["box_guard_violated"] else ""} ' if 'box_score_ratio' in metrics else '')
                + f'score={metrics["score"]:.4f}'
                f'{" *best*" if improved else ""} train {train_time:.0f}s val {metrics["val_seconds"]:.0f}s')
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
