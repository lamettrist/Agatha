"""Convert openly licensed external physics datasets into APPROXIMATE Agatha episode shards.

    CUDA_VISIBLE_DEVICES="" .venv-train/bin/python -I scripts/prepare_external_data.py \
        --external data/external --out data/external_v2 --workers 16

Output: <out>/<dataset>/test/shard_*.npz + <out>/<dataset>/meta.json + <out>/meta.json, in the
data/v2 layout (see scripts/generate_contact_data.py FORMAT) plus the extra keys listed in
EXTRA_FORMAT.  Licences are recorded in DATA_LICENSES.md.  The inputs are untrusted downloads:
they are only parsed with numpy (allow_pickle=False), json and h5py; nothing in them is executed.

EVERY RESULT DERIVED FROM THESE SHARDS IS APPROXIMATE.  None of the three sources is real video:
all are simulator ground truth (PyBullet / Blender-Bullet / TDW-PhysX) from other generators.

APPROXIMATIONS (all of them; keep this list in sync with docs/part-c-external-data.md)

 General
  A1  Every object is replaced by an oriented BOX with the extents of its bounding box.  Spheres,
      cylinders and cones therefore roll/collide differently from their box stand-in.  `shape`
      keeps the source label and `box_like` marks true cuboids; metrics are reported for box-like
      objects / box-like-only episodes separately.  Spheres get the identity orientation and zero
      angular velocity (their bounding box does not rotate; a spinning box would dig its corners
      into the ground), so a sphere can never be 'tipped'.  Cylinders/cones keep their rotation
      and their bounding box can still dip below the ground when they roll.
  A2  The source ground plane is unbounded.  Shards store it as a large Agatha support slab
      (w = d = max(8 m, trajectory extent + 2 m), thickness 0.08 m) whose TOP is at
      support_height = 0.75 m: all heights are shifted by +0.75 m.  Agatha was trained on
      0.9-1.5 m x 0.65-1.0 m tables, so this context is out of distribution.  eval_external.py can
      alternatively map the ground to Agatha's y = 0 floor with support collisions disabled
      (`floor` mode, derived from the same shards); both are evaluated and reported.
  A3  20 Hz resampling: positions linear, orientations slerp.  Velocities: see per dataset.
  A4  Episodes are cut to at most 80 steps (4 s, the training horizon); shorter episodes are
      padded by repeating the last frame and `n_frames` holds the true length.
  A5  `tipped` is NOT agatha.physics.summarize's absolute rule.  Tossed objects start in arbitrary
      orientations, so it is: the body axis that pointed up at the first frame is later more than
      35 deg away from world up.  For spheres/cylinders this is just 'rolled'.
  A6  `fell` uses the summarize rule (centre outside the support footprint and below its top).
      With the large slab nothing can fall, so it is all False: a predicted fall is a false alarm.
  A7  contact_support = lowest box corner within 5 mm of the ground (geometric guess from the BOX
      approximation); contact_floor and contact_object are all False (not derived);
      rest = contact_support and |v| < 0.01 and |w| < 0.05 on the approximate velocities.
  A8  chaotic = any dynamic object tipped (A5) or displaced by more than 0.30 m (the collision
      speed criterion of data/v2 is not evaluated).  live_physics = False everywhere.
  A9  rolling/spinning friction 0, support restitution 0, tilt 0 (not given by any source).
      support friction 0.5 unless stated below.

 Kubric MOVi-A (PyBullet, Z-up, 12 fps, 24 frames)
  K1  Similarity rescale: length x0.36, time x0.6.  Gravity stays 10 m/s^2 (measured in the source),
      12 fps becomes exactly 0.05 s per frame, and the 0.7 / 1.4 unit objects become 0.252 / 0.504 m
      (the larger one is still above the 0.35 m training maximum).  Episodes last 1.15 s rescaled.
  K2  Velocities are the dataset's own (x0.6 linear, /0.6 angular), not finite differences
      (12 fps finite differences across bounces are much worse).  --kubric-fd switches.
  K3  Masses x0.36 (a global factor, which rigid-body dynamics is invariant to) so that they fall
      inside the 0.05-3 kg training range; friction/restitution straight from the metadata
      (restitution 0.7 for rubber exceeds the 0-0.6 training range).
  K4  Z-up -> Y-up by the proper rotation (x, y, z) -> (x, z, -y); quaternion wxyz -> xyzw.
  K5  Objects are still in flight / tumbling at the last frame for most slots (`settled` marks
      the others), so the 'rest position' is mostly the position at 1.15 s, not a resting pose.

 CLEVRER (Blender/Bullet, Z-up, 128 annotated frames)
  C1  Units, object sizes, masses, friction and the time base are NOT documented.  Assumed: video
      time base 25 fps (the annotated velocities imply ~36 Hz and are ignored), every object is a
      0.4 unit bounding cube (centres sit at z = 0.2), length x0.5 so cubes are 0.2 m.
  C2  Velocities are central finite differences of the 25 fps annotations.
  C3  Friction is CALIBRATED, not read: the median deceleration of freely sliding cubes in the
      first 200 annotation files is 5.57e-4 units/frame^2 = 0.174 m/s^2 after C1, i.e. mu_eff =
      0.0177 with g = 9.81; object friction = support friction = 0.133 (Bullet multiplies them).
      Mass 0.5 kg for every object, restitution 0, gravity 9.81 (not observable in planar motion).
  C4  Orientation is assumed to be Blender XYZ Euler radians (R = Rz Ry Rx).
  C5  Objects outside the camera view are kept (their simulated state is annotated).

 Physion Dominoes / Support (TDW / PhysX, left-handed Y-up, 100 Hz, metres)
  P1  Froude similarity rescale: length x0.5, time x sqrt(0.5); gravity stays 9.81.  Dominoes
      become 0.05 x 0.25 x 0.125 m and tower blocks 0.08-0.39 m.  Mass x0.125 (same density).
  P2  Left-handed -> right-handed by mirroring z.  Positions are the `center` arrays (the
      `positions` arrays are the bottom-centre model origin).
  P3  Velocities are central finite differences at the 100 Hz source rate, interpolated to 20 Hz
      (avoids relying on the undocumented units of the stored angular velocities).
  P4  Episodes start 2 source frames after `push_time`, i.e. after the scripted push was applied,
      with the measured post-push velocities as the initial state.
  P5  Distractor / occluder meshes (static props off to the side) are dropped.  The static 'zone'
      mat (1 cm thick in the source, up to 5 m wide, lying on the ground) is dropped as well, so
      objects resting on it hover 5 mm above the slab after P1.  Friction = dynamic_friction
      clipped to [0.01, 2] (PhysX combines frictions differently from Bullet; the ground friction
      is unknown and left at 0.5), restitution = bounciness.
  P6  Only 10 trials were downloaded (8 Dominoes, 2 Support): numbers are anecdotal.
"""
import argparse
import glob
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(Path(__file__).resolve().parent)):
    if _p not in sys.path: sys.path.insert(0, _p)
import numpy as np

STEP = .05
MAX_STEPS = 80
SUPPORT_HEIGHT = .75
SUPPORT_THICKNESS = .08
MIN_SLAB = 8.
MAX_SLAB = 100.          # agatha.schema Positive limit
TIP_COS = math.cos(math.radians(35))
MOVED_FAR_M = .30
REST_LIN, REST_ANG = .01, .05
SETTLED_LIN, SETTLED_ANG = .05, .25   # 'has come to rest by the last frame' (looser: approximate velocities)
CONTACT_EPS = .005
PROPS_DIM, CONTEXT_DIM = 9, 10
NOTE = 'APPROXIMATE external data converted with documented approximations; simulator output, not real-world video.'

EXTRA_FORMAT = dict(
    n_frames='int64 [E] number of valid frames (<= T+1); later frames repeat the last valid one',
    shape='<U16 [E,N] source shape label', box_like='bool [E,N] source object is a true cuboid',
    moved_far=f'bool [E,N] final displacement > {MOVED_FAR_M} m', settled='bool [E,N] |v|<0.05 and |w|<0.25 at the last valid frame',
    rest_pos='float32 [E,N,3] mean position over the last 3 valid frames', source_id='<U80 [E] source episode / file name',
    tipped='bool [E,N] RELATIVE rule: initial up axis rotated > 35 deg (differs from data/v2)',
)


# ----------------------------------------------------------------------------- quaternion / frame helpers (xyzw)
def q_normalize(q):
    q = np.asarray(q, np.float64)
    return q / np.clip(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12, None)


def q_mul(a, b):
    ax, ay, az, aw = np.moveaxis(np.asarray(a, np.float64), -1, 0); bx, by, bz, bw = np.moveaxis(np.asarray(b, np.float64), -1, 0)
    return np.stack((aw*bx+ax*bw+ay*bz-az*by, aw*by-ax*bz+ay*bw+az*bx, aw*bz+ax*by-ay*bx+az*bw, aw*bw-ax*bx-ay*by-az*bz), -1)


def q_conj(q):
    return np.asarray(q, np.float64) * np.array([-1., -1., -1., 1.])


def q_to_matrix(q):
    x, y, z, w = np.moveaxis(q_normalize(q), -1, 0)
    return np.stack((np.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)), -1),
                     np.stack((2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)), -1),
                     np.stack((2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), -1)), -2)


def q_to_rotvec(q):
    q = q_normalize(q); q = np.where(q[..., 3:] < 0, -q, q)
    norm = np.linalg.norm(q[..., :3], axis=-1, keepdims=True)
    angle = 2 * np.arctan2(norm, q[..., 3:])
    return np.where(norm > 1e-9, q[..., :3] / np.clip(norm, 1e-12, None) * angle, 2 * q[..., :3])


def euler_xyz_to_quat(e):
    """Blender 'XYZ' Euler (R = Rz @ Ry @ Rx) -> xyzw."""
    e = np.asarray(e, np.float64); h = e / 2; zero = np.zeros_like(h[..., 0])
    qx = np.stack((np.sin(h[..., 0]), zero, zero, np.cos(h[..., 0])), -1)
    qy = np.stack((zero, np.sin(h[..., 1]), zero, np.cos(h[..., 1])), -1)
    qz = np.stack((zero, zero, np.sin(h[..., 2]), np.cos(h[..., 2])), -1)
    return q_mul(qz, q_mul(qy, qx))


def sign_continuous(q, axis=0):
    """Flip signs along `axis` so consecutive quaternions have a non-negative dot product."""
    q = np.moveaxis(q_normalize(q), axis, 0).copy()
    for t in range(1, len(q)):
        flip = (q[t] * q[t-1]).sum(-1, keepdims=True) < 0
        q[t] = np.where(flip, -q[t], q[t])
    return np.moveaxis(q, 0, axis)


def slerp(q0, q1, u):
    """Shortest-arc spherical interpolation; u broadcastable to q[..., 0]."""
    q0 = q_normalize(q0); q1 = q_normalize(q1); u = np.asarray(u, np.float64)[..., None]
    dot = (q0 * q1).sum(-1, keepdims=True)
    q1 = np.where(dot < 0, -q1, q1); dot = np.clip(np.abs(dot), 0, 1)
    theta = np.arccos(dot); s = np.sin(theta); small = s < 1e-6
    safe = np.where(small, 1., s)
    a = np.where(small, 1 - u, np.sin((1 - u) * theta) / safe); b = np.where(small, u, np.sin(u * theta) / safe)
    return q_normalize(a * q0 + b * q1)


def _bracket(t_src, t_dst):
    t_src = np.asarray(t_src, np.float64); t_dst = np.asarray(t_dst, np.float64)
    if t_dst.min() < t_src[0] - 1e-9 or t_dst.max() > t_src[-1] + 1e-9: raise ValueError('resampling would extrapolate')
    hi = np.clip(np.searchsorted(t_src, t_dst, side='right'), 1, len(t_src) - 1); lo = hi - 1
    u = np.clip((t_dst - t_src[lo]) / (t_src[hi] - t_src[lo]), 0, 1)
    return lo, hi, u


def interp_linear(t_src, values, t_dst):
    """values [T, ...] -> [K, ...] by linear interpolation along axis 0 (no extrapolation)."""
    lo, hi, u = _bracket(t_src, t_dst); values = np.asarray(values, np.float64)
    u = u.reshape(-1, *([1] * (values.ndim - 1)))
    return values[lo] * (1 - u) + values[hi] * u


def interp_slerp(t_src, quats, t_dst):
    """quats [T, ..., 4] xyzw -> [K, ..., 4] by slerp along axis 0; output is sign-continuous."""
    lo, hi, u = _bracket(t_src, t_dst); quats = np.asarray(quats, np.float64)
    u = u.reshape(-1, *([1] * (quats.ndim - 2)))
    return sign_continuous(slerp(quats[lo], quats[hi], u))


def finite_difference(t, pos, quat):
    """Central (one-sided at the ends) linear and world-frame angular velocity. pos [T,...,3], quat [T,...,4]."""
    t = np.asarray(t, np.float64); pos = np.asarray(pos, np.float64); quat = sign_continuous(quat)
    lo = np.r_[0, np.arange(len(t) - 2), len(t) - 2]; hi = np.r_[1, np.arange(2, len(t)), len(t) - 1]
    dt = (t[hi] - t[lo]).reshape(-1, *([1] * (pos.ndim - 1)))
    return (pos[hi] - pos[lo]) / dt, q_to_rotvec(q_mul(quat[hi], q_conj(quat[lo]))) / dt


def zup_to_yup_vec(v):
    """Right-handed Z-up -> right-handed Y-up, proper rotation (x, y, z) -> (x, z, -y). Also valid for angular velocity."""
    v = np.asarray(v, np.float64)
    return np.stack((v[..., 0], v[..., 2], -v[..., 1]), -1)


def zup_to_yup_quat(q):
    """xyzw; R' = C R C^T, so body axes are permuted too: use zup_to_yup_size for the extents."""
    q = np.asarray(q, np.float64)
    return np.stack((q[..., 0], q[..., 2], -q[..., 1], q[..., 3]), -1)


def zup_to_yup_size(s):
    s = np.asarray(s, np.float64)
    return np.stack((s[..., 0], s[..., 2], s[..., 1]), -1)


def mirror_z_vec(v):
    """Left-handed <-> right-handed by mirroring z (true vectors: positions, linear velocity)."""
    return np.asarray(v, np.float64) * np.array([1., 1., -1.])


def mirror_z_quat(q):
    """xyzw; R' = M R M with M = diag(1,1,-1): rotation axis (pseudo-vector) -> (-x, -y, z). Extents unchanged."""
    return np.asarray(q, np.float64) * np.array([-1., -1., 1., 1.])


def wxyz_to_xyzw(q):
    q = np.asarray(q)
    return np.concatenate((q[..., 1:], q[..., :1]), -1)


# ----------------------------------------------------------------------------- labels
def tipped_relative(quat):
    """quat [..., F, N, 4] xyzw -> bool [..., N]: initial up axis rotated more than 35 deg in any frame."""
    quat = np.asarray(quat, np.float64)
    rel = q_mul(quat, q_conj(quat[..., :1, :, :]))
    x, z = rel[..., 0], rel[..., 2]; n = (rel * rel).sum(-1).clip(1e-12)
    return ((1 - 2 * (x * x + z * z) / n) < TIP_COS).any(-2)


def box_corners(pos, quat, size):
    signs = np.array([[x, y, z] for x in (-1., 1.) for y in (-1., 1.) for z in (-1., 1.)])
    local = signs * (np.asarray(size, np.float64)[..., None, :] / 2)                    # [...,8,3]
    return np.asarray(pos, np.float64)[..., None, :] + np.einsum('...ij,...kj->...ki', q_to_matrix(quat), local)


def slab_side(pos):
    """Square slab side that contains the whole trajectory (+1 m margin), within the schema limit."""
    extent = float(np.abs(np.asarray(pos)[..., [0, 2]]).max()) if np.size(pos) else 0.
    return float(np.clip(math.ceil(2 * extent + 2), MIN_SLAB, MAX_SLAB))


# ----------------------------------------------------------------------------- episode assembly
def make_episode(pos, quat, vel, ang, size, mass, friction, restitution, static, shape, box_like, kind, source_id,
                 gravity=9.81, support_friction=.5, slots=None):
    """Arrays already at 20 Hz, Y-up metres, ground plane at y = 0. pos/vel/ang [F,N,3], quat [F,N,4] xyzw."""
    pos = np.asarray(pos, np.float64)[:MAX_STEPS + 1]; frames, count = pos.shape[:2]
    quat = sign_continuous(np.asarray(quat)[:frames]); vel = np.asarray(vel, np.float64)[:frames]; ang = np.asarray(ang, np.float64)[:frames]
    static = np.asarray(static, bool); slots = slots or count
    sphere = np.array([str(s) == 'sphere' for s in shape])                                 # A1: bounding box of a sphere never rotates
    quat = np.where(sphere[None, :, None], np.array([0., 0., 0., 1.]), quat); ang = np.where(sphere[None, :, None], 0., ang)
    vel = np.where(static[None, :, None], 0., vel); ang = np.where(static[None, :, None], 0., ang)
    states = np.zeros((MAX_STEPS + 1, slots, 13), np.float32); states[..., 6] = 1
    shifted = pos + np.array([0., SUPPORT_HEIGHT, 0.])
    states[:frames, :count] = np.concatenate((shifted, quat, vel, ang), -1)
    states[frames:] = states[frames - 1]; states[frames:, :, 7:] = states[frames - 1, :, 7:]
    props = np.zeros((slots, PROPS_DIM), np.float32); props[:, 3] = 1
    props[:count, :3] = size; props[:count, 3] = mass; props[:count, 4] = friction; props[:count, 5] = static; props[:count, 6] = restitution
    side = slab_side(pos)
    context = np.array([side, side, SUPPORT_HEIGHT, gravity, 0, 0, 0, 1, support_friction, 0], np.float32)
    mask = np.zeros(slots, bool); mask[:count] = True
    dyn = mask.copy(); dyn[:count] &= ~static
    pad = lambda a, fill: np.concatenate((np.asarray(a), np.full((slots - count, *np.shape(a)[1:]), fill, np.asarray(a).dtype)))
    tipped = pad(tipped_relative(quat), False) & dyn
    displacement = pad(np.linalg.norm(pos[-1] - pos[0], axis=-1), 0.)
    moved_far = (displacement > MOVED_FAR_M) & dyn
    speed, spin = np.linalg.norm(vel, axis=-1), np.linalg.norm(ang, axis=-1)
    settled = pad((speed[-1] < SETTLED_LIN) & (spin[-1] < SETTLED_ANG), False) & dyn
    lowest = box_corners(pos, quat, np.asarray(size)[None])[..., 1].min(-1)               # [F,N]
    contact = np.zeros((MAX_STEPS + 1, slots), bool); contact[:frames, :count] = lowest < CONTACT_EPS
    rest = np.zeros_like(contact); rest[:frames, :count] = contact[:frames, :count] & (speed < REST_LIN) & (spin < REST_ANG)
    contact[frames:] = contact[frames - 1]; rest[frames:] = rest[frames - 1]
    contact &= dyn[None]; rest &= dyn[None]
    half = side / 2; local_y = shifted[..., 1] - (SUPPORT_HEIGHT - SUPPORT_THICKNESS / 2)
    off = (np.abs(shifted[..., 0]) > half) | (np.abs(shifted[..., 2]) > half)
    fell = pad((off & (local_y < .01)).any(0), False) & dyn
    rest_pos = np.zeros((slots, 3), np.float32); rest_pos[:count] = shifted[-min(3, frames):].mean(0)
    return dict(states=states, props=props, context=context, mask=mask, contact_support=contact, contact_floor=np.zeros_like(contact),
                contact_object=np.zeros_like(contact), rest=rest, fell=fell, tipped=tipped, kind=kind,
                chaotic=bool((tipped | moved_far | fell).any()), live_physics=False, n_frames=frames,
                shape=pad(np.asarray(shape, '<U16'), ''), box_like=pad(np.asarray(box_like, bool), False), moved_far=moved_far, settled=settled,
                rest_pos=rest_pos, source_id=str(source_id))


def stack_episodes(episodes: list[dict]) -> dict:
    out = {k: np.stack([e[k] for e in episodes]) for k in episodes[0]}
    out['kind'] = out['kind'].astype('<U24'); out['source_id'] = out['source_id'].astype('<U80')
    out['episode_index'] = np.arange(len(episodes), dtype=np.int64)
    return out


def init_penetration(data: dict) -> np.ndarray:
    """Deepest box-approximation overlap at the first frame (m): a direct measure of approximation A1."""
    import torch
    from agatha.contact_gnn import penetration_depths
    out = []
    for start in range(0, len(data['states']), 512):
        sl = slice(start, start + 512)
        t = [torch.from_numpy(data[k][sl]) for k in ('props', 'context', 'mask')]
        depth = penetration_depths(torch.from_numpy(data['states'][sl, 0]), *t)
        dyn = (t[2] & (t[0][..., 5] < .5)).float()
        out.append((torch.stack([depth[k] for k in ('support', 'floor', 'objects')]).amax(0) * dyn).amax(-1).numpy())
    return np.concatenate(out).astype(np.float32)


def write_dataset(out: Path, name: str, episodes: list[dict], info: dict, shard_size: int = 500) -> dict:
    target = out / name / 'test'; target.mkdir(parents=True, exist_ok=True)
    for old in target.glob('shard_*.npz'): old.unlink()
    data = stack_episodes(episodes); data['init_penetration'] = init_penetration(data)
    for index, start in enumerate(range(0, len(episodes), shard_size)):
        np.savez_compressed(target / f'shard_{index:05d}.npz', **{k: v[start:start + shard_size] for k, v in data.items()})
    dyn = data['mask'] & (data['props'][..., 5] < .5); box = dyn & data['box_like']
    frac = lambda a, sel: float(a[sel].mean()) if sel.any() else None
    meta = dict(note=NOTE, dataset=name, step_s=STEP, steps=MAX_STEPS, episodes=len(episodes), max_objects=int(data['mask'].shape[1]),
                support=dict(mode='slab', height=SUPPORT_HEIGHT, thickness=SUPPORT_THICKNESS, side_m=[float(data['context'][:, 0].min()), float(data['context'][:, 0].max())]),
                n_frames=dict(min=int(data['n_frames'].min()), max=int(data['n_frames'].max())),
                dynamic_objects=int(dyn.sum()), box_like_objects=int(box.sum()),
                box_like_only_episodes=int(((dyn & ~data['box_like']).sum(1) == 0).sum()),
                shapes={s: int((data['shape'][dyn] == s).sum()) for s in sorted(set(data['shape'][dyn].tolist()))},
                tipped_fraction=dict(all=frac(data['tipped'], dyn), box_like=frac(data['tipped'], box)),
                moved_far_fraction=dict(all=frac(data['moved_far'], dyn), box_like=frac(data['moved_far'], box)),
                settled_fraction=dict(all=frac(data['settled'], dyn), box_like=frac(data['settled'], box)),
                fell_objects=int(data['fell'].sum()), chaotic_fraction=float(data['chaotic'].mean()),
                size_m=dict(min=float(data['props'][..., :3][dyn].min()), max=float(data['props'][..., :3][dyn].max())),
                mass_kg=dict(min=float(data['props'][..., 3][dyn].min()), max=float(data['props'][..., 3][dyn].max())),
                init_penetration_m=dict(mean=float(data['init_penetration'].mean()), max=float(data['init_penetration'].max()),
                                        frac_over_5mm=float((data['init_penetration'] > .005).mean())),
                format_extra=EXTRA_FORMAT, **info)
    (out / name / 'meta.json').write_text(json.dumps(meta, indent=2))
    return meta


# ----------------------------------------------------------------------------- Kubric MOVi-A
KUBRIC_LENGTH, KUBRIC_TIME, KUBRIC_FPS, KUBRIC_MASS = .36, .6, 12., .36
KUBRIC_SHAPES = {0: 'cube', 1: 'cylinder', 2: 'sphere'}


def convert_kubric(root: Path, use_fd: bool = False, limit: int | None = None) -> tuple[list[dict], dict]:
    path = root / 'kubric-movi' / 'movi_a_128_train_subset.npz'
    with np.load(path, allow_pickle=False) as z:
        keys = ('positions', 'quaternions_wxyz', 'velocities', 'angular_velocities', 'bboxes_3d', 'mass', 'friction', 'restitution',
                'shape_label', 'mask', 'video_name', 'shard')
        a = {k: z[k][:limit] for k in keys}
    frames = a['positions'].shape[2]
    t_src = np.arange(frames) / KUBRIC_FPS * KUBRIC_TIME; t_dst = np.arange(frames) * STEP     # identical by construction (K1)
    slots = a['mask'].shape[1]; episodes = []
    for e in range(len(a['mask'])):
        keep = np.flatnonzero(a['mask'][e])
        pos = zup_to_yup_vec(a['positions'][e, keep].transpose(1, 0, 2)) * KUBRIC_LENGTH                  # [F,N,3]
        quat = sign_continuous(zup_to_yup_quat(wxyz_to_xyzw(a['quaternions_wxyz'][e, keep]).transpose(1, 0, 2)))
        pos_r = interp_linear(t_src, pos, t_dst); quat_r = interp_slerp(t_src, quat, t_dst)
        if use_fd: vel, ang = finite_difference(t_dst, pos_r, quat_r)
        else:
            vel = interp_linear(t_src, zup_to_yup_vec(a['velocities'][e, keep].transpose(1, 0, 2)) * (KUBRIC_LENGTH / KUBRIC_TIME), t_dst)
            ang = interp_linear(t_src, zup_to_yup_vec(a['angular_velocities'][e, keep].transpose(1, 0, 2)) / KUBRIC_TIME, t_dst)
        box = a['bboxes_3d'][e, keep, 0].astype(np.float64)                                             # world corners at frame 0
        edges = np.stack([np.linalg.norm(box[:, i] - box[:, 0], axis=-1) for i in (4, 2, 1)], -1)       # x, y, z body extents
        size = zup_to_yup_size(edges) * KUBRIC_LENGTH
        shape = [KUBRIC_SHAPES.get(int(s), 'unknown') for s in a['shape_label'][e, keep]]
        episodes.append(make_episode(pos_r, quat_r, vel, ang, size, a['mass'][e, keep] * KUBRIC_MASS, a['friction'][e, keep],
                                     a['restitution'][e, keep], np.zeros(len(keep), bool), shape, [s == 'cube' for s in shape],
                                     'movi_a_toss', f"movi_a/train/shard{int(a['shard'][e]):05d}/{a['video_name'][e]}", gravity=10., slots=slots))
    info = dict(source=str(path), simulator='PyBullet (Kubric)', velocities='finite difference' if use_fd else 'dataset (rescaled)',
                rescale=dict(length=KUBRIC_LENGTH, time=KUBRIC_TIME, mass=KUBRIC_MASS), gravity=10., source_fps=KUBRIC_FPS)
    return episodes, info


# ----------------------------------------------------------------------------- CLEVRER
CLEVRER_FPS, CLEVRER_LENGTH, CLEVRER_SIZE_UNITS, CLEVRER_FRICTION, CLEVRER_MASS = 25., .5, .4, .133, .5


def _clevrer_file(name: str):
    with open(name) as handle: j = json.load(handle)
    props = j['object_property']; ids = [int(o['object_id']) for o in props]; count = len(ids)
    track = j['motion_trajectory']
    loc = np.zeros((len(track), count, 3)); ori = np.zeros((len(track), count, 3))
    for f, frame in enumerate(track):
        by_id = {int(o['object_id']): o for o in frame['objects']}
        for i, oid in enumerate(ids): loc[f, i] = by_id[oid]['location']; ori[f, i] = by_id[oid]['orientation']
    if not np.isfinite(loc).all() or not np.isfinite(ori).all(): return None
    loc[..., 2] -= 0.   # ground plane is z = 0 in the source
    t_src = np.arange(len(track)) / CLEVRER_FPS
    pos = zup_to_yup_vec(loc) * CLEVRER_LENGTH
    quat = sign_continuous(zup_to_yup_quat(euler_xyz_to_quat(ori)))
    vel_s, ang_s = finite_difference(t_src, pos, quat)
    steps = min(MAX_STEPS, int((t_src[-2] - STEP) / STEP))
    t_dst = STEP + np.arange(steps + 1) * STEP                   # start one step in: central differences on both sides
    size = np.full((count, 3), CLEVRER_SIZE_UNITS * CLEVRER_LENGTH)
    shape = [str(o['shape']) for o in props]
    return make_episode(interp_linear(t_src, pos, t_dst), interp_slerp(t_src, quat, t_dst), interp_linear(t_src, vel_s, t_dst),
                        interp_linear(t_src, ang_s, t_dst), size, np.full(count, CLEVRER_MASS), np.full(count, CLEVRER_FRICTION),
                        np.zeros(count), np.zeros(count, bool), shape, [s == 'cube' for s in shape], 'clevrer_slide',
                        f"clevrer/validation/{Path(name).stem}", gravity=9.81, support_friction=CLEVRER_FRICTION, slots=6)


def convert_clevrer(root: Path, limit: int = 1000, workers: int = 16) -> tuple[list[dict], dict]:
    files = sorted(glob.glob(str(root / 'clevrer' / 'annotation_validation' / '**' / '*.json'), recursive=True))[:limit]
    if workers > 1:
        import multiprocessing as mp
        with mp.get_context('fork').Pool(workers) as pool: episodes = pool.map(_clevrer_file, files, chunksize=8)
    else: episodes = [_clevrer_file(f) for f in files]
    kept = [e for e in episodes if e is not None]
    info = dict(source=str(root / 'clevrer' / 'annotation_validation'), simulator='Blender / Bullet (CLEVRER)', files=len(files), skipped=len(files) - len(kept),
                velocities='central finite difference at 25 fps', assumed=dict(fps=CLEVRER_FPS, length_scale=CLEVRER_LENGTH, cube_side_units=CLEVRER_SIZE_UNITS,
                friction_object_and_support=CLEVRER_FRICTION, mass_kg=CLEVRER_MASS, restitution=0., gravity=9.81))
    return kept, info


# ----------------------------------------------------------------------------- Physion
PHYSION_LENGTH = .5
PHYSION_TIME = math.sqrt(PHYSION_LENGTH)
PHYSION_DT = .01
PHYSION_START_AFTER_PUSH = 2
PHYSION_PRIMITIVES = ('cube', 'sphere', 'cone', 'cylinder', 'pyramid', 'bowl', 'torus', 'pentagon', 'octahedron', 'dumbbell', 'triangular_prism', 'platonic')


def _physion_file(name: str):
    import h5py
    with h5py.File(name, 'r') as h:
        static = h['static']; keys = sorted(h['frames'].keys())
        names = [n.decode() for n in static['model_names'][()]]
        skip = {n.decode() for key in ('distractors', 'occluders') if key in static for n in static[key][()]}
        mass = np.asarray(static['mass'][()], np.float64); count = len(mass)
        if any(n in skip for n in names[:count]) or any(n not in skip for n in names[count:]):
            return None                                             # property arrays cannot be aligned with certainty
        scale = np.asarray(static['scale'][()], np.float64)[:count]
        friction = np.clip(np.asarray(static['dynamic_friction'][()], np.float64)[:count], .01, 2.)
        bounce = np.clip(np.asarray(static['bounciness'][()], np.float64)[:count], 0., 1.)
        zone = int(np.flatnonzero(static['object_ids'][()] == static['zone_id'][()])[0]) if 'zone_id' in static else -1
        push = int(static['push_time'][()]) if 'push_time' in static else 0
        centre = np.stack([h['frames'][k]['objects']['center'][()] for k in keys])[:, :count].astype(np.float64)
        rot = np.stack([h['frames'][k]['objects']['rotations'][()] for k in keys])[:, :count].astype(np.float64)
        labels = h['frames'][keys[-1]]['labels']
        did_fall = bool(labels['did_fall'][()]) if 'did_fall' in labels else None
    keep = np.array([i != zone and mass[i] < 100 for i in range(count)])           # P5: drop the static zone mat
    names = [n for n, k in zip(names, keep) if k]; mass, scale, friction, bounce = mass[keep], scale[keep], friction[keep], bounce[keep]
    centre, rot = centre[:, keep], rot[:, keep]; count = int(keep.sum()); is_static = np.zeros(count, bool)
    t_src = np.arange(len(keys)) * PHYSION_DT * PHYSION_TIME                       # rescaled seconds
    pos = mirror_z_vec(centre) * PHYSION_LENGTH; quat = sign_continuous(mirror_z_quat(rot))
    vel_s, ang_s = finite_difference(t_src, pos, quat)
    start = t_src[min(push + PHYSION_START_AFTER_PUSH, len(keys) - 3)]
    steps = min(MAX_STEPS, int((t_src[-2] - start) / STEP))
    if steps < 5: return None
    t_dst = start + np.arange(steps + 1) * STEP
    shape = names[:count]
    kind = 'physion_support' if 'towers' in Path(name).name or did_fall is not None else 'physion_dominoes'
    episode = make_episode(interp_linear(t_src, pos, t_dst), interp_slerp(t_src, quat, t_dst), interp_linear(t_src, vel_s, t_dst),
                           interp_linear(t_src, ang_s, t_dst), scale * PHYSION_LENGTH, np.where(is_static, 1., mass * PHYSION_LENGTH ** 3),
                           friction, bounce, is_static, shape, [s == 'cube' for s in shape], kind, f'physion/{Path(name).stem}'[:80], slots=8)
    episode['_did_fall'] = did_fall
    return episode


def convert_physion(root: Path) -> tuple[list[dict], dict]:
    files = sorted(glob.glob(str(root / 'physion' / '*' / '*' / '*' / '*.hdf5')))
    episodes, skipped, labels = [], [], {}
    for name in files:
        try: episode = _physion_file(name)
        except (OSError, KeyError, ValueError) as error:
            skipped.append(f'{Path(name).name}: {type(error).__name__}'); continue
        if episode is None or int(episode['mask'].sum()) > 8: skipped.append(Path(name).name); continue
        labels[episode['source_id']] = episode.pop('_did_fall'); episodes.append(episode)
    info = dict(source=str(root / 'physion'), simulator='TDW / PhysX (Physion)', files=len(files), skipped=skipped, did_fall_label=labels,
                velocities='central finite difference at 100 Hz, interpolated', rescale=dict(length=PHYSION_LENGTH, time=PHYSION_TIME, mass=PHYSION_LENGTH ** 3),
                start_frames_after_push=PHYSION_START_AFTER_PUSH)
    return episodes, info


# ----------------------------------------------------------------------------- main
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--external', type=Path, default=ROOT / 'data' / 'external')
    parser.add_argument('--out', type=Path, default=ROOT / 'data' / 'external_v2')
    parser.add_argument('--datasets', nargs='+', default=['kubric_movi_a', 'clevrer', 'physion'])
    parser.add_argument('--clevrer-max', type=int, default=1000)
    parser.add_argument('--kubric-max', type=int, default=None)
    parser.add_argument('--kubric-fd', action='store_true', help='finite-difference velocities instead of the dataset ones')
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True); summary = {}
    for name in args.datasets:
        began = time.time()
        if name == 'kubric_movi_a': episodes, info = convert_kubric(args.external, args.kubric_fd, args.kubric_max)
        elif name == 'clevrer': episodes, info = convert_clevrer(args.external, args.clevrer_max, min(args.workers, 24))
        elif name == 'physion': episodes, info = convert_physion(args.external)
        else: raise SystemExit(f'unknown dataset {name}')
        if not episodes: print(f'{name}: no episodes converted'); continue
        meta = write_dataset(args.out, name, episodes, info)
        summary[name] = meta
        print(f"{name}: {meta['episodes']} episodes, {meta['dynamic_objects']} dynamic objects ({meta['box_like_objects']} box-like), "
              f"tipped {meta['tipped_fraction']}, settled {meta['settled_fraction']}, init penetration {meta['init_penetration_m']} "
              f"[{time.time() - began:.1f}s]")
    existing = json.loads((args.out / 'meta.json').read_text()) if (args.out / 'meta.json').exists() else {}
    datasets = {**existing.get('datasets', {}), **{k: dict(episodes=v['episodes'], path=str(args.out / k / 'test')) for k, v in summary.items()}}
    (args.out / 'meta.json').write_text(json.dumps(dict(note=NOTE, step_s=STEP, datasets=datasets), indent=2))


if __name__ == '__main__':
    main()
