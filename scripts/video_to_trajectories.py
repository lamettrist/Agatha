"""RGB video -> APPROXIMATE per-object 3D trajectories in Agatha's episode-shard format ("training on video").

    CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/video_to_trajectories.py \
        --sources kubric clevrer deskspace --out data/video_v1 --workers 20 \
        [--clevrer-videos DIR_WITH_video_1xxxx.mp4]

The learned world model is STATE based (3D box poses).  Video therefore enters training only after it has
been converted to object trajectories by segmentation + tracking + geometry, and those trajectories are
MUCH noisier than simulator states (the measured error is written to <out>/<source>/meta.json and
docs/video-training.md).  Intended use: a small loss weight, position-only, noise-tolerant loss; keep the
result only if the simulator metrics do not get worse.

WHAT COMES FROM PIXELS, WHAT DOES NOT
  from pixels      object positions over time (hence velocities), which frames an object is observed in.
  from metadata    the static attribute list of each video (colour, shape, size class, material) - it says WHICH
                   objects to look for and what box / mass / friction to give them; the camera of Kubric
                   (pose + focal length per episode).  No annotated position, velocity or mask is read for the
                   videos that are converted.
  from calibration CLEVRER: ONE image->ground homography for the fixed camera, fitted on the annotated
                   positions of a handful of CALIBRATION videos that are in no split.  Kubric: the
                   area-vs-depth gate thresholds were chosen from an error-vs-ground-truth diagnostic on the first
                   200 episodes; only the first 40 are kept out of every split (the rest include 10 test videos).
  assumed          physics parameters: identical to scripts/prepare_external_data.py (K1/K3, C1/C3), so the
                   only difference to the dataset-ground-truth shards is where the states come from.

PIPELINE
  1. background: per-pixel median over frames of many different videos (pixels only);
  2. colour segmentation against the CLEVR palette (both datasets use the same 8 colours): a pixel is
     'coloured' if its chromaticity differs from the background's, and gets the palette colour (restricted to
     the colours in the video's attribute list) with the most similar hue direction; gray objects are
     low-chroma pixels much darker / brighter than the background (shadows are low-chroma and a bit darker);
  3. connected components per colour, the k largest for a colour with k objects; nearest-neighbour tracking
     inside a colour group; a group whose members cannot be told apart (same colour, different shape, equal
     size) is DROPPED (never observed); different sizes are told apart by apparent size;
  4. lifting: Kubric - camera ray through the mask centroid intersected with the plane z = object half
     height (object assumed to sit on the floor); a frame is 'observed' only if the apparent size agrees
     with the size class at that depth (airborne objects are nearer than the floor hit point and look too
     big; occluded ones look too small).  CLEVRER - homography (planar motion);
  5. unobserved frames hold / interpolate the neighbouring observed positions (obs_mask False), 20 Hz
     resampling exactly as the converter, velocities = least-squares slope over +-2 frames, orientation =
     identity (has_orientation False everywhere: orientation is NOT estimated), angular velocity 0.

EXTRA KEYS (beyond data/v2 and data/external_v2): obs_mask bool [E,T+1,N], pos_sigma float32 [E,N] (m; 68th
percentile horizontal error of that source / shape on the calibration videos), has_orientation bool [E,N],
source <U24 [E], video_derived bool [E], n_untracked int64 [E] (objects of the video that are NOT in `mask`
because they were never tracked: the scene is then incomplete).
Splits are by video.  <out>/<source>/test_gt holds the DATASET GROUND TRUTH (converter output) of the
same held-out videos, slot-aligned with test/.
"""
import argparse
import hashlib
import itertools
import json
import math
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(Path(__file__).resolve().parent)):
    if _p not in sys.path: sys.path.insert(0, _p)
import numpy as np
import prepare_external_data as ped
from prepare_external_data import STEP, MAX_STEPS, SUPPORT_HEIGHT, interp_linear, make_episode, zup_to_yup_vec

SLOTS = 8
NOTE = ('APPROXIMATE trajectories derived from RGB video by colour segmentation, tracking and ground-plane geometry; '
        'far noisier than simulator states; orientation not estimated. See scripts/video_to_trajectories.py.')
# CLEVR material colours (sRGB), used by both CLEVRER and Kubric MOVi-A
PALETTE = dict(gray=(87, 87, 87), red=(173, 35, 35), blue=(42, 75, 215), green=(29, 105, 20), brown=(129, 74, 25),
               purple=(129, 38, 192), cyan=(41, 208, 208), yellow=(255, 238, 51))
KUBRIC_COLOURS = ('blue', 'brown', 'cyan', 'gray', 'green', 'purple', 'red', 'yellow')
KUBRIC_DENSITY = {0: 2.7, 1: 1.1}                       # metal, rubber (dataset_info.json)
KUBRIC_MATERIAL = {0: (.4, .3), 1: (.8, .7)}           # friction, restitution
KUBRIC_SIZE = {0: .7, 1: 1.4}                          # units
KUBRIC_VOLUME = dict(cube=1., cylinder=math.pi / 4, sphere=math.pi / 6)   # x size^3
KUBRIC_GATE = (.90, 1.06)                               # apparent-size ratio accepted as 'on the floor, unoccluded'


# ----------------------------------------------------------------------------- geometry (unit tested)
def quat_wxyz_matrix(q):
    w, x, y, z = np.asarray(q, np.float64) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def pixel_rays(uv, rotation, focal, sensor, aspect=1.):
    """uv [...,2] normalised image coordinates (x right, y DOWN, in 0..1) -> world ray directions [...,3].
    Blender / Kubric camera: looks along its -Z, +Y up; `rotation` is camera-to-world."""
    uv = np.asarray(uv, np.float64)
    cam = np.stack(((uv[..., 0] - .5) * sensor, -(uv[..., 1] - .5) * sensor / aspect, np.full(uv.shape[:-1], -float(focal))), -1)
    return cam @ np.asarray(rotation, np.float64).T


def project(points, position, rotation, focal, sensor, aspect=1.):
    """World points [...,3] -> (uv [...,2] normalised, depth along the optical axis)."""
    cam = (np.asarray(points, np.float64) - position) @ np.asarray(rotation, np.float64)
    depth = -cam[..., 2]
    u = cam[..., 0] / depth * focal / sensor + .5; v = -cam[..., 1] / depth * focal / sensor * aspect + .5
    return np.stack((u, v), -1), depth


def ray_plane(origin, directions, height, axis=2):
    """Intersect rays origin + s * d with the plane x[axis] = height. Returns (points [...,3], valid)."""
    directions = np.asarray(directions, np.float64); origin = np.asarray(origin, np.float64)
    denom = directions[..., axis]
    with np.errstate(divide='ignore', invalid='ignore'): s = (height - origin[axis]) / denom
    valid = np.isfinite(s) & (s > 0)
    return origin + np.where(valid, s, np.nan)[..., None] * directions, valid


def fit_homography(src, dst):
    """Least-squares (normalised DLT) homography H with dst ~ H @ [src, 1]. src, dst [M,2], M >= 4."""
    src = np.asarray(src, np.float64); dst = np.asarray(dst, np.float64)
    def norm(p):
        c = p.mean(0); s = math.sqrt(2) / max(np.linalg.norm(p - c, axis=1).mean(), 1e-12)
        return np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1]])
    ts, td = norm(src), norm(dst)
    a = apply_homography(ts, src); b = apply_homography(td, dst); rows = []
    for (x, y), (u, v) in zip(a, b):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u]); rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    h = np.linalg.svd(np.asarray(rows))[2][-1].reshape(3, 3)
    h = np.linalg.inv(td) @ h @ ts
    return h / h[2, 2]


def apply_homography(h, points):
    p = np.asarray(points, np.float64); q = np.concatenate((p, np.ones((*p.shape[:-1], 1))), -1) @ np.asarray(h).T
    return q[..., :2] / q[..., 2:]


def fit_homography_ransac(src, dst, threshold, iterations=400, seed=0):
    rng = np.random.default_rng(seed); src = np.asarray(src, np.float64); dst = np.asarray(dst, np.float64)
    best = np.zeros(len(src), bool)
    for _ in range(iterations):
        pick = rng.choice(len(src), 4, replace=False)
        try: h = fit_homography(src[pick], dst[pick])
        except np.linalg.LinAlgError: continue
        with np.errstate(all='ignore'): inlier = np.linalg.norm(apply_homography(h, src) - dst, axis=1) < threshold
        if inlier.sum() > best.sum(): best = inlier
    return fit_homography(src[best], dst[best]), best


# ----------------------------------------------------------------------------- time series (unit tested)
def fill_gaps(pos, obs):
    """pos [F,3] with garbage where ~obs -> linear interpolation inside gaps, hold at the ends. None if never observed."""
    obs = np.asarray(obs, bool); index = np.flatnonzero(obs)
    if not len(index): return None
    frames = np.arange(len(obs))
    return np.stack([np.interp(frames, index, np.asarray(pos, np.float64)[index, k]) for k in range(3)], -1)


def resample(t_src, pos, obs, t_dst):
    """Linear resampling; a destination sample is observed only if BOTH bracketing source frames are."""
    lo, hi, _ = ped._bracket(t_src, t_dst); obs = np.asarray(obs, bool)
    return interp_linear(t_src, pos, t_dst), obs[lo] & obs[hi]


def smooth_velocity(pos, dt, half=2):
    """Least-squares slope over +-half frames (window shrinks at the ends). pos [F,...,3]."""
    pos = np.asarray(pos, np.float64); out = np.zeros_like(pos); count = len(pos)
    for f in range(count):
        lo, hi = max(0, f - half), min(count, f + half + 1)
        if hi - lo < 2: continue
        t = (np.arange(lo, hi) - f) * dt; t = t - t.mean()
        out[f] = np.tensordot(t, pos[lo:hi] - pos[lo:hi].mean(0), axes=(0, 0)) / (t * t).sum()
    return out


# ----------------------------------------------------------------------------- segmentation / tracking
def _hue_dir(rgb):
    c = np.asarray(rgb, np.float64); c = c / np.clip(c.sum(-1, keepdims=True), 1e-6, None) - 1 / 3
    return c / np.clip(np.linalg.norm(c, axis=-1, keepdims=True), 1e-6, None)


PALETTE_DIR = {k: _hue_dir(v) for k, v in PALETTE.items() if k != 'gray'}


def colour_label_image(rgb, background, present, sat_min=.045, diff_min=22.):
    """rgb, background float [H,W,3] -> int label image: -1 none, else index into `present` (colour names)."""
    lum, lum_bg = rgb.mean(-1), background.mean(-1)
    chroma = rgb / np.clip(rgb.sum(-1, keepdims=True), 1e-6, None)
    chroma_bg = background / np.clip(background.sum(-1, keepdims=True), 1e-6, None)
    sat = np.linalg.norm(chroma - chroma_bg, axis=-1)
    label = np.full(rgb.shape[:2], -1, np.int32)
    coloured = [i for i, c in enumerate(present) if c != 'gray']
    if coloured:
        direction = _hue_dir(rgb)
        score = np.stack([direction @ PALETTE_DIR[present[i]] for i in coloured], -1)
        # brown and yellow / red share hue directions under shading: break ties with brightness
        names = [present[i] for i in coloured]
        ratio = lum / np.clip(lum_bg, 1., None)
        if 'brown' in names and 'yellow' in names:
            score[..., names.index('yellow')] += np.where(ratio > 1.05, .08, -.08)
        best = score.argmax(-1)
        is_col = (sat > sat_min) & (lum > 12) & (score.max(-1) > .5)
        label[is_col] = np.asarray(coloured)[best[is_col]]
    if 'gray' in present:
        ratio = lum / np.clip(lum_bg, 1., None)
        diff = np.abs(rgb - background).max(-1)
        is_gray = (sat <= sat_min * .7) & (diff > diff_min) & ((ratio < .62) | (ratio > 1.13))
        label[is_gray & (label < 0)] = present.index('gray')
    return label


def detections(label, index, k, min_area, open_kernel=0):
    """Up to k largest connected components of label == index -> rows (u, v, area, x0, y0, x1, y1) in pixels."""
    import cv2
    mask = (label == index).astype(np.uint8)
    if open_kernel: mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((open_kernel, open_kernel), np.uint8))
    count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    rows = [(centroids[i][0], centroids[i][1], stats[i, 4], stats[i, 0], stats[i, 1], stats[i, 0] + stats[i, 2] - 1, stats[i, 1] + stats[i, 3] - 1)
            for i in range(1, count) if stats[i, 4] >= min_area]
    rows.sort(key=lambda r: -r[2])
    return np.asarray(rows[:k], np.float64).reshape(-1, 7)


def track_group(per_frame, k, gate):
    """per_frame: list of [m_f, 7] detections (m_f <= k) -> [F, k, 7] (nan where a track has no detection).
    Nearest-neighbour assignment (exhaustive over permutations, k is small); the gate grows while a track is lost."""
    frames = len(per_frame); out = np.full((frames, k, 7), np.nan)
    last = np.full((k, 2), np.nan); missed = np.zeros(k, int)
    for f, det in enumerate(per_frame):
        m = len(det)
        if m:
            best, best_cost = None, np.inf
            for perm in itertools.permutations(range(k), m):               # perm[j] = track of detection j
                cost = 0.
                for j, t in enumerate(perm):
                    if np.isnan(last[t, 0]): cost += gate * .9
                    else:
                        d = float(np.linalg.norm(det[j, :2] - last[t])); limit = gate * min(1 + missed[t], 4)
                        cost += d if d <= limit else 10 * gate + d
                if cost < best_cost: best, best_cost = perm, cost
            for j, t in enumerate(best):
                if not np.isnan(last[t, 0]) and np.linalg.norm(det[j, :2] - last[t]) > gate * min(1 + missed[t], 4): continue
                out[f, t] = det[j]; last[t] = det[j, :2]; missed[t] = -1
        missed += 1
    return out


def segment_video(frames, background, colours, min_area, gate, open_kernel=0):
    """frames: iterable of RGB uint8 images; colours: per-object colour names. -> tracks [F, n_objects, 7] where the
    objects of one colour are in ARBITRARY order inside their colour group, plus flags [F, n, 2] (border, occluded)."""
    present = sorted(set(colours)); groups = {c: [i for i, x in enumerate(colours) if x == c] for c in present}
    per_colour = {c: [] for c in present}; shape = None
    for frame in frames:
        rgb = frame.astype(np.float32); shape = rgb.shape[:2]
        label = colour_label_image(rgb, background, present)
        for ci, c in enumerate(present): per_colour[c].append(detections(label, ci, len(groups[c]), min_area, open_kernel))
    count = len(per_colour[present[0]]) if present else 0
    tracks = np.full((count, len(colours), 7), np.nan)
    for c in present:
        k = len(groups[c])
        if k == 1:
            for f, det in enumerate(per_colour[c]):
                if len(det): tracks[f, groups[c][0]] = det[0]
        else: tracks[:, groups[c]] = track_group(per_colour[c], k, gate)
    flags = np.zeros((count, len(colours), 2), bool)
    if count:
        h, w = shape; x0, y0, x1, y1 = (tracks[..., i] for i in (3, 4, 5, 6))
        with np.errstate(invalid='ignore'):
            flags[..., 0] = (x0 <= 1) | (y0 <= 1) | (x1 >= w - 2) | (y1 >= h - 2)
            # occluded: another object's box touches this one and reaches lower in the image (nearer to the camera)
            touch = ((x0[:, :, None] - 2 <= x1[:, None, :]) & (x1[:, :, None] + 2 >= x0[:, None, :]) &
                     (y0[:, :, None] - 2 <= y1[:, None, :]) & (y1[:, :, None] + 2 >= y0[:, None, :]) & (y1[:, None, :] > y1[:, :, None]))
        touch &= ~np.eye(len(colours), dtype=bool)[None]
        flags[..., 1] = touch.any(-1)
    return tracks, flags


# ----------------------------------------------------------------------------- episode assembly
def build_episode(pos_src, obs_src, t_src, t_dst, size, mass, friction, restitution, shape, kind, source_id, source, sigma, **context):
    """pos_src [F,n,3] Y-up metres with the ground at y = 0 (garbage where ~obs_src [F,n]) -> episode dict (SLOTS slots)."""
    count = pos_src.shape[1]; steps = len(t_dst)
    pos = np.zeros((steps, count, 3)); obs = np.zeros((steps, count), bool); tracked = np.zeros(count, bool)
    for i in range(count):
        filled = fill_gaps(pos_src[:, i], obs_src[:, i])
        if filled is None: pos[:, i] = [0., size[i][1] / 2, 0.]; continue
        pos[:, i], obs[:, i] = resample(t_src, filled, obs_src[:, i], t_dst); tracked[i] = obs[:, i].any()
    vel = smooth_velocity(pos, STEP); quat = np.tile([0., 0., 0., 1.], (steps, count, 1))
    episode = make_episode(pos, quat, vel, np.zeros_like(pos), size, mass, friction, restitution, np.zeros(count, bool), shape,
                           [s == 'cube' for s in shape], kind, source_id, slots=SLOTS, **context)
    frames = episode['n_frames']; pad = SLOTS - count
    obs_mask = np.zeros((MAX_STEPS + 1, SLOTS), bool); obs_mask[:frames, :count] = obs[:frames]
    episode['mask'] = episode['mask'] & np.r_[tracked, np.zeros(pad, bool)]
    for key in ('tipped', 'moved_far', 'settled', 'fell'): episode[key] = episode[key] & episode['mask']
    for key in ('contact_support', 'rest'): episode[key] = episode[key] & episode['mask'][None]
    episode.update(obs_mask=obs_mask, pos_sigma=np.r_[np.asarray(sigma, np.float32), np.zeros(pad, np.float32)].astype(np.float32),
                   has_orientation=np.zeros(SLOTS, bool), source=source, video_derived=True, n_untracked=int((~tracked).sum()))
    return episode


def gt_to_slots(episode):
    """Converter (ground-truth) episode with any slot count -> SLOTS slots (objects beyond SLOTS never occur: filtered before)."""
    out = {}
    slots = episode['mask'].shape[0]
    for key, value in episode.items():
        value = np.asarray(value)
        axis = 1 if key in ('states', 'contact_support', 'contact_floor', 'contact_object', 'rest') else 0 if value.ndim and value.shape[0] == slots and key != 'context' else None
        if axis is None: out[key] = value; continue
        if slots >= SLOTS: out[key] = np.take(value, np.arange(SLOTS), axis=axis); continue
        shape = list(value.shape); shape[axis] = SLOTS - slots
        fill = np.zeros(shape, value.dtype)
        if key == 'states': fill[..., 6] = 1
        if key == 'props': fill[..., 3] = 1
        out[key] = np.concatenate((value, fill), axis)
    out.update(obs_mask=np.broadcast_to(out['mask'][None], (MAX_STEPS + 1, SLOTS)) & (np.arange(MAX_STEPS + 1)[:, None] < int(out['n_frames'])),
               pos_sigma=np.zeros(SLOTS, np.float32), has_orientation=out['mask'].copy(), source='dataset_gt', video_derived=False, n_untracked=0)
    return out


def write_split(target: Path, episodes, shard_size=500):
    target.mkdir(parents=True, exist_ok=True)
    for old in target.glob('shard_*.npz'): old.unlink()
    if not episodes: return 0
    data = {k: np.stack([np.asarray(e[k]) for e in episodes]) for k in episodes[0]}
    data['kind'] = data['kind'].astype('<U24'); data['source_id'] = data['source_id'].astype('<U80'); data['source'] = data['source'].astype('<U24')
    data['episode_index'] = np.arange(len(episodes), dtype=np.int64)
    try: data['init_penetration'] = ped.init_penetration(data)
    except Exception: data['init_penetration'] = np.zeros(len(episodes), np.float32)
    for index, start in enumerate(range(0, len(episodes), shard_size)):
        np.savez_compressed(target / f'shard_{index:05d}.npz', **{k: v[start:start + shard_size] for k, v in data.items()})
    return len(episodes)


def split_of(name: str, fractions=(.8, .1)) -> str:
    value = int(hashlib.sha1(name.encode()).hexdigest()[:8], 16) / 0xffffffff
    return 'train' if value < fractions[0] else 'val' if value < fractions[0] + fractions[1] else 'test'


# ----------------------------------------------------------------------------- evaluation against dataset ground truth
def _group_permutations(groups):
    for perms in itertools.product(*[itertools.permutations(g) for g in groups]):
        yield [i for p in perms for i in p]


def evaluate(pairs, visible=None):
    """pairs: list of (video episode, gt episode [same slot order], interchangeable groups [lists of slots]).
    Errors are 3D position errors in metres on frames < n_frames.  Objects that are interchangeable (same colour and
    attributes) are matched to the ground truth by the permutation with the lowest mean error (evaluation only)."""
    err = {k: [] for k in ('all', 'cube')}; rest = {k: [] for k in ('all', 'cube')}; horiz = {}
    seen = dict(all=0, cube=0); total = dict(all=0, cube=0); vis_total = dict(all=0, cube=0); vis_seen = dict(all=0, cube=0)
    switches = 0; multi_tracks = 0; objects = dict(all=0, cube=0); objects_tracked = dict(all=0, cube=0)
    for index, (video, gt, groups) in enumerate(pairs):
        frames = int(gt['n_frames']); count = int(gt['mask'].sum())
        pv, pg = video['states'][:frames, :, :3].astype(np.float64), gt['states'][:frames, :, :3].astype(np.float64)
        obs = video['obs_mask'][:frames]; order = np.arange(SLOTS)
        for group in groups:
            if len(group) < 2: continue
            best, best_cost = None, np.inf
            for perm in itertools.permutations(group):            # video slot group[j] <-> gt slot perm[j]
                d = np.linalg.norm(pv[:, group] - pg[:, list(perm)], axis=-1); o = obs[:, group]
                cost = d[o].mean() if o.any() else 0.
                if cost < best_cost: best, best_cost = perm, cost
            order[group] = best
            # identity switches: the nearest ground-truth object of the group changes along a track
            d_all = np.linalg.norm(pv[:, group, None] - pg[:, None, group], axis=-1)        # [F, track, gt]
            for j in range(len(group)):
                o = obs[:, group[j]]
                if o.sum() < 2: continue
                near = d_all[o, j].argmin(-1); multi_tracks += 1; switches += int((np.diff(near) != 0).sum())
        d = np.linalg.norm(pv - pg[:, order], axis=-1)
        vis = visible[index][:frames] if visible is not None else np.ones((frames, SLOTS), bool)
        for i in range(count):
            g = int(order[i]); cube = bool(gt['box_like'][g])
            for key in ('all', 'cube') if cube else ('all',):
                err[key].extend(d[obs[:, i], i].tolist()); seen[key] += int(obs[:, i].sum()); total[key] += frames
                vis_total[key] += int(vis[:, g].sum()); vis_seen[key] += int((obs[:, i] & vis[:, g]).sum())
                objects[key] += 1; objects_tracked[key] += int(obs[:, i].any())
                if bool(gt['settled'][g]) and obs[max(frames - 3, 0):frames, i].all():
                    rest[key].append(float(np.linalg.norm(pv[frames - 3:frames, i].mean(0) - gt['rest_pos'][g])))
            horiz.setdefault(str(gt['shape'][g]), []).extend(np.linalg.norm((pv[:, i] - pg[:, g])[obs[:, i]][:, [0, 2]], axis=-1).tolist())
    def stats(values):
        v = np.asarray(values)
        return dict(n=int(len(v)), median_cm=float(np.median(v) * 100), p90_cm=float(np.percentile(v, 90) * 100), mean_cm=float(v.mean() * 100)) if len(v) else dict(n=0)
    out = dict(videos=len(pairs))
    for key in ('all', 'cube'):
        out[key] = dict(position_error=stats(err[key]), rest_position_error=stats(rest[key]),
                        recall_all_object_frames=seen[key] / max(total[key], 1), recall_visible_object_frames=vis_seen[key] / max(vis_total[key], 1),
                        objects=objects[key], objects_tracked_at_all=objects_tracked[key], observed_object_frames=seen[key], gt_object_frames=total[key])
    out['identity_switches'] = dict(switches=switches, tracks_in_same_colour_groups=multi_tracks)
    out['sigma_p68_horizontal_m'] = {k: float(np.percentile(v, 68)) for k, v in horiz.items() if len(v)}
    return out


def transitions(episodes):
    """(cube object, step) pairs observed at both t and t+1, and episodes in which every object is tracked."""
    cube = obs_pairs = full = 0
    for e in episodes:
        frames = int(e['n_frames']); target = e['mask'] & e['box_like']; o = e['obs_mask'][:frames]
        obs_pairs += int((o[:-1] & o[1:])[:, target].sum()); cube += int(target.sum()); full += int(e['n_untracked'] == 0)
    return dict(episodes=len(episodes), cube_tracks=cube, cube_transitions_observed=obs_pairs, episodes_with_every_object_tracked=full)


# ----------------------------------------------------------------------------- Kubric MOVi-A
def _kubric_frames(root, shard, name, count=24):
    import cv2
    with zipfile.ZipFile(root / 'kubric-movi' / 'video' / f'train-{shard:05d}.zip') as z:
        return [cv2.cvtColor(cv2.imdecode(np.frombuffer(z.read(f'{name}/{f:02d}.png'), np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) for f in range(count)]


def _kubric_segment(args):
    root, shard, name, colours, background = args
    return segment_video(_kubric_frames(root, shard, name), background, colours, min_area=10, gate=28.)


def kubric_lift(track, flags, cam_pos, cam_quat, focal, sensor, size_units, width=128):
    """track [F,7], flags [F,2] -> (world Z-up positions [F,3] on the plane z = size/2, ratio [F]) ; ratio = apparent
    size / size expected for an object of this size class at the floor hit point (1 = consistent)."""
    rotation = quat_wxyz_matrix(cam_quat)
    uv = (track[:, :2] + .5) / width
    points, valid = ray_plane(cam_pos, pixel_rays(uv, rotation, focal, sensor), size_units / 2)
    depth = -((points - cam_pos) @ rotation)[:, 2]
    with np.errstate(invalid='ignore'): ratio = np.sqrt(track[:, 2]) * depth / (focal / sensor * width * size_units)
    return points, np.where(valid, ratio, np.nan)


def convert_kubric_video(root: Path, out: Path, workers: int, limit=None, calibration=40):
    import multiprocessing as mp
    with np.load(root / 'kubric-movi' / 'movi_a_128_train_subset.npz', allow_pickle=False) as z: a = {k: z[k][:limit] for k in z.files}
    gt_all, _ = ped.convert_kubric(root, limit=limit)
    count = len(a['mask']); n_obj = a['mask'].sum(1)
    names = [f"movi_a/train/shard{int(a['shard'][e]):05d}/{a['video_name'][e]}" for e in range(count)]
    # background: per-pixel median of one frame from each of 150 episodes (pixels only)
    rng = np.random.default_rng(0); sample = rng.choice(count, min(150, count), replace=False)
    background = np.median(np.stack([_kubric_frames(root, int(a['shard'][e]), str(a['video_name'][e]))[int(rng.integers(24))] for e in sample]).astype(np.float32), 0)
    colours = [[KUBRIC_COLOURS[c] for c in a['color_label'][e][a['mask'][e]]] for e in range(count)]
    jobs = [(root, int(a['shard'][e]), str(a['video_name'][e]), colours[e], background) for e in range(count)]
    with mp.get_context('fork').Pool(workers) as pool: segmented = pool.map(_kubric_segment, jobs, chunksize=8)
    t = np.arange(24) * STEP
    shape_names = [[ped.KUBRIC_SHAPES[int(s)] for s in a['shape_label'][e][a['mask'][e]]] for e in range(count)]
    # pass 1: slot assignment inside colour groups + apparent-size ratio; k per shape is self-calibrated from late frames (pixels only)
    lifted = []; ratios = {s: [] for s in KUBRIC_VOLUME}; dropped_ambiguous = dropped_gray = 0
    for e in range(count):
        tracks, flags = segmented[e]; tracks = tracks.copy(); n = int(n_obj[e]); sizes = a['size_label'][e][a['mask'][e]]
        groups = {}; ok = np.array([c != 'gray' for c in colours[e]])      # gray objects on the gray floor: segmentation too unreliable, dropped
        dropped_gray += int((~ok).sum())
        for i, c in enumerate(colours[e]): groups.setdefault(c, []).append(i)
        inter = []
        for c, g in groups.items():
            if len(g) == 1 or c == 'gray': continue
            attrs = [(shape_names[e][i], int(sizes[i])) for i in g]
            if len(set(attrs)) == 1: inter.append(g); continue
            if len(set(s for _, s in attrs)) == len(g):                 # all sizes differ: order tracks by apparent size
                scale = []
                for j, i in enumerate(g):
                    _, r = kubric_lift(tracks[:, i], flags[:, i], a['camera_positions'][e, 0], a['camera_quaternions_wxyz'][e, 0], 35., 32., .7)
                    scale.append(np.nanmedian(r[12:]) if np.isfinite(r[12:]).any() else -1.)
                by_track = np.argsort(scale); by_slot = sorted(range(len(g)), key=lambda j: sizes[g[j]])
                new = tracks[:, g].copy()
                for rank in range(len(g)): new[:, by_slot[rank]] = tracks[:, g[by_track[rank]]]
                tracks[:, g] = new
            else: ok[g] = False; dropped_ambiguous += len(g)
        points = np.zeros((24, n, 3)); ratio = np.full((24, n), np.nan)
        for i in range(n):
            points[:, i], ratio[:, i] = kubric_lift(tracks[:, i], flags[:, i], a['camera_positions'][e, 0], a['camera_quaternions_wxyz'][e, 0],
                                                    float(a['camera_focal_length'][e]), float(a['camera_sensor_width'][e]), KUBRIC_SIZE[int(sizes[i])])
            if ok[i]:
                good = np.isfinite(ratio[18:, i]) & ~flags[18:, i].any(-1)
                ratios[shape_names[e][i]].extend(ratio[18:, i][good].tolist())
        lifted.append((points, ratio, flags, ok, inter))
    k_shape = {s: float(np.median(v)) for s, v in ratios.items()}
    # pass 2: gate, assemble
    calib = set(range(min(calibration, count)))                           # in no split (gate thresholds were chosen on these)
    splits = {s: [] for s in ('train', 'val', 'test', 'calibration')}; gts = {s: [] for s in splits}; pairs = {s: [] for s in splits}; vis = {s: [] for s in splits}
    skipped_slots = 0; mass_ratio = []
    for e in range(count):
        n = int(n_obj[e])
        if n > SLOTS: skipped_slots += 1; continue
        points, ratio, flags, ok, inter = lifted[e]; keep = np.flatnonzero(a['mask'][e]); sizes = a['size_label'][e][keep]; mats = a['material_label'][e][keep]
        k = np.array([k_shape[s] for s in shape_names[e]])
        with np.errstate(invalid='ignore'):
            obs = np.isfinite(ratio) & ~flags.any(-1) & (ratio > KUBRIC_GATE[0] * k) & (ratio < KUBRIC_GATE[1] * k) & ok[None]
        obs = _drop_isolated(obs, 4)
        pos = zup_to_yup_vec(np.nan_to_num(points)) * ped.KUBRIC_LENGTH
        edge = np.array([KUBRIC_SIZE[int(s)] for s in sizes]) * ped.KUBRIC_LENGTH; size = np.repeat(edge[:, None], 3, 1)
        mass = np.array([KUBRIC_DENSITY[int(m)] * KUBRIC_VOLUME[sh] * KUBRIC_SIZE[int(s)] ** 3 for m, sh, s in zip(mats, shape_names[e], sizes)]) * ped.KUBRIC_MASS
        mass_ratio.extend((mass / (a['mass'][e][keep] * ped.KUBRIC_MASS)).tolist())
        friction = np.array([KUBRIC_MATERIAL[int(m)][0] for m in mats]); restitution = np.array([KUBRIC_MATERIAL[int(m)][1] for m in mats])
        episode = build_episode(pos, obs, t, t, size, mass, friction, restitution, shape_names[e], 'movi_a_toss_video', names[e], 'kubric_movi_a_video',
                                np.zeros(n), gravity=10., support_friction=ped.KUBRIC_FLOOR_FRICTION, support_restitution=ped.KUBRIC_FLOOR_RESTITUTION)
        which = 'calibration' if e in calib else split_of(names[e])
        gt = gt_to_slots(gt_all[e]); visible = np.zeros((24, SLOTS), bool); visible[:, :n] = (a['visibility'][e][keep] > 80).T
        splits[which].append(episode); gts[which].append(gt); pairs[which].append((episode, gt, inter)); vis[which].append(visible)
    sigma = evaluate(pairs['calibration'], vis['calibration'])['sigma_p68_horizontal_m']
    for which in splits:
        for episode in splits[which]:
            episode['pos_sigma'] = np.array([sigma.get(str(s), 0.) if s else 0. for s in episode['shape']], np.float32)
    target = out / 'kubric_movi_a'
    for which in ('train', 'val', 'test'): write_split(target / which, splits[which])
    write_split(target / 'test_gt', gts['test'])
    meta = dict(note=NOTE, source='kubric_movi_a_video', video='data/external/kubric-movi/video/*.zip (128x128 RGB, 24 frames)', step_s=STEP, max_objects=SLOTS,
                splits={w: transitions(splits[w]) for w in splits}, split_rule='sha1(source_id) 80/10/10; first 40 episodes = calibration (in no split)',
                test_source_ids=[str(e['source_id']) for e in splits['test']],
                leakage_note='data/external_v2/kubric_movi_a/test contains ALL 1137 episodes; a model trained on this train split must be evaluated on test_gt/ (or on the test_source_ids subset), not on the whole external_v2 set.',
                episodes_skipped_more_than_8_objects=skipped_slots, objects_dropped_ambiguous_colour_group=dropped_ambiguous, objects_dropped_gray=dropped_gray,
                apparent_size_constant=k_shape, gate=KUBRIC_GATE, pos_sigma_m=sigma,
                size_class_mass_vs_metadata=dict(median_ratio=float(np.median(mass_ratio)), min=float(np.min(mass_ratio)), max=float(np.max(mass_ratio))),
                accuracy={w: evaluate(pairs[w], vis[w]) for w in pairs if pairs[w]},
                params='as scripts/prepare_external_data.py K1/K3: length x0.36, time x0.6, gravity 10, ground friction 0.3 / restitution 0.5; '
                       'object friction / restitution from the material label; size 0.252 / 0.504 m cube from the size label; mass = density x volume x 0.36',
                extra_keys='obs_mask, pos_sigma, has_orientation, source, video_derived, n_untracked')
    (target / 'meta.json').write_text(json.dumps(meta, indent=1))
    return meta


def _drop_isolated(obs, min_run=3):
    """Remove observed runs shorter than min_run frames (single-frame matches are mostly wrong)."""
    obs = obs.copy()
    for i in range(obs.shape[1]):
        f = 0
        while f < len(obs):
            if not obs[f, i]: f += 1; continue
            g = f
            while g < len(obs) and obs[g, i]: g += 1
            if g - f < min_run: obs[f:g, i] = False
            f = g
    return obs


# ----------------------------------------------------------------------------- CLEVRER
def _clevrer_frames(path, stride=1):
    import cv2
    capture = cv2.VideoCapture(str(path)); frames = []
    while True:
        ok, frame = capture.read()
        if not ok: break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    return frames[::stride]


def _clevrer_annotation(root, index):
    lo = index // 1000 * 1000
    return root / 'clevrer' / 'annotation_validation' / f'annotation_{lo}-{lo + 1000}' / f'annotation_{index}.json'


def _clevrer_segment(args):
    path, colours, background = args
    frames = _clevrer_frames(path)
    if len(frames) < 100: return None
    return segment_video(frames, background, colours, min_area=40, gate=45., open_kernel=3)


def convert_clevrer_video(root: Path, videos: Path, out: Path, workers: int, limit=None, calibration=20):
    import multiprocessing as mp
    files = sorted(videos.glob('video_*.mp4'))[:limit]; ids = [int(f.stem.split('_')[1]) for f in files]
    props = []
    for index in ids:
        with open(_clevrer_annotation(root, index)) as handle: props.append(json.load(handle)['object_property'])   # static attributes only
    rng = np.random.default_rng(0); sample = rng.choice(len(files), min(80, len(files)), replace=False)
    background = np.median(np.stack([_clevrer_frames(files[i])[int(rng.integers(100))] for i in sample]).astype(np.float32), 0)
    jobs = [(files[i], [str(o['color']) for o in props[i]], background) for i in range(len(files))]
    with mp.get_context('fork').Pool(workers) as pool: segmented = pool.map(_clevrer_segment, jobs, chunksize=2)
    calib = set(range(len(files) - calibration, len(files)))            # the LAST videos: calibration only, in no split
    def groups_of(p):
        groups = {}
        for i, o in enumerate(p): groups.setdefault(str(o['color']), []).append(i)
        return groups
    # homography pixel -> ground (CLEVRER units) from the annotated positions of the calibration videos (unique colours only)
    src, dst = [], []
    for i in calib:
        if segmented[i] is None: continue
        tracks, flags = segmented[i]
        with open(_clevrer_annotation(root, ids[i])) as handle: annotation = json.load(handle)
        for g in groups_of(props[i]).values():
            if len(g) != 1: continue
            oid = int(props[i][g[0]]['object_id'])
            for f in range(0, min(len(tracks), len(annotation['motion_trajectory'])), 4):
                if np.isnan(tracks[f, g[0], 0]) or flags[f, g[0]].any(): continue
                obj = [o for o in annotation['motion_trajectory'][f]['objects'] if int(o['object_id']) == oid][0]
                src.append(tracks[f, g[0], :2]); dst.append(obj['location'][:2])
    homography, inlier = fit_homography_ransac(np.asarray(src), np.asarray(dst), threshold=.15)
    residual = np.linalg.norm(apply_homography(homography, np.asarray(src)) - np.asarray(dst), axis=1)
    splits = {s: [] for s in ('train', 'val', 'test', 'calibration')}; gts = {s: [] for s in splits}; pairs = {s: [] for s in splits}; vis = {s: [] for s in splits}
    dropped_ambiguous = failed = 0
    n_split = len(files) - calibration
    for i in range(len(files)):
        if segmented[i] is None: failed += 1; continue
        gt = ped._clevrer_file(str(_clevrer_annotation(root, ids[i])))
        if gt is None: failed += 1; continue
        tracks, flags = segmented[i]; n = len(props[i]); frames = len(tracks)
        shapes = [str(o['shape']) for o in props[i]]; ok = np.ones(n, bool); inter = []
        for g in groups_of(props[i]).values():
            if len(g) == 1: continue
            if len(set(shapes[j] for j in g)) == 1: inter.append(g)
            else: ok[g] = False; dropped_ambiguous += len(g)
        obs = np.isfinite(tracks[..., 0]) & ~flags.any(-1) & ok[None]
        obs = _drop_isolated(obs, 5)
        with np.errstate(invalid='ignore'): ground = apply_homography(homography, np.nan_to_num(tracks[..., :2]))
        loc = np.concatenate((ground, np.full((frames, n, 1), ped.CLEVRER_SIZE_UNITS / 2)), -1)
        pos = zup_to_yup_vec(loc) * ped.CLEVRER_LENGTH
        t_src = np.arange(frames) / ped.CLEVRER_FPS
        steps = min(MAX_STEPS, int((t_src[-2] - STEP) / STEP)); t_dst = STEP + np.arange(steps + 1) * STEP
        size = np.full((n, 3), ped.CLEVRER_SIZE_UNITS * ped.CLEVRER_LENGTH)
        episode = build_episode(pos, obs, t_src, t_dst, size, np.full(n, ped.CLEVRER_MASS), np.full(n, ped.CLEVRER_FRICTION), np.zeros(n), shapes,
                                'clevrer_slide_video', f'clevrer/validation/annotation_{ids[i]}', 'clevrer_video', np.zeros(n),
                                gravity=9.81, support_friction=ped.CLEVRER_FRICTION)
        with open(_clevrer_annotation(root, ids[i])) as handle: annotation = json.load(handle)
        inside = np.zeros((MAX_STEPS + 1, SLOTS), bool)
        for step, tt in enumerate(t_dst):
            frame = annotation['motion_trajectory'][min(int(round(tt * ped.CLEVRER_FPS)), len(annotation['motion_trajectory']) - 1)]
            by_id = {int(o['object_id']): o for o in frame['objects']}
            for j, o in enumerate(props[i]): inside[step, j] = bool(by_id[int(o['object_id'])]['inside_camera_view'])
        order = i if i < n_split else None
        which = 'calibration' if i in calib else ('train' if order < .8 * n_split else 'val' if order < .9 * n_split else 'test')
        gt = gt_to_slots(gt)
        splits[which].append(episode); gts[which].append(gt); pairs[which].append((episode, gt, inter)); vis[which].append(inside)
    sigma = evaluate(pairs['calibration'], vis['calibration'])['sigma_p68_horizontal_m']
    for which in splits:
        for episode in splits[which]:
            episode['pos_sigma'] = np.array([sigma.get(str(s), 0.) if s else 0. for s in episode['shape']], np.float32)
    target = out / 'clevrer'
    for which in ('train', 'val', 'test'): write_split(target / which, splits[which])
    write_split(target / 'test_gt', gts['test'])
    meta = dict(note=NOTE, source='clevrer_video', video=f'{len(files)} validation videos {min(ids)}..{max(ids)} (480x320, 25 fps, 128 frames)', step_s=STEP, max_objects=SLOTS,
                splits={w: transitions(splits[w]) for w in splits},
                split_rule=f'by video index order: first 80 % train, next 10 % val, next 10 % test; the last {calibration} videos are calibration only (in no split)',
                test_source_ids=[str(e['source_id']) for e in splits['test']],
                leakage_note='data/external_v2/clevrer/test is built from annotations 10000..10999; none of the videos used here is in that range, so that whole set stays held out.',
                videos_failed=failed, objects_dropped_ambiguous_colour_group=dropped_ambiguous,
                homography=dict(matrix=homography.tolist(), calibration_points=len(src), inlier_fraction=float(inlier.mean()),
                                residual_units=dict(median=float(np.median(residual)), p90=float(np.percentile(residual, 90))),
                                note='pixel centroid -> ground (CLEVRER units, x0.5 = metres); annotations of the calibration videos are used for this fit ONLY'),
                pos_sigma_m=sigma, accuracy={w: evaluate(pairs[w], vis[w]) for w in pairs if pairs[w]},
                params='as scripts/prepare_external_data.py C1/C3: 25 fps, length x0.5, every object a 0.2 m box of 0.5 kg, friction 0.133 (object and support), restitution 0',
                extra_keys='obs_mask, pos_sigma, has_orientation, source, video_derived, n_untracked')
    (target / 'meta.json').write_text(json.dumps(meta, indent=1))
    return meta


# ----------------------------------------------------------------------------- team clip (static scene -> rest episodes)
def convert_team_clip(tracks_dir: Path, out: Path, name='deskspace'):
    """Uses the output of scripts/track_video.py.  The clip is a hand-held walk-around of a desk on which nothing moves:
    there is NO dynamic trajectory in it.  What is written: ONE 'objects at rest stay at rest' episode with the objects
    at their first-frame table-plane positions (the only frame the plane was fitted on), zero velocity, 80 steps."""
    report = json.loads((tracks_dir / 'tracks.json').read_text())
    with np.load(tracks_dir / 'test' / 'shard_00000.npz', allow_pickle=False) as z: states, props, mask, observed, labels, context = (z[k][0] for k in ('states', 'props', 'mask', 'observed', 'shape', 'context'))
    keep = [i for i in np.flatnonzero(mask) if observed[:3, i].any()][:SLOTS]         # tracks that exist in the first frames
    target = out / f'team_{name}'
    episodes = []
    if keep:
        n = len(keep); pos = np.repeat(states[:1, keep, :3].astype(np.float64), MAX_STEPS + 1, 0); pos[..., 1] -= SUPPORT_HEIGHT
        t = np.arange(MAX_STEPS + 1) * STEP
        episode = build_episode(pos, np.ones((MAX_STEPS + 1, n), bool), t, t, props[keep, :3], props[keep, 3], props[keep, 4], np.zeros(n),
                                [str(s) for s in labels[keep]], 'team_clip_rest', f'team/{name}.mp4', 'team_clip_video', np.full(n, .10),
                                support_size=(float(context[0]), float(context[1])))
        episode['box_like'] = np.zeros(SLOTS, bool)
        episodes.append(episode)
    write_split(target / 'train', episodes)
    for which in ('val', 'test'): write_split(target / which, [])
    meta = dict(note=NOTE, source='team_clip_video', clip=report.get('video'), tracker='scripts/track_video.py', dynamic_trajectories=0,
                rest_episodes=len(episodes), objects=[str(labels[i]) for i in keep], tracks_reported=report.get('tracks'),
                camera_shift_fraction_of_width=report.get('camera_shift_px', {}).get('fraction_of_width'), warnings=report.get('warnings'),
                honest_summary='Hand-held clip of a static desk: no object moves, the camera does. No dynamic trajectory, no measured accuracy (no ground truth). '
                               'Only an "at rest stays at rest" episode with detector-class prior sizes / masses, an assumed 1.2 m table and an assumed pos_sigma of 0.10 m; '
                               'box_like is False for every object (bottles etc.), so a cubes-only trainer ignores it.',
                splits=dict(train=transitions(episodes)))
    (target / 'meta.json').write_text(json.dumps(meta, indent=1))
    return meta


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--external', type=Path, default=ROOT / 'data' / 'external')
    parser.add_argument('--out', type=Path, default=ROOT / 'data' / 'video_v1')
    parser.add_argument('--sources', nargs='+', default=['kubric', 'clevrer', 'deskspace'])
    parser.add_argument('--clevrer-videos', type=Path, default=ROOT / 'data' / 'external' / 'clevrer' / 'videos_validation_11000')
    parser.add_argument('--team-tracks', type=Path, default=ROOT / 'data' / 'external_v2' / 'track_video_test' / 'deskspace')
    parser.add_argument('--workers', type=int, default=20)
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args(argv)
    summary = {}
    if 'kubric' in args.sources: summary['kubric_movi_a'] = convert_kubric_video(args.external, args.out, args.workers, args.limit)
    if 'clevrer' in args.sources: summary['clevrer'] = convert_clevrer_video(args.external, args.clevrer_videos, args.out, args.workers, args.limit)
    if 'deskspace' in args.sources: summary['team_deskspace'] = convert_team_clip(args.team_tracks, args.out)
    for name, meta in summary.items():
        print('==', name, json.dumps(meta.get('splits'), indent=0).replace('\n', ' '))
        for which, acc in (meta.get('accuracy') or {}).items():
            print(f"  {which:11s} videos {acc['videos']:4d} | cube err median {acc['cube']['position_error'].get('median_cm', float('nan')):.1f} p90 {acc['cube']['position_error'].get('p90_cm', float('nan')):.1f} cm"
                  f" | recall all {acc['cube']['recall_all_object_frames']:.2f} visible {acc['cube']['recall_visible_object_frames']:.2f}"
                  f" | all-shape median {acc['all']['position_error'].get('median_cm', float('nan')):.1f} | id switches {acc['identity_switches']}")
    existing = {}
    path = args.out / 'meta.json'
    if path.exists(): existing = json.loads(path.read_text())
    existing.update({k: dict(splits=v.get('splits'), accuracy=v.get('accuracy')) for k, v in summary.items()})
    args.out.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(existing, indent=1))


if __name__ == '__main__':
    main()
