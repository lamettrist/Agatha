"""APPROXIMATE object tracks from a team-recorded phone clip -> Agatha episode shard + fell labels.

    CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 .venv/bin/python scripts/track_video.py CLIP.mp4 \
        [--out data/external_v2/team_clips] [--table-width 1.2] [--stride 3] [--max-seconds 8]

(use .venv, not .venv-train: the YOLO / torchvision stack only imports cleanly there)

Only for clips the team recorded itself (see DATA_LICENSES.md); never for scraped video.

Pipeline (everything is an estimate; outputs carry approximate=True):
  1. frames are read with OpenCV; every `--stride`-th frame is passed to the repo's YOLO detector
     (agatha.reconstruction.detect, weights models/vision/yolov8n.pt, CPU), classes in PRIORS only;
  2. a greedy IoU / centroid tracker links detections of the same class into 2D tracks;
  3. the repo's relative-depth + RANSAC table-plane code (agatha.reconstruction.relative_depth /
     backproject / fit_plane, imported unmodified) is run ONCE on the first sampled frame.  Each
     track's bottom-centre pixel ray is intersected with that plane (pinhole, assumed 60 deg FOV)
     and scaled with the assumed table width, exactly like agatha.reconstruction.reconstruct.
     If depth / plane fitting is unavailable, a flat image-plane mapping is used instead;
  4. tracks are resampled to 20 Hz, velocities are finite differences, orientation is identity
     (not estimated), sizes / masses are the per-class priors;
  5. `fell` = the track's bottom-centre dropped by more than --drop-frac of the image height below
     its starting level (median of the first observations) and never came back, after subtracting
     the cumulative global image shift (crude compensation of camera translation); when a plane was
     found, leaving the image region covered by plane inliers is reported as `left_table_region`.

KNOWN LIMITS (reported in the JSON as warnings where detectable)
  * STATIC CAMERA ASSUMED.  The plane is fitted on one frame; a hand-held / panning clip makes the
    3D positions and the fell labels meaningless.  A global image shift estimate (phase
    correlation) is reported and a warning is raised above 5 % of the image width.
  * No metric scale: relative depth + an assumed table width; positions are table-plane
    coordinates of the box bottom, heights are support_height + prior_height / 2.
  * After a detected fall the 3D position is unknown (the ray no longer hits the table plane):
    such frames are flagged in `observed` = False and hold the last on-table position.
  * `tipped` is not estimated (all False).  Occluded / undetected objects simply have no track.
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(Path(__file__).resolve().parent)):
    if _p not in sys.path: sys.path.insert(0, _p)
import numpy as np
from prepare_external_data import STEP, SUPPORT_HEIGHT, finite_difference, interp_linear

NOTE = 'APPROXIMATE: monocular tracks from a team-recorded clip (assumed static camera, assumed table width, class-prior sizes); not measured trajectories.'
FOV_DEG = 60.


# ----------------------------------------------------------------------------- tracker (pure numpy, unit tested)
def iou(a, b) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0., x2 - x1) * max(0., y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.


class Tracker:
    """Greedy same-class association: best IoU first, then nearest centroid within `max_shift` box diagonals."""

    def __init__(self, min_iou: float = .25, max_shift: float = .8, max_age: int = 8):
        self.min_iou, self.max_shift, self.max_age = min_iou, max_shift, max_age
        self.tracks: list[dict] = []

    def update(self, frame: int, detections: list[dict]) -> None:
        live = [t for t in self.tracks if frame - t['frames'][-1] <= self.max_age]
        pairs = []
        for ti, track in enumerate(live):
            last = track['boxes'][-1]; diagonal = math.hypot(last[2] - last[0], last[3] - last[1])
            for di, det in enumerate(detections):
                if det['label'] != track['label']: continue
                overlap = iou(last, det['bbox'])
                shift = math.hypot((last[0] + last[2] - det['bbox'][0] - det['bbox'][2]) / 2, (last[1] + last[3] - det['bbox'][1] - det['bbox'][3]) / 2)
                if overlap >= self.min_iou: pairs.append((2 + overlap, ti, di))
                elif shift <= self.max_shift * diagonal: pairs.append((1 - shift / max(self.max_shift * diagonal, 1e-9), ti, di))
        used_t, used_d = set(), set()
        for _, ti, di in sorted(pairs, reverse=True):
            if ti in used_t or di in used_d: continue
            used_t.add(ti); used_d.add(di)
            live[ti]['frames'].append(frame); live[ti]['boxes'].append(list(detections[di]['bbox'])); live[ti]['confidence'].append(detections[di]['confidence'])
        for di, det in enumerate(detections):
            if di not in used_d:
                self.tracks.append(dict(id=len(self.tracks), label=det['label'], frames=[frame], boxes=[list(det['bbox'])], confidence=[det['confidence']]))


def fell_from_track(bottom_y: np.ndarray, image_height: float, drop_frac: float = .15, settle: int = 3) -> tuple[bool, int | None]:
    """bottom_y: bottom-centre pixel row per observation (image y grows downwards).

    Fell = the last `settle` observations are all more than drop_frac * image_height below the
    starting level (median of the first `settle`).  Returns (fell, index of the first dropped observation)."""
    bottom_y = np.asarray(bottom_y, float)
    if len(bottom_y) < 2 * settle: return False, None
    start = float(np.median(bottom_y[:settle])); dropped = bottom_y - start > drop_frac * image_height
    if not dropped[-settle:].all(): return False, None
    first = len(dropped) - 1
    while first > 0 and dropped[first - 1]: first -= 1
    return True, int(first)


def ray_plane(pixels: np.ndarray, plane: dict, width: int, height: int) -> np.ndarray:
    """Pixel (u, v) rays (pinhole, FOV_DEG horizontal) intersected with the fitted plane -> plane coordinates (x, z), relative units."""
    focal = width / (2 * math.tan(math.radians(FOV_DEG) / 2)); pixels = np.asarray(pixels, float)
    direction = np.stack(((pixels[..., 0] - width / 2) / focal, (pixels[..., 1] - height / 2) / focal, np.ones(pixels.shape[:-1])), -1)
    denominator = direction @ plane['normal']
    scale = (plane['center'] @ plane['normal']) / np.where(np.abs(denominator) < 1e-9, np.nan, denominator)
    point = direction * scale[..., None] - plane['center']
    return np.stack((point @ plane['x'], point @ plane['z']), -1)


# ----------------------------------------------------------------------------- video
def read_frames(path: Path, stride: int, max_seconds: float | None):
    import cv2
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened(): raise SystemExit(f'cannot open {path}')
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.
    index, frames, times = 0, [], []
    while True:
        ok, frame = capture.read()
        if not ok or (max_seconds and index / fps > max_seconds): break
        if index % stride == 0:
            scale = 960 / max(frame.shape[:2])
            if scale < 1: frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)); times.append(index / fps)
        index += 1
    capture.release()
    return frames, np.asarray(times), fps


def camera_shift(frames) -> np.ndarray:
    """Cumulative global image translation (px) by phase correlation of consecutive sampled frames."""
    import cv2
    grey = [cv2.resize(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), (320, 180)).astype(np.float32) for f in frames]
    window = cv2.createHanningWindow((320, 180), cv2.CV_32F); total = np.zeros(2); out = [total.copy()]
    for a, b in zip(grey[:-1], grey[1:]):
        (dx, dy), _ = cv2.phaseCorrelate(a, b, window)
        total = total + np.array([dx, dy]) * frames[0].shape[1] / 320; out.append(total.copy())
    return np.asarray(out)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('video', type=Path)
    parser.add_argument('--out', type=Path, default=ROOT / 'data' / 'external_v2' / 'team_clips')
    parser.add_argument('--table-width', type=float, default=1.2, help='assumed table width (m) used for scale')
    parser.add_argument('--stride', type=int, default=3, help='run the detector on every n-th frame')
    parser.add_argument('--max-seconds', type=float, default=None)
    parser.add_argument('--min-observations', type=int, default=8)
    parser.add_argument('--drop-frac', type=float, default=.15)
    parser.add_argument('--no-depth', action='store_true', help='skip depth / plane fitting (flat image-plane mapping)')
    args = parser.parse_args(argv)
    os.environ.setdefault('YOLO_MODEL', str(ROOT / 'models' / 'vision' / 'yolov8n.pt'))
    from PIL import Image
    from agatha import reconstruction as rec                      # imported, not modified
    began = time.time(); warnings = []
    frames, times, fps = read_frames(args.video, args.stride, args.max_seconds)
    if len(frames) < 4: raise SystemExit('clip too short')
    height, width = frames[0].shape[:2]
    tracker = Tracker(); detections_total = 0; labels_seen = {}
    for index, frame in enumerate(frames):
        found = rec.detect(Image.fromarray(frame))
        for d in found: labels_seen[d['label']] = labels_seen.get(d['label'], 0) + 1
        kept = [d for d in found if d['label'] in rec.PRIORS]; detections_total += len(kept)
        tracker.update(index, kept)
    tracks = [t for t in tracker.tracks if len(t['frames']) >= args.min_observations]
    shift = camera_shift(frames); max_shift = float(np.linalg.norm(shift, axis=1).max())
    if max_shift > .05 * width:
        warnings.append(f'camera moves (cumulative global image shift up to {max_shift:.0f} px = {max_shift / width:.0%} of the width): '
                        'the static-camera assumption is violated, 3D positions and fell labels are NOT reliable')
    plane = extent = None; unit_scale = args.table_width; mapping = 'flat_image_plane'
    if not args.no_depth:
        try:
            image = Image.fromarray(frames[0]); depth = rec.relative_depth(image); cloud = rec.backproject(depth, width, height)
            mask = np.indices(depth.shape)[0] > depth.shape[0] * .3
            for t in tracks:
                if t['frames'][0] > 2: continue
                x1, y1, x2, y2 = t['boxes'][0]
                mask[int(y1 / height * mask.shape[0]):int(y2 / height * mask.shape[0]) + 1, int(x1 / width * mask.shape[1]):int(x2 / width * mask.shape[1]) + 1] = False
            plane = rec.fit_plane(cloud[mask])
            local = cloud.reshape(-1, 3) - plane['center']; px, pz = local @ plane['x'], local @ plane['z']
            unit_scale = args.table_width / max(float(np.percentile(px, 95) - np.percentile(px, 5)), .1)
            inlier = np.abs(local @ plane['normal']) < .035 * float(np.median(np.linalg.norm(cloud.reshape(-1, 3), axis=1)))
            extent = dict(x=np.percentile(px[inlier], [2, 98]) * unit_scale, z=np.percentile(pz[inlier], [2, 98]) * unit_scale)
            mapping = 'relative_depth_ransac_plane_first_frame'
        except Exception as error:                                # the repo's reconstruct() degrades the same way
            warnings.append(f'depth / plane unavailable ({type(error).__name__}: {str(error)[:120]}); flat image-plane mapping used')
            plane = None
    steps = int((times[-1] - times[0]) / STEP); grid = times[0] + np.arange(steps + 1) * STEP
    count = max(len(tracks), 1)
    states = np.zeros((steps + 1, count, 13), np.float32); states[..., 6] = 1
    props = np.zeros((count, 9), np.float32); props[:, 3] = 1
    observed = np.zeros((steps + 1, count), bool); fell = np.zeros(count, bool); left_region = np.zeros(count, bool)
    summary = []
    for i, t in enumerate(tracks):
        boxes = np.asarray(t['boxes'], float); t_obs = times[np.asarray(t['frames'])]
        bottom = np.stack(((boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]), -1)
        prior, _, _, mass = rec.PRIORS[t['label']]
        # crude compensation of camera translation: subtract the cumulative global image shift before the drop test
        raw_fall, _ = fell_from_track(bottom[:, 1], height, args.drop_frac)
        did_fall, first_drop = fell_from_track(bottom[:, 1] - shift[np.asarray(t['frames']), 1], height, args.drop_frac)
        if plane is not None: table = ray_plane(bottom, plane, width, height) * unit_scale
        else: table = np.stack(((bottom[:, 0] / width - .5) * args.table_width, (bottom[:, 1] / height - .5) * args.table_width * 2 / 3), -1)
        if not np.isfinite(table).all(): table = np.nan_to_num(table)
        on_table = np.ones(len(boxes), bool)
        if did_fall: on_table[first_drop:] = False; table[first_drop:] = table[max(first_drop - 1, 0)]
        position = np.stack((table[:, 0], np.where(on_table, SUPPORT_HEIGHT + prior[1] / 2, prior[1] / 2), table[:, 1]), -1)
        inside = (grid >= t_obs[0]) & (grid <= t_obs[-1]); sample = np.clip(grid, t_obs[0], t_obs[-1])
        states[:, i, :3] = interp_linear(t_obs, position, sample)
        observed[:, i] = inside & (interp_linear(t_obs, on_table.astype(float), sample) > .5)
        velocity, _ = finite_difference(grid, states[:, i, :3].astype(np.float64), np.tile([0., 0., 0., 1.], (steps + 1, 1)))
        states[:, i, 7:10] = velocity * inside[:, None]
        props[i] = [*prior, mass, .5, 0, 0, 0, 0]
        fell[i] = did_fall
        if extent is not None:
            end = table[first_drop - 1] if did_fall and first_drop else table[-1]
            left_region[i] = bool(end[0] < extent['x'][0] or end[0] > extent['x'][1] or end[1] < extent['z'][0] or end[1] > extent['z'][1])
        summary.append(dict(track=t['id'], label=t['label'], observations=len(boxes), t_first=float(t_obs[0]), t_last=float(t_obs[-1]),
                            mean_confidence=float(np.mean(t['confidence'])), fell=bool(did_fall), fell_without_shift_compensation=bool(raw_fall), fell_time_s=float(t_obs[first_drop]) if did_fall else None,
                            left_table_region=bool(left_region[i]) if extent is not None else None,
                            bottom_drop_frac=float((bottom[-3:, 1].mean() - np.median(bottom[:3, 1])) / height),
                            image_path_px=float(np.linalg.norm(np.diff(bottom, axis=0), axis=1).sum()),
                            table_displacement_m=float(np.linalg.norm(table[-1] - table[0]))))
    mask = np.zeros(count, bool); mask[:len(tracks)] = True
    context = np.array([args.table_width, args.table_width * 2 / 3, SUPPORT_HEIGHT, 9.81, 0, 0, 0, 1, .5, 0], np.float32)
    target = args.out / args.video.stem / 'test'; target.mkdir(parents=True, exist_ok=True)
    zeros = np.zeros((1, steps + 1, count), bool)
    labels = np.array([t['label'] for t in tracks] + [''] * (count - len(tracks)), '<U16')
    np.savez_compressed(target / 'shard_00000.npz', states=states[None], props=props[None], context=context[None], mask=mask[None],
                        contact_support=zeros, contact_floor=zeros, contact_object=zeros, rest=zeros, fell=fell[None] & mask[None],
                        tipped=np.zeros((1, count), bool), kind=np.array(['team_clip'], '<U24'), chaotic=np.array([bool(fell.any())]),
                        live_physics=np.array([False]), episode_index=np.array([0]), init_penetration=np.zeros(1, np.float32),
                        n_frames=np.array([steps + 1]), shape=labels[None], box_like=np.zeros((1, count), bool), observed=observed[None],
                        left_table_region=left_region[None], source_id=np.array([args.video.name], '<U80'))
    report = dict(approximate=True, note=NOTE, video=str(args.video), fps=float(fps), frames_used=len(frames), stride=args.stride,
                  image_size=[width, height], duration_s=float(times[-1] - times[0]), detector='ultralytics YOLO via agatha.reconstruction.detect',
                  detections_all_classes=labels_seen, detections_kept=detections_total, raw_tracks=len(tracker.tracks), tracks=len(tracks),
                  mapping=mapping, assumed=dict(table_width_m=args.table_width, fov_deg=FOV_DEG, support_height_m=SUPPORT_HEIGHT, static_camera=True),
                  camera_shift_px=dict(max=max_shift, final=float(np.linalg.norm(shift[-1])), fraction_of_width=max_shift / width),
                  plane=None if plane is None else dict(inlier_ratio=plane['inlier_ratio'], normal_camera=np.asarray(plane['normal']).tolist(),
                                                        table_extent_m={k: np.asarray(v).tolist() for k, v in extent.items()}),
                  labels_reliable=not warnings, objects=summary, fell=[s['track'] for s in summary if s['fell']], warnings=warnings, shard=str(target / 'shard_00000.npz'),
                  seconds=time.time() - began)
    (args.out / args.video.stem / 'tracks.json').write_text(json.dumps(report, indent=1))
    print(json.dumps({k: report[k] for k in ('video', 'frames_used', 'duration_s', 'detections_all_classes', 'raw_tracks', 'tracks', 'mapping',
                                               'camera_shift_px', 'fell', 'labels_reliable', 'warnings', 'seconds')}, indent=1))
    for s in summary: print(s)


if __name__ == '__main__':
    main()
