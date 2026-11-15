"""APPROXIMATE side-by-side evaluation of several ContactGNN checkpoints for the video fine-tune gate.

All checkpoints are rolled out on exactly the same episodes, so differences are paired:

  external ground truth (data/external_v2, scripts/prepare_external_data.py):
    clevrer                 CLEVRER validation 10000-10999 (disjoint from the videos used for training)
    kubric_movi_a_heldout   only the episodes whose source_id is in data/video_v1/kubric_movi_a/meta.json
                            `test_source_ids` (the rest of the Kubric set overlaps the video-derived training set)
    physion, real_iam_box_toss, real_contactnets_cube   (never trained on)
  video-derived held-out split (data/video_v1/<source>/test) scored against the dataset ground truth
  (data/video_v1/<source>/test_gt, slot-aligned): the rollout starts from the VIDEO-DERIVED state at the first
  frame where every dynamic object is observed at t and t+1, and the predicted positions are compared with the
  ground-truth positions at the last valid frame (and averaged over the rollout).

Metrics reuse scripts/eval_external.py (object_stats / summarize / paired_bootstrap); nothing there is modified.
Engines per checkpoint: `<arm>_projected` (model + contact projection, the shipped configuration) and `<arm>_raw`.

Usage:
  CUDA_VISIBLE_DEVICES=7 .venv-train/bin/python scripts/eval_video_ft.py \
      --arm shipped=models/contact_gnn.pt --arm control=models/runs/video_ft_control/best.pt \
      --arm video=models/runs/video_ft_video/best.pt --out models/runs/video_ft_compare.json --device cuda:0
"""
from __future__ import annotations

import argparse, glob, hashlib, json, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_external as ex      # noqa: E402
ev = ex.ev
ROOT = Path(__file__).resolve().parents[1]
REAL = ('real_iam_box_toss', 'real_contactnets_cube')


def evaluate(engines: dict, data: dict, pen_fn, fell_meaningful: bool, device: str, batch_size: int = 256) -> dict:
    """eval_external.evaluate on an explicit device."""
    count, steps = len(data['states']), data['states'].shape[1] - 1
    collected = {name: [] for name in engines}
    with torch.inference_mode():
        for start in range(0, count, batch_size):
            sl = slice(start, min(start + batch_size, count))
            batch = dict(states=torch.from_numpy(data['states'][sl]).float().to(device), props=torch.from_numpy(data['props'][sl]).float().to(device),
                         context=torch.from_numpy(data['context'][sl]).float().to(device), mask=torch.from_numpy(data['mask'][sl]).bool().to(device),
                         n_frames=torch.from_numpy(data['n_frames'][sl]).long().to(device))
            for name, step_fn in engines.items():
                if step_fn is None: traj, diverged = batch['states'], None
                else: traj, diverged = ev.rollout(step_fn, batch['states'][:, 0], batch['props'], batch['context'], batch['mask'], steps)
                traj = ex.freeze_after(traj, batch['n_frames'])
                collected[name].append({k: v.cpu().numpy() for k, v in ex.object_stats(traj, batch, diverged, pen_fn, fell_meaningful).items()})
    return {name: {k: np.concatenate([p[k] for p in parts]) for k in parts[0]} for name, parts in collected.items()}


def mean_ci(stats: dict, select: np.ndarray, key: str = 'pos_final', resamples: int = 2000, seed: int = 0) -> dict | None:
    """Episode-level bootstrap CI of the pooled mean of one engine."""
    episodes = np.flatnonzero(select.any(1))
    if len(episodes) < 2: return None
    sums = np.where(select, stats[key], 0.)[episodes].sum(1); counts = select[episodes].sum(1)
    pick = np.random.default_rng(seed).integers(0, len(episodes), (resamples, len(episodes)))
    samples = sums[pick].sum(1) / counts[pick].sum(1)
    return dict(mean=float(sums.sum() / counts.sum()), ci95=[float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))],
                n_objects=int(counts.sum()), n_episodes=int(len(episodes)))


def load_split(path: Path, keys) -> dict:
    files = sorted(glob.glob(str(path / 'shard_*.npz')))
    if not files: raise FileNotFoundError(path)
    parts = {k: [] for k in keys}
    for name in files:
        with np.load(name, allow_pickle=False) as shard:
            for k in keys: parts[k].append(shard[k])
    return {k: np.concatenate(v) for k, v in parts.items()}


def video_pairs(root: Path) -> dict | None:
    """Held-out video-derived episodes re-based at the first fully observed frame, with ground-truth targets."""
    keys = ex.KEYS + ('obs_mask',)
    try: vid, gt = load_split(root / 'test', keys), load_split(root / 'test_gt', keys)
    except FileNotFoundError: return None
    assert (vid['source_id'] == gt['source_id']).all(), 'test and test_gt are not aligned'
    frames = vid['states'].shape[1]; rows = []
    for e in range(len(vid['states'])):
        dyn = vid['mask'][e] & (vid['props'][e, :, 5] < .5); n = int(vid['n_frames'][e])
        if not dyn.any() or n < 3: continue
        obs = vid['obs_mask'][e][:n][:, dyn]                                   # [n, objects]
        ok = obs[:-1].all(1) & obs[1:].all(1) & np.isfinite(vid['states'][e, :n - 1][:, dyn]).all((1, 2))
        start = np.flatnonzero(ok)
        if not len(start) or n - 1 - start[0] < 4: continue                     # at least 0.2 s to predict
        rows.append((e, int(start[0])))
    if not rows: return None
    index = np.array([r[0] for r in rows]); t0 = np.array([r[1] for r in rows])
    take = np.minimum(t0[:, None] + np.arange(frames)[None], frames - 1)       # [E, F] shifted frame index
    out = {k: v[index] for k, v in vid.items()}
    truth = gt['states'][index][np.arange(len(index))[:, None], take]
    video_states = vid['states'][index][np.arange(len(index))[:, None], take]
    out['states'] = truth.copy(); out['states'][:, 0] = video_states[:, 0]     # start = video-derived state, targets = ground truth
    out['n_frames'] = vid['n_frames'][index] - t0
    out['mask'] = vid['mask'][index] & gt['mask'][index]
    out['_gt_start'] = truth[:, 0]; out['_n_total'] = len(vid['states']); out['_t0'] = t0
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--arm', action='append', required=True, help='name=checkpoint (first arm = reference "shipped")')
    parser.add_argument('--data', type=Path, default=ROOT / 'data' / 'external_v2')
    parser.add_argument('--video-data', type=Path, default=ROOT / 'data' / 'video_v1')
    parser.add_argument('--datasets', nargs='+', default=['clevrer', 'kubric_movi_a', 'physion', *REAL])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--resamples', type=int, default=2000)
    args = parser.parse_args(argv)
    from agatha import contact_gnn
    arms, engines, info = [], dict(reference=None, hold_initial=ex.hold_step), {}
    for item in args.arm:
        name, path = item.split('=', 1); checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        model = contact_gnn.build_from_checkpoint(checkpoint).eval().to(args.device)
        raw = (lambda m: lambda state, props, context, mask: m(state, props, context, mask))(model)
        engines[f'{name}_raw'] = raw; engines[f'{name}_projected'] = ev.projected_step(raw, contact_gnn.project)
        arms.append(name); info[name] = dict(path=path, sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(), epoch=checkpoint.get('epoch'))
    pairs = [(a, b) for i, a in enumerate(arms) for b in arms[i + 1:]]
    result = dict(approximate=True, note=ex.BANNER, generated=time.strftime('%Y-%m-%d %H:%M:%S'), arms=info, device=args.device,
                  paired_bootstrap_note='episode-level paired bootstrap of the pooled mean difference B - A (negative = B better), 95% CI',
                  datasets={}, video_derived={})

    def block(data: dict, fell_meaningful: bool, subsets: dict, extra_keys=()) -> dict:
        stats = evaluate(engines, data, contact_gnn.penetration_depths, fell_meaningful, args.device, args.batch_size)
        out = dict(engines={}, paired=[])
        for engine, s in stats.items():
            out['engines'][engine] = {}
            for subset, select in subsets.items():
                row = ex.summarize(s, data, select, fell_meaningful)
                row['rest_pos_err_ci'] = mean_ci(s, select, 'pos_final', args.resamples)
                out['engines'][engine][subset] = row
        for subset, select in subsets.items():
            for a, b in pairs:
                for kind in ('projected', 'raw'):
                    for key, metric in (('pos_final', 'rest_pos_err_m'), ('pos_rollout_sum', 'pos_err_rollout_sum_m'), *extra_keys):
                        boot = ex.paired_bootstrap(stats[f'{a}_{kind}'], stats[f'{b}_{kind}'], select, key, args.resamples)
                        if boot: out['paired'].append(dict(subset=subset, a=f'{a}_{kind}', b=f'{b}_{kind}', metric=metric, result=boot))
        return out

    heldout = set(json.loads((args.video_data / 'kubric_movi_a' / 'meta.json').read_text())['test_source_ids'])
    for name in args.datasets:
        began = time.time(); base = ex.load_dataset(args.data / name); label = name
        if name == 'kubric_movi_a':
            keep = np.array([s in heldout for s in base['source_id'].tolist()]); base = {k: v[keep] for k, v in base.items()}; label = 'kubric_movi_a_heldout'
        real = ex.is_real(base); subsets = ex.subset_masks(base)
        if real: subsets = dict(all_objects=subsets['all_objects'])
        entry = dict(n_episodes=int(len(base['states'])), real_world=real, floor_modes={})
        for mode in ('slab', 'floor'):
            data = base if mode == 'slab' else ex.to_floor_mode(base, keep_pair_friction=real)
            entry['floor_modes'][mode] = block(data, mode == 'slab', subsets,
                                               (('ori_final', 'ori_err_final_deg'), ('face_wrong', 'rest_face_error_rate')) if real else ())
        result['datasets'][label] = entry
        print(f'{label}: {len(base["states"])} episodes [{time.time() - began:.1f}s]', flush=True)

    for source in ('kubric_movi_a', 'clevrer'):
        began = time.time(); data = video_pairs(args.video_data / source)
        if data is None: continue
        start_error = np.linalg.norm(data['states'][:, 0, :, :3] - data['_gt_start'][..., :3], axis=-1)
        subsets = ex.subset_masks(data); entry = dict(n_episodes_used=int(len(data['states'])), n_episodes_in_split=int(data['_n_total']),
                                                      start_frame_median=float(np.median(data['_t0'])),
                                                      horizon_s_median=float(np.median(data['n_frames'] - 1) * ex.STEP),
                                                      start_pos_err_video_vs_gt_m={k: (float(start_error[m].mean()) if m.any() else None) for k, m in subsets.items()},
                                                      note='rollout from the video-derived state at the first fully observed frame; errors vs dataset ground truth; '
                                                           '`reference` is the ground truth with the video-derived first frame, `hold_initial` keeps the video-derived start position',
                                                      floor_modes={})
        clean = {k: v for k, v in data.items() if not k.startswith('_')}
        for mode in ('slab', 'floor'):
            entry['floor_modes'][mode] = block(clean if mode == 'slab' else ex.to_floor_mode(clean), mode == 'slab', subsets)
        result['video_derived'][source] = entry
        print(f'video-derived {source}: {entry["n_episodes_used"]} of {entry["n_episodes_in_split"]} episodes [{time.time() - began:.1f}s]', flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1))
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
