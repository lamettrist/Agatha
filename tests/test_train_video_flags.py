"""Opt-in video-derived data in scripts/train_contact_ddp.py (--video-data / --video-frac / --video-weight / --video-horizon).

The model is state-based, so "video" here always means approximate object trajectories derived from video
(tracking + geometry), stored in the data/v2 shard format plus obs_mask / pos_sigma / has_orientation.
"""
import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts' / 'train_contact_ddp.py'


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def trainer():
    return load_module(SCRIPT, 'train_contact_ddp_video_test')


def default_args(trainer, *extra):
    old = sys.argv
    sys.argv = ['x', '--data', 'unused', '--out', 'unused', *extra]
    try:
        return trainer.parse_args()
    finally:
        sys.argv = old


def synthetic_episodes(episodes=12, steps=24, slots=4, seed=0):
    """Small boxes sliding/falling near a 1.2 x 0.8 m table at 0.75 m (data/v2 layout; physics is only roughly plausible)."""
    rng = np.random.default_rng(seed)
    states = np.zeros((episodes, steps + 1, slots, 13), np.float32)
    props = np.zeros((episodes, slots, 9), np.float32)
    mask = np.zeros((episodes, slots), bool)
    context = np.tile(np.array([1.2, .8, .75, 9.81, 0, 0, 0, 1, .5, .1], np.float32), (episodes, 1))
    states[..., 6] = 1
    for e in range(episodes):
        for n in range(int(rng.integers(2, slots + 1))):
            mask[e, n] = True
            size = rng.uniform(.05, .12, 3)
            props[e, n] = [*size, .3, .5, 0, .1, 0, 0]
            pos = np.array([-.45 + .3 * n, .75 + size[1] / 2 + rng.uniform(0, .15), rng.uniform(-.2, .2)])
            vel = np.array([rng.uniform(-.3, .3), 0., rng.uniform(-.3, .3)])
            for t in range(steps + 1):
                states[e, t, n, :3] = pos
                states[e, t, n, 7:10] = vel
                vel[1] -= 9.81 * .05
                pos = pos + vel * .05
                floor = .75 + size[1] / 2
                if pos[1] < floor:
                    pos[1], vel[1] = floor, 0.
                    vel *= .9
    speed = np.linalg.norm(states[..., 7:10], axis=-1)
    rest = (speed < .01) & mask[:, None]
    zeros = np.zeros((episodes, steps + 1, slots), bool)
    return dict(states=states, props=props, context=context, mask=mask, rest=rest, contact_support=rest.copy(),
                contact_floor=zeros, contact_object=zeros.copy(), chaotic=np.arange(episodes) % 2 == 0)


def to_video(sim, seed=0, sigma_m=.01, drop=.15, keep_frames=None):
    """Fake 'video-derived' shard from clean episodes: position noise, dropped observations, finite-difference
    velocities, no orientation / angular velocity for half of the objects, no contact or rest flags."""
    rng = np.random.default_rng(seed)
    states = sim['states'].copy()
    E, frames, N = states.shape[:3]
    if keep_frames:
        states, frames = states[:, :keep_frames], keep_frames
    mask = sim['mask']
    states[..., :3] += rng.normal(0, sigma_m, states[..., :3].shape).astype(np.float32)
    states[:, 1:, :, 7:10] = (states[:, 1:, :, :3] - states[:, :-1, :, :3]) / .05
    states[:, 0, :, 7:10] = states[:, 1, :, 7:10]
    has_orientation = (rng.random((E, N)) < .5) & mask
    states[..., 3:7] = np.where(has_orientation[:, None, :, None], states[..., 3:7], np.array([0, 0, 0, 1], np.float32))
    states[..., 10:13] *= has_orientation[:, None, :, None]
    obs = (rng.random((E, frames, N)) > drop) & mask[:, None]
    states[~obs & mask[:, None]] = np.nan                  # unobserved frames carry no usable state
    return dict(states=states, props=sim['props'], context=sim['context'], mask=mask, obs_mask=obs,
                pos_sigma=np.full((E, N), sigma_m, np.float32), has_orientation=has_orientation, video_derived=np.ones(E, bool))


def write_split(root, split, arrays):
    (root / split).mkdir(parents=True, exist_ok=True)
    np.savez(root / split / 'shard_00000.npz', **arrays)


# ------------------------------------------------------------------------------------------- defaults unchanged
def test_flags_default_to_off(trainer):
    args = default_args(trainer)
    assert args.video_data is None and args.video_frac == 0 and args.video_weight == .2 and args.video_horizon == 0
    args = default_args(trainer, '--video-data', 'a', 'b', '--video-frac', '.1', '--video-horizon', '6')
    assert args.video_data == ['a', 'b'] and args.video_frac == .1 and args.video_horizon == 6


def _fixed_run(module, sim, seed=3, k=4):
    """Sampler indices and the training loss of one fixed batch, for comparing two versions of the trainer."""
    args = default_args(module, '--w-move', '3', '--w-impulse', '3', '--w-speed', '1', '--event-frac', '.3')
    data = module.to_device({k_: v for k_, v in sim.items() if k_ in ('states', 'props', 'context', 'mask', 'rest')}, 'cpu')
    speed = data['states'][..., 7:10].norm(dim=-1)
    active = (speed > .05).any(-1)
    sampler = module.WindowSampler(active, seed, 0, 2, module.event_frames(data), args.event_frac)
    sampler.set(1, k)
    picks = [torch.stack(sampler.next(16)) for _ in range(3)]
    torch.manual_seed(seed)
    model = module.cg.ContactGNN(width=32, rounds=2)
    gen = torch.Generator().manual_seed(seed)
    batch = module.gather(data, *picks[0], k, torch.device('cpu'))
    loss, stats, bad = module.Trainee(model, args, None)(*batch, k, gen)
    loss.backward()
    grad = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
    return picks, loss.detach(), stats, grad


def test_default_behaviour_matches_the_committed_trainer(trainer, tmp_path):
    """With the video flags unset, sampler indices, loss, statistics and gradients equal those of the committed file.
    (Vacuous once this change itself is committed; the comparison against the pre-change file was also run by hand.)"""
    try:
        old = subprocess.run(['git', 'show', 'HEAD:scripts/train_contact_ddp.py'], cwd=ROOT, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        pytest.skip('git not available')
    if old.returncode != 0:
        pytest.skip('no committed trainer to compare against')
    # the committed file resolves the repository root from its own location
    (tmp_path / 'scripts').mkdir()
    path = tmp_path / 'scripts' / 'train_contact_ddp.py'
    path.write_text(old.stdout.replace('sys.path.insert(0, str(Path(__file__).resolve().parents[1]))', f'sys.path.insert(0, {str(ROOT)!r})'))
    reference = load_module(path, 'train_contact_ddp_committed')
    sim = synthetic_episodes()
    new, ref = _fixed_run(trainer, sim), _fixed_run(reference, sim)
    for a, b in zip(new[0], ref[0]):
        assert torch.equal(a, b)
    assert torch.equal(new[1], ref[1]) and torch.equal(new[2], ref[2]) and torch.equal(new[3], ref[3])


# ------------------------------------------------------------------------------------------- loader / sampler
def test_video_loader_pads_and_marks_valid_starts(trainer, tmp_path):
    sim = synthetic_episodes()
    long = to_video(sim, seed=1)
    three = {k: v[:5, :3] if k in ('props', 'mask') else v[:5, :, :3] if v.ndim >= 3 else v[:5] for k, v in sim.items()}
    short = to_video(three, seed=2, keep_frames=13)
    short['mask'][:, 2] = False
    short['obs_mask'] &= short['mask'][:, None]
    write_split(tmp_path / 'long', 'train', long)
    write_split(tmp_path / 'short', 'train', short)
    data, names = trainer.load_video_split([tmp_path / 'long', tmp_path / 'short', tmp_path / 'missing'], 'train')
    assert names == ['long', 'short']
    assert data['states'].shape == (17, 25, 4, 13) and np.isfinite(data['states']).all()
    assert np.allclose(np.linalg.norm(data['states'][..., 3:7], axis=-1), 1, atol=1e-5)
    assert not data['obs'][12:, 13:].any() and not data['mask'][12:, 3].any()      # padded frames / slots
    assert list(data['source']) == [0] * 12 + [1] * 5
    dyn = data['mask'] & (data['props'][..., 5] < .5)
    ok = data['start_ok']
    assert ok.any() and not ok[12:, 13:].any()
    e, t = np.argwhere(ok)[0]
    assert data['obs'][e, t][dyn[e]].all()
    e, t = np.argwhere(~ok[:12, 1:-1])[0]
    t += 1
    seen = data['obs'][e]
    assert not all(seen[t, n] and (seen[t - 1, n] or seen[t + 1, n]) for n in np.flatnonzero(dyn[e]))
    assert trainer.load_video_split([tmp_path / 'long'], 'val')[0] is None

    tensors = trainer.to_device(data, 'cpu')
    picks = []
    for rank in range(2):
        sampler = trainer.VideoSampler(tensors, 0, rank, 2)
        sampler.set(0, 4)
        e, t0 = sampler.next(sampler.pool.numel())
        assert bool(tensors['start_ok'][e, t0].all()) and int(t0.max()) <= 24 - 4
        assert bool((tensors['obs_count'][e[:, None], t0[:, None] + torch.arange(1, 5)].sum(1) > 0).all())
        picks.append(set((e * 100 + t0).tolist()))
    assert not picks[0] & picks[1] and len(picks[0] | picks[1]) == sampler.windows   # disjoint rank slices cover the pool


def test_missing_video_keys_fall_back_conservatively(trainer, tmp_path):
    sim = synthetic_episodes(episodes=3)
    write_split(tmp_path / 'plain', 'train', {k: sim[k] for k in ('states', 'props', 'context', 'mask')})
    data, _ = trainer.load_video_split([tmp_path / 'plain'], 'train')
    assert data['obs'][data['mask'][:, None].repeat(25, 1)].all() and not data['ori'].any() and (data['sigma'] == 0).all()


# ------------------------------------------------------------------------------------------- loss
def _video_batch(trainer, tmp_path, k=4):
    write_split(tmp_path / 'v', 'train', to_video(synthetic_episodes(), seed=5))
    data = trainer.to_device(trainer.load_video_split([tmp_path / 'v'], 'train')[0], 'cpu')
    sampler = trainer.VideoSampler(data, 0, 0, 1)
    sampler.set(0, k)
    return list(trainer.gather_video(data, *sampler.next(24), k, torch.device('cpu')))


def test_video_loss_uses_only_observed_positions(trainer, tmp_path):
    k = 4
    args = default_args(trainer)
    torch.manual_seed(0)
    model = trainer.cg.ContactGNN(width=32, rounds=2)
    batch = _video_batch(trainer, tmp_path, k)
    states, obs, props, context, mask, sigma, ori = batch
    loss, stats, bad = trainer.unroll_video(model, *batch, k, args, None)
    assert not bool(bad) and torch.isfinite(loss) and float(loss) > 0 and stats.shape == (len(trainer.VIDEO_TERMS), )
    assert float(stats[0]) == pytest.approx(float(loss))
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)

    changed = states.clone()
    changed[:, 1:, :, 7:13] += 5.                          # velocity targets are never read
    hidden = ~obs[:, 1:]
    changed[:, 1:][hidden] += 3.                           # nor anything at unobserved frames
    assert hidden.any()
    again = trainer.unroll_video(model, changed, obs, props, context, mask, sigma, ori, k, args, None)[0]
    assert torch.equal(again, loss)

    no_ori = trainer.unroll_video(model, changed, obs, props, context, mask, sigma, torch.zeros_like(ori), k, args, None)[1]
    assert float(no_ori[3]) == 0 and float(stats[3]) > 0   # orientation term only where has_orientation is set
    turned = states.clone()
    turned[:, 1:, :, 3:7] = torch.tensor([0., .6, 0., .8])
    assert torch.equal(trainer.unroll_video(model, turned, obs, props, context, mask, sigma, torch.zeros_like(ori), k, args, None)[0],
                       trainer.unroll_video(model, states, obs, props, context, mask, sigma, torch.zeros_like(ori), k, args, None)[0])

    wide = trainer.unroll_video(model, states, obs, props, context, mask, sigma * 10, ori, k, args, None)[1]
    assert float(wide[1]) < float(stats[1])                # larger pos_sigma -> weaker position term
    assert float(wide[4]) == pytest.approx(float(stats[4]))  # the reported error in metres does not depend on it
    floor = trainer.unroll_video(model, states, obs, props, context, mask, sigma * 0, ori, k, args, None)[1]
    assert float(floor[1]) == pytest.approx(float(stats[1]))   # sigma 0.01 m is already at the 0.02 m scale floor


def test_video_penetration_penalty_needs_no_reference(trainer, tmp_path):
    """An object pushed into the table is penalised from the predicted state alone."""
    k = 1
    args = default_args(trainer)

    class Sink(torch.nn.Module):
        out_mean, out_std = torch.zeros(1), torch.ones(1)

        def __init__(self, depth):
            super().__init__()
            self.depth = depth

        def predict_raw(self, state, props, context, mask):
            return state

        def integrate(self, state, raw, props, context, mask):
            out = state.clone()
            out[..., 1] = context[:, 2, None] + props[..., 1] / 2 - self.depth
            return out
    batch = _video_batch(trainer, tmp_path, k)
    shallow = trainer.unroll_video(Sink(.001), *batch, k, args, None)[1]
    deep = trainer.unroll_video(Sink(.02), *batch, k, args, None)[1]
    assert float(shallow[2]) == 0 and float(deep[2]) > 1


def test_trainee_adds_weighted_video_loss_and_keeps_sim_stats(trainer, tmp_path):
    k = 4
    sim = synthetic_episodes()
    args = default_args(trainer, '--video-weight', '0.25')
    torch.manual_seed(0)
    model = trainer.cg.ContactGNN(width=32, rounds=2)
    data = trainer.to_device({k_: sim[k_] for k_ in ('states', 'props', 'context', 'mask', 'rest')}, 'cpu')
    e, t0 = torch.arange(8), torch.zeros(8, dtype=torch.long)
    batch = trainer.gather(data, e, t0, k, torch.device('cpu'))
    video = _video_batch(trainer, tmp_path, k)
    trainee = trainer.Trainee(model, args, None)
    sim_loss, sim_stats, _ = trainee(*batch, k, torch.Generator().manual_seed(1))
    loss, stats, bad, vstats = trainee(*batch, k, torch.Generator().manual_seed(1), tuple(video) + (k, ))
    video_loss = trainer.unroll_video(model, *video, k, args, None)[0]
    # same k and slot count: the video windows ride in the sim model calls, but are scored separately
    assert torch.allclose(stats, sim_stats, rtol=1e-4, atol=1e-6) and not bool(bad)
    assert float(loss) == pytest.approx(float(sim_loss + .25 * video_loss), rel=1e-4)
    assert float(vstats[0]) == pytest.approx(float(video_loss), rel=1e-4)
    # different horizon: separate unroll, sim statistics exactly those of a sim-only forward
    short = [video[0][:, :3], video[1][:, :3], *video[2:]]
    loss, stats, bad, vstats = trainee(*batch, k, torch.Generator().manual_seed(1), tuple(short) + (2, ))
    video_loss = trainer.unroll_video(model, *short, 2, args, None)[0]
    assert torch.equal(stats, sim_stats) and torch.equal(loss, sim_loss + .25 * video_loss) and float(vstats[0]) == float(video_loss)


# ------------------------------------------------------------------------------------------- end to end (CPU)
def test_cpu_run_logs_video_separately_and_scores_on_sim_only(tmp_path):
    sim = synthetic_episodes(episodes=16, steps=40)
    keys = ('states', 'props', 'context', 'mask', 'rest', 'contact_support', 'contact_floor', 'contact_object', 'chaotic')
    for split in ('train', 'val'):
        write_split(tmp_path / 'sim', split, {k: sim[k] for k in keys})
        write_split(tmp_path / 'vid', split, to_video(sim, seed=7))
    common = [sys.executable, str(SCRIPT), '--data', str(tmp_path / 'sim'), '--width', '32', '--rounds', '2', '--epochs', '1',
              '--windows-per-epoch', '96', '--batch', '16', '--horizon', '4', '--start-horizon', '4', '--log-every', '3',
              '--val-windows', '64', '--val-rollout-episodes', '16', '--data-device', 'cpu', '--amp', 'off']
    env = {'CUDA_VISIBLE_DEVICES': '', 'PATH': '/usr/bin:/bin', 'OMP_NUM_THREADS': '2'}
    runs = {}
    for name, extra in (('video', ['--video-data', str(tmp_path / 'vid'), '--video-frac', '.25', '--video-horizon', '3']),
                        ('eval_only', ['--video-data', str(tmp_path / 'vid')]), ('plain', [])):
        out = tmp_path / name
        done = subprocess.run(common + ['--out', str(out), *extra], capture_output=True, text=True, timeout=600, env=env, cwd=ROOT)
        assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]
        rows = [json.loads(line) for line in (out / 'train_log.jsonl').read_text().splitlines()]
        runs[name] = ([r for r in rows if 'event' not in r], [r for r in rows if r.get('event') == 'epoch'][-1],
                      torch.load(out / 'best.pt', map_location='cpu', weights_only=True), done.stdout)
    steps, epoch, ckpt, stdout = runs['video']
    assert all(set(('video_loss', 'video_pos', 'video_pen', 'video_rot', 'video_err_m')) <= set(r) and r['video_k'] == 3 for r in steps)
    assert epoch['video_val_windows'] > 0 and epoch['video_val_pos_err_k'] > 0 and 'video_val_pos_err_k/vid' in epoch
    assert 'train_video_loss' in epoch and 'video (not scored)' in stdout
    # the selection score is exactly the simulator score
    sim_score = epoch['pos_err_2s_chaotic'] + epoch['pos_err_2s_gentle'] + 10 * epoch['mean_penetration'] + .5 * epoch['rest_drift']
    assert epoch['score'] == pytest.approx(sim_score, rel=1e-9)
    assert ckpt['kind'] == 'contact_gnn' and ckpt['train_args']['video_frac'] == .25
    # --video-data without --video-frac only adds the video validation: training is identical to a run without the flag
    plain_steps, plain_epoch, plain_ckpt, _ = runs['plain']
    eval_steps, eval_epoch, eval_ckpt, _ = runs['eval_only']
    assert [r['loss'] for r in eval_steps] == [r['loss'] for r in plain_steps] and 'video_loss' not in eval_steps[0]
    assert eval_epoch['score'] == plain_epoch['score'] and 'video_val_pos_err_k' in eval_epoch and 'video_val_windows' not in plain_epoch
    assert all(torch.equal(v, eval_ckpt['state_dict'][k]) for k, v in plain_ckpt['state_dict'].items())
