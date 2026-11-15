import json, math, sys
from pathlib import Path
import numpy as np
import pytest
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import eval_contact as ev
from agatha.physics import simulate
from agatha.schema import Scene, SceneObject, Support, Push, Tilt, SupportCollisions
from agatha.world_model import prepare, state_array

N = 8


def physics_episode(scene, actions, steps=40):
    """Ground-truth 20 Hz episode in the contact-v2 tensor layout + the physics events."""
    result = simulate(scene, actions, duration=steps * .05, fps=20)
    prepared, _, gravity, tilt = prepare(scene, actions)
    ids = [o.id for o in prepared.objects]; count = len(ids)
    states = np.zeros((steps + 1, N, 13), np.float32); states[:, :, 6] = 1
    for t, frame in enumerate(result['frames']): states[t, :count] = state_array(frame, ids)
    props = np.zeros((N, 9), np.float32); props[:, 3] = 1
    for i, o in enumerate(prepared.objects): props[i, :6] = [*o.size, o.mass_kg, o.friction, float(o.is_static)]
    context = np.array([*scene.support.size, scene.support.height, gravity, *tilt, float(prepared.support.collisions_enabled), .5, 0], np.float32)
    mask = np.zeros(N, bool); mask[:count] = True
    events = result['events']
    fell = np.array([i < count and ids[i] in events['fell'] for i in range(N)]); tipped = np.array([i < count and ids[i] in events['tipped'] for i in range(N)])
    return states, props, context, mask, fell, tipped


def scenes():
    box = lambda id, x, size=(.1, .1, .1), static=False: SceneObject(id=id, label='box', size=size, position=(x, .75 + size[1] / 2, 0), is_static=static)
    tall = (.05, .3, .05)
    return [
        (Scene(objects=[box('a', .5), box('b', 0.), box('s', -.3, static=True)]), [Push(type='push', id='a', direction=(1, 0), impulse_Ns=1.)]),
        (Scene(objects=[box('a', .2), box('t', -.2, tall)]), [Tilt(type='tilt_support', angle_deg=35, axis='z')]),
        (Scene(objects=[box('a', .2), box('b', -.2)]), [Tilt(type='tilt_support', angle_deg=-20, axis='x')]),
        (Scene(objects=[box('a', 0.), box('t', .3, tall)]), [SupportCollisions(type='set_support_collisions', enabled=False)]),
        (Scene(objects=[box('a', 0.), box('t', .3, tall)], support=Support(size=(.9, .7), height=.75)),
         [Push(type='push', id='t', direction=(1, 0, 0), impulse_Ns=.6)]),
        (Scene(objects=[box('a', 0.), box('b', .3)]), []),
    ]


@pytest.fixture(scope='module')
def episodes():
    parts = [physics_episode(scene, actions) for scene, actions in scenes()]
    return [np.stack(column) for column in zip(*parts)]


def test_fell_and_tipped_rule_match_physics_summarize(episodes):
    states, props, context, mask, fell, tipped = (torch.from_numpy(a) for a in episodes)
    got_fell, got_tipped = ev.fell_tipped(states, props, context, mask)
    assert torch.equal(got_fell, fell) and torch.equal(got_tipped, tipped)
    assert fell.any() and (~fell[mask & (props[..., 5] < .5)]).any() and tipped.any()   # the scenes exercise both outcomes
    assert not got_fell[0, 2] and not got_fell[:, 3:].any()                               # static / padded never fall


def test_orientation_error_is_geodesic_and_sign_invariant():
    identity = torch.tensor([0., 0., 0., 1.])
    quarter = torch.tensor([0., math.sin(math.pi / 4), 0., math.cos(math.pi / 4)])
    small = torch.tensor([math.sin(math.radians(.5)), 0., 0., math.cos(math.radians(.5))])
    got = ev.orientation_error_deg(torch.stack((quarter, -identity, small, 3 * quarter)), torch.stack((identity, identity, identity, quarter)))
    assert torch.allclose(got, torch.tensor([90., 0., 1., 0.]), atol=1e-3)


def toy_batch(episodes=4, steps=20):
    states = torch.zeros(episodes, steps + 1, N, 13); states[..., 6] = 1; states[..., 1] = .8
    props = torch.zeros(episodes, N, 9); props[..., :3] = .1; props[..., 3] = 1; props[:, 2, 5] = 1   # slot 2 static
    context = torch.tensor([[1.2, .8, .75, 9.81, 0, 0, 0, 1, .5, 0]]).repeat(episodes, 1)
    mask = torch.zeros(episodes, N, dtype=torch.bool); mask[:, :3] = True
    rest = torch.zeros(episodes, steps + 1, N, dtype=torch.bool)
    none = torch.zeros(episodes, N, dtype=torch.bool)
    return dict(states=states, props=props, context=context, mask=mask, rest=rest, fell=none.clone(), tipped=none.clone())


def test_rest_drift_uses_predicted_motion_on_true_rest_frames():
    batch = toy_batch(); batch['rest'][:, 11:, 0] = True; batch['rest'][:, :, 2] = True; batch['rest'][:, :, 5] = True
    traj = batch['states'].clone()
    time = torch.arange(traj.shape[1]) * ev.STEP
    traj[:, :, 0, 0] = .02 * time                       # object 0 slides at 2 cm/s but reports zero velocity
    traj[:, :, 1, 0] = 1. * time                        # object 1 moves fast but is never at rest in the truth
    traj[:, :, 0, 9] = .5                               # state velocity (z) is reported separately
    stats = {k: v.numpy() for k, v in ev.episode_stats(traj, batch).items()}
    drift = ev.aggregate(stats, np.ones(4, bool))['rest_drift']
    assert drift['n_object_frames'] == 4 * 10           # static slot 2 and padded slot 5 are ignored
    assert drift['fd_speed_mps'] == pytest.approx(.02, rel=1e-4) and drift['state_speed_mps'] == pytest.approx(.5)
    truth = ev.aggregate({k: v.numpy() for k, v in ev.episode_stats(batch['states'], batch).items()}, np.ones(4, bool))
    assert truth['rest_drift']['fd_speed_mps'] == 0 and truth['pos_err_m']['1s'] == 0 and truth['pos_err_m']['4s'] is None


def test_position_error_subsets_and_fell_confusion():
    batch = toy_batch(); traj = batch['states'].clone()
    for episode, error in enumerate((.1, .2, .3, .4)): traj[episode, :, :2, 2] += error    # both dynamic objects off by `error`
    traj[:, :, 2, 0] += 5.                                                                 # static slot must not count
    batch['fell'][0, 0] = True; batch['fell'][1, 0] = True
    traj[0, 5, 0, :3] = torch.tensor([2., .2, 0.])                                         # true positive
    traj[2, 5, 1, :3] = torch.tensor([2., .2, 0.])                                         # false positive
    stats = {k: v.numpy() for k, v in ev.episode_stats(traj, batch).items()}
    chaotic = np.array([False, False, True, True]); live = np.array([True, False, True, False])
    kinds = np.array(['push', 'push', 'stack', 'stack'])
    masks = ev.subset_masks(chaotic, live)
    assert masks['live_physics']['gentle'].tolist() == [True, False, False, False] and masks['all_physics']['chaotic'].tolist() == [False, False, True, True]
    report = ev.engine_report(stats, chaotic, live, kinds)
    sub = report['subsets']
    assert sub['all_physics']['gentle']['pos_err_m']['0.5s'] == pytest.approx(.15, rel=1e-5)
    assert sub['all_physics']['chaotic']['pos_err_m']['1s'] == pytest.approx(.35, rel=1e-5)
    assert sub['live_physics']['chaotic']['pos_err_m']['1s'] == pytest.approx(.3, rel=1e-5)
    assert sub['live_physics']['all']['n_episodes'] == 2 and sub['live_physics']['all']['n_dynamic_objects'] == 4
    assert report['per_kind']['stack']['all_physics']['pos_err_m']['1s'] == pytest.approx(.35, rel=1e-5)
    fell = sub['all_physics']['all']['fell']
    assert fell['confusion'] == dict(tp=1, fp=1, fn=1, tn=5) and fell['accuracy'] == .75 and fell['precision'] == .5 and fell['recall'] == .5
    assert sub['all_physics']['all']['penetration']['mean_m'] is None                      # no penetration function supplied


def test_rollout_is_autoregressive_and_freezes_diverged_episodes():
    batch = toy_batch(episodes=3, steps=6); calls = []
    def step(state, props, context, mask):
        calls.append(state.clone()); out = state.clone(); out[..., 0] += 1.
        if len(calls) == 3: out[1, 0, 0] = float('nan')
        if len(calls) == 5: out[2, 0, 7] = 1e9
        return out
    traj, diverged = ev.rollout(step, batch['states'][:, 0], batch['props'], batch['context'], batch['mask'], 6)
    assert traj.shape == (3, 7, N, 13) and diverged.tolist() == [False, True, True] and torch.isfinite(traj).all()
    assert traj[0, :, 0, 0].tolist() == [0, 1, 2, 3, 4, 5, 6]
    assert traj[1, :, 0, 0].tolist() == [0, 1, 2, 2, 2, 2, 2] and traj[2, :, 0, 0].tolist() == [0, 1, 2, 3, 4, 4, 4]
    assert torch.equal(calls[2], traj[:, 2])                                               # fed its own prediction
    stats = ev.episode_stats(traj, batch, diverged)
    assert ev.aggregate({k: v.numpy() for k, v in stats.items()}, np.ones(3, bool))['diverged_episodes'] == 2


def test_penetration_statistics_use_worst_component_on_dynamic_objects():
    batch = toy_batch(episodes=2, steps=4)
    def pen_fn(state, props, context, mask):
        zero = torch.zeros(state.shape[:2])
        support = zero.clone(); support[:, 0] = .002; support[:, 2] = 1.               # static slot ignored
        objects = zero.clone(); objects[0, 1] = .02
        return dict(support=support, floor=zero, objects=objects)
    stats = {k: v.numpy() for k, v in ev.episode_stats(batch['states'], batch, pen_fn=pen_fn).items()}
    pen = ev.aggregate(stats, np.ones(2, bool))['penetration']
    assert pen['n_object_frames'] == 2 * 2 * 4 and pen['max_m'] == pytest.approx(.02)
    assert pen['mean_m'] == pytest.approx((.002 * 2 + .02) / 4) and pen['frac_over_5mm'] == pytest.approx(.25)


def test_ship_decision_and_bootstrap_ci():
    batch = toy_batch(episodes=40, steps=40); rng = torch.Generator().manual_seed(0)
    def engine(scale, depth):
        traj = batch['states'].clone(); traj[:, :, :2, 0] += scale * (1 + torch.rand(40, 1, 1, generator=rng))
        pen = lambda s, p, c, m: dict(support=torch.full(s.shape[:2], depth), floor=torch.zeros(s.shape[:2]), objects=torch.zeros(s.shape[:2]))
        return {k: v.numpy() for k, v in ev.episode_stats(traj, batch, pen_fn=pen).items()}
    stats = dict(old_mlp=engine(.2, .01), new_projected=engine(.1, .001), new_raw=engine(.3, .02))
    chaotic = np.arange(40) % 2 == 0; live = np.ones(40, bool)
    decision = ev.ship_decision(stats, chaotic, live, resamples=200, seed=1)
    projected, raw = decision['variants']['new_projected'], decision['variants']['new_raw']
    assert decision['ship'] is True and projected['includes_non_learned_projection'] and not raw['includes_non_learned_projection']
    assert raw['ship'] is False and decision['model_alone_passes'] is False
    delta = projected['chaotic_pos_err_2s_m']
    assert delta['ci95'][0] <= delta['delta'] <= delta['ci95'][1] < 0 and delta['n_episodes'] == 20
    assert projected['mean_penetration_m']['delta'] == pytest.approx(-.009, rel=1e-4)
    assert ev.ship_decision(stats, chaotic, live, resamples=200, seed=1) == decision          # seeded
    empty = ev.ship_decision(stats, chaotic, np.zeros(40, bool), resamples=10)
    assert empty['ship'] is None


def write_shards(root: Path, columns):
    states, props, context, mask, fell, tipped = columns
    count, frames = states.shape[:2]
    flags = np.zeros((count, frames, N), bool); rest = flags.copy()
    rest[:, 1:] = (np.linalg.norm(states[:, 1:, :, 7:10], axis=-1) < .01) & mask[:, None]
    (root / 'test').mkdir(parents=True)
    for shard, select in enumerate((slice(0, 3), slice(3, None))):
        np.savez_compressed(root / 'test' / f'shard_{shard:05d}.npz', states=states[select], props=props[select], context=context[select],
                            mask=mask[select], contact_support=flags[select], contact_floor=flags[select], contact_object=flags[select],
                            rest=rest[select], fell=fell[select], tipped=tipped[select], kind=np.array(['push', 'tilt', 'tilt', 'noclip', 'push', 'still'])[select],
                            chaotic=(fell | tipped).any(1)[select], live_physics=np.array([True, True, False, True, True, True])[select])
    (root / 'meta.json').write_text(json.dumps(dict(note='synthetic test shards')))


def test_baseline_only_cli_end_to_end(tmp_path, episodes):
    if not (ROOT / 'models/mlp_multistep.pt').exists(): pytest.skip('old checkpoint not present')
    write_shards(tmp_path / 'data', episodes)
    out = tmp_path / 'eval.json'
    ev.main(['--data', str(tmp_path / 'data'), '--split', 'test', '--old', str(ROOT / 'models/mlp_multistep.pt'),
             '--old-graph', str(ROOT / 'models/graph_impulse.pt'), '--out', str(out), '--device', 'cpu', '--bootstrap', '20'])
    result = json.loads(out.read_text())
    assert {'physics_reference', 'old_mlp', 'old_graph'} <= set(result['engines']) and 'new_raw' not in result['engines']
    reference = result['engines']['physics_reference']['subsets']
    assert reference['all_physics']['all']['pos_err_m']['2s'] == 0 and reference['all_physics']['all']['fell']['accuracy'] == 1
    assert result['data']['counts']['live_physics']['all'] == 5 and result['data']['kinds']['tilt'] == 2
    old = result['engines']['old_mlp']
    assert old['subsets']['live_physics']['chaotic']['pos_err_m']['2s'] > 0 and old['latency_ms_per_step'] > 0
    assert old['subsets']['all_physics']['all']['pos_err_m']['4s'] is None                    # 2 s test episodes
    assert result['ship_decision']['variants'] == {} and result['ship_decision']['ship'] is None
    ui = ev.summarize_for_ui(result)
    assert ui['old']['chaotic']['pos_err_m']['2s'] == old['subsets']['live_physics']['chaotic']['pos_err_m']['2s'] and 'new_raw' not in ui
    assert len(json.dumps(ui)) < 20000 and '| old_mlp |' in out.with_suffix('.md').read_text()
    limited = ev.load_split(tmp_path / 'data', 'test', max_episodes=3)
    assert len(limited['states']) == 3
