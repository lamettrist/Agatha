import sys
from pathlib import Path
import json
import numpy as np
import pybullet as p
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import generate_contact_data as g
from agatha.schema import Scene, SceneObject, Push, Tilt
from agatha.physics import World, simulate

N, T = 8, 20

def load(root, split):
    files = sorted((root/split).glob('shard_*.npz'))
    data = [np.load(f) for f in files]
    return {k: np.concatenate([d[k] for d in data]) for k in data[0].files}

def test_generation_shapes_dtypes_and_determinism(tmp_path):
    kwargs = dict(train_episodes=16, val_episodes=4, test_episodes=4, steps=T, max_objects=N, workers=2,
                  shard_size=10, seed=5, chunk=3, quiet=True)
    meta = g.generate(tmp_path/'a', **kwargs)
    g.generate(tmp_path/'b', **kwargs)
    assert len(list((tmp_path/'a'/'train').glob('shard_*.npz'))) == 2
    assert json.loads((tmp_path/'a'/'meta.json').read_text())['step_s'] == .05
    assert sum(meta['counts'].values())+sum(meta['dropped'].values()) == 24
    assert meta['transitions'] == sum(meta['counts'].values())*T
    for key in ('chaotic_fraction', 'live_physics_fraction', 'rest_fraction_object_frames', 'kinds'):
        assert key in meta['splits']['train']
    for split in g.SPLITS:
        a, b = load(tmp_path/'a', split), load(tmp_path/'b', split)
        e = len(a['states'])
        assert e == meta['counts'][split]
        shapes = dict(states=(e, T+1, N, 13), props=(e, N, 9), context=(e, 10), mask=(e, N), contact_support=(e, T+1, N),
                      contact_floor=(e, T+1, N), contact_object=(e, T+1, N), rest=(e, T+1, N), fell=(e, N), tipped=(e, N),
                      kind=(e,), chaotic=(e,), live_physics=(e,))
        for key, shape in shapes.items():
            assert a[key].shape == shape, key
            assert np.array_equal(a[key], b[key]), key  # same seed -> identical arrays
        for key in ('states', 'props', 'context'): assert a[key].dtype == np.float32
        for key in ('mask', 'contact_support', 'contact_floor', 'contact_object', 'rest', 'fell', 'tipped', 'chaotic', 'live_physics'):
            assert a[key].dtype == bool, key
        assert a['kind'].dtype == np.dtype('<U24') and set(a['kind']) <= set(g.KIND_WEIGHTS)
        assert np.isfinite(a['states']).all() and a['mask'].any(1).all()
        assert np.allclose(np.linalg.norm(a['states'][..., 3:7], axis=-1), 1, atol=1e-4)
        pad = ~a['mask']
        assert np.all(a['states'][:, :, :, [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]].transpose(0, 2, 1, 3)[pad] == 0)
        assert np.all(a['states'][..., 6].transpose(0, 2, 1)[pad] == 1)
        assert np.all(a['props'][pad] == [0, 0, 0, 1, 0, 0, 0, 0, 0])
        assert not (a['rest'] | a['contact_support'] | a['contact_floor'] | a['contact_object']).transpose(0, 2, 1)[pad].any()
        # quaternion sign is continuous in time
        assert (np.sum(a['states'][:, 1:, :, 3:7]*a['states'][:, :-1, :, 3:7], -1) >= 0).all()
        live = a['live_physics']
        assert np.all(a['props'][live][:, :, 6:] == 0) and np.allclose(a['context'][live][:, 8:], [.5, 0])
        assert np.all(a['live_physics'][a['kind'] == 'legacy_layout'])
    # different splits use independent seeds
    assert not np.array_equal(load(tmp_path/'a', 'val')['states'], load(tmp_path/'a', 'test')['states'])

def test_every_scenario_builds_and_rest_stays_still():
    for kind in g.KIND_WEIGHTS:
        for i in range(2):
            case = g.make_case(np.random.default_rng([3, i]), N, kind)
            episode = g.simulate_case(case, steps=12, max_objects=N)
            assert episode is not None and episode['kind'] == kind and episode['mask'].any()
            if kind == 'rest':
                dynamic = episode['mask'] & (episode['props'][:, 5] < .5)
                assert np.abs(episode['states'][-1, :, :3]-episode['states'][0, :, :3]).max() < 2e-3
                assert episode['rest'][4:, dynamic].all() and not episode['chaotic']
                assert episode['init_penetration'] < 1e-3

def _scene():
    return Scene(objects=[SceneObject(id='a', label='a', size=(.1, .1, .1), position=(0, .8, 0)),
                          SceneObject(id='b', label='b', size=(.08, .05, .08), position=(0, .875, 0)),
                          SceneObject(id='c', label='c', size=(.06, .2, .06), position=(.3, .9, .1), friction=.7)])

def _rollout(world, steps=120):
    world.apply([Push(type='push', id='a', direction=(1, .2), impulse_Ns=.4), Tilt(type='tilt_support', angle_deg=8, axis='z')])
    world.step(steps)
    return world.snapshot(0)

def test_default_world_unchanged_and_overrides():
    with World(_scene()) as w: base = _rollout(w)
    explicit = dict(objects={}, support_friction=.5, support_restitution=0)
    with World(_scene(), dynamics=explicit) as w: assert _rollout(w) == base
    with World(_scene(), dynamics=None) as w:
        info = p.getDynamicsInfo(w.bodies['a'], -1, physicsClientId=w.client)
        assert info[1] == .5 and info[5] == 0 and info[6] == 0 and info[7] == 0
    dynamics = dict(objects=dict(a=dict(restitution=.5, rolling_friction=.01, spinning_friction=.02)),
                    support_friction=.2, support_restitution=.4)
    with World(_scene(), dynamics=dynamics) as w:
        info = p.getDynamicsInfo(w.bodies['a'], -1, physicsClientId=w.client)
        assert np.allclose([info[5], info[6], info[7]], [.5, .01, .02])
        support = p.getDynamicsInfo(w.bodies['support'], -1, physicsClientId=w.client)
        assert np.allclose([support[1], support[5]], [.2, .4])
        assert p.getDynamicsInfo(w.bodies['floor'], -1, physicsClientId=w.client)[1] == .5
        assert _rollout(w) != base
    # the public simulate() result is still deterministic and unaffected
    assert simulate(_scene(), []) == simulate(_scene(), [])

def test_contact_flags():
    with World(_scene()) as w:
        w.step(240)
        flags = w.contact_flags()
        assert flags['a'] == (True, False, True) and flags['b'] == (False, False, True)
        assert flags['c'] == (True, False, False)
        assert ('a', 'support') in w.contact_pairs()

def test_supplementary_kinds_build_and_leave_the_default_mix_alone(tmp_path):
    assert not set(g.EXTRA_KINDS) & set(g.KIND_WEIGHTS) and set(g.HARD_MIX) == set(g.EXTRA_KINDS)
    for kind in g.EXTRA_KINDS:
        for i in range(3):
            case = g.make_case(np.random.default_rng([4, i]), N, kind)
            episode = g.simulate_case(case, steps=12, max_objects=N)
            assert episode is not None and episode['kind'] == kind and episode['mask'].any()
            real = episode['mask']
            if kind == 'bouncy_toss':
                assert not episode['live_physics']
                assert np.all((episode['props'][real, 6] >= .3) & (episode['props'][real, 6] <= .7))
                assert .3 <= episode['context'][9] <= .7
            if kind == 'tilt_api':
                assert 8 <= np.degrees(np.abs(episode['context'][[4, 6]]).max()) <= 30.001
            if kind == 'domino_chain': assert real.sum() >= 3
            if kind == 'hard_push':
                assert any(a.type == 'push' and a.impulse_Ns >= .07 for a in case['actions'])
    # the default mix never draws a supplementary kind and ignores the override machinery
    a = g.make_case(np.random.default_rng(9), N); b = g.make_case(np.random.default_rng(9), N, kind_weights=g.KIND_WEIGHTS)
    assert a['kind'] == b['kind'] and a['scene'] == b['scene'] and a['kind'] in g.KIND_WEIGHTS
    kwargs = dict(train_episodes=12, val_episodes=2, test_episodes=2, steps=T, max_objects=N, workers=2, shard_size=12,
                  seed=5, chunk=4, quiet=True)
    meta = g.generate(tmp_path/'hard', kind_weights=g.HARD_MIX, **kwargs)
    assert meta['kind_weights'] == g.HARD_MIX and set(load(tmp_path/'hard', 'train')['kind']) <= set(g.EXTRA_KINDS)
    assert g._ACTIVE_WEIGHTS is None
    default = g.generate(tmp_path/'default', **kwargs)
    assert default['kind_weights'] == g.KIND_WEIGHTS and set(load(tmp_path/'default', 'train')['kind']) <= set(g.KIND_WEIGHTS)

def test_tilt_kinds_land_flat_or_stay_leaned_and_leave_the_other_mixes_alone(tmp_path):
    assert set(g.TILT_MIX) == set(g.TILT_KINDS) and g.MIXES['tilt'] is g.TILT_MIX
    assert not set(g.TILT_KINDS) & (set(g.KIND_WEIGHTS) | set(g.HARD_MIX) | set(g.EXTRA_KINDS))
    assert np.allclose(g.tilt_from_flat_deg([[0, 0, 0, 1], g.q_axis((1, 0, 0), np.radians(30)), g.q_axis((0, 0, 1), np.radians(100))]),
                       [0, 30, 10], atol=1e-6)
    flat = {k: [] for k in g.TILT_KINDS}
    for kind in g.TILT_KINDS:
        for i in range(6):
            case = g.make_case(np.random.default_rng([6, i]), N, kind)
            episode = g.simulate_case(case, steps=80, max_objects=N)
            assert episode is not None and episode['kind'] == kind and episode['mask'].any()
            assert not case['actions'] and episode['init_penetration'] < 1e-3
            assert np.all(episode['context'][4:7] == 0) and abs(episode['context'][3]-9.81) < 1e-4
            dynamic = episode['mask'] & (episode['props'][:, 5] < .5)
            tilt = g.tilt_from_flat_deg(episode['states'][:, :, 3:7])
            flat[kind].append(bool((tilt[-1][dynamic] < 1).all()))
            if kind == 'tilt_release': assert tilt[0][dynamic].max() > .2
            if kind == 'settle_wobble':
                assert .9 < tilt[0][dynamic].max() < 15.01                       # starts 1-15 deg from flat ...
                first = np.flatnonzero(dynamic)[0]
                assert tilt[-1][first] < 1 and episode['rest'][-1][first]        # ... and rocks back onto its face
            if kind == 'lean_stable':
                assert g.stays_leaned(episode) and tilt[-1][1] > 10 and episode['rest'][-1][1]
                assert episode['props'][0, 3] >= 1.5 and episode['props'][1, 3] <= .5   # heavy block, light plank
    assert np.mean(flat['tilt_release']) >= .8 and np.mean(flat['settle_wobble']) >= .8 and not any(flat['lean_stable'])
    # the default and hard mixes never draw a tilt kind, and the same seed still gives the same default recipe
    for weights in (None, g.HARD_MIX):
        for i in range(40):
            assert g.make_case(np.random.default_rng([8, i]), N, kind_weights=weights)['kind'] not in g.TILT_KINDS
    kwargs = dict(train_episodes=12, val_episodes=2, test_episodes=2, steps=T, max_objects=N, workers=2, shard_size=12,
                  seed=5, chunk=4, quiet=True)
    meta = g.generate(tmp_path/'tilt', kind_weights=g.TILT_MIX, **kwargs)
    assert meta['kind_weights'] == g.TILT_MIX and set(load(tmp_path/'tilt', 'train')['kind']) <= set(g.TILT_KINDS)
    assert 'tilt_release' in meta['ranges'] and g._ACTIVE_WEIGHTS is None
