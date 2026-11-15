import sys
from pathlib import Path
import json
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import generate_soft_data as g

T = 12
V, M, F, N = g.V_MAX, g.M_MAX, g.F_MAX, g.N_RIGID

def load(root, split):
    data = [np.load(f) for f in sorted((root/split).glob('shard_*.npz'))]
    return {k: np.concatenate([d[k] for d in data]) for k in data[0].files}

def test_generation_shapes_dtypes_and_determinism(tmp_path):
    kwargs = dict(train_episodes=10, val_episodes=3, test_episodes=3, steps=T, workers=2, shard_size=6, seed=11,
                  chunk=2, quiet=True)
    meta = g.generate(tmp_path/'a', **kwargs)
    g.generate(tmp_path/'b', **kwargs)
    assert len(list((tmp_path/'a'/'train').glob('shard_*.npz'))) == 2
    saved = json.loads((tmp_path/'a'/'meta.json').read_text())
    assert saved['step_s'] == .05 and saved['substeps_per_step'] == 12 and saved['max_vertices'] == 128
    assert sum(meta['counts'].values())+sum(meta['dropped'].values()) == 16
    assert sum(meta['drop_reasons'].values()) == sum(meta['dropped'].values())
    assert set(meta['drop_reasons']) <= set(g.DROP_RULES)
    assert meta['transitions'] == sum(meta['counts'].values())*T
    for key in ('kinds', 'mean_max_strain', 'rest_frame_fraction', 'sim_seconds_per_episode'):
        assert key in meta['overall']
    for split in g.SPLITS:
        a, b = load(tmp_path/'a', split), load(tmp_path/'b', split)
        e = len(a['verts'])
        assert e == meta['counts'][split] and set(a) == set(g.ARRAY_KEYS) == set(g.FORMAT)
        shapes = dict(verts=(e, T+1, V, 3), vert_vel=(e, T+1, V, 3), vert_mask=(e, V), rest_edges=(e, M, 2), rest_len=(e, M),
                      faces=(e, F, 3), soft_props=(e, 6), soft_raw=(e, 10), states=(e, T+1, N, 13), props=(e, N, 9),
                      mask=(e, N), context=(e, 10), kind=(e,), final_com=(e, 3), max_deformation=(e,), max_penetration=(e,),
                      min_volume_ratio=(e,), rest=(e, T+1), fell=(e,), soft_shape=(e,), n_verts=(e,), n_edges=(e,),
                      n_faces=(e,), episode_index=(e,))
        for key, shape in shapes.items():
            assert a[key].shape == shape, key
            assert np.array_equal(a[key], b[key]), key   # same seed -> identical arrays
        for key in ('verts', 'vert_vel', 'rest_len', 'soft_props', 'soft_raw', 'states', 'props', 'context', 'final_com',
                    'max_deformation', 'max_penetration', 'min_volume_ratio'):
            assert a[key].dtype == np.float32 and np.isfinite(a[key]).all(), key
        for key in ('vert_mask', 'mask', 'rest', 'fell'): assert a[key].dtype == bool, key
        for key in ('rest_edges', 'faces', 'n_verts', 'n_edges', 'n_faces'): assert a[key].dtype == np.int16, key
        assert a['kind'].dtype == np.dtype('<U24') and set(a['kind']) <= set(g.KIND_WEIGHTS)
        assert set(a['soft_shape']) <= {'cube', 'ball', 'cloth'}
        for i in range(e):
            n, m, f = int(a['n_verts'][i]), int(a['n_edges'][i]), int(a['n_faces'][i])
            assert 0 < n <= V and a['vert_mask'][i].sum() == n and a['vert_mask'][i, :n].all()
            assert np.all(a['verts'][i, :, n:] == 0) and np.all(a['vert_vel'][i, :, n:] == 0)
            edges = a['rest_edges'][i]
            assert np.all(edges[m:] == -1) and np.all(a['rest_len'][i, m:] == 0)
            assert edges[:m].min() >= 0 and edges[:m].max() < n and np.all(edges[:m, 0] < edges[:m, 1])
            assert len(np.unique(edges[:m], axis=0)) == m
            assert np.all(a['faces'][i, f:] == -1) and a['faces'][i, :f].min() >= 0 and a['faces'][i, :f].max() < n
            assert set(a['faces'][i, :f].ravel()) <= set(edges[:m].ravel())
            # frame 0 is the undeformed mesh: edge lengths equal the rest lengths
            x0 = a['verts'][i, 0]
            assert np.allclose(np.linalg.norm(x0[edges[:m, 0]]-x0[edges[:m, 1]], axis=1), a['rest_len'][i, :m], atol=1e-5)
            cloth = a['soft_shape'][i] == 'cloth'
            assert a['soft_props'][i, 4] == float(cloth) and (a['kind'][i] in g.CLOTH_KINDS) == cloth
            assert np.allclose(a['final_com'][i], a['verts'][i, -1, :n].mean(0), atol=1e-5)
            length = np.linalg.norm(a['verts'][i][:, edges[:m, 0]]-a['verts'][i][:, edges[:m, 1]], axis=-1)
            assert np.isclose(np.abs(length/a['rest_len'][i, :m]-1).max(), a['max_deformation'][i], atol=1e-4)
            assert a['max_deformation'][i] <= g.STRAIN_LIMIT['cloth' if cloth else 'volumetric']
            assert a['verts'][i, :, :n, 1].min() > -.03 and a['max_penetration'][i] <= g.TUNNEL_DEPTH
            assert a['min_volume_ratio'][i] > 0
        assert np.all(a['soft_props'][:, 0] > 0) and np.all(a['soft_props'][:, 5] > 0) and np.all(np.abs(a['soft_props'][:, 1]) < 1.6)
        # rigid slots follow the box format conventions
        pad = ~a['mask']
        assert np.allclose(np.linalg.norm(a['states'][..., 3:7], axis=-1), 1, atol=1e-4)
        assert np.all(a['states'][:, :, :, [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]].transpose(0, 2, 1, 3)[pad] == 0)
        assert np.all(a['states'][..., 6].transpose(0, 2, 1)[pad] == 1)
        assert np.all(a['props'][pad] == [0, 0, 0, 1, 0, 0, 0, 0, 0])
        assert (np.sum(a['states'][:, 1:, :, 3:7]*a['states'][:, :-1, :, 3:7], -1) >= 0).all()
        assert np.all(a['context'][:, 4:8] == [0, 0, 0, 1]) and np.all(a['context'][:, 9] == 0)
    assert not np.array_equal(load(tmp_path/'a', 'val')['verts'], load(tmp_path/'a', 'test')['verts'])

def test_every_scenario_builds_valid_meshes_and_placements():
    for kind in g.KIND_WEIGHTS:
        for i in range(3):
            case = g.make_case(np.random.default_rng([i, len(kind)]), kind)
            mesh = case.mesh; world = case.world_verts()
            assert len(mesh['pts']) <= V and len(mesh['edges']) <= M and len(mesh['faces']) <= F and len(case.rigid) <= N
            assert world[:, 1].min() >= case.h+g.COLLISION_MARGIN-1e-9   # starts above the table plane
            if mesh['tets'] is not None:
                a, b, c, d = (mesh['pts'][mesh['tets'][:, j]] for j in range(4))
                assert (np.einsum('ij,ij->i', np.cross(b-a, c-a), d-a) > 0).all()
                expected = np.prod(case.soft['size'])*(1 if case.soft['shape'] == 'cube' else np.pi/6)
                assert np.isclose(mesh['measure'], expected, rtol=.25 if case.soft['shape'] == 'ball' else 1e-6)
                # closed, outward oriented surface: divergence theorem reproduces the volume
                f = mesh['faces']; p0, p1, p2 = (mesh['pts'][f[:, j]] for j in range(3))
                assert np.isclose(np.einsum('ij,ij->i', p0, np.cross(p1, p2)).sum()/6, mesh['measure'], rtol=1e-6)
            for b in case.rigid: assert b['pos'][1]-b['size'][1]/2 >= case.h-1e-9

def test_free_fall_and_velocity_convention():
    case = g.make_case(np.random.default_rng(3), 'soft_drop')
    case.gravity = 9.81
    case.place_soft(0., 0., case.h+.6, g.q_yaw(.3))
    case.soft['linvel'] = np.asarray([.4, 0., -.2]); case.soft['angvel'] = np.zeros(3)
    episode = g.simulate_case(case, steps=4)
    n = int(episode['n_verts']); x, v = episode['verts'][:, :n].astype(float), episode['vert_vel'][:, :n].astype(float)
    t = np.arange(5)*g.STEP
    assert np.allclose(v[:, :, 1].mean(1), -9.81*t, atol=.02) and np.allclose(v[:, :, [0, 2]].mean(1), [.4, -.2], atol=1e-3)
    assert np.allclose(x[:, :, 1].mean(1)-x[0, :, 1].mean(), -.5*9.81*t*t, atol=.01)
    assert np.allclose(x[4, :, 0].mean()-x[0, :, 0].mean(), .4*t[4], atol=2e-3)
    assert episode['max_deformation'] < .02 and not episode['rest'].any() and not episode['mask'].any()
