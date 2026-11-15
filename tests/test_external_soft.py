"""MeshGraphNets -> soft shard conversion (scripts/prepare_external_soft.py) on synthetic trajectories; no external data needed."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import prepare_external_soft as pe


def grid_mesh(n=21):
    axis = np.linspace(0., 1., n)
    pos = np.stack(np.meshgrid(axis, axis, indexing='ij'), -1).reshape(-1, 2)
    index = lambda i, j: i*n+j
    cells = [(index(i, j), index(i+1, j), index(i+1, j+1)) for i in range(n-1) for j in range(n-1)]
    cells += [(index(i, j), index(i+1, j+1), index(i, j+1)) for i in range(n-1) for j in range(n-1)]
    return pos.astype(np.float32), np.array(cells, np.int32)


def synthetic(frames=101, n=21, dt=.01):
    """Z-up cloth sheet in free fall along -z (handles fixed), plus a sphere of radius 0.1 moving along +y."""
    pos, cells = grid_mesh(n)
    t = np.arange(frames)*dt
    cloth = np.zeros((frames, len(pos), 3), np.float32)
    cloth[:, :, 0], cloth[:, :, 1] = pos[:, 0], pos[:, 1]
    cloth[:, :, 2] = (-.5*pe.GRAVITY*t**2)[:, None]
    handles = np.array([0, n*(n-1)])
    cloth[:, handles, 2] = 0.
    rng = np.random.default_rng(0)
    unit = rng.normal(size=(40, 3)); unit /= np.linalg.norm(unit, axis=1, keepdims=True)
    unit[:6] = np.concatenate((np.eye(3), -np.eye(3)))                             # the bounding box is centred on the sphere
    centre = np.stack((np.full(frames, .5), -3+t*.5, np.full(frames, -8.)), -1)
    sphere = (centre[:, None]+.1*unit[None]).astype(np.float32)
    node_type = np.zeros(len(pos)+40, np.int32); node_type[handles] = pe.NODE_HANDLE; node_type[len(pos):] = pe.NODE_OBSTACLE
    world = np.concatenate((cloth, sphere), 1)
    mesh_pos = np.concatenate((pos, np.zeros((40, 2), np.float32)))
    meta = dict(dt=dt, trajectory_length=frames, features=dict(
        cells=dict(shape=[1, len(cells), 3], dtype='int32'), mesh_pos=dict(shape=[1, len(mesh_pos), 2], dtype='float32'),
        node_type=dict(shape=[frames, len(node_type), 1], dtype='int32'), world_pos=dict(shape=[frames, len(node_type), 3], dtype='float32')))
    example = dict(cells=cells[None].tobytes(), mesh_pos=mesh_pos[None].tobytes(),
                   node_type=np.repeat(node_type[None, :, None], frames, 0).tobytes(), world_pos=world.tobytes())
    return meta, example, handles


def test_tfrecord_roundtrip(tmp_path):
    meta, example, _ = synthetic(frames=5, n=4)
    pe.write_tfrecord(tmp_path/'a.tfrecord', [example, example])
    found = list(pe.records(tmp_path/'a.tfrecord'))
    assert len(found) == 2
    traj = pe.trajectory(found[1], meta)
    assert traj['world_pos'].shape == (5, 56, 3) and traj['cells'].dtype == np.int32
    assert traj['world_pos'].tobytes() == example['world_pos']


def test_coarsen_keeps_handles_and_connectivity():
    pos, cells = grid_mesh(21)
    kept, edges, faces = pe.coarsen(pos.astype(np.float64), cells, 128, keep=[0, 420])
    assert len(kept) == 128 and len(set(kept.tolist())) == 128 and set(kept[:2].tolist()) == {0, 420}
    assert (edges[:, 0] < edges[:, 1]).all() and edges.max() < 128 and len(np.unique(edges, axis=0)) == len(edges)
    assert pe.connected(128, edges) and 200 < len(edges) <= 768
    assert len(faces) > 100 and (faces[:, 0] != faces[:, 1]).all()
    face_edges = np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1)
    assert {tuple(e) for e in face_edges.tolist()} <= {tuple(e) for e in edges.tolist()}      # every face edge is a mesh edge
    same = pe.coarsen(pos.astype(np.float64), cells, 10_000)                                   # no sub-sampling: the mesh itself
    assert len(same[0]) == len(pos) and len(same[2]) == len(cells)


def test_conversion_units_axes_gravity_and_sphere(tmp_path):
    meta, example, handles = synthetic()
    pe.write_tfrecord(tmp_path/'t.tfrecord', [example])
    traj = pe.trajectory(next(pe.records(tmp_path/'t.tfrecord')), meta)
    scale = .36
    episode, reason = pe.convert_trajectory(traj, meta['dt'], scale=scale, max_verts=128, handle_height=5.)
    assert reason is None
    verts = episode['verts']                                                       # [T+1,128,3], 20 Hz, Y-up
    assert verts.shape == (13, 128, 3) and episode['stride'] == pytest.approx(.05/(.01*.6))
    free = ~episode['pinned']
    assert episode['pinned'].sum() == 2 and np.allclose(verts[:, episode['pinned'], 1], 5.)
    acc = np.diff(verts[:, free], 2, axis=0)/pe.STEP**2
    assert np.allclose(acc[..., 1], -pe.GRAVITY, rtol=4e-3) and np.abs(acc[..., [0, 2]]).max() < 1e-4    # gravity keeps its value (linear interpolation: -0.2%), along -y
    assert np.allclose(episode['vert_vel'][4, free, 1], -pe.GRAVITY*4*pe.STEP, rtol=1e-3)
    assert np.ptp(verts[0, :, 0]) == pytest.approx(scale, rel=1e-6) and np.ptp(verts[0, :, 2]) == pytest.approx(scale, rel=1e-6)
    length = np.linalg.norm(verts[0][episode['rest_edges'][:, 0]]-verts[0][episode['rest_edges'][:, 1]], axis=-1)
    assert np.allclose(length, episode['rest_len'], rtol=1e-5)
    # sphere: exact radius, rounded-box props, kinematic velocity; source +y motion -> -z
    assert episode['props'][0, 9] == pytest.approx(.1*scale, rel=1e-3) and episode['props'][0, 0] == pytest.approx(.2*scale, rel=1e-3)
    assert episode['states'][3, 0, 9] == pytest.approx(-.5*scale/.6, rel=1e-3) and episode['props'][0, 5] == 1
    assert episode['context'][7] == 0 and episode['context'][3] == pe.GRAVITY
    episode['source_index'] = 0
    arrays = pe.pack([episode], 'mgn_test')
    assert arrays['verts'].shape == (1, 13, 128, 3) and arrays['pinned'].sum() == 2 and arrays['props'].shape == (1, 1, 10)
    assert arrays['rest_edges'].shape == (1, 768, 2) and arrays['faces'].shape == (1, 256, 3) and arrays['mask'].all()
    # the trainer loader reads it (and defaults 'pinned' / pads props for soft_v1-style shards without them)
    import train_soft_ddp as ts
    (tmp_path/'d'/'test').mkdir(parents=True)
    np.savez(tmp_path/'d'/'test'/'shard_x_00000.npz', **arrays)
    loaded = ts.load_split(tmp_path/'d', 'test')
    assert loaded['pinned'].sum() == 2 and loaded['props'].shape[-1] == 10 and loaded['kind'][0] == 'mgn_test'
    plain = {k: v for k, v in arrays.items() if k != 'pinned'}; plain['props'] = arrays['props'][..., :9]
    (tmp_path/'e'/'test').mkdir(parents=True)
    np.savez(tmp_path/'e'/'test'/'shard_00000.npz', **plain)
    loaded = ts.load_split(tmp_path/'e', 'test')
    assert not loaded['pinned'].any() and loaded['props'].shape[-1] == 10 and float(np.abs(loaded['props'][..., 9]).max()) == 0


def test_skipped_sets_are_refused():
    with pytest.raises(SystemExit): pe.main(['--sets', 'flag_minimal', '--out', '/nonexistent'])
    assert 'wind' in pe.SKIPPED['flag_minimal'] and pe.split_of(8) == 'val' and pe.split_of(19) == 'test' and pe.split_of(0) == 'train'
