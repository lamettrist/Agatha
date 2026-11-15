"""Routing across the box / joint / soft models (agatha/router.py). CPU, a few seconds per test."""
import json
import math
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from agatha import main, router, world_model
from agatha.schema import Scene, Push, Remove, Gravity, Tilt
from agatha.storage import ROOT, Store

pytestmark = pytest.mark.skipif(not all((ROOT/'models'/n).exists() for n in ('world_model.pt', 'joint_gnn.pt', 'soft_gnn.pt')),
                                reason='needs the three shipped checkpoints')

def load(name: str) -> Scene:
    return Scene.model_validate_json((ROOT/'scenes'/f'demo_{name}.json').read_text())

def verts(frame: dict, name: str) -> np.ndarray:
    return np.asarray(next(s for s in frame['soft_bodies'] if s['id'] == name)['vertices']).reshape(-1, 3)

def joint_report(scene: Scene, result: dict):
    from agatha.joint_gnn import joint_violation, joint_coordinates
    ids = [o['id'] for o in result['frames'][0]['objects']]
    rows = torch.from_numpy(router.joint_rows(scene.joints, ids))[None]; mask = torch.ones(1, rows.shape[1], dtype=torch.bool)
    gaps, angles = [], []
    for frame in result['frames']:
        state = torch.tensor([[o['position']+o['quaternion']+o['velocity']+o['angular_velocity'] for o in frame['objects']]])
        gaps.append(float(joint_violation(state, rows, mask)['anchor_gap'].max()))
        angles.append(float(joint_coordinates(state, rows, mask).flatten()[0]))
    return gaps, angles

def test_schema_is_additive_and_validates_references():
    desk = load('desk')
    assert desk.joints == [] and desk.soft_bodies == []
    raw = json.loads((ROOT/'scenes/demo_laptop.json').read_text())
    raw['joints'][0]['child'] = 'nobody'
    with pytest.raises(ValueError): Scene.model_validate(raw)
    raw = json.loads((ROOT/'scenes/demo_soft.json').read_text())
    raw['soft_bodies'][0]['id'] = raw['objects'][1]['id']
    with pytest.raises(ValueError): Scene.model_validate(raw)

def test_route_decisions():
    assert router.route(load('desk'))['rigid'] == 'contact_gnn' and router.route(load('desk'))['soft'] is False
    assert router.route(load('laptop'))['rigid'] == 'joint_gnn' and router.route(load('laptop'))['soft'] is False
    decision = router.route(load('soft'))
    assert decision['rigid'] == 'contact_gnn' and decision['soft'] is True and len(decision['reasons']) == 2
    both = load('laptop'); both.soft_bodies = load('soft').soft_bodies
    assert router.route(both) == dict(rigid='joint_gnn', soft=True, reasons=router.route(both)['reasons'])

def test_box_only_scene_is_bit_identical_to_world_model():
    scene = load('desk'); actions = [Push(type='push', id='mug', direction=(1, 0), impulse_Ns=.55)]
    expected = world_model.rollout(scene, actions, 45, 30, model=world_model.load_model('default'))
    for architecture in ('auto', 'default'):
        routed = router.rollout(scene, actions, 45, 30, architecture=architecture)
        assert routed['frames'] == expected['frames']
        assert {k: v for k, v in routed.items() if k not in ('routing', 'joints', 'soft_meshes')} == expected
        assert routed['routing']['rigid'] == 'contact_gnn' and routed['joints'] == [] and routed['soft_meshes'] == {}
    explicit = router.rollout(scene, actions, 20, 20, architecture='mlp_multistep')
    assert explicit['model_architecture'] == 'mlp_multistep'
    assert explicit['frames'] == world_model.rollout(scene, actions, 20, 20, model=world_model.load_model('mlp_multistep'))['frames']
    for forced in ('joint_gnn', 'soft_gnn'):
        with pytest.raises(ValueError): router.rollout(scene, actions, 20, 20, architecture=forced)

def test_jointed_demo_keeps_the_hinge_closed_and_the_lid_moves(monkeypatch):
    scene = load('laptop')
    result = router.rollout(scene, [], 60, 30)
    assert result['model_architecture'] == 'joint_gnn' and result['projection_applied'] is True
    assert result['joints'][0]['id'] == 'laptop_hinge' and len(result['frames']) == 61
    assert np.isfinite([o['position']+o['quaternion'] for f in result['frames'] for o in f['objects']]).all()
    gaps, angles = joint_report(scene, result)
    assert max(gaps) < 1e-3                                    # anchor gap below 1 mm with the projection
    assert abs(angles[0]-math.radians(100)) < 1e-3 and abs(angles[-1]-angles[0]) > .1   # the open lid falls under gravity
    pushed = router.rollout(scene, [Push(type='push', id='laptop_lid', direction=(0, 0, 1), impulse_Ns=.4)], 30, 30)
    assert max(joint_report(scene, pushed)[0]) < 1e-3
    assert joint_report(scene, pushed)[1][15] < angles[15]-.05   # pushed towards closing: smaller hinge angle than unpushed
    monkeypatch.setenv(router.JOINT_PROJECTION_ENV, '0')
    assert router.rollout(scene, [], 10, 20)['projection_applied'] is False
    removed = router.rollout(scene, [Remove(type='remove', id='laptop_lid')], 10, 20)
    assert removed['joints'] == [] and 'laptop_lid' not in [o['id'] for o in removed['frames'][0]['objects']]

def test_soft_demo_vertices_stay_finite_and_above_the_table():
    scene = load('soft'); top = scene.support.height
    result = router.rollout(scene, [], 90, 30)
    assert result['model_architecture'] == 'contact_gnn+soft_gnn' and result['routing']['coupling'] == 'one_way_rigid_to_soft'
    assert 'One-way coupling' in result['disclaimer']
    assert set(result['soft_meshes']) == {'ball', 'cloth'} and len(result['frames']) == 91
    for name, mesh in result['soft_meshes'].items():
        count = mesh['vertex_count']; assert max(mesh['faces']) < count and len(mesh['faces']) % 3 == 0
        for frame in result['frames']:
            v = verts(frame, name)
            assert v.shape == (count, 3) and np.isfinite(v).all()
            inside = (np.abs(v[:, 0]) < scene.support.size[0]/2) & (np.abs(v[:, 2]) < scene.support.size[1]/2)
            assert v[inside, 1].min() > top-.005                # never more than 5 mm inside the table
    ball = [verts(f, 'ball')[:, 1].mean() for f in result['frames']]
    assert ball[0]-ball[-1] > .05 and abs(ball[-1]-ball[45]) < .01   # dropped, lands, then stays
    # rigid frames are those of the box model alone: soft bodies do not act on rigid bodies
    alone = world_model.rollout(scene, [], 90, 30, model=world_model.load_model('default'))
    assert [f['objects'] for f in result['frames']] == [f['objects'] for f in alone['frames']]

def test_resting_soft_cube_stays_put_and_actions_reach_soft_bodies():
    scene = load('soft'); top = scene.support.height
    cube = scene.soft_bodies[0].model_copy(update=dict(id='cube', kind='cube', size=(.1, .1, .1), position=(-.25, top+.05+.005, .08)))
    scene.soft_bodies = [cube]
    rest = router.rollout(scene, [], 90, 30)
    start, end = verts(rest['frames'][0], 'cube'), verts(rest['frames'][-1], 'cube')
    assert np.abs(end.mean(0)-start.mean(0)).max() < .01        # centre within 1 cm over 3 s
    assert min(verts(f, 'cube')[:, 1].min() for f in rest['frames']) > top-.005
    pushed = router.rollout(scene, [Push(type='push', id='cube', direction=(1, 0), impulse_Ns=.3)], 30, 30)
    assert verts(pushed['frames'][-1], 'cube')[:, 0].mean()-start[:, 0].mean() > .05   # 1 m/s start: it slides in +x
    removed = router.rollout(scene, [Remove(type='remove', id='cube'), Gravity(type='set_gravity', g=5)], 10, 20)
    assert removed['model_architecture'] == 'contact_gnn' and removed['soft_meshes'] == {} and removed['frames'][0]['soft_bodies'] == []
    with pytest.raises(ValueError):
        router.rollout(scene, [Remove(type='remove', id='cube'), Push(type='push', id='cube', direction=(1, 0), impulse_Ns=.3)], 10, 20)

def test_more_than_four_rigid_boxes_are_reported():
    scene = load('soft')
    for i in range(4):
        scene.objects.append(scene.objects[1].model_copy(update=dict(id=f'extra{i}', position=(-.5+.1*i, scene.support.height+.0505, .3))))
    result = router.rollout(scene, [], 4, 20)
    assert any('4 nearest rigid boxes' in note for note in result['approximations'])

def test_missing_checkpoint_is_file_not_found(monkeypatch, tmp_path):
    monkeypatch.setitem(router.CHECKPOINTS, 'joint_gnn', tmp_path/'absent.pt')
    monkeypatch.setitem(router.CHECKPOINTS, 'soft_gnn', tmp_path/'absent.pt')
    with pytest.raises(FileNotFoundError): router.rollout(load('laptop'), [], 10, 20)
    with pytest.raises(FileNotFoundError): router.rollout(load('soft'), [], 10, 20)
    assert router.rollout(load('desk'), [], 10, 20)['model_architecture'] == 'contact_gnn'
    client = TestClient(main.app)
    key = client.post('/api/scenes/demo?scenario=laptop').json()['scene_id']
    assert client.post('/api/simulate', json=dict(scene_id=key, engine='learned', duration=.5)).status_code == 503

def test_api_round_trip_for_the_three_demos(tmp_path, monkeypatch):
    monkeypatch.setattr(main, 'store', Store(tmp_path))
    client = TestClient(main.app)
    expected = dict(desk='contact_gnn', laptop='joint_gnn', soft='contact_gnn+soft_gnn')
    for scenario, architecture in expected.items():
        record = client.post(f'/api/scenes/demo?scenario={scenario}').json(); key = record['scene_id']
        assert Scene.model_validate(record['scene']) == load(scenario)
        for engine in ('learned', 'both'):
            response = client.post('/api/simulate', json=dict(scene_id=key, engine=engine, duration=1, fps=30))
            assert response.status_code == 200, response.text
            body = response.json(); learned = body['learned']
            assert learned['model_architecture'] == architecture and len(learned['frames']) == 31
            assert learned['routing']['rigid'] == architecture.split('+')[0] and learned['routing']['soft'] == (scenario == 'soft')
            assert {'t', 'objects'} <= set(learned['frames'][0]) and isinstance(learned['disclaimer'], str)
            if scenario == 'soft':
                first = learned['frames'][0]['soft_bodies'][0]
                assert len(first['vertices']) == 3*learned['soft_meshes'][first['id']]['vertex_count']
            if scenario == 'laptop': assert learned['joints'][0]['type'] == 'hinge'
            if engine == 'both' and scenario == 'desk':
                assert body['physics']['engine'] == 'physics' and body['error']['mean_position_error_m'] >= 0
            elif engine == 'both':   # no PyBullet reference for jointed / soft scenes: said, not faked
                assert body['physics'] is None and body['error'] is None
                assert any(router.REFERENCE_WARNING in w for w in body['warnings'])
        forced = client.post('/api/simulate', json=dict(scene_id=key, engine='learned', duration=.5, learned_architecture='soft_gnn'))
        assert forced.status_code == (200 if scenario == 'soft' else 422)

def test_existing_demo_scenes_are_unchanged():
    for name in ('desk', 'bottle'):
        raw = json.loads((ROOT/'scenes'/f'demo_{name}.json').read_text())
        assert 'joints' not in raw and 'soft_bodies' not in raw
        scene = Scene.model_validate(raw)
        assert scene.joints == [] and scene.soft_bodies == []
        dumped = scene.model_dump(mode='json', exclude={'joints', 'soft_bodies'})
        assert [o['id'] for o in dumped['objects']] == [o['id'] for o in raw['objects']]

def test_review_fixes_joint_notes_validation_and_wording():
    scene = load('laptop'); raw = scene.model_dump(mode='json')
    for action, word in ((Tilt(type='tilt_support', axis='x', angle_deg=20), 'level support'), (Gravity(type='set_gravity', g=30), 'gravity'),
                         (Push(type='push', id='laptop_lid', direction=(0, 0, 1), impulse_Ns=.2), 'centre-of-mass')):
        result = router.rollout(scene, [action], 4, 20)
        assert any(word in note for note in result['approximations']), (word, result['approximations'])
        assert 'Approximations: ' in result['disclaimer']
    assert router.rollout(scene, [], 4, 20)['approximations'] == []
    # schema: lower > upper, two joints on one child, cycles
    hinge = raw['joints'][0]
    for joints in ([dict(hinge, lower=2, upper=0)], [hinge, dict(hinge, id='second')],
                   [hinge, dict(hinge, id='back', parent=hinge['child'], child=hinge['parent'])]):
        with pytest.raises(ValueError): Scene.model_validate(dict(raw, joints=joints))
    # a pose that does not satisfy the joint is refused instead of teleporting the lid
    moved = Scene.model_validate(raw); lid = next(o for o in moved.objects if o.id == 'laptop_lid')
    lid.position = (lid.position[0], lid.position[1]+.2, lid.position[2])
    with pytest.raises(ValueError, match='does not satisfy its joints'): router.rollout(moved, [], 4, 20)
    assert 'only the learned' not in router.reference(scene, [], 1, 30)[1]
    soft = load('soft')
    gone = router.rollout(soft, [Remove(type='remove', id=s.id) for s in soft.soft_bodies], 4, 20)
    assert gone['routing']['soft'] is False and not any('soft_gnn pass with' in r for r in gone['routing']['reasons'])
