"""Router gaps closed for the written API spec: assistant tools, object edit on jointed/soft scenes, /api/health."""
from types import SimpleNamespace as NS
from unittest.mock import Mock
import json
from fastapi.testclient import TestClient
from agatha import main, router, world_model
from agatha.assistant import Assistant, simulation_summary
from agatha.schema import Scene
from agatha.storage import Store

class ToolMessage:
    def __init__(self, name, args):
        self.tool_calls = [] if name is None else [NS(id='call_'+name, function=NS(name=name, arguments=json.dumps(args)))]
    def model_dump(self, **kwargs):
        return dict(role='assistant', tool_calls=[dict(id=c.id, type='function', function=dict(name=c.function.name, arguments=c.function.arguments)) for c in self.tool_calls])

def ask(store, scenario, actions):
    record = store.demo(scenario)
    messages = [ToolMessage('get_scene', {}), ToolMessage('simulate', {'actions': actions}), ToolMessage('world_model_rollout', {'actions': actions}), ToolMessage(None, {})]
    client = NS(chat=NS(completions=NS(create=Mock(side_effect=[NS(choices=[NS(message=m)]) for m in messages]))))
    return record, Assistant(store).query(record['scene_id'], 'what happens?', client)

def test_assistant_routes_jointed_scene_like_simulate(tmp_path):
    record, result = ask(Store(tmp_path), 'laptop', [])
    trace = {t['tool']: t['result'] for t in result['tool_trace']}
    assert trace['get_scene']['joints'][0]['id'] == 'laptop_hinge'
    assert router.REFERENCE_WARNING in trace['simulate']['error']          # no loose-box PyBullet run is reported
    assert trace['world_model_rollout']['model_architecture'] == 'joint_gnn'
    assert [r['engine'] for r in result['simulation_results']] == ['learned']
    assert router.REFERENCE_WARNING in result['answer'] and 'Physics over 3 s' not in result['answer']
    # same numbers as /api/simulate's learned side
    scene = Scene.model_validate(record['scene'])
    assert trace['world_model_rollout']['final_state'] == router.rollout(scene, [], 90, 30)['final_state']

def test_assistant_routes_soft_scene(tmp_path):
    _, result = ask(Store(tmp_path), 'soft', [])
    trace = {t['tool']: t['result'] for t in result['tool_trace']}
    assert 'soft bodies' in trace['simulate']['error']
    assert trace['world_model_rollout']['model_architecture'] == 'contact_gnn+soft_gnn'
    assert trace['world_model_rollout']['final_state']['soft_bodies']

def test_assistant_box_scene_summary_is_unchanged(tmp_path):
    actions = [dict(type='push', id='mug', direction=[1, 0], impulse_Ns=.55)]
    record, result = ask(Store(tmp_path), 'desk', actions)
    scene = Scene.model_validate(record['scene']); parsed = main.SimulationRequest(scene_id=record['scene_id'], actions=actions).actions
    keys = {'engine', 'support_collisions_enabled', 'fell', 'tipped', 'total_displacement_m', 'chaos_score', 'final_state', 'disclaimer'}
    physics, learned = result['simulation_results']
    assert set(physics) == keys and set(learned) == keys
    assert physics == simulation_summary(main.simulate(scene, parsed))
    assert learned == simulation_summary(world_model.rollout(scene, parsed, 90, 30))

def test_object_edit_on_jointed_and_soft_scenes(tmp_path, monkeypatch):
    monkeypatch.setattr(main, 'store', Store(tmp_path)); client = TestClient(main.app)
    laptop = client.post('/api/scenes/demo?scenario=laptop').json(); key = laptop['scene_id']
    for body in (dict(size=[.3, .02, .2]), dict(shape='cylinder'), dict(mass_kg=.5, size=[.3, .02, .2])):
        response = client.post(f'/api/scenes/{key}/objects/laptop_lid', json=body)
        assert response.status_code == 422 and 'joints or soft bodies' in response.json()['detail']
    assert client.get('/api/scenes/'+key).json() == laptop                # the saved scene is untouched
    edited = client.post(f'/api/scenes/{key}/objects/laptop_lid', json=dict(mass_kg=.5)).json()
    assert edited['scene_id'] != key and edited['report']['edited_from'] == key
    before = {o['id']: o for o in laptop['scene']['objects']}
    for o in edited['scene']['objects']:
        expected = dict(before[o['id']], mass_kg=.5, mass_source='user') if o['id'] == 'laptop_lid' else before[o['id']]
        assert o == expected                                              # poses exactly preserved: no settle
    assert edited['scene']['joints'] == laptop['scene']['joints']
    run = client.post('/api/simulate', json=dict(scene_id=edited['scene_id'], engine='learned', duration=.5)).json()
    assert run['learned']['model_architecture'] == 'joint_gnn'           # still a valid jointed scene (no 422)
    soft = client.post('/api/scenes/demo?scenario=soft').json()
    assert client.post(f"/api/scenes/{soft['scene_id']}/objects/box_a", json=dict(size=[.1, .1, .1])).status_code == 422
    kept = client.post(f"/api/scenes/{soft['scene_id']}/objects/box_a", json=dict(mass_kg=1.)).json()
    assert kept['scene']['soft_bodies'] == soft['scene']['soft_bodies']
    assert client.post(f'/api/scenes/{key}/objects/nope', json=dict(size=[.1, .1, .1])).status_code == 404

def test_object_edit_on_box_scene_still_settles(tmp_path, monkeypatch):
    monkeypatch.setattr(main, 'store', Store(tmp_path)); client = TestClient(main.app)
    key = client.post('/api/scenes/demo').json()['scene_id']
    edited = client.post(f'/api/scenes/{key}/objects/mug', json=dict(mass_kg=.4, shape='cylinder', size=[.09, .1, .09]))
    assert edited.status_code == 200
    mug = next(o for o in edited.json()['scene']['objects'] if o['id'] == 'mug')
    assert mug['shape'] == 'cylinder' and mug['shape_method'] == 'user_edited' and mug['mass_source'] == 'user'
    assert 'contacts_after_settle' in edited.json()['report']['metrics']

def test_health_lists_scenarios_architectures_and_experimental_flags():
    health = TestClient(main.app).get('/api/health').json()
    assert health['status'] == 'ok' and health['objects_field'] == 'scene.objects'      # existing fields kept
    assert health['demo_scenarios'] == ['desk', 'bottle', 'laptop', 'soft']
    assert set(health['learned_architectures']) == {'default', 'auto', 'contact_gnn', 'joint_gnn', 'soft_gnn', 'mlp_multistep', 'graph_impulse'}
    assert health['experimental_features'] == dict(joints=True, soft_bodies=True, video_upload=True)
    assert health['physics_reference'] == dict(box_scenes=True, joints=False, soft_bodies=False)
    assert all((main.ROOT/path).suffix in ('.md', '.json') for path in health['docs'].values())
