"""contact_gnn behind LearnedModel / rollout / the API (CPU only, never steps PyBullet).

Uses models/contact_gnn.pt when it exists, otherwise the interim models/runs/prelim.pt;
skips cleanly when neither is present.
"""
import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import TypeAdapter
from agatha import contact_gnn
from agatha import world_model
from agatha.schema import Scene, Action
from agatha.storage import ROOT
from agatha.physics import World
from agatha.world_model import LearnedModel, rollout, prepare, state_array, load_model

SHIPPED = ROOT/'models/contact_gnn.pt'
PRELIM = ROOT/'models/runs/prelim.pt'
CHECKPOINT = SHIPPED if SHIPPED.exists() else PRELIM if PRELIM.exists() else None
needs_checkpoint = pytest.mark.skipif(CHECKPOINT is None, reason='no contact_gnn checkpoint (models/contact_gnn.pt or models/runs/prelim.pt)')
ACTIONS = TypeAdapter(list[Action])
PUSH = [dict(type='push', id='mug', direction=[1, 0], impulse_Ns=.55)]

def demo_scene() -> Scene:
    return Scene.model_validate_json((ROOT/'scenes/demo_desk.json').read_text())

@pytest.fixture(scope='module')
def projected() -> LearnedModel: return LearnedModel(CHECKPOINT, projection=True)

@pytest.fixture(scope='module')
def raw() -> LearnedModel: return LearnedModel(CHECKPOINT, projection=False)

@pytest.fixture(autouse=True)
def no_physics_step(monkeypatch):
    """Nothing in this file may advance a PyBullet world except the API's own physics engine."""
    original = World.step
    def fail(*a, **k): raise AssertionError('contact_gnn rollout must not call World.step')
    monkeypatch.setattr(World, 'step', fail)
    return original

def finite(result: dict) -> bool:
    return all(np.isfinite(s[key]).all() for f in result['frames'] for s in f['objects']
               for key in ('position', 'quaternion', 'velocity', 'angular_velocity'))

def max_penetration(scene: Scene, actions: list, result: dict) -> dict[str, float]:
    prepared, _, gravity, tilt = prepare(scene, actions)
    ids = [o.id for o in prepared.objects]; worst = dict(support=0., floor=0., objects=0.)
    for f in result['frames']:
        depths = contact_gnn.penetration_depths(*contact_gnn.scene_tensors(prepared, state_array(f, ids), gravity, tilt))
        for key, value in depths.items(): worst[key] = max(worst[key], float(value.max()))
    return worst

@needs_checkpoint
def test_rest_stays_put_without_penetration(projected):
    scene = demo_scene()
    result = rollout(scene, [], steps=90, fps=30, model=projected)
    assert result['engine'] == 'learned' and result['model_architecture'] == 'contact_gnn' and len(result['frames']) == 91
    assert result['projection_applied'] is True and 'non-learned contact projection' in result['disclaimer']
    assert result['approximations'] == [] and 'Approximations' not in result['disclaimer']
    initial = {o.id: np.asarray(o.position) for o in scene.objects}
    moved = max(float(np.linalg.norm(np.asarray(s['position'])-initial[s['id']])) for f in result['frames'] for s in f['objects'])
    assert moved < .002, moved
    assert max(max_penetration(scene, [], result).values()) < 1e-4
    assert result['events']['fell'] == [] and result['events']['tipped'] == []

@needs_checkpoint
def test_mug_push_is_finite_and_does_not_sink(projected):
    scene = demo_scene(); actions = ACTIONS.validate_python(PUSH)
    result = rollout(scene, actions, steps=90, fps=30, model=projected)
    assert finite(result)
    worst = max_penetration(scene, actions, result)
    assert worst['support'] < .003 and worst['floor'] < .003, worst
    mug = [s for s in result['frames'][-1]['objects'] if s['id'] == 'mug'][0]
    assert mug['position'][0] > scene.objects[2].position[0]+.05   # it actually moved along +X

@needs_checkpoint
def test_projection_switch(monkeypatch, raw):
    calls = []
    original = contact_gnn.project
    monkeypatch.setattr(contact_gnn, 'project', lambda *a, **k: (calls.append(1), original(*a, **k))[1])
    scene = demo_scene(); actions = ACTIONS.validate_python(PUSH)
    off = rollout(scene, actions, steps=20, fps=20, model=raw)
    assert not calls and off['projection_applied'] is False and 'no projection applied' in off['disclaimer'] and finite(off)
    follows_env = LearnedModel(CHECKPOINT)          # projection=None -> environment, default on
    monkeypatch.delenv(world_model.CONTACT_PROJECTION_ENV, raising=False)
    assert follows_env.projection_applied is True
    on = rollout(scene, actions, steps=20, fps=20, model=follows_env)
    assert len(calls) == 20 and on['projection_applied'] is True
    monkeypatch.setenv(world_model.CONTACT_PROJECTION_ENV, '0')
    calls.clear()
    disabled = rollout(scene, actions, steps=20, fps=20, model=follows_env)
    assert not calls and disabled['projection_applied'] is False and 'no projection applied' in disabled['disclaimer']
    # An explicit argument wins over the environment.
    assert LearnedModel(CHECKPOINT, projection=True).projection_applied is True
    assert disabled['frames'][-1] == off['frames'][-1]

@needs_checkpoint
@pytest.mark.parametrize('projection', [True, False])
@pytest.mark.parametrize('actions', [
    [dict(type='tilt_support', angle_deg=15, axis='z')],
    [dict(type='tilt_support', angle_deg=-15, axis='x')],
    [dict(type='set_support_collisions', enabled=False)],
    [dict(type='set_gravity', g=3), *PUSH],
    [dict(type='set_gravity', g=0)],
    [dict(type='push', id='mug', direction=[.3, 1, 0], impulse_Ns=1.2)],   # toss
    [dict(type='remove', id='laptop'), dict(type='set_mass', id='mug', kg=5), dict(type='push', id='book1', direction=[1, 0], impulse_Ns=2)],
    [dict(type='remove', id=name) for name in ('desk', 'laptop', 'mug', 'book1', 'book2', 'box', 'phone')],
], ids=['tilt_z', 'tilt_x', 'noclip', 'gravity3_push', 'gravity0', 'toss', 'remove_mass_push', 'remove_all'])
def test_actions_give_finite_results(actions, projection, projected, raw):
    scene = demo_scene(); parsed = ACTIONS.validate_python(actions)
    result = rollout(scene, parsed, steps=40, fps=20, model=projected if projection else raw)
    assert finite(result) and len(result['frames']) == 41
    assert len(result['frames'][0]['objects']) == len(scene.objects)-sum(a['type'] == 'remove' for a in actions)
    assert result['support_collisions_enabled'] is not any(a['type'] == 'set_support_collisions' for a in actions)

@needs_checkpoint
def test_noclip_drops_everything(projected):
    scene = demo_scene(); actions = ACTIONS.validate_python([dict(type='set_support_collisions', enabled=False)])
    result = rollout(scene, actions, steps=60, fps=20, model=projected)
    assert result['events']['fell'] == sorted(o.id for o in scene.objects if not o.is_static)
    assert max_penetration(scene, actions, result)['floor'] < .003

@needs_checkpoint
def test_unsupported_scene_features_are_approximated(projected):
    data = demo_scene().model_dump()
    data['support']['thickness'] = .05
    mug = data['objects'][2]
    mug.update(shape='cylinder', collision=dict(parts=[dict(type='cylinder', radius=.0425, height=.11)]), restitution=.9, linear_damping=.2)
    data['objects'][3].update(shape='compound', soft_approx=True, collision=dict(parts=[
        dict(type='box', size=(.2, .035, .14)), dict(type='convex_hull', vertices=[(0, 0, 0), (.01, 0, 0), (0, .01, 0), (0, 0, .01)])]))
    scene = Scene.model_validate(data)
    result = rollout(scene, ACTIONS.validate_python(PUSH), steps=40, fps=20, model=projected)
    assert finite(result)
    notes = ' '.join(result['approximations'])
    for expected in ('bounding box', '0.08 m thick', 'damping', 'soft', 'restitution'): assert expected in notes
    assert 'Approximations:' in result['disclaimer']

def test_old_kinds_are_unchanged():
    result = rollout(demo_scene(), [], steps=20, fps=20, model=load_model('mlp_multistep'))
    assert result['model_architecture'] == 'mlp_multistep' and 'projection_applied' not in result and 'approximations' not in result
    assert result['disclaimer'] == 'Learned mlp_multistep prediction, trained on synthetic boxes; contacts are not detected. Autoregressive error grows with time.'
    assert load_model('mlp_multistep').projection_applied is False

def test_api_contact_gnn(monkeypatch, no_physics_step):
    from agatha.main import app
    monkeypatch.setattr(World, 'step', no_physics_step)   # engine='both' legitimately runs PyBullet for the physics side
    load_model.cache_clear()
    client = TestClient(app)
    key = client.post('/api/scenes/demo').json()['scene_id']
    body = dict(scene_id=key, actions=PUSH, learned_architecture='contact_gnn', duration=2)
    both = client.post('/api/simulate', json=dict(body, engine='both'))
    only = client.post('/api/simulate', json=dict(body, engine='learned'))
    assert both.status_code == 200, both.text
    if SHIPPED.exists():
        learned = both.json()['learned']
        assert learned['model_architecture'] == 'contact_gnn' and isinstance(learned['projection_applied'], bool)
        assert both.json()['error']['mean_position_error_m'] >= 0
        assert only.status_code == 200, only.text
        assert only.json()['learned']['model_architecture'] == 'contact_gnn' and only.json()['physics'] is None
    else:
        # Documented unavailable behaviour: physics still answers, learned-only is a 503.
        assert both.json()['learned'] is None and both.json()['physics']['engine'] == 'physics'
        assert any('Learned model unavailable' in w for w in both.json()['warnings'])
        assert only.status_code == 503
        with pytest.raises(FileNotFoundError): load_model('contact_gnn')
    assert client.post('/api/simulate', json=dict(body, learned_architecture='nope')).status_code == 422
