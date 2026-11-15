from pathlib import Path
import numpy as np
from fastapi.testclient import TestClient
from agatha.schema import Scene,Push,Remove,SetMass
from agatha.storage import ROOT
from agatha.world_model import prepare,features,rollout,compare
from agatha.physics import simulate,World
from agatha.main import app

def demo_scene(): return Scene.model_validate_json((ROOT/'scenes/demo_desk.json').read_text())

def test_action_preparation_matches_physics():
    scene=demo_scene(); actions=[SetMass(type='set_mass',id='mug',kg=.5),Push(type='push',id='mug',direction=(1,0),impulse_Ns=.5),Remove(type='remove',id='phone')]
    prepared,states,g,tilt=prepare(scene,actions)
    expected=simulate(scene,actions,.1,20)['frames'][0]['objects']
    for s,e in zip(states,expected):
        assert np.allclose(s[:3],e['position'])
        assert np.allclose(s[7:10],e['velocity'])
    assert features(prepared,states,g,tilt).shape==(6,81)
    assert scene.objects[2].mass_kg==.32

def test_autoregressive_does_not_step_physics(monkeypatch):
    def fail(*a,**k): raise AssertionError('Learned model must not call physics step')
    monkeypatch.setattr(World,'step',fail)
    scene=demo_scene()
    result=rollout(scene,[],steps=60,fps=30)
    assert result['engine']=='learned' and len(result['frames'])==61
    assert all(np.isfinite(s['position']).all() for f in result['frames'] for s in f['objects'])
    assert np.allclose(result['final_state']['objects'][0]['position'],scene.objects[0].position)

def test_both_endpoint_and_error():
    client=TestClient(app)
    key=client.post('/api/scenes/demo').json()['scene_id']
    r=client.post('/api/simulate',json=dict(scene_id=key,actions=[dict(type='push',id='mug',direction=[1,0],impulse_Ns=.55)],engine='both',duration=2))
    assert r.status_code==200,r.text
    result=r.json()
    assert result['physics']['engine']=='physics' and result['learned']['engine']=='learned'
    assert result['error']['mean_position_error_m']>=0
    assert len(result['physics']['frames'])==len(result['learned']['frames'])
    assert client.post('/api/simulate',json=dict(scene_id=key,actions=[dict(type='remove',id='missing')])).status_code==422
