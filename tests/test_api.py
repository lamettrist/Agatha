from fastapi.testclient import TestClient
from agatha.main import app
from agatha import main
from agatha.storage import Store

def test_full_offline_flow_and_score_verification(tmp_path,monkeypatch):
    monkeypatch.setattr(main,'store',Store(tmp_path))
    client=TestClient(app)
    record=client.post('/api/scenes/demo').json(); key=record['scene_id']
    assert client.get('/').status_code==200
    assert client.get('/static/vendor/three.module.js').status_code==200
    assert client.get('/api/scenes/'+key).json()==record
    result=client.post('/api/simulate',json=dict(scene_id=key,actions=[dict(type='push',id='mug',direction=[1,0],impulse_Ns=.55)],engine='both')).json()
    assert result['physics']['events']['fell']==['mug']
    score=result['physics']['chaos_score']
    assert client.post('/api/leaderboard',json=dict(name='Team',scene_id=key,chaos_score=score+1)).status_code==422
    assert client.post('/api/leaderboard',json=dict(name='Team',scene_id=key,chaos_score=score)).status_code==200
    assert client.get('/api/leaderboard').json()['entries'][0]['chaos_score']==score
    assert Store(tmp_path).leaderboard()[0]['name']=='Team'
    assert client.get('/api/scenes/'+key).json()==record

def test_model_unavailable_preserves_physics(monkeypatch):
    client=TestClient(app);key=client.post('/api/scenes/demo').json()['scene_id']
    def unavailable(*args,**kwargs): raise FileNotFoundError('checkpoint absent')
    monkeypatch.setattr(main,'rollout',unavailable)
    r=client.post('/api/simulate',json=dict(scene_id=key,engine='both',duration=.5)).json()
    assert r['physics'] and r['learned'] is None and r['warnings']
    assert client.post('/api/simulate',json=dict(scene_id=key,engine='learned')).status_code==503
    assert client.post('/api/simulate',json=dict(scene_id='missing',engine='physics')).status_code==404

def test_validation_and_cors():
    client=TestClient(app);key=client.post('/api/scenes/demo').json()['scene_id']
    for actions in [[dict(type='push',id='mug',direction=[0,0],impulse_Ns=.1)], [dict(type='set_mass',id='mug',kg=-1)]]:
        assert client.post('/api/simulate',json=dict(scene_id=key,actions=actions)).status_code==422
    assert client.post('/api/leaderboard',json=dict(name='x',scene_id=key,chaos_score=0)).status_code==422
    assert client.options('/api/simulate',headers={'Origin':'http://localhost:8000','Access-Control-Request-Method':'POST'}).headers['access-control-allow-origin']=='http://localhost:8000'


def test_bottle_toss_noclip_and_architecture_selection():
    client=TestClient(app);record=client.post('/api/scenes/demo?scenario=bottle',json={}).json()
    key=record['scene_id']
    for kind in ('mlp_multistep','graph_impulse'):
        response=client.post('/api/simulate',json=dict(scene_id=key,engine='both',learned_architecture=kind,duration=1,fps=20,
            actions=[dict(type='push',id='bottle',direction=[0,1,0],impulse_Ns=1.5),dict(type='set_support_collisions',enabled=False)]))
        assert response.status_code==200,response.text
        result=response.json()
        assert result['learned']['model_architecture']==kind
        assert result['physics']['support_collisions_enabled'] is False
        assert result['learned']['support_collisions_enabled'] is False
        assert result['physics']['frames'][4]['objects'][0]['position'][1]>.87
    assert client.post('/api/scenes/demo?scenario=unknown').status_code==422
    assert client.post('/api/simulate',json=dict(scene_id=key,learned_architecture='invented')).status_code==422
