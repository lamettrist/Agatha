import json
from fastapi.testclient import TestClient
from agatha import main
from agatha.main import app

def test_metrics_returns_file_content(tmp_path,monkeypatch):
    payload={'mean_position_error_m':{'0.5':.06,'1.0':.1,'2.0':.15},'fell_agreement':.95,
             'contact_v2':{'old':{'all':{'fell_recall':None,'mean_penetration_m':.047}},'ship_decision':{'ship':True},'shipped':True}}
    path=tmp_path/'metrics.json';path.write_text(json.dumps(payload))
    monkeypatch.setattr(main,'METRICS_PATH',path)
    response=TestClient(app).get('/api/metrics')
    assert response.status_code==200 and response.json()==payload

def test_metrics_missing_is_a_clear_404(tmp_path,monkeypatch):
    monkeypatch.setattr(main,'METRICS_PATH',tmp_path/'absent.json')
    response=TestClient(app).get('/api/metrics')
    assert response.status_code==404 and 'metrics.json' in response.json()['detail']

def test_metrics_default_path_is_the_shipped_metrics_file():
    assert main.METRICS_PATH==main.ROOT/'models/metrics.json'
