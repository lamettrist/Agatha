from io import BytesIO
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient
from agatha import reconstruction as r
from agatha.main import app
from agatha.storage import Store

def test_ransac_plane():
    rng=np.random.default_rng(5)
    points=rng.uniform(-1,1,(1000,3)); points[:800,1]=.4+rng.normal(0,.001,800)
    plane=r.fit_plane(points)
    assert plane['inlier_ratio']>.79
    assert abs(plane['normal'][1])>.99

def test_flat_fallback_and_scale(monkeypatch):
    monkeypatch.setattr(r,'detect',lambda im:[dict(label='cell phone',bbox=[100,100,200,300],confidence=.9)])
    def fail(im): raise RuntimeError('offline')
    monkeypatch.setattr(r,'relative_depth',fail)
    s,report=r.reconstruct(Image.new('RGB',(500,500)),1.2)
    assert len(s.objects)==1
    assert report['depth']=='flat_table_fallback'
    assert report['scale_confidence']=='medium'
    assert s.objects[0].mass_source=='estimated'
    assert s.objects[0].position[1]==.75+s.objects[0].size[1]/2

def test_detector_failure_is_honest(monkeypatch):
    def fail(im): raise RuntimeError('offline')
    monkeypatch.setattr(r,'detect',fail)
    s,report=r.reconstruct(Image.new('RGB',(100,100)))
    assert not s.objects and report['detector']=='unavailable'

def test_photo_endpoint_and_disk(monkeypatch,tmp_path):
    from agatha import main
    monkeypatch.setattr(main,'store',Store(tmp_path))
    monkeypatch.setattr(main,'reconstruct',r.reconstruct)
    monkeypatch.setattr(r,'detect',lambda im:[])
    client=TestClient(app); buf=BytesIO(); Image.new('RGB',(50,50)).save(buf,format='PNG')
    response=client.post('/api/scenes',files={'image':('a.png',buf.getvalue(),'image/png')})
    assert response.status_code==200
    key=response.json()['scene_id']
    assert Store(tmp_path).get(key)['scene']==response.json()['scene']
    assert client.get('/api/scenes/'+key).status_code==200
    assert client.post('/api/scenes',files={'image':('x','invalid')}).status_code==422
    assert client.get('/api/scenes/../../etc/passwd').status_code==404
