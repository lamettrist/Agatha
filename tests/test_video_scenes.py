from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import pytest
from fastapi.testclient import TestClient

from agatha.schema import Scene,SceneObject,Support
from agatha.scene_reconstruction.geometry import backproject
from agatha.scene_reconstruction.pipeline import assign_video_supports
from agatha.scene_reconstruction.video import crop_padding,fuse_track_points,sample_frames


def test_padding_crop_preserves_camera_rays():
    images=[Image.new('RGB',(518,518))]
    k=[[443.,0,259.],[0,443.,259.],[0,0,1]]
    camera=dict(fx=443.,fy=443.,cx=259.,cy=259.)
    depth=np.ones((1,518,518),np.float32)
    cloud=backproject(depth[0],camera)[None]
    cameras=[dict(intrinsics=k,center=[0,0,0],camera_to_world=np.eye(3).tolist())]
    images,depths,points,confidence,adjusted=crop_padding(images,depth,cloud,depth,cameras,(1920,1080))
    assert images[0].size==(518,294)
    assert depths.shape==(1,294,518)
    assert adjusted[0]['intrinsics'][1][2]==147
    camera['cy']=147.
    assert np.allclose(backproject(depths[0],camera),points[0])
    assert cameras[0]['intrinsics'][1][2]==259.


def test_secondary_table_support_and_floor_furniture():
    table=SceneObject(id='other_table',label='table',size=(1,.03,.8),position=(2,1.185,0),is_static=True)
    bottle=SceneObject(id='bottle',label='bottle',size=(.07,.21,.07),position=(2,1.305,0),supported_by='support')
    chair=SceneObject(id='chair',label='chair',size=(.4,.8,.4),position=(0,.4,0),supported_by='floor')
    assigned=assign_video_supports([table,bottle,chair],Support())
    assert assigned['bottle']=='other_table'
    assert chair.supported_by=='floor'


def test_fused_track_rejects_jumps_to_other_furniture():
    rng=np.random.default_rng(4)
    reference=rng.uniform(-.1,.1,(100,3))
    next_view=np.concatenate([reference+.005,reference+[2,0,0]])
    points=fuse_track_points([reference,next_view])
    assert len(points)>=190
    assert points[:,0].max()<.2


def test_fused_surface_is_converted_from_world_to_body_coordinates(monkeypatch,tmp_path):
    import trimesh
    from agatha.scene_reconstruction import meshes
    from agatha.scene_reconstruction.assets import rotation
    from agatha.schema import Collision,CollisionPart
    obj=SceneObject(id='bag',label='backpack',size=(.2,.3,.1),position=(1,.9,.2),rotation_euler=(0,.6,0))
    expected=trimesh.creation.box(extents=obj.size)
    world=expected.copy();world.vertices=world.vertices@rotation(obj.rotation_euler).T+obj.position
    monkeypatch.setattr(meshes,'point_mesh',lambda points:world.copy())
    monkeypatch.setattr(meshes,'convex_parts',lambda mesh:Collision(parts=[CollisionPart(size=obj.size)]))
    result=meshes.job(0,None,None,obj,None,tmp_path,fused=np.zeros((101,3)))
    assert result['status']=='ok'
    local=trimesh.load_mesh(result['surface'])
    assert np.allclose(local.vertices,expected.vertices,atol=1e-6)


def test_parallel_track_groups_keep_global_object_ids(monkeypatch):
    from types import SimpleNamespace
    from agatha.scene_reconstruction import video
    monkeypatch.setattr(video.torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(video.torch.cuda,'empty_cache',lambda:None)
    monkeypatch.setattr(video,'registry',lambda:SimpleNamespace(devices=[0,1,2],device=lambda i:f'cuda:{i}'))
    seen=[]
    def track(images,instances,clouds,confidence,device):
        ids=[inst['track_id'] for inst in instances];seen.extend(ids)
        return {i:np.array([[i,0,1]]) for i in ids},[dict(frame=0,track_id=i) for i in ids]
    monkeypatch.setattr(video,'track_group',track)
    fused,records=video.track_instances([], [{} for _ in range(7)],None,None)
    assert sorted(seen)==list(range(7))
    assert sorted(fused)==list(range(7))
    assert [row['track_id'] for row in records]==list(range(7))


def test_tracker_keeps_ids_when_processor_consumes_pending_inputs(monkeypatch):
    from types import SimpleNamespace
    import torch
    from agatha.scene_reconstruction import video
    class Processor:
        def init_video_session(self,**kwargs):return SimpleNamespace()
        def add_inputs_to_inference_session(self,session,**kwargs):
            session.pending=kwargs['obj_ids'];session.count=len(session.pending)
        def post_process_masks(self,masks,sizes):return [masks[0].bool()]
    class Model:
        device='cpu'
        def __call__(self,inference_session,frame_idx):inference_session.pending.clear()
        def propagate_in_video_iterator(self,session,**kwargs):
            yield SimpleNamespace(frame_idx=0,pred_masks=torch.ones((session.count,1,4,4)))
    monkeypatch.setattr(video,'tracker_model',lambda device:(Processor(),Model()))
    instances=[dict(track_id=i,mask=np.ones((4,4),bool)) for i in (2,9)]
    fused,observations=video.track_group([Image.new('RGB',(4,4))],instances,np.ones((1,4,4,3)),np.ones((1,4,4)),'cpu')
    assert [row['track_id'] for row in observations]==[2,9]
    assert all(len(points)>0 for points in fused.values())


def test_video_settling_does_not_enlarge_a_measured_table():
    from agatha.scene_reconstruction.bodies import settle
    scene=Scene(objects=[SceneObject(id='outside',label='book',size=(.1,.1,.1),position=(2,.8,0),supported_by='support')])
    settled,metrics=settle(scene,seconds=0,extend_support=False)
    assert settled.support.size==scene.support.size
    assert metrics['inferred_support_extension_m']==[0.,0.]
    assert metrics['floating_ids']==['outside']


def test_sequential_sampling_keeps_sharp_frames_in_time_order(tmp_path):
    path=tmp_path/'frames.avi'
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),10,(160,120))
    if not writer.isOpened():pytest.skip('MJPG codec unavailable')
    random=np.random.default_rng(8)
    for index in range(20):
        frame=random.integers(0,256,(120,160,3),dtype=np.uint8) if index%2 else np.full((120,160,3),128,np.uint8)
        writer.write(frame)
    writer.release()
    paths,report=sample_frames(path,tmp_path/'sampled',count=8)
    accepted=[row for row in report['candidates'] if row['accepted']]
    assert len(paths)==8
    assert all(row['laplacian_variance']>=50 for row in accepted)
    assert [row['frame'] for row in accepted]==sorted(row['frame'] for row in accepted)


@pytest.mark.parametrize('field,filename,mime',[
    ('image','deskspace.mp4','video/mp4'),
    ('file','deskspace.mp4','application/octet-stream'),
    ('video','clip.mov','video/quicktime'),
])
@pytest.mark.parametrize('endpoint',['/api/scenes','/api/scenes/video'])
def test_generic_scene_route_dispatches_video(monkeypatch,tmp_path,field,filename,mime,endpoint):
    from agatha import main
    from agatha.storage import Store
    from agatha.scene_reconstruction import video
    monkeypatch.setattr(main,'store',Store(tmp_path/'records'))
    observed=[]
    scene=Scene(objects=[SceneObject(id='bottle',label='bottle',size=(.07,.21,.07),position=(0,.855,0))])
    def reconstruct(path,width):
        observed.append((Path(path).read_bytes(),width))
        return scene,{'pipeline':'test_video_dispatch','warnings':[]}
    monkeypatch.setattr(video,'reconstruct_video',reconstruct)
    client=TestClient(main.app)
    response=client.post(endpoint,files={field:(filename,b'video-payload',mime)},data={'table_width_m':'1.2'})
    assert response.status_code==200,response.text
    assert observed==[(b'video-payload',1.2)]
    record=response.json()
    assert record['scene']['objects'][0]['id']=='bottle'
    assert client.get('/api/scenes/'+record['scene_id']).json()==record


def test_empty_video_is_a_reported_error(monkeypatch):
    from agatha import main
    from agatha.scene_reconstruction import video
    monkeypatch.setattr(video,'reconstruct_video',lambda *args:(Scene(objects=[]),{'detector':'unavailable','warnings':['Missing model weights']}))
    response=TestClient(main.app).post('/api/scenes/video',files={'file':('clip.mp4',b'video','video/mp4')})
    assert response.status_code==503
    assert response.json()['detail']['report']['warnings']==['Missing model weights']


def test_precognition_cors_and_health():
    from agatha.main import app
    client=TestClient(app)
    response=client.options('/api/scenes/video',headers={'Origin':'https://precognition.base44.app','Access-Control-Request-Method':'POST'})
    assert response.status_code==200
    assert response.headers['access-control-allow-origin']=='https://precognition.base44.app'
    health=client.get('/api/health').json()
    assert health['objects_field']=='scene.objects'
    assert health['video_endpoint']=='/api/scenes/video'
