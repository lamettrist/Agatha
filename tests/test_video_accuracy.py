import os
import numpy as np
import pytest

from agatha.schema import Support,SceneObject
from agatha.scene_reconstruction.segmentation import deduplicate,split_depth_clusters,uncovered_automatic_masks
from agatha.scene_reconstruction.structure import table_surfaces


def instance(label,mask,confidence=.9,source='sam2_box'):
    yy,xx=np.where(mask)
    return dict(label=label,mask=mask,confidence=confidence,mask_source=source,
                bbox=[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)])


def test_chair_back_and_seat_remain_one_tracked_body():
    mask=np.ones((30,30),bool)
    depth=np.where(np.indices(mask.shape)[0]<15,1.,1.8)
    kept=split_depth_clusters([instance('chair',mask)],depth)
    assert len(kept)==1
    assert np.array_equal(kept[0]['mask'],mask)


def test_nested_chair_part_is_duplicate_even_at_different_depth():
    whole=np.ones((30,30),bool);part=np.zeros_like(whole);part[:10]=True
    depth=np.where(part,1.,2.)
    kept,dropped=deduplicate([instance('chair',whole),instance('chair',part,.6)],depth)
    assert len(kept)==1
    assert dropped[0]['reason']=='duplicate_or_part_mask'


def test_automatic_proposals_only_classify_unexplained_connected_pixels():
    furniture=np.zeros((30,30),bool);furniture[:,:10]=True
    wall=np.zeros_like(furniture);wall[:10,10:]=True
    proposal=np.ones_like(furniture)
    kept=uncovered_automatic_masks([instance('chair',furniture),instance('wall',wall),
                                  instance('unknown object',proposal,source='sam2_automatic')])
    auto=kept[-1]
    assert auto['bbox']==[10.,10.,30.,30.]
    assert not (auto['mask']&(furniture|wall)).any()
    assert auto['mask'].sum()==400


def test_disjoint_coplanar_tables_do_not_create_bridge_slab():
    x,z=np.meshgrid(np.linspace(-.4,.4,20),np.linspace(-.3,.3,20))
    left=np.stack([x,np.ones_like(x)*.8,z],-1)
    right=left+[2,0,0]
    cloud=np.concatenate([left,right],axis=1)
    mask=np.ones(cloud.shape[:2],bool)
    tables=table_surfaces([dict(mask=mask)],cloud,mask,Support(size=(1,.8),height=.8))
    assert len(tables)==1
    assert np.allclose(tables[0]['position'],[2,.785,0],atol=.02)
    assert max(tables[0]['size'])<1


def test_silhouette_refinement_uses_each_bodys_support_height(monkeypatch):
    from agatha.scene_reconstruction import refinement
    table=SceneObject(id='rear_table',label='table',is_static=True,
                      size=(1,.03,.8),position=(2,1.185,0))
    bottle=SceneObject(id='bottle',label='bottle',supported_by=table.id,
                       size=(.07,.21,.07),position=(2,1.305,0))
    heights=[]
    def fit(*args):
        heights.append(args[-1]);return {'id':args[0].id}
    monkeypatch.setattr(refinement,'fit_one',fit)
    report=refinement.refine([table,bottle],[{'object_id':'bottle'}],None,None,.75)
    assert np.allclose(heights,[1.2])
    assert np.isclose(report[0]['support_height_m'],1.2)


# ---- 2026-11-15 regression tests: multi-view consensus, oriented support, filtering, supports, settle ----

def box_points(rng,center,size,count=600):
    return rng.uniform(-.5,.5,(count,3))*size+center


def test_view_consensus_rejects_track_drift_and_ghost_points():
    from agatha.scene_reconstruction.structure import fuse_views
    rng=np.random.default_rng(0);stool=([0,.55,0],[.5,1.1,.5])
    views=[(f,box_points(rng,*stool)) for f in range(6)]
    views+=[(6,box_points(rng,[.9,.5,0],[.5,1.,.5])),          # SAM 2 jumped to the neighbouring stool
            (7,np.concatenate([box_points(rng,*stool),box_points(rng,[0,.02,.9],[1.5,.04,1.5])]))]  # bled onto the floor
    views[2]=(2,np.concatenate([views[2][1],box_points(rng,[.33,.55,0],[.1,1.1,.1],80)]))            # single-view ghost
    points,stats=fuse_views(views)
    assert stats['views_observed']==8 and stats['kept_frames']==[0,1,2,3,4,5] and stats['voted']
    lo,hi=np.percentile(points,[2,98],axis=0)
    assert np.all(hi-lo<[.56,1.15,.56]) and points[:,0].max()<.30


def test_view_consensus_reports_objects_without_consistent_views():
    from agatha.scene_reconstruction.structure import fuse_views
    rng=np.random.default_rng(1)
    views=[(f,box_points(rng,[f*1.0,.5,0],[.2,.2,.2],100)) for f in range(5)]
    points,stats=fuse_views(views)
    assert stats['views_kept']<=1


def test_table_rectangle_recovers_an_oblique_table_from_partial_views():
    from agatha.scene_reconstruction.structure import table_rectangle
    rng=np.random.default_rng(2);yaw=np.deg2rad(-58);axis=np.array([np.cos(yaw),np.sin(yaw)]);side=np.array([-axis[1],axis[0]])
    def top(count,along=(-.85,.85)):
        a=rng.uniform(*along,count);b=rng.uniform(-.475,.475,count)
        return np.column_stack([a[:,None]*axis+b[:,None]*side])
    seed=top(4000,(-.85,.35))                                   # the reference frame misses the far end
    points=[];frames=[]
    for frame in range(12):
        xz=top(3000,(-.85,.85) if frame>3 else (-.85,.35))
        points.append(np.column_stack([xz[:,0],rng.normal(0,.004,len(xz)),xz[:,1]]));frames.append(np.full(len(xz),frame))
    floor=rng.uniform(-3,3,(20000,2));points.append(np.column_stack([floor[:,0],np.full(len(floor),-1.),floor[:,1]]));frames.append(np.zeros(len(floor),int))
    rect=table_rectangle(np.concatenate(points),np.concatenate(frames),seed)
    assert rect['source']=='multi_frame_min_area_rectangle'
    assert abs(rect['size'][0]-1.7)<.12 and abs(rect['size'][1]-.95)<.10
    assert abs(abs(rect['axis']@axis)-1)<.01 and np.linalg.norm(rect['center'])<.06
    # An axis-aligned box of the same points would be far from the real 1.7 x 0.95 m.
    lo,hi=np.percentile(np.concatenate(points[:-1])[:,[0,2]],[1,99],axis=0)
    assert min(hi-lo)>1.2
    alone=table_rectangle(np.empty((0,3)),np.empty(0,int),seed)
    assert alone['source']=='reference_frame_min_area_rectangle' and abs(alone['size'][0]-1.2)<.1


def test_reference_selection_drops_automatic_masks_and_confidence_floor_clutter():
    from agatha.scene_reconstruction.video import select_trackable
    confidence=np.ones((20,20));confidence[:10]=3.
    near=np.zeros((20,20),bool);near[2:8,2:8]=True;far=np.zeros((20,20),bool);far[12:18,2:8]=True
    kept,dropped=select_trackable([instance('bottle',near),instance('laptop',far),instance('monitor',near,source='sam2_automatic')],confidence)
    assert [i['label'] for i in kept]==['bottle']
    assert {d['label']:d['reason'] for d in dropped}=={'laptop':'low_geometry_confidence','monitor':'automatic_mask_not_tracked'}
    # Geometry without a calibrated confidence (COLMAP fallback) must not lose every detection.
    kept,_=select_trackable([instance('laptop',far)],confidence,'COLMAP_poses_with_metric_monocular_depth')
    assert len(kept)==1


def test_bodies_with_nothing_under_them_are_not_assigned_to_the_main_table():
    from agatha.scene_reconstruction.pipeline import assign_video_supports,UNSUPPORTED
    support=Support(size=(1.6,.9),height=1.02)
    on=SceneObject(id='on',label='bottle',size=(.07,.2,.07),position=(.2,1.12,0),supported_by='support')
    far=SceneObject(id='far',label='laptop',size=(.3,.02,.2),position=(3,.9,0),supported_by='support')
    low=SceneObject(id='low',label='can',size=(.06,.12,.06),position=(1.5,.07,0),supported_by='support')
    under=SceneObject(id='under',label='backpack',size=(.3,.4,.2),position=(0,.2,0),supported_by='support')
    assigned=assign_video_supports([on,far,low,under],support)
    assert assigned=={'on':'support','far':UNSUPPORTED,'low':'floor','under':'floor'}


def test_multiview_lift_gives_class_plausible_upright_bodies_or_rejects():
    from agatha.scene_reconstruction.pipeline import lift_multiview,Implausible
    rng=np.random.default_rng(3)
    theta=rng.uniform(0,2*np.pi,3000);bottle=np.column_stack([.03*np.cos(theta)+.4,rng.uniform(1.02,1.26,3000),.03*np.sin(theta)-.1])
    bottle=np.concatenate([bottle,bottle[:60]+[.05,0,.05]])   # residual ghost must not double the diameter
    obj,details=lift_multiview(dict(label='bottle',confidence=.8),bottle,0,1.02)
    assert obj.shape=='cylinder' and .05<=obj.size[0]<=.08 and abs(obj.size[1]-.24)<.02
    assert abs(obj.position[1]-obj.size[1]/2-1.02)<1e-6 and obj.rotation_euler==(0,0,0) and obj.supported_by=='support'
    stool=box_points(rng,[1,.57,1],[.52,1.14,.5],5000)
    chair,_=lift_multiview(dict(label='chair',confidence=.5),stool,1,1.02)
    assert chair.supported_by=='floor' and chair.rotation_euler[0]==0 and chair.rotation_euler[2]==0
    assert max(chair.size[0],chair.size[2])<=.70 and .6<=chair.size[1]<=1.3 and abs(chair.position[1]-chair.size[1]/2)<1e-6
    for part in chair.collision.parts:               # no part may leave the measured envelope
        half=np.array(part.size)/2 if part.type=='box' else np.array([part.radius,part.height/2,part.radius])
        assert np.all(np.abs(part.position)+half<=np.array(chair.size)/2+1e-6)
    with pytest.raises(Implausible):lift_multiview(dict(label='chair',confidence=.5),box_points(rng,[0,.4,0],[2.2,.75,.7],4000),2,1.02)
    with pytest.raises(Implausible):lift_multiview(dict(label='backpack',confidence=.5),box_points(rng,[0,1.5,0],[1.47,.49,.18],400),3,1.02)
    with pytest.raises(Implausible):lift_multiview(dict(label='keyboard',confidence=.3),box_points(rng,[0,.02,0],[.71,.03,.2],400),4,1.02)


def test_gpu_list_defaults_to_visible_devices_and_ignores_missing_ids(monkeypatch):
    from agatha.scene_reconstruction import models,video
    from types import SimpleNamespace
    monkeypatch.setattr(models.torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(models.torch.cuda,'device_count',lambda:1)
    monkeypatch.delenv('RECON_GPU_IDS',raising=False)
    assert models.gpu_devices()==[0]
    monkeypatch.setenv('RECON_GPU_IDS','0,1,2,3,4,5,6')
    assert models.gpu_devices()==[0]
    monkeypatch.setattr(models.torch.cuda,'device_count',lambda:4)
    monkeypatch.setenv('RECON_GPU_IDS','1,3')
    assert models.gpu_devices()==[1,3]
    monkeypatch.setattr(video,'registry',lambda:SimpleNamespace(devices=[1,3]))
    monkeypatch.delenv('RECON_TRACK_WORKERS',raising=False)
    assert video.track_workers(9)==2
    monkeypatch.setenv('RECON_TRACK_WORKERS','1')
    assert video.track_workers(9)==1
    monkeypatch.setattr(video,'registry',lambda:SimpleNamespace(devices=[0]))
    monkeypatch.setenv('RECON_TRACK_WORKERS','7')
    assert video.track_workers(9)==1


def test_vggt_out_of_memory_retries_with_smaller_chunks(monkeypatch):
    from agatha.scene_reconstruction import video
    calls=[]
    def geometry(paths,work,chunk=None):
        calls.append(chunk)
        if chunk>6:raise video.torch.OutOfMemoryError('CUDA out of memory')
        return ('images','depths','clouds','confidence','cameras',[])
    monkeypatch.setattr(video,'vggt_geometry',geometry);monkeypatch.setattr(video.torch.cuda,'empty_cache',lambda:None)
    monkeypatch.delenv('RECON_VIDEO_CHUNK',raising=False)
    report=dict(warnings=[])
    assert video.vggt_with_memory_retry([],None,report)[-1]==6 and calls==[10,8,6]
    assert 'out of GPU memory' in report['warnings'][0]


def test_measured_settle_slides_a_stool_out_of_the_table_and_keeps_bottles():
    from agatha.schema import Scene
    from agatha.scene_reconstruction.bodies import settle,stability
    from agatha.scene_reconstruction.pipeline import chair_collision
    stool=SceneObject(id='stool',label='chair',shape='compound',collision=chair_collision(.5,1.15,.5,.8),size=(.5,1.15,.5),
        position=(.3,.575,.50),supported_by='floor',mass_kg=6,friction=.8)        # back rest inside the slab edge
    bottle=SceneObject(id='bottle',label='bottle',shape='cylinder',size=(.065,.23,.065),position=(.3,1.2,.30),supported_by='support',mass_kg=.55,friction=.8)
    scene=Scene(objects=[stool,bottle],support=Support(size=(1.6,.9),height=1.02,thickness=.03))
    settled,metrics=settle(scene,extend_support=False,measured_poses=True)
    assert metrics['float_count']==0 and metrics['penetration_count']==0
    assert 'stool' in metrics['static_overlap_corrections_m']
    moved={o.id:np.linalg.norm(np.array(o.position)[[0,2]]-np.array(old.position)[[0,2]]) for o,old in zip(settled.objects,scene.objects)}
    assert moved['bottle']<.005 and .01<moved['stool']<.4
    assert stability(settled,seconds=2)['max_drift_m']<.005


def test_floor_furniture_is_not_reported_as_fallen():
    from agatha.schema import Scene,Push
    from agatha.physics import simulate
    stool=SceneObject(id='stool',label='chair',size=(.4,.8,.4),position=(1.5,.4,0),supported_by='floor',mass_kg=6)
    bottle=SceneObject(id='bottle',label='bottle',shape='cylinder',size=(.065,.23,.065),position=(0,.865,.30),supported_by='support',mass_kg=.55,friction=.8)
    scene=Scene(objects=[stool,bottle])
    assert simulate(scene,[],1,10)['events']['fell']==[]
    assert simulate(scene,[Push(type='push',id='bottle',direction=(0,0,1),impulse_Ns=1.5)],3,10)['events']['fell']==['bottle']


def test_table_rectangle_is_not_inflated_by_pose_drift_between_views():
    """Views that disagree by centimetres must not union into an oversized slab."""
    from agatha.scene_reconstruction.structure import table_rectangle
    rng=np.random.default_rng(5);yaw=np.deg2rad(30);axis=np.array([np.cos(yaw),np.sin(yaw)]);side=np.array([-axis[1],axis[0]])
    def top(count,shift=(0,0)):
        a=rng.uniform(-.75,.75,count);b=rng.uniform(-.40,.40,count)
        return a[:,None]*axis+b[:,None]*side+np.asarray(shift)
    seed=top(4000);points=[];frames=[]
    for frame in range(16):
        # Four of sixteen views are displaced by 8-10 cm (chunk seams); the rest by a centimetre or two.
        shift=rng.normal(0,.015,2) if frame%4 else .09*np.array([np.cos(frame),np.sin(frame)])
        xz=top(3000,shift);points.append(np.column_stack([xz[:,0],rng.normal(0,.003,len(xz)),xz[:,1]]));frames.append(np.full(len(xz),frame))
    rect=table_rectangle(np.concatenate(points),np.concatenate(frames),seed)
    assert rect['source']=='multi_frame_min_area_rectangle'
    assert abs(rect['size'][0]-1.5)<.06 and abs(rect['size'][1]-.8)<.06,rect['size']
    assert np.linalg.norm(rect['center'])<.04 and abs(abs(rect['axis']@axis)-1)<.01
    # The union of all views, which the trim replaces, overhangs the real top.
    assert rect['untrimmed_size'][0]>rect['size'][0]+.04 and rect['untrimmed_size'][1]>rect['size'][1]+.04,(rect['untrimmed_size'],rect['size'])


def _look_at(eye,target):
    """Camera pose (pipeline convention: x right, y down, z forward) looking at target, world Y up."""
    forward=np.asarray(target,float)-eye;forward/=np.linalg.norm(forward)
    right=np.cross(forward,[0,1,0]);right/=np.linalg.norm(right);down=np.cross(forward,right)
    return dict(origin_world=list(map(float,eye)),world_to_camera=np.array([right,down,forward]).tolist())


def test_stool_yaw_is_chosen_over_several_views_not_one_occluded_view():
    import cv2
    from agatha.scene_reconstruction.pipeline import orient_chairs,chair_collision,frame_view,consistent_frames
    from agatha.scene_reconstruction.refinement import vertices
    from agatha.scene_reconstruction.assets import rotation,camera_points,project
    camera=dict(fx=300.,fy=300.,cx=180.,cy=120.,width=360,height=240);truth=np.deg2rad(100);position=np.array([.2,.575,.9])
    def silhouette(pose,yaw):
        mask=np.zeros((240,360),np.uint8)
        for part in chair_collision(.5,1.15,.5,.8).parts:
            cam=camera_points((vertices(part)@rotation(part.rotation_euler).T+part.position)@rotation([0,yaw,0]).T+position,pose)
            cv2.fillConvexPoly(mask,cv2.convexHull(np.int32(project(cam,camera))),1)
        return mask>0
    poses=[_look_at(np.array([2.6*np.cos(a)+.2,1.5,2.6*np.sin(a)+.9]),position) for a in np.linspace(0,2.4,5)]
    masks=[silhouette(pose,truth) for pose in poses]
    occluded=masks[0].copy();occluded[:150]=False      # the reference view sees only the base through table legs
    def stool():
        return SceneObject(id='object_000',label='chair',shape='compound',collision=chair_collision(.5,1.15,.5,.8),size=(.5,1.15,.5),
            position=tuple(position),rotation_euler=(0,truth+np.pi/2,0),supported_by='floor',shape_method='multiview_stool_compound')
    inst=dict(object_id='object_000',mask=occluded)
    several=stool();report=orient_chairs([several],[inst],camera,poses[0],{'object_000':[(camera,poses[0],occluded,None),*[(camera,p,m,None) for p,m in zip(poses[1:],masks[1:])]]})
    error=abs((several.rotation_euler[1]-truth+np.pi)%(2*np.pi)-np.pi)
    assert error<.01 and report[0]['frames_scored']==5 and report[0]['silhouette_iou_proxy']>.8
    assert np.hypot(*report[0]['shift_m'])<1e-9         # a correctly placed stool is not shifted
    # Frame poses given in first-camera coordinates map to the same world poses.
    first=poses[0];w2c=np.array(first['world_to_camera']);other=poses[2]
    entry=dict(intrinsics=[[300,0,180],[0,300,120],[0,0,1]],camera_to_world=(w2c@np.array(other['world_to_camera']).T).tolist(),
               center=((np.array(other['origin_world'])-first['origin_world'])@w2c.T/2.).tolist())
    _,pose=frame_view(first,2.,entry,camera)
    assert np.allclose(pose['origin_world'],other['origin_world']) and np.allclose(pose['world_to_camera'],other['world_to_camera'])
    # Track frames that jumped to a neighbour 0.6 m away are not used for scoring.
    rng=np.random.default_rng(0);on=rng.normal(position,.05,(200,3));off=on+[.6,0,0]
    assert consistent_frames([(0,on),(1,off),(2,on)],position)==[0,2]


def test_fixed_camera_video_is_recognised_and_reconstructed_as_one_photo(monkeypatch,tmp_path):
    from agatha.schema import Scene
    from agatha.scene_reconstruction import video
    from PIL import Image
    floor=np.ones((6,20,30));walk=floor.copy();walk[:,:,:12]=3.
    assert not video.geometry_consistency(floor)['usable'] and video.geometry_consistency(walk)['usable']
    assert video.geometry_consistency(floor,'COLMAP_poses_with_metric_monocular_depth')['usable']   # no confidence channel there
    scene=Scene(objects=[SceneObject(id='object_000',label='bottle',size=(.07,.2,.07),position=(.1,.85,.1)),
                         SceneObject(id='object_001',label='person',size=(.4,1.7,.3),position=(0,-4.8,-11.)),
                         SceneObject(id='object_002',label='chair',size=(.5,1.,.5),position=(6.,.5,0))])
    monkeypatch.setattr(video,'reconstruct',lambda image,width:(scene.model_copy(deep=True),dict(run_id='a'*64,warnings=['w'],pipeline='metric')))
    path=tmp_path/'frame.jpg';Image.new('RGB',(32,24)).save(path)
    result,report=video.photo_fallback([path],None,dict(warnings=['v'],stages={}),0.,'fixed camera',tmp_path)
    assert [o.id for o in result.objects]==['object_000']
    assert {r['reason'] for r in report['video_fallback']['removed']}=={'below_floor','far_from_support'}
    assert report['pipeline']==video.PHOTO_FALLBACK and report['warnings']==['w','v','fixed camera'] and report['quality_status']=='needs_review'
    assert (tmp_path/'scene.json').exists() and (tmp_path/'reconstruction_report.json').exists()


def test_video_without_a_gpu_fails_immediately(monkeypatch,tmp_path):
    from agatha.scene_reconstruction import video
    monkeypatch.setattr(video.torch.cuda,'is_available',lambda:False)
    monkeypatch.delenv('RECON_ALLOW_CPU',raising=False);monkeypatch.delenv('VISION_OFFLINE',raising=False)
    with pytest.raises(RuntimeError,match='needs a CUDA GPU'):video.reconstruct_video(tmp_path/'missing.mp4')


def test_degraded_or_reconfigured_video_runs_are_not_served_from_cache(monkeypatch,tmp_path):
    """A run that fell back to smaller VGGT chunks, or was made with other settings, must be redone."""
    import json
    from hashlib import sha256
    from agatha.schema import Scene
    from agatha.scene_reconstruction import video,pipeline
    monkeypatch.setenv('RECON_RUN_DIR',str(tmp_path));monkeypatch.setenv('RECON_ALLOW_CPU','1')
    for name in ('RECON_VIDEO_CHUNK','RECON_VIDEO_STATIC_FURNITURE'):monkeypatch.delenv(name,raising=False)
    source=tmp_path/'clip.mp4';source.write_bytes(b'not a real video')
    class Reached(Exception):pass
    def sampling(*args,**kwargs):raise Reached()
    monkeypatch.setattr(video,'sample_frames',sampling)
    def cached_run(chunk):
        for directory in tmp_path.iterdir():
            if directory.is_dir():
                import shutil;shutil.rmtree(directory)
        with pytest.raises(Reached):video.reconstruct_video(source)          # creates the run directory for this configuration
        work=next(d for d in tmp_path.iterdir() if d.is_dir())
        (work/'scene.json').write_text(Scene(objects=[SceneObject(id='object_000',label='bottle',size=(.07,.2,.07),position=(0,.85,0))]).model_dump_json())
        (work/'reconstruction_report.json').write_text(json.dumps(dict(status='complete',video=dict(stages=dict(
            geometry=dict(method='VGGT-1B',chunk=chunk),tracking=dict(status='ok',observations=5,fused_point_counts={'0':40}))))))
        return work
    cached_run(10)
    assert video.reconstruct_video(source)[1]['cache_hit'] is True
    cached_run(4)
    with pytest.raises(Reached):video.reconstruct_video(source)
    work=cached_run(10);monkeypatch.setenv('RECON_VIDEO_STATIC_FURNITURE','0')
    with pytest.raises(Reached):video.reconstruct_video(source)
    assert len([d for d in tmp_path.iterdir() if d.is_dir()])==2             # a different run directory


def test_photo_report_states_quality_and_warns_without_a_table():
    """Source-level guard: the photo path publishes the same quality fields as the video path."""
    import inspect
    from agatha.scene_reconstruction import pipeline
    source=inspect.getsource(pipeline.reconstruct)
    assert "report['quality_status']" in source and 'No table or desk was detected' in source


@pytest.mark.skipif(not os.getenv('RECON_REAL_RUN'),reason='set RECON_REAL_RUN to a run captured by scripts/recon_refit.py from tests/images/deskspace.mp4')
def test_real_video_refit_acceptance(tmp_path,monkeypatch):
    """Opt-in: deterministic refit of the cached neural outputs must give the demo scene."""
    import time
    from agatha.scene_reconstruction import video
    from agatha.physics import simulate
    from agatha.schema import Push
    monkeypatch.setenv('RECON_RUN_DIR',str(tmp_path/'lift'));monkeypatch.setenv('RECON_MESH','0')
    bundle=video.load_neural_bundle(os.environ['RECON_REAL_RUN'])
    fused,tracking=video.refit_tracks(bundle['instances'],bundle['track_masks'],bundle['clouds'],bundle['confidence'])
    report=bundle['video_report'];video.record_tracking(report,fused,tracking,len(bundle['instances']))
    scene,report=video.finalize_video(tmp_path,bundle['digest'],bundle['images'],bundle['depths'],bundle['cameras'],bundle['detections'],
        bundle['masks'],bundle['instances'],fused,report,None,time.perf_counter(),fusion=False,
        track_masks=bundle['track_masks'],clouds=bundle['clouds'],confidence=bundle['confidence'])
    support=scene.support;dynamic=[o for o in scene.objects if not o.is_static or o.label=='chair']
    # The real top is about 1.5 x 0.8 m (independent wood-colour fit over 30 frames); the union of drifting views gave 1.65 x 0.90.
    assert .95<=support.height<=1.10 and 1.42<=max(support.size)<=1.62 and .72<=min(support.size)<=.87
    bottles=[o for o in dynamic if o.label=='bottle'];stools=[o for o in dynamic if o.label=='chair']
    assert len(bottles)==2 and len(stools)>=2 and len(dynamic)<=8
    for b in bottles:
        assert b.supported_by=='support' and .05<=b.size[0]<=.10 and .12<=b.size[1]<=.30
        assert abs(b.position[1]-b.size[1]/2-support.height)<.005
    assert np.hypot(bottles[0].position[0]-bottles[1].position[0],bottles[0].position[2]-bottles[1].position[2])>.2
    for c in stools:
        assert c.supported_by=='floor' and max(c.size[0],c.size[2])<=.70 and .7<=c.size[1]<=1.3 and abs(c.position[1]-c.size[1]/2)<.005
    assert max(max(o.size) for o in dynamic)<1.35
    contacts=report['metrics']['contacts_after_settle']
    assert contacts['float_count']==0 and contacts['penetration_count']==0
    assert report['metrics']['stability']['max_drift_m']<.005
    silhouettes=report['metrics']['multiview_silhouette']['per_object']
    assert all(silhouettes[b.id]['mean_iou']>=.45 for b in bottles)
    assert all(c.is_static for c in stools)   # published static after being drift-tested as dynamic
    assert any('fixed in place' in warning for warning in report['warnings'])
    orientation={c['id']:c for c in report['stages']['lift']['chair_orientation']}
    assert all(orientation[c.id]['frames_scored']>=3 for c in stools)       # yaw is chosen over several views
    assert report['video']['stages']['reference_selection']['tracked']>=5
    assert simulate(scene,[],5,10)['events']['fell']==[]
    pushed=simulate(scene,[Push(type='push',id=bottles[0].id,direction=(0,0,-1),impulse_Ns=1.5)],5,10)
    assert bottles[0].id in pushed['events']['tipped']
