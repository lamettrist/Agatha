import math
import numpy as np
import pytest
from agatha.scene_reconstruction.geometry import solve_scale,fit_plane,fit_box,fit_cylinder,clean_depth
from agatha.scene_reconstruction.bodies import settle, stability
from agatha.schema import Scene,SceneObject,Collision,CollisionPart,Push
from agatha.physics import simulate


def test_robust_scale_and_manual():
    refs=[dict(id=str(i),reference_m=.21,observed_m=v) for i,v in enumerate([.105,.106,.104,.04])]
    s=solve_scale(refs)
    assert s['factor']==pytest.approx(2,rel=.02)
    assert len(s['rejected'])==1
    assert solve_scale(refs,1.5)['factor']==1.5
    assert solve_scale([])['source']=='metric_model_unverified'
    with pytest.raises(ValueError): solve_scale([],float('nan'))


def test_plane_with_outliers_and_right_handed_basis():
    rng=np.random.default_rng(7); pts=rng.uniform(-1,1,(1000,3))
    pts[:800,1]=.4+rng.normal(0,.001,800)
    plane=fit_plane(pts,normal_hint=[0,-1,0])
    assert plane['inlier_ratio']>.79
    assert plane['rms_m']<.004
    assert np.linalg.det(plane['basis'])==pytest.approx(1)
    assert plane['normal'][1]<-.99


def test_box_rotated_dimensions():
    rng=np.random.default_rng(2); pts=rng.uniform(-.5,.5,(6000,3))*[.31,.025,.22]
    yaw=.45; c,s=math.cos(yaw),math.sin(yaw)
    rot=np.array([[c,0,s],[0,1,0],[-s,0,c]])
    fit=fit_box(pts@rot.T+[.1,.8,.2])
    assert fit['size']==pytest.approx([.31,.025,.22],rel=.07)
    assert abs(fit['rotation_euler'][1]-yaw)<.03


def test_partial_cylinder_completes_hidden_side():
    rng=np.random.default_rng(3); angles=rng.uniform(-1.2,1.2,1800)
    points=np.c_[.035*np.cos(angles),rng.uniform(0,.21,1800),.035*np.sin(angles)]
    fit=fit_cylinder(points+rng.normal(0,.0003,points.shape))
    assert fit['radius']==pytest.approx(.035,abs=.001)
    assert fit['height']==pytest.approx(.21,abs=.006)


def bottle_scene():
    return Scene(objects=[SceneObject(id='bottle',label='bottle',shape='cylinder',size=(.07,.21,.07),
        position=(0,1.1,0),mass_kg=.4,friction=.8,supported_by='support',
        collision=Collision(parts=[CollisionPart(type='cylinder',radius=.035,height=.21)]))])


def test_settle_stability_and_tip():
    s,report=settle(bottle_scene())
    assert report['float_count']==0 and report['penetration_count']==0
    assert s.objects[0].position[1]==pytest.approx(.855,abs=.002)
    measured=stability(s)
    assert measured['max_drift_m']<.01
    assert 'bottle' in simulate(s,[Push(type='push',id='bottle',direction=(1,0),impulse_Ns=.35)],2)['events']['tipped']


def test_overlap_is_reported_not_hidden():
    s=bottle_scene(); s.objects.append(s.objects[0].model_copy(update={'id':'other'}))
    _,report=settle(s,seconds=0)
    assert report['penetration_count']>0


def test_edge_pixels_excluded_from_geometry():
    depth=np.ones((50,50),np.float32); depth[:,25:]=2
    _,valid,report=clean_depth(depth,np.full((50,50,3),128,np.uint8))
    assert not valid[:,24:27].any()
    assert report['edge_pixels']>0
