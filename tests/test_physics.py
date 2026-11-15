import numpy as np
import pybullet as p
import pytest
from agatha.schema import Scene, SceneObject, Support, Push, Remove, Tilt, SetMass, Gravity
from agatha.coordinates import to_bullet, from_bullet, quat_to_bullet, quat_from_bullet
from agatha.physics import simulate

def scene(static=False):
    return Scene(objects=[SceneObject(id="box",label="box",size=(.1,.1,.1),position=(0,1.3,0),is_static=static)])

def test_falls_and_rests():
    r=simulate(scene(),[])
    assert r['frames'][10]['objects'][0]['position'][1]<1.3
    assert r['final_state']['objects'][0]['position'][1]==pytest.approx(.8,abs=.002)
    assert ['box','support'] in [list(x['ids']) for x in r['events']['contacts']]

def test_push_and_determinism():
    a=[Push(type='push',id='box',direction=(1,0),impulse_Ns=.3)]
    r=simulate(scene(),a)
    assert r==simulate(scene(),a)
    assert r['final_state']['objects'][0]['position'][0]>.1

def test_static_stays():
    assert simulate(scene(True),[Push(type='push',id='box',direction=(1,0),impulse_Ns=2)])['final_state']['objects'][0]['position']==(0,1.3,0)

def test_remove_does_not_mutate():
    s=scene(); before=s.model_dump()
    assert simulate(s,[Remove(type='remove',id='box')])['final_state']['objects']==[]
    assert s.model_dump()==before

def test_coordinate_roundtrips_and_rotation():
    rng=np.random.default_rng(42)
    for _ in range(100):
        v=rng.normal(size=3); q=rng.normal(size=4); q/=np.linalg.norm(q)
        assert np.allclose(from_bullet(to_bullet(v)),v)
        assert np.allclose(quat_from_bullet(quat_to_bullet(q)),q)
        r=np.array(p.getMatrixFromQuaternion(q)).reshape(3,3)
        rb=np.array(p.getMatrixFromQuaternion(quat_to_bullet(q))).reshape(3,3)
        assert np.allclose(to_bullet(r@v),rb@to_bullet(v))

def test_other_actions_and_unknown_id():
    s=scene()
    simulate(s,[SetMass(type='set_mass',id='box',kg=2),Tilt(type='tilt_support',angle_deg=12,axis='x'),Gravity(type='set_gravity',g=5)])
    assert s.objects[0].mass_kg==.5
    with pytest.raises(ValueError): simulate(s,[Remove(type='remove',id='unknown')])
