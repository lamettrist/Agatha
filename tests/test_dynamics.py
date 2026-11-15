import numpy as np
import torch
from agatha.schema import Scene,SceneObject
from agatha.world_model import prepare,features
from agatha.dynamics import mlp_features,GraphImpulseModel


def inputs():
    scene=Scene(objects=[SceneObject(id=f'b{i}',label='box',size=(.1,.1,.1),position=(i*.15,.8,0)) for i in range(4)])
    prepared,states,g,tilt=prepare(scene,[])
    state=torch.zeros(1,6,13);state[:,:,6]=1;state[0,:4]=torch.from_numpy(states)
    props=torch.zeros(1,6,6);props[:,:,3]=1
    for i,o in enumerate(scene.objects):props[0,i]=torch.tensor([*o.size,o.mass_kg,o.friction,float(o.is_static)])
    context=torch.tensor([[1.2,.8,.75,g,*tilt,1.]])
    mask=torch.tensor([[True,True,True,True,False,False]])
    return prepared,states,g,tilt,state,props,context,mask


def test_tensor_mlp_features_match_existing_features():
    scene,states,g,tilt,s,p,c,m=inputs()
    expected=features(scene,states,g,tilt)
    got=mlp_features(s,p,c,m)[0,:4,:81].numpy()
    assert np.allclose(got,expected,atol=1e-6)


def test_graph_pair_impulses_and_permutation_equivariance():
    *_,state,props,context,mask=inputs()
    torch.manual_seed(23);model=GraphImpulseModel()
    torch.nn.init.normal_(model.impulse[-1].weight,std=.01)
    with torch.inference_mode():
        impulses,_=model.pair_impulses(state,props,context,mask)
        assert torch.allclose(impulses,-impulses.transpose(1,2),atol=1e-7)
        assert torch.allclose(impulses.sum((1,2)),torch.zeros(1,6),atol=1e-6)
        order=torch.tensor([2,0,3,1,4,5]);inverse=order.argsort()
        a=model(state,props,context,mask)
        b=model(state[:,order],props[:,order],context,mask[:,order])[:,inverse]
        assert torch.allclose(a,b,atol=1e-5)


def test_collision_disabled_support_has_no_messages():
    *_,state,props,context,mask=inputs();model=GraphImpulseModel()
    context[:,7]=0
    impulses,_=model.pair_impulses(state,props,context,mask)
    assert not impulses[:,:,6].any()


def test_checkpoint_adapters_run_without_physics(tmp_path,monkeypatch):
    from agatha.dynamics import MultiStepMLP
    from agatha.world_model import LearnedModel,rollout
    from agatha.physics import World
    from agatha.schema import Push,SupportCollisions
    def fail(*args,**kwargs):raise AssertionError('Learned model stepped physics')
    monkeypatch.setattr(World,'step',fail)
    scene=Scene(objects=[SceneObject(id='bottle',label='bottle',size=(.075,.24,.075),position=(0,.87,0))])
    for model in (MultiStepMLP(),GraphImpulseModel()):
        path=tmp_path/(model.kind+'.pt');torch.save(dict(kind=model.kind,state_dict=model.state_dict()),path)
        result=rollout(scene,[Push(type='push',id='bottle',direction=(0,1,0),impulse_Ns=1.5),SupportCollisions(type='set_support_collisions',enabled=False)],3,20,LearnedModel(path))
        assert result['model_architecture']==model.kind
        assert result['support_collisions_enabled'] is False
        assert len(result['frames'])==4


def test_table_penetration_metric_distinguishes_deliberate_noclip():
    import sys
    sys.path.insert(0,str(__import__('pathlib').Path(__file__).resolve().parents[1]/'scripts'))
    from benchmark_ab import geometry
    scene=Scene(objects=[SceneObject(id='box',label='box',size=(.1,.1,.1),position=(0,.8,0))])
    result=dict(support_collisions_enabled=True,frames=[dict(objects=[dict(id='box',position=(0,.8,0),quaternion=(0,0,0,1))]),dict(objects=[dict(id='box',position=(0,.72,0),quaternion=(0,0,0,1))]),dict(objects=[dict(id='box',position=(0,.5,0),quaternion=(0,0,0,1))])])
    measured=geometry(scene,result)
    assert measured['penetration_object_frames']>0
    assert measured['passed_through_ids']==['box']
    result['support_collisions_enabled']=False
    assert geometry(scene,result)['eligible_object_frames']==0


def test_tensor_features_for_rotated_support_and_objects():
    import pybullet as p
    scene,states,g,_,s,props,c,mask=inputs()
    tilt=[.2,0,-.1];c[:,4:7]=torch.tensor(tilt)
    for i in range(4):
        states[i,3:7]=p.getQuaternionFromEuler([.1*i,.2,-.15])
    s[0,:4]=torch.from_numpy(states)
    assert np.allclose(mlp_features(s,props,c,mask)[0,:4,:81].numpy(),features(scene,states,g,tilt),atol=2e-6)
