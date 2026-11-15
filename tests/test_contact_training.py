import torch
from agatha.contact_training import contact_loss


def scene_inputs():
    state=torch.zeros(1,6,13);state[:,:,6]=1;state[0,0,1]=.8
    props=torch.zeros(1,6,6);props[:,:,3]=1;props[0,0,:3]=.1
    context=torch.tensor([[1.2,.8,.75,9.81,0,0,0,1.]])
    mask=torch.tensor([[True,False,False,False,False,False]])
    return state,props,context,mask


def test_contact_loss_zero_for_teacher_and_penalizes_sinking():
    target,props,context,mask=scene_inputs()
    assert contact_loss(target,target,props,context,mask)==0
    predicted=target.clone();predicted[0,0,1]-=.025;predicted[0,0,8]=-.2;predicted.requires_grad_()
    loss=contact_loss(predicted,target,props,context,mask)
    assert loss>.1;loss.backward()
    assert predicted.grad[0,0,1]<0  # gradient descent raises the sunk body


def test_contact_loss_respects_disabled_table_and_edge_falls():
    target,props,context,mask=scene_inputs();predicted=target.clone();predicted[0,0,1]-=.025
    context[:,7]=0
    assert contact_loss(predicted,target,props,context,mask)==0
    context[:,7]=1;target[0,0,0]=.59;predicted[0,0,0]=.59;target[0,0,8]=-.5
    assert contact_loss(predicted,target,props,context,mask)==0


def test_real_floor_contact_still_trains_with_disabled_table():
    target,props,context,mask=scene_inputs();context[:,7]=0;target[0,0,1]=.05
    predicted=target.clone();predicted[0,0,1]=.02
    assert contact_loss(predicted,target,props,context,mask)>.1


def test_stable_partial_overhang_is_not_treated_as_a_fall():
    target,props,context,mask=scene_inputs();target[0,0,0]=.59
    predicted=target.clone();predicted[0,0,1]-=.025
    assert contact_loss(predicted,target,props,context,mask)>.1
    assert contact_loss(predicted,target,props,context,mask,refined=False)==0
