"""Training-only contact diagnostics/losses. Never used to modify inference poses."""
import torch
from torch import nn
from .dynamics import rotation,euler_quaternion


def support_geometry(state: torch.Tensor,props: torch.Tensor,context: torch.Tensor) -> tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
    """Oriented-box lower extent and normal velocity in support coordinates."""
    r=rotation(state[...,3:7]);support_r=rotation(euler_quaternion(context[:,4:7]))
    center=torch.zeros_like(state[...,:3]);center[...,1]=context[:,2,None]-.04
    local=torch.einsum('bij,bnj->bni',support_r.transpose(-1,-2),state[...,:3]-center)
    local_r=torch.matmul(support_r[:,None].transpose(-1,-2),r)
    half=torch.matmul(local_r.abs(),(props[...,:3]/2)[...,None]).squeeze(-1)
    gap=local[...,1]-.04-half[...,1]
    interior=(local[...,0].abs()+half[...,0]<context[:,0,None]/2-.005)&(local[...,2].abs()+half[...,2]<context[:,1,None]/2-.005)
    normal_v=torch.einsum('bij,bnj->bni',support_r.transpose(-1,-2),state[...,7:10])[...,1]
    floor_half=torch.matmul(r.abs(),(props[...,:3]/2)[...,None]).squeeze(-1)[...,1]
    floor_gap=state[...,1]-floor_half
    return gap,normal_v,interior,floor_gap


def contact_loss(pred: torch.Tensor,target: torch.Tensor,props: torch.Tensor,context: torch.Tensor,mask: torch.Tensor,refined: bool=True) -> torch.Tensor:
    """Penalize contact drift only where the teacher actually supports the body.

    A finite-table interior test avoids inventing support under legitimate edge
    falls. The table term is disabled for ghost tables; floor contact stays real.
    Targets can be moving on impact: velocity is matched, not blindly zeroed.
    """
    pg,pv,_,pf=support_geometry(pred,props,context)
    tg,tv,interior,tf=support_geometry(target,props,context)
    dynamic=mask & (props[...,5]<.5)
    if refined:
        r=rotation(euler_quaternion(context[:,4:7]))
        center=torch.zeros_like(target[...,:3]);center[...,1]=context[:,2,None]-.04
        local=torch.einsum('bij,bnj->bni',r.transpose(-1,-2),target[...,:3]-center)
        com_inside=(local[...,0].abs()<context[:,0,None]/2-.005)&(local[...,2].abs()<context[:,1,None]/2-.005)
        stable=(target[...,7:10].norm(dim=-1)<.1)&(target[...,10:13].norm(dim=-1)<.3)
        interior=interior|(com_inside&stable)
    table=dynamic & (context[:,7,None]>.5) & interior & (tg.abs()<.015)
    floor=dynamic & (tf.abs()<.015)
    def term(pred_gap,true_gap,pred_v,true_v,active):
        # Ignore sub-millimeter target solver jitter rather than forcing zero gap.
        gap_error=((pred_gap-true_gap).abs()-.0015).clamp_min(0)/.01
        gap=nn.functional.smooth_l1_loss(gap_error,torch.zeros_like(gap_error),reduction='none')
        velocity=nn.functional.smooth_l1_loss(pred_v/.1,true_v/.1,reduction='none')
        penalty=.12*gap+.025*velocity
        if refined:
            allowed=torch.minimum(true_gap-.001,torch.full_like(true_gap,-.002))
            penetration=(allowed-pred_gap).clamp_min(0)/.005
            penalty=penalty+.06*nn.functional.smooth_l1_loss(penetration,torch.zeros_like(penetration),reduction='none')
        return (penalty*active).sum()/active.sum().clamp_min(1)
    loss=term(pg,tg,pv,tv,table)+term(pf,tf,pred[...,8],target[...,8],floor)
    settled=(table|floor)&(target[...,7:10].norm(dim=-1)<.1)&(target[...,10:13].norm(dim=-1)<.3)
    angular=nn.functional.smooth_l1_loss(pred[...,10:13],target[...,10:13],reduction='none').mean(-1)
    loss=loss+.02*(angular*settled).sum()/settled.sum().clamp_min(1)
    if refined:
        sign=torch.where((pred[...,3:7]*target[...,3:7]).sum(-1,keepdim=True)<0,-1.,1.)
        orientation=nn.functional.smooth_l1_loss(pred[...,3:7]*sign/.05,target[...,3:7]/.05,reduction='none').mean(-1)
        loss=loss+.03*(orientation*settled).sum()/settled.sum().clamp_min(1)
    return loss
