"""Differentiable MLP and graph dynamics for the measured A/B experiment.

The graph predicts antisymmetric pair impulses. Gravity/drag and static support
are external influences. Full angular-momentum conservation is not claimed.
"""
from typing import Any
import numpy as np
import torch
from torch import nn
from .schema import Scene

STEP=.05

def normalize_q(q: torch.Tensor) -> torch.Tensor:
    return q/q.norm(dim=-1,keepdim=True).clamp_min(1e-8)

def rotation(q: torch.Tensor) -> torch.Tensor:
    x,y,z,w=normalize_q(q).unbind(-1)
    return torch.stack((1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),
                        2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),
                        2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)),dim=-1).reshape(*q.shape[:-1],3,3)

def euler_quaternion(e: torch.Tensor) -> torch.Tensor:
    x,y,z=(e/2).unbind(-1);cx,sx=x.cos(),x.sin();cy,sy=y.cos(),y.sin();cz,sz=z.cos(),z.sin()
    return torch.stack((sx*cy*cz-cx*sy*sz,cx*sy*cz+sx*cy*sz,cx*cy*sz-sx*sy*cz,cx*cy*cz+sx*sy*sz),-1)

def integrate_q(q: torch.Tensor,omega: torch.Tensor) -> torch.Tensor:
    vector=q[...,:3]+STEP/2*(q[...,3:4]*omega+torch.cross(omega,q[...,:3],dim=-1))
    scalar=q[...,3:4]-STEP/2*(omega*q[...,:3]).sum(-1,keepdim=True)
    return normalize_q(torch.cat((vector,scalar),-1))

def tensor_baseline(state: torch.Tensor,props: torch.Tensor,context: torch.Tensor,mask: torch.Tensor) -> torch.Tensor:
    g=context[:,3,None]
    pos=state[...,:3]+state[...,7:10]*STEP
    offset=torch.zeros_like(pos);offset[...,1]=g*STEP*(STEP+1/240)/2
    output=torch.cat((pos-offset,integrate_q(state[...,3:7],state[...,10:13]),torch.zeros_like(state[...,7:13])),-1)
    return torch.where((mask & (props[...,5]<.5))[...,None],output,state)

def mlp_features(state: torch.Tensor,props: torch.Tensor,context: torch.Tensor,mask: torch.Tensor) -> torch.Tensor:
    """Exactly the old 81 features plus a table collision-enabled bit."""
    batch,count,_=state.shape
    pos,q,vel,av=state[...,:3],state[...,3:7],state[...,7:10],state[...,10:13]
    size=props[...,:3]
    r=rotation(q);support_r=rotation(euler_quaternion(context[:,4:7]))
    center=torch.zeros_like(pos);center[...,1]=context[:,2,None]-.04
    local=torch.einsum('bij,bnj->bni',support_r.transpose(-1,-2),pos-center)
    ext=torch.matmul(r.abs(),(size/2)[...,None]).squeeze(-1)
    local_r=torch.matmul(support_r[:,None].transpose(-1,-2),r)
    half=torch.matmul(local_r.abs(),(size/2)[...,None]).squeeze(-1)
    gaps=torch.stack((local[...,1]-.04-half[...,1],pos[...,1]-ext[...,1],
                      context[:,0,None]/2-local[...,0].abs(),context[:,1,None]/2-local[...,2].abs()),-1)
    base=torch.cat((state,size,props[...,3:4].clamp_min(1e-6).log(),props[...,4:6],
                    context[:,None,:4].expand(-1,count,-1),context[:,None,4:7].expand(-1,count,-1),local,gaps),-1)
    dp=pos[:,None,:,:]-pos[:,:,None,:]
    distance=dp.norm(dim=-1)
    valid=mask[:,None,:].expand(-1,count,-1) & ~torch.eye(count,dtype=torch.bool,device=state.device)[None]
    # Ignore a static desk alias of the explicit support.
    alias=(props[...,5]>.5)&(size[...,0]>=context[:,0,None]*.95)&(size[...,2]>=context[:,1,None]*.95)
    valid=valid & ~alias[:,None,:] & (distance<.65)
    distance=distance.masked_fill(~valid,float('inf'))
    _,indices=distance.topk(3,dim=-1,largest=False)
    neighbors=[]
    for k in range(3):
        index=indices[:,:,k]
        other=torch.gather(state,1,index[...,None].expand(-1,-1,13))
        prop=torch.gather(props,1,index[...,None].expand(-1,-1,6))
        present=torch.gather(valid,2,index[...,None]).squeeze(-1)
        row=torch.cat((present[...,None].float(),other[...,:3]-pos,prop[...,:3],other[...,7:10],other[...,3:7],
                       prop[...,3:4].clamp_min(1e-6).log(),prop[...,5:6]),-1)
        neighbors.append(row*present[...,None])
    return torch.cat((base,*neighbors,context[:,None,7:8].expand(-1,count,-1)),-1)

class MultiStepMLP(nn.Module):
    kind='mlp_multistep'
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(82,192),nn.SiLU(),nn.Linear(192,192),nn.SiLU(),nn.Linear(192,128),nn.SiLU(),nn.Linear(128,13))
        self.register_buffer('xmean',torch.zeros(82));self.register_buffer('xstd',torch.ones(82))
        self.register_buffer('ymean',torch.zeros(13));self.register_buffer('ystd',torch.ones(13))
    def forward(self,state,props,context,mask):
        x=mlp_features(state,props,context,mask)
        delta=self.net((x-self.xmean)/self.xstd)*self.ystd+self.ymean
        output=tensor_baseline(state,props,context,mask)+delta
        output=torch.cat((output[...,:3],normalize_q(output[...,3:7]),output[...,7:13]),-1)
        return torch.where((mask & (props[...,5]<.5))[...,None],output,state)

class GraphImpulseModel(nn.Module):
    kind='graph_impulse'
    def __init__(self,width: int=64,rounds: int=3):
        super().__init__();self.rounds=rounds
        self.encoder=nn.Sequential(nn.Linear(20,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU())
        self.message=nn.Sequential(nn.Linear(width*2+13,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU())
        self.update=nn.Sequential(nn.Linear(width*2,width),nn.SiLU(),nn.Linear(width,width))
        self.impulse=nn.Sequential(nn.Linear(width*2+13,width),nn.SiLU(),nn.Linear(width,6))
        nn.init.zeros_(self.impulse[-1].weight);nn.init.zeros_(self.impulse[-1].bias)

    def nodes(self,state,props,context,mask):
        batch=state.shape[0];device=state.device
        extra_state=torch.zeros(batch,2,13,device=device);extra_state[:,:,6]=1
        extra_state[:,0,1]=context[:,2]-.04;extra_state[:,1,1]=-.05
        extra_state[:,0,3:7]=euler_quaternion(context[:,4:7])
        extra_props=torch.ones(batch,2,6,device=device)
        extra_props[:,0,:3]=torch.stack((context[:,0],torch.full_like(context[:,0],.08),context[:,1]),-1)
        extra_props[:,1,:3]=torch.tensor([20.,.1,20.],device=device)
        extra_props[:,:,4]=.5
        extra_mask=torch.stack((context[:,7]>.5,torch.ones(batch,dtype=torch.bool,device=device)),-1)
        # A static duplicate desk is represented only by the support node.
        alias=(props[...,5]>.5)&(props[...,0]>=context[:,0,None]*.95)&(props[...,2]>=context[:,1,None]*.95)
        return torch.cat((state,extra_state),1),torch.cat((props,extra_props),1),torch.cat((mask & ~alias,extra_mask),1)

    def pair_impulses(self,state,props,context,mask):
        all_state,all_props,valid=self.nodes(state,props,context,mask)
        pos,q,v,av=all_state[...,:3],all_state[...,3:7],all_state[...,7:10],all_state[...,10:13]
        r=rotation(q);extent=torch.matmul(r.abs(),(all_props[...,:3]/2)[...,None]).squeeze(-1)
        dp=pos[:,None,:,:]-pos[:,:,None,:];dv=v[:,None,:,:]-v[:,:,None,:]
        separation=dp.abs()-extent[:,:,None,:]-extent[:,None,:,:]
        edge_valid=valid[:,:,None]&valid[:,None,:]&(separation.clamp_min(0).norm(dim=-1)<.3)
        count=pos.shape[1]
        edge_valid=edge_valid & ~torch.eye(count,dtype=torch.bool,device=state.device)[None]
        edge=torch.cat((dp.clamp(-3,3),dv/5,separation.clamp(-2,2),q[:,None,:,:]-q[:,:,None,:]),-1)
        node=torch.cat((pos.clamp(-5,5)/2,q,v/5,av/20,all_props[...,:3].clamp_max(3),
                        all_props[...,3:4].clamp_min(.001).log()/3,all_props[...,4:6],
                        context[:,None,3:4].expand(-1,count,-1)/10),-1)
        h=self.encoder(node)*valid[...,None]
        degree=edge_valid.sum(-1,keepdim=True).clamp_min(1).sqrt()
        for _ in range(self.rounds):
            pair=torch.cat((h[:,:,None,:].expand(-1,-1,count,-1),h[:,None,:,:].expand(-1,count,-1,-1),edge),-1)
            message=self.message(pair)*edge_valid[...,None]
            h=(h+self.update(torch.cat((h,message.sum(2)/degree),-1))/self.rounds)*valid[...,None]
        pair=torch.cat((h[:,:,None,:].expand(-1,-1,count,-1),h[:,None,:,:].expand(-1,count,-1,-1),edge),-1)
        raw=self.impulse(pair)
        # Antisymmetry guarantees zero net pairwise linear impulse for the full graph.
        impulses=(raw-raw.transpose(1,2))/2*edge_valid[...,None]
        return impulses,all_props

    def forward(self,state,props,context,mask):
        count=state.shape[1]
        pair,_=self.pair_impulses(state,props,context,mask)
        impulse=pair.sum(2)[:,:count]
        mass=props[...,3:4].clamp_min(.01)
        v=state[...,7:10];av=state[...,10:13]
        # Bullet's configured damping is an external influence on linear momentum.
        drag=(1-.04*STEP*(1+v.norm(dim=-1,keepdim=True))).clamp_min(0)
        acceleration=torch.zeros_like(v);acceleration[...,1]=-context[:,3,None]
        next_v=v*drag+acceleration*STEP+impulse[...,:3]/mass
        sx,sy,sz=props[...,:3].unbind(-1)
        inertia=mass/12*torch.stack((sy*sy+sz*sz,sx*sx+sz*sz,sx*sx+sy*sy),-1)
        r=rotation(state[...,3:7]);torque_local=torch.matmul(r.transpose(-1,-2),impulse[...,3:6,None]).squeeze(-1)
        next_av=av+torch.matmul(r,(torque_local/inertia.clamp_min(1e-5))[...,None]).squeeze(-1)
        next_av=next_av.clamp(-40,40);next_v=next_v.clamp(-20,20)
        pos=state[...,:3]+(v+next_v)*STEP/2
        offset=torch.zeros_like(pos);offset[...,1]=context[:,3,None]*STEP/240/2
        q=integrate_q(state[...,3:7],(av+next_av)/2)
        output=torch.cat((pos-offset,q,next_v,next_av),-1)
        return torch.where((mask & (props[...,5]<.5))[...,None],output,state)

def build_model(kind: str) -> nn.Module:
    if kind=='mlp_multistep': return MultiStepMLP()
    if kind=='graph_impulse': return GraphImpulseModel()
    raise ValueError(f'Unknown learned architecture: {kind}')

class StateModelAdapter:
    def __init__(self,checkpoint: dict[str,Any]):
        self.model=build_model(checkpoint['kind']);self.model.load_state_dict(checkpoint['state_dict']);self.model.eval()
    def step(self,scene: Scene,states: np.ndarray,gravity: float,tilt: list[float]) -> np.ndarray:
        n=len(states)
        if not n:return states.copy()
        count=max(6,n);state=np.zeros((1,count,13),np.float32);state[:,:,6]=1;state[0,:n]=states
        props=np.zeros((1,count,6),np.float32);props[:,:,3]=1
        for i,o in enumerate(scene.objects):props[0,i]=[*o.size,o.mass_kg,o.friction,float(o.is_static)]
        mask=np.zeros((1,count),bool);mask[0,:n]=True
        context=np.array([[*scene.support.size,scene.support.height,gravity,*tilt,float(scene.support.collisions_enabled)]],np.float32)
        with torch.inference_mode():result=self.model(torch.from_numpy(state),torch.from_numpy(props),torch.from_numpy(context),torch.from_numpy(mask))
        return result[0,:n].numpy()
