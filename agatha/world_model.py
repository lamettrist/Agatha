"""Small learned dynamics model; autoregressive rollout never steps PyBullet.

Known free-flight kinematics provide a residual baseline. A trained MLP predicts
all contact/friction/rotation corrections. No collision solver in learned rollout.
"""
from functools import lru_cache
import math
import os
from pathlib import Path
from typing import Any
import numpy as np
import pybullet as p  # quaternion math only, never a physics world in rollout
import torch
from torch import nn
from .schema import Scene, Action
from .physics import summarize
from .storage import ROOT

STEP = .05  # model learned at 20 Hz
STATE_DIM = 13  # p(3), q_xyzw(4), v(3), angular_v(3)
NEIGHBORS = 3

def state_array(frame: dict, ids: list[str]) -> np.ndarray:
    states={s['id']:s for s in frame['objects']}
    return np.asarray([list(states[id]['position'])+list(states[id]['quaternion'])+
                       list(states[id]['velocity'])+list(states[id]['angular_velocity']) for id in ids],dtype=np.float32).reshape(-1,13)

def prepare(scene: Scene, actions: list[Action]) -> tuple[Scene,np.ndarray,float,list[float]]:
    scene=scene.model_copy(deep=True)
    objects={o.id:o for o in scene.objects}
    states={o.id: np.asarray([*o.position,*p.getQuaternionFromEuler(o.rotation_euler),0,0,0,0,0,0],dtype=np.float32) for o in scene.objects}
    gravity=9.81; tilt=[0.,0.,0.]
    for a in actions:
        if hasattr(a,'id') and a.id not in states: raise ValueError(f'Unknown or removed object: {a.id}')
        if a.type=='remove':
            del states[a.id]; del objects[a.id]
        elif a.type=='set_mass': objects[a.id].mass_kg=a.kg; objects[a.id].mass_source='user'
        elif a.type=='push' and not objects[a.id].is_static:
            n=math.hypot(*a.direction)
            direction=a.direction if len(a.direction)==3 else (a.direction[0],0,a.direction[1])
            states[a.id][7:10]+=np.asarray(direction)*a.impulse_Ns/n/objects[a.id].mass_kg
        elif a.type=='set_support_collisions': scene.support.collisions_enabled=a.enabled
        elif a.type=='set_gravity': gravity=a.g
        elif a.type=='tilt_support': tilt[0 if a.axis=='x' else 2]=math.radians(a.angle_deg)
    scene.objects=list(objects.values())
    return scene,np.asarray(list(states.values()),dtype=np.float32).reshape(-1,13),gravity,tilt

def features(scene: Scene, states: np.ndarray, gravity: float, tilt: list[float]) -> np.ndarray:
    rows=[]
    w,d=scene.support.size; h=scene.support.height
    support_rot=np.asarray(p.getMatrixFromQuaternion(p.getQuaternionFromEuler(tilt))).reshape(3,3)
    sizes=np.asarray([o.size for o in scene.objects])
    for i,(o,s) in enumerate(zip(scene.objects,states)):
        pos,q,v,av=s[:3],s[3:7],s[7:10],s[10:13]
        r=np.asarray(p.getMatrixFromQuaternion(q)).reshape(3,3)
        ext=np.abs(r)@np.asarray(o.size)/2
        local=support_rot.T@(pos-[0,h-.04,0])
        half=np.abs(support_rot.T@r)@np.asarray(o.size)/2
        row=[*pos,*q,*v,*av,*o.size,math.log(o.mass_kg),o.friction,float(o.is_static),
             w,d,h,gravity,*tilt,*local,local[1]-.04-half[1],pos[1]-ext[1],
             w/2-abs(local[0]),d/2-abs(local[2])]
        others=[]
        for j,other in enumerate(scene.objects):
            if j==i: continue
            # A demo desk collider duplicates the explicit support; avoid double counting it.
            if other.is_static and other.size[0]>=w*.95 and other.size[2]>=d*.95: continue
            dp=states[j,:3]-pos
            distance=np.linalg.norm(dp)
            if distance<.65: others.append((distance,j,dp))
        others.sort(key=lambda x:x[0])
        for k in range(NEIGHBORS):
            if k<len(others):
                _,j,dp=others[k]; other=scene.objects[j]
                row.extend([1.,*dp,*sizes[j],*states[j,7:10],*states[j,3:7],
                            math.log(other.mass_kg),float(other.is_static)])
            else: row.extend([0.]*16)
        rows.append(row)
    return np.asarray(rows,dtype=np.float32).reshape(len(states),81)

def baseline(states: np.ndarray, gravity: float, static: np.ndarray) -> np.ndarray:
    out=states.copy()
    out[:,:3]+=states[:,7:10]*STEP
    out[:,1]-=gravity*STEP*(STEP+1/240)/2
    out[:,7:13]=0  # Next velocities are learned directly for stable autoregression.
    # World angular velocity integrated into xyzw quaternions.
    q=states[:,3:7]; omega=states[:,10:13]
    out[:,3:6]+=STEP/2*(q[:,3:4]*omega+np.cross(omega,q[:,:3]))
    out[:,6]-=STEP/2*np.sum(omega*q[:,:3],axis=1)
    out[:,3:7]/=np.maximum(np.linalg.norm(out[:,3:7],axis=1,keepdims=True),1e-8)
    out[static]=states[static]
    return out

class DynamicsMLP(nn.Module):
    def __init__(self, inputs: int = 81):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(inputs,192),nn.SiLU(),nn.Linear(192,192),nn.SiLU(),
                               nn.Linear(192,128),nn.SiLU(),nn.Linear(128,13))
    def forward(self,x): return self.net(x)

CONTACT_PROJECTION_ENV='AGATHA_CONTACT_PROJECTION'

def contact_projection_default() -> bool:
    """Projection is on unless AGATHA_CONTACT_PROJECTION is 0/false/off/no (read at call time)."""
    return os.environ.get(CONTACT_PROJECTION_ENV,'1').strip().lower() not in ('0','false','off','no')

def contact_approximations(scene: Scene) -> list[str]:
    """Where a valid Scene leaves what contact_gnn was trained on (boxes, 0.08 m slab, default damping)."""
    from .contact_gnn import SUPPORT_THICKNESS
    notes=[]; objects=scene.objects
    if any(o.shape!='box' or o.collision is not None for o in objects):
        notes.append('non-box shapes/collision parts are approximated by their bounding box')
    if abs(scene.support.thickness-SUPPORT_THICKNESS)>1e-6:
        notes.append(f'the support slab is modelled {SUPPORT_THICKNESS:g} m thick (scene says {scene.support.thickness:g} m)')
    if any(abs(o.linear_damping-.04)>1e-9 or abs(o.angular_damping-.04)>1e-9 for o in objects):
        notes.append('custom damping is ignored (trained with 0.04)')
    if any(o.soft_approx for o in objects): notes.append('soft objects are treated as rigid boxes')
    if any(o.restitution>.6 for o in objects): notes.append('restitution above the trained range is clamped to 0.6')
    return notes

class ContactGNNAdapter:
    """contact_gnn checkpoint behind the LearnedModel.step interface; optional non-learned projection."""
    def __init__(self,checkpoint: dict[str,Any]):
        from . import contact_gnn
        self.module=contact_gnn
        self.model=contact_gnn.build_from_checkpoint(checkpoint)
    def step(self,scene: Scene,states: np.ndarray,gravity: float,tilt: list[float],projection: bool = True) -> np.ndarray:
        n=len(states)
        if not n: return states.copy()
        state,props,context,mask=self.module.scene_tensors(scene,states,gravity,tilt)
        # Per-object restitution is a trained input (0-0.6); everything else uses the bounding box.
        for i,o in enumerate(scene.objects[:n]):
            if o.restitution: props[0,i,6]=min(o.restitution,.6)
        with torch.inference_mode():
            output=self.model(state,props,context,mask)
            if projection: output=self.module.project(state,output,props,context,mask)
        return output[0,:n].numpy().astype(np.float32)

class LearnedModel:
    def __init__(self,path: str | Path = ROOT/'models/world_model.pt',projection: bool | None = None):
        torch.set_num_threads(4)
        self._projection=projection  # None: follow AGATHA_CONTACT_PROJECTION (default on); contact_gnn only
        path=Path(path)
        if not path.exists(): raise FileNotFoundError('Learned checkpoint missing. Regenerate it with the README training commands; physics remains available.')
        checkpoint=torch.load(path,map_location='cpu',weights_only=True)
        self.kind=checkpoint.get('kind','mlp_one_step')
        self.adapter=None
        if self.kind in ('mlp_multistep','graph_impulse'):
            from .dynamics import StateModelAdapter
            self.adapter=StateModelAdapter(checkpoint)
            self.model=self.adapter.model
            return
        if self.kind=='contact_gnn':
            self.adapter=ContactGNNAdapter(checkpoint)
            self.model=self.adapter.model
            return
        self.model=DynamicsMLP(); self.model.load_state_dict(checkpoint['state_dict']); self.model.eval()
        self.xmean=checkpoint['xmean']; self.xstd=checkpoint['xstd']
        self.ymean=checkpoint['ymean']; self.ystd=checkpoint['ystd']

    @property
    def projection_applied(self) -> bool:
        """Whether the non-learned contact projection runs after each step (always False for old kinds)."""
        if self.kind!='contact_gnn': return False
        return contact_projection_default() if self._projection is None else bool(self._projection)

    def step(self,scene: Scene,states: np.ndarray,gravity: float,tilt: list[float]) -> np.ndarray:
        if not len(states): return states.copy()
        if self.adapter is not None:
            if self.kind=='contact_gnn': output=self.adapter.step(scene,states,gravity,tilt,projection=self.projection_applied)
            else: output=self.adapter.step(scene,states,gravity,tilt)
            if not np.isfinite(output).all(): raise ValueError('Learned model diverged')
            return output
        if not scene.support.collisions_enabled:
            raise RuntimeError('This legacy checkpoint has no table-collision input; train the multi-step models for the noclip counterfactual')
        static=np.array([o.is_static for o in scene.objects])
        x=torch.from_numpy(features(scene,states,gravity,tilt))
        with torch.inference_mode():
            correction=(self.model((x-self.xmean)/self.xstd)*self.ystd+self.ymean).numpy()
        out=baseline(states,gravity,static)+correction
        # Numerical bounds only; these do not impose floor or support collisions.
        out[:,7:10]=np.clip(out[:,7:10],-20,20)
        out[:,10:13]=np.clip(out[:,10:13],-40,40)
        out[:,3:7]/=np.maximum(np.linalg.norm(out[:,3:7],axis=1,keepdims=True),1e-8)
        out[static]=states[static]
        if not np.isfinite(out).all(): raise ValueError('Learned model diverged')
        return out.astype(np.float32)

@lru_cache(maxsize=4)
def load_model(architecture: str = 'default') -> LearnedModel:
    filenames={'default':'world_model.pt','mlp_multistep':'mlp_multistep.pt','graph_impulse':'graph_impulse.pt','contact_gnn':'contact_gnn.pt'}
    if architecture not in filenames: raise ValueError('Unknown learned architecture')
    try: return LearnedModel(ROOT/'models'/filenames[architecture])
    except FileNotFoundError: raise
    except Exception as exc:
        raise RuntimeError(f'Learned checkpoint could not load ({type(exc).__name__}); physics remains available') from exc

def frame(states: np.ndarray, scene: Scene, t: float) -> dict:
    return dict(t=t,objects=[dict(id=o.id,position=s[:3].tolist(),quaternion=s[3:7].tolist(),
             velocity=s[7:10].tolist(),angular_velocity=s[10:13].tolist()) for o,s in zip(scene.objects,states)])

def rollout(scene: Scene, actions: list[Action], steps: int = 60, fps: int = 20,
            model: LearnedModel | None = None) -> dict[str,Any]:
    if fps not in (10,20,30) or not 1<=steps<=150: raise ValueError('Invalid rollout length/fps')
    prepared,states,g,tilt=prepare(scene,actions)
    model=model or load_model()
    count=math.ceil(steps/fps/STEP)
    predictions=[states]
    for _ in range(count): predictions.append(model.step(prepared,predictions[-1],g,tilt))
    frames=[]
    for i in range(steps+1):
        u=i/fps/STEP; low=min(int(u),count); high=min(low+1,count); weight=u-low
        a,b=predictions[low],predictions[high].copy()
        if len(a):
            flip=np.sum(a[:,3:7]*b[:,3:7],axis=1)<0; b[flip,3:7]*=-1
        s=(1-weight)*a+weight*b
        if len(s): s[:,3:7]/=np.maximum(np.linalg.norm(s[:,3:7],axis=1,keepdims=True),1e-8)
        frames.append(frame(s,prepared,i/fps))
    if model.kind=='contact_gnn':
        applied=model.projection_applied
        notes=contact_approximations(prepared)
        disclaimer=('Learned contact graph network trained on synthetic PyBullet boxes; '+
                    ('a non-learned contact projection (push-out along the minimum-overlap axis, clamp to the support plane, sleep of tiny velocities) is applied after each step.'
                     if applied else 'no projection applied, so objects may interpenetrate or drift.')+
                    ' Contacts are not reported. Autoregressive error grows with time.'+
                    (' Approximations: '+'; '.join(notes)+'.' if notes else ''))
        return dict(engine='learned',**summarize(prepared,frames,tilt=tilt),support_rotation_euler=tilt,support_collisions_enabled=prepared.support.collisions_enabled,
                    estimated=True,contacts_available=False,model_architecture='contact_gnn',projection_applied=applied,
                    approximations=notes,disclaimer=disclaimer)
    return dict(engine='learned',**summarize(prepared,frames,tilt=tilt),support_rotation_euler=tilt,support_collisions_enabled=prepared.support.collisions_enabled,
                estimated=True,contacts_available=False,model_architecture=model.kind,
                disclaimer=f'Learned {model.kind} prediction, trained on synthetic boxes; contacts are not detected. Autoregressive error grows with time.')

def compare(physics: dict,learned: dict) -> dict:
    per_frame=[]
    for a,b in zip(physics['frames'],learned['frames']):
        expected={o['id']:np.asarray(o['position']) for o in a['objects']}
        errors=[float(np.linalg.norm(np.asarray(o['position'])-expected[o['id']])) for o in b['objects'] if o['id'] in expected]
        per_frame.append(dict(t=a['t'],mean_position_error_m=float(np.mean(errors)) if errors else 0.))
    return dict(mean_position_error_m=float(np.mean([v['mean_position_error_m'] for v in per_frame])),
                final_position_error_m=per_frame[-1]['mean_position_error_m'],per_frame=per_frame,
                fell_agreement=physics['events']['fell']==learned['events']['fell'])
