"""Refine the existing MLP on learned contact/landing behavior; no inference clamp."""
import argparse,hashlib,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch import nn
from agatha.dynamics import MultiStepMLP,normalize_q
from agatha.contact_training import contact_loss,support_geometry
from train_ab import trajectory_loss


def landing_frames(data: dict[str,torch.Tensor]) -> np.ndarray:
    count=len(data['states']);frames=np.full(count,-1,dtype=np.int64)
    for begin in range(0,count,128):
        batch=data['states'][begin:begin+128];n=len(batch);time_steps=batch.shape[1]
        props=data['props'][begin:begin+n].repeat_interleave(time_steps,0)
        context=data['context'][begin:begin+n].repeat_interleave(time_steps,0)
        states=batch.reshape(n*time_steps,*batch.shape[2:])
        gap,_,interior,floor=support_geometry(states,props,context)
        dynamic=data['mask'][begin:begin+n] & (data['props'][begin:begin+n,:,5]<.5)
        contact=(((gap.abs()<.015)&interior&(context[:,7,None]>.5))|(floor.abs()<.015)).reshape(n,time_steps,-1)&dynamic[:,None]
        events=((~contact[:,:-1])&contact[:,1:]).any(-1).numpy()
        for j in range(n):
            indices=np.flatnonzero(events[j])
            if len(indices):frames[begin+j]=int(indices[0])+1
    return frames


def refine(source: Path,output: Path,path: Path,epochs: int,horizon: int,samples: int,lr: float,basic_contact: bool=False):
    torch.set_num_threads(4);torch.manual_seed(144)
    file=np.load(path);data={key:torch.from_numpy(file[key]) for key in ('states','props','context','mask')}
    count=len(data['states']);train_ids=np.flatnonzero(np.arange(count)%10!=0);valid_ids=np.flatnonzero(np.arange(count)%10==0)
    metadata=json.loads(path.with_suffix('.json').read_text());base_count=metadata['base_dataset']['total_scenes']
    extra_ids=train_ids[train_ids>=base_count];ghost_ids=train_ids[data['context'][train_ids,7].numpy()<.5]
    checkpoint=torch.load(source,map_location='cpu',weights_only=True)
    if checkpoint.get('kind')!='mlp_multistep':raise ValueError('Contact refinement requires the existing multi-step MLP')
    model=MultiStepMLP();model.load_state_dict(checkpoint['state_dict'])
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-5)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,epochs,eta_min=lr/10)
    events=landing_frames(data);rng=np.random.default_rng(773318);val_rng=np.random.default_rng(713)
    val_scene=val_rng.choice(valid_ids,1024);val_horizon=12 if horizon<=8 else 20;val_time=val_rng.integers(0,61-val_horizon,len(val_scene))
    noise_rng=torch.Generator().manual_seed(774);batch_size=96
    output.parent.mkdir(parents=True,exist_ok=True);best=float('inf');start=time.perf_counter();best_epoch=0
    for epoch in range(epochs):
        index=np.concatenate((np.repeat(train_ids,samples),rng.choice(extra_ids,len(extra_ids)*2),rng.choice(ghost_ids,len(ghost_ids)*2)))
        rng.shuffle(index);times=rng.integers(0,61-horizon,len(index))
        # Half of windows with a real teacher touchdown include that event.
        focus=(events[index]>=0)&(rng.random(len(index))<.5)
        times[focus]=np.clip(events[index[focus]]-rng.integers(1,horizon+1,focus.sum()),0,60-horizon)
        model.train();running=0.;seen=0
        for begin in range(0,len(index),batch_size):
            ids=torch.from_numpy(index[begin:begin+batch_size]);t=torch.from_numpy(times[begin:begin+batch_size])
            props=data['props'][ids];context=data['context'][ids];mask=data['mask'][ids];state=data['states'][ids,t].clone()
            dynamic=mask & (props[...,5]<.5)
            noise=torch.randn(state.shape,generator=noise_rng)
            scale=torch.tensor([.0015,.0015,.0015,.002,.002,.002,.002,.025,.025,.025,.05,.05,.05])
            state=state+noise*scale*dynamic[...,None]
            state=torch.cat((state[...,:3],normalize_q(state[...,3:7]),state[...,7:13]),-1)
            loss=0.
            for step in range(horizon):
                state=model(state,props,context,mask);target=data['states'][ids,t+step+1]
                loss=loss+(trajectory_loss(state,target,props,mask)+contact_loss(state,target,props,context,mask,refined=not basic_contact))/horizon
            optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            running+=float(loss.detach())*len(ids);seen+=len(ids)
        scheduler.step();model.eval();validation=0.
        with torch.inference_mode():
            for begin in range(0,len(val_scene),batch_size):
                ids=torch.from_numpy(val_scene[begin:begin+batch_size]);t=torch.from_numpy(val_time[begin:begin+batch_size])
                props=data['props'][ids];context=data['context'][ids];mask=data['mask'][ids];state=data['states'][ids,t]
                loss=0.
                for step in range(val_horizon):
                    state=model(state,props,context,mask);target=data['states'][ids,t+step+1]
                    loss=loss+(trajectory_loss(state,target,props,mask)+contact_loss(state,target,props,context,mask,refined=not basic_contact))/val_horizon
                validation+=float(loss)*len(ids)
        validation/=len(val_scene)
        if validation<best:
            best=validation;best_epoch=epoch+1
            torch.save(dict(kind='mlp_multistep',state_dict=model.state_dict(),model_step_s=.05,epoch=best_epoch,contact_refined=True),output)
        print(f'contact epoch {epoch+1}/{epochs} train={running/seen:.5f} val{val_horizon}={validation:.5f} best={best_epoch} seconds={time.perf_counter()-start:.1f}',flush=True)
    metrics=dict(kind='mlp_multistep',contact_refined=True,parameters=sum(p.numel() for p in model.parameters()),
        warm_start_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),dataset=metadata,epochs=epochs,best_epoch=best_epoch,
        training_seconds=time.perf_counter()-start,training_horizon_steps=horizon,validation_horizon_steps=val_horizon,
        contact_loss_revision=1 if basic_contact else 2,validation_loss=best,learning_rate=lr,samples_per_scene=samples,training_seed=144,data_order_seed=773318,noise_seed=774,
        objective='Trajectory loss plus teacher-gated table/floor contact height, normal velocity and resting angular velocity; no inference snapping/projection',
        sampling='Uniform windows plus targeted contacts/ghost cases; half of available landing windows aligned to teacher touchdown')
    output.with_suffix('.json').write_text(json.dumps(metrics,indent=2));print(json.dumps(metrics,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=Path('models/mlp_multistep.pt'));p.add_argument('--output',type=Path,default=Path('/tmp/agatha-contact-candidate/world_model.pt'))
    p.add_argument('--data',type=Path,default=Path('data/contact_episodes.npz'));p.add_argument('--epochs',type=int,default=20);p.add_argument('--horizon',type=int,default=8)
    p.add_argument('--samples-per-scene',type=int,default=2);p.add_argument('--lr',type=float,default=.00015)
    p.add_argument('--basic-contact',action='store_true')
    a=p.parse_args();refine(a.source,a.output,a.data,a.epochs,a.horizon,a.samples_per_scene,a.lr,a.basic_contact)
