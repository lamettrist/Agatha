"""Matched three-step/noise training: per-object MLP versus graph impulses."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch import nn
from agatha.dynamics import build_model,MultiStepMLP,mlp_features,tensor_baseline


def trajectory_loss(pred,target,props,mask):
    dynamic=mask & (props[...,5]<.5)
    position=nn.functional.smooth_l1_loss(pred[...,:3]/.05,target[...,:3]/.05,reduction='none').mean(-1)
    velocity=nn.functional.smooth_l1_loss(pred[...,7:10],target[...,7:10],reduction='none').mean(-1)
    angular=nn.functional.smooth_l1_loss(pred[...,10:13]/10,target[...,10:13]/10,reduction='none').mean(-1)
    orientation=1-(pred[...,3:7]*target[...,3:7]).sum(-1).square().clamp_max(1)
    loss=position+.3*velocity+.05*angular+.1*orientation
    return (loss*dynamic).sum()/dynamic.sum().clamp_min(1)


def initialize_mlp(model,data,train_ids):
    rng=np.random.default_rng(113)
    ids=torch.from_numpy(rng.choice(train_ids,20000));times=torch.from_numpy(rng.integers(0,60,20000))
    xs=[];ys=[]
    for i in range(0,len(ids),256):
        index=ids[i:i+256];t=times[i:i+256]
        state=data['states'][index,t];props=data['props'][index];context=data['context'][index];mask=data['mask'][index]
        x=mlp_features(state,props,context,mask)
        y=data['states'][index,t+1]-tensor_baseline(state,props,context,mask)
        xs.append(x[mask]);ys.append(y[mask])
    x=torch.cat(xs);y=torch.cat(ys)
    model.xmean.copy_(x.mean(0));model.xstd.copy_(x.std(0).clamp_min(.05))
    model.ymean.copy_(y.mean(0));model.ystd.copy_(y.std(0).clamp_min(.005))


def train(kind: str,path: Path,output: Path,epochs: int=30,samples_per_scene: int=4,batch_size: int=96):
    torch.set_num_threads(4);torch.manual_seed(42)
    file=np.load(path);data={key:torch.from_numpy(file[key]) for key in ('states','props','context','mask')}
    count=len(data['states']);train_ids=np.flatnonzero(np.arange(count)%10!=0);valid_ids=np.flatnonzero(np.arange(count)%10==0)
    model=build_model(kind)
    if isinstance(model,MultiStepMLP):initialize_mlp(model,data,train_ids)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-5)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,epochs,eta_min=.0001)
    rng=np.random.default_rng(42);val_rng=np.random.default_rng(377)
    val_scene=val_rng.choice(valid_ids,2048);val_time=val_rng.integers(0,58,2048)
    noise_rng=torch.Generator().manual_seed(991)
    output.parent.mkdir(parents=True,exist_ok=True);best=float('inf');best_epoch=0;start=time.perf_counter()
    for epoch in range(epochs):
        ids=np.repeat(train_ids,samples_per_scene);rng.shuffle(ids);times=rng.integers(0,58,len(ids))
        model.train();running=0.;seen=0
        for begin in range(0,len(ids),batch_size):
            index=torch.from_numpy(ids[begin:begin+batch_size]);t=torch.from_numpy(times[begin:begin+batch_size])
            props=data['props'][index];context=data['context'][index];mask=data['mask'][index]
            state=data['states'][index,t].clone();dynamic=mask & (props[...,5]<.5)
            noise=torch.randn(state.shape,generator=noise_rng)
            scale=torch.tensor([.001,.001,.001,0,0,0,0,.02,.02,.02,.05,.05,.05])
            state=state+noise*scale*dynamic[...,None]
            loss=0
            for step in range(3):
                state=model(state,props,context,mask)
                loss=loss+trajectory_loss(state,data['states'][index,t+step+1],props,mask)/3
            optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            running+=float(loss.detach())*len(index);seen+=len(index)
        scheduler.step();model.eval();validation=0.
        with torch.inference_mode():
            for begin in range(0,len(val_scene),batch_size):
                index=torch.from_numpy(val_scene[begin:begin+batch_size]);t=torch.from_numpy(val_time[begin:begin+batch_size])
                props=data['props'][index];context=data['context'][index];mask=data['mask'][index];state=data['states'][index,t]
                loss=0
                for step in range(3):
                    state=model(state,props,context,mask)
                    loss=loss+trajectory_loss(state,data['states'][index,t+step+1],props,mask)/3
                validation+=float(loss)*len(index)
        validation/=len(val_scene)
        if validation<best:
            best=validation;best_epoch=epoch+1
            torch.save(dict(kind=kind,state_dict=model.state_dict(),model_step_s=.05,epoch=best_epoch),output)
        print(f'{kind} epoch {epoch+1}/{epochs}: train={running/seen:.5f} val3={validation:.5f} best={best_epoch} seconds={time.perf_counter()-start:.1f}',flush=True)
    metrics=dict(kind=kind,parameters=sum(p.numel() for p in model.parameters()),training_seconds=time.perf_counter()-start,
        epochs=epochs,best_epoch=best_epoch,validation_loss=best,training_seed=42,data_order_seed=42,noise_seed=991,
        horizon_steps=3,message_passing_rounds=3 if kind=='graph_impulse' else 0,samples_per_scene_per_epoch=samples_per_scene,
        batch_size=batch_size,dataset=json.loads(path.with_suffix('.json').read_text()),
        training='Both models from scratch, identical scene split, sampled windows, loss, noise, optimizer/schedule and epochs. Checkpoints selected on validation three-step loss.')
    from train_world_model import evaluate
    metrics.update(evaluate(output,200,20261115))
    metrics['architecture']='three-round graph impulses' if kind=='graph_impulse' else '82 -> 192 -> 192 -> 128 -> 13 multi-step MLP'
    output.with_suffix('.json').write_text(json.dumps(metrics,indent=2))
    if output.name=='world_model.pt':
        output.with_name('metrics.json').write_text(json.dumps(metrics,indent=2))
    print(json.dumps(metrics,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=['mlp_multistep','graph_impulse'],required=True)
    p.add_argument('--data',type=Path,default=Path('data/ab_episodes.npz'));p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=30);p.add_argument('--samples-per-scene',type=int,default=4);p.add_argument('--batch-size',type=int,default=96)
    a=p.parse_args();train(a.kind,a.data,a.output,a.epochs,a.samples_per_scene,a.batch_size)
