"""Fit learned residual dynamics, then evaluate autoregressive held-out scenes."""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch import nn
from agatha.world_model import DynamicsMLP,LearnedModel,rollout
from agatha.physics import simulate
from generate_data import random_case

def evaluate(checkpoint: Path, scenes: int=50) -> dict:
    rng=np.random.default_rng(90731); model=LearnedModel(checkpoint)
    errors={str(t):[] for t in [.5,1.,2.]}; agreements=[]; fall_counts=[0,0]
    for i in range(scenes):
        scene,actions=random_case(rng)
        expected=simulate(scene,actions,2,20); predicted=rollout(scene,actions,40,20,model)
        for t in [.5,1.,2.]:
            a={s['id']:s for s in expected['frames'][round(t*20)]['objects']}
            for s in predicted['frames'][round(t*20)]['objects']:
                if not next(o for o in scene.objects if o.id==s['id']).is_static:
                    errors[str(t)].append(float(np.linalg.norm(np.asarray(s['position'])-a[s['id']]['position'])))
        for o in scene.objects:
            if o.is_static or any(a.type=='remove' and a.id==o.id for a in actions): continue
            a=o.id in expected['events']['fell']; b=o.id in predicted['events']['fell']
            agreements.append(a==b); fall_counts[0]+=a; fall_counts[1]+=b
        if (i+1)%10==0: print(f'evaluated {i+1}/{scenes}',flush=True)
    return dict(evaluation_seed=90731,held_out_scenes=scenes,mean_position_error_m={t:float(np.mean(e)) for t,e in errors.items()},
        fell_agreement=float(np.mean(agreements)),evaluated_objects=len(agreements),physics_fell_count=fall_counts[0],learned_fell_count=fall_counts[1],
        note='Independent held-out scenes and actions; errors are autoregressive, not teacher-forced. Synthetic boxes only; no guarantee on photos.')

def train(data: Path,checkpoint: Path,epochs: int):
    torch.set_num_threads(4); torch.manual_seed(42); np.random.seed(42)
    datafile=np.load(data); x=torch.from_numpy(datafile['x']); y=torch.from_numpy(datafile['y']); groups=datafile['groups']
    # Split by entire scene, never by adjacent frames.
    valid=torch.from_numpy(groups%10==0); train=~valid
    xmean=x[train].mean(0); xstd=x[train].std(0).clamp_min(.05)
    ymean=y[train].mean(0); ystd=y[train].std(0).clamp_min(.005)
    x=(x-xmean)/xstd; y=(y-ymean)/ystd
    xt,yt=x[train],y[train]; xv,yv=x[valid],y[valid]
    model=DynamicsMLP(); optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-5)
    start=time.perf_counter(); best=float('inf'); best_epoch=0
    for epoch in range(epochs):
        model.train(); order=torch.randperm(len(xt)); running=0
        for indices in order.split(1024):
            prediction=model(xt[indices]); loss=nn.functional.smooth_l1_loss(prediction,yt[indices])
            optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),2); optimizer.step()
            running+=loss.item()*len(indices)
        model.eval()
        with torch.inference_mode(): val=nn.functional.smooth_l1_loss(model(xv),yv).item()
        if val<best:
            best=val; best_epoch=epoch+1
            torch.save(dict(state_dict=model.state_dict(),xmean=xmean,xstd=xstd,ymean=ymean,ystd=ystd,
                            model_step_s=.05,epoch=best_epoch),checkpoint)
        if (epoch+1)%5==0: print(f'epoch {epoch+1}/{epochs} train={running/len(xt):.5f} val={val:.5f} seconds={time.perf_counter()-start:.1f}',flush=True)
    training_s=time.perf_counter()-start
    metrics=evaluate(checkpoint)
    metrics.update(training_seconds=training_s,epochs=epochs,best_epoch=best_epoch,validation_loss=best,
                   parameters=sum(p.numel() for p in model.parameters()),training_seed=42,
                   dataset=json.loads(data.with_suffix('.json').read_text()),architecture='81 -> 192 -> 192 -> 128 -> 13 SiLU MLP; free-flight residual, 3 nearest neighbors')
    checkpoint.with_name('metrics.json').write_text(json.dumps(metrics,indent=2))
    print(json.dumps(metrics,indent=2),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--data',type=Path,default=Path('models/transitions.npz'))
    parser.add_argument('--output',type=Path,default=Path('models/world_model.pt')); parser.add_argument('--epochs',type=int,default=40)
    args=parser.parse_args(); train(args.data,args.output,args.epochs)
