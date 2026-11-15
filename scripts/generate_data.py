"""Generate real PyBullet state/action transitions on CPU. No API keys required."""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from agatha.schema import Scene,SceneObject,Support,Push,SetMass,Remove,Tilt,Gravity
from agatha.physics import World
from agatha.world_model import features,baseline,state_array,STEP

def random_case(rng: np.random.Generator) -> tuple[Scene,list]:
    w=float(rng.uniform(.9,1.5)); d=float(rng.uniform(.65,1.0)); h=float(rng.uniform(.6,.9))
    objects=[]
    for i in range(int(rng.integers(2,7))):
        size=rng.uniform([.06,.02,.06],[.24,.23,.24])
        if rng.random()<.3: size[1]=rng.uniform(.015,.045)
        x=float(rng.uniform(-w*.46,w*.46)); z=float(rng.uniform(-d*.44,d*.44))
        if objects and rng.random()<.3:
            old=objects[int(rng.integers(len(objects)))]; x=old.position[0]+float(rng.uniform(-.03,.03)); z=old.position[2]+float(rng.uniform(-.03,.03))
        bottom=h
        for old in objects:
            if abs(x-old.position[0])<(size[0]+old.size[0])/2 and abs(z-old.position[2])<(size[2]+old.size[2])/2:
                bottom=max(bottom,old.position[1]+old.size[1]/2)
        objects.append(SceneObject(id=f'b{i}',label='box',size=tuple(size),position=(x,bottom+size[1]/2,z),
                                   mass_kg=float(rng.uniform(.1,2)),friction=float(rng.uniform(.25,.8)),is_static=rng.random()<.04))
    scene=Scene(objects=objects,support=Support(size=(w,d),height=h))
    id=objects[int(rng.integers(len(objects)))].id
    actions=[]
    if rng.random()<.8:
        theta=float(rng.uniform(-np.pi,np.pi))
        actions.append(Push(type='push',id=id,direction=(np.cos(theta),np.sin(theta)),impulse_Ns=float(rng.uniform(.05,1.4))))
    choice=rng.random()
    if choice<.12: actions.insert(0,SetMass(type='set_mass',id=id,kg=float(rng.uniform(.1,2.5))))
    elif choice<.22:
        removable=next((o.id for o in objects if o.id!=id),None)
        if removable: actions.append(Remove(type='remove',id=removable))
    elif choice<.4: actions.append(Tilt(type='tilt_support',angle_deg=float(rng.uniform(-25,25)),axis=str(rng.choice(['x','z']))))
    elif choice<.48: actions.append(Gravity(type='set_gravity',g=float(rng.uniform(2,15))))
    return scene,actions

def generate(scenes: int,seed: int,path: Path):
    rng=np.random.default_rng(seed); xs=[]; ys=[]; groups=[]; cases=[]; transitions=0
    start=time.perf_counter()
    for index in range(scenes):
        scene,actions=random_case(rng)
        with World(scene) as world:
            world.apply(actions)
            metadata=world.scene.model_copy(deep=True)
            metadata.objects=[o for o in metadata.objects if o.id in world.bodies]
            ids=[o.id for o in metadata.objects]
            static=np.asarray([o.is_static for o in metadata.objects])
            current=state_array(world.snapshot(0),ids)
            for step in range(60):
                x=features(metadata,current,world.gravity,world.tilt)
                world.step(12)
                target=state_array(world.snapshot((step+1)*STEP),ids)
                flip=np.sum(current[:,3:7]*target[:,3:7],axis=1)<0; target[flip,3:7]*=-1
                y=target-baseline(current,world.gravity,static)
                xs.append(x); ys.append(y); groups.extend([index]*len(x)); transitions+=1
                current=target
        cases.append(dict(scene=scene.model_dump(mode='json'),actions=[a.model_dump(mode='json') for a in actions]))
        if (index+1)%50==0: print(f'{index+1}/{scenes} scenes; {transitions} transitions; {time.perf_counter()-start:.1f}s',flush=True)
    np.savez_compressed(path,x=np.concatenate(xs),y=np.concatenate(ys),groups=np.asarray(groups))
    elapsed=time.perf_counter()-start
    path.with_name('transitions_cases.json').write_text(json.dumps(cases))
    info=dict(action_encoding='Actions at t=0 encoded in post-action velocities, masses, membership, support tilt and gravity. groups index complete scene/action recipes in transitions_cases.json.',seed=seed,scenes=scenes,scene_transitions=transitions,object_transitions=len(groups),seconds=elapsed,step_s=STEP)
    path.with_suffix('.json').write_text(json.dumps(info,indent=2))
    print(json.dumps(info),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--scenes',type=int,default=500); parser.add_argument('--seed',type=int,default=2026)
    parser.add_argument('--output',type=Path,default=Path('models/transitions.npz')); args=parser.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True); generate(args.scenes,args.seed,args.output)
