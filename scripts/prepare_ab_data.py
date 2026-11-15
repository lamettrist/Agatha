"""Reuse the 5k-scene dataset and add launch/no-table-collision trajectories.

Both architectures train on these identical episode tensors and scene splits.
"""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from agatha.schema import Scene,SceneObject,Support,Push,SupportCollisions
from agatha.physics import World
from agatha.world_model import state_array,baseline,prepare
from generate_data import random_case

MAX_OBJECTS=6

def scenario(rng: np.random.Generator,kind: str) -> tuple[Scene,list]:
    if kind=='layout':return random_case(rng)
    if kind=='noclip':
        scene,actions=random_case(rng)
        return scene,actions+[SupportCollisions(type='set_support_collisions',enabled=False)]
    w=float(rng.uniform(.9,1.5));d=float(rng.uniform(.65,.95));h=float(rng.uniform(.6,.85))
    bottle=SceneObject(id='bottle',label='bottle',size=(float(rng.uniform(.06,.10)),float(rng.uniform(.18,.30)),float(rng.uniform(.06,.10))),
        position=(float(rng.uniform(-.15,.15)),h+.16,float(rng.uniform(-.1,.1))),mass_kg=float(rng.uniform(.3,1.0)),friction=.5)
    bottle.position=(bottle.position[0],h+bottle.size[1]/2,bottle.position[2])
    scene=Scene(objects=[bottle],support=Support(size=(w,d),height=h))
    direction=(float(rng.uniform(-.4,.4)),1.,float(rng.uniform(-.4,.4)))
    actions=[Push(type='push',id='bottle',direction=direction,impulse_Ns=float(rng.uniform(2,4))*bottle.mass_kg)]
    if kind=='toss_noclip':actions.append(SupportCollisions(type='set_support_collisions',enabled=False))
    return scene,actions

def episode(scene: Scene,actions: list) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    prepared,_,g,tilt=prepare(scene,actions)
    count=len(prepared.objects)
    states=np.zeros((61,MAX_OBJECTS,13),np.float32);states[:,:,6]=1
    props=np.zeros((MAX_OBJECTS,6),np.float32);props[:,3]=1
    mask=np.zeros(MAX_OBJECTS,bool);mask[:count]=True
    for i,o in enumerate(prepared.objects):props[i]=[*o.size,o.mass_kg,o.friction,float(o.is_static)]
    context=np.asarray([*scene.support.size,scene.support.height,g,*tilt,float(prepared.support.collisions_enabled)],np.float32)
    with World(scene) as world:
        world.apply(actions);ids=[o.id for o in prepared.objects]
        states[0,:count]=state_array(world.snapshot(0),ids)
        for i in range(60):
            world.step(12);current=state_array(world.snapshot((i+1)*.05),ids)
            previous=states[i,:count]
            flip=np.sum(previous[:,3:7]*current[:,3:7],axis=1)<0;current[flip,3:7]*=-1
            states[i+1,:count]=current
    return states,props,context,mask

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--extra',type=int,default=1000)
    parser.add_argument('--output',type=Path,default=Path('data/ab_episodes.npz'));args=parser.parse_args()
    start=time.perf_counter();data=np.load('models/transitions.npz');x=data['x'];y=data['y'];groups=data['groups']
    cases=json.loads(Path('models/transitions_cases.json').read_text())
    states=[];props=[];contexts=[];masks=[];kinds=[];offset=0
    for index,case in enumerate(cases):
        scene=Scene.model_validate(case['scene'])
        from pydantic import TypeAdapter
        from agatha.schema import Action
        actions=TypeAdapter(list[Action]).validate_python(case['actions'])
        prepared,_,g,tilt=prepare(scene,actions);n=len(prepared.objects);rows=n*60
        assert np.all(groups[offset:offset+rows]==index)
        current=x[offset:offset+rows,:13].reshape(60,n,13)
        target=baseline(current[-1],g,np.array([o.is_static for o in prepared.objects]))+y[offset+rows-n:offset+rows]
        episode_states=np.zeros((61,MAX_OBJECTS,13),np.float32);episode_states[:,:,6]=1
        episode_states[:60,:n]=current;episode_states[60,:n]=target
        prop=np.zeros((MAX_OBJECTS,6),np.float32);prop[:,3]=1
        for i,o in enumerate(prepared.objects):prop[i]=[*o.size,o.mass_kg,o.friction,float(o.is_static)]
        context=np.array([*scene.support.size,scene.support.height,g,*tilt,1.],np.float32)
        mask=np.zeros(MAX_OBJECTS,bool);mask[:n]=True
        states.append(episode_states);props.append(prop);contexts.append(context);masks.append(mask);kinds.append('layout');offset+=rows
    del x,y,groups,data
    rng=np.random.default_rng(41872)
    for index in range(args.extra):
        kind=('toss','noclip','toss_noclip')[index%3]
        scene,actions=scenario(rng,kind)
        s,p,c,m=episode(scene,actions);states.append(s);props.append(p);contexts.append(c);masks.append(m);kinds.append(kind)
        if (index+1)%100==0:print(f'extra {index+1}/{args.extra}, seconds={time.perf_counter()-start:.1f}',flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output,states=np.stack(states),props=np.stack(props),context=np.stack(contexts),mask=np.stack(masks),kinds=np.array(kinds))
    metadata=dict(base_scenes=len(cases),extra_scenes=args.extra,total_scenes=len(states),scene_transitions=len(states)*60,
                  extra_seed=41872,seconds=time.perf_counter()-start,step_s=.05,split='scene index % 10 == 0 held out for validation',
                  kinds={k:kinds.count(k) for k in sorted(set(kinds))},note='Identical episodes for graph/MLP, including vertical impulses and explicit table collision state. Bottles are boxes; no fluid model.')
    args.output.with_suffix('.json').write_text(json.dumps(metadata,indent=2));print(json.dumps(metadata,indent=2))
if __name__=='__main__':main()
