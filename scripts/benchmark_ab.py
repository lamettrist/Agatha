"""Paired physics/MLP/graph evaluation, including bottle toss and ghost table.

Penetration samples box vertices, face/edge centers and COM in the finite table slab, with a 5mm tolerance.
Pass-through separately counts a COM crossing below the table while over its footprint.
These geometric diagnostics are approximations, applied identically to all engines.
"""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pybullet as p
from agatha.schema import Scene,SceneObject,Push
from agatha.physics import simulate
from agatha.world_model import LearnedModel,rollout
from prepare_ab_data import scenario

HORIZONS=(.5,1.,2.)

def stack_case(rng: np.random.Generator,objects: int) -> tuple[Scene,list]:
    objs=[]
    for i in range(objects):
        size=(float(rng.uniform(.10,.18)),.045,float(rng.uniform(.10,.16)))
        x=(-.15 if i%2==0 else .15)+float(rng.uniform(-.01,.01));z=float(rng.uniform(-.01,.01))
        objs.append(SceneObject(id=f's{i}',label='book',size=size,position=(x,.75+size[1]*(i//2+.5),z),mass_kg=float(rng.uniform(.2,.6))))
    return Scene(objects=objs),[Push(type='push',id='s0',direction=(1,0),impulse_Ns=.25)]

def geometry(scene: Scene,result: dict) -> dict:
    if not result.get('support_collisions_enabled',True):
        return dict(penetration_object_frames=0,eligible_object_frames=0,max_penetration_m=None,passed_through_ids=[])
    tilt=result.get('support_rotation_euler',[0,0,0]);r=np.array(p.getMatrixFromQuaternion(p.getQuaternionFromEuler(tilt))).reshape(3,3)
    center=np.array([0,scene.support.height-.04,0]);half=np.array([scene.support.size[0]/2,.04,scene.support.size[1]/2])
    corners=np.array([[x,y,z] for x in (-1,0,1) for y in (-1,0,1) for z in (-1,0,1)])
    lookup={o.id:o for o in scene.objects};bad=eligible=0;maximum=0.;above=set();passed=set()
    for frame in result['frames']:
        for obj in frame['objects']:
            o=lookup[obj['id']]
            if o.is_static:continue
            eligible+=1
            position=np.array(obj['position']);qrot=np.array(p.getMatrixFromQuaternion(obj['quaternion'])).reshape(3,3)
            points=(corners*np.array(o.size)/2)@qrot.T+position
            local=(points-center)@r;com=(position-center)@r
            footprint=(np.abs(local[:,0])<half[0])&(np.abs(local[:,2])<half[2])
            slab=footprint&(local[:,1]>.04-.08)&(local[:,1]<.04-.005)
            if slab.any():bad+=1;maximum=max(maximum,float((.04-local[slab,1]).max()))
            if com[1]>=.04:above.add(o.id)
            if o.id in above and abs(com[0])<half[0] and abs(com[2])<half[2] and com[1]<-.04:
                passed.add(o.id)
    return dict(penetration_object_frames=bad,eligible_object_frames=eligible,max_penetration_m=maximum,passed_through_ids=sorted(passed))

def benchmark(checkpoints: dict[str,Path],per_suite: int=30,layout_scenes: int=100) -> dict:
    models={name:LearnedModel(path) for name,path in checkpoints.items()}
    output={name:{} for name in ['physics',*models]}
    cases=[]
    for suite in ('layout','toss','noclip','toss_noclip','stacks','more_objects'):
        rng=np.random.default_rng(824911+sum(map(ord,suite)))
        count=layout_scenes if suite=='layout' else per_suite
        for i in range(count):
            if suite in ('stacks','more_objects'):scene,actions=stack_case(rng,6 if suite=='stacks' else 9)
            else:scene,actions=scenario(rng,suite)
            cases.append((suite,i,scene,actions))
    accum={name:{} for name in output}
    for index,(suite,case_index,scene,actions) in enumerate(cases):
        start=time.perf_counter();reference=simulate(scene,actions,2,20);physics_ms=(time.perf_counter()-start)*1000
        results={'physics':reference};speeds={'physics':physics_ms}
        for name,model in models.items():
            start=time.perf_counter();results[name]=rollout(scene,actions,40,20,model);speeds[name]=(time.perf_counter()-start)*1000
        retained={o['id'] for o in reference['frames'][0]['objects']};objects=[o for o in scene.objects if not o.is_static and o.id in retained]
        for name,result in results.items():
            a=accum[name].setdefault(suite,dict(errors={str(t):[] for t in HORIZONS},tp=0,fp=0,tn=0,fn=0,
                penetration=0,eligible=0,max_penetration=0.,passed_through=0,stable=0,scenes=0,milliseconds=[]))
            for t in HORIZONS:
                expected={o['id']:o for o in reference['frames'][round(t*20)]['objects']}
                actual={o['id']:o for o in result['frames'][round(t*20)]['objects']}
                for o in objects:a['errors'][str(t)].append(float(np.linalg.norm(np.asarray(actual[o.id]['position'])-expected[o.id]['position'])))
            for o in objects:
                truth=o.id in reference['events']['fell'];pred=o.id in result['events']['fell']
                a['tp' if truth and pred else 'fp' if pred else 'fn' if truth else 'tn']+=1
            g=geometry(scene,result);a['penetration']+=g['penetration_object_frames'];a['eligible']+=g['eligible_object_frames']
            a['max_penetration']=max(a['max_penetration'],g['max_penetration_m'] or 0.);a['passed_through']+=len(g['passed_through_ids'])
            states=[o['position'] for f in result['frames'] for o in f['objects']]
            a['stable']+=bool(np.isfinite(states).all() and np.max(np.abs(states),initial=0)<10)
            a['scenes']+=1;a['milliseconds'].append(speeds[name])
        if (index+1)%25==0:print(f'benchmarked {index+1}/{len(cases)} cases',flush=True)
    for name,suites in accum.items():
        for suite,a in suites.items():
            n=a['tp']+a['tn']+a['fp']+a['fn']
            output[name][suite]=dict(scenes=a['scenes'],mean_position_error_m={t:float(np.mean(e)) for t,e in a['errors'].items()},
                fall_confusion={key:a[key] for key in ('tp','fp','tn','fn')},fell_agreement=(a['tp']+a['tn'])/max(n,1),
                fall_precision=a['tp']/(a['tp']+a['fp']) if a['tp']+a['fp'] else None,
                fall_recall=a['tp']/(a['tp']+a['fn']) if a['tp']+a['fn'] else None,
                table_penetration_frame_fraction=a['penetration']/a['eligible'] if a['eligible'] else None,
                max_table_penetration_m=a['max_penetration'] if a['eligible'] else None,
                passed_through_solid_table_objects=a['passed_through'] if a['eligible'] else None,
                stable_scene_fraction=a['stable']/a['scenes'],median_rollout_ms=float(np.median(a['milliseconds'])),
                mean_rollout_ms=float(np.mean(a['milliseconds'])))
    return dict(protocol=dict(seed_basis=824911,horizons_s=list(HORIZONS),fps=20,duration_s=2,layout_scenes=layout_scenes,per_other_suite=per_suite,
        matched_cases=True,model_selection='Development A/B benchmark, not a blind research test.',
        penetration='27 box sample points within finite support slab and >5mm below top; geometric approximation. Pass-through measured separately. Disabled table has null penetration metrics, since passing through is intentional.',
        limitations=['All bodies including bottles are boxes; no water/fluid simulation.','Training uses up to six objects; more_objects evaluates nine.','Conservation enforced for graph internal linear pair impulses only, not full angular momentum.']),
        checkpoints={k:str(v) for k,v in checkpoints.items()},models=output)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mlp',type=Path,default=Path('/tmp/agatha-ab/mlp.pt'));parser.add_argument('--graph',type=Path,default=Path('/tmp/agatha-ab/graph.pt'))
    parser.add_argument('--output',type=Path,default=Path('models/ab_results.json'));parser.add_argument('--per-suite',type=int,default=30);parser.add_argument('--layout-scenes',type=int,default=100)
    a=parser.parse_args();results=benchmark({'mlp_multistep':a.mlp,'graph_impulse':a.graph},a.per_suite,a.layout_scenes)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))
