"""Paired contact-refinement benchmarks on the unchanged stress set and new cases."""
import argparse,hashlib,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pybullet as pb
from agatha.world_model import LearnedModel,rollout
from agatha.physics import simulate
from prepare_contact_data import contact_case
from benchmark_ab import benchmark,geometry


def series(scene,result) -> dict[str,list]:
    tilt=result.get('support_rotation_euler',[0,0,0]);r=np.asarray(pb.getMatrixFromQuaternion(pb.getQuaternionFromEuler(tilt))).reshape(3,3)
    center=np.array([0,scene.support.height-.04,0]);lookup={o.id:o for o in scene.objects};values={}
    for f in result['frames']:
        for s in f['objects']:
            o=lookup[s['id']]
            if o.is_static:continue
            local=r.T@(np.asarray(s['position'])-center)
            rot=np.asarray(pb.getMatrixFromQuaternion(s['quaternion'])).reshape(3,3)
            half=np.abs(r.T@rot)@np.asarray(o.size)/2
            interior=abs(local[0])+half[0]<scene.support.size[0]/2-.005 and abs(local[2])+half[2]<scene.support.size[1]/2-.005
            values.setdefault(o.id,[]).append(dict(t=f['t'],gap=float(local[1]-.04-half[1]),
                normal_velocity=float((r.T@np.asarray(s['velocity']))[1]),interior=bool(interior),
                speed=float(np.linalg.norm(s['velocity'])),spin=float(np.linalg.norm(s['angular_velocity'])),position=s['position']))
    return values


def touchdown(samples: list[dict]) -> float | None:
    airborne=False
    for sample in samples:
        airborne|=sample['gap']>.03
        if airborne and sample['interior'] and -.01<=sample['gap']<=.015 and abs(sample['normal_velocity'])<.25:
            return sample['t']
    return None


def landings(checkpoints: dict[str,Path],count: int=200,seed: int=882107) -> dict:
    models={k:LearnedModel(p) for k,p in checkpoints.items()};rng=np.random.default_rng(seed)
    kinds=np.repeat(['landing_toss','drop','rest','edge'],count//4);rng.shuffle(kinds)
    output={k:{} for k in models}
    for index,kind in enumerate(kinds):
        scene,actions=contact_case(rng,str(kind));reference=simulate(scene,actions,3,20);truth=series(scene,reference)
        for name,model in models.items():
            result=rollout(scene,actions,60,20,model);predictions=series(scene,result);geo=geometry(scene,result)
            a=output[name].setdefault(str(kind),dict(errors={str(t):[] for t in (.5,1.,2.,3.)},eligible_contacts=0,settled_contacts=0,
                contact_gap_errors=[],normal_velocity_errors=[],touchdown_errors=[],missed_touchdowns=0,reference_touchdowns=0,
                crossings=0,penetration_frames=0,eligible_frames=0,scenes=0))
            for id,samples in truth.items():
                actual=predictions[id]
                for t in (.5,1.,2.,3.):
                    a['errors'][str(t)].append(float(np.linalg.norm(np.asarray(actual[round(t*20)]['position'])-samples[round(t*20)]['position'])))
                expected=samples[-1];pred=actual[-1]
                if expected['interior'] and abs(expected['gap'])<.01 and expected['speed']<.1:
                    a['eligible_contacts']+=1
                    a['contact_gap_errors'].append(abs(pred['gap']-expected['gap']))
                    a['normal_velocity_errors'].append(abs(pred['normal_velocity']-expected['normal_velocity']))
                    a['settled_contacts']+=bool(pred['interior'] and abs(pred['gap']-expected['gap'])<=.01 and abs(pred['normal_velocity'])<=.15 and pred['spin']<=.5)
                t_true=touchdown(samples);t_pred=touchdown(actual)
                if t_true is not None:
                    a['reference_touchdowns']+=1
                    if t_pred is None:a['missed_touchdowns']+=1
                    else:a['touchdown_errors'].append(abs(t_true-t_pred))
            a['crossings']+=len(geo['passed_through_ids']);a['penetration_frames']+=geo['penetration_object_frames'];a['eligible_frames']+=geo['eligible_object_frames'];a['scenes']+=1
        if (index+1)%25==0:print(f'contact held-out cases {index+1}/{count}',flush=True)
    for name,suites in output.items():
        for kind,a in suites.items():
            a['mean_position_error_m']={t:float(np.mean(e)) for t,e in a.pop('errors').items()}
            for field in ('contact_gap_errors','normal_velocity_errors','touchdown_errors'):
                data=a.pop(field);a['mean_'+field]=float(np.mean(data)) if data else None
            a['settled_contact_fraction']=a['settled_contacts']/a['eligible_contacts'] if a['eligible_contacts'] else None
            a['table_penetration_frame_fraction']=a['penetration_frames']/a['eligible_frames'] if a['eligible_frames'] else None
    return dict(seed=seed,cases=count,horizon_s=3,models=output,
        definitions=dict(contact_eligible='Teacher at 3s: full box footprint inside table, bottom gap <1cm, speed <0.1m/s',
            settled='Predicted interior, bottom gap error <=1cm, absolute normal speed <=0.15m/s, angular speed <=0.5rad/s',
            touchdown='After rising >3cm above support: first interior frame with gap [-1cm,1.5cm] and |normal speed|<0.25m/s',
            note='Independent generated cases, never used for gradient updates/checkpoint selection. Geometric penetration diagnostic is approximate.'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--before',type=Path,default=Path('/tmp/agatha-contact-baseline/world_model.pt'))
    p.add_argument('--after',type=Path,default=Path('/tmp/agatha-contact-candidate/world_model.pt'));p.add_argument('--output',type=Path,default=Path('models/contact_results.json'))
    p.add_argument('--skip-stress',action='store_true');p.add_argument('--skip-landings',action='store_true')
    args=p.parse_args();checkpoints={'before':args.before,'contact_refined':args.after}
    results=dict(checkpoint_sha256={k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in checkpoints.items()})
    if not args.skip_stress:results['stress']=benchmark(checkpoints)
    if not args.skip_landings:results['landings']=landings(checkpoints)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))
