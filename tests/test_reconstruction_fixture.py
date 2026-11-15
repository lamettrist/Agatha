"""Opt-in real acceptance run. No mocked model output is accepted here."""
import json,os
from pathlib import Path
import pytest
from PIL import Image

@pytest.mark.skipif(os.getenv('RUN_RECONSTRUCTION_FIXTURE')!='1',reason='real GPU fixture; enable RUN_RECONSTRUCTION_FIXTURE=1')
def test_real_desk_acceptance():
    from agatha.scene_reconstruction.pipeline import reconstruct
    from agatha.scene_reconstruction.models import registry,canonical
    from agatha.physics import simulate
    from agatha.schema import Push
    path=Path('tests/images/acceptance.jpg')
    assert path.exists(),'Acceptance photo must be supplied; no synthetic replacement'
    registry().preload()
    scene,report=reconstruct(Image.open(path),use_cache=False)
    requirements={'bottle':4,'tumbler':1,'laptop':4,'backpack':2,'jacket':1,'gloves':1,'mouse':1,'phone':1}
    counts={key:sum(canonical(obj.label)==key for obj in scene.objects if not obj.is_static) for key in requirements}
    missing={key:count-counts[key] for key,count in requirements.items() if counts[key]<count}
    metrics=report.get('metrics',{})
    contacts=metrics.get('contacts_after_settle',{})
    left_bottles=[o for o in scene.objects if 'black' in o.label and 'bottle' in o.label]
    tip=False
    if left_bottles:
        bottle=min(left_bottles,key=lambda o:o.position[0])
        result=simulate(scene,[Push(type='push',id=bottle.id,direction=(1,0),impulse_Ns=.55)],3)
        tip=bottle.id in result['events']['tipped']
    checks=dict(metric_pipeline=report.get('pipeline','').startswith('metric-photo'),inventory=not missing,
        photo_latency=report.get('duration_s',float('inf'))<60,
        silhouette_iou=metrics.get('reprojection_mean_iou',0)>=.75,
        no_floating=contacts.get('float_count',-1)==0,no_penetration=contacts.get('penetration_count',-1)==0,
        stable=metrics.get('stability',{}).get('max_drift_m',float('inf'))<.01,left_bottle_tips=tip)
    report['acceptance_checks']=checks;report['inventory_counts']=counts;report['missing_inventory']=missing
    report['acceptance_pass']=all(checks.values())
    report['metric_size_validation']='Independent measurements absent; standard bottle/laptop size priors are not ground truth.'
    if report.get('run_id'):
        from agatha.scene_reconstruction.pipeline import root_dir
        (root_dir()/report['run_id']/'reconstruction_report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    Path('docs/desk-reconstruction-report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps(report,indent=2,allow_nan=False))
    assert all(checks.values()),json.dumps(checks)
