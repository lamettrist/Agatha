#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
API_BASE="${API_BASE:-http://127.0.0.1:8017}"
"$PYTHON_BIN" - "$API_BASE" <<'PY'
import json,sys,urllib.request
base=sys.argv[1].rstrip('/')
def post(path,payload):
    request=urllib.request.Request(base+path,data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(request,timeout=60) as response: return json.load(response)
record=post('/api/scenes/demo',{})
print('Loaded offline desk:',record['scene_id'])
actions=[dict(type='push',id='mug',direction=[1,0],impulse_Ns=.55)]
result=post('/api/simulate',dict(scene_id=record['scene_id'],actions=actions,engine='both'))
for engine in ('physics','learned'):
    run=result.get(engine)
    if run:
        print(json.dumps(dict(engine=engine,frames=len(run['frames']),fell=run['events']['fell'],tipped=run['events']['tipped'],chaos_score=run['chaos_score']),indent=2))
    else: print(engine,'unavailable:',result['warnings'])
if result['error']:
    print('Mean position error (m):',result['error']['mean_position_error_m'])
    print('Final position error (m):',result['error']['final_position_error_m'])
    print('Fall agreement:',result['error']['fell_agreement'])
assert result['physics']['events']['fell']==['mug'], 'Offline mug push regression'
assert result['learned'], 'Learned checkpoint missing/unavailable'
print('Demo OK. Estimates and box approximations; not a guaranteed real-world outcome.')
PY
