"""Record real endpoint request/response excerpts for Base44 integration."""
import json
from pathlib import Path
import sys
import tempfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient
from agatha import main
from agatha.storage import Store
from agatha.assistant import Assistant

def capture():
    with tempfile.TemporaryDirectory() as directory:
        main.store=Store(directory); main.assistant=Assistant(main.store)
        client=TestClient(main.app)
        scene=client.post('/api/scenes/demo',json={}).json(); key=scene['scene_id']
        request=dict(scene_id=key,actions=[dict(type='push',id='mug',direction=[1,0],impulse_Ns=.55)],engine='both',duration=3,fps=30)
        response=client.post('/api/simulate',json=request).json()
        for engine in ('physics','learned'):
            result=response[engine]
            if result:
                result['frame_count']=len(result['frames']);result['frames']=[result['frames'][0],result['frames'][-1]]
        response['error']['per_frame']=[response['error']['per_frame'][0],response['error']['per_frame'][-1]]
        assistant_request=dict(scene_id=key,question='What if I push the mug?')
        # Avoid billable requests in a documentation script: capture the honest no-key mode.
        import os
        previous=os.environ.pop('OPENAI_API_KEY',None)
        try: assistant_response=client.post('/api/assistant/query',json=assistant_request).json()
        finally:
            if previous is not None: os.environ['OPENAI_API_KEY']=previous
        score_request=dict(name='Example team',scene_id=key,chaos_score=response['physics']['chaos_score'])
        score_response=client.post('/api/leaderboard',json=score_request).json()
        examples=dict(note='Actual local endpoint executions. Simulation frames/per-frame error are excerpted to first and last; frame_count records actual count. Scene IDs belong to this isolated capture, not the live server.',
            demo=dict(method='POST',path='/api/scenes/demo',request={},response=scene),
            get_scene=dict(method='GET',path='/api/scenes/'+key,response=client.get('/api/scenes/'+key).json()),
            simulate=dict(method='POST',path='/api/simulate',request=request,response_excerpt=response),
            assistant=dict(method='POST',path='/api/assistant/query',request=assistant_request,response=assistant_response),
            leaderboard_post=dict(method='POST',path='/api/leaderboard',request=score_request,response=score_response),
            leaderboard_get=dict(method='GET',path='/api/leaderboard',response=client.get('/api/leaderboard').json()))
        photo=Path('data/validation_laptop.jpg')
        if photo.exists():
            upload=client.post('/api/scenes',files={'image':('laptop.jpg',photo.read_bytes(),'image/jpeg')},data={'table_width_m':'1.2'})
            examples['photo']=dict(method='POST',path='/api/scenes',request_multipart=dict(image='real laptop photograph',table_width_m='1.2'),response=upload.json())
        Path('docs/api-examples.json').write_text(json.dumps(examples,indent=2,allow_nan=False))
        print('Saved actual request/response examples to docs/api-examples.json')
if __name__=='__main__': capture()
