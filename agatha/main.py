from io import BytesIO
import os
from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from .reconstruction import reconstruct
from .storage import Store

app=FastAPI(title='Agatha · Foresight',version='0.1.0')
app.add_middleware(CORSMiddleware,allow_origins=[o.strip() for o in os.getenv('CORS_ORIGINS','http://localhost:8000,http://127.0.0.1:8000').split(',') if o.strip()],allow_methods=['GET','POST'],allow_headers=['*'])
store=Store()

def get_record(key: str) -> dict:
    try: return store.get(key)
    except KeyError: raise HTTPException(404,'Scene not found')

@app.post('/api/scenes/demo')
def demo(): return store.demo()

@app.get('/api/scenes/{scene_id}')
def get_scene(scene_id: str): return get_record(scene_id)

@app.post('/api/scenes')
async def from_image(image: UploadFile = File(...), table_width_m: float | None = Form(None)):
    data=await image.read(10*1024*1024+1)
    if len(data)>10*1024*1024: raise HTTPException(413,'Image exceeds 10 MB')
    try:
        im=Image.open(BytesIO(data))
        if im.width*im.height>20_000_000: raise HTTPException(413,'Image exceeds 20 megapixels')
        im.load()
    except (UnidentifiedImageError,OSError,Image.DecompressionBombError):
        raise HTTPException(422,'Upload a valid JPEG/PNG/WebP image')
    from starlette.concurrency import run_in_threadpool
    try: scene,report=await run_in_threadpool(reconstruct,im,table_width_m)
    except ValueError as exc: raise HTTPException(422,str(exc))
    return store.add(scene,report)

from .schema import Scene, SimulationRequest
from .physics import simulate
from .world_model import rollout, compare

@app.post('/api/simulate')
def run_simulation(request: SimulationRequest):
    scene=Scene.model_validate(get_record(request.scene_id)['scene'])
    result={'scene_id':request.scene_id,'physics':None,'learned':None,'error':None,'warnings':[]}
    try:
        if request.engine in ('physics','both'):
            result['physics']=simulate(scene,request.actions,request.duration,request.fps)
        if request.engine in ('learned','both'):
            try: result['learned']=rollout(scene,request.actions,round(request.duration*request.fps),request.fps)
            except (FileNotFoundError,RuntimeError) as exc:
                if request.engine=='learned': raise HTTPException(503,str(exc))
                result['warnings'].append(f'Learned model unavailable: {exc}')
        if result['physics'] and result['learned']:
            result['error']=compare(result['physics'],result['learned'])
    except ValueError as exc: raise HTTPException(422,str(exc))
    return result

from pydantic import Field
from .schema import StrictModel
from .assistant import Assistant
assistant=Assistant(store)

class AssistantRequest(StrictModel):
    scene_id: str
    question: str = Field(min_length=1,max_length=2000)

@app.post('/api/assistant/query')
def assistant_query(request: AssistantRequest):
    get_record(request.scene_id)
    return assistant.query(request.scene_id,request.question)
