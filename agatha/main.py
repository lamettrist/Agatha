from io import BytesIO
import os
from dotenv import load_dotenv
load_dotenv()
import torch
torch.set_num_threads(4)
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from .scene_reconstruction.pipeline import reconstruct
from .storage import Store

app=FastAPI(title='Agatha · Foresight',version='0.1.0')
app.add_middleware(CORSMiddleware,allow_origins=[o.strip() for o in os.getenv('CORS_ORIGINS','http://localhost:8017,http://127.0.0.1:8017,http://localhost:8000,http://127.0.0.1:8000,https://precognition.base44.app').split(',') if o.strip()],allow_methods=['GET','POST'],allow_headers=['*'])
store=Store()

@app.on_event('startup')
def preload_reconstruction():
    if os.getenv('VISION_OFFLINE') != '1' and os.getenv('RECON_PRELOAD','1') == '1':
        from .scene_reconstruction.models import registry
        app.state.reconstruction_models = registry().preload()


def get_record(key: str) -> dict:
    try: return store.get(key)
    except KeyError: raise HTTPException(404,'Scene not found')

from typing import get_args
DEMO_SCENARIOS=('desk','bottle','laptop','soft')

@app.post('/api/scenes/demo')
def demo(scenario: str = 'desk'):
    if scenario not in DEMO_SCENARIOS: raise HTTPException(422,'scenario must be desk, bottle, laptop or soft')
    return store.demo(scenario)

@app.get('/api/scenes/{scene_id}')
def get_scene(scene_id: str): return get_record(scene_id)

@app.post('/api/scenes')
async def from_image(image: UploadFile | None = File(None), table_width_m: float | None = Form(None), distance_reference: str | None = Form(None), file: UploadFile | None = File(None), video: UploadFile | None = File(None)):
    # Keep the original image field while accepting clients that send all media
    # through the established /api/scenes endpoint with a generic file field.
    image=image or file or video
    if image is None: raise HTTPException(422,'Provide a multipart image, file, or video field')
    from pathlib import Path
    suffix=Path(image.filename or '').suffix.lower()
    if (image.content_type or '').startswith('video/') or suffix in ('.mp4','.mov','.m4v','.webm','.avi','.mkv'):
        return await from_video(image,table_width_m)
    if suffix in ('.ply','.obj','.usdz'):
        return await from_scan(image)
    data=await image.read(10*1024*1024+1)
    if len(data)>10*1024*1024: raise HTTPException(413,'Image exceeds 10 MB')
    try:
        im=Image.open(BytesIO(data))
        if im.width*im.height>20_000_000: raise HTTPException(413,'Image exceeds 20 megapixels')
        im.load()
    except (UnidentifiedImageError,OSError,Image.DecompressionBombError):
        raise HTTPException(422,'Upload a valid JPEG/PNG/WebP image')
    from starlette.concurrency import run_in_threadpool
    try:
        reference=None
        if distance_reference:
            import json, math
            reference=json.loads(distance_reference)
            if len(reference)!=3 or len(reference[0])!=2 or len(reference[1])!=2:
                raise ValueError('distance_reference must be [[x1,y1],[x2,y2],meters] in resized image coordinates')
            from PIL import ImageOps
            w,h=ImageOps.exif_transpose(im).size; ratio=min(1,1280/max(w,h))
            if not all(0<=pt[0]<int(w*ratio) and 0<=pt[1]<int(h*ratio) for pt in reference[:2]):
                raise ValueError('Reference points outside resized image')
            if not math.isfinite(float(reference[2])) or reference[2]<=0 or reference[0]==reference[1]:
                raise ValueError('Reference requires distinct points and a positive finite distance')
        scene,report=await run_in_threadpool(reconstruct,im,table_width_m,**({'distance_reference':reference} if reference is not None else {}))
    except (ValueError,TypeError,KeyError,IndexError) as exc: raise HTTPException(422,str(exc))
    return store.add(scene,report)

from .schema import Scene, SimulationRequest
from .physics import simulate
from .world_model import rollout, compare, load_model
from .router import rollout, reference  # routed entry point: box / joint / soft models (docs/router.md)

@app.post('/api/simulate')
def run_simulation(request: SimulationRequest):
    scene=Scene.model_validate(get_record(request.scene_id)['scene'])
    result={'scene_id':request.scene_id,'physics':None,'learned':None,'error':None,'warnings':[]}
    if request.engine in ('both','learned') and any(o.shape!='box' for o in scene.objects):
        result['warnings'].append('Learned dynamics uses bounding-box dimensions; it was not trained on these collision shapes.')
    try:
        if request.engine in ('physics','both'):
            result['physics'],reference_warning=reference(scene,request.actions,request.duration,request.fps)
            if reference_warning: result['warnings'].append(reference_warning)
            if result['physics']: store.record_score(request.scene_id,result['physics']['chaos_score'])
        if request.engine in ('learned','both'):
            try: result['learned']=rollout(scene,request.actions,round(request.duration*request.fps),request.fps,architecture=request.learned_architecture)
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

class LeaderboardRequest(StrictModel):
    name: str = Field(min_length=1,max_length=40)
    scene_id: str
    chaos_score: float = Field(ge=0)

@app.get('/api/leaderboard')
def leaderboard(): return {'entries':store.leaderboard()}

@app.post('/api/leaderboard')
def add_leaderboard_entry(request: LeaderboardRequest):
    get_record(request.scene_id)
    try: return store.add_score(request.name,request.scene_id,request.chaos_score)
    except ValueError as exc: raise HTTPException(422,str(exc))

from .storage import ROOT
METRICS_PATH=ROOT/'models/metrics.json'

@app.get('/api/metrics')
def metrics():
    """Held-out evaluation numbers written by the training/eval scripts (read-only)."""
    import json
    try: return json.loads(METRICS_PATH.read_text(),parse_constant=lambda _: None)  # NaN/Infinity -> null (JSONResponse rejects them)
    except FileNotFoundError: raise HTTPException(404,'models/metrics.json not found: train or evaluate a world model first')
    except ValueError as exc: raise HTTPException(500,f'models/metrics.json is not valid JSON: {exc}')

from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from .storage import ROOT
app.mount('/static',StaticFiles(directory=ROOT/'static'),name='static')

@app.get('/',include_in_schema=False)
def index(): return FileResponse(ROOT/'static/index.html')


@app.get('/api/reconstructions/{run_id}/{filename}')
def reconstruction_asset(run_id: str, filename: str):
    from .scene_reconstruction.pipeline import root_dir
    import re
    if not re.fullmatch(r'[a-f0-9]{64}',run_id): raise HTTPException(404,'Run not found')
    if not re.fullmatch(r'[a-zA-Z0-9_]+\.(jpg|png|json|ply|glb|npy|npz)',filename):
        raise HTTPException(404,'Asset not found')
    path=root_dir()/run_id/filename
    if not path.is_file(): raise HTTPException(404,'Asset not found')
    return FileResponse(path)

class ObjectEdit(StrictModel):
    mass_kg: float | None = Field(default=None,gt=0,le=1000)
    shape: str | None = None
    size: tuple[float,float,float] | None = None

@app.post('/api/scenes/{scene_id}/objects/{object_id}')
def edit_scene_object(scene_id: str, object_id: str, edit: ObjectEdit):
    from .schema import Collision, CollisionPart
    record=get_record(scene_id);scene=Scene.model_validate(record['scene'])
    obj=next((o for o in scene.objects if o.id==object_id),None)
    if obj is None: raise HTTPException(404,'Object not found')
    articulated=bool(scene.joints or scene.soft_bodies)
    if articulated and (edit.shape is not None or edit.size is not None):
        # settle() knows neither joints nor soft bodies: a resize would open joint anchors / leave soft bodies behind.
        raise HTTPException(422,'Shape/size edits are not supported for scenes with joints or soft bodies; only mass_kg can be edited')
    if edit.mass_kg is not None: obj.mass_kg=edit.mass_kg;obj.mass_source='user'
    if articulated:
        # Mass-only edit: keep every pose exactly (no re-settle), so joint anchors stay closed.
        report=dict(record.get('report') or {})
        report['edited_from']=scene_id;report['acceptance_pass']=False
        report['warnings']=[*report.get('warnings',[]),'Manually edited scene (mass only; poses unchanged, not re-settled because the scene has joints or soft bodies).']
        return store.add(scene,report)
    if edit.shape is not None or edit.size is not None:
        shape=edit.shape or obj.shape
        if shape not in ('box','cylinder'): raise HTTPException(422,'Choose box or cylinder for a manual shape')
        size=edit.size or obj.size
        if not all(0<v<=100 for v in size): raise HTTPException(422,'Size must be positive meters')
        obj.size=size;obj.shape=shape;obj.shape_method='user_edited';obj.visual_mesh_url=None
        obj.collision=Collision(parts=[CollisionPart(size=size)] if shape=='box' else
            [CollisionPart(type='cylinder',radius=size[0]/2,height=size[1])])
    from .scene_reconstruction.bodies import settle,stability
    scene,contacts=settle(scene)
    report=dict(record.get('report') or {})
    report['metrics']={'contacts_after_settle':contacts,'stability':stability(scene),
                       'reprojection_mean_iou':None,'depth_mae_m':None,'coverage_nonplane_pixels':None}
    report['edited_from']=scene_id;report['acceptance_pass']=False
    report['warnings']=[*report.get('warnings',[]),'Manually edited scene; original image metrics no longer apply.']
    return store.add(scene,report)


@app.post('/api/scenes/video')
async def from_video(file: UploadFile | None = File(None), table_width_m: float | None = Form(None), image: UploadFile | None = File(None), video: UploadFile | None = File(None)):
    from .scene_reconstruction.video import reconstruct_video
    from starlette.concurrency import run_in_threadpool
    import tempfile
    from pathlib import Path
    file=file or image or video
    if file is None:raise HTTPException(422,'Provide a multipart file, image, or video field')
    data=await file.read(100*1024*1024+1)
    if len(data)>100*1024*1024: raise HTTPException(413,'Video exceeds 100 MB')
    with tempfile.TemporaryDirectory(prefix='agatha-video-') as tmp:
        path=Path(tmp)/'input.mp4';path.write_bytes(data)
        try: scene,report=await run_in_threadpool(reconstruct_video,path,table_width_m)
        except (ValueError,RuntimeError) as exc: raise HTTPException(422,str(exc))
    if not scene.objects:
        status=503 if report.get('detector')=='unavailable' else 422
        raise HTTPException(status,detail={
            'message':'No physical bodies were reconstructed from this video. Inspect the report instead of displaying a successful zero-object result.',
            'report':report,
        })
    return store.add(scene,report)

@app.post('/api/scenes/scan')
async def from_scan(file: UploadFile = File(...)):
    from .scene_reconstruction.scan import reconstruct_scan
    from starlette.concurrency import run_in_threadpool
    import tempfile
    from pathlib import Path
    suffix=Path(file.filename or '').suffix.lower()
    if suffix not in ('.ply','.obj','.usdz'):raise HTTPException(422,'Upload PLY, OBJ or USDZ')
    data=await file.read(100*1024*1024+1)
    if len(data)>100*1024*1024:raise HTTPException(413,'Scan exceeds 100 MB')
    with tempfile.TemporaryDirectory(prefix='agatha-scan-') as tmp:
        path=Path(tmp)/('input'+suffix);path.write_bytes(data)
        try:scene,report=await run_in_threadpool(reconstruct_scan,path)
        except (ValueError,RuntimeError,ImportError) as exc:raise HTTPException(422,str(exc))
    return store.add(scene,report)


@app.get('/api/health')
def reconstruction_health():
    from .scene_reconstruction.models import registry
    models=registry()
    return {
        'status':'ok',
        'reconstruction_api_version':2,
        'media_types':['photo','video','ply','obj','usdz'],
        'photo_endpoint':'/api/scenes',
        'video_endpoint':'/api/scenes/video',
        'scan_endpoint':'/api/scenes/scan',
        'objects_field':'scene.objects',
        'loaded_models':list(models.models),
        'model_load_failures':dict(models.failures),
        'cuda_available':torch.cuda.is_available(),
        'demo_scenarios':list(DEMO_SCENARIOS),
        'demo_endpoint':'/api/scenes/demo',
        'learned_architectures':list(get_args(SimulationRequest.model_fields['learned_architecture'].annotation)),
        'default_learned_architecture':'default',
        'experimental_features':{'joints':True,'soft_bodies':True,'video_upload':True},
        'physics_reference':{'box_scenes':True,'joints':False,'soft_bodies':False},
        'docs':{'api_reference':'docs/api.md','frontend_guide':'docs/base44-integration.md','examples':'docs/api-examples.json','router':'docs/router.md'},
    }
