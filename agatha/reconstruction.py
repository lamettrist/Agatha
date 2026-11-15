"""Single-image box reconstruction. Relative depth is not metric calibration."""
from functools import lru_cache
import math
import os
from pathlib import Path
from threading import Lock
from typing import Any
import numpy as np
from PIL import Image, ImageOps
from .schema import Scene, SceneObject, Support

VISION_LOCK = Lock()
# width/height/depth prior, min/max dimensions, estimated mass in kg.
PRIORS = {
    'laptop': ((.33,.025,.23),(.20,.012,.14),(.45,.08,.34),1.4),
    'cell phone': ((.075,.009,.15),(.05,.006,.10),(.10,.025,.19),.19),
    'cup': ((.085,.11,.085),(.05,.05,.05),(.16,.20,.16),.32),
    'bottle': ((.075,.24,.075),(.04,.10,.04),(.14,.40,.14),.5),
    'book': ((.20,.035,.14),(.09,.015,.07),(.35,.09,.28),.45),
    'bowl': ((.16,.07,.16),(.08,.03,.08),(.30,.16,.30),.35),
    'keyboard': ((.35,.025,.13),(.20,.012,.08),(.50,.05,.23),.65),
    'mouse': ((.065,.04,.11),(.04,.02,.07),(.10,.07,.15),.10),
    'box': ((.13,.17,.13),(.04,.03,.04),(.45,.45,.45),.6),
}

@lru_cache(maxsize=1)
def detector():
    if os.getenv('VISION_OFFLINE') == '1':
        raise RuntimeError('VISION_OFFLINE=1')
    from ultralytics import YOLO
    path = os.getenv('YOLO_MODEL','models/vision/yolov8n.pt')
    if not Path(path).exists():
        raise RuntimeError('YOLO weights missing; run python scripts/prepare_vision.py')
    return YOLO(path)

@lru_cache(maxsize=1)
def depth_estimator():
    if os.getenv('VISION_OFFLINE') == '1':
        raise RuntimeError('VISION_OFFLINE=1')
    from transformers import pipeline
    return pipeline('depth-estimation',model=os.getenv('DEPTH_MODEL','depth-anything/Depth-Anything-V2-Small-hf'),device=-1)

def detect(image: Image.Image) -> list[dict]:
    result = detector().predict(image, conf=.25, verbose=False, device='cpu')[0]
    return [dict(label=result.names[int(b.cls.item())],bbox=[float(v) for v in b.xyxy[0]],confidence=float(b.conf.item()))
            for b in result.boxes]

def relative_depth(image: Image.Image) -> np.ndarray:
    prediction = depth_estimator()(image)['predicted_depth']
    return np.asarray(prediction.squeeze().cpu(),dtype=float)

def fit_plane(points: np.ndarray, iterations: int = 80) -> dict[str, Any]:
    """Deterministic RANSAC in the arbitrary units of a relative point cloud."""
    rng = np.random.default_rng(17)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 30: raise ValueError('Insufficient support pixels')
    if len(points)>2000: points=points[rng.choice(len(points),2000,replace=False)]
    best = np.zeros(len(points), dtype=bool)
    threshold = .035 * float(np.median(np.linalg.norm(points,axis=1)))
    for _ in range(iterations):
        a,b,c=points[rng.choice(len(points),3,replace=False)]
        n=np.cross(b-a,c-a); norm=np.linalg.norm(n)
        if norm<1e-8: continue
        n/=norm
        # Support must have some vertical camera component (camera looks down).
        if abs(n[1])<.20: continue
        mask=np.abs((points-a)@n)<threshold
        if mask.sum()>best.sum(): best=mask
    if best.sum()<max(25,len(points)*.15): raise ValueError('No reliable support plane')
    center=points[best].mean(axis=0)
    _,_,vh=np.linalg.svd(points[best]-center,full_matrices=False)
    n=vh[-1]
    if n[1]>0: n=-n
    x=np.array([1.,0,0]); x-=x.dot(n)*n; x/=np.linalg.norm(x)
    z=np.cross(x,n)
    return dict(normal=n,center=center,x=x,z=z,inlier_ratio=float(best.mean()))

def backproject(depth: np.ndarray, width: int, height: int) -> np.ndarray:
    # DA V2 output is inverse-relative depth. This monotonic mapping is an
    # explicit approximation; it cannot recover true camera distance.
    lo,hi=np.percentile(depth,[3,97])
    if hi-lo<1e-7: raise ValueError('Depth map is constant')
    inv=np.clip((depth-lo)/(hi-lo),0,1)
    distance=1/(.4+inv)
    yy,xx=np.indices(depth.shape)
    u=xx/(depth.shape[1]-1)*width; v=yy/(depth.shape[0]-1)*height
    focal=width/(2*math.tan(math.radians(60)/2))
    return np.stack(((u-width/2)/focal*distance,(v-height/2)/focal*distance,distance),axis=-1)

def reconstruct(image: Image.Image, table_width_m: float | None = None) -> tuple[Scene, dict]:
    image=ImageOps.exif_transpose(image).convert('RGB')
    image.thumbnail((1280,1280))
    w,h=image.size
    warnings=[]
    with VISION_LOCK:
        try:
            detections=detect(image)
            detector_status='yolo'
        except Exception as exc:
            detections=[]; detector_status='unavailable'
            warnings.append(f'YOLO unavailable ({type(exc).__name__}); no detections fabricated. Use offline demo scene.')
        cloud=plane=None
        if detections:
            try:
                depth=relative_depth(image)
                cloud=backproject(depth,w,h)
                mask=np.indices(depth.shape)[0] > depth.shape[0]*.3
                for det in detections:
                    if det['label'] in ('dining table',): continue
                    x1,y1,x2,y2=det['bbox']
                    mask[int(y1/h*mask.shape[0]):int(y2/h*mask.shape[0])+1,
                         int(x1/w*mask.shape[1]):int(x2/w*mask.shape[1])+1]=False
                plane=fit_plane(cloud[mask])
            except Exception as exc:
                cloud=plane=None
                warnings.append(f'Depth/plane unavailable ({type(exc).__name__}); bbox-only flat-table placement.')
    kept=[d for d in detections if d['label'] in PRIORS][:30]
    scale_confidence='low'
    table_width=1.2
    scale_basis='default 1.2 m support width'
    if table_width_m is not None:
        if not .3<=table_width_m<=3: raise ValueError('table_width_m must be between 0.3 and 3')
        table_width=table_width_m; scale_confidence='medium'; scale_basis='user table width reference'
    else:
        known=next((d for d in kept if d['label'] in ('cell phone','laptop')),None)
        if known:
            span=max(known['bbox'][2]-known['bbox'][0],known['bbox'][3]-known['bbox'][1])
            reference=.15 if known['label']=='cell phone' else .33
            table_width=float(np.clip(reference*w/max(span,1),.5,2.5))
            scale_confidence='medium'; scale_basis=f"assumed {known['label']} longest extent {reference} m"
    support=Support(size=(table_width,table_width*2/3),height=.75)
    objects=[]
    unit_scale=table_width
    if plane is not None:
        projection=(cloud.reshape(-1,3)-plane['center'])@plane['x']
        unit_scale=table_width/max(float(np.percentile(projection,95)-np.percentile(projection,5)),.1)
    for i,d in enumerate(kept):
        prior,minimum,maximum,mass=PRIORS[d['label']]
        x1,y1,x2,y2=d['bbox']; cx=(x1+x2)/2; by=y2
        bw=(x2-x1)/w*table_width
        bh=(y2-y1)/h*table_width*.6
        if cloud is not None:
            row=int(np.clip((y1+y2)/2/h*(cloud.shape[0]-1),0,cloud.shape[0]-1))
            left=int(np.clip(x1/w*(cloud.shape[1]-1),0,cloud.shape[1]-1))
            right=int(np.clip(x2/w*(cloud.shape[1]-1),0,cloud.shape[1]-1))
            bw=float(np.linalg.norm(cloud[row,right]-cloud[row,left])*unit_scale)
            distance=float(cloud[row,(left+right)//2,2])
            focal=w/(2*math.tan(math.radians(60)/2))
            bh=(y2-y1)/focal*distance*unit_scale
        # Thin horizontal objects use a class thickness prior; all are labeled estimates.
        height=prior[1] if d['label'] in ('laptop','cell phone','keyboard','book','mouse') else bh
        dims=np.clip([bw,height,bw*prior[2]/prior[0]],minimum,maximum)
        if plane is not None:
            yy=int(np.clip(by/h*(cloud.shape[0]-1),0,cloud.shape[0]-1))
            xx=int(np.clip(cx/w*(cloud.shape[1]-1),0,cloud.shape[1]-1))
            point=cloud[yy,xx]-plane['center']
            x=float(point@plane['x']*unit_scale); z=float(point@plane['z']*unit_scale)
        else:
            x=(cx/w-.5)*table_width
            z=(by/h-.5)*support.size[1]
        x=float(np.clip(x,-table_width/2+dims[0]/2,table_width/2-dims[0]/2))
        z=float(np.clip(z,-support.size[1]/2+dims[2]/2,support.size[1]/2-dims[2]/2))
        # Resolve overlaps conservatively by stacking, never creating interpenetrating boxes.
        bottom=support.height
        for old in objects:
            if abs(x-old.position[0])<(dims[0]+old.size[0])/2 and abs(z-old.position[2])<(dims[2]+old.size[2])/2:
                bottom=max(bottom,old.position[1]+old.size[1]/2)
        objects.append(SceneObject(id=f'obj_{i}',label=d['label'],size=tuple(dims),
            position=(x,bottom+dims[1]/2,z),mass_kg=mass,confidence=d['confidence']))
        d['object_id']=f'obj_{i}'
    report=dict(detected_count=len(detections),kept_count=len(objects),scale_confidence=scale_confidence,
        scale_basis=scale_basis,detector=detector_status,depth='relative_depth_ransac' if plane else 'flat_table_fallback',
        image_size=[w,h],detections=detections,warnings=warnings,
        estimated=['all box dimensions','object positions and rotations','all masses (per-class priors)',
                   'camera intrinsics (60 degree FOV)','support height 0.75 m','relative depth to distance mapping',
                   'support depth and photo occlusions'],
        support_plane=None if plane is None else dict(normal_camera=plane['normal'].tolist(),
            center_camera=plane['center'].tolist(),inlier_ratio=plane['inlier_ratio']),
        disclaimer='Approximate monocular reconstruction; boxes and masses are estimates, not measured geometry.')
    return Scene(objects=objects,support=support),report
