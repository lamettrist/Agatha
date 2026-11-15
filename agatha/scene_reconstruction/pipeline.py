"""Photo -> metric cloud -> instances -> separate physical bodies -> measured report."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from hashlib import sha256
import json
import os
from pathlib import Path
from threading import RLock
import time
import cv2
import numpy as np
from PIL import Image,ImageOps
from agatha.schema import Scene,SceneObject,Support,Collision,CollisionPart
from . import geometry as g
from .models import registry,MODEL_IDS,PLANE_LABELS
from .segmentation import deduplicate,split_depth_clusters
from .bodies import settle,stability
from .assets import write_debug,export_visual

VERSION='metric-photo-v17'
RUN_LOCK=RLock()
# kg, friction, rigid approximation, dimensions used ONLY when geometry fails.
PRIORS={
    'bottle':(.55,.8,False,[.07,.21,.07]), 'tumbler':(.35,.65,False,[.085,.18,.085]),
    'cup':(.3,.6,False,[.08,.11,.08]), 'can':(.27,.6,False,[.053,.134,.053]),
    'laptop':(1.8,.65,False,[.31,.02,.22]),'phone':(.19,.65,False,[.075,.009,.15]),
    'backpack':(1.5,.9,True,[.30,.42,.16]),'bag':(1.2,.9,True,[.3,.25,.18]),
    'jacket':(.8,1.,True,[.35,.08,.25]),'gloves':(.12,1.,True,[.12,.025,.20]),
    'mouse':(.10,.65,False,[.065,.035,.11]),'keyboard':(.6,.6,False,[.35,.025,.13]),
    'book':(.5,.7,False,[.2,.03,.14]),'paper':(.005,.6,False,[.216,.002,.2794]),
    'monitor':(3.,.6,False,[.5,.35,.12]),'chair':(6.,.8,False,[.45,.8,.45]),
    'person':(70.,.8,True,[.45,1.7,.3]),
    'plant':(1.,.8,True,[.2,.3,.2]),'box':(.5,.6,False,[.15,.15,.15]),
    'unknown object':(.5,.7,False,[.12,.12,.12]),
}


def root_dir():
    from agatha.storage import ROOT
    return Path(os.getenv('RECON_RUN_DIR',ROOT/'data'/'reconstructions'))


@contextmanager
def timed(report,stage):
    start=time.perf_counter(); record={'status':'running'};report['stages'][stage]=record
    try:
        yield record
        if record['status']=='running': record['status']='ok'
    except Exception as exc:
        record.update(status='failed',reason=f'{type(exc).__name__}: {str(exc)[:400]}')
        raise
    finally: record['duration_s']=round(time.perf_counter()-start,4)


def safe_report_instances(instances):
    return [{k:v for k,v in item.items() if k!='mask'} for item in instances]


FLOOR_CLASSES=('chair','person')
UNSUPPORTED='unsupported'
# Video bodies outside these measured ranges are rejected, not clamped to a template:
# label -> (min height, max height, max horizontal extent), metres, robust extents.
VIDEO_LIMITS={
    'bottle':(.08,.36,.16),'tumbler':(.06,.30,.16),'cup':(.04,.20,.16),'can':(.06,.20,.12),
    'laptop':(.0,.40,.50),'keyboard':(.0,.10,.55),'phone':(.0,.20,.20),'mouse':(.0,.10,.15),
    'chair':(.55,1.35,.90),'person':(.8,2.1,1.0),'backpack':(.05,.70,.70),'bag':(.05,.70,.70),
    'jacket':(.0,.9,.9),'monitor':(.15,.70,.90),
}
VIDEO_DEFAULT_LIMIT=(.0,.8,.8)
# Clamp ranges of upright cylinders (diameter, height) once the measurement is accepted.
CYLINDER_RANGES={'bottle':((.05,.11),(.12,.34)),'tumbler':((.06,.11),(.08,.25)),
                 'cup':((.06,.11),(.06,.16)),'can':((.05,.08),(.09,.17))}


class Implausible(ValueError):
    """A measured body that no object of its class can have; the body is dropped."""


def assign_video_supports(objects,support):
    """Choose what each measured body rests on; never default to a slab it is not over.

    A body inside a table footprint with its bottom near that table top rests on it.
    Otherwise it stands on the floor when its bottom is near the floor (or it is
    floor furniture), and is reported UNSUPPORTED so the caller can drop it.
    """
    from .assets import object_mesh,rotation
    tables=[o for o in objects if o.is_static and o.label=='table']
    assigned={}
    for obj in objects:
        if obj.is_static:continue
        vertices=object_mesh(obj).vertices@rotation(obj.rotation_euler).T+obj.position
        bottom=float(vertices[:,1].min());x,_,z=obj.position
        if obj.supported_by=='floor':
            assigned[obj.id]='floor';continue
        candidates=[]
        if abs(x)<=support.size[0]/2+.03 and abs(z)<=support.size[1]/2+.03 and bottom>support.height-.15:
            candidates.append((abs(bottom-support.height),'support'))
        for table in tables:
            local=rotation(table.rotation_euler).T@(np.array(obj.position)-table.position)
            top=table.position[1]+table.size[1]/2
            if abs(local[0])<=table.size[0]/2+.03 and abs(local[2])<=table.size[2]/2+.03 and bottom>top-.15:
                candidates.append((abs(bottom-top),table.id))
        if candidates:obj.supported_by=min(candidates)[1]
        elif bottom<.25:obj.supported_by='floor'
        else:obj.supported_by=UNSUPPORTED
        assigned[obj.id]=obj.supported_by
    return assigned


def chair_collision(w,h,d,seat_height=None):
    """Stool/chair completion: base disc, stem, seat narrower than the envelope, narrow back.

    Local origin is the centre of the w x h x d envelope; the back is on the -Z side.
    """
    seat_height=float(np.clip(.62*h if seat_height is None else seat_height,.35*h,.80*h))
    seat_thickness=.05;base_height=.03;seat_w,seat_d=.8*w,.8*d
    stem=max(seat_height-seat_thickness-base_height,.05)
    back=max(h-seat_height,.05)
    return Collision(parts=[
        CollisionPart(type='cylinder',radius=min(w,d)*.45,height=base_height,position=(0,-h/2+base_height/2,0)),
        CollisionPart(type='cylinder',radius=.03,height=stem,position=(0,-h/2+base_height+stem/2,0)),
        CollisionPart(size=(seat_w,seat_thickness,seat_d),position=(0,-h/2+seat_height-seat_thickness/2,0)),
        CollisionPart(size=(.7*seat_w,back,.04),position=(0,-h/2+seat_height+back/2,-seat_d/2+.02)),
    ])


def robust_extent(points):
    lo,hi=np.percentile(points,[2,98],axis=0)
    return lo,hi


def check_video_limits(label,height,horizontal):
    low,high,width=VIDEO_LIMITS.get(label,VIDEO_DEFAULT_LIMIT)
    if not low<=height<=high or horizontal>width:
        raise Implausible(f'{label} measured {horizontal:.2f} m wide x {height:.2f} m tall, outside {width:.2f} m x {low:.2f}-{high:.2f} m')


def lift_multiview(inst,points,index,support_height):
    """Class-aware body from a multi-view consensus cloud (metric world, Y up)."""
    label=inst['label'];mass,friction,soft,prior=PRIORS.get(label,PRIORS['unknown object'])
    lo,hi=robust_extent(points);extent=hi-lo;extra=None;fallback=None
    if label in CYLINDER_RANGES:
        check_video_limits(label,float(extent[1]),float(max(extent[0],extent[2])))
        (dmin,dmax),(hmin,hmax)=CYLINDER_RANGES[label]
        diameter=float(np.clip(np.sqrt(extent[0]*extent[2]),dmin,dmax))
        # The base stands on the supporting surface; the measured top is the reliable end.
        bottom=support_height if abs(lo[1]-support_height)<.08 else float(lo[1])
        height=float(np.clip(hi[1]-bottom,hmin,hmax));center=np.median(points,0)
        size=np.array([diameter,height,diameter]);position=np.array([center[0],bottom+height/2,center[2]])
        collision=Collision(parts=[CollisionPart(type='cylinder',radius=diameter/2,height=height)])
        shape='cylinder';method='multiview_extent_cylinder';euler=np.zeros(3)
        extra=dict(measured_diameter_m=float(np.sqrt(extent[0]*extent[2])),measured_height_m=float(extent[1]))
    elif label=='chair':
        fit=g.fit_box(points);local=(points-fit['center'])@_yaw(fit['rotation_euler'][1])
        llo,lhi=np.percentile(local,[2,98],axis=0);measured=lhi-llo
        check_video_limits(label,float(hi[1]),float(max(measured[0],measured[2])))
        w=float(np.clip(measured[0],.35,.70));d=float(np.clip(measured[2],.35,.70));h=float(np.clip(hi[1],.60,1.30))
        center=fit['center']+_yaw(fit['rotation_euler'][1])@np.array([(llo[0]+lhi[0])/2,0,(llo[2]+lhi[2])/2])
        # The seat is the densest horizontal band between 35 % and 80 % of the height.
        band=points[(points[:,1]>.35*h)&(points[:,1]<.85*h),1];seat=None
        if len(band)>50:
            histogram,edges=np.histogram(band,bins=max(4,int(.45*h/.03)));seat=float(edges[histogram.argmax()+1])
            if not .45*h<=seat<=.80*h:seat=None  # an occluded seat: use the class proportion
        size=np.array([w,h,d]);position=np.array([center[0],h/2,center[2]])
        collision=chair_collision(w,h,d,seat);shape='compound';method='multiview_stool_compound'
        euler=np.array([0,float(fit['rotation_euler'][1]),0])
        extra=dict(measured_envelope_m=[float(measured[0]),float(hi[1]),float(measured[2])],seat_height_m=seat,
                   completion='base disc, stem, seat and back are a class completion of the measured envelope')
    else:
        fit=g.fit_box(points);local=(points-fit['center'])@_yaw(fit['rotation_euler'][1])
        llo,lhi=np.percentile(local,[2,98],axis=0);size=np.maximum(lhi-llo,.004)
        check_video_limits(label,float(size[1]),float(max(size[0],size[2])))
        position=fit['center']+_yaw(fit['rotation_euler'][1])@((llo+lhi)/2);euler=np.array([0,float(fit['rotation_euler'][1]),0])
        shape='box';method='multiview_gravity_aligned_obb'
        if label=='laptop':
            fit=dict(size=size.copy(),center=position.copy(),rotation_euler=euler)
            shape,collision,method,extra=build_laptop(points,inst,fit);size=fit['size']
            position[1]=float(lo[1]+.009)  # compound origin is the base slab centre
        else:
            if label in ('phone','paper','keyboard') and size[1]>.08:size[1]=prior[1];fallback='thin_object_thickness_prior'
            collision=Collision(parts=[CollisionPart(size=tuple(np.maximum(size,.002)))])
    obj=SceneObject(id=f'object_{index:03}',label=label,confidence=inst['confidence'],shape=shape,collision=collision,
        shape_method=method,size=tuple(np.maximum(size,.002)),position=tuple(map(float,position)),rotation_euler=tuple(map(float,euler)),
        mass_kg=mass,friction=friction,soft_approx=soft,supported_by='floor' if label in FLOOR_CLASSES else 'support',
        linear_damping=.6 if soft else .04,angular_damping=.8 if soft else .04)
    inst['object_id']=obj.id
    return obj,dict(id=obj.id,label=obj.label,detector_label=inst.get('detector_label'),shape_method=method,valid_points=len(points),
                    fallback=fallback,residual_m=None,details=extra)


def _yaw(yaw):
    c,s_=np.cos(yaw),np.sin(yaw)
    return np.array([[c,0,s_],[0,1,0],[-s_,0,c]])


def frame_view(camera_pose,factor,frame_camera,base_camera):
    """Intrinsics and world pose of another video frame, from its pose in first-camera coordinates."""
    world_to_camera=np.array(camera_pose['world_to_camera']);origin=np.array(camera_pose['origin_world'])
    k=np.array(frame_camera['intrinsics']);rotation=np.array(frame_camera['camera_to_world']);center=np.array(frame_camera['center'])
    camera=dict(fx=float(k[0,0]),fy=float(k[1,1]),cx=float(k[0,2]),cy=float(k[1,2]),width=base_camera['width'],height=base_camera['height'])
    return camera,dict(origin_world=(origin+factor*center@world_to_camera).tolist(),world_to_camera=(rotation.T@world_to_camera).tolist())


def consistent_frames(views_world,center,radius=.2):
    """Frames in which a track's points still sit on the fitted body (SAM 2 tracks jump to neighbours)."""
    return [int(frame) for frame,points in views_world
            if len(points)>=12 and float(np.hypot(*(np.median(points,0)-center)[[0,2]]))<=radius]


def _support_top_mask(support,camera,pose,shape,scale):
    """Image region hidden by the table top for a camera above it (occluder for floor furniture)."""
    from .assets import camera_points,project
    mask=np.zeros(shape,np.uint8)
    if support is None or pose['origin_world'][1]<=support.height:return mask>0
    hx,hz=support.size[0]/2,support.size[1]/2;corners=np.array([[-hx,-hz],[hx,-hz],[hx,hz],[-hx,hz]])
    ring=np.concatenate([np.linspace(corners[i],corners[(i+1)%4],30,endpoint=False) for i in range(4)])
    cam=camera_points(np.column_stack([ring[:,0],np.full(len(ring),support.height),ring[:,1]]),pose);ok=cam[:,2]>.1
    if ok.sum()>=3:cv2.fillPoly(mask,[np.int32(np.clip(project(cam[ok],camera)*scale,-10000,10000))],1)
    return mask>0


def orient_chairs(objects,instances,camera,camera_pose,frames_by_id=None,support=None):
    """A near-round stool has no reliable PCA yaw: pick the yaw whose completed silhouette fits the masks.

    frames_by_id: object id -> [(camera, pose, mask)] of the frames in which the track is on the body.
    Scoring one view only mis-orients a stool seen through table legs in the reference frame, so
    candidates (8 yaws x a small horizontal shift) are ranked by their mean IoU over those frames.
    """
    from .refinement import vertices
    from .assets import rotation,camera_points,project
    by_id={inst.get('object_id'):inst for inst in instances};reports=[]
    for obj in objects:
        if obj.shape_method!='multiview_stool_compound' or obj.id not in by_id:continue
        frames=list((frames_by_id or {}).get(obj.id) or [(camera,camera_pose,by_id[obj.id]['mask'],None)])
        if len(frames)>10:frames=[frames[i] for i in np.linspace(0,len(frames)-1,10).round().astype(int)]
        targets=[]
        for cam_k,pose_k,mask_k,others in frames:
            scale=min(1.,360/cam_k['width']);shape=(round(cam_k['height']*scale),round(cam_k['width']*scale))
            small=lambda m:cv2.resize(np.asarray(m).astype(np.uint8),(shape[1],shape[0]),interpolation=cv2.INTER_NEAREST)>0
            target=small(mask_k);hidden=_support_top_mask(support,cam_k,pose_k,shape,scale)
            if others is not None:hidden|=small(others)
            if target.sum()>=30:targets.append((cam_k,pose_k,target,scale,hidden&~target))
        if not targets:continue
        w,h,d=obj.size;seat=obj.collision.parts[2].position[1]+h/2+.025;scores=[]
        parts=[(vertices(part)@rotation(part.rotation_euler).T+part.position) for part in chair_collision(w,h,d,seat).parts]
        shifts=[(0.,0.)] if len(targets)<3 else [(0.,0.),(.08,0.),(-.08,0.),(0.,.08),(0.,-.08)]
        for dx,dz in shifts:
            position=np.array(obj.position)+[dx,0,dz]
            for k in range(8):
                yaw=obj.rotation_euler[1]+k*np.pi/4;rot=rotation([0,yaw,0]);values=[]
                for cam_k,pose_k,target,scale,hidden in targets:
                    mask=np.zeros(target.shape,np.uint8)
                    for part in parts:
                        cam=camera_points(part@rot.T+position,pose_k)
                        if np.any(cam[:,2]<.05):continue
                        cv2.fillConvexPoly(mask,cv2.convexHull(np.int32(np.clip(project(cam,cam_k)*scale,-10000,10000))),1)
                    drawn=(mask>0)&~hidden
                    values.append(float((drawn&target).sum()/max((drawn|target).sum(),1)))
                # A shift must earn its keep: the measured position wins ties.
                scores.append((float(np.mean(values))-(.01 if dx or dz else 0.),yaw,dx,dz,values[0]))
        best=max(scores);obj.rotation_euler=(0.,float((best[1]+np.pi)%(2*np.pi)-np.pi),0.)
        obj.position=(float(obj.position[0]+best[2]),float(obj.position[1]),float(obj.position[2]+best[3]))
        reports.append(dict(id=obj.id,yaw_rad=obj.rotation_euler[1],silhouette_iou_proxy=best[0],frames_scored=len(targets),
            shift_m=[best[2],best[3]],candidates=[sc for sc,_,dx,dz,_ in scores if not dx and not dz]))
    return reports


def build_laptop(points,inst,fit):
    """Fit visible screen plane where possible; hinge convention is explicit."""
    size=fit['size'].copy(); width=float(max(size[0],.08)); depth=float(max(size[2],width*.6))
    # A low vertical extent denotes a closed laptop; never force all laptops open.
    if size[1]<.065 or ('closed' in inst.get('detector_label','') and 'open' not in inst.get('detector_label','')):
        size=np.array([width,.018,depth]);fit['size']=size
        return 'box',Collision(parts=[CollisionPart(size=tuple(size))]),'box_closed_laptop',None
    yaw=float(fit['rotation_euler'][1]); height=float(size[1]); angle=np.deg2rad(105.)
    measured=False
    # Upper mask points generally represent the screen, avoiding the base/table.
    screen_points=points[points[:,1]>np.percentile(points[:,1],45)]
    try:
        plane=g.fit_plane(screen_points,threshold=.008,iterations=80)
        n=plane['normal']; angle=np.arccos(np.clip(abs(n[1]),0,1))
        angle=float(np.clip(angle,np.deg2rad(65),np.deg2rad(120)))
        measured=plane['inlier_ratio']>.6
    except ValueError: pass
    screen_height=height/max(np.sin(angle),.5)
    base_height=.018; screen_thickness=.006
    # Base centered at origin; hinge lies at back edge (-Z). Screen leans back.
    screen_center=[0,base_height/2+screen_height*np.sin(angle)/2,-depth/2+screen_height*np.cos(angle)/2]
    parts=[CollisionPart(size=(width,base_height,depth)),
           CollisionPart(size=(width,screen_height,screen_thickness),position=tuple(screen_center),
                         rotation_euler=(float(np.pi/2-angle),0,0))]
    fit['size']=np.array([width,base_height+screen_height*np.sin(angle),depth+max(0,-screen_height*np.cos(angle))])
    fit['center'][1]=.75+base_height/2
    return 'compound',Collision(parts=parts),'compound_laptop_screen_fit' if measured else 'compound_laptop_template',{
        'hinge_angle_rad':float(angle),'hinge_measured':measured,'base_depth_estimated':True}


def lift(inst,cloud_world,valid,index,support_height,points_override=None):
    label=inst['label'];mass,friction,soft,prior=PRIORS.get(label,PRIORS['unknown object'])
    mask=inst['mask'];points=cloud_world[mask&valid] if points_override is None else points_override
    fallback=None;extra=None
    if len(points)<25:
        # Keep the piece even when black/glossy pixels all fail confidence checks.
        raw=cloud_world[mask];raw=raw[np.isfinite(raw).all(1)]
        center=np.median(raw,0) if len(raw) else np.array([0,support_height,0])
        fit=dict(size=np.array(prior),center=center,rotation_euler=np.zeros(3),method='class_template',residual_m=None)
        fallback='insufficient_high_confidence_depth'
    else:
        try:
            fit=g.fit_cylinder(points) if label in ('bottle','cup','tumbler','can') else g.fit_box(points)
        except ValueError as exc:
            fit=g.fit_box(points); fallback=str(exc)
    size=np.asarray(fit['size'])
    if not np.isfinite(size).all() or max(size)>3:
        size=np.array(prior);fit['size']=size;fallback='unreliable_depth_extent';fit['method']='class_template'
    shape='box';method=fit['method']
    if label in ('bottle','cup','tumbler','can'):
        radius=fit.get('radius',float(np.sqrt(size[0]*size[2])/2))
        # Aspect/radius failure falls back transparently to a class template.
        if not .012<radius<.12 or not .04<size[1]<.65:
            size=np.array(prior);radius=size[0]/2;fallback='cylinder_extent_outside_plausible_range';method='class_template'
        size=np.array([radius*2,size[1],radius*2]);shape='cylinder'
        collision=Collision(parts=[CollisionPart(type='cylinder',radius=float(radius),height=float(size[1]))])
        if 'radius' not in fit: method='symmetry_cylinder_from_obb' if fallback is None else 'template_cylinder'
    elif label=='laptop':
        shape,collision,method,extra=build_laptop(points,inst,fit);size=fit['size']
    elif label=='chair':
        # A solid OBB fills the air between the seat, back and stand and collides
        # with nearby tables. Complete those rigid parts separately in one body.
        w,h,d=map(float,size)
        h=max(h,.55);w=max(w,.25);d=max(d,.25);size=np.array([w,h,d])
        seat_y=h*.10;seat_thickness=min(.045,h*.07)
        stem_height=seat_y+h/2-seat_thickness/2-.025
        collision=Collision(parts=[
            CollisionPart(size=(w,seat_thickness,d),position=(0,seat_y,0)),
            CollisionPart(size=(w,h*.36,.035),position=(0,h*.32,-d*.43)),
            CollisionPart(type='cylinder',radius=min(w,d)*.40,height=.025,position=(0,-h/2+.0125,0)),
            CollisionPart(type='cylinder',radius=min(w,d)*.06,height=stem_height,position=(0,-h/2+.025+stem_height/2,0)),
        ])
        shape='compound';method='compound_chair_from_points'
        extra={'seat_back_and_stand':'class completion of observed bounds'}
    else:
        if label in ('phone','paper','keyboard') and size[1]>.08:
            size[1]=prior[1];fallback='thin_object_thickness_prior'
        collision=Collision(parts=[CollisionPart(size=tuple(np.maximum(size,.002)))])
        if soft: fallback='image_to_3d_unavailable; gravity_aligned_obb';method='gravity_aligned_obb'
    obj=SceneObject(id=f'object_{index:03}',label=inst.get('detector_label',label) if inst.get('detector_label') else label,
        confidence=inst['confidence'],shape=shape,collision=collision,shape_method=method,
        size=tuple(np.maximum(size,.002)),position=tuple(fit['center']),rotation_euler=tuple(fit['rotation_euler']),
        mass_kg=mass,friction=friction,soft_approx=soft,supported_by='support',
        linear_damping=.6 if soft else .04,angular_damping=.8 if soft else .04)
    if label in ('chair','person'):obj.supported_by='floor'
    inst['object_id']=obj.id
    return obj,dict(id=obj.id,label=obj.label,shape_method=method,valid_points=len(points),fallback=fallback,
                    residual_m=fit.get('residual_m'),details=extra)


def reconstruct(image: Image.Image,table_width_m: float | None=None, *, distance_reference=None,
                use_cache=True,geometry_override=None) -> tuple[Scene,dict]:
    if table_width_m is not None and not .3<=table_width_m<=3:
        raise ValueError('table_width_m must be between 0.3 and 3')
    oriented=ImageOps.exif_transpose(image);camera=g.intrinsics(oriented)
    image=oriented.convert('RGB');original_size=image.size;image.thumbnail((1280,1280))
    ratio=image.width/original_size[0]
    for k in ('fx','fy','cx','cy'): camera[k]*=ratio
    camera.update(width=image.width,height=image.height)
    if geometry_override is not None and 'camera' in geometry_override: camera=dict(geometry_override['camera'])
    config=dict(version=VERSION,models=MODEL_IDS,table_width_m=table_width_m,distance_reference=distance_reference,
                mesh_enabled=os.getenv('RECON_MESH','1'),auto_grid=os.getenv('RECON_AUTO_GRID','12'),detection_threshold=os.getenv('RECON_DETECTION_THRESHOLD','.22'))
    digest=sha256(image.tobytes()+json.dumps(config,sort_keys=True).encode()).hexdigest()
    if geometry_override is not None: digest=sha256((digest+geometry_override['hash']).encode()).hexdigest()
    directory=root_dir()/digest;start=time.perf_counter()
    # Requests are serialized for cache atomicity and GPU memory limits; model stages overlap.
    with RUN_LOCK:
        if use_cache and (directory/'reconstruction_report.json').exists() and (directory/'scene.json').exists():
            report=json.loads((directory/'reconstruction_report.json').read_text());report['cache_hit']=True
            report['request_duration_s']=time.perf_counter()-start
            return Scene.model_validate_json((directory/'scene.json').read_text()),report
        report=dict(pipeline=VERSION,input='photo',image_size=list(image.size),cache_hit=False,run_id=digest,
            stages={},warnings=[],estimated=['mass, friction and restitution priors','hidden surfaces','support height .75 m'],
            models=MODEL_IDS,metrics={},debug_url=f'/api/reconstructions/{digest}',objects=[])
        model=registry()
        with timed(report,'input') as stage: stage.update(original_size=list(original_size),camera=camera)
        try:
            with timed(report,'geometry_and_detection') as stage:
                with ThreadPoolExecutor(2) as pool:
                    depth_future=pool.submit(model.depth,image) if geometry_override is None else None
                    detections_future=pool.submit(lambda:geometry_override['detections']) if geometry_override and 'detections' in geometry_override else pool.submit(model.detect,image)
                    raw_depth=depth_future.result() if depth_future else geometry_override['depth']
                    detections=detections_future.result()
                stage.update(depth_model=MODEL_IDS['depth'] if depth_future else geometry_override['method'],
                             detector=MODEL_IDS['detector'],detected_count=len(detections))
        except Exception as exc:
            # Preserve the pre-existing demo/photo path without inventing detections.
            from agatha.reconstruction import reconstruct as legacy
            scene,old=legacy(image,table_width_m)
            old.update(pipeline='legacy_fallback',stages=report['stages'],cache_hit=False,
                       metrics={k:None for k in ['reprojection_mean_iou','depth_mae_m','coverage_nonplane_pixels']},
                       fallback_reason=str(exc)[:500])
            old['warnings'].append('Metric reconstruction unavailable; legacy approximate boxes used.')
            return scene,old
        if not detections and not (geometry_override and any(m['label'] not in PLANE_LABELS for m in geometry_override.get('masks',[]))):
            from agatha.reconstruction import reconstruct as legacy
            scene,old=legacy(image,table_width_m)
            old['pipeline']='legacy_no_detections';old['stages']=report['stages']
            return scene,old
        with timed(report,'depth_cleanup') as stage:
            depth,valid,details=g.clean_depth(raw_depth,np.asarray(image));stage.update(details)
            cloud=g.backproject(depth,camera)
        with timed(report,'segmentation') as stage:
            try: masks=geometry_override['masks'] if geometry_override and 'masks' in geometry_override else model.segment(image,detections)
            except Exception as exc:
                report['warnings'].append(f'SAM2 failed: {type(exc).__name__}: {str(exc)[:250]}; rectangle masks used.')
                masks=[]
                for det in detections:
                    mask=np.zeros(depth.shape,bool);x1,y1,x2,y2=map(int,det['bbox'])
                    mask[max(0,y1):max(0,y2),max(0,x1):max(0,x2)]=True
                    masks.append(dict(det,mask=mask,mask_source='rectangle_fallback',mask_score=None))
                stage['status']='fallback'
            stage['raw_masks']=len(masks)
        try: model.classify(image,[m for m in masks if m['label']=='unknown object'])
        except Exception as exc: report['warnings'].append(f'Automatic-mask classification failed: {type(exc).__name__}: {str(exc)[:200]}')
        with timed(report,'structure') as stage:
            table_regions=[]
            for region in sorted((m for m in masks if m['label']=='table'),key=lambda m:-m['confidence']):
                count,components,stats,_=cv2.connectedComponentsWithStats(region['mask'].astype('uint8'),8)
                for component in range(1,count):
                    if stats[component,cv2.CC_STAT_AREA]<max(100,image.width*image.height*.001):continue
                    mask=components==component
                    if any((mask&old['mask']).sum()/max(min(mask.sum(),old['mask'].sum()),1)>.85 for old in table_regions):continue
                    table_regions.append(dict(region,mask=mask))
            # Disjoint tables must not become one room-sized slab between them.
            primary_table=max(table_regions,key=lambda m:int(m['mask'].sum()),default=None)
            table_masks=[primary_table['mask']] if primary_table is not None else []
            exclude=np.logical_or.reduce([m['mask'] for m in masks if m['label'] not in PLANE_LABELS]) if any(m['label'] not in PLANE_LABELS for m in masks) else np.zeros(depth.shape,bool)
            candidates=(np.logical_or.reduce(table_masks)&valid) if table_masks else ((np.indices(depth.shape)[0]>image.height*.55)&valid&~exclude)
            try:
                plane=g.fit_plane(cloud[candidates],normal_hint=[0,-1,-.5])
            except ValueError:
                plane=g.fit_plane(cloud[valid&~exclude],normal_hint=[0,-1,-.5]);stage['status']='fallback'
                report['warnings'].append('Table mask plane failed; used dominant gravity-compatible plane.')
            depth_normal=plane['normal'].copy()
            try:
                n,vertical_report=g.vertical_gravity(masks,camera)
                plane['normal']=n
                x=np.array([1.,0,0]);x-=x.dot(n)*n;x/=np.linalg.norm(x)
                plane['basis']=np.array([x,n,np.cross(x,n)])
                # Keep metric camera height from the robust table depth consensus.
                distance=float(np.median(cloud[candidates]@n))
                plane['center']=n*distance
                stage['vertical_alignment']=vertical_report
                stage['depth_normal_disagreement_deg']=float(np.degrees(np.arccos(np.clip(n@depth_normal,-1,1))))
            except ValueError as exc: stage['vertical_alignment_fallback']=str(exc)
            basis=plane['basis'];origin=plane['center']
            cloud_world=(cloud-origin)@basis.T
            surface=candidates
            if geometry_override is not None:
                from .structure import cloud_components
                union=np.logical_or.reduce([region['mask'] for region in table_regions]) if table_regions else candidates
                surface=union&valid&(np.abs(cloud_world[:,:,1])<.025)
                if surface.sum()>30:
                    components=cloud_components(cloud[surface])
                    overlap=candidates[surface]
                    chosen=int(np.bincount(components,weights=overlap.astype(float)).argmax())
                    selected=np.zeros(surface.shape,bool);selected[surface]=components==chosen;surface=selected
            extents=np.percentile(cloud_world[surface][:,[0,2]],[1,99],axis=0) if surface.sum()>30 else np.array([[-.6,-.4],[.6,.4]])
            if geometry_override is not None and surface.sum()>30:
                # The world X axis is camera-0's; a table oblique to the first view gets an inflated
                # axis-aligned box, and Support has no yaw. Align the world frame to the table instead.
                from .structure import table_rectangle
                multiview=geometry_override.get('multiview_points')
                rectangle=table_rectangle((multiview[0]-origin)@basis.T if multiview is not None else np.empty((0,3)),
                    multiview[1] if multiview is not None else np.empty(0,int),cloud_world[surface][:,[0,2]])
                if rectangle is not None:
                    n=basis[1];x=np.array([rectangle['axis'][0],0,rectangle['axis'][1]])@basis
                    if x@basis[0]<0:x=-x
                    origin=origin+basis.T@np.array([rectangle['center'][0],0,rectangle['center'][1]])
                    basis=np.array([x,n,np.cross(x,n)]);plane['basis']=basis
                    half=rectangle['size']/2;extents=np.array([-half,half])
                    stage['table_rectangle']=dict(source=rectangle['source'],size_unscaled=rectangle['size'].tolist(),
                        yaw_from_camera_x_deg=float(np.degrees(np.arctan2(rectangle['axis'][1],rectangle['axis'][0]))),
                        seed_size_unscaled=rectangle.get('seed_size'))
            center_xz=extents.mean(0);origin=origin+basis.T@np.array([center_xz[0],0,center_xz[1]])
            cloud_world=(cloud-origin)@basis.T
            plane_mask=valid & (np.abs(cloud_world[:,:,1])<.018)
            if geometry_override is not None:
                # Automatic masks can outline patches of a floor/wall and label
                # them as furniture. Remove only pixels supported by an observed
                # structural region AND its fitted plane, not arbitrary flat items.
                for region in masks:
                    if region['label'] not in PLANE_LABELS:continue
                    observed=region['mask']&valid
                    if observed.sum()<100:continue
                    try:structure_plane=g.fit_plane(cloud_world[observed],threshold=.012)
                    except ValueError:continue
                    near=np.abs((cloud_world-structure_plane['center'])@structure_plane['normal'])<.018
                    plane_mask|=observed&near
            stage.update(normal_camera=plane['normal'].tolist(),inlier_ratio=plane['inlier_ratio'],rms_m=plane['rms_m'],
                         gravity_source='table_plane_normal',world_basis=basis.tolist())
        with timed(report,'instances') as stage:
            instances,dropped=deduplicate(masks,depth,plane_mask)
            # A tracked instance owns one multi-view cloud; splitting its mask would duplicate the body.
            if geometry_override is None:instances=split_depth_clusters(instances,depth)
            try: model.classify(image,instances)
            except Exception as exc: report['warnings'].append(f'CLIP unavailable: {type(exc).__name__}: {str(exc)[:250]}')
            # Reject automatic background and small parts contained within known bodies.
            plane_union=np.logical_or.reduce([m['mask'] for m in masks if m['label'] in PLANE_LABELS]) if any(m['label'] in PLANE_LABELS for m in masks) else plane_mask
            filtered=[]
            for inst in instances:
                if inst['mask_source']=='sam2_automatic':
                    overlap=(inst['mask']&plane_union).sum()/max(inst['mask'].sum(),1)
                    contained=any((inst['mask'][max(0,int(other['bbox'][1])):int(other['bbox'][3]),max(0,int(other['bbox'][0])):int(other['bbox'][2])].sum()/max(inst['mask'].sum(),1))>.90 for other in instances if other is not inst and other['mask_source']=='sam2_box' and other['label'] not in PLANE_LABELS)
                    if overlap>.60 or inst.get('classifier_label') in PLANE_LABELS or inst.get('classifier_confidence',0)<.35 or (contained and inst['label']!='gloves'):
                        dropped.append(dict(label=inst['label'],reason='automatic_structure_mask'));continue
                filtered.append(inst)
            # A keyboard wholly inside an open laptop detection belongs to that laptop.
            for accessory in filtered:
                if accessory['label']=='gloves':
                    for cloth in filtered:
                        if cloth['label']=='jacket':cloth['mask']=cloth['mask']&~accessory['mask']
            instances=[i for i in filtered if not (i['label']=='keyboard' and any(
                o['label']=='laptop' and i['bbox'][0]>=o['bbox'][0]-15 and i['bbox'][2]<=o['bbox'][2]+15
                and i['bbox'][1]>=o['bbox'][1] and i['bbox'][3]<=o['bbox'][3]+15 for o in filtered))]
            stage.update(kept=len(instances),dropped=dropped)
        with timed(report,'scale') as stage:
            references=[]
            for i,inst in enumerate(instances):
                if inst['label'] not in g.REFERENCES: continue
                # Tall insulated bottles and occluded laptops are unreliable standard references.
                if inst['label']=='bottle' and 'plastic' not in inst.get('detector_label',''): continue
                if min(inst['bbox'][:2])<2 or inst['bbox'][2]>image.width-2 or inst['bbox'][3]>image.height-2: continue
                try:
                    fit=g.fit_box(cloud_world[inst['mask']&valid]);target,axis=g.REFERENCES[inst['label']]
                    observed=float(fit['size'][axis]) if axis>=0 else float(max(fit['size']))
                    references.append(dict(id=i,label=inst['label'],reference_m=target,observed_m=observed))
                except ValueError: pass
            # Standard-bottle silhouette + gravity calibrates camera height directly.
            calibration=None
            try:
                camera_height,calibration=g.reference_camera_height(instances,camera,plane['normal'])
                raw_height=abs(float(origin@plane['normal']))
                references=[dict(id='camera_height_from_bottles',label='bottle_silhouette',reference_m=camera_height,observed_m=raw_height)]
                stage['bottle_camera_height']=calibration
            except ValueError: pass
            visible_width=float(extents[1,0]-extents[0,0])
            manual=table_width_m/max(visible_width,.001) if table_width_m else None
            if distance_reference is not None:
                a,b,meters=distance_reference
                pa=cloud[int(a[1]),int(a[0])];pb=cloud[int(b[1]),int(b[0])]
                manual=float(meters)/np.linalg.norm(pa-pb)
            prior_only=False
            if geometry_override is not None and manual is None and calibration is None:
                # VGGT units are not metres. Without a bottle reference a hand-held camera
                # height is a far better guess than factor 1.0, and is reported as such.
                references=[];raw_camera=float(((-origin)@basis.T)[1]);floor_raw=None
                floor_pixels=[m['mask'] for m in masks if m['label']=='floor']
                if floor_pixels:
                    try:floor_raw=float(g.fit_plane(cloud_world[np.logical_or.reduce(floor_pixels)&valid],normal_hint=[0,1,0])['center'][1])
                    except ValueError:pass
                if floor_raw is not None and raw_camera-floor_raw>1e-3:
                    references=[dict(id='camera_height_prior',label='handheld_camera_height',reference_m=1.45,observed_m=raw_camera-floor_raw)]
                    prior_only=True
                    report['warnings'].append('No metric reference object found; scene scale assumes a 1.45 m hand-held camera height.')
                else:report['warnings'].append('No metric reference found; scene scale is unverified.')
            scale=g.solve_scale(references,manual);stage.update(scale);stage['prior_only']=prior_only
            factor=scale['factor'];cloud_world*=factor;depth*=factor
            support_size=np.maximum((extents[1]-extents[0])*factor,[.3,.3])
            if np.max(support_size)>10: raise ValueError('Implausible support dimensions; provide a scale reference')
            cloud_world[:,:,1]+=.75
            camera_pose=dict(origin_world=((-origin)@basis.T*factor+[0,.75,0]).tolist(),world_to_camera=basis.T.tolist())
        support_height=.75
        floor_masks=[m['mask'] for m in masks if m['label']=='floor']
        if geometry_override is not None and not floor_masks:
            lower=np.indices(depth.shape)[0]>image.height*.55
            occupied=np.logical_or.reduce([m['mask'] for m in masks if m['label'] not in PLANE_LABELS]) if any(m['label'] not in PLANE_LABELS for m in masks) else np.zeros(depth.shape,bool)
            floor_masks=[lower&~occupied]
        if floor_masks:
            floor_pixels=np.logical_or.reduce(floor_masks)&valid
            try:
                floor=g.fit_plane(cloud_world[floor_pixels],threshold=.025,normal_hint=[0,1,0])
                floor_y=float(floor['center'][1]);measured_height=.75-floor_y
                if .2<measured_height<3 and abs(floor['normal'][1])>.95:
                    support_height=measured_height;cloud_world[:,:,1]-=floor_y
                    camera_pose['origin_world'][1]-=floor_y
                    report['stages']['structure']['floor_height_correction_m']=floor_y
                    report['stages']['structure']['support_height_m']=support_height
            except ValueError:pass
        support=Support(size=tuple(support_size),height=support_height,thickness=.03)
        with timed(report,'lift') as stage:
            objects=[];fused_world_by_id={};lift_dropped=[];stage['dropped']=lift_dropped
            def to_world(points):return (points-origin)@basis.T*factor+[0,support.height,0]
            track_views=geometry_override.get('track_views') if geometry_override else None
            max_range=float(os.getenv('RECON_VIDEO_MAX_RANGE_M','3.0'))
            for i,inst in enumerate(instances[:240]):
                try:
                    fused=None;consensus=None
                    if track_views is not None and inst.get('track_id') in track_views:
                        from .structure import fuse_views
                        fused,consensus=fuse_views([(frame,to_world(points)) for frame,points in track_views[inst['track_id']]])
                        reason=None
                        if consensus['views_kept']<min(3,consensus['views_observed']) or consensus['views_kept']<2 or consensus['points']<150:
                            reason=f"insufficient_multiview_support ({consensus['views_kept']} consistent views, {consensus['points']} points)"
                        elif float(np.hypot(*np.median(fused,0)[[0,2]]))>max_range:
                            reason=f'beyond_{max_range:g}m_of_main_support'
                        if reason:
                            lift_dropped.append(dict(index=i,track_id=inst.get('track_id'),label=inst['label'],reason=reason));continue
                        try:obj,details=lift_multiview(inst,fused,i,support.height)
                        except Implausible as exc:
                            lift_dropped.append(dict(index=i,track_id=inst.get('track_id'),label=inst['label'],reason=f'implausible_size: {exc}'));continue
                        details['multiview']=consensus
                    else:
                        if geometry_override and inst.get('track_id') in geometry_override.get('fused',{}):
                            fused=to_world(geometry_override['fused'][inst['track_id']])
                        obj,details=lift(inst,cloud_world,valid,i,support.height,points_override=fused)
                    details['fused_multiview_points']=0 if fused is None else len(fused)
                    objects.append(obj);report['objects'].append(details)
                    if fused is not None and len(fused):fused_world_by_id[obj.id]=fused
                except Exception as exc:
                    # A bad piece cannot drop the rest of the scene, but is explicitly counted.
                    report['warnings'].append(f'Object {i} ({inst["label"]}) failed: {type(exc).__name__}: {exc}')
            # Detected walls become static finite slabs fitted from their visible points.
            for j,mask in enumerate([m for m in masks if m['label']=='wall'][:3]):
                pts=cloud_world[mask['mask']&valid]
                if len(pts)<30: continue
                try:
                    wall_plane=g.fit_plane(pts,threshold=.015)
                except ValueError:continue
                if abs(wall_plane['normal'][1])>.35:
                    report['warnings'].append(f'Wall candidate {j} rejected: its observed plane is not vertical.')
                    continue
                if geometry_override is not None:
                    # A mask spanning a pillar corner or two walls is not one slab; a box around
                    # it cuts through the room. Keep only single planes clear of the main table.
                    near=np.abs((pts-wall_plane['center'])@wall_plane['normal'])<.03
                    clearance=abs(float(wall_plane['center']@wall_plane['normal']))-float(np.hypot(*support.size))/2
                    if near.mean()<.8 or clearance<.1:
                        lift_dropped.append(dict(label='wall',reason=f'not_a_single_clear_plane (inliers {near.mean():.2f}, clearance {clearance:.2f} m)'));continue
                    pts=pts[near]
                fitted=g.fit_box(pts);dims=fitted['size'];dims[2]=.03
                objects.append(SceneObject(id=f'wall_{j}',label='wall',size=tuple(dims),position=tuple(fitted['center']),
                    rotation_euler=tuple(fitted['rotation_euler']),is_static=True,shape_method='plane_slab',confidence=mask['confidence']))
            if geometry_override is not None:
                from .structure import table_surfaces
                surfaces=[t for t in table_surfaces(table_regions,cloud_world,valid,support,min_area=.3)
                          if float(np.hypot(t['position'][0],t['position'][2]))<=max_range]
                for j,surface in enumerate(surfaces):
                    details=dict(surface);details.pop('observed_points')
                    objects.append(SceneObject(id=f'table_{j}',label='table',**details,
                        is_static=True,shape_method='connected_plane_slab',confidence=.6))
                stage['secondary_table_surfaces']=surfaces
            stage['body_count']=len(objects)
        # Recenter/expand the slab using observed table bounds and contact anchors.
        # Hidden table extent is an estimate, recorded instead of dropping bodies off its edge.
        contact_points=[]
        by_id={o.id:o for o in objects}
        for inst in instances:
            obj=by_id.get(inst.get('object_id'))
            if obj is None or obj.is_static:continue
            if obj.id in fused_world_by_id and len(fused_world_by_id[obj.id])>=100:
                # Multi-view points already define a world position. Applying a
                # photo contact ray or table-depth clamp here destroys that fit.
                if obj.supported_by!='floor':contact_points.append(np.array(obj.position)[[0,2]])
                continue
            try:
                bottom,top=g.mask_anchor(inst)
                if inst['label']=='laptop':bottom=g.horizontal_mask_anchor(inst)
                direction=g.ray(bottom,camera)
                normal=plane['normal'];offset=float(origin@normal)*factor
                if obj.supported_by=='floor':offset-=support.height
                denom=float(direction@normal)
                if denom>=-.01: continue
                camera_contact=direction*(offset/denom)
                world_contact=camera_contact@basis.T+np.array(camera_pose['origin_world'])
                # Silhouette fitting replaces unreliable dark/glossy depth extents explicitly.
                label=inst['label']
                if label in ('bottle','tumbler','cup','can'):
                    ray_top=g.ray(top,camera)
                    _,height=np.linalg.lstsq(np.column_stack([ray_top,-normal]),camera_contact,rcond=None)[0]
                    yy,xx=np.where(inst['mask']); rows=[]
                    low,hi=np.percentile(yy,[25,85])
                    for row in range(int(low),int(hi)):
                        xs=np.flatnonzero(inst['mask'][row])
                        if len(xs)>4:rows.append(xs[-1]-xs[0])
                    width=float(np.median(rows)) if rows else inst['bbox'][2]-inst['bbox'][0]
                    radius=float(np.clip(width*camera_contact[2]/camera['fx']/2,.012,.12))
                    height=float(np.clip(height,.04,.6))
                    obj.size=(radius*2,height,radius*2)
                    obj.collision=Collision(parts=[CollisionPart(type='cylinder',radius=radius,height=height)])
                    obj.shape_method='silhouette_cylinder_metric_reference'
                    obj.rotation_euler=(0,0,0)
                if label not in ('bottle','tumbler','cup','can'):
                    mass,friction,soft,prior=PRIORS.get(label,PRIORS['unknown object'])
                    # Complete occluded objects with class templates; use observed silhouette
                    # dimensions only where the contact ray is inside the visible table depth.
                    table_camera=cloud[candidates]*factor
                    back_depth=float(np.percentile(table_camera[:,2],98))
                    contact_depth=min(camera_contact[2],back_depth+.08)
                    width_px=inst['bbox'][2]-inst['bbox'][0]
                    projected_width=width_px*contact_depth/camera['fx']
                    height_px=inst['bbox'][3]-inst['bbox'][1]
                    projected_height=height_px*contact_depth/camera['fy']
                    if obj.supported_by!='floor' and camera_contact[2]>back_depth+.08:
                        camera_contact=direction*(back_depth+.08)
                        world_contact=camera_contact@basis.T+np.array(camera_pose['origin_world'])
                    if label in ('backpack','bag','jacket','gloves'):
                        w=float(np.clip(projected_width,prior[0]*.65,prior[0]*1.4))
                        h=float(np.clip(projected_height,prior[1]*.6,prior[1]*1.3))
                        if label=='backpack' and camera_contact[2]>=back_depth+.079:h=prior[1]
                        dims=(w,h,prior[2]);obj.size=dims;obj.rotation_euler=(0,0,0)
                        obj.collision=Collision(parts=[CollisionPart(size=dims)])
                        obj.shape_method='occlusion_completed_box_template'
                    elif label=='laptop':
                        # Known laptop class dimensions regularize glossy depth outliers.
                        width=float(np.clip(projected_width,.28,.38));base_depth=width*.70
                        if ('closed' in inst.get('detector_label','') and 'open' not in inst.get('detector_label','')) or (height_px<150 and 'partially' not in inst.get('detector_label','')):
                            dims=(width,.018,base_depth);obj.size=dims;obj.shape='box'
                            obj.collision=Collision(parts=[CollisionPart(size=dims)]);obj.shape_method='closed_laptop_template'
                        else:
                            sh=width*.62;angle=np.deg2rad(105);obj.shape='compound';obj.size=(width,sh+.018,base_depth+.05)
                            obj.collision=Collision(parts=[CollisionPart(size=(width,.018,base_depth)),
                                CollisionPart(size=(width,sh,.006),position=(0,.009+sh*np.sin(angle)/2,-base_depth/2+sh*np.cos(angle)/2),rotation_euler=(float(np.pi/2-angle),0,0))])
                            obj.shape_method='open_laptop_template_glossy_depth'
                        yaw,axis_details=g.image_axis_yaw(image,inst,[world_contact[0],support.height+.12,world_contact[2]],camera,camera_pose)
                        if 'partially' in inst.get('detector_label',''):
                            try:
                                screen=g.fit_plane(cloud_world[inst['mask']&valid],threshold=.008)
                                normal_screen=screen['normal'];yaw=float(np.arctan2(normal_screen[0],normal_screen[2]))
                                yaw=(yaw+np.pi/2)%np.pi-np.pi/2
                            except ValueError:pass
                        obj.rotation_euler=(0,yaw,0)
                        inst['image_axis_fit']=axis_details
                        world_contact[0]-=np.sin(yaw)*base_depth/2
                        world_contact[2]-=np.cos(yaw)*base_depth/2
                    elif label in ('phone','mouse'):
                        obj.size=tuple(prior);obj.rotation_euler=(0,0,0)
                        obj.collision=Collision(parts=[CollisionPart(size=tuple(prior))]);obj.shape_method='small_object_template'
                obj.position=(float(world_contact[0]),obj.position[1],float(world_contact[2]))
                if obj.supported_by!='floor':contact_points.append(world_contact[[0,2]])
            except (ValueError,IndexError): continue
        if geometry_override is not None:
            assignments=assign_video_supports(objects,support);report['stages']['lift']['support_assignments']=assignments
            # Nothing is under these bodies: placing them on the main table would be an invention.
            for obj in [o for o in objects if o.supported_by==UNSUPPORTED]:
                objects.remove(obj);fused_world_by_id.pop(obj.id,None)
                report['stages']['lift']['dropped'].append(dict(id=obj.id,label=obj.label,reason='no_support_under_body'))
                for inst in instances:
                    if inst.get('object_id')==obj.id:inst.pop('object_id')
            report['objects']=[d for d in report['objects'] if any(o.id==d['id'] for o in objects)]
            frames_by_id={};frame_cameras=geometry_override.get('cameras');frame_masks=geometry_override.get('track_masks') or {}
            details_by_id={d['id']:d for d in report['objects']}
            for inst in instances:
                obj=by_id.get(inst.get('object_id'))
                if obj is None or obj not in objects or track_views is None or inst.get('track_id') not in track_views:continue
                agreeing=consistent_frames([(frame,to_world(points)) for frame,points in track_views[inst['track_id']]],np.array(obj.position))
                if 'multiview' in details_by_id.get(obj.id,{}):details_by_id[obj.id]['multiview']['consistent_frames']=agreeing
                if frame_cameras:
                    # Other tracked objects hide parts of this one; their masks are "cannot be seen here" regions.
                    frames_by_id[obj.id]=[(*frame_view(camera_pose,factor,frame_cameras[frame],camera),frame_masks[(frame,inst['track_id'])],
                        np.logical_or.reduce([np.zeros(depth.shape,bool),*[m for (f,t),m in frame_masks.items() if f==frame and t!=inst['track_id']]]))
                        for frame in agreeing if (frame,inst['track_id']) in frame_masks]
            report['stages']['lift']['chair_orientation']=orient_chairs(objects,instances,camera,camera_pose,frames_by_id,support)
            # Do not stretch the main tabletop to reach furniture on other tables.
            contact_points=[]
        if contact_points:
            pts=np.array(contact_points)
            extent=np.vstack([-support_size/2,support_size/2,pts])
            lo,hi=extent.min(0)-.04,extent.max(0)+.04
            center=(lo+hi)/2;support.size=tuple(np.minimum(hi-lo,10))
            for obj in objects: obj.position=(obj.position[0]-center[0],obj.position[1],obj.position[2]-center[1])
            cloud_world-=np.array([center[0],0,center[1]])
            fused_world_by_id={key:pts-[center[0],0,center[1]] for key,pts in fused_world_by_id.items()}
            camera_pose['origin_world']=(np.array(camera_pose['origin_world'])-[center[0],0,center[1]]).tolist()
            report['estimated'].append('hidden table bounds expanded to include observed object contact rays')
        for details in report['objects']:
            obj=next((o for o in objects if o.id==details['id']),None)
            if obj is not None:
                details['initial_shape_method']=details['shape_method'];details['shape_method']=obj.shape_method
        directory.mkdir(parents=True,exist_ok=True)
        if os.getenv('RECON_REFINE','1')=='1':
            with timed(report,'silhouette_refinement') as stage:
                from .refinement import refine
                stage['objects']=refine(objects,instances,camera,camera_pose,support.height)
        if os.getenv('RECON_MESH','1')=='1':
            with timed(report,'irregular_meshes') as stage:
                from .meshes import generate_objects
                mesh_reports=generate_objects(image,instances,objects,cloud_world,valid,directory,fused_by_id=fused_world_by_id,camera=camera,camera_pose=camera_pose,support_height=support.height)
                stage['objects']=mesh_reports
                stage['workers']=min(7,len(registry().devices))
                for item in mesh_reports:
                    if item['status']=='fallback':report['warnings'].append(f"{item['id']} mesh fallback: {item['reason']}")
        scene=Scene(objects=objects,support=support)
        with timed(report,'settle') as stage:
            scene,contacts=settle(scene,extend_support=geometry_override is None,measured_poses=geometry_override is not None);stage.update(contacts);report['metrics']['contacts_after_settle']=contacts
        with timed(report,'stability') as stage:
            measured=stability(scene);stage.update(measured);report['metrics']['stability']=measured
            if geometry_override is not None and os.getenv('RECON_VIDEO_STATIC_FURNITURE','1')=='1':
                # Stools and chairs were settled and drift-tested as dynamic bodies above. They are
                # published static: a bottle cannot move them, and bodies this large are outside the
                # learned world model's trained range (it drifts them by centimetres with no action).
                frozen=[obj.id for obj in scene.objects if not obj.is_static and obj.label in FLOOR_CLASSES]
                for obj in scene.objects:
                    if obj.id in frozen:obj.is_static=True
                stage['published_static_after_measurement']=frozen
                if frozen:report['warnings'].append(f"Furniture is fixed in place ({', '.join(frozen)}): pushing it has no effect. Set RECON_VIDEO_STATIC_FURNITURE=0 to make it movable.")
        report.update(detected_count=len(detections),kept_count=len(instances),detector='grounding_dino',depth='metric_depth',
            detections=safe_report_instances(instances),scale_confidence='medium' if (scale['references'] and not prior_only) or manual else 'low',
            scale_basis=scale['source'],scale=scale,camera=camera,camera_pose=camera_pose)
        report['body_count']=len(scene.objects)
        report['dynamic_body_count']=sum(not obj.is_static for obj in scene.objects)
        final_objects={obj.id:obj for obj in scene.objects}
        for details in report['objects']:
            obj=final_objects.get(details['id'])
            if obj is not None:details['shape_method']=obj.shape_method
        directory.mkdir(parents=True,exist_ok=True)
        with timed(report,'visual_assets') as stage:
            by_id={i.get('object_id'):i for i in instances}
            for obj in scene.objects:
                if obj.id not in by_id: continue
                try:
                    export_visual(obj,image,by_id[obj.id],camera,camera_pose,directory/(obj.id+'.glb'))
                    obj.visual_mesh_url=report['debug_url']+'/'+obj.id+'.glb'
                except Exception as exc: report['warnings'].append(f'{obj.id} GLB export failed: {type(exc).__name__}: {exc}')
        with timed(report,'debug_and_metrics'):
            write_debug(directory,image,depth,cloud_world,valid,instances,scene,camera,camera_pose,report)
        if geometry_override is None:
            # Same measured gates as the video path, so a client can tell a fitted scene from a guess.
            metrics=report['metrics'];contacts=metrics.get('contacts_after_settle',{});iou=metrics.get('reprojection_mean_iou')
            if primary_table is None:
                report['warnings'].append('No table or desk was detected; the support plane is a guess from the lower part of the image.')
            if iou is not None and iou<.2:
                report['warnings'].append(f'Reconstructed bodies barely overlap their image masks (mean silhouette IoU {iou:.2f}); treat this scene as unreliable.')
            report['quality_checks']={'table_detected':primary_table is not None,
                'contacts':contacts.get('float_count')==0 and contacts.get('penetration_count')==0,
                'stability':metrics.get('stability',{}).get('max_drift_m',float('inf'))<.01,
                'reprojection':iou is not None and iou>=.75}
            report['quality_status']='passed_measured_checks' if all(report['quality_checks'].values()) else 'needs_review'
            report['status']='complete'
        report['duration_s']=time.perf_counter()-start
        report['photo_latency_pass']=report['duration_s']<60
        report['acceptance_pass']=False
        report['acceptance_note']='Requires independent object inventory, metric references and fixture annotations; not inferred from model confidence.'
        (directory/'reconstruction_report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
        return scene,report
