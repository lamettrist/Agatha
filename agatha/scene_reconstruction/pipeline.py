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

VERSION='metric-photo-v11'
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
    if label=='chair':obj.supported_by='floor'
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
        if not detections:
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
            table_masks=[m['mask'] for m in masks if m['label']=='table']
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
            extents=np.percentile(cloud_world[surface][:,[0,2]],[1,99],axis=0) if surface.sum()>30 else np.array([[-.6,-.4],[.6,.4]])
            center_xz=extents.mean(0);origin=origin+basis.T@np.array([center_xz[0],0,center_xz[1]])
            cloud_world=(cloud-origin)@basis.T
            plane_mask=valid & (np.abs(cloud_world[:,:,1])<.018)
            stage.update(normal_camera=plane['normal'].tolist(),inlier_ratio=plane['inlier_ratio'],rms_m=plane['rms_m'],
                         gravity_source='table_plane_normal',world_basis=basis.tolist())
        with timed(report,'instances') as stage:
            instances,dropped=deduplicate(masks,depth,plane_mask)
            instances=split_depth_clusters(instances,depth)
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
            scale=g.solve_scale(references,manual);stage.update(scale)
            factor=scale['factor'];cloud_world*=factor;depth*=factor
            support_size=np.maximum((extents[1]-extents[0])*factor,[.3,.3])
            if np.max(support_size)>10: raise ValueError('Implausible support dimensions; provide a scale reference')
            cloud_world[:,:,1]+=.75
            camera_pose=dict(origin_world=((-origin)@basis.T*factor+[0,.75,0]).tolist(),world_to_camera=basis.T.tolist())
        support_height=.75
        floor_masks=[m['mask'] for m in masks if m['label']=='floor']
        if floor_masks:
            floor_pixels=np.logical_or.reduce(floor_masks)&valid
            try:
                floor=g.fit_plane(cloud_world[floor_pixels],threshold=.025,normal_hint=[0,1,0])
                floor_y=float(floor['center'][1]);measured_height=.75-floor_y
                if .2<measured_height<3:
                    support_height=measured_height;cloud_world[:,:,1]-=floor_y
                    camera_pose['origin_world'][1]-=floor_y
                    report['stages']['structure']['floor_height_correction_m']=floor_y
                    report['stages']['structure']['support_height_m']=support_height
            except ValueError:pass
        support=Support(size=tuple(support_size),height=support_height,thickness=.03)
        with timed(report,'lift') as stage:
            objects=[];fused_world_by_id={}
            for i,inst in enumerate(instances[:240]):
                try:
                    fused=None
                    if geometry_override and inst.get('track_id') in geometry_override.get('fused',{}):
                        fused=geometry_override['fused'][inst['track_id']]
                        fused=(fused-origin)@basis.T*factor+[0,support.height,0]
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
                fitted=g.fit_box(pts);dims=fitted['size'];dims[int(np.argmin(dims))]=.03
                objects.append(SceneObject(id=f'wall_{j}',label='wall',size=tuple(dims),position=tuple(fitted['center']),
                    rotation_euler=tuple(fitted['rotation_euler']),is_static=True,shape_method='plane_slab',confidence=mask['confidence']))
            stage['body_count']=len(objects)
        # Recenter/expand the slab using observed table bounds and contact anchors.
        # Hidden table extent is an estimate, recorded instead of dropping bodies off its edge.
        contact_points=[]
        for inst,obj in zip(instances,objects):
            if obj.is_static: continue
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
                    if camera_contact[2]>back_depth+.08:
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
            scene,contacts=settle(scene);stage.update(contacts);report['metrics']['contacts_after_settle']=contacts
        with timed(report,'stability') as stage:
            measured=stability(scene);stage.update(measured);report['metrics']['stability']=measured
        report.update(detected_count=len(detections),kept_count=len(instances),detector='grounding_dino',depth='metric_depth',
            detections=safe_report_instances(instances),scale_confidence='medium' if scale['references'] or manual else 'low',
            scale_basis=scale['source'],scale=scale,camera=camera,camera_pose=camera_pose)
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
        report['duration_s']=time.perf_counter()-start
        report['photo_latency_pass']=report['duration_s']<60
        report['acceptance_pass']=False
        report['acceptance_note']='Requires independent object inventory, metric references and fixture annotations; not inferred from model confidence.'
        (directory/'reconstruction_report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
        return scene,report
