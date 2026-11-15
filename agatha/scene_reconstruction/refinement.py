"""Constrained silhouette refinement. All objective proxies are labeled as proxies."""
import cv2
import numpy as np
from scipy.optimize import minimize
from agatha.schema import Collision,CollisionPart
from .assets import rotation,camera_points,project

CORNERS=np.array([[x,y,z] for x in (-.5,.5) for y in (-.5,.5) for z in (-.5,.5)])
ANGLES=np.linspace(0,2*np.pi,36,endpoint=False)


def vertices(part):
    if part.type=='box':return CORNERS*np.array(part.size)
    if part.type=='cylinder':return np.array([[part.radius*np.cos(a),y,part.radius*np.sin(a)] for a in ANGLES for y in (-part.height/2,part.height/2)])
    return np.asarray(part.vertices)


def fit_one(obj,instance,instances,camera,camera_pose,support_height):
    label=instance['label']
    if label not in ('bottle','tumbler','cup','laptop','backpack','bag','jacket','gloves','mouse','phone'):return None
    if obj.supported_by=='floor':return None
    scale=min(1.,360/camera['width']);shape=(round(camera['height']*scale),round(camera['width']*scale))
    target=cv2.resize(instance['mask'].astype(np.uint8),(shape[1],shape[0]),interpolation=cv2.INTER_NEAREST)>0
    if target.sum()<15:return None
    occlusion=np.zeros(shape,bool)
    distance=instance.get('median_depth_m')
    for other in instances:
        if other is instance:continue
        other_distance=other.get('median_depth_m')
        if distance is not None and other_distance is not None and other_distance<distance-.04:
            occlusion|=cv2.resize(other['mask'].astype(np.uint8),(shape[1],shape[0]),interpolation=cv2.INTER_NEAREST)>0
    pos=np.array(obj.position);yaw=obj.rotation_euler[1]
    width,height,depth=obj.size
    compound=obj.shape=='compound';cylinder=obj.shape=='cylinder'
    if compound:
        base=obj.collision.parts[0];width,base_height,depth=base.size
        screen=obj.collision.parts[1];height=screen.size[1]
    initial=np.array([pos[0],pos[2],yaw,width,height,depth])
    bounds=[(pos[0]-.08,pos[0]+.08),(pos[2]-.10,pos[2]+.10),(yaw-.35,yaw+.35),
            (max(.01,width*.8),width*1.15),(max(.006,height*.8),height*1.15),(max(.006,depth*.75),depth*1.15)]
    if cylinder:
        bounds[2]=(yaw-.001,yaw+.001);bounds[5]=(depth*.999,depth*1.001)
    def parts(params):
        x,z,angle,w,h,d=params
        if cylinder:return [CollisionPart(type='cylinder',radius=w/2,height=h)]
        if compound:
            hinge=np.deg2rad(105.)
            return [CollisionPart(size=(w,.018,d)),CollisionPart(size=(w,h,.006),
                position=(0,.009+h*np.sin(hinge)/2,-d/2+h*np.cos(hinge)/2),rotation_euler=(float(np.pi/2-hinge),0,0))]
        return [CollisionPart(size=(w,h,d))]
    def predicted(params):
        x,z,angle,w,h,d=params;rot=rotation([0,angle,0]);pieces=parts(params)
        local=[vertices(part)@rotation(part.rotation_euler).T+part.position for part in pieces]
        lowest=min((v@rot.T)[:,1].min() for v in local)
        center=np.array([x,support_height-lowest,z]);mask=np.zeros(shape,np.uint8)
        for v in local:
            cam=camera_points(v@rot.T+center,camera_pose)
            if np.any(cam[:,2]<.05):continue
            uv=project(cam,camera)*scale
            hull=cv2.convexHull(np.int32(np.clip(uv,-10000,10000)))
            cv2.fillConvexPoly(mask,hull,1)
        return (mask>0)&~occlusion,center,pieces
    def score(params):
        mask,_,_=predicted(params)
        return float((mask&target).sum()/max((mask|target).sum(),1))
    def objective(params):
        # Keep hidden dimensions close to class completion; measured pixels drive visible fit.
        regularizer=.008*np.sum(((params-initial)/np.maximum(abs(initial),.1))**2)
        return 1-score(params)+regularizer
    before=score(initial)
    fit=minimize(objective,initial,method='Powell',bounds=bounds,options={'maxiter':12,'maxfev':230,'xtol':.001,'ftol':.002})
    after=score(fit.x)
    if after>before+.015:
        mask,center,pieces=predicted(fit.x);obj.position=tuple(center);obj.rotation_euler=(0,float(fit.x[2]),0)
        obj.collision=Collision(parts=pieces)
        obj.size=(float(fit.x[3]),float(fit.x[4]+.018 if compound else fit.x[4]),float(fit.x[5]))
        if cylinder:obj.size=(obj.size[0],obj.size[1],obj.size[0])
        obj.shape_method+='_silhouette_refined'
    return dict(id=obj.id,initial_iou_proxy=before,optimized_iou_proxy=after,accepted=after>before+.015,
                evaluations=int(fit.nfev),method='bounded_Powell_on_projected_primitives')


def refine(scene_objects,instances,camera,camera_pose,height):
    by_id={inst.get('object_id'):inst for inst in instances};reports=[]
    for obj in scene_objects:
        if obj.id not in by_id or obj.is_static:continue
        report=fit_one(obj,by_id[obj.id],instances,camera,camera_pose,height)
        if report is not None:reports.append(report)
    return reports
