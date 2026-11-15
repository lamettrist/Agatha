"""Persistent one-worker-per-GPU crop-to-mesh jobs with point alignment and CoACD."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
import os,sys,time
import numpy as np
import torch
from PIL import Image
from agatha.schema import Collision,CollisionPart
from .models import registry

MODELS={};MODEL_LOCKS={};INIT_LOCK=Lock()
POOL=ThreadPoolExecutor(max_workers=7,thread_name_prefix='object-mesh')


def load_model(device):
    with INIT_LOCK:
        if device in MODELS:return MODELS[device]
        source=Path(os.getenv('TRIPOSR_SOURCE','models/vision/source/TripoSR')).resolve()
        weights=Path(os.getenv('RECON_MODEL_DIR','models/vision/reconstruction'))/'triposr'
        if not (source/'tsr/system.py').exists() or not (weights/'model.ckpt').exists():
            raise RuntimeError('Official TripoSR code/weights are not prepared')
        sys.path.insert(0,str(source)) if str(source) not in sys.path else None
        from tsr.system import TSR
        # Upstream fetches just the encoder config. Force local-only access here.
        import tsr.models.tokenizers.image as tokenizer
        from huggingface_hub import hf_hub_download
        tokenizer.hf_hub_download=lambda **kwargs:hf_hub_download(**kwargs,local_files_only=True)
        model=TSR.from_pretrained(str(weights),config_name='config.yaml',weight_name='model.ckpt').to(device).eval()
        model.renderer.set_chunk_size(8192)
        MODELS[device]=model;MODEL_LOCKS[device]=Lock()
        return model


def masked_crop(image,instance):
    rgba=np.dstack([np.array(image),instance['mask'].astype(np.uint8)*255])
    x1,y1,x2,y2=map(int,instance['bbox']);crop=Image.fromarray(rgba).crop((x1,y1,x2,y2))
    size=max(crop.size);canvas=Image.new('RGBA',(size,size),(255,255,255,0))
    canvas.alpha_composite(crop,((size-crop.width)//2,(size-crop.height)//2))
    background=Image.new('RGBA',canvas.size,(255,255,255,255));background.alpha_composite(canvas)
    return background.convert('RGB')


def align_mesh(mesh,obj,observed):
    """Fit class-completed metric bounds, then accept only small robust ICP corrections."""
    import open3d as o3d
    from .assets import rotation
    mesh=mesh.copy()
    bounds=mesh.bounds;center=bounds.mean(0);extent=np.maximum(bounds[1]-bounds[0],1e-4)
    mesh.vertices=(mesh.vertices-center)/extent*np.array(obj.size)
    # The observed cloud is partial. Prevent ICP from collapsing a completed shape.
    source=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(mesh.vertices@rotation(obj.rotation_euler).T+obj.position))
    if len(source.points)>10000:source=source.voxel_down_sample(.006)
    target=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(observed)))
    target=target.voxel_down_sample(.008)
    transformation=np.eye(4);fitness=None
    if len(target.points)>30:
        fit=o3d.pipelines.registration.registration_icp(source,target,.025,np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=25))
        fitness=float(fit.fitness)
        if np.linalg.norm(fit.transformation[:3,3])<.04:
            transformation=fit.transformation
    # Keep local geometry centered at the body's COM; add the accepted rigid correction.
    rot=rotation(obj.rotation_euler)
    world=mesh.vertices@rot.T+obj.position
    world=world@transformation[:3,:3].T+transformation[:3,3]
    mesh.vertices=(world-obj.position)@rot
    return mesh,dict(icp_fitness=fitness,icp_translation_m=float(np.linalg.norm(transformation[:3,3])),
                     alignment='metric_template_bounds_then_bounded_rigid_ICP')


def convex_parts(mesh):
    import json,subprocess,tempfile
    with tempfile.TemporaryDirectory(prefix='agatha-hulls-') as tmp:
        source=Path(tmp)/'mesh.ply';target=Path(tmp)/'hulls.json';mesh.export(source)
        env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        completed=subprocess.run([sys.executable,'-m','agatha.scene_reconstruction.decompose',str(source),str(target)],
            env=env,capture_output=True,text=True,timeout=20)
        if completed.returncode or not target.exists():
            raise RuntimeError(f'CoACD subprocess failed ({completed.returncode}): {completed.stderr[-250:]}')
        return Collision.model_validate_json(target.read_text())


def point_mesh(points):
    import open3d as o3d
    pcd=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points));pcd=pcd.voxel_down_sample(.006)
    if len(pcd.points)<100:raise ValueError('Too few fused points for a surface mesh')
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=.035,max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(20)
    mesh,density=o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd,depth=6,n_threads=2)
    mesh.remove_vertices_by_mask(np.asarray(density)<np.quantile(density,.05))
    mesh=mesh.crop(pcd.get_axis_aligned_bounding_box())
    import trimesh
    return trimesh.Trimesh(np.asarray(mesh.vertices),np.asarray(mesh.triangles),process=False)


def job(index,image,inst,obj,observed,directory,fused=None):
    start=time.perf_counter();result=dict(id=obj.id)
    try:
        if fused is not None and len(fused)>100:
            from .assets import rotation
            mesh=point_mesh(fused)
            mesh.vertices=(mesh.vertices-np.asarray(obj.position))@rotation(obj.rotation_euler)
            method='fused_multiview_poisson';alignment={'fused_points':len(fused)}
        else:
            device=registry().device(index);model=load_model(device)
            with MODEL_LOCKS[device],torch.inference_mode():
                code=model([masked_crop(image,inst)],device=device)
                mesh=model.extract_mesh(code,has_vertex_color=False,resolution=96)[0]
            mesh,alignment=align_mesh(mesh,obj,observed);method='triposr_metric_icp'
        if len(mesh.faces)<4:raise ValueError('Generated mesh is empty')
        collision=convex_parts(mesh)
        path=Path(directory)/(obj.id+'_surface.ply');mesh.export(path)
        result.update(status='ok',method=method,collision=collision,surface=path,alignment=alignment,
                      vertices=len(mesh.vertices),convex_hulls=len(collision.parts))
    except Exception as exc:
        result.update(status='fallback',method=obj.shape_method,reason=f'{type(exc).__name__}: {str(exc)[:350]}')
    result['duration_s']=time.perf_counter()-start
    return result



def silhouette_score(mesh,obj,instance,camera,camera_pose,support_height):
    import cv2
    from .assets import rotation,camera_points,project
    vertices=np.asarray(mesh.vertices)
    world=vertices@rotation(obj.rotation_euler).T+obj.position
    world[:,1]+=support_height-float(world[:,1].min())
    pixels=project(camera_points(world,camera_pose),camera)
    # A conservative convex silhouette is sufficient to reject clearly bad candidates.
    scale=min(1.,400/camera['width']);shape=(round(camera['height']*scale),round(camera['width']*scale))
    mask=np.zeros(shape,np.uint8)
    finite=np.isfinite(pixels).all(1)
    if finite.sum()<4:return 0.
    hull=cv2.convexHull(np.int32(np.clip(pixels[finite]*scale,-10000,10000)))
    cv2.fillConvexPoly(mask,hull,1)
    target=cv2.resize(instance['mask'].astype(np.uint8),(shape[1],shape[0]),interpolation=cv2.INTER_NEAREST)>0
    return float(((mask>0)&target).sum()/max(((mask>0)|target).sum(),1))


def generate_objects(image,instances,objects,cloud,valid,directory,fused_by_id=None,camera=None,camera_pose=None,support_height=.75):
    by_id={i.get('object_id'):i for i in instances};futures=[]
    for index,obj in enumerate(objects):
        if not obj.soft_approx or obj.id not in by_id:continue
        inst=by_id[obj.id];observed=cloud[inst['mask']&valid]
        futures.append((obj,POOL.submit(job,index,image,inst,obj,observed,directory,(fused_by_id or {}).get(obj.id))))
    reports=[]
    for obj,future in futures:
        result=future.result()
        if result['status']=='ok':
            import trimesh
            from .assets import object_mesh
            mesh=trimesh.load_mesh(result['surface'])
            baseline=silhouette_score(object_mesh(obj),obj,by_id[obj.id],camera,camera_pose,support_height)
            candidate=silhouette_score(mesh,obj,by_id[obj.id],camera,camera_pose,support_height)
            result['baseline_silhouette_iou_proxy']=baseline;result['candidate_silhouette_iou_proxy']=candidate
            if candidate>=baseline+.01:
                obj.collision=result.pop('collision');obj.shape='convex_hull';obj.shape_method=result['method']
                result['surface']=str(result['surface'])
            else:
                result.pop('collision');Path(result['surface']).unlink(missing_ok=True)
                result.update(status='fallback',method=obj.shape_method,reason='Generated mesh silhouette is worse than the completed box; kept the better fit',surface=None)
        reports.append(result)
    return reports
