"""Sharp-frame sampling, chunked VGGT geometry, SAM 2 propagation and TSDF fusion."""
from __future__ import annotations
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path
from threading import RLock
import json,os,shutil,subprocess,tempfile,time
import cv2
import numpy as np
import torch
from PIL import Image
from .models import registry
from .pipeline import reconstruct,root_dir
from .geometry import backproject
from .segmentation import deduplicate

VIDEO_LOCK=RLock()
TRACKER_INIT_LOCK=RLock()
VIDEO_VERSION='video_multiview_v5'


def crop_padding(images,depths,clouds,confidence,cameras,source_size):
    """Undo VGGT's documented 518-square padding without changing camera rays."""
    width,height=source_size
    side=images[0].width
    if width>=height:
        content_w=side;content_h=round(height*(side/width)/14)*14
    else:
        content_h=side;content_w=round(width*(side/height)/14)*14
    left=(side-content_w)//2;top=(side-content_h)//2
    box=(left,top,left+content_w,top+content_h)
    cropped=[im.crop(box) for im in images]
    adjusted=[]
    for camera in cameras:
        camera=dict(camera);k=np.array(camera['intrinsics'],copy=True)
        k[0,2]-=left;k[1,2]-=top;camera['intrinsics']=k.tolist()
        adjusted.append(camera)
    region=(slice(None),slice(top,top+content_h),slice(left,left+content_w))
    return cropped,depths[region].copy(),clouds[region].copy(),confidence[region].copy(),adjusted


def fuse_track_points(chunks):
    """Reject tracking jumps outside the first observed object's 3D neighborhood."""
    if not chunks:return np.empty((0,3))
    reference=np.asarray(chunks[0]);reference=reference[np.isfinite(reference).all(1)]
    if len(reference)<12:return reference
    lower,upper=np.percentile(reference,[1,99],axis=0)
    margin=np.clip((upper-lower)*.5,.035,.20)
    kept=[]
    for chunk in chunks:
        valid=np.isfinite(chunk).all(1)&(chunk>=lower-margin).all(1)&(chunk<=upper+margin).all(1)
        kept.append(chunk[valid])
    return np.concatenate(kept)


def sample_frames(path, directory, count=30, min_variance=50.):
    capture=cv2.VideoCapture(str(path));total=int(capture.get(cv2.CAP_PROP_FRAME_COUNT));fps=capture.get(cv2.CAP_PROP_FPS)
    if not capture.isOpened() or total<2 or not fps: raise ValueError('Video has no decodable frames')
    if total/fps>60: raise ValueError('Video must be at most 60 seconds')
    candidates=np.unique(np.linspace(0,total-1,min(total,count*4)).astype(int))
    frames=[];records=[];directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    wanted=set(map(int,candidates));last=int(candidates[-1])
    # H.264 random seeks repeatedly decode whole GOPs. Decode once in order and
    # materialize only candidate frames, preserving the same sampling positions.
    for index in range(last+1):
        if not capture.grab():break
        if index not in wanted:continue
        ok,frame=capture.retrieve()
        if not ok:continue
        h,w=frame.shape[:2];ratio=min(1,960/max(w,h))
        frame=cv2.resize(frame,(round(w*ratio),round(h*ratio)))
        sharp=float(cv2.Laplacian(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),cv2.CV_64F).var())
        records.append(dict(frame=int(index),time_s=float(index/fps),laplacian_variance=sharp,accepted=False))
        frames.append((frame,len(records)-1))
    capture.release()
    # Prefer one sharp candidate per temporal bin, preserving camera trajectory.
    chosen=[]
    for group in np.array_split(np.arange(len(frames)),min(count,len(frames))):
        eligible=[int(i) for i in group if records[frames[int(i)][1]]['laplacian_variance']>=min_variance]
        if eligible:chosen.append(max(eligible,key=lambda i:records[frames[i][1]]['laplacian_variance']))
    # Fill empty temporal bins from remaining sharp candidates; never admit blurry frames.
    remaining=[i for i,(_,record) in enumerate(frames) if records[record]['laplacian_variance']>=min_variance and i not in chosen]
    while len(chosen)<count and remaining:
        candidate=max(remaining,key=lambda i:min(abs(i-j) for j in chosen) if chosen else records[frames[i][1]]['laplacian_variance'])
        chosen.append(candidate);remaining.remove(candidate)
    chosen.sort()
    paths=[]
    for i,index in enumerate(chosen):
        frame,record=frames[index];records[record]['accepted']=True
        target=directory/f'frame_{i:03}.jpg';cv2.imwrite(str(target),frame);paths.append(target)
    if len(paths)<2:raise ValueError('Fewer than two sharp frames; hold the camera steady and improve lighting')
    return paths,dict(duration_s=total/fps,source_fps=fps,selected_count=len(paths),target_count=count,
                      minimum_laplacian_variance=min_variance,candidates=records,
                      warning=None if len(paths)>=30 else 'Fewer than 30 sharp frames available; reduced view coverage')


def similarity(source,target):
    """Umeyama proper similarity, used only to align adjacent overlapping VGGT chunks."""
    source=np.asarray(source);target=np.asarray(target)
    valid=np.isfinite(source).all(1)&np.isfinite(target).all(1)
    source=source[valid][::5];target=target[valid][::5]
    if len(source)<30:raise ValueError('Insufficient overlap for multi-view chunk alignment')
    if len(source)>12000:
        indices=np.linspace(0,len(source)-1,12000).astype(int);source=source[indices];target=target[indices]
    # Repeatedly trim depth-edge/padding inconsistencies at the shared views.
    for _ in range(3):
        ac,bc=source.mean(0),target.mean(0);a=source-ac;b=target-bc
        u,s,vh=np.linalg.svd(b.T@a/len(a));sign=np.eye(3);sign[-1,-1]=np.linalg.det(u@vh)
        rot=u@sign@vh;scale0=float((s*np.diag(sign)).sum()/max((a*a).sum()/len(a),1e-9))
        residual=np.linalg.norm(a@rot.T*scale0-b,axis=1)
        keep=residual<=np.percentile(residual,85)
        source=source[keep];target=target[keep]
    ac,bc=source.mean(0),target.mean(0);a=source-ac;b=target-bc
    u,s,vh=np.linalg.svd(b.T@a/len(a));sign=np.eye(3);sign[-1,-1]=np.linalg.det(u@vh)
    rotation=u@sign@vh;scale=float((s*np.diag(sign)).sum()/max((a*a).sum()/len(a),1e-9))
    if not .01<scale<100:raise ValueError('Degenerate chunk scale')
    translation=bc-scale*rotation@ac
    residual=np.linalg.norm(source@rotation.T*scale+translation-target,axis=1)
    return scale,rotation,translation,float(np.median(residual))


@lru_cache(1)
def multiview_model():
    if os.getenv('VISION_OFFLINE')=='1':raise RuntimeError('VISION_OFFLINE=1')
    from vggt.models.vggt import VGGT
    path=Path(os.getenv('RECON_MODEL_DIR','models/vision/reconstruction'))/'vggt'
    if not (path/'model.safetensors').exists():raise RuntimeError('VGGT weights missing; prepare video models')
    return VGGT.from_pretrained(str(path)).to(registry().device(4)).eval()


@torch.inference_mode()
def vggt_geometry(paths,directory):
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    model=multiview_model();chunk=int(os.getenv('RECON_VIDEO_CHUNK','10'))
    if chunk<3:raise ValueError('RECON_VIDEO_CHUNK must be at least three with two overlapping views')
    tensors=load_and_preprocess_images([str(p) for p in paths],mode='pad')
    images=[Image.fromarray(np.uint8(t.permute(1,2,0).numpy()*255)) for t in tensors]
    depths=[];clouds=[];confidences=[];cameras=[];alignments=[]
    index=0
    while index<len(paths):
        overlap=min(2,index);start=max(0,index-overlap);end=min(len(paths),start+chunk)
        if end<=index:break
        with torch.autocast(device_type='cuda',dtype=torch.float16,enabled=next(model.parameters()).is_cuda):
            predictions=model(tensors[start:end].to(next(model.parameters()).device))
        extrinsics,intrinsics=pose_encoding_to_extri_intri(predictions['pose_enc'],tensors.shape[-2:])
        ex=extrinsics[0].float().cpu().numpy();ks=intrinsics[0].float().cpu().numpy()
        ds=predictions['depth'][0,...,0].float().cpu().numpy();cs=predictions['depth_conf'][0].float().cpu().numpy()
        local=[]
        for depth,k,e in zip(ds,ks,ex):
            camera=dict(fx=float(k[0,0]),fy=float(k[1,1]),cx=float(k[0,2]),cy=float(k[1,2]),width=depth.shape[1],height=depth.shape[0],source='VGGT_predicted')
            points=backproject(depth,camera)
            local.append((points-e[:,3])@e[:,:3])  # camera -> chunk world
        if start==0:scale,rotation,translation=1.,np.eye(3),np.zeros(3)
        else:
            scale,rotation,translation,residual=similarity(np.concatenate([p.reshape(-1,3) for p in local[:overlap]]),np.concatenate([p.reshape(-1,3) for p in clouds[-overlap:]]))
            alignments.append(dict(overlap_frame=start,scale=scale,median_residual=residual))
        for j in range(0 if start==0 else overlap,len(local)):
            world=local[j]@rotation.T*scale+translation
            center=-ex[j,:,:3].T@ex[j,:,3];center=rotation@center*scale+translation
            camera_to_world=rotation@ex[j,:,:3].T
            cameras.append(dict(intrinsics=ks[j].tolist(),camera_to_world=camera_to_world.tolist(),center=center.tolist()))
            depths.append(ds[j]*scale);clouds.append(world);confidences.append(cs[j])
        del predictions
        index=end
    # Express everything in the first camera coordinates for the photo lift stage.
    first=cameras[0];rotation=np.array(first['camera_to_world']);origin=np.array(first['center'])
    clouds=[(p-origin)@rotation for p in clouds]
    for camera in cameras:
        camera['center']=((np.array(camera['center'])-origin)@rotation).tolist()
        camera['camera_to_world']=(rotation.T@np.array(camera['camera_to_world'])).tolist()
    with Image.open(paths[0]) as source:
        images,depths,clouds,confidences,cameras=crop_padding(images,np.array(depths),np.array(clouds),np.array(confidences),cameras,source.size)
    np.savez_compressed(Path(directory)/'multiview.npz',depth=depths,confidence=confidences,points=clouds)
    (Path(directory)/'cameras.json').write_text(json.dumps(cameras,indent=2))
    return images,np.array(depths),np.array(clouds),np.array(confidences),cameras,alignments


@lru_cache(7)
def tracker_model(device=None):
    from transformers import Sam2VideoProcessor,Sam2VideoModel
    path=Path(os.getenv('RECON_MODEL_DIR','models/vision/reconstruction'))/'sam'
    # Transformers' initialization contexts affect global module state. Concurrent
    # checkpoint loads can leave meta tensors; inference itself stays parallel.
    with TRACKER_INIT_LOCK:
        return (Sam2VideoProcessor.from_pretrained(str(path),local_files_only=True),
                Sam2VideoModel.from_pretrained(str(path),local_files_only=True).to(device or registry().device(2)).eval())


@torch.inference_mode()
def track_group(images,instances,clouds,confidence,device):
    processor,model=tracker_model(device)
    session=processor.init_video_session(video=images,inference_device=str(model.device),
        processing_device='cpu',video_storage_device='cpu',inference_state_device='cpu')
    # Seed actual SAM 2 masks, rather than repeating box detections independently.
    if not instances:raise ValueError('No masks available for video tracking')
    ids=[inst['track_id'] for inst in instances]
    # The processor retains this list as mutable pending-input state and the
    # model removes IDs after conditioning. Keep our output mapping independent.
    processor.add_inputs_to_inference_session(session,frame_idx=0,obj_ids=ids.copy(),
        input_masks=[inst['mask'] for inst in instances],original_size=images[0].size[::-1])
    fused={i:[] for i in ids};track_report=[]
    model(inference_session=session,frame_idx=0)
    for output in model.propagate_in_video_iterator(session,start_frame_idx=0):
        frame=int(output.frame_idx)
        masks=processor.post_process_masks(output.pred_masks.unsqueeze(0).cpu(),[images[frame].size[::-1]])[0]
        threshold=float(np.percentile(confidence[frame],25))
        valid=(confidence[frame]>=threshold)&np.isfinite(clouds[frame]).all(-1)
        for index,mask in zip(ids,masks):
            mask=np.asarray(mask[0],bool)
            points=clouds[frame][mask&valid][::3]
            if len(points):fused[index].append(points)
            track_report.append(dict(frame=frame,track_id=index,pixels=int(mask.sum()),sampled_points=len(points)))
    if not track_report:raise RuntimeError('SAM2 returned no object observations')
    return {i:fuse_track_points(chunks) for i,chunks in fused.items()},track_report


def track_instances(images,instances,clouds,confidence):
    """Independent object memories can propagate concurrently on separate GPUs."""
    if not instances:raise ValueError('No masks available for video tracking')
    for index,inst in enumerate(instances):inst['track_id']=index
    workers=min(len(instances),len(registry().devices),max(1,int(os.getenv('RECON_TRACK_WORKERS','7')))) if torch.cuda.is_available() else 1
    groups=[instances[index::workers] for index in range(workers)]
    fused={};observations=[]
    # VGGT activation buffers are no longer needed when propagation starts.
    if torch.cuda.is_available():torch.cuda.empty_cache()
    with ThreadPoolExecutor(max_workers=workers,thread_name_prefix='video-track') as pool:
        futures=[pool.submit(track_group,images,group,clouds,confidence,registry().device(index))
                 for index,group in enumerate(groups)]
        for group,future in zip(groups,futures):
            try:
                points,records=future.result()
            except Exception as exc:
                # A worker failure cannot discard objects tracked on other GPUs.
                points={};records=[]
                for inst in group:
                    mask=inst['mask']&np.isfinite(clouds[0]).all(-1)
                    selected=clouds[0][mask][::3];track=inst['track_id']
                    points[track]=selected
                    records.append(dict(frame=0,track_id=track,pixels=int(mask.sum()),sampled_points=len(selected),
                        fallback='reference_frame_only',reason=f'{type(exc).__name__}: {str(exc)[:250]}'))
            fused.update(points);observations.extend(records)
    observations.sort(key=lambda item:(item['frame'],item['track_id']))
    return fused,observations


def tsdf_fuse(images,depths,cameras,directory):
    import open3d as o3d
    volume=o3d.pipelines.integration.ScalableTSDFVolume(voxel_length=.006,sdf_trunc=.03,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    for image,depth,camera in zip(images,depths,cameras):
        k=np.array(camera['intrinsics']);intrinsic=o3d.camera.PinholeCameraIntrinsic(image.width,image.height,k[0,0],k[1,1],k[0,2],k[1,2])
        pose=np.eye(4);pose[:3,:3]=camera['camera_to_world'];pose[:3,3]=camera['center']
        rgbd=o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(np.asarray(image).copy()),
            o3d.geometry.Image(depth.astype(np.float32)),depth_scale=1.,depth_trunc=50,convert_rgb_to_intensity=False)
        volume.integrate(rgbd,intrinsic,np.linalg.inv(pose))
    cloud=volume.extract_point_cloud();o3d.io.write_point_cloud(str(Path(directory)/'fused.ply'),cloud)
    if not len(cloud.points):
        raise RuntimeError('TSDF produced an empty cloud; raw multi-view points are saved separately')
    return dict(points=len(cloud.points),voxel_size_unscaled=.006,scale_note='VGGT units; scene scale solved separately')


def reconstruct_video(path,table_width_m=None,*,use_cache=True):
    """Return a saved multi-view scene; degraded/empty runs never block a retry."""
    from .pipeline import VERSION as photo_version
    start=time.perf_counter()
    if table_width_m is not None and not .3<=table_width_m<=3:
        raise ValueError('table_width_m must be between 0.3 and 3')
    config=dict(version=VIDEO_VERSION,photo_version=photo_version,table_width_m=table_width_m,
                chunk=os.getenv('RECON_VIDEO_CHUNK','10'),mesh=os.getenv('RECON_MESH','1'),
                threshold=os.getenv('RECON_DETECTION_THRESHOLD','.22'),auto_grid=os.getenv('RECON_AUTO_GRID','12'))
    digest=sha256(Path(path).read_bytes()+json.dumps(config,sort_keys=True).encode()).hexdigest()
    work=root_dir()/digest;work.mkdir(parents=True,exist_ok=True)
    with VIDEO_LOCK:
        from agatha.schema import Scene
        report_path=work/'reconstruction_report.json';scene_path=work/'scene.json'
        if use_cache and report_path.exists() and scene_path.exists():
            cached=json.loads(report_path.read_text())
            cached_scene=Scene.model_validate_json(scene_path.read_text())
            cached_tracking=cached.get('video',{}).get('stages',{}).get('tracking',{})
            if (cached.get('status')=='complete' and cached_scene.objects and cached_tracking.get('status')=='ok'
                    and cached_tracking.get('observations',0)>0 and any(cached_tracking.get('fused_point_counts',{}).values())):
                cached['cache_hit']=True;cached['request_duration_s']=time.perf_counter()-start
                return cached_scene,cached
        stage_start=time.perf_counter()
        paths,sampling=sample_frames(path,work/'frames')
        video_report=dict(sampling=sampling,stages={},warnings=[],timings_s={'sampling':time.perf_counter()-stage_start})
        stage_start=time.perf_counter()
        if sampling['warning']:video_report['warnings'].append(sampling['warning'])
        try:
            images,depths,clouds,confidence,cameras,alignments=vggt_geometry(paths,work)
            video_report['stages']['geometry']=dict(method='VGGT-1B',alignments=alignments,frames=len(images),metric_scale=False)
        except Exception as exc:
            video_report['warnings'].append(f'VGGT failed: {type(exc).__name__}: {str(exc)[:300]}')
            try:
                images,depths,clouds,confidence,cameras,alignments=colmap_geometry(paths,work)
                video_report['stages']['geometry']=dict(method='COLMAP_poses_with_metric_monocular_depth',frames=len(images),metric_scale=False)
                video_report['stages']['colmap']=dict(status='ok',dense_method='Depth Anything V2; CPU COLMAP has no CUDA PatchMatch')
            except Exception as colmap_exc:
                video_report['stages']['colmap']=dict(status='failed',reason=f'{type(colmap_exc).__name__}: {str(colmap_exc)[:300]}')
                scene,report=reconstruct(Image.open(paths[0]),table_width_m)
                report=dict(report,input='video',video=video_report,pipeline='video_representative_photo_fallback')
                report['warnings']=[*report.get('warnings',[]),*video_report['warnings'],'Multi-view geometry unavailable; reconstructed one sharp frame only.']
                report['duration_s']=time.perf_counter()-start;report['video_latency_pass']=report['duration_s']<180
                return scene,report
        video_report['timings_s']['geometry']=time.perf_counter()-stage_start
        stage_start=time.perf_counter()
        model=registry();detections=model.detect(images[0]);masks=model.segment(images[0],detections)
        model.classify(images[0],[m for m in masks if m['label']=='unknown object'])
        instances,dropped=deduplicate(masks,depths[0])
        instances=[inst for inst in instances if inst['mask_source']!='sam2_automatic' or
                   (inst['label'] not in ('table','wall','floor','unknown object') and inst.get('classifier_confidence',0)>=.35)]
        # Automatic masks often describe a seat/leg within an already detected chair.
        # Track independent pieces, rather than assigning another whole chair to each part.
        retained=[]
        for inst in instances:
            if inst['mask_source']=='sam2_automatic':
                nested=False
                for other in instances:
                    if other is inst or other['mask_source']!='sam2_box' or other['label']!=inst['label']:continue
                    overlap=(inst['mask']&other['mask']).sum()/max(inst['mask'].sum(),1)
                    if overlap>.65:nested=True;break
                if nested:continue
            retained.append(inst)
        instances=retained
        video_report['timings_s']['detection_and_masks']=time.perf_counter()-stage_start
        stage_start=time.perf_counter()
        try:
            fused,tracking=track_instances(images,instances,clouds,confidence)
            video_report['stages']['tracking']=dict(method='SAM2_video_propagation',tracks=len(fused),observations=len(tracking),
                workers=min(len(instances),len(registry().devices),max(1,int(os.getenv('RECON_TRACK_WORKERS','7')))) if torch.cuda.is_available() else 1)
            failed_tracks=[row['track_id'] for row in tracking if row.get('fallback')]
            empty_tracks=[track for track,points in fused.items() if not len(points)]
            failed_tracks=sorted(set(failed_tracks)|set(empty_tracks))
            video_report['stages']['tracking']['status']='partial' if failed_tracks else 'ok'
            video_report['stages']['tracking']['reference_frame_fallback_tracks']=failed_tracks
            video_report['stages']['tracking']['fused_point_counts']={str(track):len(points) for track,points in fused.items()}
            video_report['stages']['tracking']['point_filter']='reference_frame_spatial_consistency'
            if failed_tracks:video_report['warnings'].append(f'{len(failed_tracks)} tracks fell back to reference-frame points after a worker failure.')
            (work/'tracks.json').write_text(json.dumps(tracking))
            np.savez_compressed(work/'object_points.npz',**{str(i):p for i,p in fused.items()})
        except Exception as exc:
            fused={};video_report['stages']['tracking']=dict(status='failed',reason=f'{type(exc).__name__}: {str(exc)[:300]}')
            video_report['warnings'].append('Video tracking failed; body fits use the reference frame only.')
        video_report['timings_s']['tracking']=time.perf_counter()-stage_start
        stage_start=time.perf_counter()
        try: video_report['stages']['fusion']=tsdf_fuse(images,depths,cameras,work)
        except Exception as exc:
            video_report['stages']['fusion']=dict(status='failed',reason=f'{type(exc).__name__}: {str(exc)[:300]}')
            video_report['warnings'].append('TSDF fusion failed; raw multi-view geometry is retained.')
        video_report['timings_s']['fusion']=time.perf_counter()-stage_start
        k=np.array(cameras[0]['intrinsics']);camera=dict(fx=k[0,0],fy=k[1,1],cx=k[0,2],cy=k[1,2],
            width=images[0].width,height=images[0].height,source='VGGT_predicted')
        override=dict(depth=depths[0],camera=camera,method='VGGT-1B',hash=digest,
                      detections=detections,masks=[*instances,*[m for m in masks if m['label'] in ('table','wall','floor')]],fused=fused)
        stage_start=time.perf_counter()
        scene,report=reconstruct(images[0],table_width_m,geometry_override=override,use_cache=False)
        video_report['timings_s']['scene_lift_and_validation']=time.perf_counter()-stage_start
        # Publish one complete, reloadable video bundle. Photo-stage assets keep their
        # exact geometry/texture contents while their URLs move into this run directory.
        photo_run=report.get('run_id')
        if photo_run:
            source=root_dir()/photo_run
            for asset in source.iterdir():
                if asset.is_file() and asset.name not in ('scene.json','reconstruction_report.json'):
                    shutil.copy2(asset,work/asset.name)
        report.update(input='video',pipeline=VIDEO_VERSION,video=video_report,run_id=digest,
                      photo_run_id=photo_run,debug_url=f'/api/reconstructions/{digest}',cache_hit=False)
        for obj in scene.objects:
            if obj.visual_mesh_url:obj.visual_mesh_url=f"{report['debug_url']}/{obj.id}.glb"
        report.pop('photo_latency_pass',None)
        report['duration_s']=time.perf_counter()-start
        report['video_latency_pass']=report['duration_s']<180
        report['warnings'].extend(video_report['warnings'])
        metrics=report.get('metrics',{});contacts=metrics.get('contacts_after_settle',{})
        iou=metrics.get('reprojection_mean_iou')
        report['quality_checks']={
            'tracking':video_report['stages'].get('tracking',{}).get('status')=='ok',
            'contacts':contacts.get('float_count')==0 and contacts.get('penetration_count')==0,
            'stability':metrics.get('stability',{}).get('max_drift_m',float('inf'))<.01,
            'reprojection':iou is not None and iou>=.75,
        }
        report['quality_status']='passed_measured_checks' if all(report['quality_checks'].values()) else 'needs_review'
        if report['quality_status']=='needs_review':
            failed=', '.join(name for name,passed in report['quality_checks'].items() if not passed)
            report['warnings'].append(f'Video scene needs review: {failed} checks did not pass. See measured metrics.')
        report['status']='complete'
        scene_path.write_text(scene.model_dump_json(indent=2))
        temporary=work/'reconstruction_report.tmp';temporary.write_text(json.dumps(report,indent=2,allow_nan=False))
        temporary.replace(report_path)
        return scene,report


def colmap_geometry(paths,directory):
    """Genuine COLMAP SfM fallback, with monocular dense depth explicitly identified.

    The installed CPU wheel lacks CUDA PatchMatch. We fuse metric model depth using
    recovered COLMAP poses, fitting per-view depth scale to triangulated points.
    """
    import pycolmap
    from .geometry import intrinsics
    directory=Path(directory);images_dir=directory/'colmap_images';images_dir.mkdir(exist_ok=True)
    for path in paths:shutil.copy2(path,images_dir/path.name)
    database=directory/'colmap.db';output=directory/'colmap_sparse';output.mkdir(exist_ok=True)
    pycolmap.extract_features(database,images_dir,camera_mode=pycolmap.CameraMode.SINGLE,
                             device=pycolmap.Device.cpu)
    pycolmap.match_exhaustive(database,device=pycolmap.Device.cpu)
    reconstructions=pycolmap.incremental_mapping(database,images_dir,output)
    if not reconstructions:raise RuntimeError('COLMAP could not register cameras; insufficient parallax or texture')
    reconstruction=max(reconstructions.values(),key=lambda r:r.num_reg_images())
    registered={image.name:image for image in reconstruction.images.values() if image.has_pose}
    if len(registered)<2:raise RuntimeError('COLMAP registered fewer than two views')
    images=[];depths=[];clouds=[];cameras=[];confidence=[]
    for path in paths:
        if path.name not in registered:continue
        image=Image.open(path).convert('RGB');image.thumbnail((960,960))
        record=registered[path.name];camera=reconstruction.cameras[record.camera_id]
        k=camera.calibration_matrix();sx=image.width/camera.width;sy=image.height/camera.height
        k[0]*=sx;k[1]*=sy
        depth=registry().depth(image)
        extrinsic=record.cam_from_world().matrix();rot=extrinsic[:,:3];translation=extrinsic[:,3]
        ratios=[]
        for point2d in record.points2D:
            if not point2d.has_point3D():continue
            xyz=reconstruction.points3D[point2d.point3D_id].xyz
            z=float((rot@xyz+translation)[2]);x,y=np.array(point2d.xy)*[sx,sy]
            if 0<=int(y)<image.height and 0<=int(x)<image.width and z>0:
                estimate=float(depth[int(y),int(x)])
                if estimate>0:ratios.append(z/estimate)
        if len(ratios)<10:continue
        scale=float(np.median(ratios));depth*=scale
        cam=dict(fx=k[0,0],fy=k[1,1],cx=k[0,2],cy=k[1,2],width=image.width,height=image.height)
        cloud=(backproject(depth,cam)-translation)@rot
        images.append(image);depths.append(depth);clouds.append(cloud);confidence.append(np.ones(depth.shape))
        cameras.append(dict(intrinsics=k.tolist(),camera_to_world=rot.T.tolist(),center=(-rot.T@translation).tolist()))
    if len(images)<2:raise RuntimeError('Insufficient triangulation for COLMAP/model depth alignment')
    first=cameras[0];rot=np.array(first['camera_to_world']);origin=np.array(first['center'])
    clouds=[(p-origin)@rot for p in clouds]
    for camera in cameras:
        camera['center']=((np.array(camera['center'])-origin)@rot).tolist()
        camera['camera_to_world']=(rot.T@np.array(camera['camera_to_world'])).tolist()
    return images,np.array(depths),np.array(clouds),np.array(confidence),cameras,[]
