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
VIDEO_VERSION='video_multiview_v8'
PHOTO_FALLBACK='video_representative_photo_fallback'


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
    model=VGGT.from_pretrained(str(path))
    # Only depth, confidence and pose are consumed; the point-map and track heads cost memory.
    model.point_head=None;model.track_head=None
    model=model.to(registry().device(4)).eval()
    if next(model.parameters()).is_cuda and os.getenv('RECON_VGGT_HALF','1')=='1':
        # The aggregator already runs under fp16 autocast, so fp16 weights change depth by a
        # median 0.03 % (measured) while cutting the footprint from 8.5+ GB to 5.7 GB for 10 frames.
        model.aggregator.half()
    return model


def make_room_for_geometry():
    """On a one-GPU server, clear the GPU for VGGT: drop the SAM 2 video tracker and the image-to-3D
    model, park the detector/segmenter/depth/CLIP weights on the CPU. Measured: without this every
    video after the first on a warm 9 GB server fell back to 4- or 6-frame chunks."""
    import gc
    from . import meshes
    tracker_model.cache_clear()
    with meshes.INIT_LOCK:meshes.MODELS.clear()
    registry().park(False);gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()


def release_multiview_model():
    """Free the ~5 GB VGGT weights; needed when every model shares one GPU."""
    import gc
    multiview_model.cache_clear();gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()


@torch.inference_mode()
def vggt_geometry(paths,directory,chunk=None):
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    model=multiview_model();chunk=int(chunk or os.getenv('RECON_VIDEO_CHUNK','10'))
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
def track_group(images,instances,clouds,confidence,device,sink=None):
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
    model(inference_session=session,frame_idx=0)
    def observed():
        for output in model.propagate_in_video_iterator(session,start_frame_idx=0):
            frame=int(output.frame_idx)
            masks=processor.post_process_masks(output.pred_masks.unsqueeze(0).cpu(),[images[frame].size[::-1]])[0]
            for index,mask in zip(ids,masks):
                mask=np.asarray(mask[0],bool)
                # Stage boundary: everything after this mask is deterministic CPU geometry.
                if sink is not None:sink[(frame,index)]=mask
                yield frame,index,mask
    return accumulate_track_points(ids,observed(),clouds,confidence)


def accumulate_track_points(ids,observations,clouds,confidence):
    """Per-track multi-view points from (frame, track_id, mask) in propagation order.

    Shared by live SAM 2 propagation and by offline refits of cached track masks.
    """
    fused={i:[] for i in ids};track_report=[];valid_by_frame={}
    for frame,index,mask in observations:
        if frame not in valid_by_frame:
            threshold=float(np.percentile(confidence[frame],25))
            valid_by_frame[frame]=(confidence[frame]>=threshold)&np.isfinite(clouds[frame]).all(-1)
        points=clouds[frame][mask&valid_by_frame[frame]][::3]
        if len(points):fused[index].append(points)
        track_report.append(dict(frame=frame,track_id=index,pixels=int(mask.sum()),sampled_points=len(points)))
    if not track_report:raise RuntimeError('SAM2 returned no object observations')
    return {i:fuse_track_points(chunks) for i,chunks in fused.items()},track_report


def reference_frame_points(group,clouds,reason):
    """Tracks whose propagation failed keep their reference-frame points only."""
    points={};records=[]
    for inst in group:
        mask=inst['mask']&np.isfinite(clouds[0]).all(-1)
        selected=clouds[0][mask][::3];track=inst['track_id']
        points[track]=selected
        records.append(dict(frame=0,track_id=track,pixels=int(mask.sum()),sampled_points=len(selected),
            fallback='reference_frame_only',reason=reason))
    return points,records


def track_workers(count):
    """One SAM 2 worker per configured GPU (RECON_GPU_IDS, default: the visible devices), capped by RECON_TRACK_WORKERS."""
    if not torch.cuda.is_available():return 1
    devices=len(registry().devices)
    return max(1,min(count,devices,int(os.getenv('RECON_TRACK_WORKERS') or devices)))


def track_instances(images,instances,clouds,confidence,sink=None):
    """Independent object memories can propagate concurrently on separate GPUs."""
    if not instances:raise ValueError('No masks available for video tracking')
    for index,inst in enumerate(instances):inst['track_id']=index
    workers=track_workers(len(instances))
    groups=[instances[index::workers] for index in range(workers)]
    fused={};observations=[]
    # VGGT activation buffers are no longer needed when propagation starts.
    if torch.cuda.is_available():torch.cuda.empty_cache()
    with ThreadPoolExecutor(max_workers=workers,thread_name_prefix='video-track') as pool:
        extra={} if sink is None else dict(sink=sink)
        futures=[pool.submit(track_group,images,group,clouds,confidence,registry().device(index),**extra)
                 for index,group in enumerate(groups)]
        for group,future in zip(groups,futures):
            try:
                points,records=future.result()
            except Exception as exc:
                # A worker failure cannot discard objects tracked on other GPUs.
                points,records=reference_frame_points(group,clouds,f'{type(exc).__name__}: {str(exc)[:250]}')
                if sink is not None:
                    for inst in group:
                        for key in [key for key in list(sink) if key[1]==inst['track_id']]:sink.pop(key,None)
            fused.update(points);observations.extend(records)
    observations.sort(key=lambda item:(item['frame'],item['track_id']))
    return fused,observations


def tsdf_fuse(images,depths,cameras,directory):
    import open3d as o3d
    from threadpoolctl import threadpool_limits
    # Open3D otherwise starts a team across the entire multi-socket server for
    # every small frame. Bound its loaded OpenMP runtime in this worker thread.
    with threadpool_limits(limits=4,user_api='openmp'):
        return _tsdf_fuse(o3d,images,depths,cameras,directory)


def _tsdf_fuse(o3d,images,depths,cameras,directory):
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
    if not torch.cuda.is_available() and os.getenv('RECON_ALLOW_CPU')!='1' and os.getenv('VISION_OFFLINE')!='1':
        # VGGT and SAM 2 propagation take tens of minutes on CPU and would hold the request and VIDEO_LOCK.
        raise RuntimeError('Video reconstruction needs a CUDA GPU (set RECON_ALLOW_CPU=1 to run it on CPU anyway)')
    config=dict(version=VIDEO_VERSION,photo_version=photo_version,table_width_m=table_width_m,
                chunk=os.getenv('RECON_VIDEO_CHUNK','10'),mesh=os.getenv('RECON_MESH','1'),
                threshold=os.getenv('RECON_DETECTION_THRESHOLD','.22'),auto_grid=os.getenv('RECON_AUTO_GRID','12'),
                min_conf=os.getenv('RECON_VIDEO_MIN_CONF','1.1'),max_range=os.getenv('RECON_VIDEO_MAX_RANGE_M','3.0'),
                static_furniture=os.getenv('RECON_VIDEO_STATIC_FURNITURE','1'),vggt_half=os.getenv('RECON_VGGT_HALF','1'),
                refine=os.getenv('RECON_REFINE','1'))
    digest=sha256(Path(path).read_bytes()+json.dumps(config,sort_keys=True).encode()).hexdigest()
    work=root_dir()/digest;work.mkdir(parents=True,exist_ok=True)
    with VIDEO_LOCK:
        from agatha.schema import Scene
        report_path=work/'reconstruction_report.json';scene_path=work/'scene.json'
        if use_cache and report_path.exists() and scene_path.exists():
            cached=json.loads(report_path.read_text())
            cached_scene=Scene.model_validate_json(scene_path.read_text())
            cached_tracking=cached.get('video',{}).get('stages',{}).get('tracking',{})
            cached_geometry=cached.get('video',{}).get('stages',{}).get('geometry',{})
            # A run degraded by a transient GPU shortage (smaller chunks, COLMAP) must not be served forever.
            degraded=cached_geometry.get('method')!='VGGT-1B' or str(cached_geometry.get('chunk'))!=config['chunk']
            tracked=(cached_tracking.get('status')=='ok' and cached_tracking.get('observations',0)>0
                     and any(cached_tracking.get('fused_point_counts',{}).values()))
            if cached.get('status')=='complete' and cached_scene.objects and not degraded and (tracked or cached.get('pipeline')==PHOTO_FALLBACK):
                cached['cache_hit']=True;cached['request_duration_s']=time.perf_counter()-start
                return cached_scene,cached
        stage_start=time.perf_counter()
        paths,sampling=sample_frames(path,work/'frames')
        video_report=dict(sampling=sampling,stages={},warnings=[],timings_s={'sampling':time.perf_counter()-stage_start})
        stage_start=time.perf_counter()
        if sampling['warning']:video_report['warnings'].append(sampling['warning'])
        try:
            single_gpu=len(registry().devices)<=1
            if single_gpu:make_room_for_geometry()
            images,depths,clouds,confidence,cameras,alignments,chunk_used=vggt_with_memory_retry(paths,work,video_report)
            video_report['stages']['geometry']=dict(method='VGGT-1B',alignments=alignments,frames=len(images),metric_scale=False,chunk=chunk_used)
        except Exception as exc:
            video_report['warnings'].append(f'VGGT failed: {type(exc).__name__}: {str(exc)[:300]}')
            if len(registry().devices)<=1:
                release_multiview_model();registry().park(True)
            try:
                images,depths,clouds,confidence,cameras,alignments=colmap_geometry(paths,work)
                video_report['stages']['geometry']=dict(method='COLMAP_poses_with_metric_monocular_depth',frames=len(images),metric_scale=False)
                video_report['stages']['colmap']=dict(status='ok',dense_method='Depth Anything V2; CPU COLMAP has no CUDA PatchMatch')
            except Exception as colmap_exc:
                video_report['stages']['colmap']=dict(status='failed',reason=f'{type(colmap_exc).__name__}: {str(colmap_exc)[:300]}')
                return photo_fallback(paths,table_width_m,video_report,start,'Multi-view geometry unavailable; reconstructed one sharp frame only.')
        video_report['timings_s']['geometry']=time.perf_counter()-stage_start
        stage_start=time.perf_counter()
        if len(registry().devices)<=1:
            release_multiview_model();registry().park(True)
        consistency=geometry_consistency(confidence,video_report['stages']['geometry']['method'])
        video_report['stages']['geometry']['consistency']=consistency
        if not consistency['usable']:
            return photo_fallback(paths,table_width_m,video_report,start,
                f"The video has no usable multi-view geometry (only {consistency['confident_fraction']:.0%} of pixels are consistent "
                'across views: a fixed camera or a moving scene); reconstructed its first sharp frame as a photo.',work)
        model=registry();detections=model.detect(images[0],include_people=True);masks=model.segment(images[0],detections)
        from .segmentation import uncovered_automatic_masks
        masks=uncovered_automatic_masks(masks)
        np.savez_compressed(work/'source_masks.npz',**{str(i):m['mask'] for i,m in enumerate(masks)})
        (work/'source_instances.json').write_text(json.dumps([
            {k:v for k,v in m.items() if k!='mask'} for m in masks]))
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
        instances,untracked=select_trackable(retained,confidence[0],video_report['stages']['geometry']['method'])
        video_report['stages']['reference_selection']=dict(tracked=len(instances),dropped=untracked)
        np.savez_compressed(work/'reference_masks.npz',**{str(i):inst['mask'] for i,inst in enumerate(instances)})
        (work/'reference_instances.json').write_text(json.dumps([
            {k:v for k,v in inst.items() if k!='mask'} for inst in instances]))
        video_report['timings_s']['detection_and_masks']=time.perf_counter()-stage_start
        stage_start=time.perf_counter()
        track_masks={}
        try:
            fused,tracking=track_instances(images,instances,clouds,confidence,sink=track_masks)
            record_tracking(video_report,fused,tracking,len(instances),work)
        except Exception as exc:
            fused={};track_masks={};video_report['stages']['tracking']=dict(status='failed',reason=f'{type(exc).__name__}: {str(exc)[:300]}')
            video_report['warnings'].append('Video tracking failed; body fits use the reference frame only.')
        video_report['timings_s']['tracking']=time.perf_counter()-stage_start
        # Last neural output. Everything below is replayable from this bundle (scripts/recon_refit.py).
        try:save_neural_bundle(work,images=images,detections=detections,masks=masks,instances=instances,
                track_masks=track_masks,video_report=video_report,table_width_m=table_width_m,digest=digest,
                geometry=(depths,clouds,confidence,cameras))
        except Exception as exc:video_report['warnings'].append(f'Neural bundle not saved: {type(exc).__name__}: {str(exc)[:200]}')
        scene,report=finalize_video(work,digest,images,depths,cameras,detections,masks,instances,fused,video_report,table_width_m,start,
                              track_masks=track_masks,clouds=clouds,confidence=confidence)
        if torch.cuda.is_available():torch.cuda.empty_cache()
        if not any(not obj.is_static for obj in scene.objects) and os.getenv('RECON_VIDEO_PHOTO_FALLBACK','1')=='1':
            return photo_fallback(paths,table_width_m,video_report,start,
                'No movable body had enough multi-view support; reconstructed the first sharp frame as a photo.',work)
        return scene,report


def geometry_consistency(confidence,method='VGGT-1B'):
    """Is there multi-view geometry to lift objects from?

    VGGT's depth confidence stays at its floor (1.0) when views do not agree: a camera on a
    tripod, or objects being moved. Measured: the walk-around clip has 26-49 % of pixels
    above 1.1, the fixed-camera clips 0-3 %.
    """
    if method!='VGGT-1B':return dict(usable=True,confident_fraction=None)
    minimum=float(os.getenv('RECON_VIDEO_MIN_CONF','1.1'));needed=float(os.getenv('RECON_VIDEO_MIN_CONSISTENT','0.10'))
    fraction=float((np.asarray(confidence)>=minimum).mean())
    return dict(usable=fraction>=needed,confident_fraction=fraction,required_fraction=needed,confidence_threshold=minimum)


def photo_fallback(paths,table_width_m,video_report,start,reason,work=None):
    """Single-frame reconstruction of a video that cannot be lifted from several views.

    Bodies a photo cannot place (below the floor, metres from the table) are removed, the
    result says plainly that it is one frame, and it is saved so a repeat upload is a cache hit.
    """
    scene,report=reconstruct(Image.open(paths[0]),table_width_m)
    limit=float(os.getenv('RECON_VIDEO_MAX_RANGE_M','3.0'));removed=[]
    for obj in list(scene.objects):
        distance=float(np.hypot(obj.position[0],obj.position[2]))
        if obj.is_static or (obj.position[1]>-.05 and distance<=limit+max(scene.support.size)/2):continue
        scene.objects.remove(obj);removed.append(dict(id=obj.id,label=obj.label,position=list(obj.position),
            reason='below_floor' if obj.position[1]<=-.05 else 'far_from_support'))
    report=dict(report,input='video',video=video_report,pipeline=PHOTO_FALLBACK,photo_run_id=report.get('run_id'),cache_hit=False)
    report['warnings']=[*report.get('warnings',[]),*video_report['warnings'],reason]
    report['video_fallback']=dict(reason=reason,removed=removed)
    report['body_count']=len(scene.objects);report['dynamic_body_count']=sum(not obj.is_static for obj in scene.objects)
    report['status']='complete';report['quality_status']='needs_review'
    report['duration_s']=time.perf_counter()-start;report['video_latency_pass']=report['duration_s']<180
    if work is not None and scene.objects:
        (Path(work)/'scene.json').write_text(scene.model_dump_json(indent=2))
        temporary=Path(work)/'reconstruction_report.tmp';temporary.write_text(json.dumps(report,indent=2,allow_nan=False,default=float))
        temporary.replace(Path(work)/'reconstruction_report.json')
    return scene,report


def vggt_with_memory_retry(paths,work,video_report):
    """VGGT with the configured chunk; on GPU OOM retry with smaller chunks instead of silently degrading to COLMAP."""
    first=int(os.getenv('RECON_VIDEO_CHUNK','10'));error=None
    for chunk in [first,*[c for c in (8,6,4) if c<first]]:
        try:
            result=vggt_geometry(paths,work,chunk)
            if chunk!=first:video_report['warnings'].append(f'VGGT ran out of GPU memory with {first}-frame chunks; used {chunk}-frame chunks (more chunk seams).')
            return (*result,chunk)
        except torch.OutOfMemoryError as exc:
            # Keep the message only: the traceback pins the failed chunk's activations on the GPU.
            error=str(exc)[:300]
        import gc
        gc.collect();torch.cuda.empty_cache()
    raise RuntimeError(f'VGGT out of GPU memory even with 4-frame chunks: {error}')


def select_trackable(instances,confidence0,method='VGGT-1B'):
    """Reference-frame instances worth tracking and lifting.

    Automatic SAM proposals are structure patches (pillar, baseboard) far more often
    than missed objects, and CLIP cannot tell on such crops. Detections whose pixels
    sit at VGGT's confidence floor (1.0) are distant clutter with no usable geometry.
    """
    minimum=float(os.getenv('RECON_VIDEO_MIN_CONF','1.1'));kept=[];dropped=[]
    for index,inst in enumerate(instances):
        reason=None
        if inst.get('mask_source')=='sam2_automatic':reason='automatic_mask_not_tracked'
        elif method=='VGGT-1B' and confidence0 is not None and inst['mask'].any():
            median=float(np.median(confidence0[inst['mask']]));inst['geometry_confidence']=median
            if median<minimum:reason='low_geometry_confidence'
        if reason:dropped.append(dict(index=index,track_id=inst.get('track_id'),label=inst['label'],reason=reason,
            geometry_confidence=inst.get('geometry_confidence')))
        else:kept.append(inst)
    return kept,dropped


def track_views(track_masks,clouds,confidence):
    """Per-track [(frame, points)] in first-camera coordinates from propagated masks."""
    views={};valid_by_frame={}
    for frame,track in sorted(track_masks):
        if frame not in valid_by_frame:
            threshold=float(np.percentile(confidence[frame],25))
            valid_by_frame[frame]=(confidence[frame]>=threshold)&np.isfinite(clouds[frame]).all(-1)
        mask=track_masks[(frame,track)];points=clouds[frame][mask&valid_by_frame[frame]]
        if len(points)>3000:points=points[::3]
        if len(points):views.setdefault(track,[]).append((frame,points))
    return views


def multiview_structure_points(clouds,confidence,stride=3):
    """Confident points of every frame (subsampled) with their frame index, for multi-frame plane extents."""
    points=[];frames=[]
    for frame,(cloud,conf) in enumerate(zip(clouds,confidence)):
        cloud=cloud[::stride,::stride].reshape(-1,3);conf=conf[::stride,::stride].reshape(-1)
        keep=np.isfinite(cloud).all(1)&(conf>=np.percentile(conf,25))
        points.append(cloud[keep]);frames.append(np.full(int(keep.sum()),frame))
    return np.concatenate(points),np.concatenate(frames)


def multiview_silhouettes(scene,report,cameras,track_masks,instances,max_frames=8):
    """Silhouette IoU of the final bodies against SAM 2 track masks in several frames.

    Uses each frame's own VGGT pose, the support slab as an occluder, and only the
    frames in which the track agreed with the fused object (kept_frames).
    """
    from agatha.schema import SceneObject
    from .assets import rasterize
    from .pipeline import frame_view
    pose=report['camera_pose'];factor=float(report['scale']['factor']);base=report['camera'];s=scene.support
    slab=SceneObject(id='support_slab',label='support',size=(s.size[0],s.thickness,s.size[1]),position=(0,s.height-s.thickness/2,0),is_static=True)
    drawn=scene.model_copy(update=dict(objects=[*scene.objects,slab]))
    tracks={inst.get('object_id'):inst['track_id'] for inst in report.get('detections',instances) if inst.get('object_id') and 'track_id' in inst}
    kept={item['id']:set(item.get('multiview',{}).get('consistent_frames',item.get('multiview',{}).get('kept_frames',[]))) for item in report.get('objects',[])}
    frames=sorted({frame for frame,_ in track_masks})
    if len(frames)>max_frames:frames=[frames[i] for i in np.linspace(0,len(frames)-1,max_frames).round().astype(int)]
    values={}
    for frame in frames:
        camera,frame_pose=frame_view(pose,factor,cameras[frame],base)
        labels,_=rasterize(drawn,camera,frame_pose)
        for index,obj in enumerate(scene.objects):
            track=tracks.get(obj.id);mask=track_masks.get((frame,track))
            if mask is None or mask.sum()<30 or (kept.get(obj.id) and frame not in kept[obj.id]):continue
            rendered=labels==index
            values.setdefault(obj.id,[]).append(float((rendered&mask).sum()/max((rendered|mask).sum(),1)))
    per_object={key:dict(mean_iou=float(np.mean(v)),frames=len(v)) for key,v in values.items()}
    return dict(frames=frames,per_object=per_object,mean_iou=float(np.mean([v['mean_iou'] for v in per_object.values()])) if per_object else None)


def record_tracking(video_report,fused,tracking,instance_count,work=None):
    """Tracking stage summary; recomputed by refits when the point filter changes."""
    video_report['stages']['tracking']=dict(method='SAM2_video_propagation',tracks=len(fused),observations=len(tracking),
        workers=track_workers(instance_count))
    failed_tracks=[row['track_id'] for row in tracking if row.get('fallback')]
    empty_tracks=[track for track,points in fused.items() if not len(points)]
    failed_tracks=sorted(set(failed_tracks)|set(empty_tracks))
    video_report['stages']['tracking']['status']='partial' if failed_tracks else 'ok'
    video_report['stages']['tracking']['reference_frame_fallback_tracks']=failed_tracks
    video_report['stages']['tracking']['fused_point_counts']={str(track):len(points) for track,points in fused.items()}
    video_report['stages']['tracking']['point_filter']='reference_frame_spatial_consistency'
    if failed_tracks:video_report['warnings'].append(f'{len(failed_tracks)} tracks fell back to reference-frame points after a worker failure.')
    if work is not None:
        (work/'tracks.json').write_text(json.dumps(tracking))
        np.savez_compressed(work/'object_points.npz',**{str(i):p for i,p in fused.items()})


def save_neural_bundle(work,*,images,detections,masks,instances,track_masks,video_report,table_width_m,digest,geometry=None):
    """Persist every neural output the deterministic stages consume (with multiview.npz/cameras.json)."""
    if geometry is not None and not ((Path(work)/'multiview.npz').exists() and (Path(work)/'cameras.json').exists()):
        # vggt_geometry writes these itself; the COLMAP fallback does not.
        depths,clouds,confidence,cameras=geometry
        np.savez_compressed(Path(work)/'multiview.npz',depth=depths,confidence=confidence,points=clouds)
        (Path(work)/'cameras.json').write_text(json.dumps(cameras,indent=2))
    def packed(items):
        return np.packbits(np.stack([np.asarray(m,bool) for m in items]).reshape(len(items),-1),axis=1) if len(items) else np.zeros((0,0),np.uint8)
    def meta(items):return [{k:v for k,v in item.items() if k!='mask'} for item in items]
    keys=sorted(track_masks)
    np.savez_compressed(Path(work)/'neural_bundle.npz',images=np.stack([np.asarray(im) for im in images]),
        masks=packed([m['mask'] for m in masks]),instance_masks=packed([i['mask'] for i in instances]),
        track_keys=np.array(keys,dtype=np.int64).reshape(-1,2),track_masks=packed([track_masks[k] for k in keys]),
        shape=np.array([images[0].height,images[0].width]))
    (Path(work)/'neural_bundle.json').write_text(json.dumps(dict(version=VIDEO_VERSION,digest=digest,table_width_m=table_width_m,
        detections=detections,masks=meta(masks),instances=meta(instances),video_report=video_report)))


def load_neural_bundle(work):
    """Inverse of save_neural_bundle plus the VGGT arrays written by vggt_geometry."""
    work=Path(work);meta=json.loads((work/'neural_bundle.json').read_text())
    with np.load(work/'neural_bundle.npz') as data:
        height,width=map(int,data['shape']);count=height*width
        def unpack(name):return np.unpackbits(data[name],axis=1)[:,:count].reshape(-1,height,width).astype(bool)
        images=[Image.fromarray(frame) for frame in data['images']]
        masks=[dict(item,mask=mask) for item,mask in zip(meta['masks'],unpack('masks'))]
        instances=[dict(item,mask=mask) for item,mask in zip(meta['instances'],unpack('instance_masks'))]
        track_masks={(int(frame),int(track)):mask for (frame,track),mask in zip(data['track_keys'],unpack('track_masks'))} if len(data['track_keys']) else {}
    with np.load(work/'multiview.npz') as geometry:
        depths,clouds,confidence=geometry['depth'],geometry['points'],geometry['confidence']
    cameras=json.loads((work/'cameras.json').read_text())
    return dict(images=images,depths=depths,clouds=clouds,confidence=confidence,cameras=cameras,
        detections=meta['detections'],masks=masks,instances=instances,track_masks=track_masks,
        video_report=meta['video_report'],table_width_m=meta['table_width_m'],digest=meta['digest'])


def refit_tracks(instances,track_masks,clouds,confidence):
    """Recompute fused track points from cached SAM 2 masks (no model)."""
    tracked=sorted({track for _,track in track_masks})
    fused,tracking=accumulate_track_points(tracked,((frame,track,track_masks[(frame,track)]) for frame,track in sorted(track_masks)),clouds,confidence) if tracked else ({},[])
    missing=[inst for inst in instances if inst['track_id'] not in tracked]
    if missing:
        points,records=reference_frame_points(missing,clouds,'no cached propagation masks')
        fused.update(points);tracking.extend(records)
    tracking.sort(key=lambda item:(item['frame'],item['track_id']))
    return fused,tracking


def finalize_video(work,digest,images,depths,cameras,detections,masks,instances,fused,video_report,table_width_m,start,*,fusion=True,
                   track_masks=None,clouds=None,confidence=None):
    """Deterministic tail of reconstruct_video: TSDF export, scene lift, quality checks, saved bundle."""
    from agatha.schema import Scene
    report_path=work/'reconstruction_report.json';scene_path=work/'scene.json'
    stage_start=time.perf_counter()
    if fusion:
        try: video_report['stages']['fusion']=tsdf_fuse(images,depths,cameras,work)
        except Exception as exc:
            video_report['stages']['fusion']=dict(status='failed',reason=f'{type(exc).__name__}: {str(exc)[:300]}')
            video_report['warnings'].append('TSDF fusion failed; raw multi-view geometry is retained.')
    video_report['timings_s']['fusion']=time.perf_counter()-stage_start
    k=np.array(cameras[0]['intrinsics']);camera=dict(fx=k[0,0],fy=k[1,1],cx=k[0,2],cy=k[1,2],
        width=images[0].width,height=images[0].height,source='VGGT_predicted')
    method=video_report['stages'].get('geometry',{}).get('method','VGGT-1B')
    # Idempotent for live runs; drops low-evidence tracks from bundles captured before the filter existed.
    instances,untracked=select_trackable(instances,None if confidence is None else confidence[0],method)
    selection=video_report['stages'].setdefault('reference_selection',dict(dropped=[]))
    selection['dropped']=[*selection.get('dropped',[]),*untracked];selection['tracked']=len(instances)
    override=dict(depth=depths[0],camera=camera,method=method,hash=digest,
                  detections=detections,masks=[*instances,*[m for m in masks if m['label'] in ('table','wall','floor')]],fused=fused)
    if track_masks and clouds is not None and confidence is not None:
        override['track_views']=track_views(track_masks,clouds,confidence)
        override['track_masks']=track_masks;override['cameras']=cameras
    if clouds is not None and confidence is not None:
        override['multiview_points']=multiview_structure_points(clouds,confidence)
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
    if track_masks:
        try:metrics['multiview_silhouette']=multiview_silhouettes(scene,report,cameras,track_masks,instances)
        except Exception as exc:report['warnings'].append(f'Multi-frame silhouette metric failed: {type(exc).__name__}: {str(exc)[:200]}')
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
