"""Diagnostics for a reconstruction refit: per-frame overlays, plan/elevation plots, body table.

Used by scripts/recon_refit.py; can also be run on an existing refit output:
  .venv-reconstruction/bin/python scripts/recon_overlay.py --run <captured run> --out <refit out dir>
All reprojections use agatha.scene_reconstruction.assets.rasterize (the z-buffer the live
metric uses) with the VGGT pose of each frame expressed in the scene's world frame.
"""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
MIN_TRACK_PIXELS=30


def color(index):
    palette=[(230,25,75),(60,180,75),(0,130,200),(245,130,48),(145,30,180),(70,240,240),(240,50,230),(210,245,60),
             (250,190,212),(0,128,128),(220,190,255),(170,110,40),(255,250,200),(128,0,0),(170,255,195),(128,128,0),
             (255,215,180),(0,0,128),(255,225,25),(100,100,255)]
    return palette[index%len(palette)]


def frame_cameras(report,cameras):
    """Per-frame (intrinsics, pose) in the scene world frame; frame 0 is report['camera_pose']."""
    pose=report['camera_pose'];world_to_camera=np.array(pose['world_to_camera']);origin=np.array(pose['origin_world'])
    factor=float(report['scale']['factor']);base=report['camera'];result=[]
    for camera in cameras:
        k=np.array(camera['intrinsics']);rotation=np.array(camera['camera_to_world']);center=np.array(camera['center'])
        result.append((dict(fx=k[0,0],fy=k[1,1],cx=k[0,2],cy=k[1,2],width=base['width'],height=base['height']),
                       dict(origin_world=(origin+factor*center@world_to_camera).tolist(),
                            world_to_camera=(rotation.T@world_to_camera).tolist())))
    return result


def to_world(points,report):
    """First-camera (VGGT unit) points -> scene world metres."""
    pose=report['camera_pose']
    return np.asarray(points)*float(report['scale']['factor'])@np.array(pose['world_to_camera'])+np.array(pose['origin_world'])


def with_support(scene):
    """The Support slab is not a SceneObject; add it so it is drawn like every other body."""
    from agatha.schema import SceneObject
    s=scene.support
    slab=SceneObject(id='support',label='support',size=(s.size[0],s.thickness,s.size[1]),
        position=(0,s.height-s.thickness/2,0),is_static=True)
    return scene.model_copy(update=dict(objects=[*scene.objects,slab]))


def world_vertices(obj):
    from agatha.scene_reconstruction.assets import object_mesh,rotation
    return object_mesh(obj).vertices@rotation(obj.rotation_euler).T+np.array(obj.position)


def put_text(image,text,origin,scale=.42,fill=(255,255,255)):
    x,y=int(origin[0]),int(origin[1])
    cv2.putText(image,text,(x,y),cv2.FONT_HERSHEY_SIMPLEX,scale,(0,0,0),3,cv2.LINE_AA)
    cv2.putText(image,text,(x,y),cv2.FONT_HERSHEY_SIMPLEX,scale,fill,1,cv2.LINE_AA)


def short(obj_id):return obj_id.replace('object_','o').replace('table_','T').replace('wall_','W').replace('support','SUP')


def panel_masks(frame,masks,labels,zoom):
    """masks: {key: bool mask}; labels: {key: text}."""
    canvas=cv2.resize(frame,None,fx=zoom,fy=zoom,interpolation=cv2.INTER_CUBIC).astype(np.float32)
    notes=[]
    for index,(key,mask) in enumerate(sorted(masks.items(),key=lambda kv:str(kv[0]))):
        if mask.sum()<MIN_TRACK_PIXELS:continue
        big=cv2.resize(mask.astype(np.uint8),(canvas.shape[1],canvas.shape[0]),interpolation=cv2.INTER_NEAREST)>0
        tint=np.array(color(index),np.float32);canvas[big]=.5*canvas[big]+.5*tint
        contours,_=cv2.findContours(big.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas,contours,-1,color(index),1)
        yy,xx=np.where(big);notes.append((labels.get(key,str(key)),(np.median(xx)-20,np.median(yy)),color(index)))
    canvas=np.clip(canvas,0,255).astype(np.uint8)
    for text,origin,tint in notes:put_text(canvas,text,origin)
    return canvas


def panel_bodies(frame,ids,objects,zoom):
    canvas=cv2.resize(frame,None,fx=zoom,fy=zoom,interpolation=cv2.INTER_CUBIC).astype(np.float32)
    big_ids=cv2.resize(ids.astype(np.int32),(canvas.shape[1],canvas.shape[0]),interpolation=cv2.INTER_NEAREST)
    notes=[]
    for index,obj in enumerate(objects):
        region=big_ids==index
        if region.sum()<20:continue
        tint=(150,150,150) if obj.id=='support' else (90,90,200) if obj.is_static else color(index)
        canvas[region]=.45*canvas[region]+.55*np.array(tint,np.float32)
        contours,_=cv2.findContours(region.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas,contours,-1,tint,2)
        yy,xx=np.where(region);notes.append((f'{short(obj.id)}:{obj.label[:14]}',(np.median(xx)-20,np.median(yy))))
    canvas=np.clip(canvas,0,255).astype(np.uint8)
    for text,origin in notes:put_text(canvas,text,origin)
    return canvas


def titled(image,title):
    bar=np.full((26,image.shape[1],3),30,np.uint8);put_text(bar,title,(6,18),.5);return np.vstack([bar,image])


def plots(out,scene,report,cloud_world,cloud_height_color,camera_centers,track_points):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon,Circle,Rectangle
    objects=with_support(scene).objects
    def footprint(obj):
        from agatha.scene_reconstruction.assets import rotation
        w,_,d=obj.size;corners=np.array([[-w/2,0,-d/2],[w/2,0,-d/2],[w/2,0,d/2],[-w/2,0,d/2]])
        return (corners@rotation(obj.rotation_euler).T+np.array(obj.position))[:,[0,2]]
    centers=np.array([o.position for o in objects]) if objects else np.zeros((1,3))
    extra=camera_centers if camera_centers is not None and len(camera_centers) else centers
    everything=np.vstack([centers,extra])
    lo=np.floor((everything.min(0)-1.)*2)/2;hi=np.ceil((everything.max(0)+1.)*2)/2
    lo=np.maximum(lo,[-8,-1,-8]);hi=np.minimum(hi,[8,4,8])
    def grid(ax,xlim,ylim):
        ax.set_xlim(*xlim);ax.set_ylim(*ylim);ax.set_aspect('equal')
        ax.set_xticks(np.arange(np.floor(xlim[0]*2)/2,xlim[1]+.01,.5));ax.set_yticks(np.arange(np.floor(ylim[0]*2)/2,ylim[1]+.01,.5))
        ax.grid(True,color='#bbbbbb',linewidth=.5);ax.tick_params(labelsize=6)
    # ---- top-down plan
    fig,ax=plt.subplots(figsize=(13,13*(hi[2]-lo[2])/max(hi[0]-lo[0],1e-6)+.6),dpi=110)
    if cloud_world is not None and len(cloud_world):
        ax.scatter(cloud_world[:,0],cloud_world[:,2],c=cloud_height_color,s=.3,alpha=.35,linewidths=0,cmap='viridis',vmin=0,vmax=2.2)
    for index,obj in enumerate(objects):
        tint='#555555' if obj.id=='support' else '#3355cc' if obj.is_static else np.array(color(index))/255
        if obj.shape=='cylinder' and not obj.is_static:
            ax.add_patch(Circle((obj.position[0],obj.position[2]),obj.size[0]/2,fill=False,edgecolor=tint,linewidth=1.6))
        else:
            ax.add_patch(Polygon(footprint(obj),closed=True,fill=obj.is_static,facecolor=tint,alpha=.18 if obj.is_static else 1,edgecolor=tint,linewidth=1.6))
            if obj.is_static:ax.add_patch(Polygon(footprint(obj),closed=True,fill=False,edgecolor=tint,linewidth=1.6))
        ax.annotate(f'{short(obj.id)} {obj.label[:12]}\ny={obj.position[1]:.2f}',(obj.position[0],obj.position[2]),fontsize=6,
                    color='black',ha='center',va='center',bbox=dict(boxstyle='round,pad=.1',fc='white',ec='none',alpha=.6))
    if camera_centers is not None and len(camera_centers):
        ax.plot(camera_centers[:,0],camera_centers[:,2],'-o',color='red',markersize=2.5,linewidth=1)
        ax.annotate('cam0',(camera_centers[0,0],camera_centers[0,2]),color='red',fontsize=7)
        ax.annotate(f'cam{len(camera_centers)-1}',(camera_centers[-1,0],camera_centers[-1,2]),color='red',fontsize=7)
    grid(ax,(lo[0],hi[0]),(hi[2],lo[2]))
    ax.set_xlabel('world X (m)');ax.set_ylabel('world Z (m)  [camera 0 looks toward -Z, top of plot]')
    ax.set_title('Top-down: footprints (bodies coloured, statics blue, support grey), red = camera path, dots = VGGT cloud coloured by height 0-2.2 m; grid 0.5 m',fontsize=8)
    fig.tight_layout();fig.savefig(out/'plan_topdown.png');plt.close(fig)
    # ---- elevations
    fig,axes=plt.subplots(2,1,figsize=(13,11),dpi=110)
    for ax,axis,name in ((axes[0],0,'X'),(axes[1],2,'Z')):
        if cloud_world is not None and len(cloud_world):
            ax.scatter(cloud_world[:,axis],cloud_world[:,1],s=.3,alpha=.25,color='#888888',linewidths=0)
        for index,obj in enumerate(objects):
            vertices=world_vertices(obj);low,high=vertices.min(0),vertices.max(0)
            tint='#555555' if obj.id=='support' else '#3355cc' if obj.is_static else np.array(color(index))/255
            ax.add_patch(Rectangle((low[axis],low[1]),high[axis]-low[axis],high[1]-low[1],fill=obj.is_static,facecolor=tint,
                                   alpha=.25 if obj.is_static else 1,edgecolor=tint,linewidth=1.5))
            if obj.is_static:ax.add_patch(Rectangle((low[axis],low[1]),high[axis]-low[axis],high[1]-low[1],fill=False,edgecolor=tint,linewidth=1.5))
            ax.annotate(f'{short(obj.id)} {obj.label[:10]}',((low[axis]+high[axis])/2,high[1]),fontsize=6,ha='center',va='bottom')
        if camera_centers is not None and len(camera_centers):
            ax.plot(camera_centers[:,axis],camera_centers[:,1],'-o',color='red',markersize=2.5,linewidth=1)
        ax.axhline(0,color='black',linewidth=1.5)
        ax.axhline(scene.support.height,color='#555555',linewidth=.8,linestyle='--')
        grid(ax,(lo[axis],hi[axis]),(-.5,max(2.5,min(4,hi[1]))))
        ax.set_xlabel(f'world {name} (m)');ax.set_ylabel('height Y (m)')
        ax.set_title(f'Side elevation along {name}: body AABBs, black = floor y=0, dashed = support top {scene.support.height:.2f} m, grey dots = VGGT cloud, red = cameras',fontsize=8)
    fig.tight_layout();fig.savefig(out/'elevation_side.png');plt.close(fig)


def diagnose(run,out,scene=None,report=None,bundle=None,frames=6,iou_stride=1):
    from agatha.schema import Scene
    from agatha.scene_reconstruction.assets import rasterize
    run=Path(run);out=Path(out)
    if report is None:report=json.loads((out/'report.json').read_text())
    if scene is None:scene=Scene.model_validate_json((out/'scene.json').read_text())
    video=(run/'neural_bundle.json').exists()
    if video and bundle is None:
        from agatha.scene_reconstruction import video as video_module
        bundle=video_module.load_neural_bundle(run)
    if video and 'fused' not in bundle:
        from agatha.scene_reconstruction import video as video_module
        bundle['fused'],bundle['tracking']=video_module.refit_tracks(bundle['instances'],bundle['track_masks'],bundle['clouds'],bundle['confidence'])
    drawn=with_support(scene);objects=drawn.objects;index_of={obj.id:i for i,obj in enumerate(objects)}
    detections=report.get('detections',[]);details={d['id']:d for d in report.get('objects',[])}
    by_object={d['object_id']:d for d in detections if d.get('object_id')}
    have_pose='camera_pose' in report and 'camera' in report and 'scale' in report
    per_frame_iou={obj.id:[] for obj in scene.objects};observed_frames={obj.id:0 for obj in scene.objects}
    camera_centers=None;cloud_world=None;cloud_color=None;zoom=2
    if video and have_pose:
        cams=frame_cameras(report,bundle['cameras']);count=len(cams)
        camera_centers=np.array([pose['origin_world'] for _,pose in cams])
        track_of={obj_id:d.get('track_id') for obj_id,d in by_object.items()}
        label_of_track={inst['track_id']:f"{inst['track_id']}:{inst['label'][:10]}" for inst in bundle['instances']}
        chosen=sorted(set(np.linspace(0,count-1,max(1,frames)).round().astype(int).tolist()))
        metric_frames=sorted(set(range(0,count,max(1,iou_stride)))|set(chosen))
        for frame in metric_frames:
            camera,pose=cams[frame];ids,_=rasterize(drawn,camera,pose)
            masks={track:bundle['track_masks'][(frame,track)] for (f,track) in bundle['track_masks'] if f==frame}
            for obj in scene.objects:
                track=track_of.get(obj.id);mask=masks.get(track)
                if mask is None or mask.sum()<MIN_TRACK_PIXELS:continue
                rendered=ids==index_of[obj.id];observed_frames[obj.id]+=1
                per_frame_iou[obj.id].append(float((rendered&mask).sum()/max((rendered|mask).sum(),1)))
            if frame in chosen:
                image=np.asarray(bundle['images'][frame])
                left=titled(cv2.resize(image,None,fx=zoom,fy=zoom,interpolation=cv2.INTER_CUBIC),f'frame {frame}/{count-1}: input')
                middle=titled(panel_masks(image,masks,label_of_track,zoom),'SAM 2 track masks (track id:label)')
                right=titled(panel_bodies(image,ids,objects,zoom),'reconstructed bodies reprojected with this frame\'s VGGT pose')
                cv2.imwrite(str(out/f'overlay_frame_{frame:03}.png'),np.hstack([left,middle,right])[:,:,::-1])
                cv2.imwrite(str(out/f'bodies_frame_{frame:03}.png'),right[:,:,::-1])
        confidence=bundle['confidence'];clouds=bundle['clouds'];sample=[]
        for frame in range(0,count,3):
            keep=(confidence[frame]>=np.percentile(confidence[frame],40))&np.isfinite(clouds[frame]).all(-1)
            sample.append(clouds[frame][keep][::23])
        cloud_world=to_world(np.concatenate(sample),report)
        near=(np.abs(cloud_world)<[8,4,8]).all(1)&(cloud_world[:,1]>-.3);cloud_world=cloud_world[near];cloud_color=cloud_world[:,1]
    elif have_pose:
        # Photo: one calibrated view; the saved masks.npz holds the kept instance masks by object id.
        camera=dict(report['camera']);pose=report['camera_pose'];camera_centers=np.array([pose['origin_world']])
        from PIL import Image
        image=np.asarray(Image.open(out/'original.jpg').convert('RGB'))
        with np.load(out/'masks.npz') as stored:masks={key:stored[key] for key in stored.files}
        ids,_=rasterize(drawn,camera,pose)
        for obj in scene.objects:
            mask=masks.get(obj.id)
            if mask is None or mask.sum()<MIN_TRACK_PIXELS:continue
            rendered=ids==index_of[obj.id];observed_frames[obj.id]+=1
            per_frame_iou[obj.id].append(float((rendered&mask).sum()/max((rendered|mask).sum(),1)))
        zoom=1
        labels={key:f"{short(key)}:{by_object.get(key,{}).get('label','?')[:10]}" for key in masks}
        left=titled(image,'input (resized)');middle=titled(panel_masks(image,masks,labels,zoom),'kept instance masks (object id:label)')
        right=titled(panel_bodies(image,ids,objects,zoom),'reconstructed bodies reprojected')
        cv2.imwrite(str(out/'overlay_frame_000.png'),np.hstack([left,middle,right])[:,:,::-1])
        cv2.imwrite(str(out/'bodies_frame_000.png'),right[:,:,::-1])
        cv2.imwrite(str(out/'masks_frame_000.png'),middle[:,:,::-1])
        depth_path=out/'depth_m.npy'
        if depth_path.exists():
            from agatha.scene_reconstruction.geometry import backproject
            depth=np.load(depth_path);points=backproject(depth,camera)[::6,::6].reshape(-1,3)
            cloud_world=points@np.array(pose['world_to_camera'])+np.array(pose['origin_world'])
            near=(np.abs(cloud_world)<[8,4,8]).all(1);cloud_world=cloud_world[near];cloud_color=cloud_world[:,1]
    try:plots(out,scene,report,cloud_world,cloud_color,camera_centers,None)
    except Exception as exc:print(f'[recon_overlay] plots failed: {type(exc).__name__}: {exc}',file=sys.stderr)
    # ---- per-body table
    metrics=report.get('metrics',{});contacts=metrics.get('contacts_after_settle',{});stability=metrics.get('stability',{})
    gaps=contacts.get('support_distance_m',{});drift=stability.get('per_object_max_drift_m',{})
    frame0={item['id']:item for item in metrics.get('per_object',[])}
    support_top={'support':scene.support.height,'floor':0.}
    support_top.update({o.id:o.position[1]+o.size[1]/2 for o in scene.objects if o.is_static})
    rows=[];records=[]
    header=['id','label','shape','size w,h,d (m)','position x,y,z (m)','support','gap(m)','bottom-top(m)','pts','conf/mask/clip','frames','cam_dist(m)','IoU_frames','IoU_f0','drift(mm)']
    for obj in scene.objects:
        d=by_object.get(obj.id,{});info=details.get(obj.id,{})
        target=None if obj.is_static else (obj.supported_by or 'support')
        gap=gaps.get(obj.id)
        low=float(world_vertices(obj)[:,1].min())
        geometric=None if obj.is_static or target not in support_top else low-support_top[target]
        distance=None
        if camera_centers is not None and len(camera_centers):
            distance=float(np.min(np.linalg.norm(camera_centers[:,[0,2]]-np.array(obj.position)[[0,2]],axis=1)))
        ious=per_frame_iou[obj.id];mean_iou=float(np.mean(ious)) if ious else None
        points=info.get('fused_multiview_points') or info.get('valid_points')
        record=dict(id=obj.id,label=obj.label,shape=obj.shape,shape_method=obj.shape_method,is_static=obj.is_static,
            size=[float(v) for v in obj.size],position=[float(v) for v in obj.position],rotation_euler=[float(v) for v in obj.rotation_euler],
            supported_by=target,support_gap_m=gap,bottom_minus_support_top_m=geometric,points=points,valid_points=info.get('valid_points'),
            fused_multiview_points=info.get('fused_multiview_points'),fallback=info.get('fallback'),
            detection_confidence=d.get('confidence'),mask_score=d.get('mask_score'),classifier_label=d.get('classifier_label'),
            classifier_confidence=d.get('classifier_confidence'),mask_source=d.get('mask_source'),track_id=d.get('track_id'),
            frames_observed=observed_frames[obj.id],camera_path_distance_m=distance,silhouette_iou_frames_mean=mean_iou,
            silhouette_iou_per_frame=ious,silhouette_iou_frame0=frame0.get(obj.id,{}).get('reprojection_iou'),
            max_drift_5s_m=drift.get(obj.id))
        records.append(record)
        fmt=lambda value,spec='.2f':'-' if value is None else format(value,spec)
        rows.append([short(obj.id)+('*' if obj.is_static else ''),obj.label[:22],f'{obj.shape}/{(obj.shape_method or "")[:26]}',
            ','.join(f'{v:.2f}' for v in obj.size),','.join(f'{v:+.2f}' for v in obj.position),str(target or '-').replace('object_','o'),
            fmt(gap,'+.3f'),fmt(geometric,'+.3f'),str(points if points is not None else '-'),
            f"{fmt(d.get('confidence'))}/{fmt(d.get('mask_score'))}/{fmt(d.get('classifier_confidence'))}",
            str(observed_frames[obj.id]),fmt(distance),fmt(mean_iou),fmt(frame0.get(obj.id,{}).get('reprojection_iou')),
            fmt(None if drift.get(obj.id) is None else drift[obj.id]*1000,'.1f')])
    widths=[max(len(str(row[i])) for row in [header,*rows]) for i in range(len(header))]
    line=lambda row:'  '.join(str(cell).ljust(width) for cell,width in zip(row,widths))
    table='\n'.join([line(header),line(['-'*w for w in widths]),*[line(row) for row in rows]])
    table+=(f'\n(* static)  support slab: size {scene.support.size[0]:.2f} x {scene.support.size[1]:.2f} m, top at {scene.support.height:.3f} m; '
            'gap = Bullet closest distance to declared support after settle; bottom-top = collision bottom minus support top; '
            'conf/mask/clip = detector confidence / SAM mask score at the reference frame / CLIP confidence (SAM 2 propagation scores are not kept); '
            'IoU_frames = mean over observed frames of IoU(z-buffered body, SAM 2 track mask); IoU_f0 = live-pipeline metric on frame 0.')
    dynamic=[r for r in records if not r['is_static']]
    scored=[r['silhouette_iou_frames_mean'] for r in dynamic if r['silhouette_iou_frames_mean'] is not None]
    sizes=[max(r['size']) for r in dynamic]
    summary=dict(kind='video' if video else 'photo',body_count=len(scene.objects),dynamic_body_count=len(dynamic),
        quality_status=report.get('quality_status'),quality_checks=report.get('quality_checks'),
        reprojection_mean_iou_frame0=metrics.get('reprojection_mean_iou'),
        silhouette_iou_all_frames_mean=float(np.mean(scored)) if scored else None,
        float_count=contacts.get('float_count'),floating_ids=contacts.get('floating_ids'),
        penetration_count=contacts.get('penetration_count'),max_drift_5s_m=stability.get('max_drift_m'),
        bodies_larger_than_1p5m=[r['id'] for r in dynamic if max(r['size'])>1.5],
        max_dynamic_extent_m=max(sizes) if sizes else None,
        max_abs_bottom_minus_support_top_m=max((abs(r['bottom_minus_support_top_m']) for r in dynamic if r['bottom_minus_support_top_m'] is not None),default=None),
        support=dict(size=list(scene.support.size),height=scene.support.height),scale=report.get('scale',{}).get('factor'),
        scale_source=report.get('scale',{}).get('source'))
    (out/'bodies.json').write_text(json.dumps(dict(summary=summary,bodies=records),indent=2))
    (out/'bodies.txt').write_text(table+'\n')
    fmt=lambda value,spec='.3f':'n/a' if value is None else format(value,spec)
    headline=[f"bodies {summary['body_count']} ({summary['dynamic_body_count']} dynamic)   quality_status {summary['quality_status']}   checks {summary['quality_checks']}",
        f"silhouette IoU: frame0 (live metric) {fmt(summary['reprojection_mean_iou_frame0'])}   all observed frames {fmt(summary['silhouette_iou_all_frames_mean'])}",
        f"floating {summary['float_count']} {summary['floating_ids']}   penetrations {summary['penetration_count']}   5 s max drift {fmt(None if summary['max_drift_5s_m'] is None else summary['max_drift_5s_m']*1000,'.1f')} mm",
        f"dynamic bodies > 1.5 m: {summary['bodies_larger_than_1p5m']}   largest dynamic extent {fmt(summary['max_dynamic_extent_m'],'.2f')} m   scale x{fmt(summary['scale'])} ({summary['scale_source']})",
        f"support {summary['support']['size'][0]:.2f} x {summary['support']['size'][1]:.2f} m at {summary['support']['height']:.3f} m"]
    return dict(table=table,headline=headline,summary=summary)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--run',required=True);parser.add_argument('--out',required=True)
    parser.add_argument('--frames',type=int,default=6);parser.add_argument('--iou-stride',type=int,default=1)
    args=parser.parse_args();result=diagnose(args.run,args.out,frames=args.frames,iou_stride=args.iou_stride)
    print(result['table']);print()
    for item in result['headline']:print(item)
