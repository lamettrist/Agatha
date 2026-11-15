"""Render actual collision surfaces and export textured GLBs separately from collision."""
import json
import math
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw
from agatha.schema import Scene


def rotation(euler):
    import pybullet as p
    return np.array(p.getMatrixFromQuaternion(p.getQuaternionFromEuler(euler))).reshape(3,3)


def object_mesh(obj):
    import trimesh
    parts=[]
    definitions=obj.collision.parts if obj.collision else []
    for part in definitions:
        if part.type=='box': mesh=trimesh.creation.box(extents=part.size)
        elif part.type=='cylinder':
            mesh=trimesh.creation.cylinder(radius=part.radius,height=part.height,sections=48)
            mesh.apply_transform(trimesh.transformations.rotation_matrix(-math.pi/2,[1,0,0]))
        else: mesh=trimesh.convex.convex_hull(np.asarray(part.vertices))
        transform=np.eye(4);transform[:3,:3]=rotation(part.rotation_euler);transform[:3,3]=part.position
        mesh.apply_transform(transform);parts.append(mesh)
    if not parts: return trimesh.creation.box(extents=obj.size)
    return trimesh.util.concatenate(parts)


def camera_points(world, camera_pose):
    return (np.asarray(world)-np.asarray(camera_pose['origin_world']))@np.asarray(camera_pose['world_to_camera']).T


def project(points, camera):
    z=points[:,2]; safe=np.maximum(z,1e-5)
    return np.c_[points[:,0]/safe*camera['fx']+camera['cx'],points[:,1]/safe*camera['fy']+camera['cy']]


def rasterize(scene, camera, camera_pose):
    """CPU triangle z-buffer in the original calibrated camera (no bbox proxy metrics)."""
    w,h=camera['width'],camera['height']; zbuf=np.full((h,w),np.inf,np.float32)
    ids=np.full((h,w),-1,np.int32)
    for index,obj in enumerate(scene.objects):
        mesh=object_mesh(obj)
        world=mesh.vertices@rotation(obj.rotation_euler).T+obj.position
        points=camera_points(world,camera_pose); pixels=project(points,camera)
        for face in mesh.faces:
            tri=pixels[face]; depths=points[face,2]
            if np.any(depths<=.02): continue
            x0,y0=np.maximum(np.floor(tri.min(0)).astype(int),0)
            x1,y1=np.minimum(np.ceil(tri.max(0)).astype(int),[w-1,h-1])
            if x1<x0 or y1<y0: continue
            xx,yy=np.meshgrid(np.arange(x0,x1+1)+.5,np.arange(y0,y1+1)+.5)
            a,b,c=tri; denom=(b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1])
            if abs(denom)<1e-8: continue
            u=((b[1]-c[1])*(xx-c[0])+(c[0]-b[0])*(yy-c[1]))/denom
            v=((c[1]-a[1])*(xx-c[0])+(a[0]-c[0])*(yy-c[1]))/denom
            t=1-u-v; inside=(u>=0)&(v>=0)&(t>=0)
            inverse=u/depths[0]+v/depths[1]+t/depths[2]
            z=1/np.maximum(inverse,1e-9)
            patch=zbuf[y0:y1+1,x0:x1+1]; replace=inside & (z<patch)
            patch[replace]=z[replace]; ids[y0:y1+1,x0:x1+1][replace]=index
    return ids,zbuf


def write_cloud(path,cloud,rgb,valid):
    # Cap export density independently of inference/fit density.
    points=cloud[valid][::2]; colors=rgb[valid][::2]
    with Path(path).open('w') as f:
        f.write('ply\nformat ascii 1.0\nelement vertex '+str(len(points))+'\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')
        for xyz,color in zip(points,colors):
            f.write(' '.join(f'{v:.6f}' for v in xyz)+' '+' '.join(str(int(v)) for v in color)+'\n')


def export_visual(obj,image,inst,camera,camera_pose,path):
    import trimesh
    mesh=object_mesh(obj)
    points=camera_points(mesh.vertices@rotation(obj.rotation_euler).T+obj.position,camera_pose)
    uv=project(points,camera)/np.array(image.size)
    # Crop the source image and preserve its projection rather than stretching a bbox.
    x1,y1,x2,y2=map(int,inst['bbox']); crop=image.crop((x1,y1,x2,y2))
    px=uv*np.array(image.size)
    uv=(px-[x1,y1])/np.maximum([x2-x1,y2-y1],1); uv[:,1]=1-uv[:,1]
    mesh.visual=trimesh.visual.TextureVisuals(uv=uv,image=crop)
    mesh.export(path)


def write_debug(directory,image,depth,cloud,valid,instances,scene,camera,camera_pose,report):
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
    image.save(directory/'original.jpg')
    finite=np.isfinite(depth)&(depth>0)
    lo,hi=np.percentile(depth[finite],[2,98]) if finite.any() else (0,1)
    norm=np.uint8(np.clip((depth-lo)/max(hi-lo,1e-6),0,1)*255)
    Image.fromarray(cv2.applyColorMap(norm,cv2.COLORMAP_TURBO)[:,:,::-1]).save(directory/'depth.png')
    np.save(directory/'depth_m.npy',depth)
    write_cloud(directory/'points.ply',cloud,np.asarray(image),valid)
    overlay=np.array(image).copy(); rng=np.random.default_rng(22)
    masks={}
    for inst in instances:
        color=rng.integers(70,255,3)
        overlay[inst['mask']]=(.55*overlay[inst['mask']]+.45*color).astype('uint8')
        if inst.get('object_id'): masks[inst['object_id']]=inst['mask']
    overlay=Image.fromarray(overlay);draw=ImageDraw.Draw(overlay)
    for inst in instances:
        draw.text(tuple(inst['bbox'][:2]),inst['label'],fill='red')
    overlay.save(directory/'masks_overlay.png')
    np.savez_compressed(directory/'masks.npz',**masks)
    labels,zbuffer=rasterize(scene,camera,camera_pose)
    render=np.full_like(np.asarray(image),235); per_object=[]
    for i,obj in enumerate(scene.objects):
        rendered=labels==i
        render[rendered]=np.random.default_rng(i+10).integers(60,230,3)
        if obj.id not in masks: continue
        mask=masks[obj.id];union=(rendered|mask).sum();intersection=(rendered&mask).sum()
        common=mask & rendered & np.isfinite(depth) & np.isfinite(zbuffer)
        per_object.append(dict(id=obj.id,label=obj.label,shape_method=obj.shape_method,
            reprojection_iou=float(intersection/union) if union else 0.,
            depth_mae_m=float(np.mean(abs(zbuffer[common]-depth[common]))) if common.any() else None,
            depth_compared_pixels=int(common.sum())))
    combined=Image.new('RGB',(image.width*2,image.height));combined.paste(image,(0,0))
    reprojected=Image.blend(image,Image.fromarray(render),.5); combined.paste(reprojected,(image.width,0))
    combined.save(directory/'reprojection.png');Image.fromarray(render).save(directory/'render.png')
    object_masks=np.logical_or.reduce(list(masks.values())) if masks else np.zeros(depth.shape,bool)
    report['metrics'].update(reprojection_mean_iou=float(np.mean([o['reprojection_iou'] for o in per_object])) if per_object else None,
        depth_mae_m=float(np.mean([o['depth_mae_m'] for o in per_object if o['depth_mae_m'] is not None]))
            if any(o['depth_mae_m'] is not None for o in per_object) else None,
        per_object=per_object,mask_pixel_assignment_fraction=float(object_masks.mean()),
        coverage_nonplane_pixels=None,
        coverage_note='Independent foreground annotation unavailable; mask union/image fraction is not foreground coverage.')
    (directory/'scene.json').write_text(scene.model_dump_json(indent=2))
    (directory/'reconstruction_report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
