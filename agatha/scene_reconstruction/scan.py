"""Metric LiDAR imports: real points, plane slabs, connected physical components."""
from hashlib import sha256
import json,time
from pathlib import Path
import numpy as np
from agatha.schema import Scene,SceneObject,Support,Collision,CollisionPart
from .pipeline import root_dir
from .geometry import fit_box
from .bodies import settle,stability


def read_scan(path):
    import open3d as o3d
    path=Path(path);metadata=dict(unit='meters',scale=1.,up_axis='Y',source='export_assumed_meters')
    if path.suffix.lower()=='.usdz':
        try:from pxr import Usd,UsdGeom
        except ImportError as exc:raise ValueError('USDZ needs the open-source usd-core package') from exc
        stage=Usd.Stage.Open(str(path))
        if stage is None:raise ValueError('Unable to open USDZ stage')
        unit=float(UsdGeom.GetStageMetersPerUnit(stage));up=str(UsdGeom.GetStageUpAxis(stage));points=[]
        cache=UsdGeom.XformCache()
        for prim in stage.Traverse():
            if not prim.IsA(UsdGeom.Mesh):continue
            mesh=UsdGeom.Mesh(prim);transform=cache.GetLocalToWorldTransform(prim)
            points.extend([list(transform.Transform(v)) for v in mesh.GetPointsAttr().Get()])
        points=np.asarray(points,dtype=float)*unit
        if up=='Z':points=points[:,[0,2,1]]*np.array([1,1,-1])
        metadata.update(source='USD_authored_units',meters_per_authored_unit=unit,up_axis=up)
        cloud=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    elif path.suffix.lower()=='.obj':
        mesh=o3d.io.read_triangle_mesh(str(path))
        if not len(mesh.vertices):raise ValueError('OBJ has no vertices')
        cloud=mesh.sample_points_uniformly(number_of_points=min(max(len(mesh.vertices),30000),250000)) if len(mesh.triangles) else o3d.geometry.PointCloud(mesh.vertices)
    elif path.suffix.lower()=='.ply':
        cloud=o3d.io.read_point_cloud(str(path))
    else:raise ValueError('Supported scan formats: .ply, .obj, .usdz')
    if not len(cloud.points):raise ValueError('Scan has no points')
    cloud=cloud.remove_non_finite_points().voxel_down_sample(.004)
    return cloud,metadata


def reconstruct_scan(path):
    import open3d as o3d
    start=time.perf_counter();cloud,metadata=read_scan(path)
    points=np.asarray(cloud.points);original=len(points)
    if len(points)<50:raise ValueError('Scan requires at least 50 finite points')
    planes=[];remaining=cloud
    for _ in range(8):
        if len(remaining.points)<100:break
        equation,indices=remaining.segment_plane(distance_threshold=.008,ransac_n=3,num_iterations=500)
        if len(indices)<max(100,original*.015):break
        part=remaining.select_by_index(indices);p=np.asarray(part.points);normal=np.asarray(equation[:3]);normal/=np.linalg.norm(normal)
        planes.append(dict(normal=normal,points=p,center=np.median(p,0),pixels=len(p)))
        remaining=remaining.select_by_index(indices,invert=True)
    horizontal=[p for p in planes if abs(p['normal'][1])>.85]
    if not horizontal:raise ValueError('No Y-up support plane found; export a gravity-aligned metric scan')
    floor=min(horizontal,key=lambda p:p['center'][1])
    # A higher tabletop is selected when a distinct floor is present.
    candidates=[p for p in horizontal if p['center'][1]>floor['center'][1]+.2]
    table=max(candidates or horizontal,key=lambda p:p['pixels'])
    center=table['center'];floor_height=floor['center'][1] if floor is not table else center[1]-.75
    shift=np.array([center[0],floor_height,center[2]])
    top=table['points']-shift;lo,hi=np.percentile(top,[1,99],axis=0)
    height=float(center[1]-floor_height)
    if not 0<=height<=5:raise ValueError('Scan support height exceeds simulation limits')
    support=Support(size=(max(float(hi[0]-lo[0]),.1),max(float(hi[2]-lo[2]),.1)),height=height,thickness=.03)
    objects=[]
    for index,plane in enumerate(planes):
        if plane is table or plane is floor:continue
        fitted=fit_box(plane['points']-shift);dims=np.maximum(fitted['size'],.02)
        if abs(plane['normal'][1])<.3:
            dims[int(np.argmin(dims))]=.03
            objects.append(SceneObject(id=f'wall_{index}',label='wall',size=tuple(dims),position=tuple(fitted['center']),
                rotation_euler=tuple(fitted['rotation_euler']),is_static=True,shape_method='lidar_plane_slab',confidence=.8))
        elif plane['center'][1]>center[1]+.1:
            objects.append(SceneObject(id=f'shelf_{index}',label='shelf',size=tuple(dims),position=tuple(fitted['center']),
                rotation_euler=tuple(fitted['rotation_euler']),is_static=True,shape_method='lidar_plane_slab',confidence=.6))
    remainder=np.asarray(remaining.points)-shift
    # Plane extraction may remove flat laptop faces; retain above-table horizontal planes
    # with object-sized area as independent components rather than silently discarding them.
    extra=[p['points']-shift for p in planes if p is not table and p is not floor
           and abs(p['normal'][1])>.85 and 0< p['center'][1]-center[1]<.2]
    if extra:remainder=np.concatenate([remainder,*extra])
    object_points=remainder[remainder[:,1]>height+.008]
    pcd=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(object_points))
    labels=np.asarray(pcd.cluster_dbscan(eps=.022,min_points=25,print_progress=False))
    details=[]
    for label in sorted(set(labels)-{-1}):
        pts=object_points[labels==label]
        if len(pts)<30:continue
        fitted=fit_box(pts);size=fitted['size']
        if max(size)>2:continue
        # Connected geometry cannot establish semantic identity; label uncertainty is explicit.
        hull=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts)).compute_convex_hull()[0]
        vertices=np.asarray(hull.vertices)-fitted['center']
        obj=SceneObject(id=f'scan_{label:03}',label='unlabeled scanned object',confidence=.25,
            shape='convex_hull',shape_method='lidar_connected_component_convex_hull',size=tuple(size),
            position=tuple(fitted['center']),mass_kg=.5,supported_by='support',
            collision=Collision(parts=[CollisionPart(type='convex_hull',vertices=[tuple(v) for v in vertices[:4096]])]))
        objects.append(obj);details.append(dict(id=obj.id,points=len(pts),label_status='unknown',collision='convex_hull'))
    if len(objects)>250:raise ValueError('More than 250 scan components; denoise or crop the scan')
    scene,contacts=settle(Scene(objects=objects,support=support))
    measured=stability(scene)
    digest=sha256(Path(path).read_bytes()+b'lidar-v1').hexdigest();directory=root_dir()/digest;directory.mkdir(parents=True,exist_ok=True)
    o3d.io.write_point_cloud(str(directory/'points.ply'),cloud)
    report=dict(input='scan',pipeline='lidar_components_v1',run_id=digest,debug_url=f'/api/reconstructions/{digest}',
        scale=metadata,depth='skipped_true_scale_scan',stages={'planes':len(planes),'components':len(details)},
        objects=details,detected_count=len(details),kept_count=len(details),scale_confidence='exported_metric',
        metrics=dict(contacts_after_settle=contacts,stability=measured,reprojection_mean_iou=None,depth_mae_m=None,
                     coverage_nonplane_pixels=None),duration_s=time.perf_counter()-start,
        warnings=['Scan labels are unknown; touching pieces may remain merged by geometric clustering.',
                  'Scan is treated as gravity-aligned Y-up unless USD specifies Z-up.'],
        estimated=['mass .5 kg per unknown body','friction','table slab thickness','occluded geometry'],acceptance_pass=False)
    (directory/'scene.json').write_text(scene.model_dump_json(indent=2));(directory/'reconstruction_report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    return scene,report
