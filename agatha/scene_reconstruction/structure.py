"""Connected measured surfaces; incomplete visibility is recorded explicitly."""
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from .geometry import fit_plane,fit_box


def cloud_components(points,radius=.12):
    points=np.asarray(points)
    if not len(points):return np.empty(0,dtype=int)
    keys,inverse=np.unique(np.floor(points/(radius/4)).astype(np.int64),axis=0,return_inverse=True)
    counts=np.bincount(inverse)
    centers=np.column_stack([np.bincount(inverse,weights=points[:,axis])/counts for axis in range(3)])
    pairs=cKDTree(centers).query_pairs(radius,output_type='ndarray')
    graph=coo_matrix((np.ones(len(pairs)),(pairs[:,0],pairs[:,1])),shape=(len(keys),len(keys))).tocsr()
    _,labels=connected_components(graph,directed=False)
    return labels[inverse]


def table_surfaces(regions,cloud,valid,support):
    patches=[]
    for region in regions:
        points=cloud[region['mask']&valid]
        for _ in range(3):
            if len(points)<50:break
            try:plane=fit_plane(points,threshold=.015,iterations=100)
            except ValueError:break
            near=np.abs((points-plane['center'])@plane['normal'])<.02
            if abs(plane['normal'][1])>.95:patches.append(points[near])
            points=points[~near]
    if not patches:return []
    points=np.concatenate(patches);labels=cloud_components(points)
    result=[]
    for label in np.unique(labels):
        part=points[labels==label]
        if len(part)<50:continue
        height=float(np.median(part[:,1]))
        if not .15<height<3:continue
        on_primary=(abs(part[:,0])<=support.size[0]/2+.04)&(abs(part[:,2])<=support.size[1]/2+.04)
        if abs(height-support.height)<.05 and on_primary.mean()>.5:continue
        fitted=fit_box(part);size=fitted['size'];size[1]=.03
        if min(size[0],size[2])<.1:continue
        center=fitted['center'];center[1]=height-.015
        result.append(dict(size=tuple(size),position=tuple(center),rotation_euler=tuple(fitted['rotation_euler']),
                           observed_points=len(part)))
    return result
