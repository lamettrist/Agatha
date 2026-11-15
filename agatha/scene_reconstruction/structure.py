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


def table_surfaces(regions,cloud,valid,support,min_area=0.):
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
        if min(size[0],size[2])<.1 or size[0]*size[2]<min_area:continue
        center=fitted['center'];center[1]=height-.015
        result.append(dict(size=tuple(size),position=tuple(center),rotation_euler=tuple(fitted['rotation_euler']),
                           observed_points=len(part)))
    return result


def fuse_views(views,*,min_points=12):
    """Multi-view consensus for one tracked object, in metric world coordinates (Y up).

    views: [(frame, Nx3 points)] in propagation order. SAM 2 tracks drift onto
    neighbouring furniture and VGGT chunks ghost by ~0.1 m, so a union of all
    observations inflates bodies. Keep only views that agree with the consensus
    of the first observations, then only voxels seen from several kept views.
    Returns (points, stats).
    """
    usable=[]
    for frame,points in views:
        points=np.asarray(points,dtype=float).reshape(-1,3);points=points[np.isfinite(points).all(1)]
        if len(points)<min_points:continue
        lo,hi=np.percentile(points,[2,98],axis=0)
        usable.append((int(frame),points,np.median(points,0),hi-lo))
    stats=dict(views_observed=len(usable),views_kept=0,kept_frames=[],points=0,voted=False)
    if not usable:return np.empty((0,3)),stats
    head=usable[:5]
    center=np.median([u[2] for u in head],0);extent=np.median([u[3] for u in head],0)
    footprint=float(max(extent[0],extent[2]))
    kept=[]
    for frame,points,median,size in usable:
        horizontal=float(np.hypot(median[0]-center[0],median[2]-center[2]))
        if horizontal>max(.12,.35*footprint):continue
        if abs(median[1]-center[1])>max(.12,.35*extent[1]):continue
        if max(size[0],size[2])>1.6*footprint+.05:continue
        kept.append((frame,points))
    if not kept:return np.empty((0,3)),stats
    points=np.concatenate([p for _,p in kept]);voted=False
    if len(kept)>=4:
        voxel=.015 if float(max(extent))<.35 else .04
        keys=np.floor(points/voxel).astype(np.int64)
        view=np.concatenate([np.full(len(p),i) for i,(_,p) in enumerate(kept)])
        _,inverse=np.unique(keys,axis=0,return_inverse=True);inverse=inverse.reshape(-1)
        pairs=np.unique(np.column_stack([inverse,view]),axis=0)
        support=np.bincount(pairs[:,0],minlength=inverse.max()+1)
        mask=support[inverse]>=max(2,int(np.ceil(.25*len(kept))))
        if mask.sum()>=min_points:points=points[mask];voted=True
    stats.update(views_kept=len(kept),kept_frames=[f for f,_ in kept],points=int(len(points)),voted=voted)
    return points,stats


def trim_rectangle(rect,component,counts,origin,cell,*,level=.5):
    """Shrink a union-of-views rectangle to the edges most views agree on.

    Poses drift by centimetres between views, so the union of every view's table
    points overhangs the real top on all sides. Cells on that smeared rim are seen
    from few views: keep cells observed at least `level` times as often as the
    typical interior cell, then put each edge where half of the cross-section is
    still occupied. Orientation is unchanged.
    """
    rows,cols=np.nonzero(component)
    if len(rows)<20:return rect
    reference=float(np.median(counts[rows,cols]))
    core=counts[rows,cols]>=max(1.,level*reference)
    if core.sum()<20:return rect
    xz=origin+(np.column_stack([cols[core],rows[core]])+.5)*cell
    axis=rect['axis'];normal=np.array([-axis[1],axis[0]])
    local=np.column_stack([(xz-rect['center'])@axis,(xz-rect['center'])@normal])
    bounds=[]
    for a in (0,1):
        bins=np.round(local[:,a]/cell).astype(int);low=bins.min();profile=np.bincount(bins-low)
        full=float(np.percentile(profile[profile>0],75))
        inside=np.flatnonzero(profile>=.5*full)
        bounds.append(((inside[0]+low-.5)*cell,(inside[-1]+low+.5)*cell))
    (u0,u1),(v0,v1)=bounds
    if u1-u0<=0 or v1-v0<=0:return rect
    center=rect['center']+axis*(u0+u1)/2+normal*(v0+v1)/2
    return dict(center=center,axis=axis,size=np.array([u1-u0,v1-v0]),untrimmed_size=np.asarray(rect['size']).tolist())


def table_rectangle(points,frames,seed_xz,*,min_frames=5):
    """Oriented rectangle of a table top from multi-frame points in a plane-aligned frame.

    points: Nx3 with the table plane at y=0 (any consistent unit), frames: N view ids,
    seed_xz: Mx2 coplanar points of the reference-frame table mask. A camera-aligned
    bounding box of one view is neither oriented with the table nor complete.
    Returns dict(center, axis (unit, long side), size (long, short), source) or None.
    """
    import cv2
    seed_xz=np.asarray(seed_xz,dtype=float)
    if len(seed_xz)<30:return None
    lo,hi=np.percentile(seed_xz,[1,99],axis=0);span=float(max(hi-lo))
    if span<=0:return None
    cell=span/50
    def rectangle(cells,origin):
        (cx,cy),(w,h),angle=cv2.minAreaRect(np.argwhere(cells)[:,::-1].astype(np.float32))
        angle=np.deg2rad(angle)
        if h>w:w,h=h,w;angle+=np.pi/2
        return dict(center=origin+(np.array([cx,cy])+.5)*cell,axis=np.array([np.cos(angle),np.sin(angle)]),
                    size=np.array([w+1,h+1])*cell)
    def grid(xz,origin,shape):
        index=np.floor((xz-origin)/cell).astype(int)
        ok=(index>=0).all(1)&(index<shape).all(1)
        return index[ok],ok
    points=np.asarray(points,dtype=float);frames=np.asarray(frames)
    origin=lo-span;shape=np.array([int(np.ceil((hi[0]-lo[0]+2*span)/cell)),int(np.ceil((hi[1]-lo[1]+2*span)/cell))])
    seed_index,_=grid(seed_xz,origin,shape)
    seed_cells=np.zeros(shape[::-1],np.uint8);seed_cells[seed_index[:,1],seed_index[:,0]]=1
    seed_cells=cv2.morphologyEx(seed_cells,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    if seed_cells.sum()<20:return None
    seed_rect=dict(rectangle(seed_cells>0,origin),source='reference_frame_min_area_rectangle')
    if len(points):
        near=np.abs(points[:,1])<cell
        index,ok=grid(points[near][:,[0,2]],origin,shape)
        if len(index):
            triples=np.unique(np.column_stack([index,frames[near][ok]]),axis=0)
            counts=np.zeros(shape[::-1],np.int32);np.add.at(counts,(triples[:,1],triples[:,0]),1)
            needed=min(min_frames,max(2,len(np.unique(frames))//4))
            occupied=cv2.morphologyEx((counts>=needed).astype(np.uint8),cv2.MORPH_OPEN,np.ones((5,5),np.uint8))
            count,labels=cv2.connectedComponents(occupied,connectivity=8)
            if count>1:
                overlap=np.bincount(labels[seed_cells>0],minlength=count);overlap[0]=0
                if overlap.max()>0:
                    component=(labels==int(overlap.argmax()))|(seed_cells>0)
                    multi=trim_rectangle(rectangle(component,origin),component,counts,origin,cell)
                    # Accept a completed footprint only when it is a plausible extension of the seed.
                    if (multi['size']>=seed_rect['size']*.7).all() and multi['size'][0]<=seed_rect['size'][0]*2 and multi['size'][1]<=seed_rect['size'][1]*1.6:
                        return dict(multi,source='multi_frame_min_area_rectangle',seed_size=seed_rect['size'].tolist())
    return seed_rect
