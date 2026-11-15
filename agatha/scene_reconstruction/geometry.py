"""Deterministic metric geometry. Camera is OpenCV XYZ; scene is right-handed Y-up."""
from __future__ import annotations
import math
import cv2
import numpy as np

REFERENCES = {'bottle': (.21, 1), 'can': (.134, 1), 'phone': (.15, -1),
              'cell phone': (.15, -1), 'laptop': (.31, 0), 'paper': (.2794, -1)}


def solve_scale(references: list[dict], manual: float | None = None) -> dict:
    """Median log-ratio rejects multiplicative outliers; no class-size clamping."""
    if manual is not None:
        if not np.isfinite(manual) or manual <= 0:
            raise ValueError('Scale override must be positive and finite')
        return dict(factor=float(manual), references=references, disagreement=0., source='manual', rejected=[])
    valid = [r for r in references if np.isfinite(r['observed_m']) and r['observed_m'] > 0
             and np.isfinite(r['reference_m']) and r['reference_m'] > 0]
    if not valid:
        return dict(factor=1., references=[], disagreement=None, source='metric_model_unverified', rejected=[])
    logs = np.log([r['reference_m']/r['observed_m'] for r in valid])
    med = np.median(logs); mad = np.median(np.abs(logs-med))
    keep = np.abs(logs-med) <= max(3*1.4826*mad, math.log(1.25))
    factor = float(np.exp(np.median(logs[keep])))
    return dict(factor=factor, references=[r for r,k in zip(valid,keep) if k],
                rejected=[r for r,k in zip(valid,keep) if not k],
                disagreement=float(np.max(np.abs(np.exp(logs[keep])/factor-1))),
                source='assumed_reference_median', all_ratio_disagreement=float(np.max(np.abs(np.exp(logs)/factor-1))))


def intrinsics(image) -> dict:
    w,h = image.size
    exif = image.getexif()
    # EXIF 41989 = focal length in 35mm film. Avoid guessing sensor size.
    focal35 = exif.get(41989)
    if focal35 and float(focal35) > 0:
        f = float(focal35)*math.hypot(w,h)/math.hypot(36,24)
        source = 'EXIF_35mm_equivalent'
    else:
        f = w/(2*math.tan(math.radians(60)/2)); source = 'assumed_60deg_horizontal_FOV'
    return dict(fx=f, fy=f, cx=(w-1)/2, cy=(h-1)/2, width=w, height=h, source=source)


def backproject(depth: np.ndarray, camera: dict) -> np.ndarray:
    yy,xx = np.indices(depth.shape)
    return np.stack(((xx-camera['cx'])/camera['fx']*depth,
                     (yy-camera['cy'])/camera['fy']*depth, depth), -1)


def clean_depth(depth: np.ndarray, image: np.ndarray, edge_fraction=.035):
    """Reject depth discontinuities and low-signal pixels; fill only for visualization.

    No model confidence is claimed: darkness, clipping and local variation are heuristics.
    The valid mask always excludes filled pixels from geometry fitting.
    """
    valid = np.isfinite(depth) & (depth > .05) & (depth < 50)
    safe = np.where(valid, depth, 0).astype(np.float32)
    dx = cv2.Sobel(safe, cv2.CV_32F, 1, 0, ksize=3)/8
    dy = cv2.Sobel(safe, cv2.CV_32F, 0, 1, ksize=3)/8
    edges = np.hypot(dx,dy) > edge_fraction*np.maximum(safe,.1)
    edges = cv2.dilate(edges.astype('uint8'),np.ones((3,3),np.uint8)) > 0
    dark = image.max(axis=-1) < 22
    clipped = image.min(axis=-1) > 250
    low = edges | dark | clipped | ~valid
    valid &= ~low
    filled = safe.copy()
    if valid.any() and low.any():
        median = float(np.median(safe[valid]))
        normalized = safe/max(median,1e-3)
        filled = cv2.inpaint(normalized, low.astype('uint8'),3,cv2.INPAINT_NS)*median
    return filled, valid, dict(edge_pixels=int(edges.sum()), dark_pixels=int(dark.sum()),
                              clipped_pixels=int(clipped.sum()), valid_fraction=float(valid.mean()),
                              confidence_method='image_and_depth_gradient_heuristic')


def fit_plane(points: np.ndarray, threshold=.012, iterations=180, normal_hint=None) -> dict:
    pts = np.asarray(points, dtype=float).reshape(-1,3)
    pts = pts[np.isfinite(pts).all(1)]
    if len(pts) < 30: raise ValueError('Fewer than 30 valid plane points')
    rng=np.random.default_rng(17)
    if len(pts)>6000: pts=pts[rng.choice(len(pts),6000,False)]
    best=np.zeros(len(pts),bool)
    hint=None if normal_hint is None else np.asarray(normal_hint)/np.linalg.norm(normal_hint)
    for _ in range(iterations):
        a,b,c=pts[rng.choice(len(pts),3,False)]
        n=np.cross(b-a,c-a); length=np.linalg.norm(n)
        if length<1e-8: continue
        n/=length
        if hint is not None and abs(n@hint)<.55: continue
        mask=np.abs((pts-a)@n)<threshold
        if mask.sum()>best.sum(): best=mask
    if best.sum()<30: raise ValueError('No plane consensus')
    center=pts[best].mean(0)
    _,_,vh=np.linalg.svd(pts[best]-center,full_matrices=False)
    n=vh[-1]
    if (n@hint if hint is not None else -n[1])<0: n=-n
    x=np.array([1.,0,0]); x-=x.dot(n)*n
    if np.linalg.norm(x)<.1:
        x=np.array([0.,0,1]); x-=x.dot(n)*n
    x/=np.linalg.norm(x); z=np.cross(x,n)
    basis=np.array([x,n,z])
    return dict(normal=n, center=center, basis=basis, inlier_ratio=float(best.mean()),
                rms_m=float(np.sqrt(np.mean(((pts[best]-center)@n)**2))))


def region_planes(cloud, valid, max_planes=5):
    """RANSAC followed by connected image-region growing; holes don't join planes."""
    remaining=valid.copy(); result=[]
    for _ in range(max_planes):
        try: plane=fit_plane(cloud[remaining])
        except ValueError: break
        near=(np.abs((cloud-plane['center'])@plane['normal'])<.018) & remaining
        count,components,stats,_=cv2.connectedComponentsWithStats(near.astype('uint8'),8)
        if count<2: break
        component=1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]))
        region=components==component
        if region.sum()<100: break
        remaining &= ~near
        plane['pixels']=int(region.sum()); plane['mask']=region
        result.append(plane)
    return result


def fit_box(points: np.ndarray) -> dict:
    points=np.asarray(points); points=points[np.isfinite(points).all(1)]
    if len(points)<12: raise ValueError('Insufficient object points')
    center=np.median(points,0); xz=points[:,[0,2]]-center[[0,2]]
    _,vectors=np.linalg.eigh(np.cov(xz.T)); axis=vectors[:,-1]
    if axis[0]<0: axis=-axis
    yaw=math.atan2(-axis[1],axis[0])
    c,s=math.cos(yaw),math.sin(yaw)
    rotation=np.array([[c,0,s],[0,1,0],[-s,0,c]])
    local=(points-center)@rotation
    lo,hi=np.percentile(local,[1,99],axis=0)
    size=np.maximum(hi-lo,.004)
    return dict(size=size,center=center+rotation@((lo+hi)/2),rotation_euler=np.array([0,yaw,0]),
                method='gravity_aligned_obb', residual_m=float(np.median(np.min(np.minimum(abs(local-lo),abs(local-hi)),axis=1))))


def fit_cylinder(points: np.ndarray) -> dict:
    """RANSAC circle on the XZ projection completes an upright cylinder by symmetry."""
    pts=np.asarray(points); pts=pts[np.isfinite(pts).all(1)]
    if len(pts)<20: raise ValueError('Insufficient cylinder points')
    rng=np.random.default_rng(19)
    if len(pts)>2500: pts=pts[rng.choice(len(pts),2500,False)]
    xz=pts[:,[0,2]]; best=None
    for _ in range(160):
        sample=xz[rng.choice(len(xz),3,False)]
        a=2*(sample[1:]-sample[0]); b=(sample[1:]**2).sum(1)-(sample[0]**2).sum()
        try: center=np.linalg.solve(a,b)
        except np.linalg.LinAlgError: continue
        radius=np.linalg.norm(sample[0]-center)
        if not .012<radius<.18: continue
        residual=abs(np.linalg.norm(xz-center,axis=1)-radius)
        inliers=residual<.006
        if best is None or inliers.sum()>best[0].sum(): best=(inliers,center,radius)
    if best is None or best[0].mean()<.45: raise ValueError('Cylinder fit has poor consensus')
    inliers,center,radius=best
    a=np.c_[2*xz[inliers],np.ones(inliers.sum())]; b=(xz[inliers]**2).sum(1)
    cx,cz,offset=np.linalg.lstsq(a,b,rcond=None)[0]
    radius=float(np.sqrt(max(offset+cx*cx+cz*cz,0)))
    lo,hi=np.percentile(pts[:,1],[1,99]); height=float(hi-lo)
    if not .01<radius<.18 or not .025<height<.65: raise ValueError('Cylinder fit outside physical range')
    return dict(size=np.array([2*radius,height,2*radius]),center=np.array([cx,(lo+hi)/2,cz]),
                radius=radius,height=height,rotation_euler=np.zeros(3),method='ransac_cylinder',
                residual_m=float(np.median(abs(np.linalg.norm(xz-[cx,cz],axis=1)-radius))))
