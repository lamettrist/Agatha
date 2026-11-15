"""Depth-aware instance merging and conservative splitting; masks remain auditable."""
import cv2
import numpy as np
from .models import PLANE_LABELS


def deduplicate(instances, depth, plane_mask=None):
    kept=[]; dropped=[]
    # Box prompts first. Automatic masks only add uncovered physical pieces.
    ordered=sorted(instances,key=lambda d:(d['mask_source']=='sam2_automatic',-d['confidence']))
    for inst in ordered:
        mask=inst['mask'].astype(bool)
        if mask.sum()<60:
            dropped.append(dict(label=inst['label'],reason='tiny_mask'));continue
        if inst['label'] in PLANE_LABELS:
            continue
        finite=mask & np.isfinite(depth) & (depth>0)
        median=float(np.median(depth[finite])) if finite.any() else None
        if plane_mask is not None and (mask & plane_mask).sum()/mask.sum()>.85:
            dropped.append(dict(label=inst['label'],reason='belongs_to_structure'));continue
        duplicate=False
        for old in kept:
            intersection=(mask & old['mask']).sum()
            iou=intersection/max((mask | old['mask']).sum(),1)
            containment=intersection/min(mask.sum(),old['mask'].sum())
            consistent=median is None or old['median_depth_m'] is None or abs(median-old['median_depth_m'])<.15
            independent_accessory=inst['label']=='gloves' and old['label'] in ('jacket','bag','backpack')
            if consistent and not independent_accessory and (iou>.65 or containment>.85):
                duplicate=True;break
        if duplicate:
            dropped.append(dict(label=inst['label'],reason='duplicate_or_part_mask'));continue
        inst=dict(inst,mask=mask,median_depth_m=median)
        kept.append(inst)
    return kept,dropped


def split_depth_clusters(instances, depth):
    result=[]
    for inst in instances:
        mask=inst['mask']; finite=mask & np.isfinite(depth) & (depth>0)
        values=depth[finite]
        if values.size<150:
            result.append(inst);continue
        # Only split when there is a true empty depth interval, not a sloped surface.
        hist,bins=np.histogram(values,bins=40)
        interior=(hist==0) & (bins[:-1]>np.percentile(values,20)) & (bins[1:]<np.percentile(values,80))
        split_indices=np.flatnonzero(interior)
        if not len(split_indices) or np.ptp(values)<.15:
            result.append(inst);continue
        cut=bins[split_indices[len(split_indices)//2]]
        sides=[finite & (depth<cut),finite & (depth>=cut)]
        components=[]
        for side in sides:
            count,labels,stats,_=cv2.connectedComponentsWithStats(side.astype('uint8'),8)
            for i in range(1,count):
                if stats[i,cv2.CC_STAT_AREA]>.22*mask.sum(): components.append(labels==i)
        if len(components)!=2:
            result.append(inst);continue
        for component in components:
            yy,xx=np.where(component)
            result.append(dict(inst,mask=component,bbox=[float(xx.min()),float(yy.min()),float(xx.max()+1),float(yy.max()+1)],
                               split_reason='disconnected_depth_clusters'))
    return result
