"""Long-lived open-weight models; downloads are an explicit preparation step."""
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from threading import RLock
import os
import numpy as np
import torch
from PIL import Image

MODEL_IDS = {
    'depth': 'depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf',
    'detector': 'IDEA-Research/grounding-dino-tiny',
    'sam': 'facebook/sam2.1-hiera-small',
    'clip': 'openai/clip-vit-base-patch32',
}
CLASSES = ['bottle','tumbler','cup','can','laptop','phone','backpack','bag','jacket','gloves',
           'mouse','keyboard','monitor','book','box','plant','chair','paper','table','wall','floor','unknown object']
PROMPTS = ['black insulated bottle','plastic water bottle','white tumbler','cup','can',
           'open laptop','closed laptop','phone','gray backpack','black backpack',
           'jacket','gloves','mouse','keyboard','monitor','book','box','plant','chair','table','wall','floor','paper']
PLANE_LABELS={'table','wall','floor'}


def canonical(label):
    label=label.lower().strip()
    for name in ['backpack','tumbler','bottle','laptop','gloves','jacket','phone','mouse','keyboard',
                 'monitor','table','wall','floor','chair','paper','plant','book','box','bag','can','cup']:
        if name in label: return name
    return 'unknown object'


class Registry:
    def __init__(self):
        torch.set_num_threads(4)
        self.models={}; self.failures={}; self.lock=RLock()
        self.locks={k:RLock() for k in MODEL_IDS}
        self.devices=[int(v) for v in os.getenv('RECON_GPU_IDS','0,1,2,3,4,5,6').split(',')]

    def device(self, index):
        return f'cuda:{self.devices[index % len(self.devices)]}' if torch.cuda.is_available() else 'cpu'

    def get(self, name):
        if os.getenv('VISION_OFFLINE')=='1': raise RuntimeError('VISION_OFFLINE=1')
        with self.lock:
            if name in self.failures: raise RuntimeError(self.failures[name])
            if name not in self.models:
                try:
                    import transformers as t
                    path=Path(os.getenv('RECON_MODEL_DIR','models/vision/reconstruction'))/name
                    source=str(path) if path.exists() else MODEL_IDS[name]
                    kwargs=dict(local_files_only=True)
                    device=self.device(list(MODEL_IDS).index(name))
                    if name=='depth':
                        processor=t.AutoImageProcessor.from_pretrained(source,**kwargs)
                        model=t.AutoModelForDepthEstimation.from_pretrained(source,**kwargs).to(device).eval()
                    elif name=='detector':
                        processor=t.AutoProcessor.from_pretrained(source,**kwargs)
                        model=t.AutoModelForZeroShotObjectDetection.from_pretrained(source,**kwargs).to(device).eval()
                    elif name=='sam':
                        processor=t.Sam2Processor.from_pretrained(source,**kwargs)
                        model=t.Sam2Model.from_pretrained(source,**kwargs).to(device).eval()
                    else:
                        processor=t.CLIPProcessor.from_pretrained(source,**kwargs)
                        model=t.CLIPModel.from_pretrained(source,**kwargs).to(device).eval()
                    self.models[name]=(processor,model)
                except Exception as exc:
                    self.failures[name]=f'{name}: {type(exc).__name__}: {str(exc)[:400]}'
                    raise RuntimeError(self.failures[name]) from exc
            return self.models[name]

    @torch.inference_mode()
    def depth(self,image):
        with self.locks['depth']:
            processor,model=self.get('depth')
            inputs=processor(images=image,return_tensors='pt').to(model.device)
            output=model(**inputs)
            return torch.nn.functional.interpolate(output.predicted_depth[:,None],size=image.size[::-1],
                mode='bicubic',align_corners=False)[0,0].cpu().numpy().astype(np.float32)

    @torch.inference_mode()
    def detect(self,image):
        with self.locks['detector']:
            processor,model=self.get('detector')
            inputs=processor(images=image,text='. '.join(PROMPTS)+'.',return_tensors='pt').to(model.device)
            output=model(**inputs)
            result=processor.post_process_grounded_object_detection(output,inputs.input_ids,
                threshold=float(os.getenv('RECON_DETECTION_THRESHOLD','.22')),text_threshold=.20,
                target_sizes=[image.size[::-1]])[0]
            return [dict(label=canonical(label),detector_label=label,bbox=box.tolist(),confidence=float(score))
                    for box,score,label in zip(result['boxes'].cpu(),result['scores'].cpu(),result['text_labels'])]

    @torch.inference_mode()
    def segment(self,image,detections):
        """One image embedding for all box prompts plus a sparse automatic point grid."""
        with self.locks['sam']:
            processor,model=self.get('sam')
            base=processor(images=image,return_tensors='pt').to(model.device)
            embeddings=model.get_image_embeddings(base.pixel_values)
            masks=[]
            if detections:
                inputs=processor(images=image,input_boxes=[[d['bbox'] for d in detections]],return_tensors='pt').to(model.device)
                inputs.pop('pixel_values')
                output=model(**inputs,image_embeddings=embeddings,multimask_output=False)
                decoded=processor.post_process_masks(output.pred_masks.cpu(),inputs['original_sizes'].cpu())[0]
                for det,mask,score in zip(detections,decoded,output.iou_scores[0].cpu()):
                    masks.append(dict(**det,mask=mask[0].numpy().astype(bool),mask_score=float(score[0]),mask_source='sam2_box'))
            w,h=image.size; side=int(os.getenv('RECON_AUTO_GRID','12'))
            points=[[[float(x),float(y)]] for y in np.linspace(h*.04,h*.96,side) for x in np.linspace(w*.04,w*.96,side)]
            for start in range(0,len(points),32):
                batch=points[start:start+32]
                inputs=processor(images=image,input_points=[batch],input_labels=[[[1]]*len(batch)],return_tensors='pt').to(model.device)
                inputs.pop('pixel_values')
                output=model(**inputs,image_embeddings=embeddings,multimask_output=True)
                decoded=processor.post_process_masks(output.pred_masks.cpu(),inputs['original_sizes'].cpu())[0]
                for candidates,scores in zip(decoded,output.iou_scores[0].cpu()):
                    idx=int(scores.argmax()); mask=candidates[idx].numpy().astype(bool)
                    if float(scores[idx])<.88 or mask.sum()<80 or mask.mean()>.45: continue
                    yy,xx=np.where(mask)
                    masks.append(dict(label='unknown object',bbox=[float(xx.min()),float(yy.min()),float(xx.max()+1),float(yy.max()+1)],
                        confidence=float(scores[idx]),mask=mask,mask_score=float(scores[idx]),mask_source='sam2_automatic'))
            return masks

    @torch.inference_mode()
    def classify(self,image,instances):
        if not instances: return
        with self.locks['clip']:
            processor,model=self.get('clip')
            crops=[]
            rgb=np.asarray(image)
            for inst in instances:
                x1,y1,x2,y2=[int(v) for v in inst['bbox']]
                crop=rgb[y1:y2,x1:x2].copy(); mask=inst['mask'][y1:y2,x1:x2]
                crop[~mask]=240
                crops.append(Image.fromarray(crop))
            inputs=processor(images=crops,text=['a photo of a '+c for c in CLASSES],padding=True,return_tensors='pt').to(model.device)
            scores=model(**inputs).logits_per_image.softmax(-1).cpu().numpy()
            for inst,probs in zip(instances,scores):
                idx=int(probs.argmax()); label=CLASSES[idx]
                inst['classifier_label']=label; inst['classifier_confidence']=float(probs[idx])
                inst['label_disagreement']=inst['label']!=label
                if inst['label']=='unknown object':
                    inst['label']=label if probs[idx]>.25 else 'unknown object'
                    inst['confidence']=min(inst['confidence'],float(probs[idx]))

    def preload(self):
        status={}
        for name in MODEL_IDS:
            try: self.get(name); status[name]='loaded'
            except Exception as exc: status[name]=str(exc)
        return status


@lru_cache(1)
def registry(): return Registry()
