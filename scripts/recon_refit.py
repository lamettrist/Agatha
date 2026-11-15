"""Fast iteration harness for scene reconstruction.

capture  : run the live pipeline ONCE (one GPU, memory capped) and keep every neural
           output on disk (VGGT geometry, masks, SAM 2 track masks, CLIP/TripoSR results).
refit    : re-run ONLY the deterministic stages from that cache on CPU by calling the
           same functions the live pipeline calls, then write scene.json, report.json,
           diagnostic overlays, plan/elevation plots and a per-body table.

  # neural stages, once (about 90 s for the video, 15 s for the photo)
  .venv-reconstruction/bin/python scripts/recon_refit.py capture tests/images/deskspace.mp4 --runs <dir>
  .venv-reconstruction/bin/python scripts/recon_refit.py capture tests/images/acceptance.jpg --runs <dir>
  # every later iteration (CPU only)
  .venv-reconstruction/bin/python scripts/recon_refit.py --run <dir>/<run id> --out <out dir>

Neural calls go through a read-through cache keyed by their exact inputs. A refit that
asks a model something new (e.g. CLIP on a mask your change produced) computes it on CPU
and remembers it; detector/segmenter/depth/TripoSR misses are refused unless
--allow-neural is given, because they mean the cached run no longer matches the code.
"""
from __future__ import annotations
import argparse,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
from threading import Lock
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
if str(ROOT/'scripts') not in sys.path:sys.path.insert(0,str(ROOT/'scripts'))
VIDEO_SUFFIXES={'.mp4','.mov','.m4v','.avi','.webm','.mkv'}


def sha(*parts):
    digest=hashlib.sha1()
    for part in parts:
        digest.update(part if isinstance(part,bytes) else str(part).encode());digest.update(b'|')
    return digest.hexdigest()[:24]


def pick_gpu():
    """The GPU with the most free memory right now; never more than one."""
    rows=subprocess.run(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],
        capture_output=True,text=True,check=True).stdout.strip().splitlines()
    free={int(row.split(',')[0]):int(row.split(',')[1]) for row in rows}
    return max(free,key=free.get),free


class NeuralCache:
    """Read-through cache of model outputs stored as files in one directory."""
    def __init__(self,directory,allow=()):
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
        self.allow=set(allow);self.hits={};self.misses={};self.lock=Lock()

    def path(self,kind,key,suffix):return self.directory/f'{kind}_{key}{suffix}'

    def count(self,kind,hit):
        with self.lock:
            target=self.hits if hit else self.misses;target[kind]=target.get(kind,0)+1

    def require(self,kind,key):
        if kind not in self.allow:
            raise RuntimeError(f'{kind} output {key} is not in the neural cache {self.directory}: the cached run no longer '
                               f'matches this code path. Re-run capture, or pass --allow-neural.')

    def summary(self):return dict(hits=dict(self.hits),computed=dict(self.misses))


def install_cache(cache):
    """Wrap the model registry and TripoSR loader with the read-through cache."""
    import numpy as np,torch
    from agatha.scene_reconstruction import models,pipeline,meshes
    import trimesh

    def image_key(image):return sha(image.size,image.mode,image.tobytes())

    class CachedClipProcessor:
        def __init__(self,owner):self.owner=owner
        def __call__(self,images,text,**kwargs):
            return ClipInputs(images,text,kwargs)

    class ClipInputs(dict):
        def __init__(self,images,text,kwargs):
            super().__init__(images=images,text=text,kwargs=kwargs)
        def to(self,device):return self

    class CachedClipModel:
        device='cpu'
        def __init__(self,owner):self.owner=owner
        def __call__(self,images,text,kwargs):
            prompt=sha(*text);keys=[sha(prompt,image_key(im)) for im in images];rows=[None]*len(keys);todo=[]
            for index,key in enumerate(keys):
                path=cache.path('clip',key,'.npy')
                if path.exists():rows[index]=np.load(path);cache.count('clip',True)
                else:todo.append(index)
            if todo:
                cache.require('clip',keys[todo[0]])
                processor,model=models.Registry.get(self.owner,'clip')
                inputs=processor(images=[images[i] for i in todo],text=text,**kwargs).to(model.device)
                logits=model(**inputs).logits_per_image.float().cpu().numpy()
                for index,row in zip(todo,logits):
                    np.save(cache.path('clip',keys[index],'.npy'),row);rows[index]=row;cache.count('clip',False)
            return SimpleNamespace(logits_per_image=torch.from_numpy(np.stack(rows)))

    class CachedRegistry(models.Registry):
        def get(self,name):
            if name=='clip':return CachedClipProcessor(self),CachedClipModel(self)
            return super().get(name)

        def depth(self,image):
            key=image_key(image);path=cache.path('depth',key,'.npy')
            if path.exists():cache.count('depth',True);return np.load(path)
            cache.require('depth',key);result=super().depth(image);np.save(path,result);cache.count('depth',False)
            return result

        def detect(self,image,*args,**kwargs):
            key=sha(image_key(image),models.PROMPTS,os.getenv('RECON_DETECTION_THRESHOLD','.22'),args,sorted(kwargs.items()))
            path=cache.path('detect',key,'.json')
            if path.exists():cache.count('detect',True);return json.loads(path.read_text())
            cache.require('detect',key);result=super().detect(image,*args,**kwargs)
            path.write_text(json.dumps(result));cache.count('detect',False)
            return json.loads(path.read_text())

        def segment(self,image,detections,*args,**kwargs):
            key=sha(image_key(image),json.dumps(detections,sort_keys=True),os.getenv('RECON_AUTO_GRID','12'),args,sorted(kwargs.items()))
            meta=cache.path('segment',key,'.json');data=cache.path('segment',key,'.npz')
            if meta.exists() and data.exists():
                cache.count('segment',True);items=json.loads(meta.read_text())
                with np.load(data) as stored:
                    height,width=map(int,stored['shape'])
                    masks=np.unpackbits(stored['masks'],axis=1)[:,:height*width].reshape(-1,height,width).astype(bool) if len(items) else []
                return [dict(item,mask=mask) for item,mask in zip(items,masks)]
            cache.require('segment',key);result=super().segment(image,detections,*args,**kwargs)
            packed=np.packbits(np.stack([m['mask'] for m in result]).reshape(len(result),-1),axis=1) if result else np.zeros((0,0),np.uint8)
            np.savez_compressed(data,masks=packed,shape=np.array([image.height,image.width]))
            meta.write_text(json.dumps([{k:v for k,v in m.items() if k!='mask'} for m in result]));cache.count('segment',False)
            return result

    instance=CachedRegistry()
    if os.getenv('RECON_GPU_IDS') is None:instance.devices=[0]
    getter=lambda:instance
    modules=[models,pipeline,meshes]
    try:
        from agatha.scene_reconstruction import video
        modules.append(video)
    except Exception as exc:print(f'[recon_refit] video module unavailable: {exc}',file=sys.stderr)
    for module in modules:module.registry=getter

    real_load=meshes.load_model
    class CachedTripo:
        """Stands in for TSR: the scene code handed to extract_mesh is the cache key."""
        def __init__(self,device):self.device=device
        def __call__(self,crops,device=None):return [image_key(crop) for crop in crops]
        def extract_mesh(self,codes,**kwargs):
            result=[]
            for key in codes:
                key=sha(key,sorted(kwargs.items()));path=cache.path('triposr',key,'.ply')
                if path.exists():
                    cache.count('triposr',True);result.append(trimesh.load_mesh(path,process=False));continue
                cache.require('triposr',key)
                raise RuntimeError('TripoSR recompute must go through capture')
            return result
    class RecordingTripo:
        def __init__(self,model):self.model=model
        def __call__(self,crops,device=None):
            self.keys=[image_key(crop) for crop in crops];return self.model(crops,device=device)
        def extract_mesh(self,code,**kwargs):
            paths=[cache.path('triposr',sha(key,sorted(kwargs.items())),'.ply') for key in self.keys]
            if all(path.exists() for path in paths):
                for _ in paths:cache.count('triposr',True)
                return [trimesh.load_mesh(path,process=False) for path in paths]
            meshes_=self.model.extract_mesh(code,**kwargs)
            for path,mesh in zip(paths,meshes_):mesh.export(path);cache.count('triposr',False)
            # Reload so capture and refit consume byte-identical geometry.
            return [trimesh.load_mesh(path,process=False) for path in paths]
    def load_model(device):
        if 'triposr' in cache.allow:
            model=real_load(device);return RecordingTripo(model)
        meshes.MODEL_LOCKS.setdefault(device,Lock());return CachedTripo(device)
    meshes.load_model=load_model
    return instance


def capture(args):
    source=Path(args.input).resolve();runs=Path(args.runs).resolve();runs.mkdir(parents=True,exist_ok=True)
    if 'CUDA_VISIBLE_DEVICES' not in os.environ or args.gpu is not None:
        gpu,free=(args.gpu,{}) if args.gpu is not None else pick_gpu()
        os.environ['CUDA_VISIBLE_DEVICES']=str(gpu);print(f'[capture] using GPU {gpu} (free MiB: {free})')
    os.environ.setdefault('RECON_GPU_IDS','0');os.environ['RECON_RUN_DIR']=str(runs)
    import torch
    if torch.cuda.is_available():
        total=torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.,args.memory_gb*2**30/total),0)
    staging=runs/f'_capture_cache_{os.getpid()}'
    cache=NeuralCache(staging,allow={'clip','depth','detect','segment','triposr'});install_cache(cache)
    start=time.perf_counter()
    if source.suffix.lower() in VIDEO_SUFFIXES:
        from agatha.scene_reconstruction import video
        import gc
        real_geometry=video.vggt_geometry
        def geometry_then_release(*a,**k):
            # One capped GPU holds every model: drop the 5 GB VGGT weights before SAM 2/DINO load.
            try:return real_geometry(*a,**k)
            finally:
                video.multiview_model.cache_clear();gc.collect()
                if torch.cuda.is_available():torch.cuda.empty_cache()
        video.vggt_geometry=geometry_then_release
        if args.reuse_geometry:
            previous=Path(args.reuse_geometry).resolve()
            def cached_geometry(paths,directory,chunk=None):
                # VGGT-1B with 10-frame chunks does not fit under the 9 GB cap. Reuse its saved
                # outputs; images are re-derived with the pipeline's own preprocessing + crop.
                import numpy as np
                from PIL import Image
                from vggt.utils.load_fn import load_and_preprocess_images
                old_frames=sorted((previous/'frames').glob('frame_*.jpg'))
                if [f.read_bytes() for f in old_frames]!=[Path(f).read_bytes() for f in paths]:
                    raise RuntimeError(f'{previous} was computed from different sampled frames')
                with np.load(previous/'multiview.npz') as data:depths,clouds,confidence=data['depth'],data['points'],data['confidence']
                cameras=json.loads((previous/'cameras.json').read_text())
                tensors=load_and_preprocess_images([str(f) for f in paths],mode='pad')
                images=[Image.fromarray(np.uint8(t.permute(1,2,0).numpy()*255)) for t in tensors]
                side=images[0].width;blank=np.zeros((len(images),side,side),np.float32)
                dummy=[dict(intrinsics=np.eye(3).tolist()) for _ in images]
                with Image.open(paths[0]) as first:
                    images,_,_,_,_=video.crop_padding(images,blank,np.zeros((len(images),side,side,3),np.float32),blank,dummy,first.size)
                if (images[0].height,images[0].width)!=depths.shape[1:]:raise RuntimeError('cached geometry has another resolution')
                shutil.copy2(previous/'multiview.npz',Path(directory)/'multiview.npz')
                shutil.copy2(previous/'cameras.json',Path(directory)/'cameras.json')
                old_report=json.loads((previous/'reconstruction_report.json').read_text())
                alignments=old_report.get('video',{}).get('stages',{}).get('geometry',{}).get('alignments',[])
                print(f'[capture] reused VGGT geometry from {previous}')
                return images,depths,clouds,confidence,cameras,alignments
            video.vggt_geometry=cached_geometry
        scene,report=video.reconstruct_video(source,args.table_width,use_cache=False)
        kind='video'
    else:
        from PIL import Image
        from agatha.scene_reconstruction.pipeline import reconstruct
        scene,report=reconstruct(Image.open(source),args.table_width,use_cache=False)
        kind='photo'
    run=runs/report['run_id'];run.mkdir(parents=True,exist_ok=True)
    target=run/'neural_cache';target.mkdir(exist_ok=True)
    for item in staging.iterdir():shutil.move(str(item),target/item.name)
    staging.rmdir()
    if kind=='photo':shutil.copy2(source,run/('input'+source.suffix.lower()))
    (run/'capture.json').write_text(json.dumps(dict(kind=kind,source=str(source),table_width_m=args.table_width,
        input=None if kind=='video' else 'input'+source.suffix.lower(),pipeline=report.get('pipeline'),
        duration_s=time.perf_counter()-start,cache=cache.summary(),
        peak_gpu_gb=torch.cuda.max_memory_allocated()/2**30 if torch.cuda.is_available() else None,
        geometry_method=report.get('video',{}).get('stages',{}).get('geometry',{}).get('method')),indent=2))
    print(f'[capture] {kind} run: {run}')
    print(f'[capture] pipeline={report.get("pipeline")} bodies={len(scene.objects)} duration={time.perf_counter()-start:.1f}s '
          f'peak_gpu={torch.cuda.max_memory_allocated()/2**30 if torch.cuda.is_available() else 0:.2f} GB cache={cache.summary()}')
    if report.get('pipeline','').startswith('legacy') or 'fallback' in report.get('pipeline',''):
        print('[capture] WARNING: the pipeline fell back; this run is not a usable neural cache',file=sys.stderr)
    for warning in report.get('warnings',[]):print('[capture] warning:',warning)


def refit(args):
    run=Path(args.run).resolve();out=Path(args.out).resolve()
    if not args.allow_neural:os.environ['CUDA_VISIBLE_DEVICES']=''
    os.environ.setdefault('RECON_GPU_IDS','0')
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(name,'8')
    if out.exists() and (out/'recon_refit_output').exists():shutil.rmtree(out)
    out.mkdir(parents=True,exist_ok=True);(out/'recon_refit_output').write_text('directory owned by scripts/recon_refit.py\n')
    lift_root=out/'_lift';os.environ['RECON_RUN_DIR']=str(lift_root)
    if args.no_mesh:os.environ['RECON_MESH']='0'
    start=time.perf_counter()
    cache=NeuralCache(run/'neural_cache',allow={'clip','depth','detect','segment','triposr'} if args.allow_neural else {'clip'})
    install_cache(cache)
    timings={'imports':time.perf_counter()-start}
    kind='video' if (run/'neural_bundle.json').exists() else 'photo'
    info=json.loads((run/'capture.json').read_text()) if (run/'capture.json').exists() else {}
    bundle=None
    if kind=='video':
        from agatha.scene_reconstruction import video
        stage=time.perf_counter();bundle=video.load_neural_bundle(run);timings['load_bundle']=time.perf_counter()-stage
        stage=time.perf_counter()
        fused,tracking=video.refit_tracks(bundle['instances'],bundle['track_masks'],bundle['clouds'],bundle['confidence'])
        video_report=bundle['video_report'];video_report['stages'].pop('tracking',None);video_report['stages'].pop('fusion',None)
        video_report['warnings']=[w for w in video_report.get('warnings',[]) if 'tracks fell back' not in w]
        video.record_tracking(video_report,fused,tracking,len(bundle['instances']),out)
        timings['track_points']=time.perf_counter()-stage;stage=time.perf_counter()
        table_width=args.table_width if args.table_width is not None else bundle['table_width_m']
        scene,report=video.finalize_video(out,bundle['digest'],bundle['images'],bundle['depths'],bundle['cameras'],
            bundle['detections'],bundle['masks'],bundle['instances'],fused,video_report,table_width,start,fusion=args.fusion,
            track_masks=bundle['track_masks'],clouds=bundle['clouds'],confidence=bundle['confidence'])
        timings['finalize_video']=time.perf_counter()-stage
        bundle['fused']=fused;bundle['tracking']=tracking
    else:
        from PIL import Image
        from agatha.scene_reconstruction.pipeline import reconstruct
        source=run/(info.get('input') or 'input.jpg')
        if not source.exists():raise SystemExit(f'{run} is neither a captured video run (neural_bundle.json) nor a captured photo run ({source.name})')
        stage=time.perf_counter()
        table_width=args.table_width if args.table_width is not None else info.get('table_width_m')
        scene,report=reconstruct(Image.open(source),table_width,use_cache=False)
        timings['reconstruct']=time.perf_counter()-stage
        produced=lift_root/report['run_id']
        for asset in produced.iterdir():
            if asset.is_file():shutil.copy2(asset,out/asset.name)
    shutil.rmtree(lift_root,ignore_errors=True)
    refit_seconds=time.perf_counter()-start
    (out/'scene.json').write_text(scene.model_dump_json(indent=2))
    report['refit']=dict(source_run=str(run),kind=kind,timings_s=timings,neural_cache=cache.summary(),
        refit_duration_s=refit_seconds,mesh_stage=os.getenv('RECON_MESH','1')=='1')
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    stage=time.perf_counter()
    import recon_overlay
    summary=recon_overlay.diagnose(run,out,scene=scene,report=report,bundle=bundle,frames=args.frames,iou_stride=args.iou_stride)
    print(summary['table'])
    print()
    for line in summary['headline']:print(line)
    computed=cache.summary()['computed']
    if computed:print(f'NOTE: neural outputs computed on CPU during this refit (now cached): {computed}')
    for item in report.get('stages',{}).get('irregular_meshes',{}).get('objects',[]):
        if item.get('status')=='fallback' and 'neural cache' in str(item.get('reason')):
            print(f"WARNING: {item['id']} needed a TripoSR mesh that is not cached; kept its primitive (re-capture to include it)")
    print(f'refit {refit_seconds:.1f}s ({", ".join(f"{k} {v:.1f}" for k,v in timings.items())}) + diagnostics {time.perf_counter()-stage:.1f}s')
    print(f'outputs: {out}')


def ensure_gl():
    """Open3D needs libGL/libEGL; without them TSDF and every soft-body mesh silently fall back."""
    import ctypes
    try:ctypes.CDLL('libGL.so.1');return
    except OSError:pass
    candidates=[os.getenv('RECON_GL_DIR'),'/home/chintd2/wsp-vla/third_party/nvgl/root/usr/lib/x86_64-linux-gnu']
    for directory in candidates:
        if directory and (Path(directory)/'libGL.so.1').exists() and os.getenv('RECON_REFIT_REEXEC')!='1':
            env=dict(os.environ,RECON_REFIT_REEXEC='1',LD_LIBRARY_PATH=directory+(':'+os.environ['LD_LIBRARY_PATH'] if os.getenv('LD_LIBRARY_PATH') else ''))
            print(f'[recon_refit] libGL.so.1 not on the loader path; re-executing with LD_LIBRARY_PATH={directory}',file=sys.stderr)
            os.execve(sys.executable,[sys.executable,*sys.argv],env)
    print('[recon_refit] WARNING: libGL.so.1 not found (set RECON_GL_DIR): Open3D stages (TSDF, soft-body meshes) will fall back',file=sys.stderr)


def main():
    ensure_gl()
    if len(sys.argv)>1 and sys.argv[1]=='capture':
        parser=argparse.ArgumentParser(prog='recon_refit.py capture',description='Run the neural stages once and cache them.')
        parser.add_argument('input');parser.add_argument('--runs',required=True,help='directory that receives <run id>/')
        parser.add_argument('--table-width',type=float);parser.add_argument('--gpu',type=int,help='physical GPU index (default: most free)')
        parser.add_argument('--memory-gb',type=float,default=9.)
        parser.add_argument('--reuse-geometry',help='earlier video run dir whose multiview.npz/cameras.json (VGGT) are reused')
        capture(parser.parse_args(sys.argv[2:]));return
    parser=argparse.ArgumentParser(description='Re-run the deterministic reconstruction stages from a captured run.')
    parser.add_argument('--run',required=True);parser.add_argument('--out',required=True)
    parser.add_argument('--table-width',type=float);parser.add_argument('--frames',type=int,default=6,help='overlay frames spread over the video')
    parser.add_argument('--iou-stride',type=int,default=3,help='use every Nth frame for the per-frame silhouette IoU')
    parser.add_argument('--no-mesh',action='store_true',help='skip the soft-body mesh stage (RECON_MESH=0)')
    parser.add_argument('--fusion',action='store_true',help='also redo the TSDF export (fused.ply, about 5 s)')
    parser.add_argument('--allow-neural',action='store_true',help='let cache misses run real models (uses the GPU if visible)')
    refit(parser.parse_args())


if __name__=='__main__':main()
