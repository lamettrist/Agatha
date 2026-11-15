"""Run real reconstruction and print its measured JSON report."""
import argparse,json
from pathlib import Path
from PIL import Image
from agatha.scene_reconstruction.pipeline import reconstruct
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('input',type=Path);p.add_argument('--table-width',type=float)
    p.add_argument('--no-cache',action='store_true');p.add_argument('--warm',action='store_true');args=p.parse_args()
    if args.warm:
        from agatha.scene_reconstruction.models import registry
        print(json.dumps(registry().preload()),flush=True)
    scene,report=reconstruct(Image.open(args.input),args.table_width,use_cache=not args.no_cache)
    print(json.dumps(report,indent=2,allow_nan=False),flush=True)
