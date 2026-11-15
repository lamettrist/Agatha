"""Fetch audited photo models once; all request-time model loads are local-only."""
import argparse
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import snapshot_download
from agatha.scene_reconstruction.models import MODEL_IDS

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--directory',default='models/vision/reconstruction')
    parser.add_argument('--warm',action='store_true'); args=parser.parse_args()
    def download(item):
        name,repo=item
        path=snapshot_download(repo,local_dir=Path(args.directory)/name,
            allow_patterns=['*.json','*.txt','*.safetensors','pytorch_model.bin','*.model','vocab*','merges*','LICENSE*','README*'],max_workers=4)
        return name,path
    print(json.dumps(dict(download(item) for item in MODEL_IDS.items()),indent=2))
    if args.warm:
        from agatha.scene_reconstruction.models import registry
        print(registry().preload())
