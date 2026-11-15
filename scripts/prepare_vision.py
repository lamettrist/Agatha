"""Download optional vision weights once before demo; safe to skip for offline desk."""
import argparse
import os
from pathlib import Path
import requests
from dotenv import load_dotenv
load_dotenv()
parser=argparse.ArgumentParser(); parser.add_argument('--depth',action='store_true'); args=parser.parse_args()
path=Path(os.getenv('YOLO_MODEL','models/vision/yolov8n.pt')); path.parent.mkdir(parents=True,exist_ok=True)
if not path.exists():
    r=requests.get('https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt',timeout=90)
    r.raise_for_status(); path.write_bytes(r.content)
print('YOLO weights:',path)
if args.depth:
    from transformers import pipeline
    pipeline('depth-estimation',model=os.getenv('DEPTH_MODEL','depth-anything/Depth-Anything-V2-Small-hf'),device=-1)
    print('Depth Anything V2 Small cached')
