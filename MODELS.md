# Reconstruction models and licenses

Weights are downloaded by preparation commands, never fetched during an upload.
No inference service or API key is required. Licenses must be retained when redistributing.

| Model / weights | Purpose | License and authoritative source |
| --- | --- | --- |
| Depth Anything V2 Metric Indoor Small (Hypersim), `depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf` | Default metric photo depth | Small models Apache-2.0 per [upstream licensing](https://github.com/DepthAnything/Depth-Anything-V2#license); [metric checkpoint](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf). Base/Large/Giant are CC-BY-NC-4.0 and are not selected. |
| Grounding DINO Tiny, `IDEA-Research/grounding-dino-tiny` | Open-vocabulary object detection | [Apache-2.0 model card](https://huggingface.co/IDEA-Research/grounding-dino-tiny) |
| SAM 2.1 Hiera Small, `facebook/sam2.1-hiera-small` | Box-prompt + automatic masks, video propagation | [Apache-2.0 model card](https://huggingface.co/facebook/sam2.1-hiera-small), [official code](https://github.com/facebookresearch/sam2) |
| CLIP ViT-B/32, `openai/clip-vit-base-patch32` | Label second opinion on masked object crops | [MIT license](https://github.com/openai/CLIP/blob/main/LICENSE), [checkpoint](https://huggingface.co/openai/clip-vit-base-patch32) |
| VGGT-1B, `facebook/VGGT-1B` | Multi-view cameras and dense geometry | [VGGT license, July 29 2025](https://github.com/facebookresearch/vggt/blob/main/LICENSE.txt), [weights](https://huggingface.co/facebook/VGGT-1B). Custom open-weight license permits use/modification subject to attribution, redistribution terms and acceptable-use terms; **not described as Apache/MIT**. |
| TripoSR, `stabilityai/TripoSR` | Optional crop-to-mesh worker | [MIT model card](https://huggingface.co/stabilityai/TripoSR), [MIT code](https://github.com/VAST-AI-Research/TripoSR). Only used when installed and prepared; missing weights/code are logged as OBB fallback. |
| Existing YOLOv8n | Existing legacy fallback only | [Ultralytics AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE); not used by the new default detector. |

three.js including GLTFLoader and BufferGeometryUtils is MIT licensed (local THREE-LICENSE.txt).
Open3D is MIT; trimesh is MIT. Stable Fast 3D is not selected because its community
license differs from TripoSR's MIT license. No model accuracy or latency from its model
card is substituted for fixture measurements.

All properties use per-class estimates unless overridden by the existing Exa/OpenAI
agent or the user. Standard-object size references are assumptions, not product identity.
Low-confidence darkness/gloss heuristics are not a model's calibrated uncertainty.

The checked-out existing environment has PyTorch 2.15/cu132 and cannot initialize
CUDA on this host's driver. `.venv-reconstruction` uses PyTorch 2.7.1/cu128 with
Transformers 4.57.6. RTX 6000 is Turing; bf16 is **not** enabled there. Photo stages
use fp32; multi-view model may use fp16 under autocast when supported.
