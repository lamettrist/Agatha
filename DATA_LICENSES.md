# Data licences (external and team data)

Scope: every dataset that was considered for Part C (external physics data for evaluating the learned
world model), whether it ended up being used or not. All licence statements below were read at the
listed URL on **2026-11-15** (and re-read by a second reviewer the same day). "Verified" means the licence text was read at an official source of the
dataset or of its repository; caveats are written out where the source does not state a dedicated
dataset licence.

**No arbitrary scraped internet video is used anywhere in this project.** Only the datasets listed
here (downloaded from their official hosts) and clips recorded by the team itself are allowed. Nothing
from the external datasets is committed to the repository: raw downloads live under `data/external/`
and converted shards under `data/external_v2/` (both git-ignored). No model has been trained or
fine-tuned on any of this data; it is used for approximate evaluation only
(`docs/part-c-external-data.md`).

| Dataset | What we use | Licence | Where the licence is stated | Checked | Verified | How it is used | Attribution / citation |
|---|---|---|---|---|---|---|---|
| **Kubric MOVi-A** (128x128, tfds 1.0.0) | Per-object ground-truth state (position, quaternion, velocities, 3D boxes, mass, friction, restitution, shape; ground friction / restitution taken from the defaults in `challenges/movi/movi_ab_worker.py`) of 1137 episodes from train shards 00000-00059 of 00512 (0.54 GB incl. RGB frames, which are not used). | **Apache-2.0** for the `google-research/kubric` repository (generation worker + KuBasic assets). **Caveat:** there is no separate dataset-specific licence statement: the MOVi README, the main README and the tfds `dataset_info.json` (citation only) say nothing about a data licence; the Kubric paper abstract (arXiv:2203.03570) says the rendered datasets are released "for reuse and modification". MOVi-A uses only Kubric's own primitive assets (no ShapeNet / GSO / HDRI third-party content, unlike MOVi-B..F). | https://github.com/google-research/kubric/blob/main/LICENSE (also read, without licence text for the data: https://github.com/google-research/kubric/blob/main/challenges/movi/README.md , https://storage.googleapis.com/kubric-public/tfds/movi_a/128x128/1.0.0/dataset_info.json , https://arxiv.org/abs/2203.03570) | 2026-11-15 | Partly: repository licence read at source; no dedicated data licence statement was found | **Eval only** (approximate, out-of-distribution toss / bounce check). Not used for fine-tuning. | Greff et al., "Kubric: A Scalable Dataset Generator", CVPR 2022. |
| **CLEVRER** | Validation annotations (`annotation_validation.zip`, 5000 JSON files; the first 1000 are converted) and 4 sample videos (downloaded, not used). | **CC0** - official README, verbatim: "## Lincense / Our dataset is under the CC0." No CC0 version number is given; we found no licence file or statement in the `chuangg/CLEVRER` GitHub repository (code only; not used here). | https://data.csail.mit.edu/clevrer/README.txt (linked as "Readme" from http://clevrer.csail.mit.edu/) | 2026-11-15 | Yes | **Eval only** (approximate planar slide / collision check, true-negative check for toppling). Not used for fine-tuning. | Yi et al., "CLEVRER: CoLlision Events for Video REpresentation and Reasoning", ICLR 2020. |
| **Physion** (NeurIPS 2021, `cogtoolslab/physics-benchmarking-neurips2021`) | 10 PhysionTest-Complete HDF5 trials (8 Dominoes, 2 Support) obtained with an HTTP range request for the first ~1 GB of two official archives (1.8 GB); only per-object state and static properties are read. | **MIT** (Copyright (c) 2019 Cognitive Tools Lab) - the repository-level licence; the README says "This repo contains code and data to reproduce the results in our NeurIPS 2021 paper", which does not explicitly name the S3-hosted stimuli. **Caveat:** the S3-hosted HDF5 / MP4 files have no separate explicit data licence statement. Distractor / occluder meshes inside the trials are TDW model-library assets (TDW's licence was not read at source); we drop those objects and do not redistribute anything. | https://github.com/cogtoolslab/physics-benchmarking-neurips2021/blob/master/LICENSE (download links and the "code and data" wording: https://github.com/cogtoolslab/physics-benchmarking-neurips2021/blob/master/README.md) | 2026-11-15 | Partly: repository licence read at source; no dedicated data licence statement was found | **Eval only** (anecdotal: 10 trials). Not used for fine-tuning. | Bear et al., "Physion: Evaluating Physical Prediction from Vision in Humans and Machines", NeurIPS 2021 Datasets and Benchmarks. |
| **Physion++** | Nothing. | Not established: a search-result snippet of its NeurIPS 2023 supplement says MIT, but the licence was **not read at source**. | - | 2026-11-15 | **No** | **Not used** (licence not verified; not downloaded). | Tung et al., "Physion++", NeurIPS 2023 (not used). |
| Other Kubric MOVi variants (B-F) | Nothing. | Same repository licence, but they embed third-party assets (ShapeNet, Google Scanned Objects, HDRI Haven) with their own terms, which were not reviewed. | https://github.com/google-research/kubric/blob/main/challenges/movi/README.md | 2026-11-15 | **No** (asset licences not reviewed) | **Not used.** | - |
| **Team-recorded phone clips** | None received so far: `data/external/team_clips/` does not exist. | Owned by the team (recorded by team members for this project); no third-party rights involved as long as no other people / copyrighted material are the subject of the clip. | n/a (own work) | 2026-11-15 | n/a | Intended: approximate fell / did-not-fall labels and noisy tracks via `scripts/track_video.py`. **Nothing evaluated yet.** | Team. |
| `tests/images/deskspace.mp4` (repository test asset) | A team-recorded walk-around clip of a desk that is already part of the repository's test images; used read-only to smoke-test `scripts/track_video.py`. | Team-owned test asset (not external data). | n/a | 2026-11-15 | n/a | Tracker smoke test only; no physics labels, no model evaluation. | Team. |

## Notes

- Downloads are treated as untrusted input: they are parsed with numpy (`allow_pickle=False`), `json`
  and `h5py` only; nothing inside them is executed.
- The two "repository-level" licences (Kubric: Apache-2.0, Physion: MIT) are recorded with their caveat
  on purpose: we found no separate licence statement for the rendered / simulated data of either project,
  so whether the repository licence covers the hosted data files is our reading, not an explicit grant. If a
  stricter reading is required, CLEVRER (explicit CC0 for the dataset) is the only one of the three with
  an unambiguous dataset licence.
- Apache-2.0 and MIT require the licence / copyright notice to accompany redistributed copies. We do not
  redistribute any of the data (raw or converted); if converted shards are ever shared, include the
  upstream licence texts and the citations above.
- All three datasets are **simulator output** (PyBullet, Blender/Bullet, TDW/PhysX), not real-world
  video. Results derived from them are labelled approximate everywhere.
