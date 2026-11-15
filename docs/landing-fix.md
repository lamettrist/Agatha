# Landing fix: a box released at an angle now comes to rest on a face

## The bug

"When we rotate something at an angle and let it fall, it doesn't land properly on the sides or on a flat face."

Reproduced with `scripts/eval_landing.py` (new, seeded): 2,432 "rotate and release" cases rolled out for 4 s with the
default learned model (`models/contact_gnn.pt`, served with the non-learned projection `agatha.contact_gnn.project`
after every step) and compared with PyBullet, which ends with a face flat on the table in 100% of the single-box cases.
Before the fix, 34% of the single boxes ended more than 2 degrees away from flat (visibly crooked), 8.8% more than
8 degrees, and 127 of 2,032 were frozen mid-topple (at rest on an edge or corner with the centre of mass not over the contact).
The learned model without the projection was better (9.9% / 6.3% / 42): the projection was making it worse.

## The cause

Two things, both in the sleep rule at the end of `project()`; the model's own weakness makes them visible.

1. **The contact patch test treated an edge contact as a face.** Sleep requires the centre of mass to be over the
   "contact patch", and the patch was the hull of the corners within 5 mm of the support. On a box 5-15 cm wide a
   1-3 degree tilt lifts the far corners by only 1-5 mm, so all four bottom corners counted as "in contact", the patch
   was the whole face, and the box was declared stable. Re-evaluated on the final poses of the baseline run: 88% of the boxes that ended
   tilted 1-2 degrees and 73% of those tilted 2-4 degrees passed the old stability test, while 3% / 0% pass a test
   with the true contact (corners within 1 mm).
2. **The sleep thresholds are far above the model's settling speed, and sleep is absorbing.** The model lets a tilted
   box sink towards flat slowly (PyBullet: 9% of boxes still tilted > 2 degrees after 0.5 s; model alone: 48%), i.e.
   well below the sleep thresholds (0.03 m/s, 0.25 rad/s = 14 deg/s). A "stable", slow box gets its velocity zeroed
   every step and, when it moved < 2 mm / 1 degree in the step, its previous pose back. From zero velocity the model
   predicts almost no motion (its known motion-onset weakness), so the box never leaves that state: in the baseline, 96-100% of the single boxes that ended
   tilted by 0.3-8 degrees had exactly zero velocity.
3. The boxes frozen at 10-45 degrees are of two kinds: near the unstable balance point on an edge (centre of mass
   within the 3 mm `stable_margin` of the edge line: slept by the same rule), and boxes the model itself parks on an
   edge with ~0.01 rad/s of creep (also present without the projection: 42 cases).

## What changed in `project()` (all of it non-learned)

The learned model is unchanged. `project()` has new keywords `landing=True, settle_deg=6., tip=1., tip_ang=None, flat_deg=.1`;
`landing=False` is the previous projection, bit for bit (checked against a snapshot of the old file over 80 steps x
2,432 cases, CPU and GPU: 0 differing values).

The landing rules only look at a box that (a) touches exactly one surface, the support top or the floor, (b) is
entirely over the support (not overhanging its edge), (c) is not within 1 cm of another box, (d) is not static and
not jointed, (e) is not on a slope it slides on, and (f) whose nearest face is more than `flat_deg` (0.1 degree) from
parallel to that surface. "Flat" is always relative to the surface the box is on (a tilted support counts). For such a box:

- **No false sleep.** It is never put to sleep and its pose is never frozen: a convex box on one plane is only at
  rest with a face flat on it.
- **Tip** (`tip`, default 1.0 = one step of gravity). If it turns slower than `tip_ang` (default: the sleep threshold)
  and its centre of mass is over the surface, its angular velocity about its contact edge/corner is raised to at
  least (previous value + g x lever / (I + r^2) x 0.05 s), the rigid-body tipping acceleration, in the direction gravity turns
  it (towards the nearest face within 5 mm of balance, where the direction would be noise). The pose turns about the
  contact by half of that gain x the step and stops where the next corner meets the surface. This is a lower bound on
  tipping, applied only in the regime where the model stalls; it never raises the centre of mass.
- **Settle** (`settle_deg`, default 6). If gravity turns it towards its nearest face (turning flat lowers the centre of
  mass), the flat pose is stable (centre of mass over the face and over the support), and it is either slow and
  tilted by less than `settle_deg`, or this step's tip rotation reaches the face, it is rotated flat about its
  contact edge/corner, re-seated on the surface and put to sleep. It moves by the lever arm about the contact only
  and gains no energy.

The contact used for the pivot is the lowest corners in a band well below the rise of the tilted face
(0.25 x sin(tilt) x smallest side, at most 1 mm), so a slightly tilted box pivots on its low edge, not on the middle of
its face.

Everything else (flat boxes, stacks, boxes leaning on other boxes, boxes on the table edge, static bodies, slopes
steeper than the friction angle, jointed bodies) is handled exactly as before; the resting demo desk is bit-identical
with and without the rules (test). With sleeping disabled (`sleep_ang=0`) the landing rules are off too.

What is learned and what is not: free flight, impact, bouncing, sliding, and tumbling at speed come from the model.
The last part of the landing, turning slower than 0.25 rad/s on an edge or corner, is now decided by the rules above.

## Before / after (`models/runs/landing_eval.{json,md}`, 4 s rollouts, `models/contact_gnn.pt`)

Tilt = angle between the box's nearest face and the support plane at 4 s; "frozen mid-topple" = at rest, tilt > 8 degrees,
touching no other box, centre of mass not over its lowest corners. Time to rest: PyBullet is the first row.

**All single-box cases** (2032 cases)

| engine | tilt median (deg) | mean | p90 | > 2 deg | > 8 deg | same face as PyBullet | rest pos err median (mm) | time to rest mean (s) | still moving at 4 s | frozen mid-topple | penetration mean (mm) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| PyBullet (truth) | 0.00 | 0.00 | 0.00 | 0.0% | 0.0% | 100.0% | 0.0 | 0.40 | 0.0% | 0 | 0.087 |
| model only (no projection) | 0.08 | 1.85 | 1.83 | 9.9% | 6.3% | 94.7% | 10.8 | 0.98 | 6.1% | 42 | 0.521 |
| model + projection, before (`landing=False`) | 1.24 | 3.93 | 5.64 | 34.1% | 8.8% | 94.9% | 13.1 | 0.59 | 0.8% | 127 | 0.000 |
| **model + projection, after (new default)** | 0.00 | 0.08 | 0.02 | 0.3% | 0.2% | 95.1% | 12.6 | 0.45 | 0.0% | 3 | 0.000 |

**Fraction of single-box cases tilted > 2 deg over time**

| engine | 0s | 0.5s | 1s | 1.5s | 2s | 2.5s | 3s | 3.5s | 4s |
|---|---|---|---|---|---|---|---|---|---|
| PyBullet (truth) | 100% | 9% | 2% | 1% | 0% | 0% | 0% | 0% | 0% |
| model only (no projection) | 100% | 48% | 28% | 21% | 18% | 15% | 13% | 11% | 10% |
| model + projection, before (`landing=False`) | 100% | 49% | 36% | 35% | 35% | 34% | 34% | 34% | 34% |
| **model + projection, after (new default)** | 100% | 24% | 2% | 1% | 0% | 0% | 0% | 0% | 0% |

multi truth-flat subset: tilt>2deg where PyBullet is flat: {'pybullet': 0.0, 'model_only': 37.8, 'projection_legacy': 50.1, 'projection': 30.8} truth flat frac 0.8675
t_rest truth mean single_all 0.3986958661417323

**Single box, friction 0.5** (1200 cases)

| engine | tilt median (deg) | mean | p90 | > 2 deg | > 8 deg | same face as PyBullet | rest pos err median (mm) | time to rest mean (s) | still moving at 4 s | frozen mid-topple | penetration mean (mm) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| PyBullet (truth) | 0.00 | 0.00 | 0.00 | 0.0% | 0.0% | 100.0% | 0.0 | 0.40 | 0.0% | 0 | 0.093 |
| model only (no projection) | 0.08 | 1.90 | 1.83 | 9.9% | 6.2% | 94.5% | 10.9 | 0.98 | 6.2% | 24 | 0.540 |
| model + projection, before (`landing=False`) | 1.27 | 3.57 | 4.86 | 34.9% | 7.2% | 95.2% | 13.6 | 0.61 | 0.9% | 64 | 0.000 |
| **model + projection, after (new default)** | 0.00 | 0.09 | 0.00 | 0.3% | 0.2% | 94.9% | 13.5 | 0.46 | 0.1% | 2 | 0.000 |

**Single box, friction 0.2-1.0** (600 cases)

| engine | tilt median (deg) | mean | p90 | > 2 deg | > 8 deg | same face as PyBullet | rest pos err median (mm) | time to rest mean (s) | still moving at 4 s | frozen mid-topple | penetration mean (mm) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| PyBullet (truth) | 0.00 | 0.00 | 0.00 | 0.0% | 0.0% | 100.0% | 0.0 | 0.41 | 0.0% | 0 | 0.090 |
| model only (no projection) | 0.07 | 1.49 | 1.07 | 8.3% | 5.7% | 94.8% | 11.9 | 0.98 | 4.7% | 14 | 0.521 |
| model + projection, before (`landing=False`) | 1.20 | 3.25 | 4.55 | 30.0% | 6.7% | 94.3% | 13.4 | 0.61 | 0.7% | 32 | 0.000 |
| **model + projection, after (new default)** | 0.00 | 0.07 | 0.02 | 0.5% | 0.2% | 94.5% | 13.5 | 0.47 | 0.0% | 1 | 0.000 |

**At rest on an edge, zero lift** (200 cases)

| engine | tilt median (deg) | mean | p90 | > 2 deg | > 8 deg | same face as PyBullet | rest pos err median (mm) | time to rest mean (s) | still moving at 4 s | frozen mid-topple | penetration mean (mm) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| PyBullet (truth) | 0.00 | 0.00 | 0.00 | 0.0% | 0.0% | 100.0% | 0.0 | 0.33 | 0.0% | 0 | 0.044 |
| model only (no projection) | 0.11 | 2.20 | 3.58 | 13.0% | 8.0% | 95.5% | 6.8 | 0.90 | 9.0% | 3 | 0.311 |
| model + projection, before (`landing=False`) | 1.09 | 6.86 | 30.65 | 33.5% | 19.5% | 95.5% | 7.7 | 0.35 | 0.5% | 21 | 0.000 |
| **model + projection, after (new default)** | 0.00 | 0.07 | 0.34 | 0.0% | 0.0% | 97.5% | 7.7 | 0.31 | 0.0% | 0 | 0.000 |

**The four reported cases x 8 yaw angles** (32 cases)

| engine | tilt median (deg) | mean | p90 | > 2 deg | > 8 deg | same face as PyBullet | rest pos err median (mm) | time to rest mean (s) | still moving at 4 s | frozen mid-topple | penetration mean (mm) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| PyBullet (truth) | 0.00 | 0.00 | 0.02 | 0.0% | 0.0% | 100.0% | 0.0 | 0.55 | 0.0% | 0 | 0.071 |
| model only (no projection) | 0.42 | 4.33 | 9.08 | 21.9% | 12.5% | 93.8% | 20.1 | 1.80 | 12.5% | 1 | 1.122 |
| model + projection, before (`landing=False`) | 4.95 | 12.07 | 29.42 | 81.2% | 40.6% | 93.8% | 23.5 | 1.00 | 3.1% | 10 | 0.000 |
| **model + projection, after (new default)** | 0.00 | 0.05 | 0.02 | 0.0% | 0.0% | 100.0% | 14.9 | 0.48 | 0.0% | 0 | 0.000 |

**All single-box cases released with zero lift** (518 cases)

| engine | tilt median (deg) | mean | p90 | > 2 deg | > 8 deg | same face as PyBullet | rest pos err median (mm) | time to rest mean (s) | still moving at 4 s | frozen mid-topple | penetration mean (mm) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| PyBullet (truth) | 0.00 | 0.00 | 0.00 | 0.0% | 0.0% | 100.0% | 0.0 | 0.35 | 0.0% | 0 | 0.054 |
| model only (no projection) | 0.10 | 1.84 | 2.61 | 11.4% | 6.6% | 94.6% | 9.9 | 0.96 | 7.3% | 8 | 0.407 |
| model + projection, before (`landing=False`) | 1.29 | 5.61 | 22.11 | 37.5% | 14.3% | 96.1% | 11.0 | 0.49 | 0.6% | 52 | 0.000 |
| **model + projection, after (new default)** | 0.00 | 0.06 | 0.04 | 0.0% | 0.0% | 96.5% | 11.5 | 0.35 | 0.0% | 0 | 0.000 |

**Tilted box released next to / onto resting boxes** (400 cases)

| engine | tilt median (deg) | mean | p90 | > 2 deg | > 8 deg | same face as PyBullet | rest pos err median (mm) | time to rest mean (s) | still moving at 4 s | frozen mid-topple | penetration mean (mm) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| PyBullet (truth) | 0.00 | 2.01 | 11.30 | 13.0% | 11.8% | 100.0% | 0.0 | 0.57 | 0.0% | 0 | 0.106 |
| model only (no projection) | 0.91 | 7.36 | 26.43 | 43.2% | 31.0% | 78.2% | 39.3 | 0.93 | 4.0% | 31 | 0.789 |
| model + projection, before (`landing=False`) | 2.50 | 7.26 | 21.96 | 54.0% | 29.0% | 76.0% | 44.0 | 0.71 | 0.8% | 34 | 0.028 |
| **model + projection, after (new default)** | 0.19 | 5.84 | 20.12 | 36.8% | 26.0% | 75.8% | 43.5 | 0.67 | 0.2% | 23 | 0.030 |

In the multi-object group PyBullet itself ends tilted (leaning on another box) in 13% of the cases; on the 87% where
PyBullet ends flat the learned engines end > 2 degrees off in 50.1% (before) / 30.8% (after) / 37.8% (model only).
The rules deliberately leave a box alone while it is within 1 cm of another one, so this group improves only where the box
ends clear of the others.

### Variants

**Variants** (all single-box cases / multi)

| `project()` keywords | single: > 2 deg | > 8 deg | frozen | moving at 4 s | pos err median (mm) | multi: > 2 deg | > 8 deg | frozen |
|---|---|---|---|---|---|---|---|---|
| `projection_legacy` ("landing": false) | 34.1% | 8.8% | 127 | 0.8% | 13.1 | 54.0% | 29.0% | 34 |
| `no_false_sleep` ("landing": true, "settle_deg": 0.0, "tip": 0.0) | 7.5% | 4.2% | 27 | 5.6% | 12.4 | 39.5% | 27.5% | 26 |
| `settle3` ("landing": true, "settle_deg": 3.0, "tip": 0.0) | 6.0% | 4.1% | 27 | 3.9% | 12.2 | 38.8% | 27.3% | 25 |
| `settle6` ("landing": true, "settle_deg": 6.0, "tip": 0.0) | 4.6% | 3.9% | 26 | 2.8% | 12.1 | 37.8% | 27.0% | 25 |
| `tip` ("landing": true, "settle_deg": 0.0, "tip": 1.0, "tip_ang": 0.25) | 0.3% | 0.2% | 3 | 0.2% | 12.7 | 36.8% | 26.0% | 22 |
| `settle3+tip` ("landing": true, "settle_deg": 3.0, "tip": 1.0, "tip_ang": 0.25) | 0.3% | 0.2% | 3 | 0.0% | 12.7 | 36.8% | 26.0% | 22 |
| `settle6+tip` ("landing": true, "settle_deg": 6.0, "tip": 1.0, "tip_ang": 0.25) | 0.3% | 0.2% | 3 | 0.0% | 12.6 | 36.5% | 26.0% | 22 |
| `settle10+tip` ("landing": true, "settle_deg": 10.0, "tip": 1.0, "tip_ang": 0.25) | 0.3% | 0.2% | 3 | 0.0% | 12.6 | 36.5% | 25.8% | 22 |
| `settle6+tip_ang1` ("landing": true, "settle_deg": 6.0, "tip": 1.0, "tip_ang": 1.0) | 0.3% | 0.2% | 4 | 0.0% | 13.3 | 36.2% | 25.5% | 19 |
| `settle6+tip_half` ("landing": true, "settle_deg": 6.0, "tip": 0.5, "tip_ang": 0.25) | 0.3% | 0.1% | 3 | 0.0% | 12.2 | 36.5% | 26.0% | 23 |

`no_false_sleep` alone already removes most of the damage (7.5% vs 34.1%) but leaves the model's own slow settling
and its parked boxes (27 frozen, 5.6% still moving at 4 s). The tip removes those; the settle angle then mostly
shortens the time to rest. The default is `settle6+tip` (`tip_ang` = the sleep threshold): `tip_ang=1` tips boxes that
are already moving and costs position error, `tip=0.5` is within noise of the default.

## Ship gate (`scripts/eval_contact.py`, `data/v2` test, 4,998 episodes, `models/runs/landing_fix_eval.{json,md}`)

Same checkpoint, `new_projected` engine; "shipped" is `models/contact_eval.json`.

| metric (live_physics) | shipped | with the landing rules | allowed | |
|---|---|---|---|---|
| chaotic pos err @2s (m) | 0.12165 | 0.12254 (+0.7%) | <= 0.12409 (+2%) | pass |
| gentle pos err @2s (m) | 0.002691 | 0.002655 | <= 0.002891 | pass |
| mean penetration (mm) | 0.054 | 0.053 | <= 0.074 | pass |
| rest drift (mm/s) | 5.54 | 5.67 (+2.4%) | <= 6.10 (+10%) | pass |
| fell precision | 0.9163 | 0.9156 | >= 0.9063 | pass |
| fell recall | 0.9428 | 0.9421 | >= 0.9328 | pass |
| orientation err @2s (deg) | 15.00 | 14.90 | - | |

all_physics: chaotic 0.11939 -> 0.11983, gentle 0.002664 -> 0.002589, penetration 0.045 -> 0.044 mm, rest drift
5.57 -> 5.65 mm/s, fell P/R 0.9157/0.9403 -> 0.9151/0.9396. `ship_decision.ship` stays true.
Cost: `project()` on CPU (batch 1, 4 threads) 3.9 ms -> 4.3 ms per step.

## Limits

- A box that ends overhanging the table edge keeps the old rules: all 7 single-box cases still tilted > 2 degrees
  after the fix (0.3%) are such boxes, and so are the 3 "frozen" ones. In PyBullet they also bear on the table edge;
  the corner-based contact polygon used here cannot represent that.
- A tilted box within 1 cm of another box (leaning, or dropped onto / beside one) is left alone, so that legitimate
  leaning poses survive. A box that should have slid off its neighbour and landed flat can still stay crooked
  (multi group: 36.8% > 2 degrees vs 13% in PyBullet).
- Which face it lands on is unchanged (95% the same face as PyBullet for single boxes, 76% in the multi group), and so
  is the rest position error (median 13 mm): both are set by the learned tumbling before the box slows down.
- The tip assumes the contact does not slip while the box pivots and uses the box inertia of a uniform solid; on a
  slope steeper than the friction angle nothing is done.
- Jointed bodies are excluded through the keywords `joint_gnn` adds when it re-compiles `project()` from source
  (`_exclude`, `_articulation`); `project()`'s signature must keep ending in `stable_margin` for that patch.
- GPU rollouts are not bit-reproducible (index_add in the model): the multi-group numbers move by a few tenths of a
  percentage point between runs.

## Reproduce

```
CUDA_VISIBLE_DEVICES=0 .venv-train/bin/python scripts/eval_landing.py --model models/contact_gnn.pt --cache <dir> --out models/runs/landing_eval.json
CUDA_VISIBLE_DEVICES=0 .venv-train/bin/python scripts/eval_contact.py --data data/v2 --split test --new models/contact_gnn.pt --out models/runs/landing_fix_eval.json
.venv/bin/python -m pytest -q tests/test_contact_gnn.py tests/test_contact_integration.py tests/test_joint_gnn.py tests/test_router.py tests/test_world_model.py tests/test_api.py
```
