"""Contact-rich PyBullet episodes for the contact-aware world model (data/v2 format).

Every episode is a full 4 s rollout (80 model steps of 0.05 s = 12 substeps at 240 Hz)
with per-frame contact / rest flags, outcome labels and randomized (but recorded)
physical parameters.  CPU only, multiprocessing, deterministic per-episode seeds.

    .venv/bin/python scripts/generate_contact_data.py --out data/v2 \
        --train-episodes 25000 --val-episodes 1000 --test-episodes 2500 --workers 56

Layout: <out>/{train,val,test}/shard_XXXXX.npz + <out>/meta.json (see FORMAT below).
"""
import argparse
import json
import math
import multiprocessing as mp
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pybullet as p
from agatha.coordinates import to_bullet, euler_to_bullet
from agatha.physics import World, summarize
from agatha.schema import (Scene, SceneObject, Support, Push, SetMass, Remove, Tilt, Gravity,
                           SupportCollisions)
from generate_data import random_case  # the old generator, used verbatim for 'legacy_layout'

STEP = .05
SUBSTEPS = 12
PROPS_DIM = 9
CONTEXT_DIM = 10
REST_LIN, REST_ANG = .01, .05
CONTACT_DISTANCE = 1e-3
FLOOR_HALF = 10.
SPLITS = ("train", "val", "test")
SPLIT_SEED_OFFSET = dict(train=0, val=7919, test=104729)

FORMAT = dict(
    states="float32 [E,T+1,N,13] pos3, quat xyzw (sign-continuous), linvel3, angvel3 (world); Y-up metres",
    props="float32 [E,N,9] size_x,size_y,size_z,mass_kg,friction,is_static,restitution,rolling_friction,spinning_friction",
    context="float32 [E,10] support_w,support_d,support_height,gravity,tilt_x,tilt_y,tilt_z,support_collisions_enabled,support_friction,support_restitution",
    mask="bool [E,N] real object slots",
    contact_support="bool [E,T+1,N]", contact_floor="bool [E,T+1,N]", contact_object="bool [E,T+1,N]",
    rest="bool [E,T+1,N] dynamic object in contact and |v|<0.01 and |w|<0.05",
    fell="bool [E,N] agatha.physics.summarize semantics", tipped="bool [E,N] agatha.physics.summarize semantics",
    kind="<U24 [E] generator scenario", chaotic="bool [E] outcome based", live_physics="bool [E]",
    episode_index="int64 [E] index within the split (seed = hash(base_seed, split, index))",
    init_penetration="float32 [E] deepest object-object/support/floor overlap at t=0 (m, >=0)",
)

# Generator mix: HARD 0.41 / GENTLE 0.59 by generator (gentle kinds still produce ~20% chaotic outcomes,
# so the outcome-based chaotic fraction lands near 0.5); measured statistics are written to meta.json.
KIND_WEIGHTS = dict(
    topple=.06, edge_fall=.055, stack_collapse=.055, collision=.06, drop=.055, toss=.035, tilt_slide=.06, noclip=.03,
    rest=.15, gentle_push=.125, slide_stop=.105, legacy_layout=.12, desk=.09,
)
HARD_KINDS = ("topple", "edge_fall", "stack_collapse", "collision", "drop", "toss", "tilt_slide", "noclip")
LIVE_FRACTION_NON_LEGACY = .43  # + always-live legacy_layout (12%) -> ~50% of all episodes

RANGES = dict(
    mass_kg="log-uniform 0.05-3 (half of objects: density 150-2500 kg/m3 clipped to that range)",
    size_m="0.02-0.35 per axis (desk items use demo_desk sizes x0.85-1.15, phone 0.009 thick)",
    object_friction="0.5 for ~35% of objects else uniform 0.1-1.0",
    support_friction="0.5 (live) or uniform 0.1-1.0", restitution="0 (live) or uniform 0-0.6 per object",
    support_restitution="0 (live) or uniform 0-0.6", rolling_friction="0 (live) or uniform 0-0.02",
    spinning_friction="0 (live) or uniform 0-0.02",
    non_live="each of the five unobservable parameters is randomized with p=0.7 (at least one)",
    table="w 0.9-1.5, d 0.65-1.0, h 0.6-0.9 (generate_data); desk layouts mostly 1.2x0.8x0.75",
    tilt_deg="tilt_slide 5-35 (12%: 35-60) on x or z (15% both); legacy +-25; sprinkled +-15 on hard kinds",
    gravity="9.81, set_gravity 2-15 in ~6% of episodes", objects="1-8, ~4% static, random yaw on ~75%",
    push="gentle 0.02-0.5 Ns (speed capped 0.8 m/s); hard pushes 0.3-4.5 m/s * mass (<=20 Ns); 2D and 3D",
    drop="gap 0.02-0.30 m, yaw-only / small tilt / fully random orientation, optional initial lin/ang velocity",
    initial_angular_velocity="topple 'spin' mode 1.5-8 rad/s; drop 0-6 rad/s",
    dropped="episodes with non-finite states, |pos|>50, or an object leaving the 20 m floor (y<-0.5 or |x|,|z|>10)",
    other_actions="set_mass ~6%, remove ~5%, set_support_collisions False ('noclip') ~3%",
)

DESK_ITEMS = dict(  # name: (size, mass) as in scenes/demo_desk.json (+ a bottle)
    laptop=((.33, .025, .23), 1.4), mug=((.085, .11, .085), .32), book=((.2, .035, .14), .45),
    phone=((.075, .009, .15), .19), box=((.13, .17, .13), .6), bottle=((.07, .24, .07), .6),
)
DEMO_LAYOUT = [  # (item, size, x, z, bottom offset above desk, mass)
    ("laptop", (.33, .025, .23), -.2, -.12, 0., 1.4), ("mug", (.085, .11, .085), .49, .16, 0., .32),
    ("book", (.2, .035, .14), .07, -.17, 0., .45), ("book", (.19, .035, .14), .07, -.17, .035, .4),
    ("box", (.13, .17, .13), -.3, .2, 0., .6), ("phone", (.075, .009, .15), .13, .19, 0., .19),
]


# ----------------------------------------------------------------------------- math helpers
def qmul(a, b):
    ax, ay, az, aw = a; bx, by, bz, bw = b
    return (aw*bx+ax*bw+ay*bz-az*by, aw*by-ax*bz+ay*bw+az*bx, aw*bz+ax*by-ay*bx+az*bw, aw*bw-ax*bx-ay*by-az*bz)

def q_axis(axis, angle):
    axis = np.asarray(axis, float); axis = axis/np.linalg.norm(axis); s = math.sin(angle/2)
    return (axis[0]*s, axis[1]*s, axis[2]*s, math.cos(angle/2))

def q_yaw(yaw):
    return (0., math.sin(yaw/2), 0., math.cos(yaw/2))

def rotmat(q):
    return np.asarray(p.getMatrixFromQuaternion(q)).reshape(3, 3)

def loguniform(rng, lo, hi):
    return float(math.exp(rng.uniform(math.log(lo), math.log(hi))))


# ----------------------------------------------------------------------------- scene builder
class Builder:
    """Places boxes on a table without initial interpenetration and collects API actions."""

    def __init__(self, rng: np.random.Generator, live: bool, table=None):
        self.rng = rng; self.live = live
        if table is None:
            table = (float(rng.uniform(.9, 1.5)), float(rng.uniform(.65, 1.0)), float(rng.uniform(.6, .9)))
        self.w, self.d, self.h = table
        names = ("restitution", "rolling_friction", "spinning_friction", "support_friction", "support_restitution")
        self.randomized = {n: (not live) and rng.random() < .7 for n in names}
        if not live and not any(self.randomized.values()):
            self.randomized[names[int(rng.integers(len(names)))]] = True
        self.support_friction = float(rng.uniform(.1, 1.)) if self.randomized["support_friction"] else .5
        self.support_restitution = float(rng.uniform(0, .6)) if self.randomized["support_restitution"] else 0.
        self.objs: list[dict] = []
        self.actions: list = []
        self.corotate = False
        self.alias = None

    # -- object factories (positions relative to the table top: y=0 is the surface, x/z cluster-local)
    def size(self, style="any"):
        rng = self.rng
        if style == "flat": return np.array([rng.uniform(.08, .35), rng.uniform(.02, .06), rng.uniform(.08, .35)])
        if style == "tall": return np.array([rng.uniform(.02, .09), rng.uniform(.12, .35), rng.uniform(.02, .09)])
        if style == "cube": return np.clip(rng.uniform(.04, .2)*rng.uniform(.8, 1.2, 3), .02, .35)
        if style == "legacy": return rng.uniform([.06, .02, .06], [.24, .23, .24])
        while True:
            s = np.array([loguniform(rng, .03, .35), loguniform(rng, .02, .35), loguniform(rng, .03, .35)])
            if s[1] <= 3.5*min(s[0], s[2]): return s

    def box(self, size=None, style=None, q=None, mass=None, friction=None, static=False, yaw=None):
        rng = self.rng
        if size is None:
            size = self.size(style or str(rng.choice(["any", "any", "flat", "cube", "legacy", "tall"],
                                                     p=[.3, .15, .15, .15, .2, .05])))
        size = np.asarray(size, float)
        if q is None:
            if yaw is None: yaw = float(rng.uniform(-math.pi, math.pi)) if rng.random() < .75 else 0.
            q = q_yaw(yaw)
        if mass is None:
            if rng.random() < .5: mass = float(np.clip(float(np.prod(size))*loguniform(rng, 150, 2500), .05, 3.))
            else: mass = loguniform(rng, .05, 3.)
        if friction is None: friction = .5 if rng.random() < .35 else float(rng.uniform(.1, 1.))
        ext = np.abs(rotmat(q))@size/2
        return dict(size=size, q=tuple(float(v) for v in q), ext=ext, pos=np.array([0., ext[1], 0.]), mass=float(mass),
                    friction=float(friction), static=bool(static), lin=None, ang=None, supports=False, pushed=False)

    def stack(self, count, collapse=False):
        rng = self.rng
        yaw = float(rng.uniform(-math.pi, math.pi)) if rng.random() < .8 else 0.
        rot = rotmat(q_yaw(yaw)); cluster = []; top = 0.; centre = np.zeros(3); prev = None
        for i in range(count):
            if prev is None:
                size = np.array([rng.uniform(.07, .3), rng.uniform(.02, .14), rng.uniform(.07, .3)])
            else:
                scale = rng.uniform(.8, 1.35, 2) if collapse and rng.random() < .4 else rng.uniform(.55, 1., 2)
                size = np.array([np.clip(prev[0]*scale[0], .03, .35), rng.uniform(.02, .14), np.clip(prev[2]*scale[1], .03, .35)])
                frac = rng.uniform(.25, .7) if collapse else rng.uniform(0, .1)
                centre = centre+rot@np.array([rng.uniform(-1, 1)*frac*prev[0], 0, rng.uniform(-1, 1)*frac*prev[2]])
            o = self.box(size, yaw=yaw+(float(rng.uniform(-.25, .25)) if i and rng.random() < .4 else 0.))
            o["pos"] = np.array([centre[0], top+size[1]/2, centre[2]]); top += size[1]
            if cluster: cluster[-1]["supports"] = True
            cluster.append(o); prev = size
        return cluster

    # -- placement
    @staticmethod
    def overlap(a, b, margin=.004):
        dx = np.abs(a["pos"]-b["pos"]); lim = a["ext"]+b["ext"]
        return dx[0] < lim[0]+margin and dx[2] < lim[2]+margin and dx[1] < lim[1]-1e-6

    def free(self, cluster):
        return not any(self.overlap(a, b) for a in cluster for b in self.objs)

    def commit(self, cluster):
        self.objs.extend(cluster); return cluster

    def place(self, cluster, near=None, inset=.015, tries=50, absolute_y=False):
        """Shift a cluster (built around x=z=0, y relative to the surface) to a free spot on the table."""
        rng = self.rng
        lo = np.min([o["pos"]-o["ext"] for o in cluster], 0); hi = np.max([o["pos"]+o["ext"] for o in cluster], 0)
        xr = (-self.w/2+inset-lo[0], self.w/2-inset-hi[0]); zr = (-self.d/2+inset-lo[2], self.d/2-inset-hi[2])
        if xr[0] > xr[1] or zr[0] > zr[1]: return None
        base = [o["pos"].copy() for o in cluster]
        for _ in range(tries):
            if near is None: x, z = rng.uniform(*xr), rng.uniform(*zr)
            else:
                x = float(np.clip(near[0]+rng.uniform(-near[2], near[2]), *xr)); z = float(np.clip(near[1]+rng.uniform(-near[2], near[2]), *zr))
            for o, b in zip(cluster, base):
                o["pos"] = b+np.array([x, 0. if absolute_y else self.h, z])
            if self.free(cluster): return self.commit(cluster)
        for o, b in zip(cluster, base): o["pos"] = b
        return None

    def fill(self, total, stacks=.2, static=.065):
        """Add resting distractors (single boxes and stable stacks) until `total` objects exist."""
        rng = self.rng; attempts = 0
        while len(self.objs) < total and attempts < 30:
            attempts += 1; room = total-len(self.objs)
            if room >= 2 and rng.random() < stacks:
                self.place(self.stack(int(rng.integers(2, min(room, 4)+1))))
            else:
                self.place([self.box(static=rng.random() < static)])

    def dynamic(self):
        return [o for o in self.objs if not o["static"]]

    def pick(self, count=1, pool=None):
        pool = self.dynamic() if pool is None else pool
        if not pool: return []
        return [pool[int(i)] for i in self.rng.permutation(len(pool))[:count]]

    # -- actions (ids are assigned in finalize; actions keep object references until then)
    def push(self, o, direction, speed=None, impulse=None):
        if impulse is None: impulse = o["mass"]*speed
        direction = [float(v) for v in direction]
        if math.hypot(*direction) < 1e-6: direction = [1., 0.] if len(direction) == 2 else [1., 0., 0.]
        o["pushed"] = True
        self.actions.append(("push", o, tuple(direction), float(np.clip(impulse, 1e-3, 20.))))

    def direction(self, theta=None, three_d=.15, up=(-.3, .6)):
        rng = self.rng
        theta = float(rng.uniform(-math.pi, math.pi)) if theta is None else theta
        if rng.random() < three_d: return (math.cos(theta), float(rng.uniform(*up)), math.sin(theta))
        return (math.cos(theta), math.sin(theta))

    def tilt(self, angle_deg, axis):
        self.actions.append(("tilt", float(np.clip(angle_deg, -60, 60)), axis))


# ----------------------------------------------------------------------------- scenarios
def count(rng, lo=1, hi=8):
    return int(rng.integers(lo, hi+1))

def s_rest(b: Builder):
    rng = b.rng; total = count(rng)
    if total >= 2 and rng.random() < .35: b.place(b.stack(int(rng.integers(2, min(total, 5)+1))))
    b.fill(total, stacks=.25)

def s_gentle_push(b: Builder):
    rng = b.rng; b.fill(count(rng))
    for o in b.pick(1 if rng.random() < .75 else 2):
        impulse = min(loguniform(rng, .02, .5), o["mass"]*.8)
        b.push(o, b.direction(three_d=.12, up=(-.3, .4)), impulse=max(impulse, .02))

def s_slide_stop(b: Builder):
    rng = b.rng; total = count(rng); primaries = []
    for _ in range(min(total, int(rng.integers(1, 4)))):
        placed = b.place([b.box(style="flat")], inset=.05)
        if placed: primaries += placed
    b.fill(total, stacks=.1)
    for o in primaries:
        if o["static"]: continue
        theta = float(rng.uniform(-math.pi, math.pi)); c, s = math.cos(theta), math.sin(theta)
        reach = min((b.w/2-abs(o["pos"][0]))/max(abs(c), 1e-6) if c*o["pos"][0] >= 0 else (b.w/2+abs(o["pos"][0]))/max(abs(c), 1e-6),
                    (b.d/2-abs(o["pos"][2]))/max(abs(s), 1e-6) if s*o["pos"][2] >= 0 else (b.d/2+abs(o["pos"][2]))/max(abs(s), 1e-6))
        distance = max(.03, min(loguniform(rng, .03, .35), .75*reach))
        mu = o["friction"]*b.support_friction  # Bullet multiplies the two lateral frictions
        b.push(o, (c, s), speed=float(np.clip(math.sqrt(2*mu*9.81*distance), .15, 3.)))

def s_desk(b: Builder):
    rng = b.rng; demo = abs(b.w-1.2) < 1e-9 and abs(b.d-.8) < 1e-9
    def item(name, size=None, mass=None, yaw=None):
        base, m = DESK_ITEMS[name]
        size = np.asarray(base)*rng.uniform(.85, 1.15, 3) if size is None else np.asarray(size, float)
        o = b.box(size, mass=(m*float(rng.uniform(.7, 1.4)) if mass is None else mass),
                  friction=.5 if rng.random() < .8 else None, yaw=yaw)
        o["item"] = name; return o
    if demo and rng.random() < .3:  # the shipped demo layout (optionally jittered / thinned)
        jitter = .012 if rng.random() < .6 else 0.
        keep = rng.random(len(DEMO_LAYOUT)) < (1. if rng.random() < .5 else .75)
        for use, (name, size, x, z, bottom, mass) in zip(keep, DEMO_LAYOUT):
            if not use and name != "book": continue
            o = item(name, size, mass, yaw=0.)
            shift = rng.normal(0, jitter, 2) if jitter and bottom == 0 else np.zeros(2)
            o["pos"] = np.array([x+shift[0], b.h+bottom+size[1]/2, z+shift[1]])
            if bottom > 0 and b.objs: o["pos"][[0, 2]] = b.objs[-1]["pos"][[0, 2]]; b.objs[-1]["supports"] = True
            if b.free([o]): b.commit([o])
    else:
        total = count(rng, 2, 7)
        if rng.random() < .35:
            lower = item("book", yaw=float(rng.uniform(-math.pi, math.pi)) if rng.random() < .5 else 0.)
            upper = item("book"); upper["q"], upper["ext"] = lower["q"], np.abs(rotmat(lower["q"]))@upper["size"]/2
            upper["size"][[0, 2]] = np.minimum(upper["size"][[0, 2]], lower["size"][[0, 2]])
            upper["ext"] = np.abs(rotmat(upper["q"]))@upper["size"]/2
            upper["pos"] = np.array([rng.uniform(-.01, .01), lower["size"][1]+upper["size"][1]/2, rng.uniform(-.01, .01)])
            lower["supports"] = True; b.place([lower, upper])
        attempts = 0
        while len(b.objs) < total and attempts < 30:
            attempts += 1
            b.place([item(str(rng.choice(list(DESK_ITEMS))), yaw=0. if rng.random() < .5 else None)])
    if rng.random() < .3:  # static desk duplicating the support, as in scenes/demo_desk.json
        size = np.array([b.w, .078, b.d]); o = b.box(size, mass=20., friction=.5, static=True, yaw=0.)
        o["pos"] = np.array([0., b.h-.04, 0.]); b.alias = o
        # physics.World tilts only its internal support and leaves this desk flat and colliding: teach that case too
        if rng.random() < .3: b.tilt(float(rng.uniform(-20, 20)), str(rng.choice(["x", "z"])))
    if rng.random() < .88:
        for o in b.pick(1 if rng.random() < .85 else 2):
            if rng.random() < .6: direction = [(1., 0.), (-1., 0.), (0., 1.), (0., -1.)][int(rng.integers(4))]
            else: direction = b.direction(three_d=.1)
            b.push(o, direction, impulse=loguniform(rng, .05, 1.))

def s_topple(b: Builder):
    rng = b.rng; total = count(rng); mode = str(rng.choice(["push", "spin", "lean", "domino"], p=[.35, .2, .25, .2]))
    def tall(**kw):
        return b.box(style="tall", friction=float(rng.uniform(.4, 1.)) if rng.random() < .7 else None, **kw)
    if mode == "domino" and total >= 2:
        n = int(rng.integers(2, min(total, 5)+1)); theta = float(rng.uniform(-math.pi, math.pi))
        size = np.array([rng.uniform(.02, .04), rng.uniform(.1, .25), rng.uniform(.04, .1)])
        gap = float(rng.uniform(.35, .8))*size[1]; cluster = []
        for i in range(n):
            o = b.box(size*rng.uniform(.9, 1.1, 3), yaw=-theta, friction=float(rng.uniform(.4, 1.)))
            o["pos"] = np.array([math.cos(theta)*gap*i, o["ext"][1], math.sin(theta)*gap*i]); cluster.append(o)
        if b.place(cluster):
            first = cluster[0]
            if rng.random() < .6: b.push(first, (math.cos(theta), math.sin(theta)), speed=float(rng.uniform(.5, 2.2)))
            else: first["ang"] = np.array([-math.sin(theta), 0, math.cos(theta)])*-float(rng.uniform(2., 6.))
        else: mode = "push"
    if mode != "domino" or total < 2:
        for _ in range(min(total, int(rng.integers(1, 4)))):
            yaw = float(rng.uniform(-math.pi, math.pi))
            if mode == "lean":
                phi = float(rng.uniform(-math.pi, math.pi)); angle = math.radians(float(rng.uniform(5, 45)))
                o = tall(q=qmul(q_axis((math.cos(phi), 0, math.sin(phi)), angle), q_yaw(yaw)))
            else: o = tall(yaw=yaw)
            if not b.place([o], inset=.03): continue
            if mode == "push" or (mode == "lean" and rng.random() < .25):
                b.push(o, b.direction(three_d=.2, up=(-.2, .5)), speed=float(rng.uniform(.4, 3.)))
            elif mode == "spin":
                phi = float(rng.uniform(-math.pi, math.pi))
                o["ang"] = np.array([math.cos(phi), 0, math.sin(phi)])*float(rng.uniform(1.5, 8.))
                if rng.random() < .3: o["lin"] = np.array([rng.uniform(-.5, .5), 0, rng.uniform(-.5, .5)])
    b.fill(total)

def s_edge_fall(b: Builder):
    rng = b.rng; total = count(rng)
    for _ in range(min(total, int(rng.integers(1, 4)))):
        o = b.box(); axis = int(rng.choice([0, 2])); sign = float(rng.choice([-1., 1.]))
        half = (b.w if axis == 0 else b.d)/2; other = 2-axis; other_half = (b.d if axis == 0 else b.w)/2
        mode = str(rng.choice(["push_out", "overhang"], p=[.65, .35]))
        if mode == "push_out": inside = float(rng.uniform(0., .1))+o["ext"][axis]*float(rng.uniform(0, 1))
        else: inside = o["ext"][axis]*float(rng.uniform(-.5, .6))  # negative: centre of mass beyond the edge
        for _ in range(12):
            pos = np.zeros(3); pos[axis] = sign*(half-inside)
            pos[other] = float(rng.uniform(-1, 1))*max(other_half-o["ext"][other]-.02, .01); pos[1] = b.h+o["ext"][1]
            o["pos"] = pos
            if b.free([o]): b.commit([o]); break
        else: continue
        if mode == "push_out" or rng.random() < .25:
            theta = math.atan2(sign if axis == 2 else 0., sign if axis == 0 else 0.)+float(rng.uniform(-.6, .6))
            speed = float(rng.uniform(.3, 2.5)) if mode == "push_out" else float(rng.uniform(.05, .5))
            b.push(o, b.direction(theta, three_d=.12), speed=speed)
    b.fill(total)

def s_stack_collapse(b: Builder):
    rng = b.rng; total = count(rng, 2); n = int(rng.integers(2, min(total, 5)+1))
    mode = str(rng.choice(["push_low", "push_any", "misaligned"], p=[.4, .25, .35]))
    cluster = b.place(b.stack(n, collapse=mode == "misaligned"), inset=.04)
    if cluster and (mode != "misaligned" or rng.random() < .2):
        o = cluster[0] if mode == "push_low" else cluster[int(rng.integers(len(cluster)))]
        b.push(o, b.direction(three_d=.1, up=(-.2, .4)), speed=float(rng.uniform(.5, 3.5)))
    b.fill(total, stacks=.3)

def s_collision(b: Builder):
    rng = b.rng; total = count(rng, 2); targets = []
    for _ in range(int(rng.integers(1, 3)) if total >= 4 else 1):
        room = total-len(b.objs)-1
        if room < 1: break
        if room >= 2 and rng.random() < .25: first = b.place(b.stack(int(rng.integers(2, min(room, 4)+1))), inset=.08)
        else: first = b.place([b.box()], inset=.08)
        if not first: continue
        group = list(first)
        for _ in range(int(rng.integers(0, 4))):  # pile-up cluster around the target
            if len(b.objs) >= total-1: break
            more = b.place([b.box()], near=(first[0]["pos"][0], first[0]["pos"][2], .2))
            if more: group += more
        target = first[0]
        for _ in range(20):
            theta = float(rng.uniform(-math.pi, math.pi)); distance = float(rng.uniform(.15, .6))
            o = b.box(style=str(rng.choice(["cube", "any", "flat", "legacy"])))
            o["pos"] = np.array([target["pos"][0]-math.cos(theta)*distance, b.h+o["ext"][1], target["pos"][2]-math.sin(theta)*distance])
            if abs(o["pos"][0]) > b.w/2-o["ext"][0] or abs(o["pos"][2]) > b.d/2-o["ext"][2] or not b.free([o]): continue
            b.commit([o]); aim = theta+float(rng.uniform(-.15, .15))
            b.push(o, b.direction(aim, three_d=.15, up=(0., .5)), speed=float(rng.uniform(.8, 4.5)))
            break
        targets += group
    b.fill(total)
    if not any(o["pushed"] for o in b.objs):
        for o in b.pick(1): b.push(o, b.direction(), speed=float(rng.uniform(.8, 4.)))

def s_drop(b: Builder):
    rng = b.rng; total = count(rng); droppers = min(total, int(rng.integers(1, 5)))
    b.fill(total-droppers, stacks=.25)
    for _ in range(droppers):
        for _ in range(25):
            r = rng.random(); yaw = float(rng.uniform(-math.pi, math.pi))
            if r < .3: q = q_yaw(yaw)
            elif r < .6:
                phi = float(rng.uniform(-math.pi, math.pi))
                q = qmul(q_axis((math.cos(phi), 0, math.sin(phi)), math.radians(float(rng.uniform(2, 25)))), q_yaw(yaw))
            else:
                q = rng.normal(size=4); q = tuple(q/np.linalg.norm(q))
            o = b.box(q=q)
            if b.objs and rng.random() < .35:
                under = b.objs[int(rng.integers(len(b.objs)))]
                x = under["pos"][0]+float(rng.uniform(-.06, .06)); z = under["pos"][2]+float(rng.uniform(-.06, .06))
            else:
                x = float(rng.uniform(-1, 1))*(b.w/2-.03); z = float(rng.uniform(-1, 1))*(b.d/2-.03)
            bottom = b.h
            for other in b.objs:
                if abs(x-other["pos"][0]) < o["ext"][0]+other["ext"][0]+.004 and abs(z-other["pos"][2]) < o["ext"][2]+other["ext"][2]+.004:
                    bottom = max(bottom, other["pos"][1]+other["ext"][1])
            o["pos"] = np.array([x, bottom+float(rng.uniform(.02, .30))+o["ext"][1], z])
            if not b.free([o]): continue
            if rng.random() < .25: o["lin"] = np.array([rng.uniform(-.8, .8), rng.uniform(-1., .5), rng.uniform(-.8, .8)])
            if rng.random() < .25:
                axis = rng.normal(size=3); o["ang"] = axis/np.linalg.norm(axis)*float(rng.uniform(0, 6.))
            b.commit([o]); break

def s_toss(b: Builder):
    rng = b.rng; total = count(rng)
    if rng.random() < .5:  # bottle-like primary, as in prepare_ab_data
        o = b.box(np.array([rng.uniform(.06, .10), rng.uniform(.18, .30), rng.uniform(.06, .10)]), mass=float(rng.uniform(.3, 1.)))
        primary = b.place([o], inset=.05)
    else: primary = None
    b.fill(total)
    chosen = list(primary or [])+b.pick(1 if rng.random() < .7 else 2)
    for o in chosen[:2]:
        if o["static"] or o["pushed"]: continue
        if rng.random() < .7: direction = (float(rng.uniform(-.4, .4)), 1., float(rng.uniform(-.4, .4)))
        else: direction = (float(rng.uniform(-1, 1)), float(rng.uniform(.5, 1.)), float(rng.uniform(-1, 1)))
        b.push(o, direction, speed=float(rng.uniform(1., 4.)))

def s_tilt_slide(b: Builder):
    rng = b.rng; b.fill(count(rng), stacks=.2)
    angle = float(rng.uniform(5, 35)) if rng.random() < .88 else float(rng.uniform(35, 60))
    axis = str(rng.choice(["x", "z"])); b.tilt(angle*float(rng.choice([-1., 1.])), axis)
    if rng.random() < .15: b.tilt(float(rng.uniform(3, 20))*float(rng.choice([-1., 1.])), "z" if axis == "x" else "x")
    b.corotate = rng.random() < .5  # objects start resting on the tilted surface instead of the API pose
    if rng.random() < .3:
        for o in b.pick(1): b.push(o, b.direction(three_d=0.), speed=float(rng.uniform(.1, 1.5)))

NOCLIP_BASES = ("rest", "gentle_push", "desk", "collision", "toss", "topple", "slide_stop")

def s_noclip(b: Builder):
    SCENARIOS[str(b.rng.choice(NOCLIP_BASES))](b)
    b.actions.append(("noclip",))

SCENARIOS = dict(rest=s_rest, gentle_push=s_gentle_push, slide_stop=s_slide_stop, desk=s_desk, topple=s_topple,
                 edge_fall=s_edge_fall, stack_collapse=s_stack_collapse, collision=s_collision, drop=s_drop,
                 toss=s_toss, tilt_slide=s_tilt_slide, noclip=s_noclip)


def sprinkle(b: Builder, kind: str):
    """Other API actions on top of the scenario: set_gravity ~6%, set_mass ~6%, remove ~5%, small tilts."""
    rng = b.rng
    if rng.random() < .06: b.actions.append(("gravity", float(rng.uniform(2, 15))))
    if rng.random() < .06 and b.dynamic():
        o = b.pick(1)[0]; kg = loguniform(rng, .05, 3.)
        # keep the intended launch speed: pushes on this object are rescaled to the new mass
        b.actions = [(a[0], a[1], a[2], float(np.clip(a[3]*kg/o["mass"], 1e-3, 20.))) if a[0] == "push" and a[1] is o else a
                     for a in b.actions]
        b.actions.insert(0, ("set_mass", o, kg))
    if rng.random() < .05 and len(b.objs) >= 2:
        gentle = kind not in HARD_KINDS
        pool = [o for o in b.objs if not o["pushed"] and not (gentle and o["supports"])]
        if pool: b.actions.append(("remove", pool[int(rng.integers(len(pool)))]))
    if kind in ("collision", "topple", "stack_collapse", "edge_fall", "toss", "drop") and rng.random() < .06:
        b.tilt(float(rng.uniform(-15, 15)), str(rng.choice(["x", "z"])))


def finalize(b: Builder, kind: str) -> dict:
    rng = b.rng
    objs = list(b.objs)+([b.alias] if b.alias is not None else [])
    for i, o in enumerate(objs): o["id"] = f"o{i}"
    scene = Scene(objects=[SceneObject(id=o["id"], label=o.get("item", "box"), size=tuple(float(v) for v in o["size"]),
                                       position=tuple(float(v) for v in o["pos"]),
                                       rotation_euler=tuple(float(v) for v in p.getEulerFromQuaternion(o["q"])),
                                       mass_kg=o["mass"], friction=o["friction"], is_static=o["static"]) for o in objs],
                  support=Support(size=(b.w, b.d), height=b.h))
    actions = []
    for a in b.actions:
        if a[0] == "push": actions.append(Push(type="push", id=a[1]["id"], direction=a[2], impulse_Ns=a[3]))
        elif a[0] == "tilt": actions.append(Tilt(type="tilt_support", angle_deg=a[1], axis=a[2]))
        elif a[0] == "gravity": actions.append(Gravity(type="set_gravity", g=a[1]))
        elif a[0] == "set_mass": actions.append(SetMass(type="set_mass", id=a[1]["id"], kg=a[2]))
        elif a[0] == "remove": actions.append(Remove(type="remove", id=a[1]["id"]))
        elif a[0] == "noclip": actions.append(SupportCollisions(type="set_support_collisions", enabled=False))
    dynamics = None; extra = {o["id"]: (0., 0., 0.) for o in objs}
    if not b.live:
        per_object = {}
        for o in objs:
            values = {}
            if b.randomized["restitution"]: values["restitution"] = float(rng.uniform(0, .6))
            if b.randomized["rolling_friction"]: values["rolling_friction"] = float(rng.uniform(0, .02))
            if b.randomized["spinning_friction"]: values["spinning_friction"] = float(rng.uniform(0, .02))
            per_object[o["id"]] = values
            extra[o["id"]] = (values.get("restitution", 0.), values.get("rolling_friction", 0.), values.get("spinning_friction", 0.))
        dynamics = dict(objects=per_object, support_friction=b.support_friction, support_restitution=b.support_restitution)
    velocities = {o["id"]: (o["lin"], o["ang"]) for o in objs if o["lin"] is not None or o["ang"] is not None}
    return dict(kind=kind, scene=scene, actions=actions, dynamics=dynamics, extra=extra, velocities=velocities,
                corotate=b.corotate, support_friction=b.support_friction if not b.live else .5,
                support_restitution=b.support_restitution if not b.live else 0., live=b.live)


def make_case(rng: np.random.Generator, max_objects: int = 8, kind: str | None = None) -> dict:
    """One randomized recipe: scene, API actions, unobservable dynamics, extra initial velocities."""
    kinds = list(KIND_WEIGHTS); weights = np.asarray([KIND_WEIGHTS[k] for k in kinds])
    if kind is None: kind = kinds[int(rng.choice(len(kinds), p=weights/weights.sum()))]
    if kind == "legacy_layout":
        scene, actions = random_case(rng)
        return dict(kind=kind, scene=scene, actions=actions, dynamics=None, extra={o.id: (0., 0., 0.) for o in scene.objects},
                    velocities={}, corotate=False, support_friction=.5, support_restitution=0., live=True)
    live = rng.random() < LIVE_FRACTION_NON_LEGACY
    for _ in range(20):
        table = (1.2, .8, .75) if kind == "desk" and rng.random() < .7 else None
        b = Builder(rng, live, table)
        SCENARIOS[kind](b)
        if b.objs and len(b.objs)+(b.alias is not None) <= max_objects:
            sprinkle(b, kind)
            return finalize(b, kind)
    raise RuntimeError(f"could not build a {kind} scene")


# ----------------------------------------------------------------------------- simulation
def _initial_pairs(world: World, present) -> tuple[set, float]:
    """Contacts before the first substep (Bullet has no manifolds yet): closest-point queries."""
    c = world.client; pairs = set(); deepest = 0.
    names = [o.id for o in present]; static = {o.id: o.is_static for o in present}
    alias = set(world._support_alias_ids())
    fixed = ["floor"]+(["support"] if world.scene.support.collisions_enabled else [])
    for i, a in enumerate(names):
        if not world.scene.support.collisions_enabled and a in alias: continue
        for other in fixed+names[i+1:]:
            if static[a] and (other in fixed or static[other]): continue
            if not world.scene.support.collisions_enabled and other in alias: continue
            points = p.getClosestPoints(world.bodies[a], world.bodies[other], CONTACT_DISTANCE, physicsClientId=c)
            if points:
                pairs.add(tuple(sorted((a, other)))); deepest = max(deepest, -min(pt[8] for pt in points))
    return pairs, deepest


def simulate_case(case: dict, steps: int = 80, max_objects: int = 8) -> dict | None:
    """Run one recipe; returns padded arrays or None when the episode is invalid."""
    scene: Scene = case["scene"]
    with World(scene, dynamics=case["dynamics"]) as world:
        c = world.client
        world.apply(case["actions"])
        present = [o for o in world.scene.objects if o.id in world.bodies]
        n = len(present)
        if n == 0 or n > max_objects: return None
        h = world.scene.support.height
        if case["corotate"]:  # rigidly carry the objects with the tilted support
            qs = euler_to_bullet(world.tilt); centre = to_bullet((0, h-.04, 0))
            for o in present:
                body = world.bodies[o.id]
                pos, q = p.getBasePositionAndOrientation(body, physicsClientId=c)
                v, av = p.getBaseVelocity(body, physicsClientId=c)
                pos, q = p.multiplyTransforms(centre, qs, [pos[i]-centre[i] for i in range(3)], q)
                p.resetBasePositionAndOrientation(body, pos, q, physicsClientId=c)
                v = p.multiplyTransforms([0, 0, 0], qs, v, [0, 0, 0, 1])[0]
                p.resetBaseVelocity(body, v, av, physicsClientId=c)
        for id, (lin, ang) in case["velocities"].items():
            if id not in world.bodies or next(o for o in present if o.id == id).is_static: continue
            v, av = p.getBaseVelocity(world.bodies[id], physicsClientId=c)
            if lin is not None: v = [v[i]+to_bullet(lin)[i] for i in range(3)]
            if ang is not None: av = [av[i]+to_bullet(ang)[i] for i in range(3)]
            p.resetBaseVelocity(world.bodies[id], v, av, physicsClientId=c)

        ids = [o.id for o in present]; index = {id: i for i, id in enumerate(ids)}
        dynamic_ids = [o.id for o in present if not o.is_static]; dynamic_set = set(dynamic_ids)
        states = np.zeros((steps+1, max_objects, 13), np.float32); states[:, :, 6] = 1
        contact = np.zeros((3, steps+1, max_objects), bool)

        def record(t, frame, pairs):
            for s in frame["objects"]:
                states[t, index[s["id"]]] = [*s["position"], *s["quaternion"], *s["velocity"], *s["angular_velocity"]]
            if t:
                flip = np.sum(states[t-1, :n, 3:7]*states[t, :n, 3:7], axis=1) < 0
                states[t, :n][flip, 3:7] *= -1
            for id, flags in world.contact_flags(pairs).items():
                contact[:, t, index[id]] = flags

        pairs, init_penetration = _initial_pairs(world, present)
        frames = [world.snapshot(0)]; record(0, frames[0], pairs)
        previous = {pair for pair in pairs if pair[0] in dynamic_set and pair[1] in dynamic_set}
        impact = False
        for t in range(1, steps+1):
            for _ in range(SUBSTEPS):
                before = None if impact else {id: p.getBaseVelocity(world.bodies[id], physicsClientId=c)[0] for id in dynamic_ids}
                world.step()
                pairs = world.contact_pairs(CONTACT_DISTANCE)
                if not impact:
                    current = {pair for pair in pairs if pair[0] in dynamic_set and pair[1] in dynamic_set}
                    for a, b in current-previous:
                        if math.dist(before[a], before[b]) > .5: impact = True
                    previous = current
            frames.append(world.snapshot(t*STEP)); record(t, frames[-1], pairs)
        result = summarize(world.scene, frames, tilt=world.tilt)
        gravity, tilt = world.gravity, list(world.tilt)
        collisions = world.scene.support.collisions_enabled

    if not np.isfinite(states).all() or np.abs(states[:, :, :3]).max() > 50: return None
    # An object that slid off the finite 20 m floor and is in free fall below it is not useful data.
    if states[:, :, 1].min() < -.5 or np.abs(states[:, :, [0, 2]]).max() > FLOOR_HALF: return None
    mask = np.zeros(max_objects, bool); mask[:n] = True
    props = np.zeros((max_objects, PROPS_DIM), np.float32); props[:, 3] = 1
    for i, o in enumerate(present):
        props[i] = [*o.size, o.mass_kg, o.friction, float(o.is_static), *case["extra"][o.id]]
    context = np.asarray([*world.scene.support.size, h, gravity, *tilt, float(collisions),
                          case["support_friction"], case["support_restitution"]], np.float32)
    dynamic = mask & (props[:, 5] < .5)
    speed = np.linalg.norm(states[:, :, 7:10], axis=-1); spin = np.linalg.norm(states[:, :, 10:13], axis=-1)
    rest = contact.any(0) & (speed < REST_LIN) & (spin < REST_ANG) & dynamic   # never set for static slots
    fell = np.zeros(max_objects, bool); tipped = np.zeros(max_objects, bool)
    for id in result["events"]["fell"]: fell[index[id]] = True
    for id in result["events"]["tipped"]: tipped[index[id]] = True
    displacement = np.linalg.norm(states[:, :, :3]-states[:1, :, :3], axis=-1).max(0)
    chaotic = bool(((fell | tipped | (displacement > .30)) & dynamic).any() or impact)
    live = bool(np.all(props[:n, 6:9] == 0) and context[8] == np.float32(.5) and context[9] == 0)
    return dict(states=states, props=props, context=context, mask=mask, contact_support=contact[0],
                contact_floor=contact[1], contact_object=contact[2], rest=rest, fell=fell, tipped=tipped,
                kind=case["kind"], chaotic=chaotic, live_physics=live, init_penetration=np.float32(init_penetration))


def episode_rng(base_seed: int, split: str, index: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([int(base_seed)+SPLIT_SEED_OFFSET[split], SPLITS.index(split), int(index)]))


def generate_episode(base_seed: int, split: str, index: int, steps: int = 80, max_objects: int = 8) -> dict | None:
    return simulate_case(make_case(episode_rng(base_seed, split, index), max_objects), steps, max_objects)


def _chunk(task):
    base_seed, split, shard, indices, steps, max_objects = task
    episodes = []
    for index in indices:
        episode = generate_episode(base_seed, split, index, steps, max_objects)
        if episode is not None: episode["episode_index"] = index
        episodes.append(episode)
    return split, shard, indices[0], episodes


ARRAY_KEYS = ("states", "props", "context", "mask", "contact_support", "contact_floor", "contact_object", "rest",
              "fell", "tipped", "kind", "chaotic", "live_physics", "episode_index", "init_penetration")

def _write_shard(path: Path, episodes: list[dict]):
    arrays = {k: np.stack([e[k] for e in episodes]) for k in ARRAY_KEYS if k not in ("kind", "episode_index")}
    arrays["kind"] = np.asarray([e["kind"] for e in episodes], dtype="<U24")
    arrays["episode_index"] = np.asarray([e["episode_index"] for e in episodes], np.int64)
    arrays["chaotic"] = arrays["chaotic"].astype(bool); arrays["live_physics"] = arrays["live_physics"].astype(bool)
    arrays["init_penetration"] = arrays["init_penetration"].astype(np.float32)
    np.savez_compressed(path, **arrays)


def _stats(episodes: list[dict], steps: int) -> dict:
    if not episodes: return dict(episodes=0)
    kinds = sorted({e["kind"] for e in episodes})
    chaotic = np.asarray([e["chaotic"] for e in episodes]); kind = np.asarray([e["kind"] for e in episodes])
    dynamic = np.stack([e["mask"] & (e["props"][:, 5] < .5) for e in episodes])
    frames = dynamic[:, None, :].repeat(steps+1, 1); total = max(int(frames.sum()), 1)
    frac = lambda key: float((np.stack([e[key] for e in episodes]) & frames).sum()/total)
    objects = max(int(dynamic.sum()), 1)
    return dict(
        episodes=len(episodes), transitions=len(episodes)*steps, object_transitions=int(dynamic.sum())*steps,
        kinds={k: int((kind == k).sum()) for k in kinds},
        chaotic_fraction=float(chaotic.mean()),
        chaotic_fraction_by_kind={k: float(chaotic[kind == k].mean()) for k in kinds},
        generator_hard_fraction=float(np.isin(kind, HARD_KINDS).mean()),
        live_physics_fraction=float(np.mean([e["live_physics"] for e in episodes])),
        rest_fraction_object_frames=frac("rest"), contact_support_fraction=frac("contact_support"),
        contact_floor_fraction=frac("contact_floor"), contact_object_fraction=frac("contact_object"),
        fell_fraction_objects=float((np.stack([e["fell"] for e in episodes]) & dynamic).sum()/objects),
        tipped_fraction_objects=float((np.stack([e["tipped"] for e in episodes]) & dynamic).sum()/objects),
        mean_objects=float(np.stack([e["mask"] for e in episodes]).sum(1).mean()),
        static_fraction_objects=float(np.sum([(e["mask"] & (e["props"][:, 5] > .5)).sum() for e in episodes])/max(np.sum([e["mask"].sum() for e in episodes]), 1)),
        noclip_fraction=float(np.mean([e["context"][7] < .5 for e in episodes])),
        gravity_changed_fraction=float(np.mean([abs(e["context"][3]-9.81) > 1e-4 for e in episodes])),
        tilted_fraction=float(np.mean([np.abs(e["context"][4:7]).max() > 0 for e in episodes])),
        init_penetration_over_1mm_fraction=float(np.mean([e["init_penetration"] > 1e-3 for e in episodes])),
    )


def generate(out: Path, train_episodes: int = 25000, val_episodes: int = 1000, test_episodes: int = 2500,
             steps: int = 80, max_objects: int = 8, workers: int = 56, shard_size: int = 500, seed: int = 2027,
             chunk: int = 10, quiet: bool = False) -> dict:
    out = Path(out); start = time.perf_counter()
    counts = dict(train=train_episodes, val=val_episodes, test=test_episodes)
    tasks = []; pending = {}; buffers = {}
    for split in SPLITS:
        (out/split).mkdir(parents=True, exist_ok=True)
        for old in (out/split).glob("shard_*.npz"): old.unlink()
        for shard, begin in enumerate(range(0, counts[split], shard_size)):
            indices = list(range(begin, min(begin+shard_size, counts[split])))
            pieces = [indices[i:i+chunk] for i in range(0, len(indices), chunk)]
            pending[split, shard] = len(pieces); buffers[split, shard] = {}
            tasks += [(seed, split, shard, piece, steps, max_objects) for piece in pieces]
    kept = {split: [] for split in SPLITS}; dropped = {split: 0 for split in SPLITS}; done = 0
    total = sum(counts.values()); writes = []
    summary_keys = ("kind", "chaotic", "live_physics", "mask", "props", "context", "rest", "contact_support",
                    "contact_floor", "contact_object", "fell", "tipped", "init_penetration")

    workers = max(1, min(workers, len(tasks) or 1))
    context = mp.get_context("fork")
    with ThreadPoolExecutor(4) as writer:
        pool = context.Pool(workers) if workers > 1 else None
        try:
            results = pool.imap_unordered(_chunk, tasks) if pool else map(_chunk, tasks)
            last = start
            for split, shard, first, episodes in results:
                buffers[split, shard][first] = episodes; pending[split, shard] -= 1; done += len(episodes)
                if pending[split, shard] == 0:
                    pieces = buffers.pop((split, shard))
                    ordered = [e for key in sorted(pieces) for e in pieces[key]]
                    good = [e for e in ordered if e is not None]; dropped[split] += len(ordered)-len(good)
                    if good: writes.append(writer.submit(_write_shard, out/split/f"shard_{shard:05d}.npz", good))
                    kept[split] += [{k: e[k] for k in summary_keys} for e in good]
                if not quiet and time.perf_counter()-last > 10:
                    last = time.perf_counter()
                    print(f"{done}/{total} episodes; {done/(last-start):.1f} episodes/s", flush=True)
        finally:
            if pool: pool.close(); pool.join()
        simulated = time.perf_counter()-start
        for w in writes: w.result()
    seconds = time.perf_counter()-start
    everything = [e for split in SPLITS for e in kept[split]]
    meta = dict(
        format=FORMAT, step_s=STEP, substeps_per_step=SUBSTEPS, steps=steps, max_objects=max_objects,
        seed=seed, split_seeds={s: seed+SPLIT_SEED_OFFSET[s] for s in SPLITS},
        seeding="np.random.SeedSequence([seed+split_offset, split_index, episode_index]) per episode",
        requested=counts, counts={s: len(kept[s]) for s in SPLITS}, dropped=dropped, shard_size=shard_size,
        episodes=len(everything), transitions=len(everything)*steps,
        seconds=seconds, simulation_seconds=simulated, workers=workers, episodes_per_second=total/max(simulated, 1e-9),
        kind_weights=KIND_WEIGHTS, hard_kinds=list(HARD_KINDS), ranges=RANGES,
        rest_definition=f"in contact and |v|<{REST_LIN} m/s and |w|<{REST_ANG} rad/s",
        contact_definition=f"Bullet contact point with distance <= {CONTACT_DISTANCE} m in the last substep (frame 0: closest-point query); a static desk duplicating the support counts as support",
        chaotic_definition="any dynamic object fell or tipped (summarize semantics), or moved > 0.30 m from its start at any frame, or a contact between two dynamic objects began after t=0 with relative COM speed > 0.5 m/s",
        live_physics_definition="restitution, rolling and spinning friction 0 for every object; support friction 0.5; support restitution 0",
        splits={s: _stats(kept[s], steps) for s in SPLITS}, overall=_stats(everything, steps),
    )
    (out/"meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("data/v2"))
    parser.add_argument("--train-episodes", type=int, default=25000)
    parser.add_argument("--val-episodes", type=int, default=1000)
    parser.add_argument("--test-episodes", type=int, default=2500)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--max-objects", type=int, default=8)
    parser.add_argument("--workers", type=int, default=56)
    parser.add_argument("--shard-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()
    meta = generate(args.out, args.train_episodes, args.val_episodes, args.test_episodes, args.steps,
                    args.max_objects, args.workers, args.shard_size, args.seed)
    brief = {k: meta[k] for k in ("counts", "dropped", "transitions", "seconds", "workers", "episodes_per_second")}
    brief["overall"] = meta["overall"]
    brief["chaotic_fraction"] = {s: meta["splits"][s].get("chaotic_fraction") for s in SPLITS}
    print(json.dumps(brief, indent=2), flush=True)


if __name__ == "__main__":
    main()
