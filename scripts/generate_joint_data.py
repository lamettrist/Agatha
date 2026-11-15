"""Articulated rigid-body PyBullet episodes for the joint-aware world model (data/joints_v1 format).

TRACK J.  Every episode is a 4 s rollout (80 model steps of 0.05 s = 12 substeps at 240 Hz) of BOX links
connected by hinge / ball / prismatic / fixed joints (Featherstone multibodies built with p.createMultiBody),
mixed with free boxes, on the same floor + support slab world as agatha.physics.World (DT 1/240, 50 solver
iterations, linear/angular damping 0.04, Y-up API coordinates, Bullet Z-up via agatha.coordinates).
~15% of the episodes are plain box scenes from generate_contact_data.make_case padded to N=12.
CPU only, multiprocessing, deterministic per-episode seeds.

    .venv/bin/python scripts/generate_joint_data.py --out data/joints_v1 \
        --train-episodes 16000 --val-episodes 600 --test-episodes 2000 --workers 40

Layout: <out>/{train,val,test}/shard_XXXXX.npz + <out>/meta.json (see FORMAT below).

Conventions that the joints array relies on (all verified numerically per episode, see check_joints):
  * every link is a box whose body frame is its centre (COM); at joint coordinate 0 the child body frame has
    the SAME orientation as the parent body frame (the world frame when parent_slot = -1);
  * hinge: R_child = R_parent * Rot(axis, q) with axis given in the parent body frame (identical in the child
    frame); joint_pos is wrapped to (-pi, pi];
  * prismatic: child anchor = parent anchor + R_parent * axis * q, no relative rotation;
  * ball: only the anchors coincide; fixed: anchors coincide and no relative rotation;
  * parent_slot = -1: anchor_parent is a fixed WORLD point and axis is a world vector;
  * two bodies that share a joint never collide; every other pair (also inside one articulation) does.
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
from agatha.coordinates import to_bullet, from_bullet, quat_to_bullet, quat_from_bullet
import generate_contact_data as box_data
from generate_contact_data import (STEP, SUBSTEPS, PROPS_DIM, CONTEXT_DIM, REST_LIN, REST_ANG, CONTACT_DISTANCE,
                                   FLOOR_HALF, SPLITS, SPLIT_SEED_OFFSET, qmul, q_axis, q_yaw, rotmat, loguniform,
                                   episode_rng)

DT = 1/240
N_SLOTS = 12
J_SLOTS = 12
JOINT_DIM = 18
JOINT_TYPES = ("hinge", "ball", "prismatic", "fixed")
HINGE_FREE = 3.2            # recorded lower/upper = -+3.2 when a hinge has no limit
PRISMATIC_FREE = 10.        # recorded lower/upper = -+10 when a slider has no limit
SUPPORT_THICKNESS = .08
CLEARANCE = 5e-4            # resting things are created this far above the surface they sit on
MAX_INIT_PENETRATION = 3e-3  # a built scene with a deeper initial overlap is rebuilt (not counted as dropped)
MAX_ANCHOR_GAP = 1e-3       # ground truth must satisfy every joint to 1 mm on every frame ...
MAX_AXIS_ERROR = 2e-3       # ... and 2 mrad of hinge-axis / prismatic / fixed misalignment
MAX_COORD_ERROR = 2e-3      # recorded joint_pos vs the coordinate recomputed from the float32 body states
MAX_SPEED, MAX_SPIN = 40., 200.   # anything faster is a solver explosion
MAX_LIMIT_EXCESS_RAD, MAX_LIMIT_EXCESS_M = .3, .03   # Bullet's limit constraint yields under sustained load
#   (a light forearm pinned under a heavy torso creeps past its stop); such episodes are dropped
PREROLL_SUBSTEPS = 3        # 'kicks' are forces applied for this many substeps BEFORE frame 0
DAMPING_STABILITY = .25    # joint damping is capped at this * reduced inertia / dt (explicit damping stability)
IDENTITY = (0., 0., 0., 1.)

FORMAT = dict(
    box_data.FORMAT,
    states="float32 [E,T+1,N=12,13] pos3 (box centre), quat xyzw (sign-continuous), linvel3, angvel3 (world); Y-up metres",
    props="float32 [E,N,9] size_x,size_y,size_z,mass_kg,friction,is_static,restitution,rolling_friction,spinning_friction",
    joints="float32 [E,J=12,18] parent_slot (-1 = world), child_slot, one-hot [hinge, ball, prismatic, fixed], "
           "anchor_parent xyz (parent body frame; WORLD point when parent_slot=-1), anchor_child xyz (child body frame), "
           "axis xyz (unit, parent body frame = child body frame at coordinate 0; world when parent_slot=-1; zeros for "
           "ball/fixed), lower, upper (rad / m; -+3.2 or -+10 when unlimited; 0,0 for ball/fixed), joint_damping "
           "(N m s/rad, N s/m for prismatic)",
    joint_mask="bool [E,J] real joint rows",
    joint_pos="float32 [E,T+1,J] hinge angle (rad, wrapped to (-pi,pi]) / prismatic displacement (m); 0 for ball/fixed",
    group="int8 [E,N] articulation id of the body (0,1,..), -1 for free bodies and empty slots",
    joint_anchor_gap="float32 [E] diagnostic: max over frames and joints of |world anchor from parent - from child| (m), "
                     "recomputed from the stored float32 states",
    joint_axis_error="float32 [E] diagnostic: max hinge-axis / prismatic / fixed misalignment (rad)",
    joint_pos_error="float32 [E] diagnostic: max |joint_pos - coordinate recomputed from the states|",
    kind="<U24 [E] generator scenario ('plain_<kind>' = jointless scene from generate_contact_data.make_case)",
    tipped="bool [E,N] body +Y axis more than 35 deg from world up on any frame (agatha.physics.summarize semantics; "
           "trivially true for links that start that way)",
)

KIND_WEIGHTS = dict(plain=.15, laptop=.12, box_lid=.07, door=.07, drawer=.09, pendulum=.11, chain=.13,
                    robot_arm=.13, ragdoll=.13)
LIVE_FRACTION = .45

RANGES = dict(
    table="w 0.9-1.5, d 0.65-1.0, h 0.6-0.9; demo desk 1.2x0.8x0.75 in 40% of jointed episodes; never tilted in jointed episodes",
    gravity="9.81; uniform 2-15 in 5% of jointed episodes",
    link_mass_kg="density 200-2000 kg/m3 x volume clipped to 0.02-5 (laptop/doll/cabinet ranges below)",
    friction="0.5 for 35% of bodies else uniform 0.1-1.0",
    unobservable="as generate_contact_data: 45% 'live' episodes (restitution/rolling/spinning 0, support friction 0.5); "
                 "otherwise each of the five parameters randomized with p=0.7",
    free_boxes="0-3 per jointed episode (0-2 with a ragdoll), resting on the table, 15% airborne, sometimes thrown at the articulation",
    laptop="base 0.25-0.38 x 0.012-0.025 x 0.18-0.27 (0.6-1.6 kg), lid 5-12 mm (0.15-0.6 kg), hinge 0..105-140 deg, "
           "damping 0.002-0.3; open at a random angle; lid pushed, base pushed / at the edge / dropped / tossed / hit",
    box_lid="box 0.1-0.3 x 0.06-0.2 x 0.1-0.3, lid 8-25 mm, hinge 0..95-175 deg",
    door="static post (70%) or free-standing heavy box (30%) + panel 0.12-0.35 wide on a vertical hinge, limits +-1.2-2.8 rad "
         "or one-sided, pushed open 1-8 rad/s or hit by a thrown box",
    drawer="cabinet 0.15-0.4 x 0.08-0.25 x 0.15-0.35 (static 45%), horizontal slider 0..0.5-0.85 x drawer depth, "
           "damping 0.01-3 N s/m, pushed up to 2 m/s; 20% vertical 'lift' slider falling to its limit",
    pendulum="world anchor above the table, rod 0.15-0.45 m (+ fixed bob 40%, second link 25%), hinge 65% / ball 35%, "
             "released up to +-1.9 rad or swung at 2-6 rad/s, bob clears the table by 5-60 mm",
    chain="3-6 links 0.06-0.14 m, ball (45%) / parallel hinges (25%) / alternating hinges (30%); hanging from a world "
          "anchor, lying on the table / draped over an edge / pushed, dropped, tossed",
    robot_arm="static base on the table, 2-4 hinge links 0.1-0.28 m with vertical/horizontal axes, damping 0.001-0.3 "
              "(15% stiff 0.3-1.5), limits +-1.6-2.9 rad / none / asymmetric, optional fixed end pad; released from a random pose",
    ragdoll="10 boxes, height 0.44-0.78 m; ball or limited-hinge neck/shoulders/hips, limited hinge elbows/knees; "
            "dropped, tossed, standing and pushed, sitting on the table edge, lying at the edge and pushed",
    joint_damping="requested value capped at 0.25 x reduced inertia of the two joined boxes / dt (Bullet integrates joint "
                  "damping explicitly); the recorded joint_damping is the value actually simulated",
    kicks=f"an impulse on one link applied as a force over {PREROLL_SUBSTEPS} substeps before frame 0",
    dropped="non-finite states, |pos|>50, y<-0.5, |x|,|z|>10, speed>40 m/s, spin>200 rad/s, anchor gap > 1 mm, "
            "axis error > 2 mrad, joint_pos mismatch > 2e-3, hinge beyond a limit by > 0.3 rad / slider by > 0.03 m, "
            "no valid scene after 30 build attempts",
)


# ----------------------------------------------------------------------------- joint math (numpy reference)
def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """[...,4] xyzw -> [...,3,3]."""
    q = np.asarray(q, np.float64); q = q/np.linalg.norm(q, axis=-1, keepdims=True)
    x, y, z, w = np.moveaxis(q, -1, 0)
    return np.stack([np.stack([1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], -1),
                     np.stack([2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)], -1),
                     np.stack([2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)], -1)], -2)


def wrap_angle(a):
    return (np.asarray(a)+np.pi) % (2*np.pi)-np.pi


def check_joints(states: np.ndarray, joints: np.ndarray, joint_mask: np.ndarray) -> dict:
    """Numpy reference for the joint geometry of one episode.

    states [T+1,N,13], joints [J,18], joint_mask [J] -> dict of [T+1,J] float64 arrays:
      anchor_gap  metres between the world anchor computed from the parent and from the child (for a prismatic
                  joint only the component perpendicular to the axis),
      axis_error  rad: hinge = angle between the axis carried by the parent and by the child; prismatic / fixed =
                  relative rotation angle; ball = 0,
      coordinate  hinge angle (wrapped) / prismatic displacement / 0,
      anchor_parent_world, anchor_child_world [T+1,J,3].
    Rows with joint_mask False are zero."""
    states = np.asarray(states, np.float64); joints = np.asarray(joints, np.float64)
    frames = states.shape[0]; count = joints.shape[0]
    gap = np.zeros((frames, count)); axis_error = np.zeros((frames, count)); coordinate = np.zeros((frames, count))
    world_parent = np.zeros((frames, count, 3)); world_child = np.zeros((frames, count, 3))
    for j in range(count):
        if not joint_mask[j]: continue
        parent, child = int(round(joints[j, 0])), int(round(joints[j, 1])); kind = int(np.argmax(joints[j, 2:6]))
        anchor_parent, anchor_child, axis = joints[j, 6:9], joints[j, 9:12], joints[j, 12:15]
        rot_child = quat_to_matrix(states[:, child, 3:7]); pos_child = states[:, child, :3]
        if parent < 0:
            rot_parent = np.broadcast_to(np.eye(3), (frames, 3, 3)); pos_parent = np.zeros((frames, 3))
        else:
            rot_parent = quat_to_matrix(states[:, parent, 3:7]); pos_parent = states[:, parent, :3]
        a = pos_parent+rot_parent@anchor_parent; b = pos_child+rot_child@anchor_child
        world_parent[:, j], world_child[:, j] = a, b
        delta = b-a
        relative = np.swapaxes(rot_parent, -1, -2)@rot_child
        angle = np.arccos(np.clip((np.trace(relative, axis1=-2, axis2=-1)-1)/2, -1, 1))
        if kind == 0:      # hinge
            gap[:, j] = np.linalg.norm(delta, axis=-1)
            axis_parent = rot_parent@axis; axis_child = rot_child@axis
            axis_error[:, j] = np.arctan2(np.linalg.norm(np.cross(axis_parent, axis_child), axis=-1),
                                          np.sum(axis_parent*axis_child, -1))
            # twist of the relative rotation about the axis: tan(q) from the antisymmetric part and the trace
            sine = (np.stack([relative[:, 2, 1]-relative[:, 1, 2], relative[:, 0, 2]-relative[:, 2, 0],
                              relative[:, 1, 0]-relative[:, 0, 1]], -1)@axis)/2
            cosine = (np.trace(relative, axis1=-2, axis2=-1)-(relative@axis)@axis)/2
            coordinate[:, j] = np.arctan2(sine, cosine)
        elif kind == 2:    # prismatic
            axis_world = rot_parent@axis; along = np.sum(delta*axis_world, -1)
            coordinate[:, j] = along
            gap[:, j] = np.linalg.norm(delta-along[:, None]*axis_world, axis=-1); axis_error[:, j] = angle
        else:              # ball, fixed
            gap[:, j] = np.linalg.norm(delta, axis=-1)
            if kind == 3: axis_error[:, j] = angle
    return dict(anchor_gap=gap, axis_error=axis_error, coordinate=coordinate,
                anchor_parent_world=world_parent, anchor_child_world=world_child)


def coordinate_error(joint_pos: np.ndarray, coordinate: np.ndarray, joints: np.ndarray, joint_mask: np.ndarray) -> np.ndarray:
    """|recorded - recomputed| per frame and joint; hinges are compared modulo 2 pi."""
    error = np.abs(np.asarray(joint_pos, np.float64)-coordinate)
    hinge = (joints[:, 2] > .5) & joint_mask
    error[:, hinge] = np.abs(wrap_angle(joint_pos[:, hinge]-coordinate[:, hinge]))
    return error*joint_mask


# ----------------------------------------------------------------------------- scene description (Y-up)
def random_quat(rng):
    q = rng.normal(size=4); q /= np.linalg.norm(q)
    return tuple(float(v) for v in q)


def horizontal(theta):
    return np.array([math.cos(theta), 0., math.sin(theta)])


class Art:
    """One Bullet multibody described in API (Y-up) coordinates: a free box, or a tree of boxes with joints.

    Body 0 is the root (a free / static base) unless the first link is attached to the world (parent -1); then
    there is no root body and `t` is the world anchor.  World root pose = (t, q*root_q)."""

    def __init__(self, builder):
        self.b = builder
        self.bodies: list[dict] = []
        self.joints: list[dict] = []
        self.anchored = False
        self.t = np.zeros(3); self.q = IDENTITY; self.root_q = IDENTITY
        self.lin = np.zeros(3); self.ang = np.zeros(3)
        self.kicks: list[tuple[int, np.ndarray]] = []
        self.fixed_yaw = False

    def _body(self, size, mass, static=False, friction=None):
        b = self.b; rng = b.rng
        if friction is None: friction = .5 if rng.random() < .35 else float(rng.uniform(.1, 1.))
        body = dict(size=np.asarray(size, float), mass=float(mass), friction=float(friction), static=bool(static),
                    restitution=float(rng.uniform(0, .6)) if b.randomized["restitution"] else 0.,
                    rolling=float(rng.uniform(0, .02)) if b.randomized["rolling_friction"] else 0.,
                    spinning=float(rng.uniform(0, .02)) if b.randomized["spinning_friction"] else 0.)
        self.bodies.append(body)
        return len(self.bodies)-1

    def root(self, size, mass, static=False, friction=None):
        assert not self.bodies
        return self._body(size, mass, static, friction)

    def link(self, parent, size, mass, kind, anchor_parent, anchor_child, axis=(0, 0, 0), lower=None, upper=None,
             damping=0., q0=None, qd0=None, friction=None):
        """parent = body index or -1 (world; only as the very first body).  q0/qd0: float (hinge, prismatic) or
        quaternion / angular velocity vector in the child frame (ball)."""
        if parent < 0:
            assert not self.bodies; self.anchored = True
        child = self._body(size, mass, False, friction)
        axis = np.asarray(axis, float)
        if kind in ("hinge", "prismatic"):
            axis = axis/np.linalg.norm(axis); free = HINGE_FREE if kind == "hinge" else PRISMATIC_FREE
            lower = -free if lower is None else lower; upper = free if upper is None else upper
        else: axis = np.zeros(3); lower = upper = 0.
        damping = min(float(damping), self.damping_cap(parent, child, kind, np.asarray(anchor_parent, float),
                                                       np.asarray(anchor_child, float), axis))
        if q0 is None: q0 = IDENTITY if kind == "ball" else 0.
        if qd0 is None: qd0 = np.zeros(3) if kind == "ball" else 0.
        if kind in ("hinge", "prismatic"): q0 = float(np.clip(q0, lower, upper))   # always start inside the limits
        self.joints.append(dict(parent=parent, child=child, kind=kind, anchor_parent=np.asarray(anchor_parent, float),
                                anchor_child=np.asarray(anchor_child, float), axis=axis, lower=float(lower),
                                upper=float(upper), damping=float(damping), q0=q0, qd0=qd0))
        return child

    def damping_cap(self, parent, child, kind, anchor_parent, anchor_child, axis):
        """Bullet integrates joint damping explicitly: damping * dt / (effective inertia) must stay well below 1 or
        light links blow up.  Cap = DAMPING_STABILITY * (reduced inertia of the two joined boxes) / dt."""
        def inertia(index, anchor):
            if index < 0 or self.bodies[index]["static"]: return math.inf
            body = self.bodies[index]; m = body["mass"]; sx, sy, sz = body["size"]
            if kind == "prismatic": return m
            principal = m/12*np.array([sy*sy+sz*sz, sx*sx+sz*sz, sx*sx+sy*sy])
            if kind == "hinge": return float(principal@(axis*axis)+m*np.sum((anchor-(anchor@axis)*axis)**2))
            return float(principal.min())
        if kind == "fixed": return 0.
        a, b = inertia(parent, anchor_parent), inertia(child, anchor_child)
        reduced = b if math.isinf(a) else a*b/(a+b)
        return DAMPING_STABILITY*reduced/DT

    def limited(self, joint):
        if joint["kind"] == "hinge": return joint["lower"] > -HINGE_FREE+1e-6 or joint["upper"] < HINGE_FREE-1e-6
        if joint["kind"] == "prismatic": return joint["lower"] > -PRISMATIC_FREE+1e-6 or joint["upper"] < PRISMATIC_FREE-1e-6
        return False

    def fk(self):
        """World (pos, quat) of every body for the current placement and initial joint coordinates."""
        n = len(self.bodies); pos = [None]*n; quat = [None]*n
        if not self.anchored:
            pos[0] = self.t.copy(); quat[0] = qmul(self.q, self.root_q)
        for j in self.joints:
            if j["parent"] < 0: parent_pos, parent_q = self.t, IDENTITY
            else: parent_pos, parent_q = pos[j["parent"]], quat[j["parent"]]
            rel = q_axis(j["axis"], j["q0"]) if j["kind"] == "hinge" else j["q0"] if j["kind"] == "ball" else IDENTITY
            shift = j["axis"]*j["q0"] if j["kind"] == "prismatic" else 0.
            q = qmul(parent_q, rel); quat[j["child"]] = q
            pos[j["child"]] = parent_pos+rotmat(parent_q)@(j["anchor_parent"]+shift)-rotmat(q)@j["anchor_child"]
        return pos, quat

    def boxes(self):
        """World AABBs [(lo, hi)] of the bodies."""
        pos, quat = self.fk(); out = []
        for body, x, q in zip(self.bodies, pos, quat):
            ext = np.abs(rotmat(q))@body["size"]/2
            out.append((x-ext, x+ext))
        return out

    def bounds(self):
        boxes = self.boxes()
        return np.min([b[0] for b in boxes], 0), np.max([b[1] for b in boxes], 0)

    def root_bounds(self):
        return self.boxes()[0]


class JB:
    """Episode builder: table, unobservable parameters, articulations and free boxes."""

    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.live = bool(rng.random() < LIVE_FRACTION)
        if rng.random() < .4: self.w, self.d, self.h = 1.2, .8, .75
        else: self.w, self.d, self.h = float(rng.uniform(.9, 1.5)), float(rng.uniform(.65, 1.)), float(rng.uniform(.6, .9))
        names = ("restitution", "rolling_friction", "spinning_friction", "support_friction", "support_restitution")
        self.randomized = {n: (not self.live) and rng.random() < .7 for n in names}
        if not self.live and not any(self.randomized.values()):
            self.randomized[names[int(rng.integers(len(names)))]] = True
        self.support_friction = float(rng.uniform(.1, 1.)) if self.randomized["support_friction"] else .5
        self.support_restitution = float(rng.uniform(0, .6)) if self.randomized["support_restitution"] else 0.
        self.gravity = float(rng.uniform(2, 15)) if rng.random() < .05 else 9.81
        self.arts: list[Art] = []
        self.ok = True

    # -- sampling helpers
    def U(self, lo, hi): return float(self.rng.uniform(lo, hi))

    def link_mass(self, size, density=None, lo=.02, hi=5.):
        density = loguniform(self.rng, 200, 2000) if density is None else density
        return float(np.clip(float(np.prod(size))*density, lo, hi))

    def free_size(self):
        rng = self.rng; style = str(rng.choice(["any", "flat", "cube", "tall"], p=[.45, .2, .2, .15]))
        if style == "flat": return np.array([rng.uniform(.08, .3), rng.uniform(.02, .06), rng.uniform(.08, .3)])
        if style == "tall": return np.array([rng.uniform(.03, .09), rng.uniform(.1, .3), rng.uniform(.03, .09)])
        if style == "cube": return np.clip(rng.uniform(.04, .18)*rng.uniform(.8, 1.2, 3), .02, .35)
        while True:
            s = np.array([loguniform(rng, .03, .3), loguniform(rng, .02, .3), loguniform(rng, .03, .3)])
            if s[1] <= 3.5*min(s[0], s[2]): return s

    def free_mass(self, size):
        rng = self.rng
        if rng.random() < .5: return float(np.clip(float(np.prod(size))*loguniform(rng, 150, 2500), .05, 3.))
        return loguniform(rng, .05, 3.)

    def box_art(self, size=None, mass=None):
        art = Art(self); size = self.free_size() if size is None else np.asarray(size, float)
        art.root(size, self.free_mass(size) if mass is None else mass)
        return art

    # -- placement
    def slots_used(self):
        return sum(len(a.bodies) for a in self.arts)

    def free(self, art, margin=.004):
        for lo, hi in art.boxes():
            for other in self.arts:
                for lo2, hi2 in other.boxes():
                    if np.all(lo < hi2+margin) and np.all(lo2 < hi+margin): return False
        return True

    def commit(self, art):
        self.arts.append(art); return True

    def place(self, art, gap=0., inset=.02, tries=60, yaw=None, near=None, level=None, tilt=IDENTITY):
        """Random yaw and a free spot with the whole footprint on the table; the lowest point (or the root's lowest
        point when level='root') ends `gap` above the surface.  near=(x, z, radius) restricts the search."""
        rng = self.rng
        for _ in range(tries):
            if not art.anchored and not art.fixed_yaw:
                angle = float(rng.uniform(-math.pi, math.pi)) if yaw is None else yaw
                art.q = qmul(q_yaw(angle), tilt)
            art.t = np.zeros(3); lo, hi = art.bounds()
            bottom = art.root_bounds()[0][1] if level == "root" else lo[1]
            t = np.zeros(3); t[1] = self.h-bottom+CLEARANCE+gap
            for axis, half in ((0, self.w/2), (2, self.d/2)):
                a, b = -half+inset-lo[axis], half-inset-hi[axis]
                if a > b: a = b = (a+b)/2
                if near is not None:
                    centre = near[0 if axis == 0 else 1]
                    a2, b2 = max(a, centre-near[2]-(lo[axis]+hi[axis])/2), min(b, centre+near[2]-(lo[axis]+hi[axis])/2)
                    if a2 <= b2: a, b = a2, b2
                t[axis] = rng.uniform(a, b) if b > a else a
            art.t = t
            if self.free(art): return self.commit(art)
        return False

    def edge(self):
        """Random table edge: (outward normal, distance of the edge from the centre, tangent half length)."""
        side = int(self.rng.integers(4))
        normal = np.array([(1., 0, 0), (-1., 0, 0), (0, 0, 1.), (0, 0, -1.)][side])
        return normal, (self.w/2 if side < 2 else self.d/2), (self.d/2 if side < 2 else self.w/2)

    def place_edge(self, art, overhang, forward=None, gap=0., tries=40, level=None):
        """Put the articulation at a table edge with `overhang` (fraction of its extent along the outward normal)
        sticking out.  forward = local horizontal vector to align with the outward normal (else random yaw)."""
        rng = self.rng
        for _ in range(tries):
            normal, reach, half = self.edge(); tangent = np.array([normal[2], 0., normal[0]])
            if forward is None: angle = float(rng.uniform(-math.pi, math.pi))
            else: angle = math.atan2(normal[0], normal[2])-math.atan2(forward[0], forward[2])
            art.q = q_yaw(angle); art.t = np.zeros(3); lo, hi = art.bounds()
            bottom = art.root_bounds()[0][1] if level == "root" else lo[1]
            far = max(float(lo@normal), float(hi@normal)); near = min(float(lo@normal), float(hi@normal))
            lateral = sorted((float(lo@tangent), float(hi@tangent)))
            a, b = -half+.02-lateral[0], half-.02-lateral[1]
            along = rng.uniform(a, b) if b > a else (a+b)/2
            t = normal*(reach+overhang*(far-near)-far)+tangent*along
            t[1] = self.h-bottom+CLEARANCE+gap
            art.t = t
            if self.free(art):
                art.edge_normal = normal
                return self.commit(art)
        return False

    def stage(self, art, mode):
        """Shared ways of starting a free-floating articulation: rest / push / edge / drop / toss."""
        rng = self.rng
        if mode == "rest": return self.place(art)
        if mode == "push":
            if not self.place(art): return False
            art.lin = horizontal(rng.uniform(-math.pi, math.pi))*float(rng.uniform(.3, 2.5))
            if rng.random() < .25: art.ang = np.array([0., rng.uniform(-4, 4), 0.])
            return True
        if mode == "nudge":      # round 2: slow slide that stops on the table (0.05-0.6 m/s), sometimes a slow yaw
            if not self.place(art, inset=.06): return False
            art.lin = horizontal(rng.uniform(-math.pi, math.pi))*float(rng.uniform(.05, .6))
            if rng.random() < .3: art.ang = np.array([0., rng.uniform(-1.5, 1.5), 0.])
            return True
        if mode == "edge_safe":  # round 2: at / over an edge with the COM well on the table; at rest, or pushed slowly along / away from the edge
            if not self.place_edge(art, float(rng.uniform(-.03, .3))): return False
            r = rng.random(); n = art.edge_normal; tangent = np.array([n[2], 0., n[0]])
            if r < .35: art.lin = tangent*float(rng.uniform(-.6, .6))
            elif r < .6: art.lin = -n*float(rng.uniform(.05, .8))+tangent*float(rng.uniform(-.3, .3))
            elif r < .75: art.lin = n*float(rng.uniform(.02, .2))
            return True
        if mode == "edge":
            if not self.place_edge(art, float(rng.uniform(-.05, .42))): return False
            if rng.random() < .7:
                theta = float(rng.normal(0, .35)); n = art.edge_normal
                direction = np.array([n[0]*math.cos(theta)-n[2]*math.sin(theta), 0., n[0]*math.sin(theta)+n[2]*math.cos(theta)])
                art.lin = direction*float(rng.uniform(.2, 1.8))
            return True
        if mode == "drop":
            style = rng.random()
            tilt = IDENTITY if style < .4 else q_axis(horizontal(rng.uniform(-math.pi, math.pi)), rng.uniform(.05, .6)) if style < .7 else random_quat(rng)
            if not self.place(art, gap=float(rng.uniform(.03, .35)), tilt=tilt): return False
            if rng.random() < .2:  # partly or fully beyond an edge
                normal, reach, _ = self.edge(); art.t = art.t+normal*float(rng.uniform(.1, .5))
                if not self.free(art): return False
            if rng.random() < .4: art.lin = np.array([rng.uniform(-1, 1), rng.uniform(-1, .5), rng.uniform(-1, 1)])
            if rng.random() < .4: art.ang = rng.normal(size=3)*float(rng.uniform(0, 3))
            return True
        if mode == "toss":
            tilt = IDENTITY if rng.random() < .5 else q_axis(horizontal(rng.uniform(-math.pi, math.pi)), rng.uniform(.05, .5))
            if not self.place(art, gap=float(rng.uniform(0, .1)), tilt=tilt): return False
            art.lin = np.array([rng.uniform(-1.5, 1.5), rng.uniform(1., 3.5), rng.uniform(-1.5, 1.5)])
            art.ang = rng.normal(size=3)*float(rng.uniform(0, 4))
            return True
        raise ValueError(mode)

    def throw_at(self, target: np.ndarray):
        """A free box launched so that it reaches `target` (world point) on a ballistic arc."""
        rng = self.rng
        if self.slots_used() >= N_SLOTS: return False
        for _ in range(20):
            art = self.box_art(size=np.clip(self.free_size(), .03, .16))
            direction = horizontal(rng.uniform(-math.pi, math.pi)); distance = float(rng.uniform(.2, .45))
            ext = np.abs(rotmat(art.q))@art.bodies[0]["size"]/2
            start = target+direction*distance
            start[1] = max(target[1], self.h+ext.max()+.02)+float(rng.uniform(0, .08))
            speed = float(rng.uniform(1., 4.)); flight = distance/speed
            art.q = q_yaw(rng.uniform(-math.pi, math.pi)); art.t = start
            art.lin = -direction*speed+np.array([0., self.gravity*flight/2+(target[1]-start[1])/flight, 0.])
            if rng.random() < .3: art.ang = rng.normal(size=3)*2.
            if self.free(art): return self.commit(art)
        return False

    def add_free(self, count, near=None):
        rng = self.rng
        for _ in range(count):
            if self.slots_used() >= N_SLOTS: return
            art = self.box_art()
            if rng.random() < .15:
                tilt = IDENTITY if rng.random() < .5 else random_quat(rng)
                if self.place(art, gap=float(rng.uniform(.02, .3)), near=near, tilt=tilt) and rng.random() < .3:
                    art.lin = np.array([rng.uniform(-1, 1), rng.uniform(-.5, .5), rng.uniform(-1, 1)])
            else:
                if self.place(art, near=near) and rng.random() < .12:
                    art.lin = horizontal(rng.uniform(-math.pi, math.pi))*float(rng.uniform(.2, 1.5))

    def centre(self, art):
        lo, hi = art.bounds(); return (lo+hi)/2


# ----------------------------------------------------------------------------- scenarios
def _lidded(b: JB, base_size, lid_thickness, base_mass, lid_mass, upper, modes, weights, calm=False):
    """Base box + a lid hinged along the back top edge (laptop, box with a lid)."""
    rng = b.rng; w, tb, d = base_size; a = Art(b)
    base = a.root((w, tb, d), base_mass)
    mode = str(rng.choice(modes, p=np.asarray(weights)/np.sum(weights)))
    r = rng.random()
    q0 = 0. if r < .15 else float(np.clip(rng.normal(math.pi/2, .15), 0, upper)) if r < .35 else float(rng.uniform(0, upper))
    qd0 = 0.
    if mode == "lid_push" or rng.random() < .15: qd0 = float(rng.uniform(-6, 6))
    if calm:   # round 2: lids at their stops (closed 40% / fully open 20%), released slowly near a stop, or let go anywhere
        r = rng.random()
        q0 = 0. if r < .4 else upper if r < .6 else float(rng.uniform(0, .25)) if r < .7 else float(rng.uniform(max(upper-.3, 0), upper)) if r < .8 else float(rng.uniform(0, upper))
        qd0 = float(rng.uniform(-1.5, 1.5)) if mode == "lid_slow" else 0.
    a.link(base, (w*b.U(.97, 1.), lid_thickness, d*b.U(.97, 1.)), lid_mass, "hinge", (0, tb/2, -d/2), (0, -lid_thickness/2, -d/2),
           axis=(-1, 0, 0), lower=0., upper=upper, damping=loguniform(rng, .002, .3), q0=q0, qd0=qd0)
    if mode in ("lid_push", "hit", "lid_slow"): ok = b.stage(a, "rest")
    else: ok = b.stage(a, mode)
    if not ok: b.ok = False; return None
    if mode == "hit": b.throw_at(b.centre(a))
    return a

def s_laptop(b: JB):
    rng = b.rng
    a = _lidded(b, (b.U(.25, .38), b.U(.012, .025), b.U(.18, .27)), b.U(.005, .012), b.U(.6, 1.6), b.U(.15, .6),
                math.radians(b.U(105, 140)), ["rest", "lid_push", "push", "edge", "drop", "toss", "hit"],
                [.18, .17, .2, .2, .1, .05, .1])
    if a is not None: b.add_free(int(rng.integers(0, 4)))

def s_lid_calm(b: JB):
    """Round 2: laptops (60%) / lidded boxes that stay on the table: resting, nudged, at an edge without falling, lid at
    a stop or closing slowly (the round-1 model slid closed laptops off the table and let resting lids creep)."""
    rng = b.rng; modes = ["rest", "nudge", "edge_safe", "lid_slow"]; weights = [.2, .3, .3, .2]
    if rng.random() < .6:
        a = _lidded(b, (b.U(.25, .38), b.U(.012, .025), b.U(.18, .27)), b.U(.005, .012), b.U(.6, 1.6), b.U(.15, .6),
                    math.radians(b.U(105, 140)), modes, weights, calm=True)
    else:
        size = (b.U(.1, .3), b.U(.06, .2), b.U(.1, .3)); thickness = b.U(.008, .025)
        a = _lidded(b, size, thickness, b.link_mass(size, lo=.15, hi=3.), b.link_mass((size[0], thickness, size[2]), lo=.03, hi=1.),
                    math.radians(b.U(95, 175)), modes, weights, calm=True)
    if a is not None: b.add_free(int(rng.integers(0, 3)))

def s_box_lid(b: JB):
    rng = b.rng; size = (b.U(.1, .3), b.U(.06, .2), b.U(.1, .3)); thickness = b.U(.008, .025)
    a = _lidded(b, size, thickness, b.link_mass(size, lo=.15, hi=3.), b.link_mass((size[0], thickness, size[2]), lo=.03, hi=1.),
                math.radians(b.U(95, 175)), ["rest", "lid_push", "push", "edge", "drop", "toss", "hit"],
                [.2, .25, .15, .12, .1, .08, .1])
    if a is not None: b.add_free(int(rng.integers(0, 4)))

def s_door(b: JB):
    rng = b.rng; a = Art(b); static = rng.random() < .7
    if static: post = np.array([b.U(.03, .06), b.U(.15, .4), b.U(.03, .06)])
    else: post = np.array([b.U(.1, .25), b.U(.15, .35), b.U(.12, .3)])
    base = a.root(post, b.link_mass(post, lo=.8, hi=5.) if not static else 2., static=static)
    width = b.U(.12, .35); height = post[1]*b.U(.75, 1.); thickness = b.U(.01, .03); lift = b.U(.004, .02)
    if static: anchor_parent = (post[0]/2+.003, (height-post[1])/2+lift, 0.)
    else: anchor_parent = (post[0]/2+.003, (height-post[1])/2+lift, post[2]/2-thickness/2)  # door on the cabinet front corner
    r = rng.random(); swing = b.U(1.2, 2.8)
    lower, upper = (-swing, swing) if r < .5 else (0., swing) if r < .8 else (-HINGE_FREE, HINGE_FREE)
    q0 = float(rng.uniform(max(lower, -2.8), min(upper, 2.8))) if rng.random() < .7 else 0.
    mode = str(rng.choice(["swing", "hit", "rest", "base"] if not static else ["swing", "hit", "rest"],
                          p=[.4, .25, .1, .25] if not static else [.6, .3, .1]))
    qd0 = float(rng.choice([-1, 1])*rng.uniform(1, 8)) if mode == "swing" else 0.
    panel = (width, height, thickness)
    a.link(base, panel, b.link_mass(panel, lo=.03, hi=2.), "hinge", anchor_parent, (-width/2, 0, 0), axis=(0, 1, 0),
           lower=lower, upper=upper, damping=loguniform(rng, 5e-4, .05), q0=q0, qd0=qd0)
    if mode == "base": ok = b.stage(a, str(rng.choice(["push", "edge", "toss"], p=[.5, .3, .2])))
    else: ok = b.place(a, level="root")
    if not ok: b.ok = False; return
    centre = b.centre(a)
    if mode == "hit": b.throw_at(centre)
    b.add_free(int(rng.integers(0, 4)), near=(centre[0], centre[2], width+.15) if rng.random() < .7 else None)

def s_drawer(b: JB):
    rng = b.rng; a = Art(b); static = rng.random() < .45
    cabinet = np.array([b.U(.15, .4), b.U(.08, .25), b.U(.15, .35)])
    base = a.root(cabinet, 3. if static else b.link_mass(cabinet, lo=.5, hi=5.), static=static)
    if rng.random() < .2:   # vertical 'lift' slider: a platform above the cabinet that drops to its lower limit
        size = np.array([cabinet[0]*b.U(.5, 1.), b.U(.015, .05), cabinet[2]*b.U(.5, 1.)])
        upper = b.U(.05, .25); q0 = b.U(0, upper); qd0 = b.U(-1, 2) if rng.random() < .4 else 0.
        a.link(base, size, b.link_mass(size, lo=.03, hi=2.), "prismatic", (0, cabinet[1]/2+.002, 0), (0, -size[1]/2, 0), axis=(0, 1, 0),
               lower=0., upper=upper, damping=loguniform(rng, .01, 3.), q0=q0, qd0=qd0)
    else:
        size = np.array([cabinet[0]*b.U(.75, .92), cabinet[1]*b.U(.5, .85), cabinet[2]*b.U(.8, .95)])
        proud = (cabinet[2]-size[2])/2+b.U(.005, .02); y = (cabinet[1]-size[1])/2*b.U(-.8, .8)
        upper = size[2]*b.U(.5, .85)
        q0 = 0. if rng.random() < .25 else b.U(0, upper)
        qd0 = b.U(-2, 2) if rng.random() < .65 else 0.
        a.link(base, size, b.link_mass(size, lo=.05, hi=2.), "prismatic", (0, y, proud), (0, 0, 0), axis=(0, 0, 1),
               lower=0., upper=upper, damping=loguniform(rng, .01, 3.), q0=q0, qd0=qd0)
    if static: ok = b.place(a, level="root")
    else: ok = b.stage(a, str(rng.choice(["rest", "push", "edge", "drop", "toss"], p=[.3, .3, .2, .12, .08])))
    if not ok: b.ok = False; return
    if rng.random() < .2: b.throw_at(b.centre(a))
    b.add_free(int(rng.integers(0, 4)))

def s_pendulum(b: JB):
    rng = b.rng
    for _ in range(20):
        a = Art(b); ball = rng.random() < .35
        thick = b.U(.015, .04); length = b.U(.15, .45); rod = (thick, length, thick*b.U(.8, 1.6))
        swing_axis = horizontal(rng.uniform(-math.pi, math.pi))
        amplitude = b.U(.3, 1.9)*float(rng.choice([-1, 1])); spin = 0.
        if rng.random() < .15: amplitude = b.U(-.1, .1); spin = b.U(2, 6)*float(rng.choice([-1, 1]))
        damping = loguniform(rng, 1e-4, .05); limited = (not ball) and rng.random() < .2
        limit = b.U(max(abs(amplitude), .5), 2.2)
        if ball:
            q0 = q_axis(swing_axis, amplitude); qd0 = swing_axis*spin+(rng.normal(size=3)*b.U(0, 2) if rng.random() < .5 else 0)
            # ball joint velocities are expressed in the child frame
            qd0 = rotmat(q0).T@qd0
        else: q0, qd0 = amplitude, spin
        first = a.link(-1, rod, b.link_mass(rod), "ball" if ball else "hinge", (0, 0, 0), (0, length/2, 0), axis=swing_axis,
                       lower=-limit if limited else -HINGE_FREE, upper=limit if limited else HINGE_FREE, damping=damping, q0=q0, qd0=qd0)
        total = length; r = rng.random()
        if r < .4:      # rigid bob
            s = b.U(.04, .1); bob = (s, s*b.U(.7, 1.2), s)
            a.link(first, bob, b.link_mass(bob), "fixed", (0, -length/2, 0), (0, bob[1]/2, 0)); total += bob[1]
        elif r < .65:   # double pendulum
            second = b.U(.1, .3); rod2 = (thick, second, rod[2]); kind = "ball" if rng.random() < .4 else "hinge"
            a.link(first, rod2, b.link_mass(rod2), kind, (0, -length/2, 0), (0, second/2, 0),
                   axis=swing_axis if rng.random() < .7 else horizontal(rng.uniform(-math.pi, math.pi)),
                   lower=-HINGE_FREE, upper=HINGE_FREE, damping=loguniform(rng, 1e-4, .05),
                   q0=(float(rng.uniform(-1, 1)) if kind == "hinge" else q_axis(horizontal(rng.uniform(-math.pi, math.pi)), rng.uniform(0, 1))))
            total += second
        a.t = np.array([rng.uniform(-b.w/2+.1, b.w/2-.1), b.h+total+b.U(.005, .06)+max(rod[0], rod[2]),
                        rng.uniform(-b.d/2+.1, b.d/2-.1)])
        if a.bounds()[0][1] < b.h+.003: continue
        b.commit(a)
        direction = np.cross(swing_axis, [0, 1., 0])
        for _ in range(int(rng.integers(0, 4))):
            if rng.random() < .75:
                spot = a.t+direction*float(rng.uniform(-.7, .7))*total
                b.add_free(1, near=(spot[0], spot[2], .05))
            else: b.add_free(1)
        return
    b.ok = False

def s_chain(b: JB, calm=False):
    rng = b.rng; n = int(rng.integers(3, 7)); a = Art(b)
    style = str(rng.choice(["ball", "hinge", "alternate"], p=[.45, .25, .3]))
    mode = str(rng.choice(["hang", "drape", "lie_push", "drop", "toss"], p=[.3, .22, .13, .22, .13]))
    if calm: mode = str(rng.choice(["drape", "lie_push", "drop"], p=[.4, .4, .2]))   # round 2: slow settling on the table
    length = b.U(.06, .14); thick = b.U(.018, .04); density = loguniform(rng, 300, 3000)
    limited = style != "ball" and rng.random() < .5; limit = b.U(1., 2.6)
    damping = loguniform(rng, 1e-4, .02)
    sizes = [np.array([thick*b.U(.9, 1.1), length*b.U(.85, 1.15), thick*b.U(.9, 1.1)]) for _ in range(n)]
    straight = mode in ("drape", "lie_push")
    def joint_args(i):
        kind = "ball" if style == "ball" else "hinge"
        axis = (1, 0, 0) if style != "alternate" or i % 2 == 0 else (0, 0, 1)
        if straight: q0 = IDENTITY if kind == "ball" else 0.
        elif kind == "ball": q0 = q_axis(rng.normal(size=3), rng.uniform(0, 1.))
        else: q0 = float(rng.uniform(-1, 1)*min(1.2, limit))
        return dict(kind=kind, axis=axis, lower=-limit if limited else -HINGE_FREE, upper=limit if limited else HINGE_FREE,
                    damping=damping, q0=q0)
    total = float(sum(s[1] for s in sizes))
    if mode == "hang":
        args = joint_args(0); height = total*b.U(.35, 1.15)
        lean = math.acos(min(1., max(0., (height-.02)/total)))+.05
        angle = float(rng.choice([-1, 1]))*b.U(min(lean, 1.6), max(1.7, lean)) if height < total+.03 else b.U(-1.7, 1.7)
        swing_axis = horizontal(rng.uniform(-math.pi, math.pi)) if args["kind"] == "ball" else np.asarray(args["axis"], float)
        args["q0"] = q_axis(swing_axis, angle) if args["kind"] == "ball" else angle
        previous = a.link(-1, sizes[0], b.link_mass(sizes[0], density), anchor_parent=(0, 0, 0), anchor_child=(0, sizes[0][1]/2, 0), **args)
        for i in range(1, n):
            args = joint_args(i)
            args["q0"] = (q_axis(rng.normal(size=3), rng.uniform(0, .25)) if args["kind"] == "ball" else float(rng.uniform(-.25, .25)))
            previous = a.link(previous, sizes[i], b.link_mass(sizes[i], density), anchor_parent=(0, -sizes[i-1][1]/2, 0),
                              anchor_child=(0, sizes[i][1]/2, 0), **args)
        for _ in range(20):
            a.t = np.array([rng.uniform(-b.w/2+.05, b.w/2-.05), b.h+height, rng.uniform(-b.d/2+.05, b.d/2-.05)])
            if rng.random() < .25:
                normal, reach, _ = b.edge(); a.t = np.where(normal != 0, normal*(reach+b.U(.02, .2)), a.t)
            lo, hi = a.bounds()
            inside = abs(a.t[0]) < b.w/2 and abs(a.t[2]) < b.d/2
            if lo[1] > b.h+.004 or (not inside and lo[1] > .05 and b._clear_of_table(a)): break
        else: b.ok = False; return
        b.commit(a)
    else:
        previous = a.root(sizes[0], b.link_mass(sizes[0], density))
        for i in range(1, n):
            previous = a.link(previous, sizes[i], b.link_mass(sizes[i], density), anchor_parent=(0, -sizes[i-1][1]/2, 0),
                              anchor_child=(0, sizes[i][1]/2, 0), **joint_args(i))
        if straight:
            a.root_q = q_axis((0, 0, 1), math.pi/2)   # the chain extends along local +x, lying on a side face
            if mode == "drape":
                ok = b.place_edge(a, b.U(.05, .45) if calm else b.U(.3, .78), forward=(1, 0, 0) if rng.random() < .5 else (-1, 0, 0))
                if ok and rng.random() < .3: a.lin = a.edge_normal*(b.U(-.4, .15) if calm else b.U(.2, 1.2))
            else:
                ok = b.place(a)
                if ok:
                    a.lin = horizontal(rng.uniform(-math.pi, math.pi))*b.U(.3, 2.5)
                    if rng.random() < .5: a.ang = np.array([0., rng.uniform(-6, 6), 0.])
                    if calm: a.lin, a.ang = a.lin*b.U(0, .25), a.ang*b.U(0, .25)
        elif calm:   # folded chain let go just above the table: link-link contacts, then rest
            ok = b.place(a, gap=b.U(.0, .04), inset=.08, tilt=q_axis(horizontal(rng.uniform(-math.pi, math.pi)), math.pi/2+b.U(-.3, .3)))
        else: ok = b.stage(a, mode)
        if not ok: b.ok = False; return
        if rng.random() < .25 and not calm:
            a.kicks.append((int(rng.integers(n)), a.bodies[0]["mass"]*rng.normal(size=3)*b.U(.5, 2.)))
    b.add_free(int(rng.integers(0, 4)))

def _clear_of_table(self, art):
    """True when no body AABB of the articulation intersects the support slab."""
    for lo, hi in art.boxes():
        if lo[0] < self.w/2 and hi[0] > -self.w/2 and lo[2] < self.d/2 and hi[2] > -self.d/2 and lo[1] < self.h+.004 and hi[1] > self.h-SUPPORT_THICKNESS:
            return False
    return True
JB._clear_of_table = _clear_of_table

def s_robot_arm(b: JB, calm=False):
    rng = b.rng; n = int(rng.integers(2, 5))
    base_size = np.array([b.U(.08, .16), b.U(.03, .1), b.U(.08, .16)])
    stiff = rng.random() < .15; density = loguniform(rng, 300, 2500)
    thick = b.U(.03, .06); lengths = [b.U(.1, .28)*(.85**i) for i in range(n)]
    axes = []; previous = None
    for i in range(n):
        r = rng.random()
        if i == 0: axis = (0, 1, 0) if r < .35 else (1, 0, 0) if r < .7 else (0, 0, 1)
        elif previous is not None and previous != (0, 1, 0) and r < .65: axis = previous
        elif r < .9: axis = (1, 0, 0) if rng.random() < .5 else (0, 0, 1)
        else: axis = (0, 1, 0)
        axes.append(axis); previous = axis
    limits = []
    for i in range(n):
        r = rng.random()
        if r < .5: s = b.U(1.6, 2.9); limits.append((-s, s))
        elif r < .8: limits.append((-HINGE_FREE, HINGE_FREE))
        else: limits.append((-b.U(.2, 1.5), b.U(1.5, 2.9)))
    dampings = [b.U(.3, 1.5) if stiff else loguniform(rng, .001, .3) for _ in range(n)]
    if calm:   # round 2: arms that settle slowly (stiff joints 50%) or start folded onto a limit / hanging down, and then REST
        stiff = rng.random() < .5; dampings = [b.U(.3, 1.5) if stiff else loguniform(rng, .02, .3) for _ in range(n)]
        calm_style = str(rng.choice(["limit", "small", "any"], p=[.35, .3, .35]))
    pad = n <= 3 and rng.random() < .3
    pad_size = np.array([b.U(.05, .1), b.U(.02, .04), b.U(.05, .1)])
    for _ in range(40):
        a = Art(b); parent = a.root(base_size, 2., static=True); anchor = (0, base_size[1]/2, 0)
        for i in range(n):
            size = (thick*(.9**i), lengths[i], thick*(.9**i))
            lo, hi = max(limits[i][0], -3.), min(limits[i][1], 3.)
            q0 = float(rng.uniform(lo, hi)) if rng.random() < .9 else float(rng.uniform(-.3, .3))
            qd0 = float(rng.uniform(-3, 3)) if rng.random() < .15 else 0.
            if calm:
                qd0 = 0.
                if calm_style == "limit" and limits[i][1] < HINGE_FREE-1e-6: q0 = float(rng.choice([lo, hi]))
                if calm_style == "limit" and limits[i][1] < HINGE_FREE-1e-6: q0 = float(np.clip(q0+rng.uniform(-.15, .15), lo, hi))
                if calm_style == "small": q0 = float(np.clip(rng.normal(0, .25), lo, hi))
            parent = a.link(parent, size, b.link_mass(size, density, lo=.05, hi=3.), "hinge", anchor, (0, -lengths[i]/2, 0), axis=axes[i],
                            lower=limits[i][0], upper=limits[i][1], damping=dampings[i], q0=q0, qd0=qd0)
            anchor = (0, lengths[i]/2, 0)
        if pad: a.link(parent, pad_size, b.link_mass(pad_size, density, lo=.03, hi=1.), "fixed", anchor, (0, -pad_size[1]/2, 0))
        boxes = a.boxes()
        if min(lo[1] for lo, _ in boxes[1:]) > boxes[0][0][1]+.006: break
    else: b.ok = False; return
    if not b.place(a, level="root", inset=.05): b.ok = False; return
    reach = sum(lengths)
    for _ in range(int(rng.integers(0, 4))):
        b.add_free(1, near=(a.t[0], a.t[2], reach+.05) if rng.random() < .8 else None)
    if rng.random() < .12 and not calm: b.throw_at(b.centre(a))

def s_ragdoll(b: JB, calm=False):
    rng = b.rng; s = b.U(.45, .8); a = Art(b); density = loguniform(rng, 250, 1100)
    mode = str(rng.choice(["drop", "toss", "sit", "stand", "lie"], p=[.3, .15, .2, .15, .2]))
    if calm: mode = str(rng.choice(["sit_in", "lie", "stand", "drop_low"], p=[.3, .3, .2, .2]))   # round 2: at rest / slowly toppling
    loose = mode in ("drop", "toss")
    def mass(size): return b.link_mass(size, density*b.U(.8, 1.2), lo=.03, hi=5.)
    def damp(): return loguniform(rng, .001, .05)
    def hinge_q(lower, upper, sit=None):
        if mode == "sit" and sit is not None: return float(sit+rng.normal(0, .08))
        if mode == "sit_in" and sit is not None: return float(np.clip((sit if sit < 0 else 0.)+rng.normal(0, .08), lower, upper))  # legs straight ahead on the table
        if loose: return float(rng.uniform(lower, upper)*.6)
        return float(np.clip(rng.normal(0, .08), lower, upper))
    def ball_q(sit=None):
        if mode in ("sit", "sit_in") and sit is not None: return qmul(q_axis((1, 0, 0), sit), q_axis(rng.normal(size=3), abs(rng.normal(0, .08))))
        return q_axis(rng.normal(size=3), rng.uniform(0, .8) if loose else abs(rng.normal(0, .08)))
    def joint(kind, lower, upper, sit=None):
        if kind == "ball": return dict(kind="ball", damping=damp(), q0=ball_q(sit))
        return dict(kind="hinge", axis=(1, 0, 0), lower=lower, upper=upper, damping=damp(), q0=hinge_q(lower, upper, sit))
    tw, th, td = s*.28*b.U(.85, 1.1), s*.36*b.U(.9, 1.1), s*.14*b.U(.85, 1.15)
    torso = a.root((tw, th, td), mass((tw, th, td)))
    hs = s*.14*b.U(.9, 1.1)
    a.link(torso, (hs, hs, hs), mass((hs, hs, hs)), anchor_parent=(0, th/2, 0), anchor_child=(0, -hs/2, 0),
           **joint("ball" if rng.random() < .4 else "hinge", -.7, .7))
    aw, ul, fw, fl = s*.07, s*.2*b.U(.9, 1.1), s*.06, s*.2*b.U(.9, 1.1)
    lw, tl, sw, sl = s*.09, s*.24*b.U(.9, 1.1), s*.08, s*.24*b.U(.9, 1.1)
    shoulder = "ball" if rng.random() < .6 else "hinge"; hip = "ball" if rng.random() < .5 else "hinge"
    for side in (-1, 1):
        upper = a.link(torso, (aw, ul, aw), mass((aw, ul, aw)), anchor_parent=(side*(tw/2+aw/2+.002), th/2-aw/2, 0),
                       anchor_child=(0, ul/2-aw/2, 0), **joint(shoulder, -2.8, 1.))
        a.link(upper, (fw, fl, fw), mass((fw, fl, fw)), anchor_parent=(0, -ul/2, 0), anchor_child=(0, fl/2, 0),
               **joint("hinge", -2.4, .05))
    for side in (-1, 1):
        thigh = a.link(torso, (lw, tl, lw), mass((lw, tl, lw)), anchor_parent=(side*tw/4, -th/2, 0), anchor_child=(0, tl/2, 0),
                       **joint(hip, -2., .5, sit=-math.pi/2))
        a.link(thigh, (sw, sl, sw), mass((sw, sl, sw)), anchor_parent=(0, -tl/2, 0), anchor_child=(0, sl/2, 0),
               **joint("hinge", -.05, 2.4, sit=math.pi/2))
    if mode in ("drop", "toss"): ok = b.stage(a, mode)
    elif mode == "drop_low":   # let go flat, a few cm above the middle of the table: lands, limbs settle, stays up
        a.root_q = q_axis((1, 0, 0), -math.pi/2 if rng.random() < .7 else math.pi/2)
        ok = b.place(a, gap=b.U(.005, .08), inset=.05)
    elif mode == "sit_in":     # sitting in the middle of the table, legs on the table: rests, or slowly topples back / sideways
        ok = False
        for _ in range(20):
            normal, reach, half = b.edge(); tangent = np.array([normal[2], 0., normal[0]])
            a.q = q_yaw(math.atan2(normal[0], normal[2]))
            a.t = normal*(reach-tl-sl-b.U(.03, .2))+tangent*float(rng.uniform(-half+tw, half-tw))
            a.t[1] = b.h+th/2+lw/2+CLEARANCE+.001
            if abs(a.t[0]) < b.w/2-.05 and abs(a.t[2]) < b.d/2-.05 and b.free(a): ok = b.commit(a); break
        if ok and rng.random() < .4: a.lin = horizontal(rng.uniform(-math.pi, math.pi))*b.U(.05, .4)
    elif mode == "stand" and calm:   # standing, barely pushed: a slow topple
        ok = b.place(a, inset=.1)
        if ok and rng.random() < .7: a.lin = horizontal(rng.uniform(-math.pi, math.pi))*b.U(.02, .3)
    elif mode == "lie" and calm:     # lying in the middle (70%) or at an edge with the COM on the table, at rest or nudged
        a.root_q = q_axis((1, 0, 0), -math.pi/2)
        if rng.random() < .3:
            ok = b.place_edge(a, b.U(-.03, .25))
            if ok and rng.random() < .5: a.lin = -a.edge_normal*b.U(.05, .6)
        else:
            ok = b.place(a, inset=.05)
            if ok and rng.random() < .6: a.lin = horizontal(rng.uniform(-math.pi, math.pi))*b.U(.05, .5)
    elif mode == "stand":
        ok = b.place(a)
        if ok:
            if rng.random() < .7: a.lin = horizontal(rng.uniform(-math.pi, math.pi))*b.U(.1, 1.2)
            else: a.kicks.append((1, a.bodies[1]["mass"]*horizontal(rng.uniform(-math.pi, math.pi))*b.U(1., 4.)))
    elif mode == "sit":
        ok = False
        for _ in range(20):
            normal, reach, half = b.edge(); tangent = np.array([normal[2], 0., normal[0]])
            a.q = q_yaw(math.atan2(normal[0], normal[2]))
            a.t = normal*(reach+sw/2+b.U(.003, .03)-tl)+tangent*float(rng.uniform(-half+tw, half-tw))
            a.t[1] = b.h+th/2+lw/2+CLEARANCE+.001
            if b.free(a): ok = b.commit(a); break
        if ok:
            r = rng.random()
            if r < .5: a.lin = normal*b.U(-.6, 1.)+tangent*b.U(-.4, .4)
            elif r < .8: a.kicks.append((1, a.bodies[1]["mass"]*(normal*b.U(-3, 3)+tangent*b.U(-2, 2))))
    else:   # lying on the back at an edge
        a.root_q = q_axis((1, 0, 0), -math.pi/2)
        if rng.random() < .7:
            ok = b.place_edge(a, b.U(0, .5))
            if ok and rng.random() < .75: a.lin = a.edge_normal*b.U(.3, 1.8)
        else:
            ok = b.place(a)
            if ok and rng.random() < .6: a.lin = horizontal(rng.uniform(-math.pi, math.pi))*b.U(.3, 2.)
    if not ok: b.ok = False; return
    if rng.random() < .12 and not calm: b.throw_at(b.centre(a))
    b.add_free(int(rng.integers(0, 3)))

def s_chain_calm(b: JB): s_chain(b, calm=True)
def s_arm_calm(b: JB): s_robot_arm(b, calm=True)
def s_ragdoll_calm(b: JB): s_ragdoll(b, calm=True)


SCENARIOS = dict(laptop=s_laptop, box_lid=s_box_lid, door=s_door, drawer=s_drawer, pendulum=s_pendulum, chain=s_chain,
                 robot_arm=s_robot_arm, ragdoll=s_ragdoll,
                 # round 2 (never drawn by the default KIND_WEIGHTS, so data/joints_v1 stays reproducible): targeted 'calm' variants
                 lid_calm=s_lid_calm, chain_calm=s_chain_calm, arm_calm=s_arm_calm, ragdoll_calm=s_ragdoll_calm)


# ----------------------------------------------------------------------------- Bullet world
class JointWorld:
    """Isolated DIRECT client with the engine parameters of agatha.physics.World, floor + support slab, and one
    Featherstone multibody per articulation.  slots[i] = (body unique id, link index or -1)."""

    def __init__(self, b: JB):
        self.b = b
        self.client = c = p.connect(p.DIRECT)
        p.setPhysicsEngineParameter(fixedTimeStep=DT, numSolverIterations=50, deterministicOverlappingPairs=1, physicsClientId=c)
        p.setGravity(0, 0, -b.gravity, physicsClientId=c)
        self.floor = self._static((20, .1, 20), (0, -.05, 0), .5, 0.)
        self.support = self._static((b.w, SUPPORT_THICKNESS, b.d), (0, b.h-SUPPORT_THICKNESS/2, 0), b.support_friction, b.support_restitution)
        self.slots: list[tuple[int, int]] = []; self.bodies: list[dict] = []; self.group: list[int] = []
        self.joints: list[dict] = []     # global rows: parent/child slots + (uid, link index) + spec
        self.kicks: list[tuple[int, np.ndarray]] = []
        self.fk_error = 0.
        groups = 0
        for art in b.arts:
            first = len(self.slots)
            uid = self._create(art)
            offset = 0 if art.anchored else -1   # body i of the art is link i+offset
            for i, body in enumerate(art.bodies):
                self.slots.append((uid, i+offset)); self.bodies.append(body)
                self.group.append(groups if art.joints else -1)
            groups += bool(art.joints)
            for k, j in enumerate(art.joints):
                anchor = j["anchor_parent"]+art.t if j["parent"] < 0 else j["anchor_parent"]
                self.joints.append(dict(j, parent_slot=-1 if j["parent"] < 0 else first+j["parent"], child_slot=first+j["child"],
                                        uid=uid, link=k, anchor_parent=anchor, limited=art.limited(j)))
            for index, impulse in art.kicks: self.kicks.append((first+index, np.asarray(impulse, float)))
            # the Bullet poses must reproduce the numpy forward kinematics that the placement logic used
            pos, quat = art.fk(); state = self.read()
            for i in range(len(art.bodies)):
                self.fk_error = max(self.fk_error, float(np.abs(state[first+i, :3]-pos[i]).max()),
                                    float(1-abs(np.dot(state[first+i, 3:7], quat[i]))))

    def _static(self, size, pos, friction, restitution):
        c = self.client
        shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=(size[0]/2, size[2]/2, size[1]/2), physicsClientId=c)
        body = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=shape, basePosition=to_bullet(pos), physicsClientId=c)
        p.changeDynamics(body, -1, lateralFriction=friction, restitution=restitution, linearDamping=.04, angularDamping=.04, physicsClientId=c)
        return body

    def _dynamics(self, uid, link, body, **extra):
        if body["rolling"]: extra["rollingFriction"] = body["rolling"]
        if body["spinning"]: extra["spinningFriction"] = body["spinning"]
        p.changeDynamics(uid, link, lateralFriction=body["friction"], restitution=body["restitution"],
                         physicsClientId=self.client, **extra)

    def _create(self, art: Art) -> int:
        c = self.client
        def shape(body, offset=(0, 0, 0)):
            size = body["size"]
            return p.createCollisionShape(p.GEOM_BOX, halfExtents=(size[0]/2, size[2]/2, size[1]/2),
                                          collisionFramePosition=to_bullet(offset), physicsClientId=c)
        kwargs = {}
        if art.joints:
            # Link frame = joint frame (at the anchor, parent orientation at q=0); the box centre / COM sits at
            # -anchor_child in it.  linkPositions are relative to the PARENT LINK frame (verified numerically).
            child_anchor = {j["child"]: j["anchor_child"] for j in art.joints}
            link_of = {j["child"]: k for k, j in enumerate(art.joints)}
            positions, parents = [], []
            for j in art.joints:
                if j["parent"] < 0 or (not art.anchored and j["parent"] == 0):
                    positions.append(to_bullet(j["anchor_parent"])); parents.append(0)
                else:
                    positions.append(to_bullet(j["anchor_parent"]-child_anchor[j["parent"]])); parents.append(link_of[j["parent"]]+1)
            types = dict(hinge=p.JOINT_REVOLUTE, ball=p.JOINT_SPHERICAL, prismatic=p.JOINT_PRISMATIC, fixed=p.JOINT_FIXED)
            count = len(art.joints)
            kwargs = dict(
                linkMasses=[art.bodies[j["child"]]["mass"] for j in art.joints],
                linkCollisionShapeIndices=[shape(art.bodies[j["child"]], -j["anchor_child"]) for j in art.joints],
                linkVisualShapeIndices=[-1]*count, linkPositions=positions, linkOrientations=[IDENTITY]*count,
                linkInertialFramePositions=[to_bullet(-j["anchor_child"]) for j in art.joints],
                linkInertialFrameOrientations=[IDENTITY]*count, linkParentIndices=parents,
                linkJointTypes=[types[j["kind"]] for j in art.joints],
                linkJointAxis=[to_bullet(j["axis"]) if j["kind"] in ("hinge", "prismatic") else (0, 0, 1) for j in art.joints],
                flags=p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT)
        if art.anchored:
            uid = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, basePosition=to_bullet(art.t), physicsClientId=c, **kwargs)
            p.changeDynamics(uid, -1, linearDamping=.04, angularDamping=.04, physicsClientId=c)
        else:
            root = art.bodies[0]
            uid = p.createMultiBody(baseMass=0 if root["static"] else root["mass"], baseCollisionShapeIndex=shape(root),
                                    basePosition=to_bullet(art.t), baseOrientation=quat_to_bullet(qmul(art.q, art.root_q)),
                                    physicsClientId=c, **kwargs)
            self._dynamics(uid, -1, root, linearDamping=.04, angularDamping=.04)
        for k, j in enumerate(art.joints):
            extra = dict(jointDamping=j["damping"])
            if art.limited(j): extra.update(jointLowerLimit=j["lower"], jointUpperLimit=j["upper"])
            self._dynamics(uid, k, art.bodies[j["child"]], **extra)
            if j["kind"] in ("hinge", "prismatic"):
                p.setJointMotorControl2(uid, k, p.VELOCITY_CONTROL, force=0, physicsClientId=c)
                p.resetJointState(uid, k, j["q0"], j["qd0"], physicsClientId=c)
            elif j["kind"] == "ball":
                p.setJointMotorControlMultiDof(uid, k, p.POSITION_CONTROL, targetPosition=[0, 0, 0, 1], targetVelocity=[0, 0, 0],
                                               positionGain=0, velocityGain=0, force=[0, 0, 0], physicsClientId=c)
                p.resetJointStateMultiDof(uid, k, quat_to_bullet(j["q0"]), list(to_bullet(np.asarray(j["qd0"], float))), physicsClientId=c)
        if not art.anchored and not art.bodies[0]["static"] and (np.any(art.lin) or np.any(art.ang)):
            p.resetBaseVelocity(uid, to_bullet(art.lin), to_bullet(art.ang), physicsClientId=c)
        return uid

    # -- queries
    def read(self) -> np.ndarray:
        """[n,13] float64 Y-up states of all slots (box centres, world velocities)."""
        c = self.client; out = np.zeros((len(self.slots), 13))
        for i, (uid, link) in enumerate(self.slots):
            if link < 0:
                pos, q = p.getBasePositionAndOrientation(uid, physicsClientId=c); v, w = p.getBaseVelocity(uid, physicsClientId=c)
            else:
                s = p.getLinkState(uid, link, computeLinkVelocity=1, computeForwardKinematics=1, physicsClientId=c)
                pos, q, v, w = s[0], s[1], s[6], s[7]
            out[i] = [*from_bullet(pos), *quat_from_bullet(q), *from_bullet(v), *from_bullet(w)]
        return out

    def joint_positions(self) -> np.ndarray:
        out = np.zeros(len(self.joints))
        for k, j in enumerate(self.joints):
            if j["kind"] == "hinge": out[k] = wrap_angle(p.getJointState(j["uid"], j["link"], physicsClientId=self.client)[0])
            elif j["kind"] == "prismatic": out[k] = p.getJointState(j["uid"], j["link"], physicsClientId=self.client)[0]
        return out

    def jointed_pairs(self) -> set:
        return {tuple(sorted((j["parent_slot"], j["child_slot"]))) for j in self.joints if j["parent_slot"] >= 0}

    def contact_pairs(self) -> set:
        """{(slot a, slot b)} with a < b; the support is 'support', the floor 'floor' (as second element)."""
        index = {slot: i for i, slot in enumerate(self.slots)}
        index[self.floor, -1] = "floor"; index[self.support, -1] = "support"
        pairs = set()
        for cp in p.getContactPoints(physicsClientId=self.client):
            if cp[8] > CONTACT_DISTANCE: continue
            a, b = index.get((cp[1], cp[3])), index.get((cp[2], cp[4]))
            if a is None or b is None or a == b: continue
            if isinstance(a, str): a, b = b, a
            if isinstance(a, str): continue
            pairs.add((a, b) if isinstance(b, str) or a < b else (b, a))
        return pairs

    def initial_pairs(self) -> tuple[set, float]:
        """Closest-point contacts before the first substep and the deepest overlap (jointed pairs excluded)."""
        c = self.client; pairs = set(); deepest = 0.; skip = self.jointed_pairs()
        static = [body["static"] for body in self.bodies]
        targets = [("floor", (self.floor, -1)), ("support", (self.support, -1))]
        for i, (uid, link) in enumerate(self.slots):
            others = ([] if static[i] else targets)+[(k, self.slots[k]) for k in range(i+1, len(self.slots))]
            for name, (uid2, link2) in others:
                if not isinstance(name, str) and ((i, name) in skip or (static[i] and static[name])): continue
                points = p.getClosestPoints(uid, uid2, CONTACT_DISTANCE, link, link2, physicsClientId=c)
                if points:
                    pairs.add((i, name)); deepest = max(deepest, -min(pt[8] for pt in points))
        return pairs, deepest

    def step(self):
        p.stepSimulation(physicsClientId=self.client)

    def close(self):
        if p.isConnected(self.client): p.disconnect(self.client)

    def __enter__(self): return self
    def __exit__(self, *args): self.close()


# ----------------------------------------------------------------------------- episode
def build_world(rng: np.random.Generator, kind: str, attempts: int = 30) -> JointWorld | None:
    """Draw scenes until one has <= N slots, <= J joints, a consistent Bullet build and no deep initial overlap."""
    for _ in range(attempts):
        b = JB(rng)
        SCENARIOS[kind](b)
        if not b.ok or not b.arts or b.slots_used() > N_SLOTS or sum(len(a.joints) for a in b.arts) > J_SLOTS: continue
        world = JointWorld(b)
        if world.fk_error < 1e-5 and world.initial_pairs()[1] <= MAX_INIT_PENETRATION: return world
        world.close()
    return None


def joint_rows(world: JointWorld) -> tuple[np.ndarray, np.ndarray]:
    joints = np.zeros((J_SLOTS, JOINT_DIM), np.float32); mask = np.zeros(J_SLOTS, bool)
    for k, j in enumerate(world.joints):
        onehot = [float(j["kind"] == name) for name in JOINT_TYPES]
        lower, upper = j["lower"], j["upper"]
        if j["kind"] == "hinge" and not j["limited"]: lower, upper = -HINGE_FREE, HINGE_FREE
        if j["kind"] == "prismatic" and not j["limited"]: lower, upper = -PRISMATIC_FREE, PRISMATIC_FREE
        joints[k] = [j["parent_slot"], j["child_slot"], *onehot, *j["anchor_parent"], *j["anchor_child"], *j["axis"], lower, upper, j["damping"]]
        mask[k] = True
    return joints, mask


def simulate_world(world: JointWorld, kind: str, steps: int = 80) -> tuple[dict | None, str]:
    """Roll the built world out; returns (arrays, 'ok') or (None, reason)."""
    b = world.b; c = world.client; n = len(world.slots); count = len(world.joints)
    if world.kicks:   # impulses as short forces before the recording starts
        for _ in range(PREROLL_SUBSTEPS):
            state = world.read()
            for slot, impulse in world.kicks:
                uid, link = world.slots[slot]
                p.applyExternalForce(uid, link, to_bullet(impulse/(PREROLL_SUBSTEPS*DT)), to_bullet(state[slot, :3]),
                                     p.WORLD_FRAME, physicsClientId=c)
            world.step()
    states = np.zeros((steps+1, N_SLOTS, 13), np.float32); states[:, :, 6] = 1
    joint_pos = np.zeros((steps+1, J_SLOTS), np.float32)
    contact = np.zeros((3, steps+1, N_SLOTS), bool)
    dynamic_slots = {i for i, body in enumerate(world.bodies) if not body["static"]}

    def record(t, pairs):
        states[t, :n] = world.read(); joint_pos[t, :count] = world.joint_positions()
        if t:
            flip = np.sum(states[t-1, :n, 3:7]*states[t, :n, 3:7], axis=1) < 0
            states[t, :n][flip, 3:7] *= -1
        for a, other in pairs:
            column = 0 if other == "support" else 1 if other == "floor" else 2
            contact[column, t, a] = True
            if column == 2: contact[2, t, other] = True

    pairs, init_penetration = world.initial_pairs()
    record(0, pairs)
    previous = {pair for pair in pairs if pair[0] in dynamic_slots and pair[1] in dynamic_slots}
    impact = False
    for t in range(1, steps+1):
        for _ in range(SUBSTEPS):
            world.step()
            pairs = world.contact_pairs()
            if not impact:
                current = {pair for pair in pairs if pair[0] in dynamic_slots and pair[1] in dynamic_slots}
                for a, other in current-previous:   # velocities of the last recorded frame (20 Hz)
                    if np.linalg.norm(states[t-1, a, 7:10]-states[t-1, other, 7:10]) > .5: impact = True
                previous = current
        record(t, pairs)

    if not np.isfinite(states).all() or not np.isfinite(joint_pos).all() or np.abs(states[:, :, :3]).max() > 50: return None, "non_finite"
    if states[:, :, 1].min() < -.5 or np.abs(states[:, :, [0, 2]]).max() > FLOOR_HALF: return None, "left_floor"
    if np.linalg.norm(states[:, :, 7:10], axis=-1).max() > MAX_SPEED or np.linalg.norm(states[:, :, 10:13], axis=-1).max() > MAX_SPIN:
        return None, "exploded"
    joints, joint_mask = joint_rows(world)
    geometry = check_joints(states, joints, joint_mask)
    anchor_gap = float(geometry["anchor_gap"].max()); axis_error = float(geometry["axis_error"].max())
    pos_error = float(coordinate_error(joint_pos, geometry["coordinate"], joints, joint_mask).max())
    if anchor_gap > MAX_ANCHOR_GAP or axis_error > MAX_AXIS_ERROR or pos_error > MAX_COORD_ERROR: return None, "inconsistent"
    excess = np.maximum(joints[:, 15]-joint_pos, 0)+np.maximum(joint_pos-joints[:, 16], 0)
    if (excess*(joints[:, 2] > .5)).max() > MAX_LIMIT_EXCESS_RAD or (excess*(joints[:, 4] > .5)).max() > MAX_LIMIT_EXCESS_M:
        return None, "limit_violation"

    mask = np.zeros(N_SLOTS, bool); mask[:n] = True
    props = np.zeros((N_SLOTS, PROPS_DIM), np.float32); props[:, 3] = 1
    for i, body in enumerate(world.bodies):
        props[i] = [*body["size"], body["mass"], body["friction"], float(body["static"]), body["restitution"], body["rolling"], body["spinning"]]
    context = np.asarray([b.w, b.d, b.h, b.gravity, 0., 0., 0., 1., b.support_friction, b.support_restitution], np.float32)
    group = np.full(N_SLOTS, -1, np.int8); group[:n] = world.group
    dynamic = mask & (props[:, 5] < .5)
    speed = np.linalg.norm(states[:, :, 7:10], axis=-1); spin = np.linalg.norm(states[:, :, 10:13], axis=-1)
    rest = contact.any(0) & (speed < REST_LIN) & (spin < REST_ANG) & dynamic
    # agatha.physics.summarize semantics (flat support): COM outside the footprint and below top - 3 cm; body
    # +Y axis more than 35 degrees from world up
    position = states[:, :, :3].astype(np.float64)
    outside = (np.abs(position[:, :, 0]) > b.w/2) | (np.abs(position[:, :, 2]) > b.d/2)
    fell = (outside & (position[:, :, 1]-(b.h-SUPPORT_THICKNESS/2) < SUPPORT_THICKNESS/2-.03)).any(0) & dynamic
    up = quat_to_matrix(states[:, :, 3:7])[..., 1, 1]
    tipped_frames = up < math.cos(math.radians(35))
    tipped = tipped_frames.any(0) & dynamic
    displacement = np.linalg.norm(states[:, :, :3]-states[:1, :, :3], axis=-1).max(0)
    chaotic = bool(((fell | (tipped & ~tipped_frames[0]) | (displacement > .30)) & dynamic).any() or impact)
    live = bool(np.all(props[:n, 6:9] == 0) and context[8] == np.float32(.5) and context[9] == 0)
    return dict(states=states, props=props, context=context, mask=mask, contact_support=contact[0], contact_floor=contact[1],
                contact_object=contact[2], rest=rest, fell=fell, tipped=tipped, kind=kind, chaotic=chaotic, live_physics=live,
                init_penetration=np.float32(init_penetration), joints=joints, joint_mask=joint_mask, joint_pos=joint_pos,
                group=group, joint_anchor_gap=np.float32(anchor_gap), joint_axis_error=np.float32(axis_error),
                joint_pos_error=np.float32(pos_error)), "ok"


def plain_episode(rng: np.random.Generator, steps: int) -> tuple[dict | None, str, str]:
    """A jointless box scene from the contact generator, padded to N_SLOTS."""
    case = box_data.make_case(rng, 8)
    kind = ("plain_"+case["kind"])[:24]
    episode = box_data.simulate_case(case, steps, N_SLOTS)
    if episode is None: return None, "plain_invalid", kind
    episode.update(kind=kind, joints=np.zeros((J_SLOTS, JOINT_DIM), np.float32), joint_mask=np.zeros(J_SLOTS, bool),
                   joint_pos=np.zeros((steps+1, J_SLOTS), np.float32), group=np.full(N_SLOTS, -1, np.int8),
                   joint_anchor_gap=np.float32(0), joint_axis_error=np.float32(0), joint_pos_error=np.float32(0))
    return episode, "ok", kind


def generate_episode(base_seed: int, split: str, index: int, steps: int = 80, kind: str | None = None,
                     kind_weights: dict | None = None) -> tuple[dict | None, str, str]:
    """(arrays or None, 'ok' or the drop reason, kind).  Deterministic in (base_seed, split, index)."""
    rng = episode_rng(base_seed, split, index)
    weights = (_ACTIVE_WEIGHTS or KIND_WEIGHTS) if kind_weights is None else kind_weights
    kinds = list(weights); probabilities = np.asarray([weights[k] for k in kinds], float)
    drawn = kinds[int(rng.choice(len(kinds), p=probabilities/probabilities.sum()))]
    kind = drawn if kind is None else kind
    if kind == "plain": return plain_episode(rng, steps)
    world = build_world(rng, kind)
    if world is None: return None, "build_failed", kind
    with world:
        episode, reason = simulate_world(world, kind, steps)
    return episode, reason, kind


_ACTIVE_WEIGHTS = None


def _chunk(task):
    base_seed, split, shard, indices, steps = task
    episodes, drops = [], []
    for index in indices:
        episode, reason, kind = generate_episode(base_seed, split, index, steps)
        if episode is not None: episode["episode_index"] = index
        else: drops.append((kind, reason))
        episodes.append(episode)
    return split, shard, indices[0], episodes, drops


ARRAY_KEYS = box_data.ARRAY_KEYS+("joints", "joint_mask", "joint_pos", "group", "joint_anchor_gap", "joint_axis_error", "joint_pos_error")
_SCALARS = dict(chaotic=bool, live_physics=bool, init_penetration=np.float32, joint_anchor_gap=np.float32,
                joint_axis_error=np.float32, joint_pos_error=np.float32)

def _write_shard(path: Path, episodes: list[dict]):
    arrays = {k: np.stack([e[k] for e in episodes]) for k in ARRAY_KEYS if k not in ("kind", "episode_index")}
    arrays["kind"] = np.asarray([e["kind"] for e in episodes], dtype="<U24")
    arrays["episode_index"] = np.asarray([e["episode_index"] for e in episodes], np.int64)
    for key, dtype in _SCALARS.items(): arrays[key] = arrays[key].astype(dtype)
    np.savez_compressed(path, **arrays)


def _summary(episode: dict) -> dict:
    """Small per-episode record kept in the parent process for meta.json."""
    mask, joints, joint_mask = episode["mask"], episode["joints"], episode["joint_mask"]
    dynamic = mask & (episode["props"][:, 5] < .5); frames = episode["states"].shape[0]
    types = joints[joint_mask, 2:6].sum(0)
    lower, upper, pos = joints[:, 15], joints[:, 16], episode["joint_pos"]
    one_dof = joint_mask & ((joints[:, 2] > .5) | (joints[:, 4] > .5))
    limited = one_dof & ~(((joints[:, 2] > .5) & (upper >= HINGE_FREE-1e-4)) | ((joints[:, 4] > .5) & (upper >= PRISMATIC_FREE-1e-4)))
    margin = np.where(joints[:, 4] > .5, .003, .02)
    at_limit = ((pos < lower+margin) | (pos > upper-margin)) & limited
    excess = (np.maximum(lower-pos, 0)+np.maximum(pos-upper, 0))*limited
    count = lambda key: int((episode[key] & dynamic).sum())
    return dict(kind=episode["kind"], chaotic=bool(episode["chaotic"]), live=bool(episode["live_physics"]),
                bodies=int(mask.sum()), dynamic=int(dynamic.sum()), static=int((mask & ~dynamic).sum()),
                jointed_bodies=int((episode["group"] >= 0).sum()), joints=int(joint_mask.sum()),
                types=types.astype(int).tolist(), world_joints=int((joint_mask & (joints[:, 0] < 0)).sum()),
                limited=int(limited.sum()), limit_frames=int(at_limit.sum()), limited_frames=int(limited.sum())*frames,
                excess_frames=int((excess*(joints[:, 2] > .5) > .05).sum()),
                limit_excess_hinge=float((excess*(joints[:, 2] > .5)).max()), limit_excess_prismatic=float((excess*(joints[:, 4] > .5)).max()),
                rest=count("rest"), contact_support=count("contact_support"), contact_floor=count("contact_floor"),
                contact_object=count("contact_object"), fell=count("fell"), tipped=count("tipped"),
                frames=frames, anchor_gap=float(episode["joint_anchor_gap"]), axis_error=float(episode["joint_axis_error"]),
                pos_error=float(episode["joint_pos_error"]), init_penetration=float(episode["init_penetration"]),
                max_speed=float(np.linalg.norm(episode["states"][:, :, 7:10], axis=-1).max()),
                max_spin=float(np.linalg.norm(episode["states"][:, :, 10:13], axis=-1).max()),
                gravity_changed=bool(abs(episode["context"][3]-9.81) > 1e-4))


def _stats(rows: list[dict], steps: int) -> dict:
    if not rows: return dict(episodes=0)
    total = lambda key, items=rows: float(sum(r[key] for r in items))
    def block(items):
        frames = max(total("dynamic", items)*(steps+1), 1); objects = max(total("dynamic", items), 1)
        return dict(
            episodes=len(items), chaotic_fraction=float(np.mean([r["chaotic"] for r in items])),
            mean_bodies=total("bodies", items)/len(items), mean_joints=total("joints", items)/len(items),
            rest_fraction_object_frames=total("rest", items)/frames, contact_support_fraction=total("contact_support", items)/frames,
            contact_floor_fraction=total("contact_floor", items)/frames, contact_object_fraction=total("contact_object", items)/frames,
            fell_fraction_objects=total("fell", items)/objects, tipped_fraction_objects=total("tipped", items)/objects,
            at_limit_fraction_of_limited_joint_frames=total("limit_frames", items)/max(total("limited_frames", items), 1),
            hinge_limit_excess_over_0p05rad_fraction_of_limited_joint_frames=total("excess_frames", items)/max(total("limited_frames", items), 1),
            max_anchor_gap_m=max(r["anchor_gap"] for r in items), max_axis_error_rad=max(r["axis_error"] for r in items),
            max_joint_pos_error=max(r["pos_error"] for r in items),
            max_limit_excess_rad=max(r["limit_excess_hinge"] for r in items), max_limit_excess_m=max(r["limit_excess_prismatic"] for r in items),
            max_speed=max(r["max_speed"] for r in items), max_spin=max(r["max_spin"] for r in items))
    kinds = sorted({r["kind"] for r in rows})
    grouped = {"plain": [r for r in rows if r["kind"].startswith("plain_")]}
    grouped.update({k: [r for r in rows if r["kind"] == k] for k in kinds if not k.startswith("plain_")})
    types = np.sum([r["types"] for r in rows], 0)
    out = block(rows)
    out.update(
        transitions=len(rows)*steps, object_transitions=int(total("dynamic")*steps),
        jointed_body_transitions=int(total("jointed_bodies")*steps), joint_transitions=int(total("joints")*steps),
        kinds={k: int(sum(r["kind"] == k for r in rows)) for k in kinds},
        jointless_fraction=float(np.mean([r["joints"] == 0 for r in rows])),
        joint_type_counts=dict(zip(JOINT_TYPES, (int(v) for v in types))),
        world_anchored_joints=int(total("world_joints")), limited_joints=int(total("limited")),
        live_physics_fraction=float(np.mean([r["live"] for r in rows])),
        static_fraction_objects=total("static")/max(total("bodies"), 1),
        gravity_changed_fraction=float(np.mean([r["gravity_changed"] for r in rows])),
        init_penetration_over_1mm_fraction=float(np.mean([r["init_penetration"] > 1e-3 for r in rows])),
        anchor_gap_percentiles_m=dict(zip(("p50", "p99", "max"), (float(v) for v in np.percentile([r["anchor_gap"] for r in rows if r["joints"]] or [0], [50, 99, 100])))),
        by_kind={k: block(v) for k, v in grouped.items() if v},
    )
    return out


def generate(out: Path, train_episodes: int = 16000, val_episodes: int = 600, test_episodes: int = 2000, steps: int = 80,
             workers: int = 40, shard_size: int = 500, seed: int = 4041, chunk: int = 10, quiet: bool = False,
             kind_weights: dict | None = None) -> dict:
    global _ACTIVE_WEIGHTS
    if kind_weights is not None:
        unknown = set(kind_weights)-set(SCENARIOS)-{"plain"}
        if unknown or not kind_weights or min(kind_weights.values()) < 0 or sum(kind_weights.values()) <= 0:
            raise ValueError(f"bad kind weights {kind_weights}")
        kind_weights = {k: float(v) for k, v in kind_weights.items()}
    _ACTIVE_WEIGHTS = kind_weights   # inherited by the forked workers
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
            tasks += [(seed, split, shard, piece, steps) for piece in pieces]
    kept = {split: [] for split in SPLITS}; dropped = {split: 0 for split in SPLITS}; reasons: dict[str, dict[str, int]] = {}
    done = 0; total = sum(counts.values()); writes = []
    workers = max(1, min(workers, len(tasks) or 1))
    context = mp.get_context("fork")
    with ThreadPoolExecutor(4) as writer:
        pool = context.Pool(workers) if workers > 1 else None
        try:
            results = pool.imap_unordered(_chunk, tasks) if pool else map(_chunk, tasks)
            last = start
            for split, shard, first, episodes, drops in results:
                buffers[split, shard][first] = episodes; pending[split, shard] -= 1; done += len(episodes)
                for kind, reason in drops:
                    kind = "plain" if kind.startswith("plain_") else kind
                    reasons.setdefault(reason, {}); reasons[reason][kind] = reasons[reason].get(kind, 0)+1
                if pending[split, shard] == 0:
                    pieces = buffers.pop((split, shard))
                    ordered = [e for key in sorted(pieces) for e in pieces[key]]
                    good = [e for e in ordered if e is not None]; dropped[split] += len(ordered)-len(good)
                    if good: writes.append(writer.submit(_write_shard, out/split/f"shard_{shard:05d}.npz", good))
                    kept[split] += [_summary(e) for e in good]
                if not quiet and time.perf_counter()-last > 10:
                    last = time.perf_counter()
                    print(f"{done}/{total} episodes; {done/(last-start):.1f} episodes/s", flush=True)
        finally:
            if pool: pool.close(); pool.join()
            _ACTIVE_WEIGHTS = None
        simulated = time.perf_counter()-start
        for w in writes: w.result()
    seconds = time.perf_counter()-start
    everything = [e for split in SPLITS for e in kept[split]]
    meta = dict(
        format=FORMAT, step_s=STEP, substeps_per_step=SUBSTEPS, steps=steps, max_objects=N_SLOTS, max_joints=J_SLOTS,
        joint_dim=JOINT_DIM, joint_types=list(JOINT_TYPES), hinge_unlimited=HINGE_FREE, prismatic_unlimited=PRISMATIC_FREE,
        seed=seed, split_seeds={s: seed+SPLIT_SEED_OFFSET[s] for s in SPLITS},
        seeding="np.random.SeedSequence([seed+split_offset, split_index, episode_index]) per episode",
        requested=counts, counts={s: len(kept[s]) for s in SPLITS}, dropped=dropped, dropped_by_reason=reasons, shard_size=shard_size,
        episodes=len(everything), transitions=len(everything)*steps,
        seconds=seconds, simulation_seconds=simulated, workers=workers, episodes_per_second=total/max(simulated, 1e-9),
        kind_weights=KIND_WEIGHTS if kind_weights is None else kind_weights, ranges=RANGES,
        engine="PyBullet Featherstone multibodies (createMultiBody links), fixedTimeStep 1/240, 50 solver iterations, linear/angular "
               "damping 0.04, joint motors disabled, self-collision on except parent-child (URDF_USE_SELF_COLLISION_EXCLUDE_PARENT)",
        consistency_thresholds=dict(anchor_gap_m=MAX_ANCHOR_GAP, axis_error_rad=MAX_AXIS_ERROR, joint_pos_error=MAX_COORD_ERROR,
                                    max_init_penetration_m=MAX_INIT_PENETRATION, max_speed=MAX_SPEED, max_spin=MAX_SPIN,
                                    max_limit_excess_rad=MAX_LIMIT_EXCESS_RAD, max_limit_excess_m=MAX_LIMIT_EXCESS_M,
                                    damping_stability=DAMPING_STABILITY),
        rest_definition=f"in contact and |v|<{REST_LIN} m/s and |w|<{REST_ANG} rad/s",
        contact_definition=f"Bullet contact point with distance <= {CONTACT_DISTANCE} m in the last substep (frame 0: closest-point query); "
                           "contact_object includes non-adjacent links of the same articulation; jointed pairs never touch",
        chaotic_definition="any dynamic body fell, became tipped (was not at frame 0), or moved > 0.30 m from its start, or a contact "
                           "between two dynamic bodies began after t=0 with relative COM speed > 0.5 m/s (velocities of the previous "
                           "20 Hz frame); plain_* episodes use the generate_contact_data definition",
        live_physics_definition="restitution, rolling and spinning friction 0 for every object; support friction 0.5; support restitution 0",
        splits={s: _stats(kept[s], steps) for s in SPLITS}, overall=_stats(everything, steps),
    )
    (out/"meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("data/joints_v1"))
    parser.add_argument("--train-episodes", type=int, default=16000)
    parser.add_argument("--val-episodes", type=int, default=600)
    parser.add_argument("--test-episodes", type=int, default=2000)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--workers", type=int, default=40)
    parser.add_argument("--shard-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=4041)
    parser.add_argument("--kind-weights", type=json.loads, default=None,
                        help='JSON {kind: weight}, kinds: plain, ' + ", ".join(SCENARIOS))
    args = parser.parse_args()
    meta = generate(args.out, args.train_episodes, args.val_episodes, args.test_episodes, args.steps, args.workers,
                    args.shard_size, args.seed, kind_weights=args.kind_weights)
    brief = {k: meta[k] for k in ("counts", "dropped", "dropped_by_reason", "transitions", "seconds", "workers", "episodes_per_second")}
    brief["overall"] = meta["overall"]
    print(json.dumps(brief, indent=2), flush=True)


if __name__ == "__main__":
    main()
