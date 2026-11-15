"""Soft-body PyBullet episodes for the experimental SoftGNN track (data/soft_v1 format).

One deformable body per episode (NeoHookean tetrahedral cube / ball, or a mass-spring cloth patch) interacting
with the table, the floor and up to four rigid boxes, simulated in Bullet's deformable world
(p.RESET_USE_DEFORMABLE_WORLD) at 240 Hz and recorded at 20 Hz (STEP 0.05 s = 12 substeps), Y-up metres.
CPU only, multiprocessing, deterministic per-episode seeds, procedural meshes written to a temp dir.

    .venv/bin/python scripts/generate_soft_data.py --out data/soft_v1 \
        --train-episodes 6000 --val-episodes 300 --test-episodes 800 --workers 40

Layout: <out>/{train,val,test}/shard_XXXXX.npz + <out>/meta.json (see FORMAT below).
Exploding / tunnelling episodes are dropped and counted per reason in meta.json.
"""
import argparse
import contextlib
import itertools
import json
import math
import multiprocessing as mp
import os
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pybullet as p
from agatha.coordinates import to_bullet, quat_to_bullet, quat_from_bullet

STEP = .05
SUBSTEPS = 12
DT = 1/240
V_MAX, M_MAX, F_MAX, N_RIGID = 128, 768, 256, 4
STATE_DIM, PROPS_DIM, CONTEXT_DIM, SOFT_PROPS_DIM, SOFT_RAW_DIM = 13, 9, 10, 6, 10
SUPPORT_THICKNESS = .08
FLOOR_HALF = 10.
COLLISION_MARGIN = .005         # Bullet keeps soft nodes about this far outside rigid surfaces
SPARSE_SDF_VOXEL = .25
SOLVER_ITERATIONS = 50
REST_SPEED = .03                # m/s, max vertex speed (and rigid speed) for a frame 'at rest'
SPLITS = ("train", "val", "test")
SPLIT_SEED_OFFSET = dict(train=0, val=7919, test=104729)

SAG_RANGE = (.01, .4)           # volumetric softness: density*9.81*max_extent/mu, log-uniform (mu follows from it)
CLOTH_K_PER_KG = (100., 1000.)  # cloth springElasticStiffness = mass * loguniform(this)
CLOTH_DAMPING_PER_K = (7e-4, 5e-3)  # cloth springDampingStiffness = k * loguniform(this), clipped to 0.015-0.3
STIFF_NORM = dict(volumetric=(3.8, 1.6), cloth=(1.48, 1.0))  # (log10 centre, half range) -> about [-1, 1]

FORMAT = dict(
    verts="float32 [E,T+1,V=128,3] soft vertex positions (zero padded), Y-up metres, fixed vertex order",
    vert_vel="float32 [E,T+1,V,3] vertex velocities: finite difference over the last 240 Hz substep (frame 0: analytic initial velocity)",
    vert_mask="bool [E,V] real vertices",
    rest_edges="int16 [E,M=768,2] unique mesh edges incl. internal tetra edges, i<j, -1 padded",
    rest_len="float32 [E,M] rest length of each edge (0 padded)",
    faces="int16 [E,F=256,3] surface triangles (outward winding for volumetric bodies), -1 padded",
    soft_props="float32 [E,6] total_mass_kg, stiffness (normalised log10, see stiffness_normalisation), damping "
               "(NeoHookeanDamping or springDampingStiffness), friction (Bullet frictionCoeff; effective contact friction is "
               "the product with the other body's friction), is_cloth (0 volumetric / 1 cloth), rest volume m3 (cloth: rest area m2)",
    soft_raw="float32 [E,10] mu, lambda, neo_damping, spring_k, spring_damping, bending_k, size_x, size_y, size_z, collision_margin "
             "(unused entries 0; size = undeformed local extents, cloth size_y = 0)",
    soft_shape="<U8 [E] cube | ball | cloth", n_verts="int16 [E]", n_edges="int16 [E]", n_faces="int16 [E]",
    states="float32 [E,T+1,N=4,13] rigid boxes: pos3, quat xyzw (sign-continuous), linvel3, angvel3 (world); pad pos 0 quat (0,0,0,1)",
    props="float32 [E,N,9] size_x,size_y,size_z,mass_kg,friction,is_static,restitution,rolling_friction,spinning_friction (pad 0,0,0,1,0..)",
    mask="bool [E,N] real rigid box slots (may be all False)",
    context="float32 [E,10] support_w,support_d,support_height,gravity,tilt_x,tilt_y,tilt_z,support_collisions_enabled,support_friction,support_restitution "
            "(tilt always 0, collisions always 1, restitution always 0 in soft_v1)",
    kind="<U24 [E] generator scenario", episode_index="int64 [E] index within the split",
    final_com="float32 [E,3] mean of the soft vertices at the last frame (nodes have equal mass)",
    max_deformation="float32 [E] max over frames and edges of |len/rest_len - 1|",
    max_penetration="float32 [E] deepest soft-vertex penetration into floor / support slab / rigid boxes over all frames (m, >= 0)",
    min_volume_ratio="float32 [E] min over frames and tetrahedra of signed volume / rest volume (cloth: 1)",
    rest="bool [E,T+1] every soft vertex and every dynamic rigid box slower than REST_SPEED",
    fell="bool [E] soft COM ended more than 0.1 m below the table top",
)

KIND_WEIGHTS = dict(soft_drop=.22, soft_rest=.08, soft_launch=.16, soft_edge_fall=.12, box_on_soft=.14,
                    soft_on_boxes=.14, cloth_drape_box=.07, cloth_drape_edge=.05, cloth_drop=.02)
CLOTH_KINDS = ("cloth_drape_box", "cloth_drape_edge", "cloth_drop")

RANGES = dict(
    table="w 0.9-1.5, d 0.65-1.0, h 0.6-0.9; slab 0.08 thick; floor top at y=0",
    gravity="9.81, uniform 2-15 in ~6% of episodes",
    support_friction="0.5 (60%) or uniform 0.3-1.0",
    volumetric="cube (55%): 0.06-0.20 m per axis, 3-5 nodes per axis; ball (45%): diameter 0.07-0.20 x (0.8-1.2 per axis), 4^3 or 5^3 "
               "grid mapped to an ellipsoid; 6-tet Kuhn split; density 100-1200 kg/m3 (mass clipped 0.03-2 kg); NeoHookean "
               "mu = density*9.81*max_extent/sag with sag log-uniform 0.01-0.4 (mu about 150-235000), lambda = mu*U(1,4), "
               "damping log-uniform 0.005-0.1; frictionCoeff 0.3-1.0",
    cloth="8-11 x 8-11 vertex grid, side 0.2-0.45 m, 0.4-1.5 kg/m2 (mass clipped 0.03-0.3), mass-spring with bending springs: "
          "springElasticStiffness k = mass*loguniform(100,1000), springDampingStiffness = k*loguniform(7e-4,5e-3) clipped "
          "0.015-0.3 (all directions; needed for stability, so the cloth drapes slowly), "
          "springBendingStiffness log-uniform 0.1-10, useFaceContact",
    rigid_boxes="0.05-0.2 m per axis, density 150-2000 kg/m3 (mass clipped 0.05-3), friction 0.5 or 0.2-1.0, ~12% static in soft_on_boxes; "
                "box_on_soft: box mass = soft mass * U(0.2,1.5) (clipped 0.03-2)",
    soft_drop="gap 0.02-0.4 m above the table, yaw-only (40%) or random orientation, optional lateral velocity <= 0.8 m/s and spin <= 5 rad/s",
    soft_rest="placed just above the table, zero velocity (sags under gravity from the undeformed shape)",
    soft_launch="on the table, lateral 0.3-2.5 m/s, 30% with 0.3-1.5 m/s upward toss, balls 50% with rolling spin",
    soft_edge_fall="pushed at 0.6-2.5 m/s towards an edge 0.02-0.3 m away (65%) or placed overhanging the edge (35%)",
    box_on_soft="soft body resting on the table, 1-2 rigid boxes dropped onto it from 0.03-0.2 m",
    soft_on_boxes="1-3 rigid boxes on the table (sometimes a stack of 2), soft body dropped onto them from 0.03-0.3 m",
    cloth_drape_box="cloth dropped 0.02-0.15 m above a rigid box (50% static) standing on the table",
    cloth_drape_edge="cloth dropped 0.01-0.1 m above the table overhanging an edge by 25-65%",
    cloth_drop="cloth with a small tilt and lateral velocity dropped onto the table",
)
DROP_RULES = dict(
    nonfinite="non-finite soft or rigid state", bounds="|x|,|z| > 5 m or y > 5 m, or a rigid box below y=-0.5",
    below_floor="a soft vertex more than 0.03 m below the floor", speed="a soft vertex faster than 25 m/s",
    strain="edge strain |len/rest-1| above 3.0 (volumetric) / 1.0 (cloth)",
    tunnel="a soft vertex more than 0.03 m inside the support slab or a rigid box",
    energy="mechanical energy (soft nodes: kinetic + potential; rigid boxes: translational kinetic + potential) rises more "
           "than 0.25 J per kg of dynamic mass above its initial value (solver kick)",
    inverted="volumetric only: a tetrahedron with non-positive signed volume in any recorded frame",
)
STRAIN_LIMIT = dict(volumetric=3., cloth=1.)
TUNNEL_DEPTH = .03
ENERGY_GAIN_LIMIT = .25         # J per kg of dynamic mass


# ----------------------------------------------------------------------------- math helpers
def loguniform(rng, lo, hi):
    return float(math.exp(rng.uniform(math.log(lo), math.log(hi))))

def q_yaw(yaw):
    return np.asarray([0., math.sin(yaw/2), 0., math.cos(yaw/2)])

def q_random(rng):
    q = rng.normal(size=4); return q/np.linalg.norm(q)

def rotmat(q):
    x, y, z, w = q
    return np.asarray([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                       [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                       [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])

def verts_to_bullet(v):
    v = np.asarray(v, float); return np.stack([v[:, 0], -v[:, 2], v[:, 1]], 1)

def verts_from_bullet(v):
    v = np.asarray(v, float); return np.stack([v[:, 0], v[:, 2], -v[:, 1]], 1)


# ----------------------------------------------------------------------------- procedural meshes (local Y-up frame)
def tet_grid(nx: int, ny: int, nz: int):
    """Regular grid in [-0.5,0.5]^3, every cell split into 6 positively oriented tetrahedra (Kuhn)."""
    idx = lambda i, j, k: (i*ny+j)*nz+k
    pts = np.asarray([[i/(nx-1), j/(ny-1), k/(nz-1)] for i in range(nx) for j in range(ny) for k in range(nz)])-.5
    tets = []
    for i, j, k in itertools.product(range(nx-1), range(ny-1), range(nz-1)):
        corner = lambda v: idx(i+v[0], j+v[1], k+v[2])
        for perm in itertools.permutations(range(3)):
            v = [0, 0, 0]; path = [corner(v)]
            for axis in perm:
                v[axis] = 1; path.append(corner(v))
            tets.append(path)
    tets = np.asarray(tets, np.int64)
    a, b, c, d = (pts[tets[:, i]] for i in range(4))
    negative = np.einsum("ij,ij->i", np.cross(b-a, c-a), d-a) < 0
    tets[negative] = tets[negative][:, [0, 2, 1, 3]]
    return pts, tets

def cube_to_ball(pts):
    """Smooth map of the cube [-0.5,0.5]^3 onto the ball of diameter 1 (keeps the grid topology)."""
    q = 2*pts; x, y, z = q[:, 0], q[:, 1], q[:, 2]
    out = np.stack([x*np.sqrt(1-y*y/2-z*z/2+y*y*z*z/3), y*np.sqrt(1-z*z/2-x*x/2+z*z*x*x/3),
                    z*np.sqrt(1-x*x/2-y*y/2+x*x*y*y/3)], 1)
    return out/2

def tet_edges(tets):
    pairs = np.concatenate([tets[:, [a, b]] for a, b in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))])
    return np.unique(np.sort(pairs, 1), axis=0)

def tet_surface(pts, tets):
    """Boundary triangles (tet faces used once) with outward winding."""
    count = {}
    for tet in tets:
        for skip in range(4):
            face = [tet[i] for i in range(4) if i != skip]
            count.setdefault(tuple(sorted(face)), []).append((face, tet[skip]))
    faces = []
    for key in sorted(count):
        if len(count[key]) != 1: continue
        (a, b, c), opposite = count[key][0]
        if np.dot(np.cross(pts[b]-pts[a], pts[c]-pts[a]), pts[opposite]-pts[a]) > 0: b, c = c, b
        faces.append((a, b, c))
    return np.asarray(faces, np.int64)

def tet_volume(pts, tets):
    a, b, c, d = (pts[tets[:, i]] for i in range(4))
    return float(np.abs(np.einsum("ij,ij->i", np.cross(b-a, c-a), d-a)).sum()/6)

def cloth_grid(nx: int, nz: int):
    """Flat grid in the local xz plane, [-0.5,0.5]^2, alternating diagonals, normals +y."""
    pts = np.asarray([[i/(nx-1)-.5, 0., k/(nz-1)-.5] for i in range(nx) for k in range(nz)])
    tris = []
    for i in range(nx-1):
        for k in range(nz-1):
            a = i*nz+k; b = a+1; c = a+nz; d = c+1
            tris += [(a, b, c), (b, d, c)] if (i+k) % 2 else [(a, b, d), (a, d, c)]
    return pts, np.asarray(tris, np.int64)

def tri_edges(tris):
    pairs = np.concatenate([tris[:, [a, b]] for a, b in ((0, 1), (1, 2), (0, 2))])
    return np.unique(np.sort(pairs, 1), axis=0)

def build_mesh(spec: dict) -> dict:
    """spec: shape cube|ball|cloth, res (nx,ny,nz) / (nx,nz), size3 -> local vertices, cells, edges, faces, measure."""
    size = np.asarray(spec["size"], float)
    if spec["shape"] == "cloth":
        pts, tris = cloth_grid(*spec["res"]); pts = pts*size
        a, b, c = (pts[tris[:, i]] for i in range(3))
        return dict(pts=pts, tets=None, tris=tris, edges=tri_edges(tris), faces=tris,
                    measure=float(np.linalg.norm(np.cross(b-a, c-a), axis=1).sum()/2))
    pts, tets = tet_grid(*spec["res"])
    if spec["shape"] == "ball": pts = cube_to_ball(pts)
    pts = pts*size
    return dict(pts=pts, tets=tets, tris=None, edges=tet_edges(tets), faces=tet_surface(pts, tets),
                measure=tet_volume(pts, tets))

def write_vtk(path, pts, tets):
    with open(path, "w") as f:
        f.write(f"# vtk DataFile Version 2.0\nagatha soft body\nASCII\nDATASET UNSTRUCTURED_GRID\nPOINTS {len(pts)} double\n")
        f.writelines("%.9g %.9g %.9g\n" % tuple(v) for v in pts)
        f.write(f"CELLS {len(tets)} {5*len(tets)}\n")
        f.writelines("4 %d %d %d %d\n" % tuple(t) for t in tets)
        f.write(f"CELL_TYPES {len(tets)}\n"+"10\n"*len(tets))

def write_obj(path, pts, tris):
    with open(path, "w") as f:
        f.writelines("v %.9g %.9g %.9g\n" % tuple(v) for v in pts)
        f.writelines("f %d %d %d\n" % tuple(t+1) for t in tris)


# ----------------------------------------------------------------------------- scenario sampling
class Case:
    """Random table, one soft body spec and helpers to place bodies; everything Y-up."""

    def __init__(self, rng: np.random.Generator, kind: str):
        self.rng = rng; self.kind = kind
        self.w, self.d, self.h = float(rng.uniform(.9, 1.5)), float(rng.uniform(.65, 1.)), float(rng.uniform(.6, .9))
        self.gravity = float(rng.uniform(2, 15)) if rng.random() < .06 else 9.81
        self.support_friction = .5 if rng.random() < .6 else float(rng.uniform(.3, 1.))
        self.rigid = []
        self.soft = self._cloth() if kind in CLOTH_KINDS else self._volumetric()
        self.mesh = build_mesh(self.soft)

    def _volumetric(self):
        rng = self.rng
        if rng.random() < .55:
            shape = "cube"; size = rng.uniform(.06, .2, 3)
            if rng.random() < .35: size[:] = size[0]
            nmax = 5 if rng.random() < .35 else 4
            res = tuple(int(np.clip(round(s/(size.max()/(nmax-1)))+1, 3, 5)) for s in size)
            volume = float(np.prod(size))
        else:
            shape = "ball"; size = rng.uniform(.07, .2)*rng.uniform(.8, 1.2, 3)
            if rng.random() < .5: size[:] = size[0]
            n = 5 if rng.random() < .35 else 4; res = (n, n, n)
            volume = float(math.pi/6*np.prod(size))
        mass = float(np.clip(volume*rng.uniform(100, 1200), .03, 2.))
        mu = mass/volume*9.81*float(size.max())/loguniform(rng, *SAG_RANGE)
        return dict(shape=shape, size=size, res=res, mass=mass,
                    mu=mu, lam=mu*float(rng.uniform(1, 4)), neo_damping=loguniform(rng, .005, .1),
                    spring_k=0., spring_damping=0., bending_k=0., friction=float(rng.uniform(.3, 1.)),
                    pos=np.zeros(3), quat=np.asarray([0., 0, 0, 1]), linvel=np.zeros(3), angvel=np.zeros(3))

    def _cloth(self):
        rng = self.rng
        res = (int(rng.integers(8, 12)), int(rng.integers(8, 12)))
        size = np.asarray([rng.uniform(.2, .45), 0., rng.uniform(.2, .45)])
        mass = float(np.clip(size[0]*size[2]*rng.uniform(.4, 1.5), .03, .3))
        spring_k = mass*loguniform(rng, *CLOTH_K_PER_KG)
        # Bullet's mass-spring cloth is only stable with enough spring damping relative to the stiffness (measured:
        # k=100, damping 0.01 explodes); low damping drapes freely, high damping creeps like a heavy blanket.
        spring_damping = float(np.clip(spring_k*loguniform(rng, *CLOTH_DAMPING_PER_K), .015, .3))
        return dict(shape="cloth", size=size, res=res, mass=mass, mu=0., lam=0., neo_damping=0.,
                    spring_k=spring_k, spring_damping=spring_damping,
                    bending_k=loguniform(rng, .1, 10.), friction=float(rng.uniform(.3, 1.)),
                    pos=np.zeros(3), quat=np.asarray([0., 0, 0, 1]), linvel=np.zeros(3), angvel=np.zeros(3))

    # ---- placement helpers
    def world_verts(self):
        return self.mesh["pts"] @ rotmat(self.soft["quat"]).T + self.soft["pos"]

    def place_soft(self, x, z, bottom_y, quat):
        """Rotate the soft body and put its lowest vertex at bottom_y with its centre over (x, z)."""
        self.soft["quat"] = np.asarray(quat, float); self.soft["pos"] = np.zeros(3)
        v = self.world_verts()
        self.soft["pos"] = np.asarray([x-v[:, 0].mean(), bottom_y-v[:, 1].min(), z-v[:, 2].mean()])

    def soft_radius(self):
        v = self.world_verts(); return float(np.linalg.norm((v-v.mean(0))[:, [0, 2]], axis=1).max())

    def soft_top(self):
        return float(self.world_verts()[:, 1].max())

    def inner_xz(self, margin):
        rng = self.rng
        return (float(rng.uniform(-1, 1)*max(self.w/2-margin, .02)), float(rng.uniform(-1, 1)*max(self.d/2-margin, .02)))

    def box(self, size=None, static=False):
        rng = self.rng
        size = rng.uniform(.05, .2, 3) if size is None else np.asarray(size, float)
        mass = float(np.clip(np.prod(size)*rng.uniform(150, 2000), .05, 3.))
        friction = .5 if rng.random() < .4 else float(rng.uniform(.2, 1.))
        return dict(size=size, mass=mass, friction=friction, is_static=bool(static), pos=np.zeros(3),
                    quat=q_yaw(rng.uniform(-math.pi, math.pi) if rng.random() < .7 else 0.), linvel=np.zeros(3), angvel=np.zeros(3))


def _yaw_or_random(rng, p_yaw=.4):
    return q_yaw(rng.uniform(-math.pi, math.pi)) if rng.random() < p_yaw else q_random(rng)

def scenario_soft_drop(c: Case):
    rng = c.rng
    c.place_soft(0, 0, 0, _yaw_or_random(rng))
    x, z = c.inner_xz(c.soft_radius()+.03)
    c.place_soft(x, z, c.h+COLLISION_MARGIN+rng.uniform(.02, .4), c.soft["quat"])
    if rng.random() < .3:
        a = rng.uniform(0, 2*math.pi); s = rng.uniform(.1, .8); c.soft["linvel"] = np.asarray([s*math.cos(a), 0, s*math.sin(a)])
    if rng.random() < .3: c.soft["angvel"] = rng.normal(size=3)*rng.uniform(.5, 5)/math.sqrt(3)

def scenario_soft_rest(c: Case):
    rng = c.rng
    c.place_soft(0, 0, 0, q_yaw(rng.uniform(-math.pi, math.pi)))
    x, z = c.inner_xz(c.soft_radius()+.03)
    c.place_soft(x, z, c.h+COLLISION_MARGIN+.001, c.soft["quat"])

def scenario_soft_launch(c: Case):
    rng = c.rng
    c.place_soft(0, 0, 0, q_yaw(rng.uniform(-math.pi, math.pi)))
    r = c.soft_radius(); x, z = c.inner_xz(r+.05)
    c.place_soft(x, z, c.h+COLLISION_MARGIN+.002, c.soft["quat"])
    speed = rng.uniform(.3, 2.5)
    if rng.random() < .7:   # aim at a point on the table so that many launches stay on it
        tx, tz = c.inner_xz(r); direction = np.asarray([tx-x, 0, tz-z])
        if np.linalg.norm(direction) < 1e-3: direction = np.asarray([1., 0, 0])
    else:
        a = rng.uniform(0, 2*math.pi); direction = np.asarray([math.cos(a), 0, math.sin(a)])
    direction /= np.linalg.norm(direction)
    c.soft["linvel"] = direction*speed
    if rng.random() < .3: c.soft["linvel"][1] = rng.uniform(.3, 1.5)
    if c.soft["shape"] == "ball" and rng.random() < .5:   # rolling without slipping
        c.soft["angvel"] = np.cross([0, 1, 0], direction)*speed/(c.soft["size"][1]/2)

def scenario_soft_edge_fall(c: Case):
    rng = c.rng
    c.place_soft(0, 0, 0, q_yaw(rng.uniform(-math.pi, math.pi)))
    r = c.soft_radius()
    axis = 0 if rng.random() < .5 else 2; sign = 1 if rng.random() < .5 else -1
    half = (c.w/2, 0, c.d/2)
    pos = [0., 0.]; other = 2-axis
    pos_other = float(rng.uniform(-1, 1)*max(half[other]-r-.03, .02))
    if rng.random() < .65:
        edge_distance = rng.uniform(.02, .3)
        along = sign*(half[axis]-edge_distance)
        velocity = np.zeros(3); velocity[axis] = sign*rng.uniform(.6, 2.5); velocity[other] = rng.uniform(-.3, .3)
        c.soft["linvel"] = velocity
    else:
        along = sign*(half[axis]+rng.uniform(-.1, .6)*r)
    xz = {axis: along, other: pos_other}
    c.place_soft(xz[0], xz[2], c.h+COLLISION_MARGIN+.002, c.soft["quat"])

def scenario_box_on_soft(c: Case):
    rng = c.rng
    c.place_soft(0, 0, 0, q_yaw(rng.uniform(-math.pi, math.pi)))
    r = c.soft_radius(); x, z = c.inner_xz(r+.08)
    c.place_soft(x, z, c.h+COLLISION_MARGIN+.001, c.soft["quat"])
    y = c.soft_top()+COLLISION_MARGIN
    for _ in range(1 if rng.random() < .7 else 2):
        b = c.box(); y += rng.uniform(.03, .2)
        b["mass"] = float(np.clip(c.soft["mass"]*rng.uniform(.2, 1.5), .03, 2.))
        offset = rng.uniform(-.5, .5, 2)*r
        b["pos"] = np.asarray([x+offset[0], y+b["size"][1]/2, z+offset[1]]); y += b["size"][1]
        if rng.random() < .2: b["quat"] = q_random(rng); b["pos"][1] += np.linalg.norm(b["size"])/2-b["size"][1]/2; y += np.linalg.norm(b["size"])
        c.rigid.append(b)

def scenario_soft_on_boxes(c: Case):
    rng = c.rng
    count = int(rng.integers(1, 4)); placed = []
    for i in range(count):
        b = c.box(static=rng.random() < .12)
        if placed and rng.random() < .3 and i == 1:   # stack on the first box
            base = placed[0]; b["size"][[0, 2]] = np.minimum(b["size"][[0, 2]], base["size"][[0, 2]])
            b["quat"] = base["quat"]; b["pos"] = base["pos"]+[0, base["size"][1]/2+b["size"][1]/2+.001, 0]
        else:
            for _ in range(30):
                radius = float(np.linalg.norm(b["size"][[0, 2]])/2)
                x, z = c.inner_xz(radius+.05) if not placed else (placed[0]["pos"][0]+rng.uniform(-.3, .3), placed[0]["pos"][2]+rng.uniform(-.3, .3))
                if abs(x) > c.w/2-radius or abs(z) > c.d/2-radius: continue
                if all(math.hypot(x-o["pos"][0], z-o["pos"][2]) > radius+np.linalg.norm(o["size"][[0, 2]])/2+.005 for o in placed): break
            else: continue
            b["pos"] = np.asarray([x, c.h+b["size"][1]/2+.001, z])
        placed.append(b)
    c.rigid += placed
    target = placed[int(rng.integers(len(placed)))] if placed else None
    c.place_soft(0, 0, 0, _yaw_or_random(rng, .5))
    r = c.soft_radius()
    if target is None: x, z = c.inner_xz(r+.03); top = c.h
    else:
        offset = rng.uniform(-.6, .6, 2)*(r+np.linalg.norm(target["size"][[0, 2]])/2)*.6
        x, z = target["pos"][0]+offset[0], target["pos"][2]+offset[1]
        top = max(b["pos"][1]+b["size"][1]/2 for b in placed)
    c.place_soft(x, z, top+COLLISION_MARGIN+rng.uniform(.03, .3), c.soft["quat"])
    if rng.random() < .25:
        a = rng.uniform(0, 2*math.pi); s = rng.uniform(.1, .8); c.soft["linvel"] = np.asarray([s*math.cos(a), 0, s*math.sin(a)])

def scenario_cloth_drape_box(c: Case):
    rng = c.rng
    b = c.box(size=rng.uniform(.06, .2, 3), static=rng.random() < .5)
    x, z = c.inner_xz(.3)
    b["pos"] = np.asarray([x, c.h+b["size"][1]/2+.001, z]); c.rigid.append(b)
    offset = rng.uniform(-.08, .08, 2)
    c.place_soft(x+offset[0], z+offset[1], b["pos"][1]+b["size"][1]/2+COLLISION_MARGIN+rng.uniform(.02, .15), q_yaw(rng.uniform(-math.pi, math.pi)))

def scenario_cloth_drape_edge(c: Case):
    rng = c.rng
    yaw = rng.uniform(-.4, .4)+(math.pi/2 if rng.random() < .5 else 0)
    c.place_soft(0, 0, 0, q_yaw(yaw))
    v = c.world_verts()
    axis = 0 if rng.random() < .5 else 2; sign = 1 if rng.random() < .5 else -1; other = 2-axis
    half = (c.w/2, 0, c.d/2); extent = v[:, axis].max()-v[:, axis].min()
    along = sign*(half[axis]+(rng.uniform(.25, .65)-.5)*extent)
    xz = {axis: along, other: float(rng.uniform(-1, 1)*max(half[other]-.3, .02))}
    c.place_soft(xz[0], xz[2], c.h+COLLISION_MARGIN+rng.uniform(.01, .1), c.soft["quat"])

def scenario_cloth_drop(c: Case):
    rng = c.rng
    tilt = rotmat(q_yaw(rng.uniform(-math.pi, math.pi)))
    axis = tilt @ np.asarray([1., 0, 0]); angle = rng.uniform(0, .5)
    quat = np.asarray([*(axis*math.sin(angle/2)), math.cos(angle/2)])
    c.place_soft(0, 0, 0, quat)
    x, z = c.inner_xz(c.soft_radius()+.05)
    c.place_soft(x, z, c.h+COLLISION_MARGIN+rng.uniform(.02, .25), quat)
    if rng.random() < .5:
        a = rng.uniform(0, 2*math.pi); s = rng.uniform(.1, 1.); c.soft["linvel"] = np.asarray([s*math.cos(a), 0, s*math.sin(a)])

SCENARIOS = dict(soft_drop=scenario_soft_drop, soft_rest=scenario_soft_rest, soft_launch=scenario_soft_launch,
                 soft_edge_fall=scenario_soft_edge_fall, box_on_soft=scenario_box_on_soft, soft_on_boxes=scenario_soft_on_boxes,
                 cloth_drape_box=scenario_cloth_drape_box, cloth_drape_edge=scenario_cloth_drape_edge, cloth_drop=scenario_cloth_drop)

def make_case(rng: np.random.Generator, kind: str | None = None, kind_weights: dict | None = None) -> Case:
    weights = kind_weights or KIND_WEIGHTS
    if kind is None:
        names = list(weights); w = np.asarray([weights[k] for k in names], float)
        kind = names[int(rng.choice(len(names), p=w/w.sum()))]
    case = Case(rng, kind)
    SCENARIOS[kind](case)
    return case


# ----------------------------------------------------------------------------- simulation
try:
    import ctypes
    _LIBC = ctypes.CDLL(None)
except Exception:   # pragma: no cover
    _LIBC = None

@contextlib.contextmanager
def _quiet_stdout():
    """Bullet prints 'Nodes/Links/Faces/Tetras' from C when loading a .vtk; silence fd 1 for that call."""
    sys.stdout.flush()
    saved = os.dup(1); null = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(null, 1); yield
    finally:
        try: _LIBC.fflush(None)   # C stdio buffers the text when stdout is a pipe; drop it into /dev/null too
        except Exception: pass
        os.dup2(saved, 1); os.close(saved); os.close(null)

def _rigid_box(c, size, pos, mass, quat=(0, 0, 0, 1), friction=.5):
    """Same construction as agatha.physics.World._box (Y-up arguments)."""
    shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=(size[0]/2, size[2]/2, size[1]/2), physicsClientId=c)
    body = p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=shape, basePosition=to_bullet(pos),
                             baseOrientation=quat_to_bullet(quat), physicsClientId=c)
    p.changeDynamics(body, -1, lateralFriction=friction, restitution=0, linearDamping=.04, angularDamping=.04, physicsClientId=c)
    return body

def _load_soft(c, case: Case, directory: str) -> int:
    soft, mesh = case.soft, case.mesh
    world = verts_to_bullet(case.world_verts())   # rotation and position are baked into the mesh file
    common = dict(basePosition=[0, 0, 0], mass=soft["mass"], collisionMargin=COLLISION_MARGIN,
                  frictionCoeff=soft["friction"], useSelfCollision=0, physicsClientId=c)
    if soft["shape"] == "cloth":
        path = os.path.join(directory, "soft.obj"); write_obj(path, world, mesh["tris"])
        kwargs = dict(useMassSpring=1, useBendingSprings=1, springElasticStiffness=soft["spring_k"],
                      springDampingStiffness=soft["spring_damping"], springDampingAllDirections=1,
                      springBendingStiffness=soft["bending_k"], useFaceContact=1)
    else:
        path = os.path.join(directory, "soft.vtk"); write_vtk(path, world, mesh["tets"])
        kwargs = dict(useNeoHookean=1, NeoHookeanMu=soft["mu"], NeoHookeanLambda=soft["lam"],
                      NeoHookeanDamping=soft["neo_damping"])
    with _quiet_stdout():
        body = p.loadSoftBody(path, **common, **kwargs)
    os.unlink(path)
    return body

def _read_verts(c, body):
    return verts_from_bullet(p.getMeshData(body, flags=p.MESH_DATA_SIMULATION_MESH, physicsClientId=c)[1])

def box_penetration(points, centre, rot, half):
    """Depth (>= 0) of points inside an oriented box; points [...,3], rot maps local -> world."""
    local = (points-centre) @ rot
    return np.clip((half-np.abs(local)).min(-1), 0, None)

def max_penetration(verts, mask, context, states, props, rigid_mask):
    """Deepest penetration of real soft vertices into floor / support slab / rigid boxes over all frames."""
    v = verts[:, mask]; w, d, h = context[:3]
    depth = np.clip(-v[..., 1], 0, None).max()
    slab = box_penetration(v, np.asarray([0, h-SUPPORT_THICKNESS/2, 0]), np.eye(3), np.asarray([w/2, SUPPORT_THICKNESS/2, d/2]))
    depth = max(depth, slab.max())
    for i in np.flatnonzero(rigid_mask):
        for t in range(len(v)):
            depth = max(depth, box_penetration(v[t], states[t, i, :3], rotmat(states[t, i, 3:7]), props[i, :3]/2).max())
    return float(depth)

def simulate_case(case: Case, steps: int = 60, return_reason: bool = False):
    """Run one episode; returns the episode dict, or None (or (None, reason)) when it is dropped."""
    soft, mesh = case.soft, case.mesh
    n = len(mesh["pts"]); edges, faces = mesh["edges"], mesh["faces"]
    assert n <= V_MAX and len(edges) <= M_MAX and len(faces) <= F_MAX and len(case.rigid) <= N_RIGID
    verts = np.zeros((steps+1, V_MAX, 3), np.float32); vel = np.zeros((steps+1, V_MAX, 3), np.float32)
    states = np.zeros((steps+1, N_RIGID, STATE_DIM), np.float32); states[..., 6] = 1
    initial = case.world_verts()
    c = p.connect(p.DIRECT)
    try:
        p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD, physicsClientId=c)
        p.setPhysicsEngineParameter(fixedTimeStep=DT, numSolverIterations=SOLVER_ITERATIONS, deterministicOverlappingPairs=1,
                                   sparseSdfVoxelSize=SPARSE_SDF_VOXEL, physicsClientId=c)
        p.setGravity(0, 0, -case.gravity, physicsClientId=c)
        _rigid_box(c, (2*FLOOR_HALF, .1, 2*FLOOR_HALF), (0, -.05, 0), 0)
        _rigid_box(c, (case.w, SUPPORT_THICKNESS, case.d), (0, case.h-SUPPORT_THICKNESS/2, 0), 0, friction=case.support_friction)
        bodies = []
        for b in case.rigid:
            body = _rigid_box(c, b["size"], b["pos"], 0 if b["is_static"] else b["mass"], b["quat"], b["friction"])
            p.resetBaseVelocity(body, to_bullet(b["linvel"]), to_bullet(b["angvel"]), physicsClientId=c)
            bodies.append(body)
        with tempfile.TemporaryDirectory(prefix=f"agatha_soft_{os.getpid()}_") as directory:
            sb = _load_soft(c, case, directory)
        loaded = _read_verts(c, sb)
        if len(loaded) != n or not np.allclose(loaded, initial, atol=1e-4):
            raise RuntimeError("Bullet changed the soft-body vertex order or count")
        if np.any(soft["linvel"]) or np.any(soft["angvel"]):
            p.resetBaseVelocity(sb, to_bullet(soft["linvel"]), to_bullet(soft["angvel"]), physicsClientId=c)

        def record(t, previous=None):
            current = _read_verts(c, sb); verts[t, :n] = current
            if previous is not None: vel[t, :n] = (current-previous)/DT
            for i, body in enumerate(bodies):
                pos, quat = p.getBasePositionAndOrientation(body, physicsClientId=c)
                lin, ang = p.getBaseVelocity(body, physicsClientId=c)
                states[t, i] = [*(pos[0], pos[2], -pos[1]), *quat_from_bullet(quat), *(lin[0], lin[2], -lin[1]), *(ang[0], ang[2], -ang[1])]
            if t and len(bodies):
                flip = np.sum(states[t-1, :len(bodies), 3:7]*states[t, :len(bodies), 3:7], axis=1) < 0
                states[t, :len(bodies)][flip, 3:7] *= -1

        record(0)
        vel[0, :n] = soft["linvel"]+np.cross(soft["angvel"], initial-initial.mean(0))
        for t in range(1, steps+1):
            for _ in range(SUBSTEPS-1): p.stepSimulation(physicsClientId=c)
            previous = _read_verts(c, sb)
            p.stepSimulation(physicsClientId=c)
            record(t, previous)
    finally:
        p.disconnect(c)

    kind_class = "cloth" if soft["shape"] == "cloth" else "volumetric"
    rest_len = np.linalg.norm(mesh["pts"][edges[:, 0]]-mesh["pts"][edges[:, 1]], axis=1)
    reason = None
    real = verts[:, :n]
    if not (np.isfinite(verts).all() and np.isfinite(vel).all() and np.isfinite(states).all()): reason = "nonfinite"
    elif (np.abs(real[..., [0, 2]]).max() > 5 or real[..., 1].max() > 5 or np.abs(states[..., :3]).max() > 5
          or states[..., 1].min() < -.5): reason = "bounds"
    elif real[..., 1].min() < -.03: reason = "below_floor"
    elif np.linalg.norm(vel, axis=-1).max() > 25: reason = "speed"
    if reason is None:
        length = np.linalg.norm(real[:, edges[:, 0]]-real[:, edges[:, 1]], axis=-1)
        deformation = float(np.abs(length/rest_len-1).max())
        if deformation > STRAIN_LIMIT[kind_class]: reason = "strain"
    volume_ratio = 1.
    if reason is None and mesh["tets"] is not None:
        tets = mesh["tets"]
        signed = lambda x: np.einsum("...ij,...ij->...i", np.cross(x[..., tets[:, 1], :]-x[..., tets[:, 0], :],
                                     x[..., tets[:, 2], :]-x[..., tets[:, 0], :]), x[..., tets[:, 3], :]-x[..., tets[:, 0], :])
        volume_ratio = float((signed(real.astype(float))/signed(mesh["pts"])).min())
        if volume_ratio <= 0: reason = "inverted"
    mask = np.zeros(V_MAX, bool); mask[:n] = True
    rigid_mask = np.zeros(N_RIGID, bool); rigid_mask[:len(case.rigid)] = True
    props = np.zeros((N_RIGID, PROPS_DIM), np.float32); props[:, 3] = 1
    for i, b in enumerate(case.rigid):
        props[i] = [*b["size"], b["mass"], b["friction"], float(b["is_static"]), 0, 0, 0]
    context = np.asarray([case.w, case.d, case.h, case.gravity, 0, 0, 0, 1, case.support_friction, 0], np.float32)
    if reason is None:
        penetration = max_penetration(verts, mask, context, states, props, rigid_mask)
        if penetration > TUNNEL_DEPTH: reason = "tunnel"
    if reason is None:
        # Nothing in these scenes adds energy (zero restitution, bodies start undeformed), so the mechanical energy
        # of the soft nodes plus the rigid boxes' translation must never rise above its initial value.
        dynamic_mass = np.where(rigid_mask & (props[:, 5] < .5), props[:, 3], 0.).astype(float)
        speed2 = (vel[:, :n].astype(float)**2).sum(-1)
        energy = soft["mass"]*(speed2.mean(1)/2+case.gravity*real[..., 1].astype(float).mean(1))
        energy = energy+((.5*(states[..., 7:10].astype(float)**2).sum(-1)+case.gravity*states[..., 1])*dynamic_mass).sum(1)
        energy_gain = float((energy.max()-energy[0])/(soft["mass"]+dynamic_mass.sum()))
        if energy_gain > ENERGY_GAIN_LIMIT: reason = "energy"
    if reason is not None:
        return (None, reason) if return_reason else None

    centre, half = STIFF_NORM[kind_class]
    raw_stiffness = soft["spring_k"] if kind_class == "cloth" else soft["mu"]
    soft_props = np.asarray([soft["mass"], (math.log10(raw_stiffness)-centre)/half,
                             soft["spring_damping"] if kind_class == "cloth" else soft["neo_damping"],
                             soft["friction"], float(kind_class == "cloth"), mesh["measure"]], np.float32)
    soft_raw = np.asarray([soft["mu"], soft["lam"], soft["neo_damping"], soft["spring_k"], soft["spring_damping"],
                           soft["bending_k"], *soft["size"], COLLISION_MARGIN], np.float32)
    rest_edges = np.full((M_MAX, 2), -1, np.int16); rest_edges[:len(edges)] = edges
    rest_length = np.zeros(M_MAX, np.float32); rest_length[:len(edges)] = rest_len
    face_array = np.full((F_MAX, 3), -1, np.int16); face_array[:len(faces)] = faces
    dynamic = rigid_mask & (props[:, 5] < .5)
    rigid_speed = np.maximum(np.linalg.norm(states[..., 7:10], axis=-1), .1*np.linalg.norm(states[..., 10:13], axis=-1))
    rest = (np.linalg.norm(vel, axis=-1).max(1) < REST_SPEED) & ((rigid_speed < REST_SPEED) | ~dynamic).all(1)
    final_com = real[-1].mean(0).astype(np.float32)
    episode = dict(
        verts=verts, vert_vel=vel, vert_mask=mask, rest_edges=rest_edges, rest_len=rest_length, faces=face_array,
        soft_props=soft_props, soft_raw=soft_raw, soft_shape=soft["shape"], n_verts=np.int16(n),
        n_edges=np.int16(len(edges)), n_faces=np.int16(len(faces)), states=states, props=props, mask=rigid_mask,
        context=context, kind=case.kind, final_com=final_com, max_deformation=np.float32(deformation),
        max_penetration=np.float32(penetration), min_volume_ratio=np.float32(volume_ratio), rest=rest, fell=bool(final_com[1] < case.h-.1))
    return (episode, None) if return_reason else episode


def episode_rng(base_seed: int, split: str, index: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([int(base_seed)+SPLIT_SEED_OFFSET[split], SPLITS.index(split), int(index)]))

_ACTIVE_WEIGHTS = None   # set by generate() before the worker pool forks

def generate_episode(base_seed: int, split: str, index: int, steps: int = 60, kind: str | None = None):
    """-> (episode | None, drop reason | None, scenario kind, simulation seconds)."""
    case = make_case(episode_rng(base_seed, split, index), kind, _ACTIVE_WEIGHTS)
    start = time.perf_counter()
    episode, reason = simulate_case(case, steps, return_reason=True)
    return episode, reason, case.kind, time.perf_counter()-start


def _chunk(task):
    base_seed, split, shard, indices, steps = task
    out = []
    for index in indices:
        episode, reason, kind, seconds = generate_episode(base_seed, split, index, steps)
        if episode is not None: episode["episode_index"] = index
        out.append((episode, reason, kind, seconds))
    return split, shard, indices[0], out


STACK_KEYS = ("verts", "vert_vel", "vert_mask", "rest_edges", "rest_len", "faces", "soft_props", "soft_raw", "states",
              "props", "mask", "context", "final_com", "max_deformation", "max_penetration", "min_volume_ratio", "rest", "fell")
ARRAY_KEYS = STACK_KEYS+("soft_shape", "n_verts", "n_edges", "n_faces", "kind", "episode_index")

def _write_shard(path: Path, episodes: list[dict]):
    arrays = {k: np.stack([e[k] for e in episodes]) for k in STACK_KEYS}
    arrays["fell"] = arrays["fell"].astype(bool)
    arrays["kind"] = np.asarray([e["kind"] for e in episodes], dtype="<U24")
    arrays["soft_shape"] = np.asarray([e["soft_shape"] for e in episodes], dtype="<U8")
    for k in ("n_verts", "n_edges", "n_faces"): arrays[k] = np.asarray([e[k] for e in episodes], np.int16)
    arrays["episode_index"] = np.asarray([e["episode_index"] for e in episodes], np.int64)
    np.savez_compressed(path, **arrays)


SUMMARY_KEYS = ("kind", "soft_shape", "n_verts", "n_edges", "mask", "props", "context", "soft_props", "max_deformation",
                "max_penetration", "min_volume_ratio", "rest", "fell", "seconds")

def _stats(episodes: list[dict], steps: int) -> dict:
    if not episodes: return dict(episodes=0)
    kind = np.asarray([e["kind"] for e in episodes]); kinds = sorted(set(kind))
    shape = np.asarray([e["soft_shape"] for e in episodes])
    get = lambda key: np.asarray([e[key] for e in episodes])
    deformation, rest, seconds, fell = get("max_deformation"), get("rest"), get("seconds"), get("fell")
    by_kind = lambda values: {k: float(values[kind == k].mean()) for k in kinds}
    return dict(
        episodes=len(episodes), transitions=len(episodes)*steps, vertex_transitions=int(get("n_verts").sum())*steps,
        kinds={k: int((kind == k).sum()) for k in kinds}, shapes={s: int((shape == s).sum()) for s in sorted(set(shape))},
        mean_vertices=float(get("n_verts").mean()), mean_edges=float(get("n_edges").mean()),
        mean_rigid_boxes=float(get("mask").sum(1).mean()), with_rigid_fraction=float(get("mask").any(1).mean()),
        mean_max_strain=float(deformation.mean()), median_max_strain=float(np.median(deformation)),
        p95_max_strain=float(np.quantile(deformation, .95)), mean_max_strain_by_kind=by_kind(deformation),
        rest_frame_fraction=float(rest.mean()), rest_frame_fraction_by_kind=by_kind(rest.mean(1)),
        final_frame_rest_fraction=float(rest[:, -1].mean()),
        fell_fraction=float(fell.mean()), fell_fraction_by_kind=by_kind(fell.astype(float)),
        mean_max_penetration_m=float(get("max_penetration").mean()), p99_max_penetration_m=float(np.quantile(get("max_penetration"), .99)),
        mean_min_volume_ratio_volumetric=float(get("min_volume_ratio")[shape != "cloth"].mean()) if (shape != "cloth").any() else None,
        gravity_changed_fraction=float(np.mean(np.abs(get("context")[:, 3]-9.81) > 1e-4)),
        sim_seconds_per_episode=float(seconds.mean()), sim_seconds_per_episode_by_kind=by_kind(seconds),
    )


def generate(out: Path, train_episodes: int = 6000, val_episodes: int = 300, test_episodes: int = 800, steps: int = 60,
             workers: int = 40, shard_size: int = 250, seed: int = 4041, chunk: int = 5, quiet: bool = False,
             kind_weights: dict | None = None) -> dict:
    global _ACTIVE_WEIGHTS
    if kind_weights is not None:
        if set(kind_weights)-set(SCENARIOS) or not kind_weights or min(kind_weights.values()) < 0 or sum(kind_weights.values()) <= 0:
            raise ValueError(f"bad kind weights {kind_weights}")
        kind_weights = {k: float(v) for k, v in kind_weights.items()}
    _ACTIVE_WEIGHTS = kind_weights
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
    kept = {split: [] for split in SPLITS}; done = 0; total = sum(counts.values()); writes = []
    dropped = {split: 0 for split in SPLITS}; reasons = {}; dropped_kinds = {}; attempted_kinds = {}; dropped_seconds = 0.

    workers = max(1, min(workers, len(tasks) or 1))
    context = mp.get_context("fork")
    with ThreadPoolExecutor(4) as writer:
        pool = context.Pool(workers) if workers > 1 else None
        try:
            results = pool.imap_unordered(_chunk, tasks) if pool else map(_chunk, tasks)
            last = start
            for split, shard, first, items in results:
                buffers[split, shard][first] = items; pending[split, shard] -= 1; done += len(items)
                if pending[split, shard] == 0:
                    pieces = buffers.pop((split, shard))
                    good = []
                    for key in sorted(pieces):
                        for episode, reason, kind, seconds in pieces[key]:
                            attempted_kinds[kind] = attempted_kinds.get(kind, 0)+1
                            if episode is None:
                                dropped[split] += 1; reasons[reason] = reasons.get(reason, 0)+1; dropped_seconds += seconds
                                dropped_kinds[kind] = dropped_kinds.get(kind, 0)+1
                            else:
                                episode["seconds"] = seconds; good.append(episode)
                    if good: writes.append(writer.submit(_write_shard, out/split/f"shard_{shard:05d}.npz", good))
                    kept[split] += [{k: e[k] for k in SUMMARY_KEYS} for e in good]
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
        format=FORMAT, step_s=STEP, substeps_per_step=SUBSTEPS, steps=steps, max_vertices=V_MAX, max_edges=M_MAX,
        max_faces=F_MAX, max_rigid=N_RIGID, seed=seed, split_seeds={s: seed+SPLIT_SEED_OFFSET[s] for s in SPLITS},
        seeding="np.random.SeedSequence([seed+split_offset, split_index, episode_index]) per episode",
        simulator=dict(engine="PyBullet deformable world (RESET_USE_DEFORMABLE_WORLD)", fixed_time_step=DT,
                       solver_iterations=SOLVER_ITERATIONS, sparse_sdf_voxel_size=SPARSE_SDF_VOXEL,
                       collision_margin=COLLISION_MARGIN, volumetric="NeoHookean tetrahedra (.vtk), no mass-spring",
                       cloth="mass-spring + bending springs (.obj), useFaceContact", self_collision=False,
                       friction="Bullet multiplies the two bodies' friction coefficients (soft frictionCoeff x rigid lateralFriction)",
                       vertex_read="p.getMeshData(body, flags=p.MESH_DATA_SIMULATION_MESH)"),
        stiffness_normalisation={k: f"(log10({'springElasticStiffness' if k == 'cloth' else 'NeoHookeanMu'}) - {c}) / {h}"
                                 for k, (c, h) in STIFF_NORM.items()},
        requested=counts, counts={s: len(kept[s]) for s in SPLITS}, dropped=dropped, drop_reasons=reasons,
        dropped_by_kind=dropped_kinds, attempted_by_kind=attempted_kinds, drop_rules=DROP_RULES,
        drop_rate=sum(dropped.values())/max(total, 1), shard_size=shard_size,
        episodes=len(everything), transitions=len(everything)*steps,
        seconds=seconds, simulation_seconds=simulated, workers=workers, episodes_per_second=total/max(simulated, 1e-9),
        kind_weights=KIND_WEIGHTS if kind_weights is None else kind_weights, ranges=RANGES,
        rest_definition=f"every soft vertex slower than {REST_SPEED} m/s and every dynamic rigid box slower than {REST_SPEED} m/s and {10*REST_SPEED} rad/s",
        splits={s: _stats(kept[s], steps) for s in SPLITS}, overall=_stats(everything, steps),
    )
    (out/"meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("data/soft_v1"))
    parser.add_argument("--train-episodes", type=int, default=6000)
    parser.add_argument("--val-episodes", type=int, default=300)
    parser.add_argument("--test-episodes", type=int, default=800)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--workers", type=int, default=40)
    parser.add_argument("--shard-size", type=int, default=250)
    parser.add_argument("--seed", type=int, default=4041)
    parser.add_argument("--kind-weights", type=json.loads, default=None,
                        help='JSON {kind: weight} overriding the default mix, e.g. \'{"soft_drop": 1, "cloth_drop": 1}\'')
    args = parser.parse_args()
    meta = generate(args.out, args.train_episodes, args.val_episodes, args.test_episodes, args.steps, args.workers,
                    args.shard_size, args.seed, kind_weights=args.kind_weights)
    brief = {k: meta[k] for k in ("counts", "dropped", "drop_rate", "drop_reasons", "dropped_by_kind", "attempted_by_kind",
                                  "transitions", "seconds", "workers", "episodes_per_second")}
    brief["overall"] = meta["overall"]
    print(json.dumps(brief, indent=2), flush=True)


if __name__ == "__main__":
    main()
