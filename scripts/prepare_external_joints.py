"""RBO articulated-object mocap recordings -> APPROXIMATE joints_v1 shards + kinematic-consistency report.

    # (optional) fetch more recordings / models: streams the Zenodo archive, keeps ONLY the pose CSVs
    .venv/bin/python -I scripts/prepare_external_joints.py --download treasurebox treasurebox07
    # convert + report (CPU, a few seconds)
    CUDA_VISIBLE_DEVICES="" .venv/bin/python -I scripts/prepare_external_joints.py \
        --external data/external/rbo-articulated --out data/external_v2 --report models/external_joints_kinematics.json

Source: "The RBO Dataset of Articulated Objects and Interactions" (Martin-Martin, Eppner, Brock,
arXiv:1806.06465), CC BY 4.0, https://tu-rbo.github.io/articulated-objects/ , data at
https://zenodo.org/records/1036660 .  See DATA_LICENSES.md.

Output: <out>/rbo_articulated_joints/test/shard_*.npz (+ meta.json) in the joint data layout read by
scripts/eval_joints.py / scripts/train_joint_ddp.py: the box keys of scripts/generate_contact_data.py
FORMAT with N = 12 body slots, plus joints [E,12,18], joint_mask [E,12], joint_pos [E,T+1,12], group
[E,12] (agatha.joint_gnn joint row), plus the EXTRA_FORMAT keys below.

THIS DATA IS HAND-DRIVEN.  Every motion in the recordings is caused by a person (hand, or a stick
with a force/torque sensor).  Those forces are not part of the state, so a rollout of a passive
dynamics model from t = 0 CANNOT be scored against these trajectories (every episode has
driven = True).  What the shards and the report are good for:
  (a) kinematic consistency: do the measured part poses satisfy the joint rows (anchor gap, axis
      error, joint coordinate vs. the dataset's own joint_states) - computed with
      agatha.joint_gnn.joint_violation / joint_coordinates, i.e. a check of those conventions on
      real data;
  (b) free-motion segments (object moving with no hand on it).  They are searched for and counted;
      a segment only counts as USABLE if the absence of contact is verified by the F/T sensor.

Untrusted input: only csv, tarfile (member read, no extraction of code), xml.etree and regular
expressions are used.  The *_out.yaml files of the model archives contain python-object YAML tags
and are never opened.

APPROXIMATIONS (keep in sync with docs/part-c-external-data.md, section 11 'Exa-discovered articulated data')
  R1  Each rigid part is replaced by ONE box: the bounding box of its scan mesh (.dae vertices placed
      with the URDF visual origin).  Hollow / concave parts (cabinet carcass, drawers, microwave
      body) become solid boxes, so the boxes of jointed parts overlap massively (joint_gnn ignores
      contact between jointed bodies; any other box-box reasoning on them is meaningless).
  R2  Box orientation: the root box is the root link at the first valid frame, turned so that the
      box y axis is the world up axis and the yaw minimises its footprint (assumes the object
      stands upright on its surface at the start).  joint_gnn's zero convention (child orientation
      == parent orientation at coordinate 0) fixes every child box frame: prismatic children share
      the parent's frame; a hinge child uses the parent's frame at the URDF angle where the pair is
      most compact (lid / door closed), stored as joint_zero.  The row's coordinate is therefore
      URDF value - joint_zero, and a hinge box is only as tight as the closed pose is box-aligned.
  R3  '/map' is treated as Z-up (inferred from the data, not documented) and mapped to Y-up by the
      proper rotation (x, y, z) -> (x, z, -y).
  R4  The resting surface is not given.  Its height is the median lowest corner of the root box;
      it becomes an Agatha support slab with top at 0.75 m (all heights shifted), centred under
      the motion, of ASSUMED size (>= 1.2 m x 0.8 m).  Floor-standing furniture gets the same slab.
  R5  100 Hz mocap is resampled to 20 Hz (linear / slerp); velocities are central finite
      differences on the 20 Hz grid (the dataset's velocity columns are all zero).
  R6  Masses are NOT in the dataset: box volume x 300 kg/m^3 clipped to the 0.05-3 kg training
      range.  Friction 0.5, restitution 0, joint damping 0 are placeholders.  Nothing is static.
  R7  URDF joint limits are placeholders (+-3.1416); rows store 'no limit' (hinge -pi..pi,
      prismatic -10..10 m).  The observed coordinate range is in meta.json.
  R8  The joint origin / axis of each URDF were estimated by the dataset authors from mocap of the
      same marker-set configuration, so the anchor-gap / axis statistics measure model-fit residual
      plus mocap noise, not an independent ground truth.
  R9  rest / contact_support are geometric guesses from the boxes; contact_floor / contact_object /
      fell are all False; tipped uses the relative rule of prepare_external_data (root links only).
"""
import argparse
import csv
import json
import math
import re
import sys
import tarfile
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(Path(__file__).resolve().parent)):
    if _p not in sys.path: sys.path.insert(0, _p)
import numpy as np
import prepare_external_data as pe

DATASET = 'rbo_articulated_joints'
ZENODO = 'https://zenodo.org/api/records/1036660/files/{}/content'
STEP = .05
N_SLOTS, J_SLOTS, JOINT_DIM = 12, 12, 18
SUPPORT_HEIGHT, MIN_SUPPORT = .75, (1.2, .8)
DENSITY, MASS_RANGE = 300., (.05, 3.)
MAX_GAP_S = .035                      # two mocap samples further apart than this are a dropout
MOVE_MIN = dict(hinge=.05, prismatic=.01)       # rad/s, m/s: 'the joint is moving'
LAUNCH_MIN = dict(hinge=.3, prismatic=.05)      # a run must reach this speed to be a free-motion candidate
SMOOTH_S, FREE_MIN_S, ABRUPT_S = .15, .30, .20  # smoothing window, shortest candidate, "stops at once" (incl. smoothing)
STICK_STILL = .02                     # m/s: the mocap-tracked F/T stick is not moving
FT_FREE_N = .5                        # |F - F_idle| below this = nothing touches the F/T stick (idle noise is ~0.2-0.8 N)
FIT_MAX_GAP_M, FIT_MAX_AXIS_RAD = .05, math.radians(15)   # median residual above this: the URDF does not describe the recording
NOTE = ('APPROXIMATE conversion of REAL, HAND-DRIVEN mocap recordings (RBO articulated objects, CC BY 4.0). External forces are '
        'not observed: valid for kinematic-consistency checks only, NOT for scoring free dynamics.')
EXTRA_FORMAT = dict(
    joints='float32 [E,12,18] agatha.joint_gnn row: parent, child, one-hot(hinge, ball, prismatic, fixed), anchor_parent, anchor_child, axis, lower, upper, damping',
    joint_mask='bool [E,12]', joint_pos='float32 [E,T+1,12] joint coordinate from the measured poses (joint_gnn convention; rad / m)',
    joint_pos_source='float32 [E,T+1,12] the dataset\'s own joint_states minus joint_zero, resampled to 20 Hz (NaN if absent)',
    joint_zero='float32 [E,12] URDF joint value at which the row\'s coordinate is 0 (hinges: most compact pose)',
    group='int8 [E,12] 0 = body of the articulated object, -1 = unused',
    driven='bool [E] True: motion caused by unobserved external (human) forces', free_motion='bool [E] window lies inside a verified free-motion segment',
    source_id='<U80 [E] recording name', start_s='float32 [E] window start within the recording',
    n_frames='int64 [E] valid frames (always T+1 here)', base_moved_m='float32 [E] displacement range of the root link in the window',
)


# ----------------------------------------------------------------------------- small maths
def rpy_matrix(rpy):
    """URDF fixed-axis roll-pitch-yaw: R = Rz(yaw) Ry(pitch) Rx(roll)."""
    r, p, y = (float(v) for v in rpy)
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr], [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr], [-sp, cp*sr, cp*cr]])


def matrix_to_quat(m):
    """[...,3,3] proper rotations -> xyzw."""
    m = np.asarray(m, np.float64); out = np.empty(m.shape[:-2] + (4,))
    for index in np.ndindex(m.shape[:-2]):
        r = m[index]; t = np.trace(r)
        if t > 0:
            s = math.sqrt(t + 1) * 2; q = [(r[2, 1]-r[1, 2])/s, (r[0, 2]-r[2, 0])/s, (r[1, 0]-r[0, 1])/s, s/4]
        else:
            i = int(np.argmax(np.diag(r))); j, k = (i+1) % 3, (i+2) % 3
            s = math.sqrt(max(r[i, i]-r[j, j]-r[k, k]+1, 1e-18)) * 2
            q = [0., 0., 0., (r[k, j]-r[j, k])/s]; q[i] = s/4; q[j] = (r[j, i]+r[i, j])/s; q[k] = (r[k, i]+r[i, k])/s
        out[index] = q
    return pe.q_normalize(out)


C_ZUP_TO_YUP = np.array([[1., 0, 0], [0, 0, 1], [0, -1, 0]])           # (x, y, z) -> (x, z, -y)


def yaw_matrix(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


# ----------------------------------------------------------------------------- model files (URDF, *_in.yaml, .dae)
def parse_in_yaml(text: str) -> dict:
    """The two fields we need from <object>_in.yaml, by regular expression (no YAML loader)."""
    block = re.search(r'link_ids:\s*\{([^}]*)\}', text)
    if not block: raise ValueError('no link_ids in *_in.yaml')
    link_ids = {int(link): int(mocap) for mocap, link in re.findall(r'(\d+)\s*:\s*"(\d+)"', block.group(1))}
    return dict(marker_of_link=link_ids)                                # link index -> mocap rigid-body id


def _floats(text, count=3):
    values = [float(v) for v in (text or '').split()]
    if len(values) != count: raise ValueError(f'expected {count} numbers, got {text!r}')
    return np.array(values)


def parse_urdf(text: str) -> dict:
    """Links rb<k> with their visual mesh (in the link frame) and the movable joints.

    RBO layout: rb0 carries its visual directly (link frame = mocap marker frame); every other link
    rb<k> is an empty link with a fixed joint to rb<k>_marker (the mocap frame, which carries the visual).
    Returns links {k: dict(marker_T 4x4 marker frame in the link frame, mesh name, mesh_T 4x4 mesh in
    the link frame)} and joints [dict(name, type, parent, child, origin, axis)].
    """
    robot = ET.fromstring(text)
    def origin(element):
        o = element.find('origin') if element is not None else None
        t = np.eye(4)
        if o is not None:
            t[:3, :3] = rpy_matrix(_floats(o.get('rpy', '0 0 0'))); t[:3, 3] = _floats(o.get('xyz', '0 0 0'))
        return t
    visuals = {}
    for link in robot.findall('link'):
        visual = link.find('visual')
        if visual is not None and visual.find('geometry/mesh') is not None:
            visuals[link.get('name')] = (origin(visual), visual.find('geometry/mesh').get('filename'))
    marker = {}; joints = []
    for joint in robot.findall('joint'):
        parent, child, kind = joint.find('parent').get('link'), joint.find('child').get('link'), joint.get('type')
        if kind == 'fixed' and child == f'{parent}_marker': marker[parent] = origin(joint)
        elif kind in ('revolute', 'continuous', 'prismatic'):
            t = origin(joint)
            if not np.allclose(t[:3, :3], np.eye(3), atol=1e-9): raise ValueError(f'joint {joint.get("name")} has a rotated origin (unsupported)')
            axis = _floats(joint.find('axis').get('xyz'))
            joints.append(dict(name=joint.get('name'), type='prismatic' if kind == 'prismatic' else 'hinge',
                               parent=int(parent[2:]), child=int(child[2:]), origin=t[:3, 3], axis=axis / np.linalg.norm(axis)))
        else:
            raise ValueError(f'unsupported joint {joint.get("name")} ({kind}: {parent} -> {child})')
    links = {}
    for link in robot.findall('link'):
        name = link.get('name')
        if not re.fullmatch(r'rb\d+', name): continue
        index = int(name[2:])
        if name in visuals: marker_t, (mesh_t, mesh) = np.eye(4), visuals[name]
        elif name in marker and f'{name}_marker' in visuals:
            marker_t = marker[name]; mesh_t, mesh = visuals[f'{name}_marker']; mesh_t = marker_t @ mesh_t
        else: raise ValueError(f'link {name} has no visual / marker frame')
        links[index] = dict(marker_T=marker_t, mesh=mesh.split('/objects/')[-1], mesh_T=mesh_t)
    return dict(links=links, joints=joints)


def read_dae_vertices(data: bytes) -> np.ndarray:
    """Vertex positions [V,3] (metres) of a COLLADA file, with the matrix of the node that instantiates each geometry."""
    root = ET.fromstring(data)
    ns = root.tag[:root.tag.index('}') + 1] if root.tag.startswith('{') else ''
    unit = root.find(f'{ns}asset/{ns}unit'); scale = float(unit.get('meter', 1)) if unit is not None else 1.
    geometry = {}
    for g in root.iter(f'{ns}geometry'):
        mesh = g.find(f'{ns}mesh')
        if mesh is None: continue
        sources = {s.get('id'): s for s in mesh.findall(f'{ns}source')}
        position = mesh.find(f"{ns}vertices/{ns}input[@semantic='POSITION']")
        if position is None: continue
        array = sources[position.get('source').lstrip('#')].find(f'{ns}float_array')
        if array is None or not (array.text or '').strip(): continue      # empty geometry
        geometry[g.get('id')] = np.array(array.text.split(), np.float64).reshape(-1, 3)
    out = []
    for node in root.iter(f'{ns}node'):
        for instance in node.findall(f'{ns}instance_geometry'):
            if any(node.find(f'{ns}{tag}') is not None for tag in ('translate', 'rotate', 'scale')):
                raise ValueError('COLLADA node uses translate/rotate/scale elements (unsupported)')
            matrix = node.find(f'{ns}matrix')
            m = np.array(matrix.text.split(), np.float64).reshape(4, 4) if matrix is not None else np.eye(4)
            v = geometry.get(instance.get('url').lstrip('#'))
            if v is None: continue
            out.append(v @ m[:3, :3].T + m[:3, 3])
    if not out: raise ValueError('no instantiated geometry in COLLADA file')
    return np.concatenate(out) * scale


def load_model(archive: Path, obj: str, date: str) -> dict:
    """Kinematic model of `obj` for the marker-set configuration of `date`, read straight from models/<obj>.tar.gz."""
    wanted = {f'{obj}/configuration_{date}/{obj}.urdf': 'urdf', f'{obj}/configuration_{date}/{obj}_in.yaml': 'yaml'}
    found = {}; meshes = {}
    with tarfile.open(archive, 'r:gz') as tar:
        for member in tar:
            name = member.name.lstrip('./')
            if not member.isfile() or member.size > 100_000_000: continue
            if name in wanted: found[wanted[name]] = tar.extractfile(member).read().decode('utf-8', 'replace')
            elif name.endswith('.dae'): meshes[name] = tar.extractfile(member).read()
    if len(found) != 2: raise FileNotFoundError(f'{archive.name} has no configuration_{date}')
    model = parse_urdf(found['urdf']); model.update(parse_in_yaml(found['yaml']), name=obj, date=date)
    for index, link in model['links'].items():
        if link['mesh'] not in meshes: raise FileNotFoundError(f'mesh {link["mesh"]} missing in {archive.name}')
        v = read_dae_vertices(meshes[link['mesh']])
        link['vertices'] = v @ link['mesh_T'][:3, :3].T + link['mesh_T'][:3, 3]          # in the link frame
        if index not in model['marker_of_link']: raise ValueError(f'link rb{index} has no mocap id')
    return model


# ----------------------------------------------------------------------------- recordings (ROS topic CSV dumps)
def read_rb_poses(path: Path):
    """rb_poses_array.csv -> stamps [T] (s, from the first), {mocap id: [T,7] pos + quat xyzw, NaN where missing}."""
    with open(path, newline='') as handle:
        reader = csv.reader(handle); header = next(reader); rows = [r for r in reader if r]
    first = header.index('field.markers0.id'); stride = (header.index('field.markers1.id') - first) if 'field.markers1.id' in header else len(header)
    pose = header.index('field.markers0.pose.position.x') - first; stamp = header.index('field.markers0.header.stamp')
    stamps = np.array([int(r[stamp]) for r in rows], np.int64)
    poses: dict[int, np.ndarray] = {}
    for t, row in enumerate(rows):
        for start in range(first, len(row) - pose - 6, stride):
            if row[start] == '': continue
            try: values = [float(v) for v in row[start + pose:start + pose + 7]]
            except ValueError: continue
            poses.setdefault(int(row[start]), np.full((len(rows), 7), np.nan))[t] = values
    for values in poses.values():
        bad = (np.linalg.norm(values[:, 3:], axis=-1) < .5) | ~np.isfinite(values).all(-1)
        values[bad] = np.nan
    return stamps, poses


def read_joint_states(path: Path):
    """<object>_joint_states.csv -> stamps [T] int64 ns, names, positions [T,J]."""
    with open(path, newline='') as handle:
        reader = csv.reader(handle); header = next(reader); rows = [r for r in reader if r]
    names = [rows[0][i] for i, c in enumerate(header) if re.fullmatch(r'field\.name\d+', c)]
    columns = [i for i, c in enumerate(header) if re.fullmatch(r'field\.position\d+', c)]
    stamp = header.index('field.header.stamp')
    return np.array([int(r[stamp]) for r in rows], np.int64), names, np.array([[float(r[i]) for i in columns] for r in rows])


def read_ft(path: Path):
    with open(path, newline='') as handle:
        reader = csv.reader(handle); header = next(reader); rows = [r for r in reader if r]
    stamp = header.index('field.header.stamp'); fx = header.index('field.wrench.force.x')
    return np.array([int(r[stamp]) for r in rows], np.int64), np.array([[float(v) for v in r[fx:fx + 3]] for r in rows])


def read_tf_frame(path: Path, child: str):
    """tf.csv -> stamps [K] int64 ns and positions [K,3] of the transform /map -> `child` (None if absent)."""
    with open(path, newline='') as handle:
        reader = csv.reader(handle); header = next(reader)
        name = header.index('field.transforms0.child_frame_id'); stamp = header.index('field.transforms0.header.stamp')
        x = header.index('field.transforms0.transform.translation.x')
        rows = [(int(r[stamp]), float(r[x]), float(r[x + 1]), float(r[x + 2])) for r in reader if len(r) > x + 2 and r[name].strip('/') == child]
    if len(rows) < 10: return None
    data = np.array(sorted(rows), np.float64)
    return np.array([r[0] for r in sorted(rows)], np.int64), data[:, 1:]


def read_index(path: Path) -> dict:
    with open(path, newline='') as handle:
        return {row['Name']: dict(object=row['Object'], date=row['recording date(marker set id)'].strip(), ft=row['force/torque sensor used'].strip() == '1',
                                  comment=row.get('comment', '') or '') for row in csv.DictReader(handle)}


# ----------------------------------------------------------------------------- geometry of one recording
def link_frames(model: dict, poses: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Link-frame poses from the mocap marker frames: position [T,L,3], rotation [T,L,3,3], valid [T] (mocap frame)."""
    order = sorted(model['links'])
    missing = [model['marker_of_link'][k] for k in order if model['marker_of_link'][k] not in poses]
    if missing: raise KeyError(f'mocap rigid bodies {missing} are not in the recording')
    frames = len(next(iter(poses.values())))
    pos = np.zeros((frames, len(order), 3)); rot = np.zeros((frames, len(order), 3, 3)); valid = np.ones(frames, bool)
    for column, k in enumerate(order):
        raw = poses[model['marker_of_link'][k]]; ok = np.isfinite(raw).all(-1); valid &= ok
        raw = np.where(ok[:, None], raw, np.array([0, 0, 0, 0, 0, 0, 1.]))
        marker_r = pe.q_to_matrix(raw[:, 3:]); inverse = np.linalg.inv(model['links'][k]['marker_T'])   # link frame in the marker frame
        rot[:, column] = marker_r @ inverse[:3, :3]; pos[:, column] = raw[:, :3] + marker_r @ inverse[:3, 3]
    return pos, rot, valid


def axis_rotation(axis, angle):
    """Rodrigues: rotation by `angle` about the unit `axis`."""
    x, y, z = axis; k = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * (k @ k)


def fit_boxes(model: dict, rotations: np.ndarray) -> dict:
    """One box per link (R1, R2). rotations [L,3,3]: link frames at the reference instant (mocap frame).

    The root box frame is upright with the footprint-minimising yaw.  A child's box frame is the parent's box
    frame at the joint's box zero: 0 for a prismatic joint; for a hinge the URDF angle `zero` (0.5 deg grid) that
    minimises child box volume / its minimum + 3 x diagonal of the pair's common bounding box / its minimum,
    i.e. the pose where the pair is most compact and the child box still tight (lid / door closed), since the
    URDF zero is an arbitrary pose (the laptop lid is ~120 deg open at URDF 0).  joint_gnn coordinate = URDF - zero.
    Returns align [L,3,3] (box frame in the link frame), size [L,3], centre [L,3] (link frame), joints [J,18],
    zero [J], order (link indices by slot) and root (slot).
    """
    order = sorted(model['links']); slot = {k: i for i, k in enumerate(order)}
    parent_joint = {j['child']: (n, j) for n, j in enumerate(model['joints'])}
    roots = [k for k in order if k not in parent_joint]
    if len(roots) != 1: raise ValueError(f'expected one root link, found {roots}')
    root = roots[0]
    vertices = {k: model['links'][k]['vertices'] for k in order}
    thin = {k: v[::max(len(v) // 3000, 1)] for k, v in vertices.items()}
    upright = (C_ZUP_TO_YUP @ rotations[slot[root]]).T
    footprint = lambda theta: np.prod(np.ptp(thin[root] @ (upright @ yaw_matrix(theta)), 0)[[0, 2]])
    coarse = min(np.radians(np.arange(0, 90, .5)), key=footprint)
    align = {root: upright @ yaw_matrix(min(coarse + np.radians(np.arange(-.5, .5001, .01)), key=footprint))}
    zero = np.zeros(len(model['joints']))
    pending = [k for k in order if k != root]
    while pending:
        ready = [k for k in pending if parent_joint[k][1]['parent'] in align]
        if not ready: raise ValueError('kinematic loop or missing parent link')
        for k in ready:
            number, joint = parent_joint[k]; base = align[joint['parent']]
            if joint['type'] == 'hinge':
                angles = np.radians(np.arange(-180, 180, .5)); own = []; common = []
                parent_points = thin[joint['parent']] @ base
                for angle in angles:
                    turned = thin[k] @ axis_rotation(joint['axis'], angle).T
                    own.append(np.prod(np.ptp(turned @ base, 0)))
                    common.append(np.linalg.norm(np.ptp(np.concatenate((parent_points, (turned + joint['origin']) @ base)), 0)))
                own, common = np.array(own), np.array(common)
                zero[number] = angles[int(np.argmin(own / own.min() + 3 * common / common.min()))]
            align[k] = axis_rotation(joint['axis'], -zero[number]) @ base
            pending.remove(k)
    size = np.zeros((len(order), 3)); centre = np.zeros((len(order), 3))
    for k in order:
        v = vertices[k] @ align[k]                                       # vertex coordinates in the box frame
        size[slot[k]] = np.ptp(v, 0); centre[slot[k]] = align[k] @ ((v.max(0) + v.min(0)) / 2)
    rows = np.zeros((len(model['joints']), JOINT_DIM), np.float32)
    for row, joint in zip(rows, model['joints']):
        p, c = slot[joint['parent']], slot[joint['child']]
        hinge = joint['type'] == 'hinge'
        row[0], row[1] = p, c
        row[2 if hinge else 4] = 1
        row[6:9] = align[joint['parent']].T @ (joint['origin'] - centre[p]); row[9:12] = align[joint['child']].T @ (-centre[c])
        row[12:15] = align[joint['parent']].T @ joint['axis']
        row[15], row[16] = (-math.pi, math.pi) if hinge else (-10., 10.)
    return dict(align=np.stack([align[k] for k in order]), size=size, centre=centre, joints=rows, zero=zero, order=order, root=slot[root])


def body_states(pos, rot, boxes) -> tuple[np.ndarray, np.ndarray]:
    """Box centre positions [T,L,3] and orientation quaternions [T,L,4] in the Y-up world (unshifted)."""
    centre = pos + np.einsum('tlij,lj->tli', rot, boxes['centre'])
    rotation = np.einsum('ij,tljk,lkm->tlim', C_ZUP_TO_YUP, rot, boxes['align'])
    return centre @ C_ZUP_TO_YUP.T, pe.sign_continuous(matrix_to_quat(rotation))


def numpy_joint_coordinates(pos, quat, rows) -> np.ndarray:
    """joint_gnn.joint_coordinates in numpy/float64 ([T,L,3], [T,L,4], [J,18] -> [T,J]); the test suite checks they agree."""
    out = np.zeros((len(pos), len(rows)))
    for j, row in enumerate(rows):
        p, c = int(round(row[0])), int(round(row[1])); axis = row[12:15].astype(np.float64); axis /= np.linalg.norm(axis)
        if row[2] > .5:
            rel = pe.q_mul(pe.q_conj(quat[:, p]), quat[:, c]); s, w = rel[:, :3] @ axis, rel[:, 3]
            out[:, j] = np.arctan2(2 * s * w, w * w - s * s)
        elif row[4] > .5:
            rp, rc = pe.q_to_matrix(quat[:, p]), pe.q_to_matrix(quat[:, c])
            gap = pos[:, c] + rc @ row[9:12].astype(np.float64) - pos[:, p] - rp @ row[6:9].astype(np.float64)
            out[:, j] = (gap * (rp @ axis)).sum(-1)
    return out


def violations(pos, quat, rows) -> dict:
    """agatha.joint_gnn.joint_violation / joint_coordinates on states [T,L,13-compatible] -> numpy [T,J] each."""
    import torch
    from agatha import joint_gnn as jg
    state = torch.zeros(len(pos), pos.shape[1], 13, dtype=torch.float64)
    state[..., :3] = torch.from_numpy(pos - pos[:1].mean(1, keepdims=True)); state[..., 3:7] = torch.from_numpy(quat)   # centred: float32 inside
    joints = torch.from_numpy(np.asarray(rows, np.float32))[None].expand(len(pos), -1, -1)
    mask = torch.ones(len(pos), len(rows), dtype=torch.bool)
    out = {k: v.double().numpy() for k, v in jg.joint_violation(state, joints, mask).items()}
    out['coordinate'] = jg.joint_coordinates(state, joints, mask).double().numpy()
    return out


# ----------------------------------------------------------------------------- free-motion search
def free_motion_segments(t, q, kinds, ft=None, stick=None) -> list[dict]:
    """Candidate free-motion segments of each joint at the source rate, and the F/T check.

    A moving run is a maximal interval with |qdot| > MOVE_MIN (0.15 s moving average) whose peak speed reaches
    LAUNCH_MIN and that ends inside the recording.  Two patterns are kept:
      decelerating_tail          peak -> end of the run, >= FREE_MIN_S: what a part coasting to a stop looks like -
                                 and also what every hand-guided motion looks like when the hand slows down;
      accelerating_abrupt_stop   start -> peak, >= FREE_MIN_S, with the speed collapsing within ABRUPT_S after the
                                 peak: what a lid falling shut looks like - or a hand pushing something against a stop.
    Kinematics alone cannot tell these apart.  A segment is verified (usable) only if, for the whole segment,
    (i) F/T data exist and the force stays within FT_FREE_N of its joint-idle median AND (ii) the mocap-tracked
    F/T stick is at rest (speed < STICK_STILL): a stick that travels along with the part is pushing it, however
    small the force (a light door needs < 0.5 N).  ft: optional (t_ft, force [K,3]); stick: optional (t, pos [K,3]).
    """
    t = np.asarray(t, np.float64); dt = float(np.median(np.diff(t))); width = max(int(round(SMOOTH_S / dt)) | 1, 3)
    kernel = np.ones(width) / width; segments = []
    pad = width // 2
    smooth = np.stack([np.convolve(np.pad(q[:, j], pad, mode='edge'), kernel, mode='valid') for j in range(q.shape[1])], -1)
    rate = np.gradient(smooth, t, axis=0)
    moving_any = np.zeros(len(t), bool)
    for j, kind in enumerate(kinds): moving_any |= np.abs(rate[:, j]) > MOVE_MIN[kind]
    deviation = None
    if ft is not None:
        force = np.stack([np.interp(t, ft[0], ft[1][:, a]) for a in range(3)], -1)
        idle = np.median(force[~moving_any], 0) if (~moving_any).sum() >= 10 else np.median(force, 0)
        deviation = np.linalg.norm(force - idle, axis=-1)
    stick_speed = None
    if stick is not None:
        order = np.r_[True, np.diff(stick[0]) > 0]
        track = np.stack([np.convolve(np.pad(np.interp(t, stick[0][order], stick[1][order, a]), pad, mode='edge'), kernel, mode='valid') for a in range(3)], -1)
        stick_speed = np.linalg.norm(np.gradient(track, t, axis=0), axis=-1)
    for j, kind in enumerate(kinds):
        speed = np.abs(rate[:, j]); moving = speed > MOVE_MIN[kind]
        edges = np.flatnonzero(np.diff(np.r_[False, moving, False].astype(int)))
        for start, stop in zip(edges[::2], edges[1::2]):                 # moving on [start, stop)
            if stop >= len(t): continue                                  # recording ends while still moving
            peak = start + int(np.argmax(speed[start:stop]))
            if speed[peak] < LAUNCH_MIN[kind]: continue
            tail, rise = t[stop - 1] - t[peak], t[peak] - t[start]
            found = []
            if tail >= FREE_MIN_S: found.append(('decelerating_tail', peak, stop - 1))
            if rise >= FREE_MIN_S and tail <= ABRUPT_S: found.append(('accelerating_abrupt_stop', start, peak))
            for pattern, a, b in found:
                worst = None if deviation is None else float(deviation[a:b + 1].max())
                fastest = None if stick_speed is None else float(stick_speed[a:b + 1].max())
                segments.append(dict(joint=j, type=kind, pattern=pattern, start_s=round(float(t[a]), 3), end_s=round(float(t[b]), 3),
                                     duration_s=round(float(t[b] - t[a]), 3), peak_speed=round(float(speed[peak]), 4),
                                     travel=round(float(abs(smooth[b, j] - smooth[a, j])), 4),
                                     ft_max_force_dev_N=None if worst is None else round(worst, 3),
                                     stick_max_speed_m_s=None if fastest is None else round(fastest, 3),
                                     verified=bool(worst is not None and worst < FT_FREE_N and fastest is not None and fastest < STICK_STILL)))
    return segments


# ----------------------------------------------------------------------------- one recording
def _summary(values, scale=1.):
    values = np.asarray(values, np.float64) * scale
    if values.size == 0: return None
    return dict(mean=round(float(values.mean()), 4), median=round(float(np.median(values)), 4), p95=round(float(np.percentile(values, 95)), 4),
                max=round(float(values.max()), 4))


def convert_recording(name: str, folder: Path, model: dict, info: dict, steps: int, stride: int) -> tuple[list[dict], dict]:
    stamps, poses = read_rb_poses(folder / 'rb_poses_array.csv')
    pos, rot, valid = link_frames(model, poses)
    t = (stamps - stamps[0]) * 1e-9
    keep = np.r_[True, np.diff(t) > 0] & valid                           # strictly increasing stamps, all links tracked
    if keep.sum() < 20: raise ValueError('fewer than 20 valid mocap frames')
    t_valid, pos, rot = t[keep], pos[keep], rot[keep]
    boxes = fit_boxes(model, rot[0])
    links = len(boxes['order']); rows = boxes['joints']; kinds = ['hinge' if r[2] > .5 else 'prismatic' for r in rows]
    centre, quat = body_states(pos, rot, boxes)
    # ---- kinematic consistency at the source rate (joint_gnn functions)
    v = violations(centre, quat, rows)
    gap, off = np.median(v['anchor_gap'], 0), np.median(v['off_axis'], 0)
    if (gap > FIT_MAX_GAP_M).any() or (off > FIT_MAX_AXIS_RAD).any():
        raise ValueError(f'kinematic model configuration_{model["date"]} does not fit this recording (median anchor gap '
                         f'{", ".join(f"{g * 1000:.0f}" for g in gap)} mm, median axis error {", ".join(f"{math.degrees(o):.0f}" for o in off)} deg)')
    q_numpy = numpy_joint_coordinates(centre, quat, rows)
    report = dict(object=model['name'], configuration=model['date'], comment=info.get('comment', ''), ft_sensor=bool(info.get('ft')),
                  duration_s=round(float(t[-1]), 2), source_frames=int(len(t)), valid_frames=int(keep.sum()),
                  dropped_frames=int((~keep).sum()), links=links, box_size_m=[[round(float(x), 4) for x in s] for s in boxes['size']], joints=[])
    source_q = None
    states_file = folder / f"{info['object']}_joint_states.csv"
    if states_file.exists():
        js_stamps, js_names, js_q = read_joint_states(states_file)
        js_t = (js_stamps - stamps[0]) * 1e-9; order = np.argsort(js_t, kind='stable')
        source_q = np.full((len(t_valid), len(rows)), np.nan)
        for j, joint in enumerate(model['joints']):
            if joint['name'] in js_names: source_q[:, j] = np.interp(t_valid, js_t[order], js_q[order, js_names.index(joint['name'])])
        source_q -= boxes['zero']                                        # the dataset's (URDF) coordinate in the box-zero convention
    for j, joint in enumerate(model['joints']):
        hinge = kinds[j] == 'hinge'; unit = 'deg' if hinge else 'mm'; scale = math.degrees(1) if hinge else 1000.
        entry = dict(name=joint['name'], type=kinds[j], parent=f"rb{joint['parent']}", child=f"rb{joint['child']}",
                     anchor_gap_mm=_summary(v['anchor_gap'][:, j], 1000), off_axis_deg=_summary(v['off_axis'][:, j], math.degrees(1)),
                     coordinate_unit='rad' if hinge else 'm', coordinate_min=round(float(v['coordinate'][:, j].min()), 4),
                     coordinate_max=round(float(v['coordinate'][:, j].max()), 4), urdf_zero_offset=round(float(boxes['zero'][j]), 4),
                     numpy_vs_joint_gnn_max=float(np.abs(q_numpy[:, j] - v['coordinate'][:, j]).max()))
        if source_q is not None and np.isfinite(source_q[:, j]).all():
            difference = np.arctan2(np.sin(q_numpy[:, j] - source_q[:, j]), np.cos(q_numpy[:, j] - source_q[:, j])) if hinge else q_numpy[:, j] - source_q[:, j]
            flipped = np.arctan2(np.sin(q_numpy[:, j] + source_q[:, j]), np.cos(q_numpy[:, j] + source_q[:, j])) if hinge else q_numpy[:, j] + source_q[:, j]
            entry[f'coordinate_vs_dataset_{unit}'] = _summary(np.abs(difference), scale)
            entry[f'coordinate_vs_negated_dataset_{unit}'] = _summary(np.abs(flipped), scale)      # would be small if the sign convention were wrong
            entry['dataset_coordinate_range'] = [round(float(source_q[:, j].min()), 4), round(float(source_q[:, j].max()), 4)]
        report['joints'].append(entry)
    # ---- free-motion search at the source rate
    ft = stick = None
    if (folder / 'ft_sensor_netft_data.csv').exists():
        ft_stamps, force = read_ft(folder / 'ft_sensor_netft_data.csv'); ft = ((ft_stamps - stamps[0]) * 1e-9, force)
        tracked = read_tf_frame(folder / 'tf.csv', 'ftSensor') if (folder / 'tf.csv').exists() else None
        if tracked is not None: stick = ((tracked[0] - stamps[0]) * 1e-9, tracked[1])
    segments = free_motion_segments(t_valid, q_numpy, kinds, ft, stick)
    report['free_motion'] = dict(candidates=len(segments), verified=sum(s['verified'] for s in segments), ft_available=ft is not None, segments=segments)
    # ---- 20 Hz episodes
    grid = np.arange(0, t_valid[-1] - t_valid[0] + 1e-9, STEP) + t_valid[0]
    lo, hi, _ = pe._bracket(t_valid, grid)
    grid_ok = (t_valid[hi] - t_valid[lo]) < MAX_GAP_S
    p20, q20 = pe.interp_linear(t_valid, centre, grid), pe.interp_slerp(t_valid, quat, grid)
    vel, ang = pe.finite_difference(grid, p20, q20)
    joint_q = numpy_joint_coordinates(p20, q20, rows)
    joint_source = np.full_like(joint_q, np.nan) if source_q is None else np.stack([np.interp(grid, t_valid, source_q[:, j]) for j in range(len(rows))], -1)
    corners = pe.box_corners(p20, q20, np.broadcast_to(boxes['size'], p20.shape))             # [T,L,8,3]
    table = float(np.median(corners[:, boxes['root'], :, 1].min(-1)))
    report['surface_height_mocap_m'] = round(table, 4)
    report['lowest_corner_range_m'] = [round(float(corners[..., 1].min((1, 2)).min() - table), 4), round(float(corners[..., 1].min((1, 2)).max() - table), 4)]
    volume = boxes['size'].prod(-1); mass = np.clip(volume * DENSITY, *MASS_RANGE)
    verified = [s for s in segments if s['verified']]
    episodes = []
    start = -stride
    while True:
        start += stride
        while start + steps < len(grid) and not grid_ok[start:start + steps + 1].all(): start += 1     # slide past mocap dropouts
        if start + steps >= len(grid): break
        window = slice(start, start + steps + 1)
        position = p20[window].copy(); c = corners[window]
        shift = np.array([-(c[..., 0].max() + c[..., 0].min()) / 2, SUPPORT_HEIGHT - table, -(c[..., 2].max() + c[..., 2].min()) / 2])
        position += shift
        states = np.zeros((steps + 1, N_SLOTS, 13), np.float32); states[..., 6] = 1
        states[:, :links] = np.concatenate((position, pe.sign_continuous(q20[window]), vel[window], ang[window]), -1)
        props = np.zeros((N_SLOTS, 9), np.float32); props[:, :3] = .1; props[:, 3] = 1; props[:, 4] = .5
        props[:links, :3] = boxes['size']; props[:links, 3] = mass
        extent = (np.ptp(c[..., 0]) + .3, np.ptp(c[..., 2]) + .3)
        context = np.array([max(MIN_SUPPORT[0], extent[0]), max(MIN_SUPPORT[1], extent[1]), SUPPORT_HEIGHT, 9.81, 0, 0, 0, 1, .5, 0], np.float32)
        mask = np.zeros(N_SLOTS, bool); mask[:links] = True
        low = c[..., 1].min(-1) - table
        contact = np.zeros((steps + 1, N_SLOTS), bool); contact[:, :links] = np.abs(low) < .005
        slow = np.zeros_like(contact); slow[:, :links] = (np.linalg.norm(vel[window], axis=-1) < .01) & (np.linalg.norm(ang[window], axis=-1) < .05)
        tipped = np.zeros(N_SLOTS, bool); tipped[boxes['root']] = bool(pe.tipped_relative(q20[window][:, boxes['root']][:, None])[0])
        joints = np.zeros((J_SLOTS, JOINT_DIM), np.float32); joints[:len(rows)] = rows
        joint_mask = np.zeros(J_SLOTS, bool); joint_mask[:len(rows)] = True
        jp = np.zeros((steps + 1, J_SLOTS), np.float32); jp[:, :len(rows)] = joint_q[window]
        js = np.full((steps + 1, J_SLOTS), np.nan, np.float32); js[:, :len(rows)] = joint_source[window]
        jz = np.zeros(J_SLOTS, np.float32); jz[:len(rows)] = boxes['zero']
        group = np.full(N_SLOTS, -1, np.int8); group[:links] = 0
        t0, t1 = grid[start], grid[start + steps]
        root_path = p20[window][:, boxes['root']]
        episodes.append(dict(
            states=states, props=props, context=context, mask=mask, contact_support=contact, contact_floor=np.zeros_like(contact),
            contact_object=np.zeros_like(contact), rest=contact & slow, fell=np.zeros(N_SLOTS, bool), tipped=tipped, kind=f'rbo_{model["name"]}',
            chaotic=False, live_physics=False, init_penetration=np.float32(0), joints=joints, joint_mask=joint_mask, joint_pos=jp,
            joint_pos_source=js, joint_zero=jz, group=group, driven=True, free_motion=any(s['start_s'] <= t0 and s['end_s'] >= t1 for s in verified),
            source_id=name, start_s=np.float32(t0), n_frames=np.int64(steps + 1), base_moved_m=np.float32(np.linalg.norm(np.ptp(root_path, 0)))))
    report['episodes'] = len(episodes)
    report['root_displacement_m'] = round(float(np.linalg.norm(np.ptp(p20[:, boxes['root']], 0))), 4)
    return episodes, report


# ----------------------------------------------------------------------------- dataset
def stack(episodes: list[dict]) -> dict:
    data = {key: np.stack([np.asarray(e[key]) for e in episodes]) for key in episodes[0]}
    data['kind'] = data['kind'].astype('<U24'); data['source_id'] = data['source_id'].astype('<U80')
    data['episode_index'] = np.arange(len(episodes), dtype=np.int64)
    return data


def shard_violations(data: dict) -> dict:
    """joint_gnn.joint_violation on the float32 shard states exactly as an evaluator would call it."""
    import torch
    from agatha import joint_gnn as jg
    e, frames = data['states'].shape[:2]
    spread = lambda x: torch.from_numpy(np.repeat(x[:, None], frames, 1).reshape(e * frames, *x.shape[1:]))
    state = torch.from_numpy(data['states'].reshape(e * frames, *data['states'].shape[2:]))
    joints, mask = spread(data['joints']), spread(data['joint_mask'])
    v = jg.joint_violation(state, joints, mask); q = jg.joint_coordinates(state, joints, mask)
    on = mask.numpy(); hinge = on & (joints[..., 2].numpy() > .5); slide = on & (joints[..., 4].numpy() > .5)
    stored = data['joint_pos'].reshape(e * frames, -1)
    return dict(joint_frames=int(on.sum()), hinge_frames=int(hinge.sum()), prismatic_frames=int(slide.sum()),
                anchor_gap_mm=_summary(v['anchor_gap'].numpy()[on], 1000),
                anchor_gap_mm_hinge=_summary(v['anchor_gap'].numpy()[hinge], 1000), anchor_gap_mm_prismatic=_summary(v['anchor_gap'].numpy()[slide], 1000),
                off_axis_deg_hinge=_summary(v['off_axis'].numpy()[hinge], math.degrees(1)), off_axis_deg_prismatic=_summary(v['off_axis'].numpy()[slide], math.degrees(1)),
                limit_excess_max=float(v['limit_excess'].numpy()[on].max()) if on.any() else 0.,
                joint_pos_vs_joint_gnn_max=float(np.abs(q.numpy() - stored)[on].max()) if on.any() else 0.)


def download(names: list[str], root: Path) -> None:
    """Models (<object>) are saved as archives; recordings (<object><NN>) are streamed and ONLY their .csv members are kept."""
    for name in names:
        if not re.fullmatch(r'[a-z]+\d*', name): raise ValueError(f'bad name {name!r}')
        if re.search(r'\d$', name):
            target = root / 'interactions' / f'{name}_o'; target.mkdir(parents=True, exist_ok=True); kept = 0
            with urllib.request.urlopen(ZENODO.format(f'{name}_o.tar.gz'), timeout=120) as response, tarfile.open(fileobj=response, mode='r|gz') as tar:
                for member in tar:
                    if member.isfile() and member.name.lower().endswith('.csv') and member.size < 50_000_000:
                        (target / Path(member.name).name).write_bytes(tar.extractfile(member).read()); kept += 1
            print(f'{name}: kept {kept} csv files in {target}', flush=True)
        else:
            target = root / 'models' / f'{name}.tar.gz'; target.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(ZENODO.format(f'{name}.tar.gz'), timeout=120) as response: target.write_bytes(response.read())
            print(f'{name}: {target.stat().st_size} bytes -> {target}', flush=True)


def markdown(report: dict) -> str:
    lines = ['# RBO articulated objects: kinematic consistency of the joint rows (APPROXIMATE, hand-driven real data)', '', report['note'], '',
             f"Recordings converted: {report['recordings_converted']} ({report['episodes']} windows of {report['steps']} steps at 20 Hz); skipped: "
             f"{len(report['skipped'])}.", '',
             '| recording | joint | type | frames | anchor gap mm (mean / p95 / max) | off-axis deg (mean / p95 / max) | coordinate range | vs dataset joint_states (mean / max) | vs NEGATED (mean) |',
             '|---|---|---|---|---|---|---|---|---|']
    for name, rec in report['recordings'].items():
        for joint in rec['joints']:
            unit = 'deg' if joint['type'] == 'hinge' else 'mm'; g, o = joint['anchor_gap_mm'], joint['off_axis_deg']
            d, n = joint.get(f'coordinate_vs_dataset_{unit}'), joint.get(f'coordinate_vs_negated_dataset_{unit}')
            lines.append(f"| {name} | {joint['name']} | {joint['type']} | {rec['valid_frames']} | {g['mean']:.2f} / {g['p95']:.2f} / {g['max']:.2f} | "
                         f"{o['mean']:.2f} / {o['p95']:.2f} / {o['max']:.2f} | {joint['coordinate_min']} .. {joint['coordinate_max']} {joint['coordinate_unit']} | "
                         + (f"{d['mean']:.3f} / {d['max']:.3f} {unit}" if d else 'n/a') + ' | ' + (f"{n['mean']:.2f} {unit}" if n else 'n/a') + ' |')
    s = report['shards']
    lines += ['', '## Shards (float32, 20 Hz, as an evaluator reads them)', '', f"{json.dumps(s)}", '', '## Free-motion segments', '',
              f"Kinematic candidates (hand presence unknown): {report['free_motion']['candidates']}; verified free of contact (usable): **{report['free_motion']['verified']}**; "
              f"recordings with F/T data: {report['free_motion']['recordings_with_ft']}.", '',
              '| recording | joint | pattern | start s | duration s | peak speed | travel | max F/T deviation N | max stick speed m/s | verified |', '|---|---|---|---|---|---|---|---|---|---|']
    for name, rec in report['recordings'].items():
        for seg in rec['free_motion']['segments']:
            lines.append(f"| {name} | {seg['joint']} ({seg['type']}) | {seg['pattern']} | {seg['start_s']} | {seg['duration_s']} | {seg['peak_speed']} | {seg['travel']} | "
                         f"{seg['ft_max_force_dev_N']} | {seg['stick_max_speed_m_s']} | {seg['verified']} |")
    if report['skipped']: lines += ['', '## Skipped', ''] + [f'- {k}: {v}' for k, v in report['skipped'].items()]
    return '\n'.join(lines) + '\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--external', type=Path, default=ROOT / 'data/external/rbo-articulated')
    parser.add_argument('--out', type=Path, default=ROOT / 'data/external_v2')
    parser.add_argument('--report', type=Path, default=ROOT / 'models/external_joints_kinematics.json')
    parser.add_argument('--steps', type=int, default=40, help='steps per episode window (40 = 2 s)')
    parser.add_argument('--stride', type=int, default=40, help='window stride in steps (default: non-overlapping)')
    parser.add_argument('--download', nargs='+', metavar='NAME', help='fetch models (e.g. laptop) / recordings (e.g. laptop04) from Zenodo and exit')
    args = parser.parse_args(argv)
    if args.download:
        download(args.download, args.external); return
    index = read_index(args.external / 'interactions_index.csv')
    episodes, recordings, skipped, models = [], {}, {}, {}
    for folder in sorted((args.external / 'interactions').glob('*_o')):
        name = folder.name[:-2]; info = index.get(name)
        try:
            if info is None: raise KeyError('not in interactions_index.csv')
            key = (info['object'], info['date'])
            if key not in models:
                archive = args.external / 'models' / f"{info['object']}.tar.gz"
                if not archive.exists(): raise FileNotFoundError(f'model archive {archive.name} not downloaded')
                models[key] = load_model(archive, *key)
            new, report = convert_recording(name, folder, models[key], info, args.steps, args.stride)
        except (KeyError, ValueError, FileNotFoundError, ET.ParseError, tarfile.TarError) as error:
            skipped[name] = f'{type(error).__name__}: {error}'; print(f'skip {name}: {skipped[name]}'); continue
        episodes += new; recordings[name] = report
        gaps = ', '.join(f"{j['name']} {j['type']} gap {j['anchor_gap_mm']['mean']:.2f} mm, off-axis {j['off_axis_deg']['mean']:.2f} deg" for j in report['joints'])
        print(f"{name}: {report['valid_frames']} frames, {len(new)} episodes, {gaps}, free candidates {report['free_motion']['candidates']} "
              f"(verified {report['free_motion']['verified']})")
    if not episodes: raise SystemExit('nothing converted')
    data = stack(episodes)
    target = args.out / DATASET / 'test'; target.mkdir(parents=True, exist_ok=True)
    for old in target.glob('shard_*.npz'): old.unlink()
    np.savez_compressed(target / 'shard_00000.npz', **data)
    summary = dict(
        note=NOTE, dataset=DATASET, source='RBO Dataset of Articulated Objects and Interactions (arXiv:1806.06465), https://zenodo.org/records/1036660',
        license='CC BY 4.0', step_s=STEP, steps=args.steps, stride=args.stride, episodes=len(episodes), recordings_converted=len(recordings),
        skipped=skipped, shards=shard_violations(data),
        free_motion=dict(candidates=sum(r['free_motion']['candidates'] for r in recordings.values()),
                         verified=sum(r['free_motion']['verified'] for r in recordings.values()),
                         recordings_with_ft=sum(r['free_motion']['ft_available'] for r in recordings.values()),
                         episodes_inside_verified_segment=int(data['free_motion'].sum())),
        recordings=recordings)
    meta = dict(summary, max_objects=N_SLOTS, max_joints=J_SLOTS, path=str(target),
                episodes_per_kind={k: int(n) for k, n in zip(*np.unique(data['kind'], return_counts=True))},
                format=dict(note='box keys as in scripts/generate_contact_data.py FORMAT', **EXTRA_FORMAT),
                assumptions=dict(density_kg_m3=DENSITY, mass_clip_kg=MASS_RANGE, friction=.5, support_height_m=SUPPORT_HEIGHT, min_support_m=MIN_SUPPORT))
    (args.out / DATASET / 'meta.json').write_text(json.dumps(meta, indent=2))
    top = args.out / 'meta.json'
    existing = json.loads(top.read_text()) if top.exists() else dict(note=pe.NOTE, step_s=STEP, datasets={})
    existing.setdefault('datasets', {})[DATASET] = dict(episodes=len(episodes), path=str(target), note=NOTE)
    top.write_text(json.dumps(existing, indent=2))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(summary, indent=2)); args.report.with_suffix('.md').write_text(markdown(summary))
    print(f"{DATASET}: {len(episodes)} episodes from {len(recordings)} recordings -> {target}; shard check {json.dumps(summary['shards'])}")
    print(f"free-motion: {summary['free_motion']}")


if __name__ == '__main__':
    main()
