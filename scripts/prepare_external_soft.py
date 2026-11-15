"""Convert DeepMind MeshGraphNets cloth data into the soft_v1 shard format (TRACK S, EXPERIMENTAL).

Source: data/external/meshgraphnets (see its README.txt; Pfaff et al., ICLR 2021, simulator ArcSim).
Licence status: no explicit data licence; used by team decision under the repository Apache-2.0
licence; model not publicly released (see DATA_LICENSES.md).  The files are only parsed
(hand-written TFRecord / tf.train.Example reader below; nothing is executed, CRCs are not checked).

What is converted
  sphere_simple   a 1 m x 1 m cloth pinned at two HANDLE nodes that falls / swings under gravity while a
                  kinematic sphere (radius 0.1 m) moves through it.  Only the locally present file
                  (valid.tfrecord, 100 trajectories) is used and split by trajectory index into
                  train / val / test = 80 / 10 / 10 (index % 10 == 8 -> val, == 9 -> test).
What is NOT converted (and why)
  flag_minimal / flag_simple   the flag is driven by WIND, an unobserved external force that is not part of
                  the model input, so the accelerations cannot be explained from the state; in addition the
                  local flag_minimal train / valid / test files are byte-identical (11 trajectories, no
                  held-out data).  Recommended mix fraction: 0.
  sphere_dynamic, deforming_plate   only meta.json is present locally (deforming_plate is a quasi-static
                  tetrahedral plate pressed by a moving actuator mesh: no gravity-driven dynamics).

Conversion (all factors are written to <out>/meta.json)
  axes      source is Z-up (gravity along -z): (x, y, z) -> (x, z, -y), i.e. Y-up, right-handed.
  scale     similarity rescale lengths by --scale (default 0.36: the cloth becomes 0.36 m, the sphere radius
            0.036 m) and times by sqrt(scale) (0.6), so that gravity keeps its numerical value.  ArcSim's
            gravity is taken as 9.8 m/s^2 from the published configs; it CANNOT be verified from the data
            (no free-falling body) and is written to context[3] as 9.8.
  time      source dt 0.01 s -> 0.006 s after the rescale; frames are linearly interpolated at 20 Hz
            (every 8.333 source frames): 501 source frames -> 61 frames (60 steps of 0.05 s).
  height    the scene is lifted so that the handles sit at y = --handle-height (default 1.0 m): the soft
            model always has an analytic floor at y = 0 and the cloth must stay clear of it.
  mesh      the irregular triangle mesh (1087 cloth nodes) is sub-sampled to --max-verts (128) nodes by
            farthest-point sampling in material coordinates, always keeping the handles.  Every fine node is
            assigned to its nearest kept node; two kept nodes are joined by an edge when a fine edge
            connects their clusters, and a fine triangle whose corners lie in three clusters gives a coarse
            face (connectivity of the fine mesh is preserved).  rest_len is the material-space distance
            times the measured world / material scale.  Vertex positions are the kept nodes' own positions.
  pinned    HANDLE nodes -> 'pinned' bool [E,V]: kinematic vertices, prescribed from the ground truth.
  obstacle  the sphere is kept EXACT: rigid slot 0 with size = diameter and a 10th props column = rounding
            radius (soft_gnn treats it as a rounded box, here a sphere); its centre is the mean of the
            obstacle nodes, it is flagged static (kinematic) and prescribed like every rigid box.
            (A sharp box would misplace the surface by 19-40% of the radius, so that was not done.)
  material  the data has no masses or material parameters: soft_props are PLACEHOLDERS (mass 0.05 kg,
            normalised stiffness 0, damping 0, friction 0.5, is_cloth 1, area) and identical for all episodes.
  context   no table: support collisions disabled (context[7] = 0).

Usage:
  .venv/bin/python scripts/prepare_external_soft.py            # -> data/external_v2/meshgraphnets_soft/{train,val,test}
"""
import argparse
import json
import struct
import time
from pathlib import Path

import numpy as np

STEP = .05
SETS = ('sphere_simple',)
SKIPPED = dict(
    flag_minimal='wind-driven (unobserved external force); local train/valid/test files are byte-identical (11 trajectories)',
    flag_simple='wind-driven (unobserved external force); only meta.json present locally',
    sphere_dynamic='only meta.json present locally',
    deforming_plate='only meta.json present locally; quasi-static actuator-driven plate, no gravity-driven dynamics')
NODE_NORMAL, NODE_OBSTACLE, NODE_HANDLE = 0, 1, 3
GRAVITY = 9.8
PLACEHOLDER_PROPS = dict(mass_kg=.05, stiffness_norm=0., damping=0., friction=.5)


# ----------------------------------------------------------------------------- TFRecord / tf.train.Example
def _varint(buf, i):
    result = shift = 0
    while True:
        byte = buf[i]
        i += 1
        result |= (byte & 0x7F) << shift
        if byte < 0x80:
            return result, i
        shift += 7


def _fields(buf):
    i, n = 0, len(buf)
    while i < n:
        key, i = _varint(buf, i)
        field, wire = key >> 3, key & 7
        if wire == 2:
            length, i = _varint(buf, i)
            yield field, buf[i:i + length]
            i += length
        elif wire == 0:
            value, i = _varint(buf, i)
            yield field, value
        elif wire == 5:
            yield field, buf[i:i + 4]
            i += 4
        elif wire == 1:
            yield field, buf[i:i + 8]
            i += 8
        else:
            raise ValueError(f'unsupported protobuf wire type {wire}')


def records(path, limit=0):
    """Raw records of a TFRecord file (length-prefixed; the CRCs are skipped, not verified)."""
    with open(path, 'rb') as handle:
        count = 0
        while True:
            head = handle.read(12)
            if len(head) < 12:
                return
            length = struct.unpack('<Q', head[:8])[0]
            data = handle.read(length)
            if len(data) < length:
                raise IOError(f'truncated record in {path}')
            handle.read(4)
            yield memoryview(data)
            count += 1
            if limit and count >= limit:
                return


def parse_example(record) -> dict:
    """{feature name: first bytes_list value} of a serialized tf.train.Example."""
    out = {}
    for _, features in _fields(record):
        for _, entry in _fields(features):
            key = value = None
            for field, payload in _fields(entry):
                if field == 1:
                    key = bytes(payload).decode()
                else:
                    for kind, items in _fields(payload):
                        if kind == 1:                                   # Feature.bytes_list
                            for _, item in _fields(items):
                                value = item
            out[key] = value
    return out


def trajectory(record, meta) -> dict:
    example = parse_example(record)
    return {name: np.frombuffer(example[name], dtype=spec['dtype']).reshape(spec['shape']) for name, spec in meta['features'].items()}


def write_tfrecord(path, examples):
    """Tiny writer for the tests: one Example of bytes features per dict {name: bytes}; CRC fields are zero."""
    def varint(value):
        out = bytearray()
        while True:
            byte = value & 0x7F
            value >>= 7
            out.append(byte | (0x80 if value else 0))
            if not value:
                return bytes(out)

    def field(number, payload):
        return varint(number << 3 | 2) + varint(len(payload)) + payload
    with open(path, 'wb') as handle:
        for example in examples:
            entries = b''.join(field(1, field(1, name.encode()) + field(2, field(1, field(1, value)))) for name, value in example.items())
            data = field(1, entries)
            handle.write(struct.pack('<Q', len(data)) + b'\0' * 4 + data + b'\0' * 4)


# ----------------------------------------------------------------------------- mesh coarsening
def farthest_points(points, count, first=()):
    """Indices of `count` points chosen by farthest-point sampling, starting with `first`."""
    chosen = list(dict.fromkeys(int(i) for i in first)) or [0]
    distance = np.full(len(points), np.inf)
    for index in chosen:
        distance = np.minimum(distance, np.linalg.norm(points - points[index], axis=-1))
    while len(chosen) < min(count, len(points)):
        index = int(distance.argmax())
        chosen.append(index)
        distance = np.minimum(distance, np.linalg.norm(points - points[index], axis=-1))
    return np.array(chosen)


def coarsen(mesh_pos, cells, count, keep=()):
    """Sub-sample a triangle mesh.  Returns (kept node indices [K], edges [M,2] i<j, faces [F,3]) with
    edges / faces indexing into the kept list; connectivity comes from the fine mesh (cluster adjacency)."""
    kept = farthest_points(mesh_pos, count, keep)
    if len(kept) == len(mesh_pos):
        cluster = np.empty(len(mesh_pos), int)
        cluster[kept] = np.arange(len(kept))
    else:
        cluster = np.linalg.norm(mesh_pos[:, None] - mesh_pos[kept][None], axis=-1).argmin(1)
    tri = cluster[cells]                                                # [C,3] cluster of each corner
    pairs = np.concatenate((tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]))
    pairs = np.sort(pairs[pairs[:, 0] != pairs[:, 1]], axis=1)
    edges = np.unique(pairs, axis=0)
    distinct = (tri[:, 0] != tri[:, 1]) & (tri[:, 1] != tri[:, 2]) & (tri[:, 0] != tri[:, 2])
    faces = tri[distinct]
    _, first = np.unique(np.sort(faces, axis=1), axis=0, return_index=True)
    return kept, edges, faces[np.sort(first)]


def connected(count, edges) -> bool:
    parent = list(range(count))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for a, b in edges:
        parent[find(int(a))] = find(int(b))
    return len({find(i) for i in range(count)}) == 1


# ----------------------------------------------------------------------------- conversion
def to_y_up(points):
    """Source Z-up (x, y, z) -> Y-up (x, z, -y)."""
    return np.stack((points[..., 0], points[..., 2], -points[..., 1]), -1)


def resample(series, positions):
    """Linear interpolation of series [T,...] at fractional frame positions [K]."""
    low = np.clip(np.floor(positions).astype(int), 0, len(series) - 2)
    weight = (positions - low).reshape(-1, *([1] * (series.ndim - 1))).astype(series.dtype)
    return series[low] * (1 - weight) + series[low + 1] * weight


def convert_trajectory(traj, source_dt, scale=.36, max_verts=128, handle_height=1., steps=0):
    """One MeshGraphNets cloth trajectory -> episode dict in the soft shard layout (unpadded), or (None, reason)."""
    node_type = traj['node_type'][0, :, 0]
    if (traj['node_type'] != traj['node_type'][:1]).any():
        return None, 'node types change over time'
    world = traj['world_pos'].astype(np.float64)
    if not np.isfinite(world).all():
        return None, 'non-finite positions'
    cloth = np.nonzero((node_type == NODE_NORMAL) | (node_type == NODE_HANDLE))[0]
    obstacle = np.nonzero(node_type == NODE_OBSTACLE)[0]
    if set(np.unique(node_type)) - {NODE_NORMAL, NODE_OBSTACLE, NODE_HANDLE}:
        return None, 'unknown node types'
    index = -np.ones(len(node_type), int)
    index[cloth] = np.arange(len(cloth))
    cells = traj['cells'][0]
    cells = index[cells[(index[cells] >= 0).all(1)]]                    # triangles of the cloth only
    mesh_pos = traj['mesh_pos'][0][cloth].astype(np.float64)
    handles = np.nonzero(node_type[cloth] == NODE_HANDLE)[0]
    kept, edges, faces = coarsen(mesh_pos, cells, max_verts, handles)
    if not connected(len(kept), edges):
        return None, 'coarse mesh is not connected'
    # time: similarity rescale keeps gravity when time scales with sqrt(length scale)
    time_scale = np.sqrt(scale)
    stride = STEP / (source_dt * time_scale)                            # source frames per 0.05 s step
    count = int(np.floor((len(world) - 1) / stride + 1e-6)) if not steps else steps
    positions = np.arange(count + 1) * stride
    if positions[-1] > len(world) - 1 + 1e-6:
        return None, 'trajectory too short'
    # material -> world scale from the fine mesh (median edge ratio at frame 0)
    fine = np.unique(np.sort(np.concatenate((cells[:, [0, 1]], cells[:, [1, 2]], cells[:, [2, 0]])), axis=1), axis=0)
    material = np.linalg.norm(mesh_pos[fine[:, 0]] - mesh_pos[fine[:, 1]], axis=-1)
    actual = np.linalg.norm(world[0, cloth[fine[:, 0]]] - world[0, cloth[fine[:, 1]]], axis=-1)
    ratio = float(np.median(actual / np.maximum(material, 1e-9)))
    verts = to_y_up(resample(world[:, cloth[kept]], positions)) * scale
    pinned = np.isin(kept, handles)
    lift = handle_height - (verts[0, pinned, 1].mean() if pinned.any() else verts[..., 1].max())
    verts[..., 1] += lift
    # finite-difference velocities (one source frame on either side, in rescaled units)
    eps = .5
    ahead, behind = resample(world[:, cloth[kept]], np.clip(positions + eps, 0, len(world) - 1)), resample(world[:, cloth[kept]], np.clip(positions - eps, 0, len(world) - 1))
    span = (np.clip(positions + eps, 0, len(world) - 1) - np.clip(positions - eps, 0, len(world) - 1)) * source_dt * time_scale
    vert_vel = to_y_up(ahead - behind) * scale / span[:, None, None]
    out = dict(verts=verts, vert_vel=vert_vel, pinned=pinned, rest_edges=edges, faces=faces,
               rest_len=np.linalg.norm(mesh_pos[kept][edges[:, 0]] - mesh_pos[kept][edges[:, 1]], axis=-1) * ratio * scale,
               material_to_world=ratio, fine_nodes=len(cloth), stride=stride, lift=float(lift))
    # obstacle: a sphere (kinematic), kept exact as a rounded box
    states = np.zeros((count + 1, 0, 13))
    props = np.zeros((0, 10))
    if len(obstacle):
        points = world[:, obstacle]
        centre = (points.max(1) + points.min(1)) / 2                    # [T,3]
        radial = np.linalg.norm(points - centre[:, None], axis=-1)
        radius = float(radial.mean())
        if radial.std() > .02 * radius:
            return None, 'obstacle is not a sphere'
        c = to_y_up(resample(centre, positions)) * scale
        c[:, 1] += lift
        c_ahead, c_behind = resample(centre, np.clip(positions + eps, 0, len(world) - 1)), resample(centre, np.clip(positions - eps, 0, len(world) - 1))
        velocity = to_y_up(c_ahead - c_behind) * scale / span[:, None]
        states = np.zeros((count + 1, 1, 13))
        states[:, 0, :3], states[:, 0, 6], states[:, 0, 7:10] = c, 1., velocity
        r = radius * scale
        props = np.array([[2 * r, 2 * r, 2 * r, 1., PLACEHOLDER_PROPS['friction'], 1., 0., 0., 0., r]])
        gap = np.linalg.norm(verts[:, :, None] - states[:, None, :, :3], axis=-1).min() - r
        out.update(sphere_radius=r, min_gap=float(gap))
    area = float(np.ptp(mesh_pos[:, 0]) * np.ptp(mesh_pos[:, 1])) * (ratio * scale) ** 2
    out.update(states=states, props=props,
               soft_props=np.array([PLACEHOLDER_PROPS['mass_kg'], PLACEHOLDER_PROPS['stiffness_norm'], PLACEHOLDER_PROPS['damping'],
                                    PLACEHOLDER_PROPS['friction'], 1., area]),
               context=np.array([1., 1., 0., GRAVITY, 0., 0., 0., 0., .5, 0.]))
    if out['verts'][..., 1].min() < .05:
        return None, 'cloth reaches the analytic floor (raise --handle-height)'
    return out, None


def pack(episodes, kind, max_verts=128, max_edges=768, max_faces=256):
    """Stack converted episodes into padded arrays (soft_v1 layout + 'pinned')."""
    count, frames = len(episodes), len(episodes[0]['verts'])
    boxes = max(e['states'].shape[1] for e in episodes)
    out = dict(verts=np.zeros((count, frames, max_verts, 3), np.float32), vert_vel=np.zeros((count, frames, max_verts, 3), np.float32),
               vert_mask=np.zeros((count, max_verts), bool), pinned=np.zeros((count, max_verts), bool),
               rest_edges=-np.ones((count, max_edges, 2), np.int16), rest_len=np.zeros((count, max_edges), np.float32),
               faces=-np.ones((count, max_faces, 3), np.int16), soft_props=np.zeros((count, 6), np.float32),
               states=np.zeros((count, frames, boxes, 13), np.float32), props=np.zeros((count, boxes, 10), np.float32),
               mask=np.zeros((count, boxes), bool), context=np.zeros((count, 10), np.float32), rest=np.zeros((count, frames), bool),
               n_verts=np.zeros(count, np.int16), n_edges=np.zeros(count, np.int16), n_faces=np.zeros(count, np.int16),
               kind=np.array([kind] * count), soft_shape=np.array(['cloth'] * count), episode_index=np.zeros(count, np.int64))
    out['states'][..., 6] = 1.
    out['props'][..., 3] = 1.
    for i, e in enumerate(episodes):
        v, m, f, n = e['verts'].shape[1], len(e['rest_edges']), min(len(e['faces']), max_faces), e['states'].shape[1]
        if m > max_edges:
            raise ValueError(f'{m} edges exceed the padded size {max_edges}')
        out['verts'][i, :, :v], out['vert_vel'][i, :, :v] = e['verts'], e['vert_vel']
        out['vert_mask'][i, :v], out['pinned'][i, :v] = True, e['pinned']
        out['rest_edges'][i, :m], out['rest_len'][i, :m], out['faces'][i, :f] = e['rest_edges'], e['rest_len'], e['faces'][:f]
        out['soft_props'][i], out['context'][i] = e['soft_props'], e['context']
        out['states'][i, :, :n], out['props'][i, :n], out['mask'][i, :n] = e['states'], e['props'], True
        out['n_verts'][i], out['n_edges'][i], out['n_faces'][i], out['episode_index'][i] = v, m, f, e['source_index']
        speed = np.linalg.norm(np.diff(e['verts'], axis=0), axis=-1).max(-1) / STEP
        out['rest'][i] = np.concatenate((speed[:1], speed)) < .03
    return out


def displacement_stats(verts, vert_mask, pinned=None):
    """Per-step displacement statistics (m) of the free vertices of episodes [E,T+1,V,3]."""
    free = vert_mask if pinned is None else vert_mask & ~pinned
    step = np.linalg.norm(np.diff(verts, axis=1), axis=-1)[np.broadcast_to(free[:, None], verts[:, 1:].shape[:-1])]
    acc = np.linalg.norm(np.diff(verts, 2, axis=1), axis=-1)[np.broadcast_to(free[:, None], verts[:, 2:].shape[:-1])] / STEP ** 2
    return dict(step_mean_m=float(step.mean()), step_median_m=float(np.median(step)), step_p95_m=float(np.percentile(step, 95)),
                step_max_m=float(step.max()), still_fraction_below_0p5mm=float((step < .0005).mean()),
                accel_mean_ms2=float(acc.mean()), accel_p95_ms2=float(np.percentile(acc, 95)))


def split_of(index: int) -> str:
    return 'val' if index % 10 == 8 else 'test' if index % 10 == 9 else 'train'


def convert_set(source: Path, name: str, out: Path, args) -> dict:
    meta = json.loads((source / name / 'meta.json').read_text())
    files = [f for f in ('valid', 'train', 'test') if (source / name / f'{f}.tfrecord').exists()]
    if not files:
        return dict(converted=False, reason='no tfrecord present locally')
    file = 'valid' if 'valid' in files else files[0]                    # one file only: the local splits may be copies
    episodes = dict(train=[], val=[], test=[])
    dropped = {}
    for index, record in enumerate(records(source / name / f'{file}.tfrecord', args.limit)):
        episode, reason = convert_trajectory(trajectory(record, meta), meta['dt'], args.scale, args.max_verts, args.handle_height)
        if episode is None:
            dropped[reason] = dropped.get(reason, 0) + 1
            continue
        episode['source_index'] = index
        episodes[split_of(index)].append(episode)
    report = dict(converted=True, source_file=f'{name}/{file}.tfrecord', source_dt_s=meta['dt'], source_frames=meta['trajectory_length'],
                  source_nodes=meta['features']['world_pos']['shape'][1], dropped=dropped, splits={})
    for split, items in episodes.items():
        (out / split).mkdir(parents=True, exist_ok=True)
        for old in (out / split).glob(f'shard_{name}_*.npz'):
            old.unlink()
        if not items:
            continue
        arrays = pack(items, f'mgn_{name}', args.max_verts)
        np.savez_compressed(out / split / f'shard_{name}_00000.npz', **arrays)
        report['splits'][split] = dict(
            episodes=len(items), frames=int(arrays['verts'].shape[1]), mean_vertices=float(arrays['n_verts'].mean()),
            mean_edges=float(arrays['n_edges'].mean()), mean_faces=float(arrays['n_faces'].mean()),
            pinned_per_episode=float(arrays['pinned'].sum(1).mean()), fine_nodes=int(items[0]['fine_nodes']),
            material_to_world=float(np.mean([e['material_to_world'] for e in items])),
            source_frames_per_step=float(items[0]['stride']), sphere_radius_m=float(np.mean([e.get('sphere_radius', 0.) for e in items])),
            min_vertex_sphere_gap_m=float(min(e.get('min_gap', np.inf) for e in items)),
            mean_abs_rest_strain_frame0=float(np.mean([np.abs(np.linalg.norm(e['verts'][0][e['rest_edges'][:, 0]] - e['verts'][0][e['rest_edges'][:, 1]], axis=-1) / e['rest_len'] - 1).mean() for e in items])),
            extent_m=[float(x) for x in np.ptp(np.concatenate([e['verts'].reshape(-1, 3) for e in items]), axis=0)],
            min_height_m=float(min(e['verts'][..., 1].min() for e in items)),
            displacement=displacement_stats(arrays['verts'], arrays['vert_mask'], arrays['pinned']))
    return report


def reference_stats(root: Path, limit=4):
    """Per-step displacement statistics of the first shards of a soft data set (for the comparison in meta.json)."""
    files = sorted((root / 'train').glob('shard_*.npz'))[:limit]
    if not files:
        return None
    parts = [np.load(f) for f in files]
    return displacement_stats(np.concatenate([p['verts'] for p in parts]), np.concatenate([p['vert_mask'] for p in parts]))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--src', type=Path, default=Path('data/external/meshgraphnets'))
    p.add_argument('--out', type=Path, default=Path('data/external_v2/meshgraphnets_soft'))
    p.add_argument('--sets', default=','.join(SETS))
    p.add_argument('--scale', type=float, default=.36, help='length scale; times scale with its square root (gravity unchanged)')
    p.add_argument('--max-verts', type=int, default=128)
    p.add_argument('--handle-height', type=float, default=1.)
    p.add_argument('--limit', type=int, default=0, help='first n trajectories only (0 = all)')
    p.add_argument('--compare', type=Path, default=Path('data/soft_v1'), help='soft data set whose displacement statistics go into meta.json')
    args = p.parse_args(argv)
    began = time.perf_counter()
    report = dict(
        source='DeepMind MeshGraphNets (Pfaff et al., ICLR 2021), https://github.com/google-deepmind/deepmind-research/tree/master/meshgraphnets',
        licence_status='no explicit data licence; used by team decision under the repository Apache-2.0 licence; model not publicly released',
        experimental=True, step_s=STEP, length_scale=args.scale, time_scale=float(np.sqrt(args.scale)), gravity_assumed=GRAVITY,
        axes='(x, y, z) Z-up -> (x, z, -y) Y-up', handle_height_m=args.handle_height, max_vertices=args.max_verts,
        split_rule='trajectory index % 10: 8 -> val, 9 -> test, else train (our own partition of ONE upstream file)',
        placeholder_soft_props=PLACEHOLDER_PROPS, pinned='HANDLE nodes -> pinned [E,V] (kinematic, prescribed from ground truth)',
        obstacle='sphere kept exact: rigid slot with size = diameter and props[9] = rounding radius; kinematic, prescribed',
        sets={}, skipped=dict(SKIPPED))
    for name in [s for s in args.sets.split(',') if s]:
        if name not in SETS:
            raise SystemExit(f'{name}: not convertible ({SKIPPED.get(name, "unknown set")})')
        report['sets'][name] = convert_set(args.src, name, args.out, args)
        print(f'[{name}] ' + json.dumps(report['sets'][name], indent=1), flush=True)
    if args.compare and (args.compare / 'train').is_dir():
        report['reference'] = dict(data=str(args.compare), displacement=reference_stats(args.compare))
        print('[reference] ' + json.dumps(report['reference']), flush=True)
    report['seconds'] = time.perf_counter() - began
    (args.out / 'meta.json').write_text(json.dumps(report, indent=1))
    print(f'[done] {args.out} in {report["seconds"]:.0f}s', flush=True)
    return report


if __name__ == '__main__':
    main()
