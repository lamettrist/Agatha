"""Fast tests for the RBO articulated-object converter (synthetic recording; no network, no downloaded data)."""
import csv
import io
import math
import sys
import tarfile
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import prepare_external_data as pe
import prepare_external_joints as pj

T0 = 1_500_000_000_000_000_000
URDF_ZERO_OPEN = 2.0                 # the synthetic lid is closed at URDF angle -2.0 (as the real laptop model: URDF 0 = open)
TABLE_Z = .5


def transform(rotation=np.eye(3), translation=(0, 0, 0)):
    t = np.eye(4); t[:3, :3] = rotation; t[:3, 3] = translation
    return t


def box_vertices(lo, hi):
    return np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])], float)


def dae(vertices):
    """Minimal COLLADA: vertices stored at double size under a node matrix that scales by 0.5."""
    numbers = ' '.join(f'{v:.9g}' for v in (vertices * 2).reshape(-1))
    return f'''<?xml version="1.0" encoding="utf-8"?>
<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">
  <asset><unit name="meter" meter="1"/><up_axis>Z_UP</up_axis></asset>
  <library_geometries><geometry id="g-mesh" name="g"><mesh>
    <source id="g-mesh-positions"><float_array id="g-mesh-positions-array" count="{vertices.size}">{numbers}</float_array></source>
    <vertices id="g-mesh-vertices"><input semantic="POSITION" source="#g-mesh-positions"/></vertices>
  </mesh></geometry></library_geometries>
  <library_visual_scenes><visual_scene id="Scene"><node id="g" name="g" type="NODE">
    <matrix sid="transform">0.5 0 0 0 0 0.5 0 0 0 0 0.5 0 0 0 0 1</matrix><instance_geometry url="#g-mesh"/>
  </node></visual_scene></library_visual_scenes>
</COLLADA>'''.encode()


def origin_tag(t):
    """<origin> of a 4x4 transform (rpy recovered from the matrix; pitch away from +-90 deg in this fixture)."""
    r = t[:3, :3]
    rpy = (math.atan2(r[2, 1], r[2, 2]), -math.asin(r[2, 0]), math.atan2(r[1, 0], r[0, 0]))
    return f"<origin xyz='{t[0, 3]:.12f} {t[1, 3]:.12f} {t[2, 3]:.12f}' rpy='{rpy[0]:.12f} {rpy[1]:.12f} {rpy[2]:.12f}'/>"


class Fixture:
    """A 'laptop with a drawer': root slab, hinged lid (URDF zero = open by 2 rad), prismatic drawer; Z-up world."""
    def __init__(self):
        self.m0 = transform(pj.rpy_matrix((.4, -.3, 1.1)), (.05, -.02, .03))           # root marker frame in the object frame
        self.f1 = transform(pj.rpy_matrix((-1., .5, .2)), (.02, .03, -.04))            # lid marker frame in the lid link frame
        self.f2 = transform(pj.rpy_matrix((.3, .9, -2.)), (0, 0, 0))                   # drawer marker frame in its link frame
        to_link0 = np.linalg.inv(self.m0)                                               # object frame -> rb0 (= marker) frame
        self.point = lambda p: to_link0[:3, :3] @ np.asarray(p, float) + to_link0[:3, 3]
        self.vector = lambda v: to_link0[:3, :3] @ np.asarray(v, float)
        self.hinge_origin, self.hinge_axis = self.point((0, .1, .01)), self.vector((1, 0, 0))
        self.slide_origin, self.slide_axis = self.point((0, 0, -.03)), self.vector((0, -1, 0))
        self.root_vertices = box_vertices((-.15, -.1, -.01), (.15, .1, .01))            # object frame, z up
        lid_closed = box_vertices((-.15, -.1, .01), (.15, .1, .03))
        drawer = box_vertices((-.1, -.05, -.05), (.1, .05, -.01))
        in_link0 = lambda v: v @ to_link0[:3, :3].T + to_link0[:3, 3]
        self.lid_vertices = (in_link0(lid_closed) - self.hinge_origin) @ pj.axis_rotation(self.hinge_axis, URDF_ZERO_OPEN).T
        self.drawer_vertices = in_link0(drawer) - self.slide_origin

    def urdf(self):
        hinge = ' '.join(f'{v:.12f}' for v in self.hinge_origin); slide = ' '.join(f'{v:.12f}' for v in self.slide_origin)
        axis1 = ' '.join(f'{v:.12f}' for v in self.hinge_axis); axis2 = ' '.join(f'{v:.12f}' for v in self.slide_axis)
        mesh = lambda name: f'<geometry><mesh filename="package://articulated_objects_db/data/objects/toy/meshes/{name}.dae" scale="1 1 1"/></geometry>'
        return f'''<?xml version="1.0"?>
<robot name="toy">
  <link name ="rb0"><visual>{origin_tag(np.linalg.inv(self.m0))}{mesh("base")}</visual></link>
  <joint name="j_0_1" type="revolute"><parent link="rb0"/><child link="rb1"/><origin xyz="{hinge}" rpy="0 0 0 "/><axis xyz="{axis1}"/>
    <limit effort="30" velocity="1.0" lower="-3.1416" upper="3.1416"/></joint>
  <joint name="j_0_fixed" type="fixed"><parent link="rb1"/><child link="rb1_marker"/>{origin_tag(self.f1)}</joint>
  <joint name="j_0_2" type="prismatic"><parent link="rb0"/><child link="rb2"/><origin xyz="{slide}" rpy="0 0 0 "/><axis xyz="{axis2}"/>
    <limit effort="30" velocity="1.0" lower="-3.1416" upper="3.1416"/></joint>
  <joint name="j_1_fixed" type="fixed"><parent link="rb2"/><child link="rb2_marker"/>{origin_tag(self.f2)}</joint>
  <link name ="rb1"></link><link name ="rb1_marker"><visual>{origin_tag(np.linalg.inv(self.f1))}{mesh("lid")}</visual></link>
  <link name ="rb2"></link><link name ="rb2_marker"><visual>{origin_tag(np.linalg.inv(self.f2))}{mesh("drawer")}</visual></link>
</robot>'''

    def archive(self, path: Path):
        files = {'toy/configuration_2017-01-01/toy.urdf': self.urdf().encode(),
                 'toy/configuration_2017-01-01/toy_in.yaml': b'name: toy\nbase_marker: "/A"\nlink_ids: {7: "0", 3: "1", 5: "2"}\njoints:\n- {rb1_id: 7, rb2_id: 3, type: REVOLUTE}\n',
                 'toy/configuration_2017-01-01/toy_out.yaml': b'!!python/object/apply:os.system ["false"]\n',      # must never be parsed
                 'toy/meshes/base.dae': dae(self.root_vertices), 'toy/meshes/lid.dae': dae(self.lid_vertices), 'toy/meshes/drawer.dae': dae(self.drawer_vertices)}
        with tarfile.open(path, 'w:gz') as tar:
            for name, data in files.items():
                info = tarfile.TarInfo(name); info.size = len(data); tar.addfile(info, io.BytesIO(data))
        return path

    def marker_poses(self, q, d, yaw=.3, position=(1., 2., TABLE_Z + .01)):
        """World poses (pos, quat xyzw) of the three marker frames for URDF hinge angle q and drawer displacement d."""
        world_object = transform(np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]]), position)
        link0 = world_object @ self.m0
        link1 = link0 @ transform(pj.axis_rotation(self.hinge_axis, q), self.hinge_origin)
        link2 = link0 @ transform(np.eye(3), self.slide_origin + d * self.slide_axis)
        return {7: link0, 3: link1 @ self.f1, 5: link2 @ self.f2, 0: transform(np.eye(3), (9, 9, 9))}       # id 0: an unrelated camera body


def write_recording(folder: Path, fixture: Fixture, q, d, missing=(), wrong=False):
    folder.mkdir(parents=True, exist_ok=True)
    fields = ('header.seq', 'header.stamp', 'header.frame_id', 'ns', 'id', 'type', 'action', 'pose.position.x', 'pose.position.y', 'pose.position.z',
              'pose.orientation.x', 'pose.orientation.y', 'pose.orientation.z', 'pose.orientation.w', 'scale.x', 'scale.y', 'scale.z', 'text')
    ids = (0, 3, 7, 5)
    with open(folder / 'rb_poses_array.csv', 'w', newline='') as handle:
        writer = csv.writer(handle); writer.writerow(['%time'] + [f'field.markers{k}.{f}' for k in range(len(ids)) for f in fields])
        for t, (angle, slide) in enumerate(zip(q, d)):
            poses = fixture.marker_poses(angle, slide); stamp = T0 + t * 10_000_000; row = [stamp + 123]
            for k, body in enumerate(ids):
                pose = poses[7 if (wrong and body == 3) else body]
                quat = pj.matrix_to_quat(pose[:3, :3])
                values = [t, stamp, '/map', '', body, 1, 0, *(f'{v:.12f}' for v in pose[:3, 3]), *(f'{v:.12f}' for v in quat), .2, .2, .2, '']
                if t in missing and body == 3: values = [''] * len(fields)
                row += values
            writer.writerow(row)
    with open(folder / 'toy_joint_states.csv', 'w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['%time', 'field.header.seq', 'field.header.stamp', 'field.header.frame_id', 'field.name0', 'field.name1', 'field.position0',
                         'field.position1', 'field.velocity0', 'field.velocity1'])
        for t, (angle, slide) in enumerate(zip(q, d)): writer.writerow([T0 + t * 10_000_000 + 99, t, T0 + t * 10_000_000, '', 'j_0_1', 'j_0_2', angle, slide, 0., 0.])


@pytest.fixture(scope='module')
def toy(tmp_path_factory):
    root = tmp_path_factory.mktemp('rbo'); fixture = Fixture()
    model = pj.load_model(fixture.archive(root / 'toy.tar.gz'), 'toy', '2017-01-01')
    frames = 301; s = np.linspace(0, 1, frames)
    q = -URDF_ZERO_OPEN + 1.5 * (.5 - .5 * np.cos(math.pi * s))               # closed -> 1.5 rad open
    d = .08 * (.5 - .5 * np.cos(2 * math.pi * s))
    write_recording(root / 'toy01_o', fixture, q, d)
    return dict(root=root, fixture=fixture, model=model, q=q, d=d, info=dict(object='toy', date='2017-01-01', ft=False, comment=''))


def test_rotation_helpers():
    rng = np.random.default_rng(0)
    for rpy in rng.uniform(-1.4, 1.4, (20, 3)):
        r = pj.rpy_matrix(rpy)
        assert np.allclose(r @ r.T, np.eye(3), atol=1e-12) and np.isclose(np.linalg.det(r), 1)
        assert np.allclose(pe.q_to_matrix(pj.matrix_to_quat(r)), r, atol=1e-9)
    assert np.allclose(pj.rpy_matrix((0, 0, math.pi / 2)) @ [1, 0, 0], [0, 1, 0])                 # yaw: x -> y
    assert np.allclose(pj.rpy_matrix((math.pi / 2, 0, 0)) @ [0, 1, 0], [0, 0, 1])                 # roll: y -> z
    assert np.allclose(pj.axis_rotation(np.array([0, 0, 1.]), math.pi / 2) @ [1, 0, 0], [0, 1, 0])
    assert np.allclose(pj.C_ZUP_TO_YUP @ [0, 0, 1], [0, 1, 0]) and np.isclose(np.linalg.det(pj.C_ZUP_TO_YUP), 1)
    assert np.allclose(pj.matrix_to_quat(np.diag([1., -1, -1])), [1, 0, 0, 0])                    # 180 deg branch


def test_model_parsing(toy):
    model, fixture = toy['model'], toy['fixture']
    assert model['marker_of_link'] == {0: 7, 1: 3, 2: 5}
    assert [(j['name'], j['type'], j['parent'], j['child']) for j in model['joints']] == [('j_0_1', 'hinge', 0, 1), ('j_0_2', 'prismatic', 0, 2)]
    assert np.allclose(model['joints'][0]['axis'], fixture.hinge_axis) and np.allclose(model['joints'][1]['origin'], fixture.slide_origin)
    assert np.allclose(model['links'][0]['marker_T'], np.eye(4)) and np.allclose(model['links'][1]['marker_T'], fixture.f1, atol=1e-9)
    # mesh vertices land in the link frame (node matrix, visual origin and marker transform all applied)
    assert np.allclose(model['links'][1]['vertices'], fixture.lid_vertices, atol=1e-7)
    assert np.allclose(model['links'][0]['vertices'], fixture.root_vertices @ np.linalg.inv(fixture.m0)[:3, :3].T + np.linalg.inv(fixture.m0)[:3, 3], atol=1e-7)
    with pytest.raises(FileNotFoundError): pj.load_model(toy['root'] / 'toy.tar.gz', 'toy', '2099-01-01')
    with pytest.raises(ValueError): pj.parse_urdf('<robot><link name="rb0"/><joint name="j" type="floating"><parent link="rb0"/><child link="rb1"/></joint></robot>')


def test_boxes_and_zero_convention(toy):
    stamps, poses = pj.read_rb_poses(toy['root'] / 'toy01_o' / 'rb_poses_array.csv')
    pos, rot, valid = pj.link_frames(toy['model'], poses)
    assert valid.all() and set(poses) == {0, 3, 5, 7}
    boxes = pj.fit_boxes(toy['model'], rot[0])
    assert np.allclose(sorted(boxes['size'][0]), [.02, .2, .3], atol=1e-4) and np.isclose(boxes['size'][0][1], .02, atol=1e-4)   # y is up
    assert np.allclose(sorted(boxes['size'][1]), [.02, .2, .3], atol=2e-3)                          # tight lid box: zero = closed pose
    assert abs(boxes['zero'][0] + URDF_ZERO_OPEN) < .01 and boxes['zero'][1] == 0
    centre, quat = pj.body_states(pos, rot, boxes)
    assert np.allclose(pe.q_to_matrix(quat[0, 0])[:, 1], [0, 1, 0], atol=1e-6)                      # root box upright in the Y-up world
    assert np.allclose(centre[0, 0], [1., TABLE_Z + .01, -2.], atol=1e-6)                           # (x, y, z) -> (x, z, -y)
    rows = boxes['joints']
    assert rows[0, 2] == 1 and rows[1, 4] == 1 and rows[0, 0] == 0 and rows[0, 1] == 1 and rows[1, 1] == 2
    q = pj.numpy_joint_coordinates(centre, quat, rows)
    assert np.allclose(q[:, 0], toy['q'] - boxes['zero'][0], atol=1e-6) and np.allclose(q[:, 1], toy['d'], atol=1e-6)
    v = pj.violations(centre, quat, rows)                                                           # agatha.joint_gnn on the same states
    assert np.abs(v['coordinate'] - q).max() < 2e-5
    assert v['anchor_gap'].max() < 2e-5 and v['off_axis'].max() < 2e-3 and v['limit_excess'].max() == 0


def test_convert_recording_and_shard_format(toy):
    episodes, report = pj.convert_recording('toy01', toy['root'] / 'toy01_o', toy['model'], toy['info'], steps=20, stride=20)
    assert len(episodes) == 3 and report['valid_frames'] == 301 and report['dropped_frames'] == 0
    for joint in report['joints']:
        unit = 'deg' if joint['type'] == 'hinge' else 'mm'
        assert joint['anchor_gap_mm']['max'] < .05 and joint['off_axis_deg']['max'] < .1
        assert joint[f'coordinate_vs_dataset_{unit}']['max'] < .05 and joint[f'coordinate_vs_negated_dataset_{unit}']['mean'] > 5
    assert report['free_motion']['verified'] == 0
    data = pj.stack(episodes)
    assert data['states'].shape == (3, 21, 12, 13) and data['states'].dtype == np.float32
    assert data['props'].shape == (3, 12, 9) and data['context'].shape == (3, 10) and data['joints'].shape == (3, 12, 18)
    assert data['joint_pos'].shape == (3, 21, 12) and data['group'].dtype == np.int8 and data['joint_mask'].sum() == 6
    for key in ('mask', 'contact_support', 'contact_floor', 'contact_object', 'rest', 'fell', 'tipped', 'kind', 'chaotic', 'live_physics',
                'episode_index', 'init_penetration', 'joint_pos_source', 'joint_zero', 'driven', 'free_motion', 'source_id', 'start_s'):
        assert key in data and len(data[key]) == 3
    assert data['driven'].all() and not data['free_motion'].any() and (data['kind'] == 'rbo_toy').all()
    assert (data['mask'].sum(1) == 3).all() and (data['group'][:, :3] == 0).all() and (data['group'][:, 3:] == -1).all()
    assert np.allclose(np.linalg.norm(data['states'][..., 3:7], axis=-1), 1, atol=1e-5)
    # the root slab rests on the support slab: bottom at support_height, flagged as supported / at rest, inside the footprint
    assert np.allclose(data['context'][:, 2], pj.SUPPORT_HEIGHT) and np.allclose(data['states'][:, :, 0, 1], pj.SUPPORT_HEIGHT + .01, atol=1e-4)
    assert data['contact_support'][:, :, 0].all() and data['rest'][:, :, 0].all()
    assert (np.abs(data['states'][:, :, :3, 0]) < data['context'][:, None, None, 0] / 2).all()
    # the lid moves (finite-difference velocities, hinge spin about the box x / z axis only), the root does not
    assert np.abs(data['states'][:, :, 0, 7:]).max() < 1e-4 and np.abs(data['states'][:, :, 1, 10:]).max() > .3
    middle = episodes[1]; step = (middle['joint_pos'][11, 0] - middle['joint_pos'][9, 0]) / (2 * pj.STEP)
    assert np.isclose(np.linalg.norm(middle['states'][10, 1, 10:]), abs(step), rtol=.02)
    check = pj.shard_violations(data)
    assert check['joint_frames'] == 3 * 21 * 2 and check['anchor_gap_mm']['max'] < .05 and check['joint_pos_vs_joint_gnn_max'] < 2e-5
    assert np.nanmax(np.abs(data['joint_pos'][:, :, :2] - data['joint_pos_source'][:, :, :2])) < 2e-3      # 20 Hz linear resampling of a smooth curve


def test_dropouts_and_misfit(toy, tmp_path):
    q, d = toy['q'], toy['d']
    write_recording(tmp_path / 'gap_o', toy['fixture'], q, d, missing=range(60, 70))
    episodes, report = pj.convert_recording('gap', tmp_path / 'gap_o', toy['model'], toy['info'], steps=20, stride=20)
    assert report['dropped_frames'] == 10 and len(episodes) == 2                                    # windows slide past the dropout (0.60-0.69 s)
    assert all(e['start_s'] > .69 for e in episodes)
    write_recording(tmp_path / 'wrong_o', toy['fixture'], q, d, wrong=True)                         # lid marker reports the root pose
    with pytest.raises(ValueError, match='does not fit'):
        pj.convert_recording('wrong', tmp_path / 'wrong_o', toy['model'], toy['info'], steps=20, stride=20)
    poses = pj.read_rb_poses(tmp_path / 'gap_o' / 'rb_poses_array.csv')[1]
    with pytest.raises(KeyError): pj.link_frames(toy['model'], {k: v for k, v in poses.items() if k != 5})


def coasting(t, start=1., speed=2., decay=1.5):
    """Joint at rest, launched at `start` with `speed`, then decelerating exponentially to rest."""
    return np.where(t < start, 0., speed / decay * (1 - np.exp(-decay * np.clip(t - start, 0, None))))[:, None]


def test_free_motion_needs_force_and_stick_evidence():
    t = np.arange(0, 6, .01); q = coasting(t)
    quiet = (t, np.tile([0., 3.6, 2.], (len(t), 1)) + .05 * np.sin(40 * t)[:, None])
    still = (t, np.tile([.5, .1, 0.], (len(t), 1)))
    plain = pj.free_motion_segments(t, q, ['hinge'])
    assert len(plain) == 1 and plain[0]['pattern'] == 'decelerating_tail' and not plain[0]['verified'] and plain[0]['duration_s'] > 1
    assert plain[0]['ft_max_force_dev_N'] is None                                                   # no F/T: a candidate, never usable
    assert [s['verified'] for s in pj.free_motion_segments(t, q, ['hinge'], quiet, still)] == [True]
    assert [s['verified'] for s in pj.free_motion_segments(t, q, ['hinge'], quiet)] == [False]      # force quiet but the stick is not tracked
    following = (t, np.stack((.5 + .1 * q[:, 0], .1 + 0 * t, 0 * t), -1))                           # the stick travels with the part: it is pushing
    assert [s['verified'] for s in pj.free_motion_segments(t, q, ['hinge'], quiet, following)] == [False]
    loaded = (t, quiet[1] + np.where((t > 1) & (t < 3), 3., 0.)[:, None])
    assert [s['verified'] for s in pj.free_motion_segments(t, q, ['hinge'], loaded, still)] == [False]
    slow = (.1 * np.clip(t - 1, 0, 3))[:, None]                                                     # quasi-static 0.1 rad/s push: never launched
    assert pj.free_motion_segments(t, slow, ['hinge']) == []
    falling = np.where(t < 1, 0., np.minimum(2. * np.clip(t - 1, 0, None) ** 2, 1.))[:, None]       # accelerates, then stops dead at 1 rad
    assert [s['pattern'] for s in pj.free_motion_segments(t, falling, ['hinge'])] == ['accelerating_abrupt_stop']


def test_csv_readers(toy, tmp_path):
    stamps, names, q = pj.read_joint_states(toy['root'] / 'toy01_o' / 'toy_joint_states.csv')
    assert names == ['j_0_1', 'j_0_2'] and q.shape == (301, 2) and stamps[1] - stamps[0] == 10_000_000
    (tmp_path / 'index.csv').write_text('Name,Object,Background,force/torque sensor used,recording date(marker set id),comment\n'
                                        'toy01,toy,plain,1,2017-01-01,hello\n')
    assert pj.read_index(tmp_path / 'index.csv') == {'toy01': dict(object='toy', date='2017-01-01', ft=True, comment='hello')}
    (tmp_path / 'tf.csv').write_text('%time,field.transforms0.header.seq,field.transforms0.header.stamp,field.transforms0.header.frame_id,'
                                     'field.transforms0.child_frame_id,field.transforms0.transform.translation.x,field.transforms0.transform.translation.y,'
                                     'field.transforms0.transform.translation.z\n'
                                     + ''.join(f'{T0 + k},{k},{T0 + k * 10},/map,{"ftSensor" if k % 2 else "Other"},{k},0,1\n' for k in range(40)))
    stick = pj.read_tf_frame(tmp_path / 'tf.csv', 'ftSensor')
    assert stick[1].shape == (20, 3) and stick[1][0, 0] == 1 and pj.read_tf_frame(tmp_path / 'tf.csv', 'Nothing') is None
    with pytest.raises(ValueError): pj.download(['../evil'], tmp_path)
