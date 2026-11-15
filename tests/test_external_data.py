"""Fast unit tests for the external-data conversion (no network, no downloaded data needed)."""
import math
import sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import prepare_external_data as pe

FORMAT_KEYS = ('states', 'props', 'context', 'mask', 'contact_support', 'contact_floor', 'contact_object', 'rest', 'fell', 'tipped',
               'kind', 'chaotic', 'live_physics', 'episode_index', 'init_penetration')


def axis_quat(axis, angle):
    axis = np.asarray(axis, float); axis = axis / np.linalg.norm(axis)
    return np.array([*(axis * math.sin(angle / 2)), math.cos(angle / 2)])


def random_quats(count, seed=0):
    q = np.random.default_rng(seed).normal(size=(count, 4))
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def test_slerp_endpoints_midpoint_and_shortest_arc():
    a, b = axis_quat([0, 1, 0], 0.), axis_quat([0, 1, 0], math.pi / 2)
    assert np.allclose(pe.slerp(a, b, 0.), a) and np.allclose(pe.slerp(a, b, 1.), b)
    assert np.allclose(pe.slerp(a, b, .5), axis_quat([0, 1, 0], math.pi / 4), atol=1e-12)
    assert np.allclose(pe.slerp(a, -b, .5), axis_quat([0, 1, 0], math.pi / 4), atol=1e-12)      # -b is the same rotation
    assert np.allclose(np.linalg.norm(pe.slerp(random_quats(50, 1), random_quats(50, 2), .3), axis=-1), 1)
    assert np.allclose(pe.slerp(a, a, .7), a)                                                    # degenerate angle


def test_resampling_is_exact_for_linear_motion_and_constant_spin():
    t_src = np.arange(25) / 12.; omega = 1.7
    pos = np.stack((2 * t_src, 1 - .5 * t_src, 0 * t_src), -1)[:, None]                          # [T,1,3]
    quat = np.stack([axis_quat([0, 0, 1], omega * t) for t in t_src])[:, None]
    t_dst = np.arange(0, 2.0001, .05)
    p, q = pe.interp_linear(t_src, pos, t_dst), pe.interp_slerp(t_src, quat, t_dst)
    assert p.shape == (41, 1, 3) and q.shape == (41, 1, 4)
    assert np.allclose(p[:, 0], np.stack((2 * t_dst, 1 - .5 * t_dst, 0 * t_dst), -1), atol=1e-12)
    expected = np.stack([axis_quat([0, 0, 1], omega * t) for t in t_dst])
    assert np.allclose(np.abs((q[:, 0] * expected).sum(-1)), 1, atol=1e-12)
    assert ((q[1:] * q[:-1]).sum(-1) > 0).all()                                                  # sign-continuous
    with pytest.raises(ValueError): pe.interp_linear(t_src, pos, np.array([2.5]))                # never extrapolates


def test_finite_difference_velocities():
    t = np.arange(0, 1.0001, .05); omega = np.array([.3, -1.1, .6])
    pos = np.stack((1.5 * t, -9.81 / 2 * t * t, .2 * t), -1)[:, None]
    angle = np.linalg.norm(omega)
    quat = np.stack([axis_quat(omega, angle * s) for s in t])[:, None]
    quat[5:] *= -1                                                                               # a sign flip must not matter
    vel, ang = pe.finite_difference(t, pos, quat)
    assert np.allclose(vel[1:-1, 0, 1], -9.81 * t[1:-1], atol=1e-9)                              # central differences are exact for quadratics
    assert np.allclose(vel[:, 0, 0], 1.5) and np.allclose(ang[:, 0], omega, atol=1e-9)           # world-frame angular velocity


def test_zup_to_yup_is_a_proper_rotation_consistent_for_points_quats_and_sizes():
    assert np.allclose(pe.zup_to_yup_vec([0, 0, 1]), [0, 1, 0])                                  # up stays up
    basis = pe.zup_to_yup_vec(np.eye(3)).T
    assert np.isclose(np.linalg.det(basis), 1)                                                   # no reflection
    rng = np.random.default_rng(3); quat = random_quats(20, 4); size = rng.uniform(.1, 1, (20, 3)); pos = rng.normal(size=(20, 3))
    source = pe.box_corners(pos, quat, size)                                                     # Z-up world corners [20,8,3]
    converted = pe.box_corners(pe.zup_to_yup_vec(pos), pe.zup_to_yup_quat(quat), pe.zup_to_yup_size(size))
    expected = pe.zup_to_yup_vec(source)
    distance = np.linalg.norm(converted[:, :, None] - expected[:, None], axis=-1).min(-1)        # same corner SET
    assert distance.max() < 1e-9


def test_mirror_z_maps_left_handed_boxes_onto_their_mirror_image():
    rng = np.random.default_rng(5); quat = random_quats(20, 6); size = rng.uniform(.1, 1, (20, 3)); pos = rng.normal(size=(20, 3))
    mirrored = pe.box_corners(pos, quat, size) * np.array([1., 1., -1.])
    converted = pe.box_corners(pe.mirror_z_vec(pos), pe.mirror_z_quat(quat), size)
    assert np.linalg.norm(converted[:, :, None] - mirrored[:, None], axis=-1).min(-1).max() < 1e-9
    assert np.allclose(np.linalg.norm(pe.mirror_z_quat(quat), axis=-1), 1)


def test_wxyz_and_euler_conventions():
    assert np.allclose(pe.wxyz_to_xyzw([1, 0, 0, 0]), [0, 0, 0, 1])
    assert np.allclose(pe.euler_xyz_to_quat([0, 0, math.pi / 2]), axis_quat([0, 0, 1], math.pi / 2))
    rot = pe.q_to_matrix(pe.euler_xyz_to_quat([math.pi / 2, 0, math.pi / 2]))                    # R = Rz @ Ry @ Rx
    assert np.allclose(rot @ [0, 1, 0], [0, 0, 1], atol=1e-12) and np.allclose(rot @ [1, 0, 0], [0, 1, 0], atol=1e-12)
    assert np.allclose(pe.q_to_matrix(pe.q_mul(axis_quat([1, 0, 0], .4), axis_quat([0, 1, 0], -.9))),
                       pe.q_to_matrix(axis_quat([1, 0, 0], .4)) @ pe.q_to_matrix(axis_quat([0, 1, 0], -.9)))


def test_tipped_is_relative_to_the_initial_orientation():
    start = axis_quat([1, 0, 0], math.radians(80))                                              # starts lying on its side
    frames = lambda q_end: np.stack([start, q_end])[:, None]
    assert not pe.tipped_relative(frames(start))[0]
    assert not pe.tipped_relative(frames(pe.q_mul(axis_quat([0, 1, 0], 2.), start)))[0]          # pure yaw about world up
    assert not pe.tipped_relative(frames(pe.q_mul(axis_quat([0, 0, 1], math.radians(30)), start)))[0]
    assert pe.tipped_relative(frames(pe.q_mul(axis_quat([0, 0, 1], math.radians(40)), start)))[0]


def synthetic_episode(frames=30, slots=5):
    t = np.arange(frames) * pe.STEP
    pos = np.zeros((frames, 3, 3)); quat = np.tile([0., 0., 0., 1.], (frames, 3, 1))
    pos[:, 0] = np.stack((.5 * t, np.full(frames, .1), 0 * t), -1)                               # slides 0.7 m
    pos[:, 1] = [1., .1, 1.]                                                                     # rests
    pos[:, 2] = [-1., .1, -1.]; quat[:, 2] = [axis_quat([0, 0, 1], min(s * 2, math.pi / 2)) for s in t]   # topples and stops
    vel, ang = pe.finite_difference(t, pos, quat)
    return pe.make_episode(pos, quat, vel, ang, np.full((3, 3), .2), [.5, .5, .5], [.5, .5, .5], [0, 0, 0], [False] * 3,
                           ['cube', 'sphere', 'cube'], [True, False, True], 'unit_test', 'synthetic/0', slots=slots)


def test_episode_schema_and_labels(tmp_path):
    episode = synthetic_episode()
    assert episode['states'].shape == (pe.MAX_STEPS + 1, 5, 13) and episode['states'].dtype == np.float32
    assert episode['props'].shape == (5, 9) and episode['context'].shape == (10,) and episode['mask'].tolist() == [True] * 3 + [False] * 2
    assert episode['n_frames'] == 30 and np.array_equal(episode['states'][30:, :, :7], np.repeat(episode['states'][29:30, :, :7], 51, 0))
    assert np.allclose(episode['states'][0, :3, 1], pe.SUPPORT_HEIGHT + .1)                      # ground -> top of the support slab
    assert episode['context'][2] == pe.SUPPORT_HEIGHT and episode['context'][0] >= pe.MIN_SLAB and episode['context'][7] == 1
    assert np.allclose(np.linalg.norm(episode['states'][..., 3:7], axis=-1), 1)
    assert episode['tipped'].tolist() == [False, False, True, False, False]
    assert episode['moved_far'].tolist() == [True, False, False, False, False]
    assert episode['settled'].tolist() == [False, True, True, False, False]
    assert not episode['fell'].any() and episode['chaotic'] and not episode['live_physics']
    assert episode['box_like'].tolist() == [True, False, True, False, False] and episode['shape'][1] == 'sphere'
    assert episode['contact_support'][0, :3].all() and episode['rest'][5, 1] and not episode['rest'][5, 0]
    assert not episode['contact_floor'].any() and not episode['contact_object'].any()
    meta = pe.write_dataset(tmp_path, 'unit', [episode, synthetic_episode()], dict(source='synthetic'))
    assert meta['episodes'] == 2 and meta['box_like_objects'] == 4 and meta['box_like_only_episodes'] == 0
    with np.load(tmp_path / 'unit' / 'test' / 'shard_00000.npz', allow_pickle=False) as shard:
        assert set(FORMAT_KEYS) <= set(shard.files) and set(pe.EXTRA_FORMAT) <= set(shard.files)
        assert shard['states'].shape == (2, pe.MAX_STEPS + 1, 5, 13) and shard['kind'].tolist() == ['unit_test'] * 2
        assert shard['init_penetration'].shape == (2,) and float(shard['init_penetration'].max()) < 1e-4


def test_sphere_bounding_box_never_rotates():
    frames = 10; t = np.arange(frames) * pe.STEP
    pos = np.tile([0., .1, 0.], (frames, 1, 1)); quat = np.stack([axis_quat([1, 0, 0], 5 * s) for s in t])[:, None]
    vel, ang = pe.finite_difference(t, pos, quat)
    episode = pe.make_episode(pos, quat, vel, ang, np.full((1, 3), .2), [1.], [.5], [0.], [False], ['sphere'], [False], 'unit_test', 's')
    assert np.allclose(episode['states'][:, 0, 3:7], [0, 0, 0, 1]) and np.allclose(episode['states'][:, 0, 10:13], 0)
    assert not episode['tipped'].any()


def test_slab_side_contains_the_trajectory_within_schema_limits():
    assert pe.slab_side(np.zeros((2, 1, 3))) == pe.MIN_SLAB
    assert pe.slab_side(np.array([[[5.2, 0, -1.]]])) == 13.
    assert pe.slab_side(np.array([[[500., 0, 0]]])) == pe.MAX_SLAB


def test_eval_helpers_floor_mode_tipping_and_freezing():
    torch = pytest.importorskip('torch')
    import eval_external as ee
    episode = synthetic_episode(); data = pe.stack_episodes([episode, episode])
    floor = ee.to_floor_mode(data)
    assert np.allclose(floor['states'][:, 0, :3, 1], .1) and (floor['context'][:, 7] == 0).all() and np.allclose(floor['context'][:, :3], [1.2, .8, .75])
    assert np.allclose(data['states'][:, 0, :3, 1], pe.SUPPORT_HEIGHT + .1)                      # the input is not modified
    traj = torch.from_numpy(data['states'])
    assert ee.tipped_relative(traj)[0].tolist() == episode['tipped'].tolist()
    noisy = traj.clone(); noisy[:, 40:, :, 0] += 9.
    frozen = ee.freeze_after(noisy, torch.tensor([30, 30]))
    assert torch.equal(frozen, traj)
    masks = ee.subset_masks(data)
    assert masks['all_objects'].sum() == 6 and masks['box_like_objects'].sum() == 4 and masks['box_like_only_episodes'].sum() == 0
    stats = ee.evaluate(dict(reference=None, hold_initial=ee.hold_step), data, None, True)
    reference = ee.summarize(stats['reference'], data, masks['all_objects'], True)
    hold = ee.summarize(stats['hold_initial'], data, masks['all_objects'], True)
    assert reference['rest_pos_err_m'] == 0 and reference['tipped']['accuracy'] == 1 and reference['fell']['fp'] == 0
    assert hold['tipped']['recall'] == 0 and hold['moved_far']['recall'] == 0
    assert hold['rest_pos_err_m'] == pytest.approx(.5 * 29 * pe.STEP / 3, rel=1e-4) and hold['n_objects_1s'] == 6


def test_video_tracker_and_fell_rule():
    import track_video as tv
    tracker = tv.Tracker()
    for frame in range(12):
        tracker.update(frame, [dict(label='cup', bbox=[100 + 5 * frame, 100, 140 + 5 * frame, 150], confidence=.9),
                               dict(label='book', bbox=[300, 200, 380, 230], confidence=.8)] + ([dict(label='cup', bbox=[500, 50, 530, 90], confidence=.5)] if frame == 4 else []))
    lengths = sorted(len(t['frames']) for t in tracker.tracks)
    assert lengths == [1, 12, 12] and {t['label'] for t in tracker.tracks} == {'cup', 'book'}
    assert tv.iou([0, 0, 2, 2], [1, 1, 3, 3]) == pytest.approx(1 / 7)
    steady = np.full(20, 300.); dropped = np.r_[np.full(8, 300.), np.linspace(320, 600, 6), np.full(6, 600.)]
    assert tv.fell_from_track(steady, 720) == (False, None)
    fell, first = tv.fell_from_track(dropped, 720)
    assert fell and 8 <= first <= 12
    assert tv.fell_from_track(np.r_[dropped, np.full(5, 300.)], 720)[0] is False                 # came back up: not a fall
    plane = dict(normal=np.array([0., -1., 0.]), center=np.array([0., 1., 2.]), x=np.array([1., 0., 0.]), z=np.array([0., 0., 1.]))
    centre = tv.ray_plane(np.array([[320., 240. + 640 / (2 * math.tan(math.radians(30))) * .5]]), plane, 640, 480)
    assert np.allclose(centre, [[0., 0.]], atol=1e-9)                                            # the ray through (0, 1, 2) hits the plane centre


# ----------------------------------------------------------------------------- real-world toss converters (synthetic tracks only)
def test_matrix_to_quat_round_trip_and_rest_face():
    quat = random_quats(200, 7)
    back = pe.matrix_to_quat(pe.q_to_matrix(quat))
    assert np.allclose(np.abs((back * quat).sum(-1)), 1, atol=1e-12)
    assert np.allclose(pe.matrix_to_quat(np.diag([1., -1., -1.])), [1, 0, 0, 0])                 # 180 deg: the w ~ 0 branch
    assert pe.rest_face(np.array([0., 0., 0., 1.])) == 4                                         # identity: -y face is down
    assert pe.rest_face(axis_quat([0, 0, 1], math.pi / 2)) == 3                                  # +x now points up, -x is down
    assert pe.rest_face(axis_quat([1, 0, 0], math.pi / 2)) == 2                                  # body +z now points along world -y
    assert pe.rest_face(axis_quat([0, 1, 0], 1.3)) == 4                                          # yaw does not change the face


def test_smooth_velocities_are_exact_for_ballistic_motion_with_constant_spin():
    dt = 1 / 148.; t = np.arange(60) * dt; omega = np.array([2., -5., 1.])
    pos = np.stack((1.2 * t, .3 + .8 * t - 9.81 / 2 * t * t, -.4 * t), -1)
    quat = np.stack([axis_quat(omega, np.linalg.norm(omega) * s) for s in t])
    vel, ang = pe.smooth_velocities(dt, pos, quat, 2)
    assert np.allclose(vel[1:-1], np.stack((np.full(60, 1.2), .8 - 9.81 * t, np.full(60, -.4)), -1)[1:-1], atol=1e-9)
    assert np.allclose(ang[1:-1], omega, atol=2e-3) and vel.shape == ang.shape == (60, 3)
    assert np.allclose(pe.fit_acceleration(dt, pos, 8)[8:-8], [0, -9.81, 0], atol=1e-6) and np.isnan(pe.fit_acceleration(dt, pos, 8)[:8]).all()


def synthetic_toss(hz=360., mu=.3, release=.3, total=2.5):
    """Y-up track of a 0.2 x 0.1 x 0.2 box: carried in the hand, released, ballistic flight, flat slide with Coulomb friction, rest."""
    dt = 1 / hz; t = np.arange(round(total * hz)) * dt; g = 9.81
    p0 = np.array([0., .25, 0.]); v0 = np.array([1.2, .5, .9]); horiz = np.array([v0[0], 0., v0[2]]); speed = np.linalg.norm(horiz)
    flight = (v0[1] + math.sqrt(v0[1] ** 2 + 2 * g * (p0[1] - .05))) / g; slide = speed / (mu * g)
    pos = np.zeros((len(t), 3))
    for i, s in enumerate(t):
        if s < release: pos[i] = p0 + np.array([.3, 1., -.2]) * (s - release)                    # hand: constant velocity, not ballistic
        elif s < release + flight: u = s - release; pos[i] = p0 + v0 * u - np.array([0, g / 2 * u * u, 0])
        else:
            u = min(s - release - flight, slide)
            pos[i] = p0 + horiz * flight + horiz / speed * (speed * u - mu * g / 2 * u * u); pos[i, 1] = .05
    return dt, t, pos, np.tile([0., 0., 0., 1.], (len(t), 1)), dict(release=release, flight=flight, slide=slide, speed=speed, v0=v0)


def test_real_toss_segmentation_friction_and_episode():
    size = np.array([.2, .1, .2]); dt, t, pos, quat, info = synthetic_toss()
    vel, ang = pe.smooth_velocities(dt, pos, quat, pe.IAM_VEL_HALF); low = pe.lowest_corner(pos, quat, size)
    start = pe.free_flight_start(dt, pos, low)
    assert info['release'] <= t[start] <= info['release'] + 14 * dt                              # after the hand, at the start of the flight
    rest = pe.rest_frame(pos, quat, start)
    assert abs(t[rest] - (info['release'] + info['flight'] + info['slide'])) < .06
    assert pe.sliding_mu(dt, pos, quat, vel, size) == pytest.approx(.3, abs=.01)
    assert pe.free_flight_start(dt, pos - np.array([0, 1., 0]), low - 1.) is None                # never clear of the plane -> skipped
    episode = pe.real_toss_episode(dt, pos, quat, vel, ang, start, rest + round(.5 / dt), size, .4, 'real_unit', 'synthetic/toss')
    n = int(episode['n_frames']); states = episode['states']
    assert n == int((t[rest + round(.5 / dt)] - t[start]) / pe.STEP + 1e-9) + 1 and states.shape == (pe.MAX_STEPS + 1, pe.REAL_SLOTS, 13)
    assert episode['mask'].tolist() == [True, False, False, False] and episode['box_like'][0] and episode['kind'] == 'real_unit'
    u = t[start] - info['release']
    assert states[0, 0, 1] == pytest.approx(pe.SUPPORT_HEIGHT + .25 + .5 * u - 9.81 / 2 * u * u, abs=1e-5)       # landing plane -> slab top
    assert states[0, 0, 8] == pytest.approx(.5 - 9.81 * u, abs=1e-4) and np.linalg.norm(states[0, 0, [7, 9]]) == pytest.approx(info['speed'], abs=1e-4)
    assert states[0, 0, 9] == pytest.approx(0, abs=1e-4) and states[0, 0, 7] > 0                 # R5: travel along +x
    assert states[n - 1, 0, 1] == pytest.approx(pe.SUPPORT_HEIGHT + .05, abs=1e-5) and states[n - 1, 0, 2] == pytest.approx(states[0, 0, 2], abs=1e-5)
    assert states[n - 1, 0, 0] == pytest.approx(-states[0, 0, 0], abs=1e-5)                      # R5: trajectory centred on the table
    assert np.allclose(np.linalg.norm(states[..., 3:7], axis=-1), 1) and episode['rest_face'].tolist() == [4, -1, -1, -1]
    assert episode['context'][0] >= pe.REAL_TABLE[0] and episode['context'][1] == pe.REAL_TABLE[1] and episode['context'][2] == pe.SUPPORT_HEIGHT
    assert episode['context'][0] / 2 - abs(states[:n, 0, 0]).max() >= pe.REAL_TABLE_MARGIN and not episode['fell'].any() and episode['settled'][0]
    assert episode['props'][0, :7].tolist() == pytest.approx([.2, .1, .2, .4, .5, 0, 0]) and episode['context'][8:].tolist() == [.5, 0]
    other = pe.with_materials(episode, .6, .2)
    assert other['props'][0, 4] == pytest.approx(.6) and other['props'][0, 6] == pytest.approx(.2) and other['context'][8:].tolist() == pytest.approx([.6, .2])
    assert episode['props'][0, 4] == .5 and other['props'][1, 4] == 0                            # the original and the empty slots are untouched
    variants = pe._real_variants([episode], pe._real_info([episode], 1, [.3], 0., [], {}))
    assert set(variants) == {'', '_calib'} and variants['_calib'][1]['materials']['mu_eff'] == pytest.approx(.3)
    assert variants['_calib'][0][0]['props'][0, 4] == pytest.approx(math.sqrt(.3))


def test_real_world_eval_helpers(tmp_path):
    torch = pytest.importorskip('torch')
    import eval_external as ee
    size = np.array([.2, .1, .2]); dt, t, pos, quat, info = synthetic_toss()
    vel, ang = pe.smooth_velocities(dt, pos, quat, 4); start = pe.free_flight_start(dt, pos, pe.lowest_corner(pos, quat, size))
    episode = pe.real_toss_episode(dt, pos, quat, vel, ang, start, len(pos) - 5, size, .4, 'real_unit', 'synthetic/toss')
    data = pe.stack_episodes([episode, episode])
    assert ee.is_real(data) and not ee.is_real(pe.stack_episodes([synthetic_episode()]))
    quats = torch.from_numpy(np.stack([random_quats(50, 9)]))
    assert ee.face_down(quats).numpy().tolist() == pe.rest_face(quats.numpy()).tolist()          # torch and numpy rules agree
    varied = ee.set_materials(data, .36, .04)
    assert np.allclose(varied['props'][:, 0, 4], .6) and np.allclose(varied['props'][:, 0, 6], .2) and np.allclose(varied['context'][:, 8:], [.6, .2])
    assert np.allclose(data['props'][:, 0, 4], .5) and (varied['props'][:, 1:, 4] == 0).all()
    floor = ee.to_floor_mode(varied, keep_pair_friction=True)
    assert np.allclose(floor['props'][:, 0, 4], .36 / .5) and (floor['context'][:, 7] == 0).all()   # floor friction is 0.5: keep the pair product
    assert np.allclose(ee.to_floor_mode(varied)['props'][:, 0, 4], .6)
    stats = ee.evaluate(dict(reference=None, hold_initial=ee.hold_step), data, None, True)
    masks = ee.subset_masks(data)
    reference = ee.summarize(stats['reference'], data, masks['all_objects'], True); hold = ee.summarize(stats['hold_initial'], data, masks['all_objects'], True)
    assert reference['rest_face'] == dict(accuracy=1., n=2, majority_class_accuracy=1.) and reference['pos_err_0.25s_m'] == 0
    assert hold['rest_face']['accuracy'] == 1. and hold['pos_err_0.25s_m'] > .1 and hold['n_objects_0.25s'] == 2      # flat throw: same face down
    boot = ee.paired_bootstrap(stats['reference'], stats['hold_initial'], masks['all_objects'], 'face_wrong')
    assert boot['delta'] == 0 and boot['n_episodes'] == 2
    block = dict(data=dict(n_episodes=2, horizon_s=[1., 1.], object=dict(name='box', size_m=[.2, .1, .2], mass_kg=.4), materials=dict(variant='default')),
                 floor_modes=dict(slab=dict(reference=dict(all_objects=reference), hold_initial=dict(all_objects=hold))), paired_bootstrap=[])
    text = '\n'.join(ee.real_markdown('real_unit', block, {}))
    assert 'REAL WORLD' in text and '| hold_initial |' in text and 'rest-face acc' in text
