"""Fast tests for scripts/video_to_trajectories.py (geometry, homography, resampling, tracking, shard schema)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import video_to_trajectories as v  # noqa: E402


def _camera():
    position = np.array([7.9, -5.9, 5.5])
    forward = -position / np.linalg.norm(position)                   # looks at the origin
    right = np.cross(forward, [0, 0, 1.]); right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    return position, np.stack((right, up, -forward), 1)             # camera-to-world, camera looks along -Z


def test_ray_plane_lifting_roundtrip():
    position, rotation = _camera()
    rng = np.random.default_rng(0)
    points = np.concatenate((rng.uniform(-4, 4, (50, 2)), np.full((50, 1), .35)), 1)
    uv, depth = v.project(points, position, rotation, 35., 32.)
    assert (depth > 0).all()
    hit, valid = v.ray_plane(position, v.pixel_rays(uv, rotation, 35., 32.), .35)
    assert valid.all() and np.allclose(hit, points, atol=1e-9)


def test_ray_plane_airborne_object_is_pushed_away_and_rays_above_horizon_are_invalid():
    position, rotation = _camera()
    uv, _ = v.project(np.array([[0., 0., 3.]]), position, rotation, 35., 32.)          # object 3 units above the floor
    hit, valid = v.ray_plane(position, v.pixel_rays(uv, rotation, 35., 32.), .35)
    assert valid.all() and np.linalg.norm(hit[0, :2]) > 2.                              # ground assumption is wrong while airborne
    _, valid = v.ray_plane(position, np.array([[0., 0., 1.]]), .35)
    assert not valid.any()


def test_kubric_apparent_size_ratio_flags_airborne():
    position, rotation = _camera()
    quat = np.array([1., 0, 0, 0])                                                      # replaced below by the matrix path
    for height, expect_big in ((.35, False), (3., True)):
        uv, depth = v.project(np.array([[1., 1., height]]), position, rotation, 35., 32.)
        area = (35. / 32. * 128 * .7 / depth[0]) ** 2                                   # a 0.7-unit object seen at its true depth
        rays = v.pixel_rays(uv, rotation, 35., 32.)
        hit, _ = v.ray_plane(position, rays, .35)
        plane_depth = -((hit - position) @ rotation)[:, 2]
        ratio = np.sqrt(area) * plane_depth[0] / (35. / 32. * 128 * .7)
        assert (ratio > 1.15) == expect_big
    assert np.allclose(v.quat_wxyz_matrix(quat), np.eye(3))


def test_homography_fit_exact_and_ransac():
    h = np.array([[.02, .001, -5.], [.0005, .03, -4.], [1e-5, .004, 1.]])
    rng = np.random.default_rng(1); src = rng.uniform(0, 480, (60, 2)); dst = v.apply_homography(h, src)
    assert np.allclose(v.fit_homography(src, dst), h, rtol=1e-6, atol=1e-8)
    bad = dst.copy(); bad[:8] += rng.uniform(1, 3, (8, 2))
    fitted, inlier = v.fit_homography_ransac(src, bad, threshold=.05)
    assert inlier[8:].all() and not inlier[:8].any()
    assert np.allclose(v.apply_homography(fitted, src[8:]), dst[8:], atol=1e-6)


def test_fill_resample_and_velocity():
    t_src = np.arange(26) / 25.; truth = np.stack((.5 * t_src, np.zeros(26), -.2 * t_src), -1)
    obs = np.ones(26, bool); obs[10:14] = False; noisy = truth.copy(); noisy[~obs] = 99.
    filled = v.fill_gaps(noisy, obs)
    assert np.allclose(filled, truth, atol=1e-12)                                       # linear motion: the gap is interpolated exactly
    assert v.fill_gaps(noisy, np.zeros(26, bool)) is None
    t_dst = .05 + np.arange(18) * .05
    pos, seen = v.resample(t_src, filled, obs, t_dst)
    assert np.allclose(pos[:, 0], .5 * t_dst) and seen.sum() < len(seen) and seen[0] and seen[-1]
    assert not seen[(t_dst > 10 / 25.) & (t_dst < 13 / 25.)].any()
    assert np.allclose(v.smooth_velocity(pos, .05), [[.5, 0, -.2]] * len(pos), atol=1e-9)
    with pytest.raises(ValueError): v.resample(t_src, filled, obs, np.array([2.]))     # no extrapolation


def test_track_group_keeps_identities_through_a_missed_detection():
    def det(x, y): return [x, y, 50, x - 3, y - 3, x + 3, y + 3]
    frames = []
    for f in range(10):
        rows = [det(10 + 2 * f, 20), det(100 - 2 * f, 60)]
        if f == 4: rows = rows[1:]                                                      # first object missed once
        if f % 2: rows = rows[::-1]                                                     # detection order is arbitrary
        frames.append(np.asarray(rows, float))
    tracks = v.track_group(frames, 2, gate=20.)
    assert np.isnan(tracks[4, :, 0]).sum() == 1
    a = tracks[:, 0, 1]; assert len(set(a[np.isfinite(a)])) == 1                        # one track never jumps to the other row


def test_colour_segmentation_synthetic():
    background = np.full((40, 60, 3), 128., np.float32); image = background.copy()
    image[5:15, 5:15] = (170, 40, 40); image[20:32, 30:45] = (40, 70, 210); image[30:38, 2:12] = (105, 105, 105)   # red, blue, shadow
    tracks, flags = v.segment_video([image.astype(np.uint8)] * 3, background, ['red', 'blue'], min_area=10, gate=20.)
    assert tracks.shape == (3, 2, 7) and np.allclose(tracks[0, 0, :2], [9.5, 9.5]) and np.allclose(tracks[0, 1, :2], [37., 25.5])
    assert not flags.any()


def test_episode_schema_and_ground_mapping():
    frames = 24; t = np.arange(frames) * .05
    pos = np.zeros((frames, 2, 3)); pos[:, 0] = np.stack((.3 * t, np.full(frames, .126), 0 * t), -1); pos[:, 1] = 7.
    obs = np.zeros((frames, 2), bool); obs[4:, 0] = True                                # second object never observed
    size = np.full((2, 3), .252)
    e = v.build_episode(pos, obs, t, t, size, [.5, .5], [.4, .4], [.3, .3], ['cube', 'sphere'], 'k', 'vid', 'src', [.03, .02], gravity=10.)
    n = v.SLOTS
    assert e['states'].shape == (81, n, 13) and e['props'].shape == (n, 9) and e['context'].shape == (10,)
    assert e['obs_mask'].shape == (81, n) and e['obs_mask'].dtype == bool and e['pos_sigma'].shape == (n,) and e['pos_sigma'].dtype == np.float32
    assert e['mask'].tolist() == [True] + [False] * (n - 1) and e['n_untracked'] == 1 and e['video_derived'] is True
    assert not e['has_orientation'].any() and np.allclose(e['states'][:, 0, 3:7], [0, 0, 0, 1])
    assert np.isclose(e['states'][5, 0, 1], .75 + .126)                                 # ground plane -> support slab top (0.75 m)
    assert np.allclose(e['states'][6:20, 0, 7], .3, atol=1e-5) and e['obs_mask'][:, 0].sum() == 20 and not e['obs_mask'][24:].any()
    assert np.allclose(e['states'][0, 0, :3], e['states'][4, 0, :3])                    # frames before the first observation hold it
    assert v.transitions([e])['cube_transitions_observed'] == 19


def test_written_shards_if_present():
    root = Path(__file__).resolve().parents[1] / 'data' / 'video_v1'
    shards = sorted(root.glob('*/train/shard_*.npz'))
    if not shards: pytest.skip('data/video_v1 not generated')
    for shard in shards:
        with np.load(shard, allow_pickle=False) as z:
            count = len(z['states'])
            assert z['states'].shape[1:] == (81, v.SLOTS, 13) and z['props'].shape == (count, v.SLOTS, 9) and z['context'].shape == (count, 10)
            assert z['obs_mask'].shape == (count, 81, v.SLOTS) and z['pos_sigma'].shape == (count, v.SLOTS) and z['has_orientation'].shape == (count, v.SLOTS)
            assert z['video_derived'].all() and np.isfinite(z['states']).all()
            assert not (z['obs_mask'] & ~z['mask'][:, None]).any()
