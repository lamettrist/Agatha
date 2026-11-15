import json
import sys
from pathlib import Path
import numpy as np
import pybullet as p
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import generate_joint_data as g

N, J, T = g.N_SLOTS, g.J_SLOTS, 20
JOINTED = list(g.SCENARIOS)


def load(root, split):
    files = sorted((root/split).glob('shard_*.npz'))
    data = [np.load(f) for f in files]
    return {k: np.concatenate([d[k] for d in data]) for k in data[0].files}


def episodes(kind, count, steps=T, seed=11):
    out, index = [], 0
    while len(out) < count and index < 4*count+8:
        episode, reason, _ = g.generate_episode(seed, 'train', index, steps, kind=kind)
        index += 1
        if episode is not None: out.append(episode)
    assert len(out) == count, f'{kind}: too many dropped episodes'
    return out


def check_structure(e, steps):
    mask, joints, joint_mask, group = e['mask'], e['joints'], e['joint_mask'], e['group']
    assert e['states'].shape == (steps+1, N, 13) and e['joints'].shape == (J, 18) and e['joint_pos'].shape == (steps+1, J)
    assert np.isfinite(e['states']).all() and np.isfinite(joints).all() and np.isfinite(e['joint_pos']).all()
    assert np.allclose(np.linalg.norm(e['states'][:, :, 3:7], axis=-1), 1, atol=1e-5)
    assert (joints[~joint_mask] == 0).all() and (e['joint_pos'][:, ~joint_mask] == 0).all()
    assert (group[~mask] == -1).all()
    rows = joints[joint_mask]
    parent, child = rows[:, 0].astype(int), rows[:, 1].astype(int)
    assert np.array_equal(rows[:, :2], np.stack([parent, child], 1))            # integer valued
    assert (parent >= -1).all() and (parent < N).all() and mask[child].all() and mask[parent[parent >= 0]].all()
    assert (parent != child).all() and len(set(child)) == len(child)             # a tree: one parent joint per body
    assert np.array_equal(rows[:, 2:6].sum(1), np.ones(len(rows))) and set(np.unique(rows[:, 2:6])) <= {0., 1.}
    one_dof = (rows[:, 2] > .5) | (rows[:, 4] > .5)
    assert np.allclose(np.linalg.norm(rows[one_dof, 12:15], axis=1), 1, atol=1e-5)
    assert (rows[~one_dof, 12:17] == 0).all()                                    # ball / fixed: no axis, no limits
    assert (rows[:, 15] <= rows[:, 16]).all() and (rows[:, 17] >= 0).all()
    assert (e['props'][child, 5] == 0).all()                                     # a child link is never static
    # the group id is exactly the connected component of the joint graph
    assert (group[child] >= 0).all() and (group[parent[parent >= 0]] == group[child[parent >= 0]]).all()
    jointed = set(child) | set(parent[parent >= 0])
    assert set(np.flatnonzero(group >= 0)) == jointed
    # unlimited hinges are recorded as +-3.2 and hinge coordinates are wrapped
    hinge = joint_mask & (joints[:, 2] > .5)
    assert np.abs(e['joint_pos'][:, hinge]).max(initial=0) <= np.pi+1e-5
    assert (e['joint_pos'][:, joint_mask & ((joints[:, 3] > .5) | (joints[:, 5] > .5))] == 0).all()
    sizes = e['props'][mask, :3]
    assert sizes.min() >= .004 and sizes.max() <= .5 and e['props'][mask, 3].min() > 0


def test_generation_shapes_dtypes_and_determinism(tmp_path):
    kwargs = dict(train_episodes=16, val_episodes=4, test_episodes=4, steps=T, workers=2, shard_size=10, seed=5,
                  chunk=3, quiet=True)
    meta = g.generate(tmp_path/'a', **kwargs)
    g.generate(tmp_path/'b', **kwargs)
    assert len(list((tmp_path/'a'/'train').glob('shard_*.npz'))) == 2
    stored = json.loads((tmp_path/'a'/'meta.json').read_text())
    assert stored['step_s'] == .05 and stored['max_objects'] == N and stored['max_joints'] == J and stored['joint_dim'] == 18
    assert stored['joint_types'] == ['hinge', 'ball', 'prismatic', 'fixed']
    assert sum(meta['counts'].values())+sum(meta['dropped'].values()) == 24
    assert sum(sum(v.values()) for v in meta['dropped_by_reason'].values()) == sum(meta['dropped'].values())
    assert meta['transitions'] == sum(meta['counts'].values())*T
    for key in ('chaotic_fraction', 'kinds', 'joint_type_counts', 'by_kind', 'max_anchor_gap_m'):
        assert key in meta['splits']['train']
    for split in g.SPLITS:
        a, b = load(tmp_path/'a', split), load(tmp_path/'b', split)
        e = len(a['states'])
        assert e == meta['counts'][split]
        shapes = dict(states=(e, T+1, N, 13), props=(e, N, 9), context=(e, 10), mask=(e, N), contact_support=(e, T+1, N),
                      contact_floor=(e, T+1, N), contact_object=(e, T+1, N), rest=(e, T+1, N), fell=(e, N), tipped=(e, N),
                      kind=(e,), chaotic=(e,), live_physics=(e,), episode_index=(e,), init_penetration=(e,),
                      joints=(e, J, 18), joint_mask=(e, J), joint_pos=(e, T+1, J), group=(e, N),
                      joint_anchor_gap=(e,), joint_axis_error=(e,), joint_pos_error=(e,))
        assert set(shapes) == set(a)
        for key, shape in shapes.items():
            assert a[key].shape == shape, key
            assert np.array_equal(a[key], b[key]), key   # same seed -> identical arrays
        for key in ('states', 'props', 'context', 'joints', 'joint_pos', 'init_penetration', 'joint_anchor_gap'):
            assert a[key].dtype == np.float32, key
        for key in ('mask', 'joint_mask', 'contact_support', 'contact_floor', 'contact_object', 'rest', 'fell', 'tipped',
                    'chaotic', 'live_physics'):
            assert a[key].dtype == bool, key
        assert a['group'].dtype == np.int8 and a['episode_index'].dtype == np.int64 and a['kind'].dtype.kind == 'U'
        assert (a['joint_anchor_gap'] < 1e-3).all()
        for i in range(e):
            check_structure({k: a[k][i] for k in a}, T)


def test_episode_is_deterministic_and_split_dependent():
    a = g.generate_episode(3, 'train', 4, T, kind='chain')[0]
    b = g.generate_episode(3, 'train', 4, T, kind='chain')[0]
    c = g.generate_episode(3, 'val', 4, T, kind='chain')[0]
    assert a is not None and c is not None
    for key in ('states', 'joints', 'joint_pos', 'props', 'context'):
        assert np.array_equal(a[key], b[key]), key
    assert a['states'].shape != c['states'].shape or not np.array_equal(a['states'], c['states'])


@pytest.mark.parametrize('kind', JOINTED)
def test_anchor_axis_and_coordinate_consistency(kind):
    """World anchor from the parent == from the child within 1 mm on EVERY ground-truth frame; hinge axes stay
    aligned; the recorded joint coordinate equals the one recomputed from the body states."""
    for e in episodes(kind, 3, steps=40):
        check_structure(e, 40)
        assert e['kind'] == kind and e['joint_mask'].any()
        geometry = g.check_joints(e['states'], e['joints'], e['joint_mask'])
        assert geometry['anchor_gap'].max() < 1e-3, geometry['anchor_gap'].max()
        assert geometry['anchor_gap'].max() < 1e-5       # in practice float32 rounding only
        assert geometry['axis_error'].max() < 2e-3
        error = g.coordinate_error(e['joint_pos'], geometry['coordinate'], e['joints'], e['joint_mask'])
        assert error.max() < 2e-3, error.max()
        assert abs(float(e['joint_anchor_gap'])-geometry['anchor_gap'].max()) < 1e-6
        # explicit anchor formula (independent of check_joints) for every joint on every frame
        for row in e['joints'][e['joint_mask']]:
            parent, child = int(row[0]), int(row[1])
            if row[4] > .5: continue   # prismatic: anchors separate along the axis by the coordinate
            for t in range(0, 41, 8):
                rc = np.asarray(p.getMatrixFromQuaternion(e['states'][t, child, 3:7].tolist())).reshape(3, 3)
                wc = e['states'][t, child, :3]+rc@row[9:12]
                if parent < 0: wp = row[6:9]
                else:
                    rp = np.asarray(p.getMatrixFromQuaternion(e['states'][t, parent, 3:7].tolist())).reshape(3, 3)
                    wp = e['states'][t, parent, :3]+rp@row[6:9]
                assert np.linalg.norm(wp-wc) < 1e-3
        # limits hold up to the solver's transient overshoot
        rows = e['joints']; limited = e['joint_mask'] & (rows[:, 2] + rows[:, 4] > .5)
        excess = np.maximum(rows[:, 15]-e['joint_pos'], 0)+np.maximum(e['joint_pos']-rows[:, 16], 0)
        assert (excess[:, limited] <= g.MAX_LIMIT_EXCESS_RAD).all()
        assert (excess[0, limited] < 1e-4).all()          # every episode starts inside its limits


def test_scenario_content():
    kinds = {k: episodes(k, 4) for k in JOINTED}
    types = lambda es: np.sum([e['joints'][e['joint_mask'], 2:6].sum(0) for e in es], 0)
    for k in ('laptop', 'box_lid', 'door'):
        for e in kinds[k]:
            assert e['joint_mask'].sum() == 1 and e['joints'][0, 2] == 1 and e['joints'][0, 0] >= 0
    for e in kinds['laptop']:
        row = e['joints'][0]
        assert row[15] == 0 and 1.8 < row[16] < 2.5
        base, lid = int(row[0]), int(row[1])
        up = g.quat_to_matrix(e['states'][0, base, 3:7])[:, 1]
        assert up@(e['states'][0, lid, :3]-e['states'][0, base, :3]) > 0   # the lid opens to the base's upper side
    for e in kinds['drawer']: assert e['joints'][0, 4] == 1 and e['joints'][0, 15] == 0 and e['joints'][0, 16] > 0
    for e in kinds['pendulum']: assert e['joints'][0, 0] == -1 and e['joints'][0, 7] > e['context'][2]   # world anchor above the table
    for e in kinds['chain']: assert 3 <= (e['group'] >= 0).sum() <= 6
    for e in kinds['robot_arm']:
        root = int(e['joints'][0, 0]); assert e['props'][root, 5] == 1
        assert np.abs(e['states'][:, root, :7]-e['states'][0, root, :7]).max() == 0 and (e['states'][:, root, 7:] == 0).all()
    for e in kinds['ragdoll']:
        assert (e['group'] == 0).sum() == 10 and e['joint_mask'].sum() == 9
    assert types(kinds['ragdoll'])[0] > 0 and types(kinds['drawer'])[2] == 4
    assert all(e['context'][7] == 1 and (e['context'][4:7] == 0).all() for es in kinds.values() for e in es)


def test_plain_episodes_have_no_joints():
    for e in episodes('plain', 4):
        check_structure(e, T)
        assert e['kind'].startswith('plain_') and not e['joint_mask'].any() and (e['group'] == -1).all()
        assert (e['joints'] == 0).all() and e['mask'].sum() <= 8


def test_jointed_bodies_never_touch_each_other_but_other_links_do():
    """Parent-child pairs are excluded from collision; contact_object therefore never comes from the joint partner."""
    rng = np.random.default_rng(0); b = g.JB(rng); a = g.Art(b)
    base = a.root((.2, .2, .2), 1.)
    child = a.link(base, (.2, .2, .2), 1., 'hinge', (.05, 0, 0), (0, 0, 0), axis=(0, 1, 0))      # overlaps its parent
    a.link(child, (.2, .2, .2), 1., 'hinge', (.3, 0, 0), (0, 0, 0), axis=(0, 1, 0))               # clear of both
    a.t = np.array([0., b.h+.5, 0.]); b.commit(a)
    with g.JointWorld(b) as world:
        pairs, deepest = world.initial_pairs()
        assert pairs == set() and deepest == 0
        assert world.jointed_pairs() == {(0, 1), (1, 2)}
    b = g.JB(rng); a = g.Art(b)
    base = a.root((.2, .2, .2), 1.)
    child = a.link(base, (.2, .2, .2), 1., 'hinge', (.3, 0, 0), (0, 0, 0), axis=(0, 1, 0))
    a.link(child, (.2, .2, .2), 1., 'hinge', (-.25, 0, 0), (0, 0, 0), axis=(0, 1, 0))             # grandchild overlaps the base
    a.t = np.array([0., b.h+.5, 0.]); b.commit(a)
    with g.JointWorld(b) as world:
        pairs, deepest = world.initial_pairs()
        assert pairs == {(0, 2)} and deepest > .1


def test_link_velocities_are_world_com_velocities():
    rng = np.random.default_rng(1); b = g.JB(rng); a = g.Art(b)
    base = a.root((.2, .05, .1), 1.)
    mid = a.link(base, (.06, .2, .04), .4, 'hinge', (.1, 0, .02), (0, .1, 0), axis=(0, 0, 1), q0=.6, qd0=2.)
    a.link(mid, (.05, .15, .05), .3, 'ball', (0, -.1, 0), (.01, .075, 0), q0=g.q_axis((1, 2, 3), .5), qd0=np.array([1., -2, .5]))
    a.link(base, (.05, .05, .2), .3, 'prismatic', (-.1, 0, 0), (0, 0, -.1), axis=(0, 0, 1), q0=.03, qd0=.7)
    a.t = np.array([0., b.h+.6, 0.]); a.q = g.q_axis((1, 1, 0), .4); a.lin = np.array([.3, 1., -.2]); a.ang = np.array([1., 2., 3.])
    b.commit(a)
    with g.JointWorld(b) as world:
        assert world.fk_error < 1e-6
        h = 1e-5
        p.setTimeStep(h, physicsClientId=world.client)
        s0 = world.read(); world.step(); s1 = world.read()
        assert np.abs((s1[:, :3]-s0[:, :3])/h-s1[:, 7:10]).max() < 2e-3
        for i in range(len(s0)):
            dq = g.qmul(tuple(s1[i, 3:7]), (-s0[i, 3], -s0[i, 4], -s0[i, 5], s0[i, 6]))   # world-frame increment
            assert np.abs(2*np.asarray(dq[:3])/h*np.sign(dq[3])-s1[i, 10:13]).max() < 2e-2
        joints, mask = g.joint_rows(world)
        states = np.zeros((1, N, 13)); states[0, :4] = s0
        geometry = g.check_joints(states, joints, mask)
        assert geometry['anchor_gap'].max() < 1e-6 and geometry['axis_error'].max() < 1e-6
        assert np.allclose(geometry['coordinate'][0, :3], [.6, 0, .03], atol=1e-6)


def test_free_box_matches_agatha_physics_world():
    """A jointless JointWorld is the same engine as agatha.physics.World."""
    from agatha.physics import World
    from agatha.schema import Scene, SceneObject, Support
    rng = np.random.default_rng(2); b = g.JB(rng); b.w, b.d, b.h = 1.2, .8, .75; b.gravity = 9.81
    b.live = True; b.randomized = {k: False for k in b.randomized}; b.support_friction, b.support_restitution = .5, 0.
    a = g.Art(b); a.root((.1, .2, .06), .7, friction=.4); a.t = np.array([.5, .75+.1+.2, .1]); a.q = g.q_axis((1, 0, 1), .5)
    a.lin = np.array([.8, 0., .3]); b.commit(a)
    scene = Scene(support=Support(size=(1.2, .8), height=.75),
                  objects=[SceneObject(id='a', label='a', size=(.1, .2, .06), position=(.5, 1.05, .1), mass_kg=.7, friction=.4)])
    with g.JointWorld(b) as mine, World(scene) as reference:
        body = reference.bodies['a']
        q = p.getQuaternionFromEuler([0, 0, 0])
        p.resetBasePositionAndOrientation(body, g.to_bullet((.5, 1.05, .1)), g.quat_to_bullet(a.q), physicsClientId=reference.client)
        p.resetBaseVelocity(body, g.to_bullet((.8, 0., .3)), [0, 0, 0], physicsClientId=reference.client)
        for _ in range(360):
            mine.step(); reference.step()
        s = reference.snapshot(0)['objects'][0]
        state = mine.read()[0]
        assert np.allclose(state[:3], s['position'], atol=1e-6) and np.allclose(state[7:10], s['velocity'], atol=1e-6)
        assert abs(abs(np.dot(state[3:7], s['quaternion']))-1) < 1e-6


def test_damping_is_capped_for_light_links():
    rng = np.random.default_rng(3); b = g.JB(rng); a = g.Art(b)
    base = a.root((.1, .1, .1), 1., static=True)
    a.link(base, (.02, .1, .02), .02, 'hinge', (0, .05, 0), (0, -.05, 0), axis=(1, 0, 0), damping=5.)
    inertia = .02/12*(.1**2+.02**2)+.02*.05**2
    assert abs(a.joints[0]['damping']-g.DAMPING_STABILITY*inertia*240) < 1e-9
    a.link(base, (.1, .1, .1), 1., 'hinge', (0, .05, 0), (0, -.05, 0), axis=(1, 0, 0), damping=.01)
    assert a.joints[1]['damping'] == .01
