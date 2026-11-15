import math
import time
from pathlib import Path
import numpy as np
import pybullet as p
import pytest
import torch
from agatha import contact_gnn as cg
from agatha import joint_gnn as jg
from agatha.contact_gnn import quat_mul, rotvec_to_quat
from agatha.coordinates import from_bullet, quat_from_bullet
from agatha.dynamics import rotation
from agatha.joint_gnn import (JointGNN, joint_coordinates, joint_violation, joint_losses, project_joints, solve_joints,
                              build_from_checkpoint, init_from_contact_gnn, jointed_pairs)

CHECKPOINT = Path(__file__).resolve().parents[1]/'models'/'contact_gnn.pt'
HINGE, BALL, PRISMATIC, FIXED = range(4)


def joint_row(parent, child, kind, anchor_parent, anchor_child, axis=(0., 0., 0.), lower=0., upper=0., damping=0.):
    onehot = [0.]*4; onehot[kind] = 1.
    return torch.tensor([float(parent), float(child), *onehot, *anchor_parent, *anchor_child, *axis, lower, upper, damping])


def scene(boxes, rows, n=None, slots=None, enabled=0.):
    """boxes: dict(size, pos, quat?, vel?, ang?, mass?, static?); rows: joint_row tensors.  Support off by default."""
    n = n or len(boxes); slots = slots or max(len(rows), 1)
    state = torch.zeros(1, n, 13); state[:, :, 6] = 1
    props = torch.zeros(1, n, 9); props[:, :, 3] = 1
    mask = torch.zeros(1, n, dtype=torch.bool)
    for i, b in enumerate(boxes):
        state[0, i, :3] = torch.tensor(b['pos'])
        if 'quat' in b: state[0, i, 3:7] = torch.as_tensor(b['quat'], dtype=torch.float32)
        state[0, i, 7:10] = torch.tensor(b.get('vel', (0., 0., 0.)))
        state[0, i, 10:13] = torch.tensor(b.get('ang', (0., 0., 0.)))
        props[0, i, :6] = torch.tensor([*b['size'], b.get('mass', .5), .5, float(b.get('static', False))])
        mask[0, i] = True
    context = torch.tensor([[1.2, .8, .75, 9.81, 0., 0., 0., enabled, .5, 0.]])
    joints = torch.zeros(1, slots, jg.JOINT_DIM); joint_mask = torch.zeros(1, slots, dtype=torch.bool)
    for j, row in enumerate(rows): joints[0, j] = row; joint_mask[0, j] = True
    return state, props, context, mask, joints, joint_mask


def axis_quat(axis, angle):
    return rotvec_to_quat(torch.tensor(axis, dtype=torch.float32)*angle)


def hinge_pair(angle=0., lower=-3.2, upper=3.2, base_quat=None, **base):
    """Two 0.2 x 0.04 x 0.1 boxes end to end along x, hinged about z at the shared end face."""
    base_quat = torch.tensor([0., 0., 0., 1.]) if base_quat is None else base_quat
    child_quat = quat_mul(base_quat, axis_quat((0., 0., 1.), angle))
    pivot = torch.tensor([0., 1.5, 0.])+rotation(base_quat)@torch.tensor([.1, 0., 0.])
    child_pos = pivot+rotation(child_quat)@torch.tensor([.1, 0., 0.])
    boxes = [dict(size=(.2, .04, .1), pos=(0., 1.5, 0.), quat=base_quat, **base),
             dict(size=(.2, .04, .1), pos=child_pos.tolist(), quat=child_quat)]
    return scene(boxes, [joint_row(0, 1, HINGE, (.1, 0., 0.), (-.1, 0., 0.), (0., 0., 1.), lower, upper)])


def random_boxes(batch=4, n=12, seed=0):
    rng = torch.Generator().manual_seed(seed)
    state = torch.zeros(batch, n, 13)
    state[..., 0] = torch.rand(batch, n, generator=rng)*.8-.4
    state[..., 1] = .78+torch.rand(batch, n, generator=rng)*.25
    state[..., 2] = torch.rand(batch, n, generator=rng)*.5-.25
    q = torch.randn(batch, n, 4, generator=rng); state[..., 3:7] = q/q.norm(dim=-1, keepdim=True)
    state[..., 7:10] = torch.randn(batch, n, 3, generator=rng)
    state[..., 10:13] = torch.randn(batch, n, 3, generator=rng)*4
    props = torch.zeros(batch, n, 9)
    props[..., :3] = .04+torch.rand(batch, n, 3, generator=rng)*.2
    props[..., 3] = .1+torch.rand(batch, n, generator=rng)*2
    props[..., 4] = .25+torch.rand(batch, n, generator=rng)*.5
    props[..., 6] = torch.rand(batch, n, generator=rng)*.3
    props[:, 1, 5] = 1
    mask = torch.ones(batch, n, dtype=torch.bool); mask[:, -1] = False
    state[:, -1] = 0; state[:, -1, 6] = 1; props[:, -1] = 0; props[:, -1, 3] = 1
    context = torch.tensor([[1.2, .8, .75, 9.81, .1, 0., -.05, 1., .5, 0.]]).repeat(batch, 1)
    context[:, 3] = torch.rand(batch, generator=rng)*10+2
    return state, props, context, mask


def random_joints(batch=4, n=12, slots=12, seed=0, world=True):
    """A random chain 0-2-3-...-(n-2) per scene (slot 1 is static, slot n-1 unused) with mixed joint types."""
    rng = torch.Generator().manual_seed(seed+100)
    joints = torch.zeros(batch, slots, jg.JOINT_DIM); joint_mask = torch.zeros(batch, slots, dtype=torch.bool)
    chain = [0, *range(2, n-1)]
    for b in range(batch):
        for j, (parent, child) in enumerate(zip(([-1] if world else [])+chain, chain)):
            if j >= slots-1: break
            kind = int(torch.randint(0, 4, (1,), generator=rng))
            axis = torch.randn(3, generator=rng); axis = axis/axis.norm()
            anchor = lambda: (torch.rand(3, generator=rng)*.2-.1).tolist()
            anchor_parent = (0., 1., 0.) if parent < 0 else anchor()
            limits = (-.5, .7) if kind == HINGE else (-.05, .1) if kind == PRISMATIC else (0., 0.)
            joints[b, j] = joint_row(parent, child, kind, anchor_parent, anchor(),
                                     axis.tolist() if kind in (HINGE, PRISMATIC) else (0., 0., 0.), *limits, damping=.1)
            joint_mask[b, j] = True
    return joints, joint_mask


@pytest.fixture(scope='module')
def checkpoint():
    if not CHECKPOINT.exists(): pytest.skip('models/contact_gnn.pt missing')
    return torch.load(CHECKPOINT, map_location='cpu', weights_only=False)


# ------------------------------------------------------------ zero-joint equivalence
@pytest.mark.parametrize('rounds', [None, 6])
def test_zero_joints_reproduce_contact_gnn(checkpoint, rounds):
    reference = cg.build_from_checkpoint(checkpoint)
    if rounds is None:
        model = build_from_checkpoint(checkpoint)                       # contact checkpoint accepted directly
        assert model.rounds == reference.rounds
    else:
        model = JointGNN(width=reference.width, rounds=rounds, edge_radius=reference.edge_radius)
        report = init_from_contact_gnn(model, checkpoint)
        assert report['identity_rounds'] == list(range(reference.rounds, rounds))
        assert all(key.startswith(('joint_', 'world_node', 'edge_updates', 'node_updates')) for key in report['new'])
        model.eval()
    worst = 0.
    for seed, n in ((0, 12), (1, 6), (2, 3)):
        state, props, context, mask = random_boxes(4, n, seed)
        joints = torch.zeros(4, 12, jg.JOINT_DIM); joint_mask = torch.zeros(4, 12, dtype=torch.bool)
        with torch.no_grad():
            expected = reference(state, props, context, mask)
            for _ in range(3):                                           # a short rollout, compared step by step
                for output in (model(state, props, context, mask), model(state, props, context, mask, joints, joint_mask)):
                    worst = max(worst, float((output-expected).abs().max()))
                state = expected; expected = reference(state, props, context, mask)
    print(f'zero-joint max |JointGNN-ContactGNN| (rounds={model.rounds}): {worst:.2e}')
    assert worst < 1e-5


def test_zero_joints_projection_and_penetration_match_contact_gnn():
    assert jg._contact_functions()['patched']
    state, props, context, mask = random_boxes(4, 12, 3)
    pred = state.clone(); pred[..., 1] -= .02
    joints = torch.zeros(4, 12, jg.JOINT_DIM); joint_mask = torch.zeros(4, 12, dtype=torch.bool)
    assert torch.equal(project_joints(state, pred, props, context, mask, joints, joint_mask), cg.project(state, pred, props, context, mask))
    ours, theirs = jg.penetration_depths(pred, props, context, mask, joints, joint_mask), cg.penetration_depths(pred, props, context, mask)
    assert all(torch.equal(ours[key], theirs[key]) for key in theirs)
    rest = torch.zeros(4, 12, dtype=torch.bool)
    ours, theirs = jg.physics_losses(pred, state, state, props, context, mask, rest, joints, joint_mask), cg.physics_losses(pred, state, state, props, context, mask, rest)
    assert all(torch.equal(ours[key], theirs[key]) for key in theirs)


def test_jointed_pairs_do_not_collide():
    # two overlapping boxes: a contact edge and a penetration without the joint, none with it
    boxes = [dict(size=(.2, .1, .1), pos=(0., 1.5, 0.)), dict(size=(.2, .1, .1), pos=(.15, 1.5, 0.))]
    state, props, context, mask, joints, joint_mask = scene(boxes, [joint_row(0, 1, BALL, (.075, 0., 0.), (-.075, 0., 0.))])
    assert jointed_pairs(joints, joint_mask, 2)[0].tolist() == [[False, True], [True, False]]
    assert float(cg.penetration_depths(state, props, context, mask)['objects'].max()) > .02
    assert float(jg.penetration_depths(state, props, context, mask, joints, joint_mask)['objects'].max()) == 0
    out = project_joints(state, state, props, context, mask, joints, joint_mask)
    assert float((out[..., :3]-state[..., :3]).abs().max()) < 1e-5            # not pushed apart
    assert float((cg.project(state, state, props, context, mask)[..., :3]-state[..., :3]).abs().max()) > .01
    model = JointGNN(width=32, rounds=2)
    _, _, receiver, sender = model.graph(state, props, context, mask)
    assert len(receiver) > 0                                                   # ContactGNN would link them
    features, joint_receiver, joint_sender = model.joint_graph(state, props, context, mask, joints, joint_mask)
    assert features.shape == (2, jg.JOINT_EDGE_DIM) and torch.isfinite(features).all()
    assert sorted(zip(joint_receiver.tolist(), joint_sender.tolist())) == [(0, 1), (1, 0)]


def test_world_joint_uses_the_world_node_and_static_bodies_receive_nothing():
    boxes = [dict(size=(.1, .1, .1), pos=(0., 1.5, 0.)), dict(size=(.1, .1, .1), pos=(.3, 1.5, 0.), static=True),
             dict(size=(.1, .1, .1), pos=(.5, 1.5, 0.))]
    rows = [joint_row(-1, 0, BALL, (0., 1.6, 0.), (0., .1, 0.)), joint_row(1, 2, HINGE, (.1, 0., 0.), (-.1, 0., 0.), (0., 0., 1.), -1., 1.)]
    state, props, context, mask, joints, joint_mask = scene(boxes, rows, slots=4)
    model = JointGNN(width=32, rounds=2)
    features, receiver, sender = model.joint_graph(state, props, context, mask, joints, joint_mask)
    total = 3+2                                                                # bodies + support + floor
    assert sorted(zip(receiver.tolist(), sender.tolist())) == [(0, total), (2, 1)]   # world node = B*total + b
    out = model(state, props, context, mask, joints, joint_mask)
    assert torch.isfinite(out).all() and torch.equal(out[0, 1], state[0, 1])


# ------------------------------------------------------------------ analytic geometry
@pytest.mark.parametrize('angle', [0., .3, -1.2, 2.9, -3.0])
def test_hinge_angle_and_zero_violation(angle):
    tilt = axis_quat((.6, .0, .8), .7)
    state, props, context, mask, joints, joint_mask = hinge_pair(angle, base_quat=tilt)
    assert float(joint_coordinates(state, joints, joint_mask)[0, 0]) == pytest.approx(angle, abs=1e-5)
    violation = joint_violation(state, joints, joint_mask)
    assert float(violation['anchor_gap'].max()) < 3e-6
    assert float(violation['off_axis'].max()) < 1e-4
    assert float(violation['limit_excess'].max()) == 0


def test_anchor_gap_limit_excess_and_off_axis():
    state, props, context, mask, joints, joint_mask = hinge_pair(.8, lower=-.5, upper=.6)
    violation = joint_violation(state, joints, joint_mask)
    assert float(violation['limit_excess'][0, 0]) == pytest.approx(.2, abs=1e-5)
    state2, *_ = hinge_pair(-.9, lower=-.5, upper=.6)
    assert float(joint_violation(state2, joints, joint_mask)['limit_excess'][0, 0]) == pytest.approx(.4, abs=1e-5)
    moved = state.clone(); moved[0, 1, :3] += torch.tensor([.012, -.016, 0.])
    assert float(joint_violation(moved, joints, joint_mask)['anchor_gap'][0, 0]) == pytest.approx(.02, abs=1e-6)
    tipped = state.clone(); tipped[0, 1, 3:7] = quat_mul(axis_quat((1., 0., 0.), .1), state[0, 1, 3:7])
    assert float(joint_violation(tipped, joints, joint_mask)['off_axis'][0, 0]) == pytest.approx(.1, abs=1e-4)
    # unused rows report nothing
    joints2 = torch.cat((joints, joints), 1); mask2 = torch.cat((joint_mask, torch.zeros_like(joint_mask)), 1)
    for value in joint_violation(moved, joints2, mask2).values(): assert float(value[0, 1]) == 0


def test_prismatic_ball_fixed_and_world_joints():
    quat = axis_quat((0., 1., 0.), .4)
    axis_world = rotation(quat)@torch.tensor([1., 0., 0.])
    boxes = [dict(size=(.3, .1, .2), pos=(0., 1.5, 0.), quat=quat),
             dict(size=(.1, .05, .1), pos=(torch.tensor([0., 1.5, 0.])+.07*axis_world).tolist(), quat=quat)]
    row = joint_row(0, 1, PRISMATIC, (0., 0., 0.), (0., 0., 0.), (1., 0., 0.), -.05, .05)
    state, props, context, mask, joints, joint_mask = scene(boxes, [row])
    assert float(joint_coordinates(state, joints, joint_mask)[0, 0]) == pytest.approx(.07, abs=1e-6)
    violation = joint_violation(state, joints, joint_mask)
    assert float(violation['anchor_gap'][0, 0]) < 3e-6                         # sliding along the axis is free
    assert float(violation['limit_excess'][0, 0]) == pytest.approx(.02, abs=1e-6)
    assert float(violation['off_axis'][0, 0]) < 1e-4
    side = state.clone(); side[0, 1, 1] += .01
    assert float(joint_violation(side, joints, joint_mask)['anchor_gap'][0, 0]) == pytest.approx(.01, abs=1e-6)
    turned = state.clone(); turned[0, 1, 3:7] = quat_mul(axis_quat((0., 0., 1.), .2), quat)
    assert float(joint_violation(turned, joints, joint_mask)['off_axis'][0, 0]) == pytest.approx(.2, abs=1e-4)
    for kind, expected in ((FIXED, .2), (BALL, 0.)):
        joints[0, 0, 2:6] = torch.eye(4)[kind]
        found = joint_violation(turned, joints, joint_mask)
        assert float(found['off_axis'][0, 0]) == pytest.approx(expected, abs=1e-4)
        assert float(found['limit_excess'][0, 0]) == 0 and float(joint_coordinates(turned, joints, joint_mask)[0, 0]) == 0
    # world-anchored hinge: anchor and axis are world coordinates
    child = dict(size=(.2, .04, .1), pos=(.1*math.cos(.5), 1.+.1*math.sin(.5), 0.), quat=axis_quat((0., 0., 1.), .5))
    state, props, context, mask, joints, joint_mask = scene([child], [joint_row(-1, 0, HINGE, (0., 1., 0.), (-.1, 0., 0.), (0., 0., 1.))])
    assert float(joint_coordinates(state, joints, joint_mask)[0, 0]) == pytest.approx(.5, abs=1e-5)
    assert float(joint_violation(state, joints, joint_mask)['anchor_gap'][0, 0]) < 3e-6


# ------------------------------------------------------------------------ projection
def test_projection_closes_anchor_gap():
    state, props, context, mask, joints, joint_mask = hinge_pair(.4, ang=(0., 0., 0.))
    props[0, 1, 3] = 2.                                                       # unequal masses
    pred = state.clone(); pred[0, 1, :3] += torch.tensor([.012, -.016, 0.])
    assert float(joint_violation(pred, joints, joint_mask)['anchor_gap'].max()) == pytest.approx(.02, abs=1e-6)
    out = project_joints(state, pred, props, context, mask, joints, joint_mask)
    violation = joint_violation(out, joints, joint_mask)
    assert float(violation['anchor_gap'].max()) < 5e-4 and float(violation['off_axis'].max()) < 1e-3
    assert torch.isfinite(out).all()
    # split by inverse mass: the centre of mass does not move
    centre = lambda s: (s[0, :, :3]*props[0, :, 3:4]).sum(0)/props[0, :, 3].sum()
    assert float((centre(out)-centre(pred)).abs().max()) < 1e-5
    assert float((out[0, 0, :3]-pred[0, 0, :3]).norm()) > float((out[0, 1, :3]-pred[0, 1, :3]).norm())   # light one moves more


def test_projection_clamps_limits_and_realigns_axis():
    state, props, context, mask, joints, joint_mask = hinge_pair(.9, lower=-.5, upper=.6)
    state[0, 1, 3:7] = quat_mul(axis_quat((1., 0., 0.), .15), state[0, 1, 3:7])   # also 0.15 rad off axis
    before = joint_violation(state, joints, joint_mask)
    assert float(before['limit_excess'].max()) > .25 and float(before['off_axis'].max()) > .14
    out = project_joints(state, state, props, context, mask, joints, joint_mask, limit_slop=(0., 0.))   # hard clamp
    after = joint_violation(out, joints, joint_mask)
    assert float(after['limit_excess'].max()) < 1e-3 and float(after['off_axis'].max()) < 1e-3
    assert float(after['anchor_gap'].max()) < 5e-4
    assert float(joint_coordinates(out, joints, joint_mask)[0, 0]) == pytest.approx(.6, abs=2e-3)
    # default: Bullet's limits are soft, so the clamp stops jg.LIMIT_SLOP outside the limit
    out = project_joints(state, state, props, context, mask, joints, joint_mask)
    after = joint_violation(out, joints, joint_mask)
    assert float(after['off_axis'].max()) < 1e-3 and float(after['anchor_gap'].max()) < 5e-4
    assert float(joint_coordinates(out, joints, joint_mask)[0, 0]) == pytest.approx(.6+jg.LIMIT_SLOP[0], abs=2e-3)
    # a true-data-sized overshoot inside the tolerance is left alone
    state2 = hinge_pair(.6+.8*jg.LIMIT_SLOP[0], lower=-.5, upper=.6)[0]
    assert float((jg.solve_joints(state2, props, context, mask, joints, joint_mask)-state2).abs().max()) < 1e-5
    # prismatic limit
    boxes = [dict(size=(.3, .1, .2), pos=(0., 1.5, 0.)), dict(size=(.1, .05, .1), pos=(.09, 1.51, 0.))]
    state, props, context, mask, joints, joint_mask = scene(boxes, [joint_row(0, 1, PRISMATIC, (0., 0., 0.), (0., 0., 0.), (1., 0., 0.), -.05, .05)])
    out = project_joints(state, state, props, context, mask, joints, joint_mask, limit_slop=(0., 0.))
    after = joint_violation(out, joints, joint_mask)
    assert float(after['limit_excess'].max()) < 5e-4 and float(after['anchor_gap'].max()) < 5e-4
    assert float(joint_coordinates(out, joints, joint_mask)[0, 0]) == pytest.approx(.05, abs=5e-4)
    out = project_joints(state, state, props, context, mask, joints, joint_mask)
    assert float(joint_coordinates(out, joints, joint_mask)[0, 0]) == pytest.approx(.05+jg.LIMIT_SLOP[1], abs=5e-4)


def test_projection_removes_violating_velocity_only():
    state, props, context, mask, joints, joint_mask = hinge_pair(.2)
    state[0, 1, 7:10] = torch.tensor([0., 0., .8])                            # child flying off along the hinge axis
    out = project_joints(state, state, props, context, mask, joints, joint_mask)
    frames = jg._frames(out, joints, joint_mask)
    slip = (out[0, 1, 7:10]+torch.cross(out[0, 1, 10:13], frames.arm_c[0, 0], dim=-1))-(out[0, 0, 7:10]+torch.cross(out[0, 0, 10:13], frames.arm_p[0, 0], dim=-1))
    spin = out[0, 1, 10:13]-out[0, 0, 10:13]
    assert float(slip.norm()) < 5e-3 and float((spin-(spin*frames.axis_p[0, 0]).sum()*frames.axis_p[0, 0]).norm()) < 5e-3
    momentum = lambda s: (s[0, :, 7:10]*props[0, :, 3:4]).sum(0)
    assert float((momentum(out)-momentum(state)).abs().max()) < 1e-5
    # spinning about the hinge axis is allowed and kept
    state, props, context, mask, joints, joint_mask = hinge_pair(.2)
    arm = jg._frames(state, joints, joint_mask).arm_c[0, 0]
    state[0, 1, 10:13] = torch.tensor([0., 0., 3.]); state[0, 1, 7:10] = torch.cross(torch.tensor([0., 0., 3.]), -arm, dim=-1)
    out = project_joints(state, state, props, context, mask, joints, joint_mask)
    assert float((out-state).abs().max()) < 1e-4
    # ... but not into a limit
    joints[0, 0, 15:17] = torch.tensor([-.5, .2])
    out = project_joints(state, state, props, context, mask, joints, joint_mask, limit_slop=(0., 0.))
    assert float((out[0, 1, 10:13]-out[0, 0, 10:13])[2]) < 1e-2
    # with the default tolerance the motion is only stopped at the end of the band
    out = project_joints(state, state, props, context, mask, joints, joint_mask)
    assert float((out[0, 1, 10:13]-out[0, 0, 10:13])[2]) > 2.9
    joints[0, 0, 16] = .2-jg.LIMIT_SLOP[0]
    out = project_joints(state, state, props, context, mask, joints, joint_mask)
    assert float((out[0, 1, 10:13]-out[0, 0, 10:13])[2]) < 1e-2


def test_projection_leaves_consistent_states_and_fixed_bodies_alone():
    state, props, context, mask, joints, joint_mask = hinge_pair(.3, lower=-1., upper=1., base_quat=axis_quat((.6, 0., .8), .7))
    out = project_joints(state, state, props, context, mask, joints, joint_mask)
    assert float((out-state).abs().max()) < 1e-5
    # static parent: only the child moves, and it ends on the parent's anchor
    props[0, 0, 5] = 1
    pred = state.clone(); pred[0, 1, :3] += torch.tensor([.0, .02, .0])
    out = project_joints(state, pred, props, context, mask, joints, joint_mask)
    assert torch.equal(out[0, 0], pred[0, 0])
    assert float(joint_violation(out, joints, joint_mask)['anchor_gap'].max()) < 5e-4
    # world anchor: the anchor point stays where the joint row says
    child = dict(size=(.2, .04, .1), pos=(.1, 1.02, .0))
    state, props, context, mask, joints, joint_mask = scene([child], [joint_row(-1, 0, BALL, (0., 1., 0.), (-.1, 0., 0.))])
    out = project_joints(state, state, props, context, mask, joints, joint_mask)
    anchor = out[0, 0, :3]+rotation(out[0, 0, 3:7])@torch.tensor([-.1, 0., 0.])
    assert float((anchor-torch.tensor([0., 1., 0.])).norm()) < 5e-4
    # both ends static: nothing moves, nothing is NaN
    props[0, 0, 5] = 1
    assert torch.equal(project_joints(state, state, props, context, mask, joints, joint_mask), state)


def test_projection_never_nan_and_tightens_random_chains():
    state, props, context, mask = random_boxes(4, 12, 5)
    joints, joint_mask = random_joints(4, 12, 12, 5)
    out = project_joints(state, state, props, context, mask, joints, joint_mask, iters=20)
    assert torch.isfinite(out).all()
    start, end = joint_violation(state, joints, joint_mask), joint_violation(out, joints, joint_mask)
    print('random chain anchor gap', float(start['anchor_gap'].max()), '->', float(end['anchor_gap'].max()))
    assert float(end['anchor_gap'].mean()) < .25*float(start['anchor_gap'].mean())
    assert torch.equal(out[:, 1], state[:, 1]) and torch.equal(out[:, -1], state[:, -1])     # static and unused slots
    broken = state.clone(); broken[0, 0] = float('nan'); broken[1, 2, 7:] = float('inf')
    assert torch.isfinite(project_joints(state, broken, props, context, mask, joints, joint_mask)).all()
    garbage = joints.clone(); garbage[0, 0, 0] = 99; garbage[0, 1, 1] = -3; garbage[1, 0, 12:15] = 0
    assert torch.isfinite(project_joints(state, state, props, context, mask, garbage, joint_mask)).all()
    assert torch.isfinite(JointGNN(width=32, rounds=2)(state, props, context, mask, garbage, joint_mask)).all()


def test_batched_results_equal_single_scene_results():
    state, props, context, mask = random_boxes(4, 12, 21)
    joints, joint_mask = random_joints(4, 12, 12, 21)
    joint_mask[2] = False                                                      # one scene without joints in the batch
    consistent = solve_joints(state, props, context, mask, joints, joint_mask, iters=30)
    pred = consistent.clone(); pred[..., :3] += torch.randn(4, 12, 3, generator=torch.Generator().manual_seed(3))*.003
    model = JointGNN(width=32, rounds=2).eval()
    one = lambda function, b, *states: function(*(s[b:b+1] for s in states), props[b:b+1], context[b:b+1], mask[b:b+1], joints[b:b+1], joint_mask[b:b+1])
    with torch.no_grad():
        for name, function, states, options in (('sweeps', solve_joints, (pred,), dict(iters=3, snap=False)), ('solve', solve_joints, (pred,), {}),
                                                ('project', project_joints, (consistent, pred), {}), ('model', model, (consistent,), {})):
            batched = function(*states, props, context, mask, joints, joint_mask, **options)
            for b in range(4):
                single = function(*(s[b:b+1] for s in states), props[b:b+1], context[b:b+1], mask[b:b+1], joints[b:b+1], joint_mask[b:b+1], **options)
                assert float((batched[b]-single[0]).abs().max()) < (2e-3 if name == 'project' else 1e-4), (name, b)
    after = joint_violation(solve_joints(pred, props, context, mask, joints, joint_mask, iters=3, snap=False), joints, joint_mask)
    assert float(after['anchor_gap'].max()) < float(joint_violation(pred, joints, joint_mask)['anchor_gap'].max())


def test_projection_refines_small_errors_on_a_chain():
    """A consistent 6-link hanging chain with millimetre noise (the size of a model error) is pulled back."""
    links = 6; boxes, rows = [], []
    for i in range(links):
        boxes.append(dict(size=(.04, .12, .04), pos=(0., 2.-.06-.12*i, 0.)))
        rows.append(joint_row(i-1, i, BALL, (0., 2., 0.) if i == 0 else (0., -.06, 0.), (0., .06, 0.)))
    state, props, context, mask, joints, joint_mask = scene(boxes, rows)
    assert float(joint_violation(state, joints, joint_mask)['anchor_gap'].max()) < 3e-6
    pred = state.clone(); pred[..., :3] += torch.randn(1, links, 3, generator=torch.Generator().manual_seed(0))*.003
    before = float(joint_violation(pred, joints, joint_mask)['anchor_gap'].max())
    out = project_joints(state, pred, props, context, mask, joints, joint_mask)
    after = float(joint_violation(out, joints, joint_mask)['anchor_gap'].max())
    print(f'chain anchor gap {before*1e3:.2f} mm -> {after*1e3:.3f} mm (6 sweeps)')
    assert after < 5e-4 and after < .15*before


# ---------------------------------------------------------------------------- model
def test_permutation_equivariance():
    torch.manual_seed(0)
    model = JointGNN(width=48, rounds=3).eval()
    state, props, context, mask = random_boxes(3, 12, 7)
    joints, joint_mask = random_joints(3, 12, 12, 7)
    order = torch.randperm(12, generator=torch.Generator().manual_seed(1))
    place = torch.empty(12, dtype=torch.long); place[order] = torch.arange(12)     # old slot -> new slot
    remapped = joints.clone()
    for column in (0, 1):
        index = joints[..., column].long()
        remapped[..., column] = torch.where(index >= 0, place[index.clamp_min(0)], index).float()
    joint_order = torch.randperm(12, generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        expected = model(state, props, context, mask, joints, joint_mask)
        output = model(state[:, order], props[:, order], context, mask[:, order], remapped[:, joint_order], joint_mask[:, joint_order])
        plain = model(state, props, context, mask)
    assert float((output-expected[:, order]).abs().max()) < 1e-4
    assert float((expected-plain).abs().max()) > 1e-4                       # the joints do change the prediction
    coords = joint_coordinates(state[:, order], remapped, joint_mask)
    assert torch.allclose(coords, joint_coordinates(state, joints, joint_mask), atol=1e-5)


def test_gradients_flow_through_joint_losses_and_model():
    torch.manual_seed(0)
    model = JointGNN(width=48, rounds=3)
    state, props, context, mask = random_boxes(3, 12, 9)
    joints, joint_mask = random_joints(3, 12, 12, 9)
    truth = solve_joints(state, props, context, mask, joints, joint_mask, iters=30)
    pred = model(truth, props, context, mask, joints, joint_mask)
    losses = joint_losses(pred, truth, joints, joint_mask)
    assert set(losses) == {'anchor', 'limit', 'axis'} and all(value.ndim == 0 and torch.isfinite(value) for value in losses.values())
    assert float(losses['anchor']) > 0
    sum(losses.values()).backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    for name in ('joint_encoder.0.weight', 'joint_updates.0.0.weight', 'edge_encoder.0.weight', 'decoder.4.weight', 'world_node'):
        assert float(dict(model.named_parameters())[name].grad.abs().sum()) > 0, name
    # the loss is zero on the truth and its gradient w.r.t. a displaced state points back
    assert float(sum(joint_losses(truth, truth, joints, joint_mask).values())) == 0
    state, props, context, mask, joints, joint_mask = hinge_pair(.9, lower=-.5, upper=.6)
    moved = state.clone(); moved[0, 1, :3] += torch.tensor([.02, 0., 0.]); moved.requires_grad_(True)
    reference, *_ = hinge_pair(.3)
    losses = joint_losses(moved, reference, joints, joint_mask)
    assert float(losses['anchor']) > 0 and float(losses['limit']) > 0
    losses['anchor'].backward()
    assert float(moved.grad[0, 1, 0]) > 0 and float(moved.grad[0, 0, 0]) < 0


def test_checkpoint_round_trip_and_warm_start_errors(checkpoint, tmp_path):
    model = JointGNN(width=128, rounds=6)
    init_from_contact_gnn(model, checkpoint)
    path = tmp_path/'joint.pt'
    torch.save(dict(kind=model.kind, config=model.config, state_dict=model.state_dict()), path)
    loaded = build_from_checkpoint(torch.load(path, map_location='cpu', weights_only=False))
    assert isinstance(loaded, JointGNN) and loaded.kind == 'joint_gnn' and loaded.rounds == 6 and not loaded.training
    state, props, context, mask = random_boxes(2, 12, 11)
    joints, joint_mask = random_joints(2, 12, 12, 11)
    with torch.no_grad():
        assert torch.equal(loaded(state, props, context, mask, joints, joint_mask), model.eval()(state, props, context, mask, joints, joint_mask))
    with pytest.raises(ValueError): init_from_contact_gnn(JointGNN(width=64, rounds=6), checkpoint)
    with pytest.raises(ValueError): init_from_contact_gnn(JointGNN(width=128, rounds=2), checkpoint)
    with pytest.raises(ValueError): build_from_checkpoint(dict(kind='world_model', state_dict={}))
    with pytest.raises(ValueError): cg.build_from_checkpoint(dict(kind='joint_gnn', config=model.config, state_dict=model.state_dict()))


def test_speed_cpu():
    torch.manual_seed(0)
    model = JointGNN(width=128, rounds=6).eval()
    state, props, context, mask = random_boxes(1, 12, 13)
    joints, joint_mask = random_joints(1, 12, 12, 13)
    def clock(function, repeat=15):
        function(); times = []
        for _ in range(repeat):
            start = time.perf_counter(); function(); times.append(time.perf_counter()-start)
        return sorted(times)[repeat//2]*1e3                                    # median: the machine may be busy
    threads = torch.get_num_threads(); torch.set_num_threads(1)
    try:
        with torch.no_grad():
            # a consistent 10-joint chain (world - 0 - 2 - ... - 10, mixed joint types) with a model-sized error
            start = solve_joints(state, props, context, mask, joints, joint_mask, iters=30)
            pred = start.clone(); pred[..., :3] += torch.randn(1, 12, 3)*.002; pred[..., 7:] += torch.randn(1, 12, 6)*.05
            forward = clock(lambda: model(start, props, context, mask, joints, joint_mask))
            plain = clock(lambda: model(start, props, context, mask))
            contact = clock(lambda: cg.project(start, pred, props, context, mask))
            projection = clock(lambda: project_joints(start, pred, props, context, mask, joints, joint_mask))
    finally:
        torch.set_num_threads(threads)
    print(f'\nJointGNN N=12 J=12 (10 joints) CPU, 1 thread, median: forward {forward:.1f} ms/step (same scene without joints '
          f'{plain:.1f}), project_joints {projection:.1f} ms/step (contact_gnn.project alone {contact:.1f})')
    assert forward+projection < 2000


# ------------------------------------------------------- PyBullet ground truth
def bullet_episode(kind, steps=20, world=False, seed=0):
    """A two-box articulation (or one box on a world pivot) simulated by PyBullet's multibody solver.

    Returns Y-up states [T+1,N,13], props [N,9], joint row and the true joint angle per frame
    / displacement (hinge, prismatic).  Built in Bullet's Z-up frame and converted with agatha.coordinates.
    """
    rng = np.random.default_rng(seed)
    client = p.connect(p.DIRECT)
    try:
        p.setGravity(0, 0, -9.81 if world else -2., physicsClientId=client)   # the free pair must not fall far
        p.setPhysicsEngineParameter(fixedTimeStep=1/240, numSolverIterations=50, physicsClientId=client)
        half_base, half_link = (.1, .05, .02), (.09, .04, .015)
        pivot_in_base = (.1, 0., .03)              # pivot in the base frame (Bullet axes)
        com_from_pivot = (.09, .01, .02)           # link centre of mass seen from the pivot
        axis = (0., 1., 0.)                        # hinge axis (Bullet axes) = -z in the Y-up API frame
        make = lambda half: p.createCollisionShape(p.GEOM_BOX, halfExtents=half, physicsClientId=client)
        quat = p.getQuaternionFromEuler(rng.uniform(-.6, .6, 3).tolist())
        body = p.createMultiBody(
            baseMass=0. if world else .8, baseCollisionShapeIndex=-1 if world else make(half_base),
            basePosition=(.05, -.1, 2.2), baseOrientation=(0, 0, 0, 1) if world else quat,
            linkMasses=[.4], linkCollisionShapeIndices=[make(half_link)], linkVisualShapeIndices=[-1],
            linkPositions=[pivot_in_base], linkOrientations=[(0, 0, 0, 1)],
            linkInertialFramePositions=[com_from_pivot], linkInertialFrameOrientations=[(0, 0, 0, 1)],
            linkParentIndices=[0], linkJointTypes=[{HINGE: p.JOINT_REVOLUTE, BALL: p.JOINT_SPHERICAL, PRISMATIC: p.JOINT_PRISMATIC, FIXED: p.JOINT_FIXED}[kind]],
            linkJointAxis=[axis], physicsClientId=client)
        p.changeDynamics(body, -1, linearDamping=0, angularDamping=0, physicsClientId=client)
        p.changeDynamics(body, 0, linearDamping=0, angularDamping=0, jointDamping=0, physicsClientId=client)
        if kind in (HINGE, PRISMATIC):
            p.setJointMotorControl2(body, 0, p.VELOCITY_CONTROL, force=0, physicsClientId=client)
            p.resetJointState(body, 0, *((.4, 2.5) if kind == HINGE else (.03, .4)), physicsClientId=client)
        elif kind == BALL:
            p.setJointMotorControlMultiDof(body, 0, p.POSITION_CONTROL, targetPosition=[0, 0, 0, 1], positionGain=0, velocityGain=0,
                                           force=[0, 0, 0], physicsClientId=client)
            p.resetJointStateMultiDof(body, 0, p.getQuaternionFromEuler((.3, -.4, .5)), (1.5, -2., 1.), physicsClientId=client)
        if not world:
            p.resetBaseVelocity(body, (.3, -.2, 1.), (1., -2., .5), physicsClientId=client)

        def snapshot():
            rows = []
            if not world:
                pos, orn = p.getBasePositionAndOrientation(body, physicsClientId=client)
                vel, ang = p.getBaseVelocity(body, physicsClientId=client)
                rows.append([*from_bullet(pos), *quat_from_bullet(orn), *from_bullet(vel), *from_bullet(ang)])
            link = p.getLinkState(body, 0, computeLinkVelocity=1, computeForwardKinematics=1, physicsClientId=client)
            rows.append([*from_bullet(link[0]), *quat_from_bullet(link[1]), *from_bullet(link[6]), *from_bullet(link[7])])
            angle = p.getJointState(body, 0, physicsClientId=client)[0] if kind in (HINGE, PRISMATIC) else 0.
            return rows, angle
        frames, angles = [], []
        for step in range(steps+1):
            if step:
                for _ in range(12): p.stepSimulation(physicsClientId=client)
            rows, angle = snapshot(); frames.append(rows); angles.append(angle)
    finally:
        p.disconnect(client)
    size = lambda half: [2*abs(v) for v in from_bullet(half)]
    props = np.zeros((1 if world else 2, 9), np.float32)
    props[-1, :6] = [*size(half_link), .4, .5, 0.]
    anchor_child = from_bullet([-v for v in com_from_pivot])
    api_axis = from_bullet(axis) if kind in (HINGE, PRISMATIC) else (0., 0., 0.)
    limits = (-3.2, 3.2) if kind == HINGE else (-10., 10.) if kind == PRISMATIC else (0., 0.)
    if world:
        row = joint_row(-1, 0, kind, from_bullet([a+b for a, b in zip((.05, -.1, 2.2), pivot_in_base)]), anchor_child, api_axis, *limits)
    else:
        props[0, :6] = [*size(half_base), .8, .5, 0.]
        row = joint_row(0, 1, kind, from_bullet(pivot_in_base), anchor_child, api_axis, *limits)
    return torch.tensor(np.asarray(frames, np.float32)), torch.from_numpy(props), row, torch.tensor(angles)


@pytest.mark.parametrize('kind,world', [(HINGE, False), (BALL, False), (PRISMATIC, False), (FIXED, False), (HINGE, True), (BALL, True)])
def test_conventions_match_pybullet(kind, world):
    states, props, row, angles = bullet_episode(kind, world=world)
    frames, count = states.shape[:2]
    assert float((states[..., :3].amax(0)-states[..., :3].amin(0)).norm(dim=-1).min()) > .1   # it really moved
    assert float(states[..., 1].min()) > .3                                                # and stayed in the air
    props = props[None].expand(frames, -1, -1)
    context = torch.tensor([[1.2, .8, .75, 9.81, 0., 0., 0., 0., .5, 0.]]).expand(frames, -1)
    mask = torch.ones(frames, count, dtype=torch.bool)
    joints = row[None, None].expand(frames, 1, -1); joint_mask = torch.ones(frames, 1, dtype=torch.bool)
    violation = joint_violation(states, joints, joint_mask)
    name = f"{jg.JOINT_TYPES[kind]}{' (world)' if world else ''}"
    print(f"\nPyBullet {name}: anchor gap max {float(violation['anchor_gap'].max())*1e3:.4f} mm, "
          f"off-axis max {math.degrees(float(violation['off_axis'].max())):.4f} deg")
    assert float(violation['anchor_gap'].max()) < 1e-3
    assert float(violation['off_axis'].max()) < math.radians(.5)
    assert float(violation['limit_excess'].max()) == 0
    if kind in (HINGE, PRISMATIC):
        error = joint_coordinates(states, joints, joint_mask)[:, 0]-angles
        if kind == HINGE: error = torch.atan2(error.sin(), error.cos())
        assert float(angles.max()-angles.min()) > (.5 if kind == HINGE else .1)   # the joint really moved (while the pair spins)
        print(f'PyBullet {name}: joint coordinate error max {float(error.abs().max()):.2e}')
        assert float(error.abs().max()) < (2e-3 if kind == HINGE else 1e-4)
    # true states are a fixed point of the joint projection, with and without the contact clean-up
    for output in (solve_joints(states, props, context, mask, joints, joint_mask),
                   project_joints(states, states, props, context, mask, joints, joint_mask)):
        moved, speed = float((output[..., :3]-states[..., :3]).norm(dim=-1).max()), float((output[..., 7:]-states[..., 7:]).abs().max())
        turned = float((1-(output[..., 3:7]*states[..., 3:7]).sum(-1).abs()).max())
        print(f'PyBullet {name}: projection moves true states by {moved*1e3:.4f} mm, velocities by {speed:.2e}')
        assert moved < 1e-3 and turned < 1e-6 and speed < 2e-2
    # joint_losses of the truth against itself vanish
    assert float(sum(joint_losses(states, states, joints, joint_mask).values())) == 0


# ----------------------------------------------------------------------------- track-J improvements
def test_articulations_label_connected_bodies():
    rows = [joint_row(0, 1, HINGE, (.1, 0, 0), (-.1, 0, 0), (0, 0, 1.)), joint_row(1, 2, BALL, (.1, 0, 0), (-.1, 0, 0)),
            joint_row(-1, 4, BALL, (0, 2., 0), (0, .1, 0)), joint_row(-1, 5, BALL, (1., 2., 0), (0, .1, 0))]
    boxes = [dict(size=(.2, .04, .1), pos=(.25*i, 1.5, 0.)) for i in range(7)]
    _, _, _, _, joints, joint_mask = scene(boxes, rows, slots=6)
    assert jg.articulations(joints, joint_mask, 7)[0].tolist() == [0, 0, 0, -1, 4, 5, -1]     # world anchors do not connect
    assert jg.articulations(joints, torch.zeros_like(joint_mask), 7)[0].tolist() == [-1]*7


def test_tracking_losses_zero_at_truth_and_finite_gradients():
    state, props, context, mask, joints, joint_mask = hinge_pair(angle=.3)
    exact = jg.joint_tracking_losses(state, state, joints, joint_mask)
    assert float(exact['coord']) < 1e-8 and float(exact['relative']) < 1e-6
    moved = hinge_pair(angle=.4)[0]
    off = jg.joint_tracking_losses(moved, state, joints, joint_mask)
    assert float(off['coord']) == pytest.approx(1.5, abs=.02)            # smooth-L1 of 0.1 rad / 0.05 rad = 2 -> 2 - 0.5
    assert float(off['relative']) > 0
    assert float(jg.joint_tracking_losses(moved, state, joints, joint_mask, weight=torch.zeros(1, 1))['coord']) == 0
    # random predictions (including relative rotations near 180 degrees): finite loss and gradients
    state, props, context, mask = random_boxes(batch=8, seed=3)
    joints, joint_mask = random_joints(batch=8, seed=3)
    pred = state.clone().requires_grad_(True)
    flipped = state.clone(); flipped[:, ::2, 3:7] = quat_mul(axis_quat((1., 0., 0.), math.pi).expand(8, 6, 4), state[:, ::2, 3:7])
    losses = jg.joint_tracking_losses(pred, flipped, joints, joint_mask)
    (losses['coord']+losses['relative']).backward()
    assert torch.isfinite(losses['coord']) and torch.isfinite(losses['relative']) and torch.isfinite(pred.grad).all()


def test_joint_motion_flags_only_moving_joints():
    still = hinge_pair(angle=.3); moving = hinge_pair(angle=.35)
    window = torch.stack((still[0], still[0], moving[0]), 1)             # [1,3,N,13]
    assert bool(jg.joint_motion(window, still[4], still[5])[0, 0])
    assert not bool(jg.joint_motion(torch.stack((still[0],)*3, 1), still[4], still[5])[0, 0])
    carried = still[0].clone(); carried[..., 0] += .3                    # both bodies translate together: the joint does not move
    assert not bool(jg.joint_motion(torch.stack((still[0], carried), 1), still[4], still[5])[0, 0])


def test_gravity_torque_features_and_legacy_checkpoints(tmp_path):
    state, props, context, mask, joints, joint_mask = hinge_pair(angle=.3)
    new, old = JointGNN(width=32, rounds=2), JointGNN(width=32, rounds=2, joint_features=jg.LEGACY_JOINT_EDGE_DIM)
    assert new.config['joint_features'] == jg.JOINT_EDGE_DIM == 80 and old.config['joint_features'] == 73
    features = new.joint_graph(state, props, context, mask, joints, joint_mask)[0]
    legacy = old.joint_graph(state, props, context, mask, joints, joint_mask)[0]
    assert features.shape[1] == 80 and legacy.shape[1] == 73 and torch.equal(features[:, :73], legacy)
    assert float(features[:, 73:].abs().max()) > 0
    weightless = context.clone(); weightless[:, 3] = 0
    assert float(new.joint_graph(state, props, weightless, mask, joints, joint_mask)[0][:, 73:].abs().max()) == 0
    # a checkpoint written before the features existed has no 'joint_features' in its config
    ckpt = dict(kind='joint_gnn', config=dict(width=32, rounds=2, edge_radius=.12), state_dict=old.state_dict())
    loaded = build_from_checkpoint(ckpt)
    assert loaded.joint_features == 73
    assert torch.equal(loaded(state, props, context, mask, joints, joint_mask), old.eval()(state, props, context, mask, joints, joint_mask))
    assert build_from_checkpoint(dict(kind='joint_gnn', config=new.config, state_dict=new.state_dict())).joint_features == 80


def _resting_laptop(depth):
    """Thin base resting `depth` inside the enabled support with a lid hinged on it, plus a free box at the same depth."""
    top = .75
    boxes = [dict(size=(.3, .02, .2), pos=(0., top+.01-depth, 0.)), dict(size=(.3, .01, .2), pos=(0., top+.025-depth, 0.)),
             dict(size=(.1, .1, .1), pos=(.4, top+.05-depth, 0.))]
    return scene(boxes, [joint_row(0, 1, HINGE, (0., .01, -.1), (0., -.005, -.1), (-1., 0., 0.), 0., 2.)], enabled=1.)


def _sliding(case):
    case[0][..., 7] = .1                                   # everything slides at 0.1 m/s: faster than the sleep rule
    return case


def test_projection_slop_and_scope():
    state, props, context, mask, joints, joint_mask = _sliding(_resting_laptop(.0015))
    args = (state, state, props, context, mask, joints, joint_mask)
    old = project_joints(*args, joint_slop=0., articulation_sleep=False)
    assert float(old[0, 0, 1]-state[0, 0, 1]) > .001                                    # previous behaviour: lifted out every step
    new = project_joints(*args)
    assert float((new[..., :3]-state[..., :3]).abs().max()) < 1e-5                      # 1.5 mm rest depth is tolerated (scene scope)
    scoped = project_joints(*args, slop_scope='articulation')
    assert float((scoped[0, :2, :3]-state[0, :2, :3]).abs().max()) < 1e-5 and float(scoped[0, 2, 1]-state[0, 2, 1]) == pytest.approx(.0015, abs=1e-4)
    deep = _sliding(_resting_laptop(.01))
    lifted = project_joints(deep[0], deep[0], *deep[1:])
    assert .005 < float(lifted[0, 0, 1]-deep[0][0, 0, 1]) < .0085                       # beyond the slop it is still corrected (to the slop)
    with pytest.raises(ValueError): project_joints(*args, slop_scope='nope')
    # no joints: identical to contact_gnn.project whatever the joint options
    none = torch.zeros_like(joint_mask)
    assert torch.equal(project_joints(state, state, props, context, mask, joints, none), cg.project(state, state, props, context, mask))


def test_articulation_sleeps_only_as_a_whole():
    state, props, context, mask, joints, joint_mask = _resting_laptop(0.)
    state[0, 1, 3:7] = axis_quat((-1., 0., 0.), 1.)                                     # lid open at 1 rad ...
    state[0, 1, :3] = torch.tensor([0., .76, -.1])+rotation(state[0, 1, 3:7])@torch.tensor([0., .005, .1])
    state[0, 1, 10:13] = torch.tensor([2., 0., 0.])                                     # ... and falling shut at 2 rad/s
    state[0, 0, 7:10] = torch.tensor([.01, 0., 0.]); state[0, 2, 7:10] = torch.tensor([.01, 0., 0.])   # base and free box creep
    args = (state, state, props, context, mask, joints, joint_mask)
    frozen = project_joints(*args, articulation_sleep=False)
    assert float(frozen[0, 0, 7].abs()) < 1e-6 and float(frozen[0, 2, 7:10].norm()) == 0   # the base's creep was put to sleep
    awake = project_joints(*args)
    assert float(awake[0, 0, 7]) > .004                                                 # the base stays awake while its lid moves
    assert float(awake[0, 2, 7:10].norm()) == 0                                         # the free box sleeps as before
    state[0, 1, 10:13] = 0
    assert float(project_joints(state, state, *args[2:])[0, 0, 7].abs()) < 1e-6         # whole articulation slow: it sleeps


def _trainer():
    import sys
    scripts = str(Path(__file__).resolve().parents[1]/'scripts')
    if scripts not in sys.path: sys.path.insert(0, scripts)
    import train_joint_ddp
    return train_joint_ddp


def _train_args(tj, *extra):
    import sys
    old = sys.argv
    try:
        sys.argv = ['train_joint_ddp.py', '--data', 'x', '--out', 'y', *extra]
        return tj.parse_args()
    finally:
        sys.argv = old


def test_trainer_new_loss_terms_default_off_and_distillation():
    tj = _trainer()
    x = torch.linspace(-4, 4, 33)
    assert torch.allclose(tj.huber(x, torch.full_like(x, 2.5)), tj.huber(x, 2.5))
    torch.manual_seed(0)
    model = JointGNN(width=32, rounds=2); teacher = JointGNN(width=32, rounds=2)
    teacher.load_state_dict(model.state_dict())
    state, props, context, mask = random_boxes(batch=4, seed=5)
    joints, joint_mask = random_joints(batch=4, seed=5); joint_mask[2:] = False        # two jointed windows, two plain ones
    states = torch.stack((state, state, state), 1); rest = torch.zeros(4, 3, 12, dtype=torch.bool)
    batch = (states, rest, props, context, mask, joints, joint_mask)
    gen = lambda: torch.Generator().manual_seed(1)
    names = tj.TERMS
    base_loss, base, _ = tj.unroll(model, *batch, 2, _train_args(tj), None, gen())
    assert float(base[names.index('distill')]) == 0 and float(base[names.index('jcoord')]) == 0 and float(base[names.index('jrel')]) == 0
    args = _train_args(tj, '--w-distill', '2', '--w-jcoord', '1', '--w-jrel', '.5', '--w-jmove', '3', '--joint-one-delta', '2.5')
    loss, stats, bad = tj.unroll(model, *batch, 2, args, None, gen(), teacher)
    assert not bool(bad) and float(stats[names.index('jcoord')]) > 0 and float(stats[names.index('jrel')]) > 0
    assert float(stats[names.index('distill')]) == 0                                    # the student still equals its teacher
    with torch.no_grad(): model.decoder[-1].bias.add_(.5)
    loss, stats, _ = tj.unroll(model, *batch, 2, args, None, gen(), teacher)
    assert float(stats[names.index('distill')]) > .05
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    # validation (no generator) never uses the train-only options but reports the tracking terms
    _, val, _ = tj.unroll(model, *batch, 2, args, None, None, teacher)
    assert float(val[names.index('distill')]) == 0 and float(val[names.index('jcoord')]) > 0


def test_shared_rounds_keep_joint_free_scenes_on_the_box_model(checkpoint):
    contact = cg.build_from_checkpoint(checkpoint)
    model = JointGNN(width=128, rounds=6, shared_rounds=4)
    init_from_contact_gnn(model, checkpoint)
    torch.manual_seed(0)
    with torch.no_grad():                                   # "train" everything that is not inherited
        for name, parameter in model.named_parameters():
            if name.startswith(('joint_', 'world_node', 'edge_updates.4', 'edge_updates.5', 'node_updates.4', 'node_updates.5')):
                parameter.add_(torch.randn_like(parameter)*.3)
    model.eval()
    state, props, context, mask = random_boxes(batch=4, seed=7)
    joints, joint_mask = random_joints(batch=4, seed=7)
    with torch.no_grad():
        box = contact(state, props, context, mask)
        assert torch.equal(model(state, props, context, mask), box)                                    # no joints at all
        joint_mask[:2] = False                                                                           # mixed batch
        mixed = model(state, props, context, mask, joints, joint_mask)
        assert torch.allclose(mixed[:2], box[:2], atol=1e-6) and not torch.allclose(mixed[2:], box[2:], atol=1e-3)
        ungated = JointGNN(width=128, rounds=6); ungated.load_state_dict(model.state_dict()); ungated.eval()
        assert not torch.allclose(ungated(state, props, context, mask), box, atol=1e-3)                # what the gate prevents
    assert model.config['shared_rounds'] == 4
    again = build_from_checkpoint(dict(kind='joint_gnn', config=model.config, state_dict=model.state_dict()))
    with torch.no_grad(): assert torch.equal(again(state, props, context, mask, joints, joint_mask), mixed)


# ----------------------------------------------------------------------------- round 2
def test_grow_rounds_is_the_identity_and_keeps_box_scenes(checkpoint):
    tj = _trainer()
    model = JointGNN(width=128, rounds=6, shared_rounds=4)
    init_from_contact_gnn(model, checkpoint)
    torch.manual_seed(0)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.startswith(('joint_', 'world_node', 'edge_updates.4', 'edge_updates.5', 'node_updates.4', 'node_updates.5')):
                parameter.add_(torch.randn_like(parameter)*.1)
    model.eval()
    grown = tj.grow_rounds(model, 2)
    assert grown.config['rounds'] == 8 and grown.config['shared_rounds'] == 4 and not grown.training
    state, props, context, mask = random_boxes(batch=4, seed=11)
    joints, joint_mask = random_joints(batch=4, seed=11)
    with torch.no_grad():
        assert torch.equal(grown(state, props, context, mask, joints, joint_mask), model(state, props, context, mask, joints, joint_mask))
        for name, parameter in grown.named_parameters():      # train the new rounds: joint-free scenes must not notice
            if name.startswith(('edge_updates.6', 'edge_updates.7', 'node_updates.6', 'node_updates.7', 'joint_updates.6', 'joint_updates.7')):
                parameter.add_(torch.randn_like(parameter)*.3+.1)
        box = cg.build_from_checkpoint(checkpoint)(state, props, context, mask)
        assert torch.equal(grown(state, props, context, mask), box)
        assert not torch.allclose(grown(state, props, context, mask, joints, joint_mask), model(state, props, context, mask, joints, joint_mask), atol=1e-4)
    with pytest.raises(ValueError): tj.grow_rounds(JointGNN(width=32, rounds=2), 1)


def test_joint_rest_flag_extends_the_rest_penalty_to_held_links():
    tj = _trainer()
    torch.manual_seed(0)
    model = JointGNN(width=32, rounds=2)
    state, props, context, mask = random_boxes(batch=4, seed=5)
    joints, joint_mask = random_joints(batch=4, seed=5)
    state[..., 7:] = 0                                      # everything truly still, no rest flag (no contact)
    states = torch.stack((state, state, state), 1); rest = torch.zeros(4, 3, 12, dtype=torch.bool)
    batch = (states, rest, props, context, mask, joints, joint_mask)
    gen = lambda: torch.Generator().manual_seed(1)
    index = tj.TERMS.index('rest')
    _, off, _ = tj.unroll(model, *batch, 2, _train_args(tj), None, gen())
    _, on, _ = tj.unroll(model, *batch, 2, _train_args(tj, '--joint-rest', '1'), None, gen())
    _, val, _ = tj.unroll(model, *batch, 2, _train_args(tj, '--joint-rest', '1'), None, None)
    assert float(off[index]) == 0 and float(on[index]) > 0 and float(val[index]) == 0     # train-only


def test_solve_joints_round2_options():
    state, props, context, mask, joints, joint_mask = hinge_pair(angle=.4)
    broken = state.clone(); broken[0, 1, :3] += torch.tensor([.004, -.003, .002]); broken[0, 1, 7:10] = torch.tensor([.3, 0., 0.])
    args = (props, context, mask, joints, joint_mask)
    assert torch.equal(solve_joints(broken, *args, gap_free=.02), broken)                 # inside the dead band: untouched
    fixed = solve_joints(broken, *args, gap_turn=False, velocity=False)
    assert float(joint_violation(fixed, joints, joint_mask)['anchor_gap'].max()) < 1e-4
    assert torch.allclose(fixed[..., 3:7], broken[..., 3:7], atol=1e-6) and torch.equal(fixed[..., 7:], broken[..., 7:])   # translation only
    assert torch.allclose(joint_coordinates(fixed, joints, joint_mask), joint_coordinates(broken, joints, joint_mask), atol=1e-5)
    default = project_joints(state, broken, *args)
    assert float(joint_violation(default, joints, joint_mask)['anchor_gap'].max()) < 1e-4
