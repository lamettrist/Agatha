import math
import time
import numpy as np
import pytest
import torch
from agatha import contact_gnn as cg
from agatha.contact_gnn import ContactGNN, penetration_depths, physics_losses, project, box_corners, scene_tensors, build_from_checkpoint
from agatha.dynamics import euler_quaternion
from agatha.physics import simulate
from agatha.schema import Scene, SceneObject, Support, Push, Tilt
from agatha.world_model import prepare, state_array


def make(boxes, n=None, w=1.2, d=.8, h=.75, tilt=(0., 0., 0.), enabled=1., g=9.81):
    """boxes: list of dict(size, pos, quat?, vel?, ang?, mass?, static?)."""
    n = n or len(boxes)
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
    context = torch.tensor([[w, d, h, g, *tilt, enabled, .5, 0.]])
    return state, props, context, mask


def random_batch(batch=5, n=6, seed=0):
    rng = torch.Generator().manual_seed(seed)
    state = torch.zeros(batch, n, 13)
    state[..., 0] = torch.rand(batch, n, generator=rng)*.8-.4
    state[..., 1] = .78+torch.rand(batch, n, generator=rng)*.25
    state[..., 2] = torch.rand(batch, n, generator=rng)*.5-.25
    q = torch.randn(batch, n, 4, generator=rng); state[..., 3:7] = q/q.norm(dim=-1, keepdim=True)
    state[..., 7:10] = torch.randn(batch, n, 3, generator=rng)
    state[..., 10:13] = torch.randn(batch, n, 3, generator=rng)*4
    props = torch.zeros(batch, n, 9); props[..., 3] = 1
    props[..., :3] = .04+torch.rand(batch, n, 3, generator=rng)*.2
    props[..., 3] = .1+torch.rand(batch, n, generator=rng)*2
    props[..., 4] = .25+torch.rand(batch, n, generator=rng)*.5
    props[..., 6] = torch.rand(batch, n, generator=rng)*.3
    props[:, 1, 5] = 1                                   # a static obstacle
    mask = torch.ones(batch, n, dtype=torch.bool); mask[:, -1] = False
    state[:, -1] = 0; state[:, -1, 6] = 1; props[:, -1] = 0; props[:, -1, 3] = 1
    context = torch.tensor([[1.2, .8, .75, 9.81, .1, 0., -.05, 1., .5, 0.]]).repeat(batch, 1)
    context[:, 3] = torch.rand(batch, generator=rng)*10+2
    return state, props, context, mask


def model(seed=3, **kwargs):
    torch.manual_seed(seed)
    m = ContactGNN(**kwargs).eval()
    m.set_normalizer(torch.randn(12)*.01, torch.rand(12)*.05+.01)
    with torch.no_grad(): m.decoder[-1].weight.mul_(10)
    return m


def same_rotation(a, b, atol):
    return ((a*b).sum(-1).abs() > 1-atol).all()


def test_round_trip_and_dimensions():
    state, props, context, mask = random_batch()
    nxt, *_ = random_batch(seed=1)
    nxt[..., :3] = state[..., :3]+torch.randn_like(state[..., :3])*.05
    m = model()
    raw = m.raw_targets(state, nxt, props, context, mask)
    back = m.integrate(state, raw, props, context, mask)
    dynamic = cg.dynamic_mask(props, context, mask)
    assert torch.allclose(back[dynamic][:, :3], nxt[dynamic][:, :3], atol=1e-5)
    assert torch.allclose(back[dynamic][:, 7:], nxt[dynamic][:, 7:], atol=1e-5)
    assert same_rotation(back[dynamic][:, 3:7], nxt[dynamic][:, 3:7], 1e-5)
    assert torch.equal(back[~dynamic], state[~dynamic]) and (raw[~dynamic] == 0).all()
    # and the other way: integrate then raw_targets
    wanted = torch.randn(5, 6, 12)*.05*dynamic[..., None]
    assert torch.allclose(m.raw_targets(state, m.integrate(state, wanted, props, context, mask), props, context, mask), wanted, atol=1e-5)
    # zero residual == free flight
    free = m.integrate(state, torch.zeros(5, 6, 12), props, context, mask)
    assert torch.allclose(free[dynamic][:, 8], state[dynamic][:, 8]-(context[:, 3, None]*cg.STEP).expand(-1, 6)[dynamic], atol=1e-6)
    node, edge, receiver, sender = m.graph(state, props, context, mask)
    assert node.shape == (5*8, cg.NODE_DIM) and edge.shape[1] == cg.EDGE_DIM and len(receiver) == len(edge) > 0
    assert torch.isfinite(node).all() and torch.isfinite(edge).all()


def test_forward_keeps_static_and_padded_and_is_permutation_equivariant():
    state, props, context, mask = random_batch()
    m = model()
    with torch.inference_mode():
        out = m(state, props, context, mask)
        perm = torch.tensor([3, 5, 0, 2, 1, 4])
        permuted = m(state[:, perm], props[:, perm], context, mask[:, perm])
    assert torch.isfinite(out).all()
    assert torch.equal(out[:, 1], state[:, 1]) and torch.equal(out[:, -1], state[:, -1])
    assert (out[:, 0] != state[:, 0]).any()
    assert torch.allclose(out[..., 3:7].norm(dim=-1), torch.ones(5, 6), atol=1e-5)
    assert torch.allclose(out[:, perm], permuted, atol=2e-5)
    # old 6/8-column layouts are accepted with live defaults
    with torch.inference_mode():
        live_props = props.clone(); live_props[..., 6:] = 0
        assert torch.allclose(m(state, props[..., :6], context[:, :8], mask), m(state, live_props, context, mask), atol=1e-6)


def test_xz_translation_invariance_without_support():
    state, props, context, mask = random_batch()
    context[:, 7] = 0
    m = model()
    shift = torch.tensor([1.7, 0., -2.3])
    moved = state.clone(); moved[..., :3] += shift*mask[..., None]
    with torch.inference_mode():
        a = m.predict_raw(state, props, context, mask); b = m.predict_raw(moved, props, context, mask)
    assert torch.allclose(a, b, atol=2e-4)


def test_alias_desk_masked_but_unchanged():
    scene = Scene(objects=[SceneObject(id='desk', label='desk', size=(1.2, .078, .8), position=(0, .71, 0), is_static=True),
                           SceneObject(id='mug', label='cup', size=(.085, .11, .085), position=(.1, .805, 0))])
    prepared, states, g, tilt = prepare(scene, [])
    state, props, context, mask = scene_tensors(prepared, states, g, tilt)
    assert props.shape == (1, 2, 9) and context.shape == (1, 10) and mask.tolist() == [[False, True]]
    assert context[0, 8:].tolist() == [.5, 0.]
    m = model()
    with torch.inference_mode():
        out = m(state, props, context, mask)
        with_alias_in_mask = m(state, props, context, torch.ones_like(mask))
    assert torch.equal(out[0, 0], state[0, 0]) and torch.equal(out, with_alias_in_mask)
    assert penetration_depths(state, props, context, torch.ones_like(mask))['objects'].max() == 0
    empty = scene_tensors(Scene(objects=[]), np.zeros((0, 13), np.float32), 9.81, [0, 0, 0])
    assert not empty[3].any() and torch.isfinite(m(*empty)).all()


def test_checkpoint_round_trip(tmp_path):
    m = model(width=32, rounds=2, edge_radius=.1)
    ckpt = {'kind': 'contact_gnn', 'config': {**m.config, 'note': 'ignored'}, 'state_dict': m.state_dict(), 'model_step_s': .05, 'epoch': 1, 'train_args': {'lr': 1e-3}}
    torch.save(ckpt, tmp_path/'m.pt')
    loaded = build_from_checkpoint(torch.load(tmp_path/'m.pt', map_location='cpu', weights_only=True))
    args = random_batch()
    assert not loaded.training and loaded.edge_radius == .1
    with torch.inference_mode(): assert torch.equal(loaded(*args), m(*args))


def test_penetration_depths_analytic():
    # 10 cm cube sunk 1 cm in the table top (top at .75)
    args = make([dict(size=(.1, .1, .1), pos=(0, .79, 0))])
    depth = penetration_depths(*args)
    assert depth['support'].item() == pytest.approx(.01, abs=1e-5) and depth['floor'].item() == 0 and depth['objects'].item() == 0
    # legal resting box
    assert penetration_depths(*make([dict(size=(.1, .1, .1), pos=(0, .8, 0))]))['support'].item() < 1e-6
    # collisions disabled
    args = make([dict(size=(.1, .1, .1), pos=(0, .79, 0))], enabled=0.)
    assert penetration_depths(*args)['support'].item() == 0
    # beside the table, lower than the top; and below the table
    args = make([dict(size=(.1, .1, .1), pos=(.7, .72, 0)), dict(size=(.1, .1, .1), pos=(0, .3, 0))])
    assert penetration_depths(*args)['support'].abs().max().item() == 0
    # hanging over the edge, resting on the top surface: no penetration
    assert penetration_depths(*make([dict(size=(.2, .1, .1), pos=(.62, .8, 0))]))['support'].item() < 1e-6
    # box 2 cm inside another (side by side along x), static partner counts, padded slot ignored
    args = make([dict(size=(.1, .1, .1), pos=(0, .9, 0)), dict(size=(.1, .2, .2), pos=(.08, .9, 0), static=True)], n=3)
    depth = penetration_depths(*args)
    assert depth['objects'][0, :2].tolist() == pytest.approx([.02, .02], abs=1e-5) and depth['objects'][0, 2] == 0
    # small box sunk 5 mm into a larger one below it
    args = make([dict(size=(.3, .1, .3), pos=(0, .8, 0)), dict(size=(.05, .05, .05), pos=(.05, .87, .02))])
    assert penetration_depths(*args)['objects'][0].tolist() == pytest.approx([.005, .005], abs=1e-5)
    # floor
    args = make([dict(size=(.1, .1, .1), pos=(2, .03, 0))])
    assert penetration_depths(*args)['floor'].item() == pytest.approx(.02, abs=1e-6)
    # tilted support: box aligned with the tilted surface at x=.3, sunk 7 mm along the normal
    tilt = (0., 0., .3)
    q = euler_quaternion(torch.tensor(tilt)); r = cg.rotation(q)
    centre = torch.tensor([0, .71, 0.])
    for sink, expected in ((.007, .007), (-.002, 0.)):
        pos = centre+r@torch.tensor([.3, .04+.05-sink, .1])
        args = make([dict(size=(.1, .1, .1), pos=pos.tolist(), quat=q)], tilt=tilt)
        assert penetration_depths(*args)['support'].item() == pytest.approx(expected, abs=1e-5)
    # differentiable
    state, props, context, mask = make([dict(size=(.1, .1, .1), pos=(0, .79, 0))])
    state.requires_grad_(True)
    penetration_depths(state, props, context, mask)['support'].sum().backward()
    assert state.grad[0, 0, 1].item() == pytest.approx(-1, abs=1e-4)


def test_box_corners():
    state, props, _, _ = make([dict(size=(.2, .1, .4), pos=(1, 2, 3))])
    corners = box_corners(state, props)
    assert corners.shape == (1, 1, 8, 3)
    assert torch.allclose(corners.amin(2)[0, 0], torch.tensor([.9, 1.95, 2.8])) and torch.allclose(corners.amax(2)[0, 0], torch.tensor([1.1, 2.05, 3.2]))


def test_project_removes_penetration():
    cases = [make([dict(size=(.1, .1, .1), pos=(0, .79, 0), vel=(.3, -1., 0))]),
             make([dict(size=(.1, .1, .1), pos=(0, .9, 0), vel=(.5, 0, 0)), dict(size=(.1, .2, .2), pos=(.08, .9, 0), static=True)], n=3),
             make([dict(size=(.3, .1, .3), pos=(0, .8, 0)), dict(size=(.05, .05, .05), pos=(.05, .87, .02), vel=(0, -.4, 0))]),
             make([dict(size=(.1, .1, .1), pos=(2, .03, 0), vel=(0, -2, 0))]),
             make([dict(size=(.1, .1, .1), pos=(0, .79, 0)), dict(size=(.1, .1, .1), pos=(.01, .885, 0)), dict(size=(.1, .1, .1), pos=(0, .98, .01))])]
    for index, (state, props, context, mask) in enumerate(cases):
        before = penetration_depths(state, props, context, mask)
        assert max(v.max().item() for v in before.values()) > .004
        prev = state.clone(); prev[..., 1] += .05
        out = project(prev, state, props, context, mask)
        after = penetration_depths(out, props, context, mask)
        assert max(v.max().item() for v in after.values()) < 1e-3, index
        assert torch.isfinite(out).all()
        static = ~cg.dynamic_mask(props, context, mask)
        assert torch.equal(out[static], state[static])
    # velocities into the contact are removed, tangential ones kept
    out = project(cases[0][0], *cases[0])
    assert out[0, 0, 8].item() == 0 and out[0, 0, 7].item() == pytest.approx(.3)
    out = project(cases[1][0], *cases[1])
    assert out[0, 0, 7].item() <= 1e-6 and out[0, 0, 0].item() == pytest.approx(-.02, abs=1e-4)
    # mass split: light box moves more
    state, props, context, mask = make([dict(size=(.1, .1, .1), pos=(0, 1.2, 0), mass=1.), dict(size=(.1, .1, .1), pos=(.09, 1.2, 0), mass=3.)])
    out = project(state, state, props, context, mask, iters=1)
    assert (out[0, :, 0]-state[0, :, 0]).tolist() == pytest.approx([-.0075, .0025], abs=1e-5)


def test_project_leaves_legal_states_and_handles_edges():
    # legal resting box with a legitimate fast velocity: untouched
    state, props, context, mask = make([dict(size=(.1, .1, .1), pos=(0, .8, 0), vel=(.5, 0, 0))])
    assert torch.allclose(project(state, state, props, context, mask), state, atol=1e-7)
    # sleep: tiny drift velocities and a sub-mm pose change are removed
    prev, *_ = make([dict(size=(.1, .1, .1), pos=(0, .8, 0))])
    state, props, context, mask = make([dict(size=(.1, .1, .1), pos=(.0006, .8002, 0), vel=(.012, .004, 0), ang=(.05, 0, 0))])
    out = project(prev, state, props, context, mask)
    assert torch.equal(out[..., 7:], torch.zeros(1, 1, 6)) and torch.allclose(out[..., :7], prev[..., :7], atol=1e-7)
    # slow but moved more than 2 mm: velocity zeroed, pose kept
    state2 = state.clone(); state2[0, 0, 0] = .004
    out = project(prev, state2, props, context, mask)
    assert out[0, 0, 0].item() == pytest.approx(.004) and out[0, 0, 7:].abs().max() == 0
    # slow in free flight (apex): not put to sleep
    state3, props3, context3, mask3 = make([dict(size=(.1, .1, .1), pos=(0, 1.0, 0), vel=(.01, .01, 0))])
    assert torch.equal(project(state3, state3, props3, context3, mask3), state3)
    # hanging beyond the table edge / beside the table and lower than its top: not lifted
    for pos in ((.66, .74, 0), (.7, .5, 0), (.3, .5, 0)):
        state, props, context, mask = make([dict(size=(.1, .1, .1), pos=pos, vel=(0, -1., 0))])
        assert torch.equal(project(state, state, props, context, mask), state), pos
    # collisions disabled: falls through
    state, props, context, mask = make([dict(size=(.1, .1, .1), pos=(0, .76, 0), vel=(0, -1., 0))], enabled=0.)
    assert torch.equal(project(state, state, props, context, mask), state)
    # NaN prediction falls back to the previous state
    state, props, context, mask = make([dict(size=(.1, .1, .1), pos=(0, .8, 0))])
    bad = state.clone(); bad[0, 0, 0] = float('nan')
    assert torch.isfinite(project(state, bad, props, context, mask)).all()
    # batched with random garbage: finite, normalized, statics untouched
    state, props, context, mask = random_batch(batch=16, n=6, seed=4)
    out = project(state, state, props, context, mask)
    assert torch.isfinite(out).all() and torch.allclose(out[..., 3:7].norm(dim=-1)[mask], torch.ones(int(mask.sum())), atol=1e-5)
    assert torch.equal(out[:, 1], state[:, 1]) and torch.equal(out[:, -1], state[:, -1])


def test_physics_losses_and_gradients():
    state, props, context, mask = make([dict(size=(.1, .1, .1), pos=(0, .8, 0)), dict(size=(.1, .1, .1), pos=(.3, .8, 0))])
    rest = torch.tensor([[True, False]])
    zero = physics_losses(state, state, state, props, context, mask, rest)
    assert all(v.item() == 0 for v in zero.values())
    pred = state.clone(); pred[0, 0, 1] -= .01; pred[0, 0, 0] += .005; pred[0, 0, 7] = .1; pred[0, 1, 7] = 3.
    pred.requires_grad_(True)
    losses = physics_losses(pred, state, state, props, context, mask, rest)
    assert all(v.item() > 0 and v.ndim == 0 for v in losses.values())
    sum(losses.values()).backward()
    assert pred.grad[0, 0, 1] < 0 and pred.grad[0, 0, 7] > 0 and pred.grad[0, 0, 0] > 0
    assert pred.grad[0, 1, 7] == 0                         # the non-resting object is free to move
    m = model().train()
    state, props, context, mask = random_batch()
    out = m(state, props, context, mask)
    loss = out.square().sum()+sum(physics_losses(out, state, state, props, context, mask, mask).values())
    loss.backward()
    grads = [p.grad for p in m.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads) and sum(g.abs().sum() for g in grads) > 0


def test_forward_speed_cpu():
    torch.set_num_threads(4)
    m = ContactGNN().eval()
    print('parameters', sum(p.numel() for p in m.parameters()))
    for n in (8, 40):
        # a crowded table: grid of boxes with ~1 cm gaps, some stacked
        boxes = [dict(size=(.08, .08, .08), pos=(-.4+.09*(i % 10), .79+.08*(i//20), -.2+.09*((i//10) % 2))) for i in range(n)]
        state, props, context, mask = make(boxes)
        edges = len(m.graph(state, props, context, mask)[2])
        with torch.inference_mode():
            for _ in range(3): m(state, props, context, mask)
            start = time.perf_counter()
            for _ in range(20): project(state, m(state, props, context, mask), props, context, mask)
            both = (time.perf_counter()-start)/20*1000
            start = time.perf_counter()
            for _ in range(20): m(state, props, context, mask)
            elapsed = (time.perf_counter()-start)/20*1000
        print(f'N={n}: {edges} edges, forward {elapsed:.1f} ms/step, forward+project {both:.1f} ms/step')
        assert elapsed < 50


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA (.venv-train)')
def test_autocast_cuda():
    m = model().cuda()
    args = [a.cuda() for a in random_batch()]
    with torch.inference_mode():
        reference = m(*args)
        with torch.autocast('cuda', dtype=torch.float16): out = m(*args)
    assert out.dtype == torch.float32 and torch.isfinite(out).all()
    assert torch.allclose(out, reference, atol=5e-3)
    m.train()
    with torch.autocast('cuda', dtype=torch.float16):
        out = m(*args)
        loss = out.square().mean()+sum(physics_losses(out, args[0], args[0], *args[1:], args[3]).values())
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters())


# ---------------------------------------------------------------- PyBullet ground truth
def physics_cases():
    stack = [SceneObject(id='a', label='box', size=(.2, .1, .2), position=(0, .8, 0)),
             SceneObject(id='b', label='box', size=(.1, .08, .1), position=(.02, .89, .01), mass_kg=.3),
             SceneObject(id='c', label='box', size=(.06, .05, .06), position=(.03, .955, 0), mass_kg=1.5),
             SceneObject(id='d', label='box', size=(.3, .02, .2), position=(-.35, .76, .1), mass_kg=1.),
             SceneObject(id='e', label='box', size=(.07, .2, .07), position=(.4, .85, -.2), rotation_euler=(0, .6, 0)),
             SceneObject(id='s', label='box', size=(.1, .15, .1), position=(.3, .825, .2), is_static=True)]
    desk = [SceneObject(id='desk', label='desk', size=(1.2, .078, .8), position=(0, .71, 0), is_static=True),
            SceneObject(id='laptop', label='laptop', size=(.33, .025, .23), position=(-.2, .7625, -.12), mass_kg=1.4),
            SceneObject(id='mug', label='cup', size=(.085, .11, .085), position=(-.2, .83, -.12), mass_kg=.3),
            SceneObject(id='book', label='book', size=(.2, .04, .15), position=(.25, .77, .1), mass_kg=.6)]
    return [('rest', Scene(objects=stack), []),
            ('desk', Scene(objects=desk), []),
            ('tilt_z', Scene(objects=stack[:5]), [Tilt(type='tilt_support', angle_deg=12, axis='z')]),
            ('tilt_x', Scene(objects=stack[:5]), [Tilt(type='tilt_support', angle_deg=-9, axis='x')]),
            ('tilt_slide', Scene(objects=stack[:5]), [Tilt(type='tilt_support', angle_deg=35, axis='z')]),
            ('push', Scene(objects=stack), [Push(type='push', id='a', direction=(1, .3), impulse_Ns=1.2), Push(type='push', id='e', direction=(-1, 0), impulse_Ns=.6)]),
            ('floor', Scene(objects=stack[:5], support=Support(collisions_enabled=False, height=.75)), [])]


@pytest.mark.parametrize('name,scene,actions', physics_cases(), ids=[c[0] for c in physics_cases()])
def test_geometry_matches_pybullet(name, scene, actions):
    result = simulate(scene, actions, duration=3, fps=20)
    prepared, initial, g, tilt = prepare(scene, actions)
    ids = [o.id for o in prepared.objects]
    frames = np.stack([state_array(f, ids) for f in result['frames']])
    assert np.allclose(frames[0, :, :7], initial[:, :7], atol=1e-5)
    assert result['support_rotation_euler'] == tilt
    # Bullet itself penetrates transiently (landing after a fall: ~14 mm decaying over ~4 frames; the
    # instant tilt at t=0 leaves boxes inside the slab), so the strict checks apply to quiet frames.
    worst = 0.; quiet = 0; settled = 0.; moved = 0.; turned = 0.; speed = 0.
    for t in range(1, len(frames)):
        prev, props, context, mask = scene_tensors(prepared, frames[t-1], g, tilt)
        state = scene_tensors(prepared, frames[t], g, tilt)[0]
        depth = max(v.max().item() for v in penetration_depths(state, props, context, torch.ones_like(mask)).values())
        worst = max(worst, depth)
        if t >= len(frames)-6: settled = max(settled, depth)
        # sleeping disabled here: the pure geometric projection must not disturb true states
        out = project(prev, state, props, context, mask, sleep_lin=0., sleep_ang=0.)
        if depth < .05: assert max(v.max().item() for v in penetration_depths(out, props, context, mask).values()) < 1e-3
        shift = (out[..., :3]-state[..., :3]).norm(dim=-1).max().item()
        assert shift <= 3*depth+1e-6
        if depth < .001:
            quiet += 1; moved = max(moved, shift)
            turned = max(turned, (1-(out[..., 3:7]*state[..., 3:7]).sum(-1).abs()).max().item())
            speed = max(speed, (out[..., 7:10]-state[..., 7:10]).norm(dim=-1).max().item())
    print(f'{name}: true-state penetration worst {worst*1000:.2f} mm (transient), settled {settled*1000:.2f} mm; '
          f'{quiet}/{len(frames)-1} quiet frames, project moved them <= {moved*1000:.2f} mm, dv <= {speed:.3f} m/s')
    assert settled < .003 and (settled < .001 or name == 'tilt_slide')
    assert quiet >= .7*(len(frames)-1) and moved < .002 and turned < 1e-6 and speed < .25
    if name in ('rest', 'desk', 'tilt_z', 'tilt_x'):
        # with default sleeping, a settled true state stays exactly where it is
        prev, props, context, mask = scene_tensors(prepared, frames[-2], g, tilt)
        state = scene_tensors(prepared, frames[-1], g, tilt)[0]
        out = project(prev, state, props, context, mask)
        assert (out[..., :3]-state[..., :3]).norm(dim=-1).max().item() < .002
        assert out[..., 7:].abs().max().item() == 0


def test_project_review_regressions():
    # a spinning box landing on a corner must not be launched: projection never adds kinetic energy
    half = math.radians(30)/2
    state, props, context, mask = make([dict(size=(.1, .1, .1), pos=(2, .06, 0), quat=(0, 0, math.sin(half), math.cos(half)),
                                             vel=(0, -2.5, 0), ang=(-20., 0, -30.))])
    assert penetration_depths(state, props, context, mask)['floor'].item() > .005
    out = project(state, state, props, context, mask)
    energy = lambda s: (.5*s[0, 0, 7:10].square().sum()+.5*(.01/6)*s[0, 0, 10:13].square().sum()).item()
    assert penetration_depths(out, props, context, mask)['floor'].item() < 1e-4
    assert energy(out) <= energy(state)+1e-6 and out[0, 0, 8].item() < .3
    # sub-millimetre penetration: pose fixed, velocity untouched
    state, props, context, mask = make([dict(size=(.1, .1, .1), pos=(2, .0495, 0), vel=(.4, -1., 0))])
    out = project(state, state, props, context, mask)
    assert out[0, 0, 1].item() == pytest.approx(.05, abs=1e-6) and torch.equal(out[0, 0, 7:], state[0, 0, 7:])
    # on a slope steeper than the friction angle a slow object is not put to sleep; below it, it is
    tilt = (0., 0., .2); q = euler_quaternion(torch.tensor(tilt)); r = cg.rotation(q)
    pos = torch.tensor([0, .71, 0.])+r@torch.tensor([.3, .04+.05, .1])
    for friction, sleeps in ((.1, False), (.5, True)):
        state, props, context, mask = make([dict(size=(.1, .1, .1), pos=pos.tolist(), quat=q, vel=(r@torch.tensor([-.01, 0., 0.])).tolist())], tilt=tilt)
        props[0, 0, 4] = friction
        out = project(state, state, props, context, mask)
        assert (out[0, 0, 7:].abs().max().item() == 0) == sleeps, friction
    # a desk duplicating the support is an alias only while the support is flat (or disabled) and at its height
    desk = dict(size=(1.2, .078, .8), pos=(0, .711, 0), static=True); mug = dict(size=(.08, .1, .08), pos=(.1, .8, 0))
    for kwargs, alias in ((dict(), True), (dict(tilt=(0., 0., .3)), False), (dict(tilt=(0., 0., .3), enabled=0.), True)):
        state, props, context, mask = make([desk, mug], **kwargs)
        assert cg.alias_mask(props, context, state)[0].tolist() == [alias, False]
    state, props, context, mask = make([dict(desk, pos=(0, .4, 0)), mug])
    assert not cg.alias_mask(props, context, state).any()
    # a tall box standing under the table (2 cm below the slab) does not penetrate the support
    state, props, context, mask = make([dict(size=(.1, .65, .1), pos=(0, .325, 0))])
    assert penetration_depths(state, props, context, mask)['support'].item() == 0
