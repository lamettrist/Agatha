import math
import time
import pytest
import torch
from agatha import soft_gnn as sg
from agatha.soft_gnn import (SoftGNN, vertex_penetration, project_soft, rollout, soft_losses, acceleration_loss,
                             edge_strain, build_from_checkpoint, prev_from_velocity)
from agatha.dynamics import euler_quaternion

V, M = 128, 768


def context(batch=1, w=1.2, d=.8, h=.75, g=9.81, tilt=(0., 0., 0.), enabled=1.):
    return torch.tensor([[w, d, h, g, *tilt, enabled, .5, 0.]]).repeat(batch, 1)


def cube(n=5, size=.12, centre=(0., .75+.06, 0.), pad_v=V, pad_m=M):
    """Grid cube: verts [1,V,3], mask [1,V], edges [1,M,2] (axis + one face diagonal + body diagonal), rest_len [1,M]."""
    axis = torch.linspace(-size/2, size/2, n)
    grid = torch.stack(torch.meshgrid(axis, axis, axis, indexing='ij'), -1).reshape(-1, 3)+torch.tensor(centre)
    index = lambda i, j, k: (i*n+j)*n+k
    pairs = set()
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for di, dj, dk in ((1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (0, 1, 1), (1, 0, 1), (1, 1, 1)):
                    a, b, c = i+di, j+dj, k+dk
                    if a < n and b < n and c < n: pairs.add((index(i, j, k), index(a, b, c)))
    pairs = sorted(pairs)[:pad_m]
    count = n**3
    verts = torch.zeros(1, pad_v, 3); verts[0, :count] = grid
    mask = torch.zeros(1, pad_v, dtype=torch.bool); mask[0, :count] = True
    edges = torch.full((1, pad_m, 2), -1, dtype=torch.int16); edges[0, :len(pairs)] = torch.tensor(pairs, dtype=torch.int16)
    rest = torch.zeros(1, pad_m)
    rest[0, :len(pairs)] = (grid[edges[0, :len(pairs), 1].long()]-grid[edges[0, :len(pairs), 0].long()]).norm(dim=-1)
    return verts, mask, edges, rest


def boxes(batch=1, n=4, pos=(0., .80, 0.), size=(.2, .1, .2)):
    """One real box at ``pos`` (slot 0), the other slots padding."""
    state = torch.zeros(batch, n, 13); state[..., 6] = 1
    props = torch.zeros(batch, n, 9); props[..., 3] = 1
    mask = torch.zeros(batch, n, dtype=torch.bool)
    state[:, 0, :3] = torch.tensor(pos); props[:, 0, :6] = torch.tensor([*size, .5, .5, 0.]); mask[:, 0] = True
    return state, props, mask


def model(seed=3, **kwargs):
    torch.manual_seed(seed)
    m = SoftGNN(**kwargs).eval()
    m.set_normalizer(torch.randn(3)*.5, torch.rand(3)*5+2, vel_scale=.7, props_mean=torch.rand(6)*.2, props_std=torch.rand(6)+.5)
    with torch.no_grad(): m.decoder[-1].weight.mul_(10)
    return m


def random_batch(batch=3, seed=0, v=V, m=M, n=4):
    rng = torch.Generator().manual_seed(seed)
    verts = torch.rand(batch, v, 3, generator=rng)*.15+torch.tensor([-.05, .78, -.05])
    prev = verts-torch.randn(batch, v, 3, generator=rng)*.02
    mask = torch.ones(batch, v, dtype=torch.bool); mask[1:, v-9:] = False
    edges = torch.randint(0, v, (batch, m, 2), generator=rng).to(torch.int16)
    edges[:, m-20:] = -1
    rest = torch.rand(batch, m, generator=rng)*.04+.01
    soft = torch.rand(batch, 6, generator=rng)
    ctx = context(batch, tilt=(.05, 0., -.03)); ctx[:, 3] = torch.rand(batch, generator=rng)*8+2
    state = torch.zeros(batch, n, 13)
    state[..., 0] = torch.rand(batch, n, generator=rng)*.3-.15; state[..., 1] = .78+torch.rand(batch, n, generator=rng)*.1
    state[..., 2] = torch.rand(batch, n, generator=rng)*.3-.15
    q = torch.randn(batch, n, 4, generator=rng); state[..., 3:7] = q/q.norm(dim=-1, keepdim=True)
    state[..., 7:13] = torch.randn(batch, n, 6, generator=rng)
    props = torch.zeros(batch, n, 9); props[..., :3] = .05+torch.rand(batch, n, 3, generator=rng)*.1
    props[..., 3] = .5; props[..., 4] = .5
    rmask = torch.ones(batch, n, dtype=torch.bool); rmask[:, -1] = False
    return verts, prev, mask, edges, rest, soft, ctx, state, props, rmask


def test_shapes_and_optional_boxes():
    m = model()
    verts, prev, mask, edges, rest, soft, ctx, state, props, rmask = random_batch()
    vert, box, mesh, contact, centre, star = m.graph(verts, prev, mask, edges, rest, soft, ctx, state, props, rmask)
    assert vert.shape == (3*V, sg.VERT_DIM) and box.shape == (3*4, sg.BOX_DIM)
    assert mesh[0].shape[1] == sg.MESH_EDGE_DIM and contact[0].shape[1] == sg.CONTACT_EDGE_DIM
    assert mesh[0].shape[0] == mesh[1].numel() == mesh[2].numel() and mesh[0].shape[0] % 2 == 0
    assert contact[0].shape[0] > 0 and int(contact[2].min()) % (V+4) >= V          # senders are box nodes
    assert all(torch.isfinite(x).all() for x in (vert, box, mesh[0], contact[0]))
    raw = m.predict_raw(verts, prev, mask, edges, rest, soft, ctx, state, props, rmask)
    nxt = m(verts, prev, mask, edges, rest, soft, ctx, state, props, rmask)
    assert raw.shape == nxt.shape == (3, V, 3) and raw.dtype == nxt.dtype == torch.float32
    assert torch.equal(nxt[~mask], verts[~mask])                                  # padded vertices never move
    without = m(verts, prev, mask, edges, rest, soft, ctx)
    assert without.shape == (3, V, 3) and not torch.allclose(without, nxt)
    assert torch.allclose(m(verts, prev, mask, edges, rest, soft, ctx, state[:, :0], props[:, :0], rmask[:, :0]), without)
    masked = m(verts, prev, mask, edges, rest, soft, ctx, state, props, torch.zeros_like(rmask))
    assert torch.allclose(masked, without, atol=1e-6)                             # masked boxes are invisible


def test_integrator_inverse_and_free_fall():
    m = model()
    verts, prev, mask, edges, rest, soft, ctx, *_ = random_batch()
    nxt = verts+torch.randn_like(verts)*.02
    raw = m.raw_targets(verts, prev, nxt, ctx, mask)
    assert torch.allclose(m.integrate(verts, prev, raw, ctx, mask)[mask], nxt[mask], atol=1e-6)
    fall = m.integrate(verts, verts, torch.zeros_like(verts), ctx, mask)
    assert torch.allclose((fall-verts)[mask][:, 1], (-ctx[:, 3]*sg.STEP**2)[:, None].expand(-1, V)[mask], atol=1e-6)
    assert torch.allclose(prev_from_velocity(verts, (verts-prev)/sg.STEP), prev, atol=1e-6)


def test_permutation_equivariance_over_vertices():
    m = model()
    verts, prev, mask, edges, rest, soft, ctx, state, props, rmask = random_batch()
    out = m(verts, prev, mask, edges, rest, soft, ctx, state, props, rmask)
    perm = torch.randperm(V, generator=torch.Generator().manual_seed(1))
    inverse = torch.empty_like(perm); inverse[perm] = torch.arange(V)
    remapped = torch.where(edges >= 0, inverse[edges.long().clamp_min(0)].to(edges.dtype), edges)
    order = torch.randperm(M, generator=torch.Generator().manual_seed(2))             # edge order is irrelevant too
    permuted = m(verts[:, perm], prev[:, perm], mask[:, perm], remapped[:, order], rest[:, order], soft, ctx, state, props, rmask)
    assert torch.allclose(permuted, out[:, perm], atol=2e-5)


def test_translation_invariance_xz_without_support_and_not_with_it():
    m = model()
    verts, prev, mask, edges, rest, soft, ctx, state, props, rmask = random_batch()
    shift = torch.tensor([.31, 0., -.17])
    moved_state = state.clone(); moved_state[..., :3] += shift
    off = ctx.clone(); off[:, 7] = 0                                              # support disabled: only the floor is left
    base = m(verts, prev, mask, edges, rest, soft, off, state, props, rmask)
    moved = m(verts+shift, prev+shift, mask, edges, rest, soft, off, moved_state, props, rmask)
    assert torch.allclose(moved-shift, base, atol=2e-5)
    base = m(verts, prev, mask, edges, rest, soft, ctx, state, props, rmask)
    moved = m(verts+shift, prev+shift, mask, edges, rest, soft, ctx, moved_state, props, rmask)
    assert not torch.allclose(moved-shift, base, atol=1e-4)                       # the slab is centred at the origin


def test_resting_cube_has_no_penetration_after_projection():
    m = model()
    verts, mask, edges, rest = cube()                                              # bottom layer exactly on the table top
    ctx = context(); soft = torch.tensor([[.3, .5, .1, .5, 1., .0017]])
    assert vertex_penetration(verts, mask, ctx)['support'].max() < 1e-6
    nxt = m(verts, verts, mask, edges, rest, soft, ctx)
    fixed = project_soft(verts, nxt, mask, ctx, edges=edges, rest_len=rest)
    depth = vertex_penetration(fixed, mask, ctx)
    assert max(float(d.max()) for d in depth.values()) < 1e-5
    held = project_soft(verts, verts, mask, ctx)                                   # a clean state is left alone
    assert torch.allclose(held, verts, atol=1e-7)
    margin = project_soft(verts, verts, mask, ctx, margin=.004)
    assert float(margin[0, mask[0], 1].min()) == pytest.approx(.754, abs=1e-5)


def test_vertices_pushed_into_table_floor_and_box_come_out():
    verts, mask, edges, rest = cube()
    ctx = context()
    sunk = verts.clone(); sunk[..., 1] -= .01                                      # 1 cm into the table
    assert float(vertex_penetration(sunk, mask, ctx)['support'].max()) == pytest.approx(.01, abs=1e-5)
    out = project_soft(verts, sunk, mask, ctx)
    assert float(vertex_penetration(out, mask, ctx)['support'].max()) < 1e-5
    assert torch.allclose(out[0, :25, 1].sort().values[:5], torch.full((5,), .75), atol=1e-5)
    # tilted table
    tilted = context(tilt=(.2, 0., 0.))
    out = project_soft(verts, sunk, mask, tilted)
    assert float(vertex_penetration(out, mask, tilted)['support'].max()) < 1e-5
    # a box: vertices 1 cm below its top face come out through the top
    state, props, rmask = boxes(pos=(0., .80, 0.), size=(.3, .1, .3))
    inside = verts.clone(); inside[..., 1] += .09                                  # cube bottom at .84 = box top .85 - 1 cm
    assert float(vertex_penetration(inside, mask, ctx, state, props, rmask)['objects'].max()) == pytest.approx(.01, abs=1e-5)
    out = project_soft(inside, inside, mask, ctx, state, props, rmask)
    assert float(vertex_penetration(out, mask, ctx, state, props, rmask)['objects'].max()) < 1e-5
    assert float(out[0, mask[0], 1].min()) == pytest.approx(.85, abs=1e-5)
    # rotated box
    state[:, 0, 3:7] = euler_quaternion(torch.tensor([[.3, .4, -.2]]))
    assert float(vertex_penetration(inside, mask, ctx, state, props, rmask)['objects'].max()) > .005
    out = project_soft(inside, inside, mask, ctx, state, props, rmask, iters=3)
    assert float(vertex_penetration(out, mask, ctx, state, props, rmask)['objects'].max()) < 1e-5
    # floor, off the table
    low = verts.clone(); low[..., 0] += 2.; low[..., 1] -= .82
    out = project_soft(low, low, mask, ctx)
    assert float(out[0, mask[0], 1].min()) == pytest.approx(0., abs=1e-6)
    # tunnelling: above the table one step ago, below the slab now
    through = verts.clone(); through[..., 1] -= .25
    out = project_soft(verts, through, mask, ctx)
    assert float(out[0, mask[0], 1].min()) >= .75-1e-5
    # non-finite predictions fall back to the previous frame
    bad = verts.clone(); bad[0, 3] = float('nan')
    assert torch.isfinite(project_soft(verts, bad, mask, ctx)).all()


def test_strain_limiting():
    verts, mask, edges, rest = cube()
    ctx = context(enabled=0.)
    stretched = verts.clone(); stretched[..., 1] = (stretched[..., 1]-.75)*3+.75
    before = edge_strain(stretched, edges, rest, mask)[0].max()
    after = edge_strain(project_soft(stretched, stretched, mask, ctx, edges=edges, rest_len=rest, iters=20, max_strain=.3),
                        edges, rest, mask)[0].max()
    assert float(before) > 1.5 and float(after) < .6
    same = project_soft(verts, verts, mask, ctx, edges=edges, rest_len=rest, max_strain=.3, max_compress=.3)
    assert torch.allclose(same, verts, atol=1e-6)


def test_losses_and_gradient_flow():
    torch.manual_seed(0)
    m = model().train()
    verts, prev, mask, edges, rest, soft, ctx, state, props, rmask = random_batch()
    true = verts+(verts-prev)+torch.randn_like(verts)*.003
    raw = m.predict_raw(verts, prev, mask, edges, rest, soft, ctx, state, props, rmask)
    pred = m.integrate(verts, prev, raw*m.out_std+m.out_mean, ctx, mask)
    second = m(pred, verts, mask, edges, rest, soft, ctx, state, props, rmask)       # two-step unroll
    terms = soft_losses(second, true, mask, edges, rest, ctx, state, props, rmask)
    terms['acceleration'] = acceleration_loss(m, raw, verts, prev, true, ctx, mask)
    assert set(terms) == {'position', 'strain', 'penetration', 'acceleration'}
    assert all(t.dim() == 0 and torch.isfinite(t) for t in terms.values())
    sum(terms.values()).backward()
    missing = [name for name, p in m.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all() or p.grad.abs().sum() == 0]
    assert not missing, missing
    # a perfect prediction costs nothing
    perfect = soft_losses(true, true, mask, edges, rest, ctx, state, props, rmask)
    assert all(float(t) == 0 for t in perfect.values())
    target = m.normalize(m.raw_targets(verts, prev, true, ctx, mask))
    assert float(acceleration_loss(m, target, verts, prev, true, ctx, mask)) < 1e-10
    # the penetration term pulls a sunk vertex up
    cube_verts, cube_mask, cube_edges, cube_rest = cube()
    sunk = (cube_verts-torch.tensor([0., .01, 0.])).requires_grad_()
    sg.penetration_loss(sunk, cube_verts, cube_mask, context()).backward()
    assert float(sunk.grad[0, 0, 1]) < 0 and float(sunk.grad[0, 5, 1]) == 0          # bottom-layer vertex vs second layer


def test_fit_normalizer_and_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(0)
    m = SoftGNN(width=32, rounds=5, box_radius=.1)
    episodes = torch.cumsum(torch.randn(6, 21, V, 3)*.01, 1)+torch.tensor([0., 1., 0.])
    mask = torch.ones(6, V, dtype=torch.bool); mask[:, 100:] = False
    episodes[:, :, 100:] = 0
    ctx = context(6)
    stats = m.fit_normalizer(episodes.numpy(), mask.numpy(), ctx.numpy(), torch.rand(6, 6).numpy())
    acc = (episodes[:, 2:]-2*episodes[:, 1:-1]+episodes[:, :-2])/sg.STEP**2
    acc[..., 1] += 9.81
    assert torch.allclose(stats['mean'], acc[:, :, :100].mean((0, 1, 2)), atol=1e-3)
    assert torch.allclose(m.out_std, acc[:, :, :100].std((0, 1, 2)), rtol=1e-2)
    assert float(m.vel_scale) == pytest.approx(.2, rel=.1) and m.props_std.min() > 0
    path = tmp_path/'soft.pt'
    torch.save({'kind': m.kind, 'config': m.config, 'state_dict': m.state_dict()}, path)
    loaded = build_from_checkpoint(torch.load(path))
    assert loaded.config == m.config == dict(width=32, rounds=5, box_radius=.1, centre=True) and not loaded.training
    verts, prev, vmask, edges, rest, soft, ctx, state, props, rmask = random_batch()
    assert torch.equal(loaded(verts, prev, vmask, edges, rest, soft, ctx, state, props, rmask),
                       m.eval()(verts, prev, vmask, edges, rest, soft, ctx, state, props, rmask))
    with pytest.raises(ValueError): build_from_checkpoint({'kind': 'contact_gnn', 'state_dict': {}})


def test_autocast_trunk_keeps_float32_geometry():
    m = model()
    verts, prev, mask, edges, rest, soft, ctx, state, props, rmask = random_batch()
    with torch.no_grad(): reference = m(verts, prev, mask, edges, rest, soft, ctx, state, props, rmask)
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        mixed = m(verts, prev, mask, edges, rest, soft, ctx, state, props, rmask)
    assert mixed.dtype == torch.float32 and torch.isfinite(mixed).all()
    assert float((mixed-reference).abs().max()) < 5e-3                             # bf16 trunk noise only


def test_rollout_60_steps_never_nan():
    m = model()
    verts, prev, mask, edges, rest, soft, ctx, state, props, rmask = random_batch()
    with torch.no_grad(): m.decoder[-1].weight.mul_(20)                            # violent, untrained outputs
    states = state[:, None].repeat(1, 61, 1, 1)
    states[:, :, :, 1] += torch.linspace(0, .2, 61)[None, :, None]                 # prescribed moving boxes
    for project in (True, False):
        path = rollout(m, verts, prev, 60, mask, edges, rest, soft, ctx, states, props, rmask, project=project)
        assert path.shape == (3, 61, V, 3) and torch.isfinite(path).all()
    path = rollout(m, verts, prev, 60, mask, edges, rest, soft, ctx, states, props, rmask, max_strain=.5, margin=.002)
    assert torch.isfinite(path).all()
    assert float(path[:, 1:, :, 1][mask[:, None].expand(-1, 60, -1)].min()) >= .002-1e-6       # never under the floor
    depth = vertex_penetration(path[:, -1], mask, ctx, states[:, -1], props, rmask)
    assert float(depth['support'].max()) < 1e-4 and float(depth['floor'].max()) == 0
    wild = rollout(m, verts*float('inf'), prev, 5, mask, edges, rest, soft, ctx)
    assert wild.shape == (3, 6, V, 3)


def test_speed_v128_m768_cpu():
    m = model()
    verts, prev, mask, edges, rest, soft, ctx, state, props, rmask = random_batch(batch=1)
    mask[:] = True; edges = torch.randint(0, V, (1, M, 2)); edges[0, :, 1] = (edges[0, :, 0]+1+torch.randint(0, V-1, (M,))) % V
    args = (verts, prev, mask, edges, rest, soft, ctx, state, props, rmask)
    with torch.no_grad():
        for _ in range(3): project_soft(verts, m(*args), mask, ctx, state, props, rmask, edges, rest)
        start = time.perf_counter()
        for _ in range(20): nxt = m(*args)
        forward = (time.perf_counter()-start)/20*1000
        start = time.perf_counter()
        for _ in range(20): project_soft(verts, nxt, mask, ctx, state, props, rmask, edges, rest, max_strain=.5)
        projection = (time.perf_counter()-start)/20*1000
    parameters = sum(p.numel() for p in m.parameters())
    print(f'\nSoftGNN width={m.width} rounds={m.rounds} params={parameters} V={V} M={M} N=4 threads={torch.get_num_threads()}: '
          f'forward {forward:.2f} ms/step, project_soft {projection:.2f} ms/step')
    assert math.isfinite(forward) and forward < 500


# ---------------------------------------------------------------- pinned vertices, sphere obstacles, centre node, sleep, rest loss
def test_pinned_vertices_follow_their_prescribed_positions():
    m = model()
    verts, prev, mask, edges, rest, soft, ctx, state, props, rmask = random_batch()
    pinned = torch.zeros_like(mask); pinned[:, :3] = True
    target = verts+torch.tensor([.01, .02, -.03])
    free = m(verts, prev, mask, edges, rest, soft, ctx, state, props, rmask)
    held = m(verts, prev, mask, edges, rest, soft, ctx, state, props, rmask, pinned=pinned)
    moved = m(verts, prev, mask, edges, rest, soft, ctx, state, props, rmask, pinned=pinned, pinned_next=target)
    assert torch.equal(held[:, :3], verts[:, :3]) and torch.equal(moved[:, :3], target[:, :3])
    assert not torch.allclose(free[:, :3], verts[:, :3])
    assert not torch.allclose(held[:, 3:], free[:, 3:])                                # the flag is a model input
    none = m(verts, prev, mask, edges, rest, soft, ctx, state, props, rmask, pinned=torch.zeros_like(mask))
    assert torch.equal(none, free)                                                    # all-False pinned == no pinned
    # projection never moves a pinned vertex, the rollout keeps it on its trajectory
    below = verts.clone(); below[:, :3, 1] = -.05
    fixed = project_soft(verts, below, mask, ctx, pinned=pinned)
    assert torch.equal(fixed[:, :3], below[:, :3]) and float(fixed[:, 3:, 1].min()) >= 0
    path = verts[:, None].repeat(1, 6, 1, 1)+torch.arange(6)[None, :, None, None]*.01
    traj = rollout(m, verts, prev, 5, mask, edges, rest, soft, ctx, state, props, rmask, pinned=pinned, pinned_verts=path)
    assert torch.allclose(traj[:, 1:, :3], path[:, 1:, :3]) and torch.isfinite(traj).all()


def test_sphere_obstacle_is_a_rounded_box():
    state, props, rmask = boxes(pos=(0., 1., 0.), size=(.2, .2, .2))
    sphere = torch.cat((props, torch.zeros(1, 4, 1)), -1); sphere[:, 0, 9] = .1          # radius = size/2: a sphere
    ctx = context(enabled=0.)
    points = torch.tensor([[[.09, 1.09, .09], [.0, 1.05, .0], [.2, 1., 0.], [.0, 1.3, .0]]])   # box corner region, inside, outside x2
    mask = torch.ones(1, 4, dtype=torch.bool)
    sdf = sg._box_point_sdf(points, state, sphere)[0][0, :, 0]
    expect = (points[0]-torch.tensor([0., 1., 0.])).norm(dim=-1)-.1
    assert torch.allclose(sdf, expect, atol=1e-6)
    assert float(vertex_penetration(points, mask, ctx, state, props, rmask)['objects'][0, 0]) > .005     # inside the sharp box ...
    assert float(vertex_penetration(points, mask, ctx, state, sphere, rmask)['objects'][0, 0]) == 0      # ... but outside the sphere
    out = project_soft(points, points, mask, ctx, state, sphere, rmask)
    radial = (out[0]-torch.tensor([0., 1., 0.])).norm(dim=-1)
    assert float(radial.min()) >= .1-1e-5 and torch.allclose(out[0, 2:], points[0, 2:])
    m = model()                                                                   # 9- and 10-column props agree for sharp boxes
    verts, prev, vmask, edges, rest, soft, c, st, pr, rm = random_batch()
    assert torch.equal(m(verts, prev, vmask, edges, rest, soft, c, st, pr, rm),
                       m(verts, prev, vmask, edges, rest, soft, c, st, torch.cat((pr, torch.zeros(3, 4, 1)), -1), rm))


def test_centre_node_gives_two_hop_reach_and_old_checkpoints_load():
    torch.manual_seed(0)
    verts, mask, edges, rest = cube()
    ctx = context(); soft = torch.tensor([[.3, .5, .1, .5, 0., .0017]])
    prev = verts.clone()
    far = prev.clone(); far[0, 0, 0] -= .02                                           # change the velocity of one corner vertex
    opposite = 124                                                                    # the opposite corner: 4 grid hops (diagonal edges) away
    for centre, reach in ((False, False), (True, True)):
        m = model(rounds=2, centre=centre)
        a = m.predict_raw(verts, prev, mask, edges, rest, soft, ctx); b = m.predict_raw(verts, far, mask, edges, rest, soft, ctx)
        assert bool((a[0, opposite]-b[0, opposite]).abs().max() > 1e-7) == reach
    old = model(centre=False)
    state = {k: (v[:, :-1] if k in ('vert_encoder.0.weight', 'box_encoder.0.weight') else v) for k, v in old.state_dict().items()}
    loaded = build_from_checkpoint({'kind': 'soft_gnn', 'config': dict(width=128, rounds=6, box_radius=.08), 'state_dict': state})
    assert not loaded.centre and loaded.vert_encoder[0].weight.shape[1] == sg.VERT_DIM
    assert float(loaded.vert_encoder[0].weight[:, -1].abs().max()) == 0
    assert torch.isfinite(loaded(verts, prev, mask, edges, rest, soft, ctx)).all()


def test_sleep_rule_holds_supported_slow_bodies_only():
    verts, mask, edges, rest = cube()                                                 # resting on the table top
    ctx = context()
    drift = verts+torch.tensor([.0008, 0., 0.])*mask[..., None]
    assert torch.equal(project_soft(verts, drift, mask, ctx, sleep_move=.0015), verts)          # slow + supported: held
    assert torch.allclose(project_soft(verts, drift, mask, ctx), drift)                         # rule off: drift stays
    fast = verts+torch.tensor([.004, 0., 0.])*mask[..., None]
    assert torch.allclose(project_soft(verts, fast, mask, ctx, sleep_move=.0015), fast)         # fast: awake
    air, air_next = verts+torch.tensor([0., .3, 0.])*mask[..., None], drift+torch.tensor([0., .3, 0.])*mask[..., None]
    assert torch.allclose(project_soft(air, air_next, mask, ctx, sleep_move=.0015), air_next)   # unsupported: never sleeps
    pinned = torch.zeros_like(mask); pinned[0, 0] = True
    assert torch.equal(project_soft(air, air_next, mask, ctx, sleep_move=.0015, pinned=pinned)[0, 1:], air[0, 1:])   # hanging from a pin counts as supported


def test_rest_loss_only_counts_resting_vertices():
    verts, mask, edges, rest = cube()
    true_cur, true_next = verts, verts.clone()
    true_next[0, :10] += torch.tensor([.02, 0., 0.])                                  # 10 vertices really move (0.4 m/s)
    pred = verts+torch.tensor([.001, 0., 0.])*mask[..., None]                          # the model drifts every vertex by 1 mm
    loss = sg.rest_loss(pred, true_cur, true_next, true_cur, mask)
    expect = (125-10)*(.001/(.005*sg.STEP)-.5)/125                                    # Smooth-L1 linear branch, x only
    assert float(loss) == pytest.approx(expect, rel=1e-4)
    assert float(sg.rest_loss(true_next, true_cur, true_next, true_cur, mask)) == 0


def test_round2_trainer_helpers_late_horizon_and_box_contact_frames():
    import sys
    from pathlib import Path
    from types import SimpleNamespace
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
    import train_soft_ddp as ts
    args = SimpleNamespace(start_horizon=0, late_horizon=8, late_from=.6)
    assert [ts.soft_horizon(s, 100, 4, 0., args) for s in (0, 10, 20, 50, 59, 60, 99)] == [1, 2, 4, 4, 4, 8, 8]
    assert ts.soft_horizon(0, 100, 4, .7, args) == 8                                  # wall-clock progress also triggers it
    args.late_horizon = 0
    assert ts.soft_horizon(99, 100, 4, 0., args) == 4                                 # off by default
    # box-contact frames: the cube sits beside a box in frame 1 only; the second episode has no box at all
    verts, mask, edges, rest = cube()
    far = verts+torch.tensor([0., .5, 0.])*mask[..., None]
    episode = torch.stack((far, verts, far), 1)                                       # [1,3,V,3]
    state, props, rmask = boxes(pos=(.17, .81, 0.), size=(.2, .12, .2))               # box face 1 cm from the cube face
    data = dict(verts=torch.cat((episode, episode)), states=state[:, None].repeat(2, 3, 1, 1), props=props.repeat(2, 1, 1),
                mask=torch.cat((rmask, torch.zeros_like(rmask))), context=context(2), vert_mask=mask.repeat(2, 1))
    near = ts.box_contact_frames(data, gap=.02)
    assert near.tolist() == [[False, True, False], [False, False, False]]
