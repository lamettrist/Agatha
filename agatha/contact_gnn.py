"""Contact-aware graph network for box dynamics, with explicit contact geometry.

Nodes are object slots plus a support node and a floor node (both plain static
boxes).  Edges carry analytic corner-in-box signed distances, contact normals and
contact-point relative velocities, so the network sees millimetre-scale gaps
directly.  ``project`` is a non-learned inference-time clean-up (penetration
removal + sleeping); ``physics_losses`` are the matching training penalties.

Conventions: Y-up metres, quaternion xyzw, state = [pos3, quat4, linvel3, angvel3(world)].
All geometry runs in float32 with autocast disabled; only the MLP trunk follows the
caller's autocast setting.
"""
from typing import Any
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from .dynamics import normalize_q, rotation, euler_quaternion

STEP = .05
PROPS_DIM = 9
CONTEXT_DIM = 10
SUPPORT_THICKNESS = .08
MAX_LIN, MAX_ANG = 20., 40.
NODE_DIM = 29
EDGE_DIM = 91

_SIGNS: dict[Any, torch.Tensor] = {}

def _signs(device) -> torch.Tensor:
    key = str(device)
    if key not in _SIGNS:
        _SIGNS[key] = torch.tensor([[x, y, z] for x in (-1., 1.) for y in (-1., 1.) for z in (-1., 1.)], device=device)
    return _SIGNS[key]

def _samples(device, count: int = 26) -> torch.Tensor:
    """Unit sample points of a box: 8 corners, then 6 face centres, then 12 edge midpoints."""
    key = (str(device), 'samples')
    if key not in _SIGNS:
        grid = [[x, y, z] for x in (-1., 0., 1.) for y in (-1., 0., 1.) for z in (-1., 0., 1.)]
        extra = sorted((g for g in grid if 0 < sum(map(abs, g)) < 3), key=lambda g: sum(map(abs, g)))
        _SIGNS[key] = torch.cat((_signs(device), torch.tensor(extra, device=device)))
    return _SIGNS[key][:count]

def _no_autocast(t: torch.Tensor):
    kind = t.device.type
    return torch.autocast(device_type=kind if kind in ('cuda', 'cpu') else 'cpu', enabled=False)

def _pad_inputs(props: torch.Tensor, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Accept the old 6-column props / 8-column context and append live defaults."""
    props, context = props.float(), context.float()
    if props.shape[-1] < PROPS_DIM:
        props = torch.cat((props, props.new_zeros(*props.shape[:-1], PROPS_DIM-props.shape[-1])), -1)
    if context.shape[-1] < CONTEXT_DIM:
        extra = context.new_tensor([.5, 0.])[CONTEXT_DIM-context.shape[-1]-2:] if context.shape[-1] >= 8 else None
        if extra is None: raise ValueError('context needs at least 8 columns')
        context = torch.cat((context, extra.expand(context.shape[0], -1)), -1)
    return props, context

def alias_mask(props: torch.Tensor, context: torch.Tensor, state: torch.Tensor | None = None) -> torch.Tensor:
    """Static slots that duplicate the support, so the support node stands in for them.

    physics.World tilts only its internal support body and leaves such a desk flat and colliding,
    so with a tilted, enabled support the desk is an ordinary static box (not an alias).  With
    ``state`` the top face must also sit at the support height (physics._support_alias_ids).
    """
    alias = (props[..., 5] > .5) & (props[..., 0] >= context[:, 0, None]*.95) & (props[..., 2] >= context[:, 1, None]*.95)
    coincident = (context[:, 4:7].abs().amax(-1) < 1e-4) | (context[:, 7] < .5)
    alias = alias & coincident[:, None]
    if state is not None:
        alias = alias & ((state[..., 1]+props[..., 1]/2-context[:, 2, None]).abs() < .03)
    return alias

def dynamic_mask(props: torch.Tensor, context: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """[B,N] slots the model actually moves: real, not static, not a support alias."""
    return mask & (props[..., 5] < .5)          # an alias is always static

# ---------------------------------------------------------------- quaternions
def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ax, ay, az, aw = a.unbind(-1); bx, by, bz, bw = b.unbind(-1)
    return torch.stack((aw*bx+ax*bw+ay*bz-az*by, aw*by-ax*bz+ay*bw+az*bx,
                        aw*bz+ax*by-ay*bx+az*bw, aw*bw-ax*bx-ay*by-az*bz), -1)

def quat_conj(q: torch.Tensor) -> torch.Tensor:
    return torch.cat((-q[..., :3], q[..., 3:]), -1)

def rotvec_to_quat(r: torch.Tensor) -> torch.Tensor:
    angle = (r.square().sum(-1, keepdim=True)+1e-20).sqrt()
    return torch.cat((r*.5*torch.sinc(angle/(2*torch.pi)), (angle/2).cos()), -1)

def quat_to_rotvec(q: torch.Tensor) -> torch.Tensor:
    q = torch.where(q[..., 3:] < 0, -q, q)
    vec, w = q[..., :3], q[..., 3:]
    norm = (vec.square().sum(-1, keepdim=True)+1e-20).sqrt()
    factor = torch.where(norm > 1e-6, 2*torch.atan2(norm, w)/norm.clamp_min(1e-6), 2/w.clamp_min(1e-6))
    return vec*factor

# ------------------------------------------------------------------- geometry
def _mm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Broadcasting product of tiny (<=3 wide) matrices [...,m,k] x [...,k,n], elementwise.

    torch.matmul lowers these to cuBLAS batched gemm over 1e5-1e6 3x3 matrices, which was
    about half of the training GPU time; a broadcast multiply + sum is several times faster.
    """
    return (a.unsqueeze(-1)*b.unsqueeze(-3)).sum(-2)

def box_corners(state: torch.Tensor, props: torch.Tensor) -> torch.Tensor:
    """World corners [B,N,8,3] of every slot's box."""
    with _no_autocast(state):
        state = state.float(); half = props[..., :3].float()/2
        local = _signs(state.device)*half[..., None, :]
        return state[..., None, :3]+_mm(local, rotation(state[..., 3:7]).transpose(-1, -2))

def _sdf(local: torch.Tensor, half: torch.Tensor) -> torch.Tensor:
    """Signed distance of points (box frame) to a box with the given half extents."""
    q = local.abs()-half
    return (q.clamp_min(0).square().sum(-1)+1e-12).sqrt()+q.max(-1).values.clamp_max(0)

def _sdf_grad(local: torch.Tensor, half: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """SDF and its (unit, box-frame) gradient; inside it is the nearest face normal."""
    q = local.abs()-half
    outside = q.clamp_min(0)
    norm = (outside.square().sum(-1, keepdim=True)+1e-12).sqrt()
    largest = q.max(-1, keepdim=True).values
    inner = F.one_hot(q.argmax(-1), 3).to(q.dtype)
    sign = torch.where(local >= 0, 1., -1.)
    grad = torch.where(largest > 0, outside/norm, inner)*sign
    return norm.squeeze(-1)+largest.squeeze(-1).clamp_max(0), grad

def _support_frame(context: torch.Tensor):
    """Support rotation [B,3,3], centre [B,3], half footprint (w/2, d/2) and enabled flag."""
    rot = rotation(euler_quaternion(context[:, 4:7]))
    centre = torch.zeros(context.shape[0], 3, device=context.device)
    centre[:, 1] = context[:, 2]-SUPPORT_THICKNESS/2
    return rot, centre, context[:, 0]/2, context[:, 1]/2, context[:, 7] > .5

def _support_corner_depth(corners: torch.Tensor, context: torch.Tensor):
    """Per-corner depth below the support top [B,N,8], split into top-exit and side-exit corners.

    A corner counts when it is inside the footprint, below the top surface and above the
    slab bottom (a box standing under the table does not penetrate it).  When the corner is clearly below the surface (> 1 cm) and the
    sideways way out is shorter, the sideways distance is the penetration (side contact).
    """
    rot, centre, hw, hd, enabled = _support_frame(context)
    local = _mm(corners-centre[:, None, None, :], rot[:, None])
    top = SUPPORT_THICKNESS/2-local[..., 1]
    lateral = torch.minimum(hw[:, None, None]-local[..., 0].abs(), hd[:, None, None]-local[..., 2].abs())
    inside = enabled[:, None, None] & (lateral > 0) & (top > 0) & (top < SUPPORT_THICKNESS)
    side = inside & (lateral < top) & (top > .01)
    depth = torch.where(side, lateral, top)*inside
    return depth, top, lateral, inside & ~side, rot

def _pair_corner_sdf(corners: torch.Tensor, pos: torch.Tensor, rot: torch.Tensor, half: torch.Tensor) -> torch.Tensor:
    """[B,a,b,K]: signed distance of every sample point of box a to box b."""
    local = _mm(corners[:, :, None]-pos[:, None, :, None, :], rot[:, None])
    return _sdf(local, half[:, None, :, None, :])

def penetration_depths(state: torch.Tensor, props: torch.Tensor, context: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
    """Max penetration depth (m, >=0) of each object into support, floor and other boxes."""
    with _no_autocast(state):
        props, context = _pad_inputs(props, context); state = state.float()
        valid = mask & ~alias_mask(props, context, state)
        corners = box_corners(state, props)
        support = _support_corner_depth(corners, context)[0].max(-1).values*valid
        floor = (-corners[..., 1]).clamp_min(0).max(-1).values*valid
        count = state.shape[1]
        if count > 1:
            # Only pairs whose bounding spheres overlap can interpenetrate: evaluate those sparsely.
            pos = state[..., :3]; rot = rotation(state[..., 3:7]); half = props[..., :3]/2
            movable = props[..., 5] < .5
            pair = valid[:, :, None] & valid[:, None, :] & (movable[:, :, None] | movable[:, None, :])
            pair = pair & ~torch.eye(count, dtype=torch.bool, device=state.device)
            with torch.no_grad():
                radius = half.norm(dim=-1)
                near = (pos[:, :, None]-pos[:, None]).norm(dim=-1) <= radius[:, :, None]+radius[:, None, :]
            b, i, j = (pair & near).nonzero(as_tuple=True)
            points = pos[b, i][:, None]+_mm(_samples(state.device)*half[b, i][:, None], rot[b, i].transpose(-1, -2))
            depth = (-_sdf(_mm(points-pos[b, j][:, None], rot[b, j]), half[b, j][:, None])).clamp_min(0).max(-1).values
            objects = floor.new_zeros(state.shape[0]*count)
            objects = objects.scatter_reduce(0, b*count+i, depth, 'amax').scatter_reduce(0, b*count+j, depth, 'amax')
            objects = objects.reshape(state.shape[0], count)
        else:
            objects = torch.zeros_like(floor)
        return dict(support=support, floor=floor, objects=objects)

# ---------------------------------------------------------------------- model
def _mlp(inputs: int, width: int, outputs: int, norm: bool = True) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Linear(inputs, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU(), nn.Linear(width, outputs)]
    if norm: layers.append(nn.LayerNorm(outputs))
    return nn.Sequential(*layers)

class ContactGNN(nn.Module):
    kind = 'contact_gnn'

    def __init__(self, width: int = 128, rounds: int = 4, edge_radius: float = .12):
        super().__init__()
        self.width, self.rounds, self.edge_radius = width, rounds, edge_radius
        self.node_encoder = _mlp(NODE_DIM, width, width)
        self.edge_encoder = _mlp(EDGE_DIM, width, width)
        self.edge_updates = nn.ModuleList(_mlp(3*width, width, width) for _ in range(rounds))
        self.node_updates = nn.ModuleList(_mlp(2*width, width, width) for _ in range(rounds))
        self.decoder = _mlp(width, width, 12, norm=False)
        with torch.no_grad():
            self.decoder[-1].weight.mul_(.1); self.decoder[-1].bias.zero_()
        self.register_buffer('out_mean', torch.zeros(12))
        self.register_buffer('out_std', torch.ones(12))

    @property
    def config(self) -> dict[str, Any]:
        return dict(width=self.width, rounds=self.rounds, edge_radius=self.edge_radius)

    def set_normalizer(self, mean, std) -> None:
        mean = torch.as_tensor(mean, dtype=torch.float32).reshape(12)
        std = torch.as_tensor(std, dtype=torch.float32).reshape(12)
        self.out_mean.copy_(mean.to(self.out_mean.device))
        self.out_std.copy_(std.clamp_min(1e-4).to(self.out_std.device))

    # ---- graph construction (float32, no autocast)
    def nodes(self, state, props, context, mask):
        """Append support and floor nodes: ([B,N+2,13], [B,N+2,9], valid[B,N+2])."""
        batch, device = state.shape[0], state.device
        extra_state = torch.zeros(batch, 2, 13, device=device); extra_state[:, :, 6] = 1
        extra_state[:, 0, 1] = context[:, 2]-SUPPORT_THICKNESS/2; extra_state[:, 1, 1] = -.05
        extra_state[:, 0, 3:7] = euler_quaternion(context[:, 4:7])
        extra_props = torch.zeros(batch, 2, PROPS_DIM, device=device)
        extra_props[:, 0, 0] = context[:, 0]; extra_props[:, 0, 1] = SUPPORT_THICKNESS; extra_props[:, 0, 2] = context[:, 1]
        extra_props[:, 1, :3] = torch.tensor([20., .1, 20.], device=device)
        extra_props[:, :, 3] = 1; extra_props[:, :, 5] = 1
        extra_props[:, 0, 4] = context[:, 8]; extra_props[:, 0, 6] = context[:, 9]; extra_props[:, 1, 4] = .5
        extra_mask = torch.stack((context[:, 7] > .5, torch.ones(batch, dtype=torch.bool, device=device)), -1)
        valid = mask & ~alias_mask(props, context, state)
        return torch.cat((state, extra_state), 1), torch.cat((props, extra_props), 1), torch.cat((valid, extra_mask), 1)

    def graph(self, state, props, context, mask):
        """Node features [B*M,NODE_DIM], edge features [K,EDGE_DIM], receiver and sender flat indices."""
        props, context = _pad_inputs(props, context); state = state.float()
        batch, count = state.shape[:2]; total = count+2; device = state.device
        all_state, all_props, valid = self.nodes(state, props, context, mask)
        pos, quat = all_state[..., :3], all_state[..., 3:7]
        vel = all_state[..., 7:10].clamp(-MAX_LIN, MAX_LIN); ang = all_state[..., 10:13].clamp(-MAX_ANG, MAX_ANG)
        rot = rotation(quat); half = all_props[..., :3]/2
        kind = torch.zeros(batch, total, 3, device=device)
        kind[:, :count, 0] = 1; kind[:, count, 1] = 1; kind[:, count+1, 2] = 1
        node = torch.cat((pos[..., 1:2].clamp(-2, 4), rot.flatten(-2), vel/2, ang/10, all_props[..., :3].clamp_max(3),
                          all_props[..., 3:4].clamp(1e-3, 1e3).log()/3, all_props[..., 4:5], all_props[..., 6:7],
                          all_props[..., 7:9]*50, all_props[..., 5:6], kind,
                          context[:, None, 3:4].expand(-1, total, -1)/10), -1)
        # Conservative gate: 6-axis SAT separation never exceeds the true box-box gap.
        dp = pos[:, None, :, :]-pos[:, :, None, :]                       # [B,i,j,3] sender minus receiver
        relative = _mm(rot.transpose(-1, -2)[:, :, None], rot[:, None])  # R_i^T R_j
        absolute = relative.abs()
        dp_i = _mm(dp[..., None, :], rot[:, :, None]).squeeze(-2)
        dp_j = _mm(dp[..., None, :], rot[:, None]).squeeze(-2)
        sep_i = dp_i.abs()-half[:, :, None]-_mm(absolute, half[:, None, :, :, None]).squeeze(-1)
        sep_j = dp_j.abs()-half[:, None]-_mm(absolute.transpose(-1, -2), half[:, :, None, :, None]).squeeze(-1)
        sat = torch.maximum(sep_i.max(-1).values, sep_j.max(-1).values)
        receiver = valid & (all_props[..., 5] < .5)
        # Swept gate: a pair that can close its gap within one step (falls reach ~0.2 m/step) gets its edge now.
        reach = ang.norm(dim=-1)*half.norm(dim=-1)*(all_props[..., 5] < .5)
        sweep = STEP*((vel[:, :, None]-vel[:, None]).norm(dim=-1)+reach[:, :, None]+reach[:, None]).clamp_max(10.)
        edge_valid = receiver[:, :, None] & valid[:, None, :] & (sat < self.edge_radius+sweep)
        edge_valid = edge_valid & ~torch.eye(total, dtype=torch.bool, device=device)
        b, i, j = edge_valid.nonzero(as_tuple=True)
        edge = self._edge_features(pos[b, i], rot[b, i], half[b, i], vel[b, i], ang[b, i],
                                   pos[b, j], rot[b, j], half[b, j], vel[b, j], ang[b, j], j == count+1, sat[b, i, j])
        return node.reshape(batch*total, NODE_DIM)*valid.reshape(-1, 1), edge, b*total+i, b*total+j

    @staticmethod
    def _edge_features(pi, ri, hi, vi, wi, pj, rj, hj, vj, wj, floor, sat):
        signs = _samples(pi.device, 14)                                        # corners + face centres
        ci = pi[:, None]+_mm(signs*hi[:, None], ri.transpose(-1, -2))   # receiver corners (world)
        cj = pj[:, None]+_mm(signs*hj[:, None], rj.transpose(-1, -2))
        sdf_ij, grad_ij = _sdf_grad(_mm(ci-pj[:, None], rj), hj[:, None])  # receiver corners in sender
        sdf_ji, grad_ji = _sdf_grad(_mm(cj-pi[:, None], ri), hi[:, None])  # sender corners in receiver
        normal_ij = _mm(grad_ij, rj.transpose(-1, -2))            # out of sender, toward receiver
        normal_ji = -_mm(grad_ji, ri.transpose(-1, -2))           # flipped to the same convention
        sdf = torch.cat((sdf_ij, sdf_ji), 1); points = torch.cat((ci, cj), 1); normals = torch.cat((normal_ij, normal_ji), 1)
        low = sdf.min(-1, keepdim=True).values
        weight = torch.softmax(-(sdf-low)/.002, -1)[..., None]              # soft deepest-corner contact
        point = (weight*points).sum(1); normal = (weight*normals).sum(1)
        normal = normal/(normal.norm(dim=-1, keepdim=True)+1e-6)
        arm_i, arm_j = point-pi, point-pj
        slip = (vi+torch.cross(wi, arm_i, dim=-1))-(vj+torch.cross(wj, arm_j, dim=-1))
        along = (slip*normal).sum(-1, keepdim=True); tangent = slip-along*normal
        dp = pj-pi
        dp = torch.where(floor[:, None], dp*dp.new_tensor([0., 1., 0.]), dp).clamp(-1, 1)   # floor is infinite in x/z
        to_i = lambda x: _mm(x[:, None], ri).squeeze(1)
        to_j = lambda x: _mm(x[:, None], rj).squeeze(1)
        clipped = sdf.clamp(-.1, .3)
        return torch.cat((dp, to_i(dp), to_j(dp), (vj-vi)/2,
                          clipped*5, torch.tanh(sdf/.01),
                          low.clamp(-.1, .3)*10, torch.tanh(low/.01),
                          sat[:, None].clamp(-.1, .3)*10, torch.tanh(sat[:, None]/.01),
                          normal, to_i(normal), arm_i.clamp(-1, 1), to_i(arm_i).clamp(-1, 1),
                          (along/2).clamp(-10, 10), (tangent/2).clamp(-10, 10), to_i(tangent/2).clamp(-10, 10)), -1)

    # ---- network
    def predict_raw(self, state, props, context, mask) -> torch.Tensor:
        """Normalized network output [B,N,12] (float32)."""
        batch, count = state.shape[:2]
        with _no_autocast(state):
            node, edge, receiver, sender = self.graph(state, props, context, mask)
        h = self.node_encoder(node); e = self.edge_encoder(edge)
        for edge_update, node_update in zip(self.edge_updates, self.node_updates):
            e = e+edge_update(torch.cat((e, h[receiver], h[sender]), -1))
            total = torch.zeros_like(h).index_add_(0, receiver, e.to(h.dtype))
            h = h+node_update(torch.cat((h, total), -1))
        return self.decoder(h).float().reshape(batch, count+2, 12)[:, :count]

    # ---- integrator and its exact inverse (float32)
    @staticmethod
    def _gravity(context: torch.Tensor) -> torch.Tensor:
        g = torch.zeros(context.shape[0], 1, 3, device=context.device); g[:, 0, 1] = -context[:, 3]
        return g

    def integrate(self, state, raw, props, context, mask, clamp: bool = False) -> torch.Tensor:
        """next_state from un-normalized outputs [dv, dw, dx, dtheta]; inverse of raw_targets."""
        with _no_autocast(state):
            props, context = _pad_inputs(props, context); state = state.float(); raw = raw.float()
            pos, quat, vel, ang = state[..., :3], normalize_q(state[..., 3:7]), state[..., 7:10], state[..., 10:13]
            next_vel = vel+self._gravity(context)*STEP+raw[..., :3]
            next_ang = ang+raw[..., 3:6]
            if clamp:
                next_vel = next_vel.clamp(-MAX_LIN, MAX_LIN); next_ang = next_ang.clamp(-MAX_ANG, MAX_ANG)
            next_pos = pos+(vel+next_vel)/2*STEP+raw[..., 6:9]
            base = quat_mul(rotvec_to_quat((ang+next_ang)/2*STEP), quat)
            next_quat = normalize_q(quat_mul(rotvec_to_quat(raw[..., 9:12]), base))
            next_quat = torch.where((next_quat*quat).sum(-1, keepdim=True) < 0, -next_quat, next_quat)
            output = torch.cat((next_pos, next_quat, next_vel, next_ang), -1)
            return torch.where(dynamic_mask(props, context, mask)[..., None], output, state)

    def raw_targets(self, state, next_state, props, context, mask) -> torch.Tensor:
        """Un-normalized targets [B,N,12] of a true transition (zeros for non-dynamic slots)."""
        with _no_autocast(state):
            props, context = _pad_inputs(props, context); state = state.float(); next_state = next_state.float()
            pos, quat, vel, ang = state[..., :3], normalize_q(state[..., 3:7]), state[..., 7:10], state[..., 10:13]
            next_vel, next_ang = next_state[..., 7:10], next_state[..., 10:13]
            dv = next_vel-(vel+self._gravity(context)*STEP)
            dw = next_ang-ang
            dx = next_state[..., :3]-(pos+(vel+next_vel)/2*STEP)
            base = quat_mul(rotvec_to_quat((ang+next_ang)/2*STEP), quat)
            dth = quat_to_rotvec(quat_mul(normalize_q(next_state[..., 3:7]), quat_conj(base)))
            return torch.cat((dv, dw, dx, dth), -1)*dynamic_mask(props, context, mask)[..., None]

    def forward(self, state, props, context, mask) -> torch.Tensor:
        out = self.predict_raw(state, props, context, mask)
        with _no_autocast(state):
            raw = out.float()*self.out_std+self.out_mean
            return self.integrate(state, raw, props, context, mask, clamp=True)

# --------------------------------------------------------------------- losses
def physics_losses(pred_next, true_next, prev_state, props, context, mask, rest_next) -> dict[str, torch.Tensor]:
    """Scalar penalties: penetration beyond the true state's, motion of resting objects."""
    with _no_autocast(pred_next):
        props, context = _pad_inputs(props, context)
        pred_next, true_next, prev_state = pred_next.float(), true_next.float(), prev_state.float()
        dynamic = dynamic_mask(props, context, mask)
        count = dynamic.sum().clamp_min(1)
        predicted = penetration_depths(pred_next, props, context, mask)
        with torch.no_grad(): allowed = penetration_depths(true_next, props, context, mask)
        penetration = pred_next.new_zeros(())
        for key in ('support', 'floor', 'objects'):
            excess = (predicted[key]-allowed[key]-.001).clamp_min(0)/.005
            penetration = penetration+(F.smooth_l1_loss(excess, torch.zeros_like(excess), reduction='none')*dynamic).sum()/count
        resting = dynamic & rest_next.bool()
        rest_count = resting.sum().clamp_min(1)
        zero = torch.zeros_like(pred_next[..., :3])
        rest = (F.smooth_l1_loss(pred_next[..., 7:10]/.05, zero, reduction='none').sum(-1)
                +F.smooth_l1_loss(pred_next[..., 10:13]/.25, zero, reduction='none').sum(-1))
        slide = F.smooth_l1_loss((pred_next[..., :3]-prev_state[..., :3])/STEP/.05, zero, reduction='none').sum(-1)
        turn = 2*quat_mul(normalize_q(pred_next[..., 3:7]), quat_conj(normalize_q(prev_state[..., 3:7])))[..., :3]
        slide = slide+.1*F.smooth_l1_loss(turn/STEP/.25, zero, reduction='none').sum(-1)
        return dict(penetration=penetration, rest=(rest*resting).sum()/rest_count, slide=(slide*resting).sum()/rest_count)

# ----------------------------------------------------------------- projection
def _combine(contributions: torch.Tensor) -> torch.Tensor:
    """Merge several corrections [B,N,K,3] per axis: largest push each way (no double counting)."""
    return contributions.clamp_min(0).max(2).values+contributions.clamp_max(0).min(2).values

@torch.no_grad()
def project(prev_state, pred_state, props, context, mask, iters: int = 4, sleep_lin: float = .03, sleep_ang: float = .25,
            slop: float = 0., vel_slop: float = .001) -> torch.Tensor:
    """Non-learned clean-up of a predicted state: remove penetrations, then put slow supported objects to sleep.

    Positions are corrected for any depth > slop.  Velocities are only edited for depths > vel_slop
    (Bullet's own resting states sit a fraction of a mm deep), and never gain energy: support/floor
    contacts get an inelastic rigid-body impulse at the contact patch (linear + angular), pair
    contacts lose at most the centre-of-mass closing speed.  Objects on a slope steeper than
    their friction angle are never put to sleep (they must be able to start sliding from rest).
    """
    with _no_autocast(pred_state):
        props, context = _pad_inputs(props, context)
        original = pred_state; prev = prev_state.float(); pred = pred_state.float()
        pred = torch.where(torch.isfinite(pred).all(-1, keepdim=True), pred, prev)
        batch, count = pred.shape[:2]; device = pred.device
        valid = mask & ~alias_mask(props, context, pred)
        dynamic = valid & (props[..., 5] < .5)
        inverse = dynamic/props[..., 3].clamp_min(1e-3)
        half = props[..., :3]/2
        pos, quat = pred[..., :3].clone(), normalize_q(pred[..., 3:7])
        vel, ang = pred[..., 7:10].clone(), pred[..., 10:13].clone()
        rot = rotation(quat)
        samples = torch.matmul(_samples(device)*half[..., None, :], rot.transpose(-1, -2))
        offsets = samples[:, :, :8]
        pair = valid[:, :, None] & valid[:, None, :] & (dynamic[:, :, None] | dynamic[:, None, :])
        pair = pair & ~torch.eye(count, dtype=torch.bool, device=device)
        order = torch.arange(count, device=device)
        first = order[:, None] < order[None, :]
        weight = inverse[:, :, None]+inverse[:, None, :]
        share_a = (inverse[:, :, None]/weight.clamp_min(1e-12))[..., None]
        share_b = (inverse[:, None, :]/weight.clamp_min(1e-12))[..., None]
        support_rot = _support_frame(context)[0]
        support_normal = support_rot[:, None, :, 1]
        enabled = (context[:, 7] > .5)[:, None, None]
        fixed = valid & ~dynamic
        grounded = fixed.clone()
        size2 = props[..., :3].square()
        inertia = 12/(size2.sum(-1, keepdim=True)-size2).clamp_min(1e-8)     # inverse box inertia per unit mass (box frame)
        up = pos.new_tensor([0., 1., 0.]).expand(batch, count, 3)

        def patch(depth):
            """Centre [B,N,3] (relative to the COM) of the corners within 2 mm of the deepest one."""
            near = ((depth > 0) & (depth >= depth.max(-1, keepdim=True).values-.002)).float()
            return (near[..., None]*offsets).sum(2)/near.sum(-1, keepdim=True).clamp_min(1)

        def halt(vel, ang, arm, normal, active):
            """Inelastic impulse at COM+arm against a fixed surface: zeroes the approach speed of the
            contact point, split between linear and angular velocity (never adds kinetic energy)."""
            approach = ((vel+torch.cross(ang, arm, dim=-1))*normal).sum(-1).clamp_max(0)
            lever = torch.cross(arm, normal.expand_as(arm), dim=-1)
            spin = _mm((_mm(lever[..., None, :], rot).squeeze(-2)*inertia)[..., None, :], rot.transpose(-1, -2)).squeeze(-2)
            impulse = (-approach/(1+(torch.cross(spin, arm, dim=-1)*normal).sum(-1).clamp_min(0))*active)[..., None]
            return vel+normal*impulse, ang+spin*impulse

        def ground(pos, vel, ang):
            """Lift out of support top / floor; returns which objects now rest on them."""
            corners = pos[:, :, None]+offsets
            _, top, lateral, top_exit, _ = _support_corner_depth(corners, context)
            depth = (top-slop).clamp_min(0)*(top_exit & (top < .06))
            lift = depth.max(-1).values*dynamic
            pos = pos+support_normal*lift[..., None]
            vel, ang = halt(vel, ang, patch(depth), support_normal, lift > vel_slop)
            low = (corners+(support_normal*lift[..., None])[:, :, None])[..., 1]
            depth = (-low-slop).clamp_min(0)
            sunk = depth.max(-1).values*dynamic
            pos = pos+sunk[..., None]*up
            vel, ang = halt(vel, ang, patch(depth), up, sunk > vel_slop)
            on_support = (enabled & (lateral > 0) & (top > -.003) & (top < .06)).any(-1)
            return pos, vel, ang, valid & (on_support | (low < .003).any(-1))

        for _ in range(iters):
            pos, vel, ang, resting = ground(pos, vel, ang)
            grounded = grounded | resting
            if count > 1:
                points = pos[:, :, None]+samples
                local = torch.matmul(points[:, :, None]-pos[:, None, :, None, :], rot[:, None])   # [B,a,b,26,3]
                excess = local.abs()-half[:, None, :, None, :]
                inside, corner = (-excess.max(-1).values).max(-1)
                depth = (inside-slop).clamp_min(0)*pair
                index = corner[..., None, None].expand(-1, -1, -1, 1, 3)
                deepest_excess = excess.gather(3, index).squeeze(3); deepest_local = local.gather(3, index).squeeze(3)
                face = F.one_hot(deepest_excess.argmax(-1), 3).float()*torch.where(deepest_local >= 0, 1., -1.)
                normal = torch.matmul(rot[:, None], face[..., None]).squeeze(-1)                # out of b, toward a
                # Shock propagation: a body resting (transitively) on the support/floor is not pushed down.
                near = pair & (inside > -.003)
                above = (near & (normal[..., 1] > .7)) | (near & (normal[..., 1] < -.7)).transpose(1, 2)   # a sits on b
                for _ in range(4): grounded = grounded | (above & grounded[:, None, :]).any(2)
                a_on_b = ((normal[..., 1] > .7) & grounded[:, None, :] & dynamic[:, :, None])[..., None]
                b_on_a = ((normal[..., 1] < -.7) & grounded[:, :, None] & dynamic[:, None, :])[..., None]
                part_a = torch.where(a_on_b, 1., torch.where(b_on_a, 0., share_a))
                part_b = torch.where(b_on_a, 1., torch.where(a_on_b, 0., share_b))
                other = depth.transpose(1, 2)
                chosen = (depth > 0) & ((depth > other) | ((depth == other) & first))
                push = normal*(depth*chosen)[..., None]
                pos = pos+_combine(torch.cat((push*part_a, -(push*part_b).transpose(1, 2)), 2))
                contact = torch.gather(points[:, :, None].expand(-1, -1, count, -1, -1), 3, index).squeeze(3)
                speed_a = vel[:, :, None]+torch.cross(ang[:, :, None].expand_as(contact), contact-pos[:, :, None], dim=-1)
                speed_b = vel[:, None, :]+torch.cross(ang[:, None, :].expand_as(contact), contact-pos[:, None, :], dim=-1)
                # Never remove more than the centres' own closing speed (a spinning box's corner is much faster).
                closing = torch.maximum(((speed_a-speed_b)*normal).sum(-1), ((vel[:, :, None]-vel[:, None, :])*normal).sum(-1))
                stop = normal*(closing.clamp_max(0)*(chosen & (depth > vel_slop)))[..., None]
                vel = vel+_combine(torch.cat((-stop*part_a, (stop*part_b).transpose(1, 2)), 2))
        pos, vel, ang, _ = ground(pos, vel, ang)
        # Sleep: slow + supported -> zero velocity; pose snaps back if it barely changed.
        corners = pos[:, :, None]+offsets
        _, top, lateral, _, _ = _support_corner_depth(corners, context)
        on_floor = (corners[..., 1].abs() < .003).any(-1)
        supported = (enabled & (lateral > 0) & (top.abs() < .003)).any(-1) | on_floor
        # Rest must not be a fixed point where physics says the object slides: on an enabled support
        # tilted beyond the friction angle (Bullet multiplies the two coefficients) nothing sleeps.
        slope = (1-support_normal[..., 1].square()).clamp_min(0).sqrt()/support_normal[..., 1].clamp_min(1e-3)   # [B,1] tan(tilt)
        slipping = enabled[:, :, 0] & (slope > .9*props[..., 4]*context[:, 8, None]) & ~on_floor
        if count > 1:
            near = (_pair_corner_sdf(pos[:, :, None]+samples, pos, rot, half).abs() < .003).any(-1) & pair
            supported = supported | near.any(2) | near.any(1)
        asleep = dynamic & supported & ~slipping & (vel.norm(dim=-1) < sleep_lin) & (ang.norm(dim=-1) < sleep_ang)
        vel = torch.where(asleep[..., None], torch.zeros_like(vel), vel)
        ang = torch.where(asleep[..., None], torch.zeros_like(ang), ang)
        previous_quat = normalize_q(prev[..., 3:7])
        still = asleep & ((pos-prev[..., :3]).norm(dim=-1) < .002) & ((quat*previous_quat).sum(-1).abs() > float(np.cos(np.radians(.5))))
        pos = torch.where(still[..., None], prev[..., :3], pos)
        quat = torch.where(still[..., None], previous_quat, quat)
        output = torch.nan_to_num(torch.cat((pos, quat, vel, ang), -1), nan=0., posinf=0., neginf=0.)
        return torch.where(dynamic[..., None], output.to(original.dtype), original)

# ------------------------------------------------------------------------- io
def build_from_checkpoint(ckpt: dict) -> ContactGNN:
    """ContactGNN in eval mode from {'kind','config','state_dict',...}."""
    if ckpt.get('kind') != ContactGNN.kind: raise ValueError(f"Not a contact_gnn checkpoint: {ckpt.get('kind')}")
    config = ckpt.get('config', {})
    model = ContactGNN(**{key: config[key] for key in ('width', 'rounds', 'edge_radius') if key in config})
    model.load_state_dict(ckpt['state_dict']); model.eval()
    return model

def scene_tensors(scene, states: np.ndarray, gravity: float, tilt: list[float]):
    """(state, props, context, mask) with batch 1 and live defaults for the unobservable columns."""
    states = np.asarray(states, np.float32).reshape(-1, 13)
    n = len(states); count = max(n, 1)
    state = np.zeros((1, count, 13), np.float32); state[:, :, 6] = 1; state[0, :n] = states
    props = np.zeros((1, count, PROPS_DIM), np.float32); props[:, :, 3] = 1
    for i, o in enumerate(scene.objects[:n]): props[0, i, :6] = [*o.size, o.mass_kg, o.friction, float(o.is_static)]
    w, d = scene.support.size
    context = np.array([[w, d, scene.support.height, gravity, *tilt, float(scene.support.collisions_enabled), .5, 0.]], np.float32)
    mask = np.zeros((1, count), bool); mask[0, :n] = True
    # same rule as alias_mask: a desk under a tilted, enabled support stays a real (flat) static box
    coincident = max((abs(float(t)) for t in tilt), default=0.) < 1e-4 or not scene.support.collisions_enabled
    level = np.abs(state[0, :, 1]+props[0, :, 1]/2-scene.support.height) < .03
    mask[0] &= ~((props[0, :, 5] > .5) & (props[0, :, 0] >= .95*w) & (props[0, :, 2] >= .95*d) & level & coincident)
    return torch.from_numpy(state), torch.from_numpy(props), torch.from_numpy(context), torch.from_numpy(mask)
