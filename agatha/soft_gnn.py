"""GNS-style graph network for basic soft bodies ("Learning to Simulate", research prototype).

Nodes are the soft-body vertices plus one node per rigid box.  The support slab and the
floor are analytic: every vertex carries its signed distance / normal to them as node
features.  Edges are the rest-mesh edges (both directions; relative vector, length, rest
length, strain) plus proximity edges box -> vertex with the analytic box SDF, normal and
contact-point relative velocity (same box SDF helpers as ``contact_gnn``).  The network
outputs a per-vertex normalised acceleration (gravity removed) that is integrated with
semi-implicit Euler at STEP = 0.05 s.

A virtual CENTRE node per body (features: COM velocity / height / size) is connected to every
vertex in both directions, so global information crosses the body in two hops.  Vertices may be
PINNED (kinematic, e.g. cloth handles): the flag is a node feature and ``integrate`` / ``forward`` /
``rollout`` keep pinned vertices at their prescribed positions.  Rigid obstacles may be ROUNDED
boxes: an optional 10th rigid_props column is the rounding radius (a sphere is size = diameter,
radius = size/2); it is 0 for ordinary boxes.

v1 scope: rigid boxes are PRESCRIBED obstacles (teacher-forced from ground truth or from
another model); only the soft vertices are predicted and nothing pushes back on the boxes.
There are no soft-soft / self-collision edges.

Conventions: Y-up metres, STEP 0.05 s; rigid state/props/context exactly as in
``contact_gnn``.  The state of the soft body is two position frames (verts_t, verts_{t-1});
the velocity is their finite difference.  Everything is invariant to xz translation except
the distances to the support slab, which is centred at the origin.  Geometry runs in
float32 with autocast disabled; only the MLP trunk follows the caller's autocast setting.

Argument order used by every function here:
    verts [B,V,3], prev_verts [B,V,3], vert_mask bool [B,V], edges int [B,M,2] (-1 pad),
    rest_len [B,M], soft_props [B,6], context [B,10],
    rigid_state [B,N,13] | None, rigid_props [B,N,9|10] | None, rigid_mask bool [B,N] | None,
    pinned bool [B,V] | None (keyword; kinematic vertices)
"""
from typing import Any
import torch
from torch import nn
import torch.nn.functional as F
from .dynamics import rotation
from .contact_gnn import (STEP, PROPS_DIM, SUPPORT_THICKNESS, MAX_LIN, MAX_ANG, _mm, _mlp, _no_autocast, _pad_inputs,
                          _sdf_grad, _support_frame, alias_mask)

SOFT_PROPS_DIM = 6
VERT_DIM = 30                  # last column: pinned flag (appended so that older checkpoints can be zero-padded)
BOX_DIM = 27                   # last column: rounding radius
CENTRE_DIM = 17
STAR_EDGE_DIM = 12
RIGID_DIM = PROPS_DIM+1        # contact_gnn props + rounding radius (m)
MESH_EDGE_DIM = 12
CONTACT_EDGE_DIM = 21

# ------------------------------------------------------------------- geometry
def _rigid(batch: int, device, rigid_state, rigid_props, rigid_mask):
    """Float32 rigid tensors (zero boxes when absent) and the mask of boxes that act as obstacles."""
    if rigid_state is None or rigid_state.shape[1] == 0:
        return (torch.zeros(batch, 0, 13, device=device), torch.zeros(batch, 0, RIGID_DIM, device=device),
                torch.zeros(batch, 0, dtype=torch.bool, device=device))
    state = rigid_state.float(); props = rigid_props.float()
    if props.shape[-1] < RIGID_DIM:
        props = torch.cat((props, props.new_zeros(*props.shape[:-1], RIGID_DIM-props.shape[-1])), -1)
    mask = torch.ones(state.shape[:2], dtype=torch.bool, device=device) if rigid_mask is None else rigid_mask.bool()
    return state, props, mask

def _context(context: torch.Tensor) -> torch.Tensor:
    return _pad_inputs(context.new_zeros(context.shape[0], 1, PROPS_DIM), context)[1]

def _obstacles(state, props, mask, context) -> torch.Tensor:
    """Boxes a vertex can collide with: real slots that are not a duplicate of the support."""
    if state.shape[1] == 0: return mask
    return mask & ~alias_mask(props, context, state)

def _round_radius(props: torch.Tensor) -> torch.Tensor:
    """Rounding radius [B,N] of each rigid slot (0 = sharp box), at most the smallest half extent."""
    if props.shape[-1] <= PROPS_DIM: return torch.zeros_like(props[..., 0])
    return torch.minimum(props[..., PROPS_DIM].clamp_min(0), props[..., :3].amin(-1)/2)

def _free(vert_mask: torch.Tensor, pinned) -> torch.Tensor:
    """Real vertices that the model moves (not pinned)."""
    return vert_mask if pinned is None else vert_mask & ~pinned.bool()

def _box_point_sdf(points: torch.Tensor, state: torch.Tensor, props: torch.Tensor):
    """Signed distance [B,V,N] of points [B,V,3] to boxes, world normal [B,V,N,3], box-frame point [B,V,N,3]."""
    rot = rotation(state[..., 3:7]); half = props[..., :3]/2
    radius = _round_radius(props)
    local = _mm((points[:, :, None]-state[:, None, :, :3])[..., None, :], rot[:, None]).squeeze(-2)
    sdf, grad = _sdf_grad(local, (half-radius[..., None])[:, None])
    sdf = sdf-radius[:, None]
    normal = _mm(grad[..., None, :], rot.transpose(-1, -2)[:, None]).squeeze(-2)
    return sdf, normal, local

def _support_local(points: torch.Tensor, context: torch.Tensor):
    """Points in the support-slab frame, plus (rot, half extents [B,1,3], enabled [B])."""
    rot, centre, hw, hd, enabled = _support_frame(context)
    half = torch.stack((hw, torch.full_like(hw, SUPPORT_THICKNESS/2), hd), -1)[:, None]
    return _mm(points-centre[:, None], rot), rot, half, enabled

def _support_push(points: torch.Tensor, context: torch.Tensor, margin: float = 0., prev: torch.Tensor | None = None):
    """Depth [B,V] (>=0) of points inside the support slab and the world vector [B,V,3] that takes them out.

    Same rule as contact_gnn._support_corner_depth: a point inside the footprint, below the top and
    above the slab bottom leaves through the top, unless it is clearly below the surface (> 1 cm)
    and the sideways way out is shorter.  With ``prev`` a point that was above the top one step
    ago and is now anywhere below it (even under the slab: tunnelling) also leaves through the top.
    """
    local, rot, half, enabled = _support_local(points, context)
    top = SUPPORT_THICKNESS/2-local[..., 1]+margin
    side_x = half[..., 0]-local[..., 0].abs(); side_z = half[..., 2]-local[..., 2].abs()
    lateral = torch.minimum(side_x, side_z)
    inside = enabled[:, None] & (lateral > 0) & (top > 0) & (top < SUPPORT_THICKNESS+margin)
    side = inside & (lateral < top) & (top > .01+margin)
    if prev is not None:
        prev_local = _support_local(prev, context)[0]
        swept = enabled[:, None] & (lateral > 0) & (top > 0) & (SUPPORT_THICKNESS/2-prev_local[..., 1] <= .005)
        side = side & ~swept; inside = inside | swept
    depth = torch.where(side, lateral+margin, top)*inside
    along_x = side_x <= side_z
    sign = torch.where(local >= 0, 1., -1.)
    zero = torch.zeros_like(depth)
    push = torch.stack((torch.where(side & along_x, depth*sign[..., 0], zero), torch.where(side, zero, depth),
                        torch.where(side & ~along_x, depth*sign[..., 2], zero)), -1)
    return depth, _mm(push, rot.transpose(-1, -2))

def vertex_penetration(verts, vert_mask, context, rigid_state=None, rigid_props=None, rigid_mask=None) -> dict[str, torch.Tensor]:
    """Penetration depth (m, >=0) [B,V] of every vertex into the support slab, the floor and the rigid boxes."""
    with _no_autocast(verts):
        verts = verts.float(); context = _context(context)
        state, props, mask = _rigid(verts.shape[0], verts.device, rigid_state, rigid_props, rigid_mask)
        support = _support_push(verts, context)[0]*vert_mask
        floor = (-verts[..., 1]).clamp_min(0)*vert_mask
        if state.shape[1]:
            sdf = _box_point_sdf(verts, state, props)[0]
            depth = (-sdf).clamp_min(0)*_obstacles(state, props, mask, context)[:, None, :]
            objects = depth.max(-1).values*vert_mask
        else:
            objects = torch.zeros_like(floor)
        return dict(support=support, floor=floor, objects=objects)

def _mesh_edges(edges: torch.Tensor, rest_len: torch.Tensor, vert_mask: torch.Tensor):
    """Valid undirected edges as (batch index, vertex i, vertex j, rest length)."""
    edges = edges.long(); count = vert_mask.shape[1]
    ok = (edges >= 0).all(-1) & (edges < count).all(-1) & (edges[..., 0] != edges[..., 1]) & (rest_len > 1e-6)
    safe = edges.clamp(0, count-1)
    ok = ok & vert_mask.gather(1, safe[..., 0]) & vert_mask.gather(1, safe[..., 1])
    b, m = ok.nonzero(as_tuple=True)
    return b, edges[b, m, 0], edges[b, m, 1], rest_len[b, m].float()

def edge_strain(verts: torch.Tensor, edges: torch.Tensor, rest_len: torch.Tensor, vert_mask: torch.Tensor):
    """Strain (len/rest - 1) of every valid mesh edge, flat [K], with its batch index [K]."""
    with _no_autocast(verts):
        b, i, j, rest = _mesh_edges(edges, rest_len, vert_mask)
        verts = verts.float()
        length = ((verts[b, j]-verts[b, i]).square().sum(-1)+1e-12).sqrt()
        return length/rest-1, b

def prev_from_velocity(verts: torch.Tensor, vert_vel: torch.Tensor) -> torch.Tensor:
    """Previous-frame positions for the first step of an episode: verts - vert_vel*STEP."""
    return verts-vert_vel*STEP

# ---------------------------------------------------------------------- model
class SoftGNN(nn.Module):
    kind = 'soft_gnn'

    def __init__(self, width: int = 128, rounds: int = 6, box_radius: float = .08, centre: bool = True):
        super().__init__()
        self.width, self.rounds, self.box_radius, self.centre = width, rounds, box_radius, bool(centre)
        if self.centre:
            self.centre_encoder = _mlp(CENTRE_DIM, width, width)
            self.star_encoder = _mlp(STAR_EDGE_DIM, width, width)
        self.vert_encoder = _mlp(VERT_DIM, width, width)
        self.box_encoder = _mlp(BOX_DIM, width, width)
        self.mesh_encoder = _mlp(MESH_EDGE_DIM, width, width)
        self.contact_encoder = _mlp(CONTACT_EDGE_DIM, width, width)
        self.edge_updates = nn.ModuleList(_mlp(3*width, width, width) for _ in range(rounds))
        self.node_updates = nn.ModuleList(_mlp(2*width, width, width) for _ in range(rounds))
        self.decoder = _mlp(width, width, 3, norm=False)
        with torch.no_grad():
            self.decoder[-1].weight.mul_(.1); self.decoder[-1].bias.zero_()
        self.register_buffer('out_mean', torch.zeros(3))            # acceleration without gravity, m/s^2
        self.register_buffer('out_std', torch.ones(3))
        self.register_buffer('vel_scale', torch.ones(1))            # m/s
        self.register_buffer('props_mean', torch.zeros(SOFT_PROPS_DIM))
        self.register_buffer('props_std', torch.ones(SOFT_PROPS_DIM))

    @property
    def config(self) -> dict[str, Any]:
        return dict(width=self.width, rounds=self.rounds, box_radius=self.box_radius, centre=self.centre)

    def set_normalizer(self, mean, std, vel_scale=None, props_mean=None, props_std=None) -> None:
        """Acceleration (gravity removed) mean/std [3]; optional speed scale and soft_props mean/std [6]."""
        put = lambda buffer, value, low=None: buffer.copy_(
            (lambda t: t if low is None else t.clamp_min(low))(torch.as_tensor(value, dtype=torch.float32).reshape(buffer.shape)).to(buffer.device))
        put(self.out_mean, mean); put(self.out_std, std, 1e-4)
        if vel_scale is not None: put(self.vel_scale, vel_scale, 1e-3)
        if props_mean is not None: put(self.props_mean, props_mean)
        if props_std is not None: put(self.props_std, props_std, 1e-3)

    @torch.no_grad()
    def fit_normalizer(self, verts, vert_mask, context, soft_props=None) -> dict[str, torch.Tensor]:
        """Fit the buffers from episodes: verts [E,T+1,V,3], vert_mask [E,V], context [E,>=8], soft_props [E,6]."""
        verts = torch.as_tensor(verts, dtype=torch.float32); mask = torch.as_tensor(vert_mask).bool()
        context = _context(torch.as_tensor(context, dtype=torch.float32))
        acc = (verts[:, 2:]-2*verts[:, 1:-1]+verts[:, :-2])/STEP**2
        acc[..., 1] += context[:, 3, None, None]
        weight = mask[:, None, :, None].expand_as(acc).float(); count = weight.sum((0, 1, 2)).clamp_min(1)
        mean = (acc*weight).sum((0, 1, 2))/count
        std = (((acc-mean).square()*weight).sum((0, 1, 2))/count).sqrt()
        speed = ((verts[:, 1:]-verts[:, :-1])/STEP).square().sum(-1)
        vel_scale = ((speed*mask[:, None]).sum()/(mask.sum()*speed.shape[1]).clamp_min(1)/3).sqrt().clamp_min(.05)
        stats = dict(mean=mean, std=std, vel_scale=vel_scale)
        if soft_props is not None:
            soft_props = torch.as_tensor(soft_props, dtype=torch.float32)
            stats.update(props_mean=soft_props.mean(0), props_std=soft_props.std(0, unbiased=False))
        self.set_normalizer(**stats)
        return stats

    # ---- graph construction (float32, no autocast)
    def graph(self, verts, prev_verts, vert_mask, edges, rest_len, soft_props, context,
              rigid_state=None, rigid_props=None, rigid_mask=None, pinned=None):
        """Vertex features [B*V,VERT_DIM], box features [B*N,BOX_DIM], mesh edges, contact edges, centre
        features [B,CENTRE_DIM] and star edges (None, None without the centre node).

        Each edge set is (features, receiver, sender) with flat indices into the node array
        [B, V+N(+1)] (vertices first, then boxes, then the body's centre node).
        """
        verts, prev_verts = verts.float(), prev_verts.float(); context = _context(context)
        batch, count = verts.shape[:2]; device = verts.device
        state, props, rmask = _rigid(batch, device, rigid_state, rigid_props, rigid_mask)
        boxes = state.shape[1]; total = count+boxes+int(self.centre)
        pin = torch.zeros_like(vert_mask) if pinned is None else pinned.bool() & vert_mask
        scale = self.vel_scale
        vel = ((verts-prev_verts)/STEP).clamp(-MAX_LIN, MAX_LIN)
        # vertices: velocity, height, analytic support / floor distances, material
        local, rot, half, enabled = _support_local(verts, context)
        sdf, grad = _sdf_grad(local, half)
        normal = _mm(grad, rot.transpose(-1, -2))
        lateral = torch.minimum(half[..., 0]-local[..., 0].abs(), half[..., 2]-local[..., 2].abs())
        above = local[..., 1]-SUPPORT_THICKNESS/2
        on = enabled[:, None, None].float()
        height = verts[..., 1:2]
        soft = ((soft_props.float()-self.props_mean)/self.props_std).clamp(-10, 10)
        per_vertex = (soft_props[:, :1].float().clamp_min(1e-6)/vert_mask.sum(1, keepdim=True).clamp_min(1)).log()/3
        expand = lambda x: x[:, None].expand(-1, count, -1)
        vert = torch.cat((vel/scale, vel.norm(dim=-1, keepdim=True)/scale,
                          height.clamp(-2, 4), torch.tanh(height/.01), torch.tanh(height/.1),
                          sdf[..., None].clamp(-.1, .3)*5*on, torch.tanh(sdf[..., None]/.01)*on, torch.tanh(sdf[..., None]/.1)*on,
                          normal*on, (vel*normal).sum(-1, keepdim=True)/scale*on,
                          above[..., None].clamp(-.3, .3)*5*on, torch.tanh(above[..., None]/.01)*on,
                          lateral[..., None].clamp(-.3, .3)*5*on, torch.tanh(lateral[..., None]/.02)*on,
                          on.expand(-1, count, -1), expand(context[:, 8:10]), expand(context[:, 3:4])/10,
                          expand(soft), expand(per_vertex), pin[..., None].float()), -1)
        vert = vert*vert_mask[..., None]
        # mesh edges, both directions
        b, i, j, rest = _mesh_edges(edges, rest_len, vert_mask)
        b2, recv, send, rest2 = torch.cat((b, b)), torch.cat((i, j)), torch.cat((j, i)), torch.cat((rest, rest))[:, None]
        delta = verts[b2, send]-verts[b2, recv]; dvel = vel[b2, send]-vel[b2, recv]
        length = (delta.square().sum(-1, keepdim=True)+1e-12).sqrt()
        strain = length/rest2-1
        unit = delta/length
        mesh = torch.cat(((delta/rest2).clamp(-5, 5), strain.clamp(-1, 4), torch.tanh(strain*5), rest2*20, (length*20).clamp_max(10),
                          dvel/scale, ((dvel*unit).sum(-1, keepdim=True)*STEP/rest2).clamp(-5, 5),
                          ((dvel*unit).sum(-1, keepdim=True)/scale)), -1)
        mesh_edges = (mesh, b2*total+recv, b2*total+send)
        # rigid boxes: nodes and proximity edges box -> vertex
        if boxes:
            obstacle = _obstacles(state, props, rmask, context)
            bvel = state[..., 7:10].clamp(-MAX_LIN, MAX_LIN); bang = state[..., 10:13].clamp(-MAX_ANG, MAX_ANG)
            brot = rotation(state[..., 3:7])
            box = torch.cat((state[..., 1:2].clamp(-2, 4), brot.flatten(-2), bvel/scale, bang/10, props[..., :3].clamp_max(3),
                             props[..., 3:4].clamp(1e-3, 1e3).log()/3, props[..., 4:5], props[..., 6:7], props[..., 5:6],
                             (bvel.norm(dim=-1, keepdim=True)/scale), torch.ones_like(props[..., :1]),
                             context[:, None, 3:4].expand(-1, boxes, -1)/10, _round_radius(props)[..., None]*10), -1)*obstacle[..., None]
            bsdf, bnormal, blocal = _box_point_sdf(verts, state, props)
            arm = verts[:, :, None]-state[:, None, :, :3]
            surface = bvel[:, None]+torch.cross(bang[:, None].expand(-1, count, -1, -1), arm, dim=-1)
            slip = vel[:, :, None]-surface
            sweep = (STEP*slip.norm(dim=-1)).clamp_max(1.)
            near = vert_mask[:, :, None] & obstacle[:, None, :] & (bsdf < self.box_radius+sweep)
            cb, ci, cj = near.nonzero(as_tuple=True)
            d, n, s, a = bsdf[cb, ci, cj][:, None], bnormal[cb, ci, cj], slip[cb, ci, cj], arm[cb, ci, cj]
            along = (s*n).sum(-1, keepdim=True); tangent = s-along*n
            half_b = props[..., :3][cb, cj]/2
            contact = torch.cat((d.clamp(-.1, .3)*5, torch.tanh(d/.01), torch.tanh(d/.05), n, s/scale, along/scale, tangent/scale,
                                 a.clamp(-1, 1), (blocal[cb, ci, cj]/half_b.clamp_min(1e-3)).clamp(-4, 4),
                                 props[..., 4][cb, cj][:, None], props[..., 5][cb, cj][:, None]), -1)
            contact_edges = (contact, cb*total+ci, cb*total+count+cj)
        else:
            box = verts.new_zeros(batch, 0, BOX_DIM)
            empty = torch.zeros(0, dtype=torch.long, device=device)
            contact_edges = (verts.new_zeros(0, CONTACT_EDGE_DIM), empty, empty)
        centre = star_edges = None
        if self.centre:
            w = vert_mask[..., None].float(); n = w.sum(1).clamp_min(1)
            com = (verts*w).sum(1)/n; cvel = (vel*w).sum(1)/n
            rel = (verts-com[:, None])*w; rvel = (vel-cvel[:, None])*w
            dist = (rel.square().sum(-1, keepdim=True)+1e-12).sqrt()
            size = ((dist.square()*w).sum(1)/n).sqrt().clamp_min(1e-3)                       # rms radius [B,1]
            centre = torch.cat((cvel/scale, cvel.norm(dim=-1, keepdim=True)/scale, com[:, 1:2].clamp(-2, 4), torch.tanh(com[:, 1:2]/.1),
                                n/128, soft, per_vertex, context[:, 3:4]/10, (pin[..., None].float()*w).sum(1)/n, size*5), -1)
            bi, vi = vert_mask.nonzero(as_tuple=True)
            r, d, rv = rel[bi, vi], dist[bi, vi], rvel[bi, vi]
            common = torch.cat(((r*5).clamp(-5, 5), (d*5).clamp_max(5), (r/size[bi]).clamp(-5, 5), rv/scale, (rv*r/d).sum(-1, keepdim=True)/scale), -1)
            flag = torch.ones_like(d)
            star = torch.cat((torch.cat((common, flag), -1), torch.cat((common, -flag), -1)))
            cnode = bi*total+count+boxes; vnode = bi*total+vi
            star_edges = (star, torch.cat((vnode, cnode)), torch.cat((cnode, vnode)))
        return vert.reshape(batch*count, VERT_DIM), box.reshape(batch*boxes, BOX_DIM), mesh_edges, contact_edges, centre, star_edges

    # ---- network
    def predict_raw(self, verts, prev_verts, vert_mask, edges, rest_len, soft_props, context,
                    rigid_state=None, rigid_props=None, rigid_mask=None, pinned=None) -> torch.Tensor:
        """Normalised per-vertex acceleration [B,V,3] (float32)."""
        batch, count = verts.shape[:2]
        with _no_autocast(verts):
            vert, box, (mesh, mesh_recv, mesh_send), (contact, contact_recv, contact_send), centre, star_edges = self.graph(
                verts, prev_verts, vert_mask, edges, rest_len, soft_props, context, rigid_state, rigid_props, rigid_mask, pinned)
        h = self.vert_encoder(vert).reshape(batch, count, self.width)
        if box.shape[0]:
            h = torch.cat((h, self.box_encoder(box).reshape(batch, -1, self.width).to(h.dtype)), 1)
        if centre is not None:
            h = torch.cat((h, self.centre_encoder(centre)[:, None].to(h.dtype)), 1)
        total = h.shape[1]; h = h.reshape(batch*total, self.width)
        e = self.mesh_encoder(mesh); receiver, sender = mesh_recv, mesh_send
        if contact.shape[0]:
            e = torch.cat((e, self.contact_encoder(contact).to(e.dtype)))
            receiver = torch.cat((receiver, contact_recv)); sender = torch.cat((sender, contact_send))
        if star_edges is not None and star_edges[0].shape[0]:
            e = torch.cat((e, self.star_encoder(star_edges[0]).to(e.dtype)))
            receiver = torch.cat((receiver, star_edges[1])); sender = torch.cat((sender, star_edges[2]))
        for edge_update, node_update in zip(self.edge_updates, self.node_updates):
            e = e+edge_update(torch.cat((e, h[receiver], h[sender]), -1))
            incoming = torch.zeros_like(h).index_add_(0, receiver, e.to(h.dtype))
            h = h+node_update(torch.cat((h, incoming), -1))
        return self.decoder(h).float().reshape(batch, total, 3)[:, :count]

    # ---- integrator and its exact inverse (float32)
    @staticmethod
    def _gravity(context: torch.Tensor) -> torch.Tensor:
        g = torch.zeros(context.shape[0], 1, 3, device=context.device); g[:, 0, 1] = -context[:, 3].float()
        return g

    def integrate(self, verts, prev_verts, raw, context, vert_mask, clamp: bool = False, pinned=None, pinned_next=None) -> torch.Tensor:
        """Semi-implicit Euler: v' = v + (raw + g)*STEP, x' = x + v'*STEP.  ``raw`` is un-normalised (m/s^2).

        Pinned (kinematic) vertices are not integrated: they take ``pinned_next`` [B,V,3] (their
        prescribed next positions) or stay where they are when it is None.
        """
        with _no_autocast(verts):
            verts, prev_verts, raw = verts.float(), prev_verts.float(), raw.float()
            vel = (verts-prev_verts)/STEP+(raw+self._gravity(context))*STEP
            if clamp: vel = vel.clamp(-MAX_LIN, MAX_LIN)
            out = torch.where(vert_mask[..., None], verts+vel*STEP, verts)
            if pinned is not None:
                out = torch.where((pinned.bool() & vert_mask)[..., None], verts if pinned_next is None else pinned_next.float(), out)
            return out

    def raw_targets(self, verts, prev_verts, next_verts, context, vert_mask) -> torch.Tensor:
        """Un-normalised acceleration targets [B,V,3] of a true transition (zeros for padded vertices)."""
        with _no_autocast(verts):
            verts, prev_verts, next_verts = verts.float(), prev_verts.float(), next_verts.float()
            return ((next_verts-2*verts+prev_verts)/STEP**2-self._gravity(context))*vert_mask[..., None]

    def normalize(self, raw: torch.Tensor) -> torch.Tensor:
        return (raw.float()-self.out_mean)/self.out_std

    def forward(self, verts, prev_verts, vert_mask, edges, rest_len, soft_props, context,
                rigid_state=None, rigid_props=None, rigid_mask=None, pinned=None, pinned_next=None) -> torch.Tensor:
        """Next vertex positions [B,V,3]; the next call takes (result, verts)."""
        out = self.predict_raw(verts, prev_verts, vert_mask, edges, rest_len, soft_props, context, rigid_state, rigid_props, rigid_mask, pinned)
        with _no_autocast(verts):
            return self.integrate(verts, prev_verts, out.float()*self.out_std+self.out_mean, context, vert_mask, clamp=True,
                                  pinned=pinned, pinned_next=pinned_next)

# --------------------------------------------------------------------- losses
def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (value*mask).sum()/mask.sum().clamp_min(1)

def acceleration_loss(model: SoftGNN, pred_raw, verts, prev_verts, next_verts, context, vert_mask) -> torch.Tensor:
    """Smooth-L1 between the normalised network output (predict_raw) and the normalised true acceleration."""
    with _no_autocast(verts):
        target = model.normalize(model.raw_targets(verts, prev_verts, next_verts, context, vert_mask))
        loss = F.smooth_l1_loss(pred_raw.float(), target, reduction='none').sum(-1)
        return _masked_mean(loss, vert_mask)

def position_loss(pred_verts, true_verts, vert_mask, scale: float = .01) -> torch.Tensor:
    """Smooth-L1 position error in units of ``scale`` metres; use it on every step of an unroll."""
    with _no_autocast(pred_verts):
        loss = F.smooth_l1_loss((pred_verts.float()-true_verts.float())/scale, torch.zeros_like(pred_verts, dtype=torch.float32),
                                reduction='none').sum(-1)
        return _masked_mean(loss, vert_mask)

def rest_loss(pred_verts, cur_verts, true_verts, true_cur, vert_mask, speed: float = .01, scale: float = .005) -> torch.Tensor:
    """Rest penalty: where the TRUE vertex moves slower than ``speed`` m/s into the predicted frame, the
    predicted step displacement (pred - cur) must match the true one (true - true_cur); Smooth-L1 in
    units of ``scale`` m/s.  ``cur_verts`` is the model input the step started from.  On the first step
    of a noisy window pass the true current frame as ``cur_verts`` only if the input was clean; the
    trainer instead uses pred - true there (the model must land on the true resting position)."""
    with _no_autocast(pred_verts):
        true_step = true_verts.float()-true_cur.float()
        still = vert_mask & (true_step.norm(dim=-1) < speed*STEP)
        error = ((pred_verts.float()-cur_verts.float())-true_step)/(scale*STEP)
        loss = F.smooth_l1_loss(error, torch.zeros_like(error), reduction='none').sum(-1)
        return (loss*still).sum()/vert_mask.sum().clamp_min(1)

def strain_loss(pred_verts, true_verts, vert_mask, edges, rest_len, scale: float = .05) -> torch.Tensor:
    """Edge-strain regulariser: predicted edge strain should match the true strain (units of ``scale``)."""
    with _no_autocast(pred_verts):
        predicted = edge_strain(pred_verts, edges, rest_len, vert_mask)[0]
        with torch.no_grad(): true = edge_strain(true_verts, edges, rest_len, vert_mask)[0]
        if predicted.numel() == 0: return pred_verts.new_zeros((), dtype=torch.float32)
        return F.smooth_l1_loss((predicted-true)/scale, torch.zeros_like(predicted))

def penetration_loss(pred_verts, true_verts, vert_mask, context, rigid_state=None, rigid_props=None, rigid_mask=None) -> torch.Tensor:
    """Vertex penetration into support / floor / boxes beyond what the true state shows (1 mm slack, 5 mm unit)."""
    with _no_autocast(pred_verts):
        predicted = vertex_penetration(pred_verts, vert_mask, context, rigid_state, rigid_props, rigid_mask)
        with torch.no_grad(): allowed = vertex_penetration(true_verts, vert_mask, context, rigid_state, rigid_props, rigid_mask)
        loss = pred_verts.new_zeros((), dtype=torch.float32)
        for key in ('support', 'floor', 'objects'):
            excess = (predicted[key]-allowed[key]-.001).clamp_min(0)/.005
            loss = loss+_masked_mean(F.smooth_l1_loss(excess, torch.zeros_like(excess), reduction='none'), vert_mask)
        return loss

def soft_losses(pred_verts, true_verts, vert_mask, edges, rest_len, context,
                rigid_state=None, rigid_props=None, rigid_mask=None) -> dict[str, torch.Tensor]:
    """Scalar position / strain / penetration terms of one predicted step (rigid_* are the boxes at that step)."""
    return dict(position=position_loss(pred_verts, true_verts, vert_mask),
                strain=strain_loss(pred_verts, true_verts, vert_mask, edges, rest_len),
                penetration=penetration_loss(pred_verts, true_verts, vert_mask, context, rigid_state, rigid_props, rigid_mask))

# ----------------------------------------------------------------- projection
@torch.no_grad()
def project_soft(prev_verts, pred_verts, vert_mask, context, rigid_state=None, rigid_props=None, rigid_mask=None,
                 edges=None, rest_len=None, iters: int = 2, max_strain: float | None = None,
                 max_compress: float | None = None, margin: float = 0., pinned=None,
                 sleep_move: float | None = None, sleep_gap: float = .015) -> torch.Tensor:
    """Non-learned clean-up of a predicted step: push vertices out of the boxes, the support slab and the floor.

    ``rigid_*`` are the boxes at the predicted step.  With ``edges``/``rest_len`` and ``max_strain``
    (e.g. 0.5) edges stretched beyond rest*(1+max_strain) are pulled back first (``max_compress``
    likewise limits compression), Jacobi averaged.  ``margin`` keeps vertices that far outside
    (PyBullet soft bodies rest about one collision margin above a surface).  Velocity is the
    position difference to ``prev_verts``, so the projection also removes the inward velocity.
    Non-finite predictions fall back to ``prev_verts``.  Pinned vertices are never moved.
    ``sleep_move`` (m per step, e.g. 0.0015; None = off) is a NON-LEARNED sleep rule analogous to
    the box sleep: a body whose free vertices all moved less than that in this step and that is
    supported (a vertex within ``sleep_gap`` of the floor / support top / an obstacle, or pinned)
    is held at ``prev_verts``.  It is stateless, so a body wakes as soon as the model predicts a
    larger step (a box landing on it, an unsupported part starting to fall).
    """
    with _no_autocast(pred_verts):
        prev = prev_verts.float(); context = _context(context)
        verts = torch.where(torch.isfinite(pred_verts).all(-1, keepdim=True), pred_verts.float(), prev)
        step = (verts-prev).clamp(-MAX_LIN*STEP, MAX_LIN*STEP)
        keep = verts                                                          # pinned vertices keep exactly what they were given
        verts = prev+step
        fixed = None if pinned is None else (pinned.bool() & vert_mask)[..., None]
        if fixed is not None: verts = torch.where(fixed, keep, verts)
        batch, count = verts.shape[:2]
        state, props, mask = _rigid(batch, verts.device, rigid_state, rigid_props, rigid_mask)
        obstacle = _obstacles(state, props, mask, context)
        limit = edges is not None and rest_len is not None and (max_strain is not None or max_compress is not None)
        if limit: b, i, j, rest = _mesh_edges(edges, rest_len, vert_mask)
        for _ in range(max(iters, 1)):
            if limit and b.numel():
                delta = verts[b, j]-verts[b, i]
                length = (delta.square().sum(-1)+1e-12).sqrt()
                low = rest*(1-max_compress) if max_compress is not None else torch.zeros_like(rest)
                high = rest*(1+max_strain) if max_strain is not None else torch.full_like(rest, float('inf'))
                excess = length-torch.maximum(torch.minimum(length, high), low)
                move = (.5*excess/length)[:, None]*delta
                active = (excess != 0).float()
                shift = torch.zeros(batch*count, 3, device=verts.device).index_add_(0, b*count+i, move).index_add_(0, b*count+j, -move)
                hits = torch.zeros(batch*count, device=verts.device).index_add_(0, b*count+i, active).index_add_(0, b*count+j, active)
                verts = verts+(shift/hits.clamp_min(1)[:, None]).reshape(batch, count, 3)
            if state.shape[1]:
                sdf, normal, _ = _box_point_sdf(verts, state, props)
                depth = (margin-sdf).clamp_min(0)*obstacle[:, None, :]
                deepest = depth.argmax(-1, keepdim=True)                      # one box per pass: no double counting
                push = depth.gather(-1, deepest)*normal.gather(2, deepest[..., None].expand(-1, -1, -1, 3)).squeeze(2)
                verts = verts+push
            verts = verts+_support_push(verts, context, margin, prev)[1]
            verts = torch.cat((verts[..., :1], verts[..., 1:2].clamp_min(margin), verts[..., 2:]), -1)
            if fixed is not None: verts = torch.where(fixed, keep, verts)
        if sleep_move is not None:
            free = _free(vert_mask, pinned)
            moved = ((verts-prev).norm(dim=-1)*free).amax(1)
            gap = verts[..., 1].clone()                                           # floor
            local, _, half, enabled = _support_local(verts, context)
            over = enabled[:, None] & (torch.minimum(half[..., 0]-local[..., 0].abs(), half[..., 2]-local[..., 2].abs()) > 0)
            gap = torch.where(over, torch.minimum(gap, (local[..., 1]-SUPPORT_THICKNESS/2).abs()), gap)
            if state.shape[1]:
                sdf = _box_point_sdf(verts, state, props)[0]
                gap = torch.minimum(gap, torch.where(obstacle[:, None, :], sdf, torch.full_like(sdf, 1e3)).amin(-1))
            supported = ((gap < sleep_gap+margin) & vert_mask).any(1)
            if pinned is not None: supported = supported | (pinned.bool() & vert_mask).any(1)
            verts = torch.where((supported & (moved < sleep_move))[:, None, None], prev, verts)
            if fixed is not None: verts = torch.where(fixed, keep, verts)
        return torch.where(vert_mask[..., None], verts, pred_verts.float().nan_to_num(0., 0., 0.)).to(pred_verts.dtype)

@torch.no_grad()
def rollout(model: SoftGNN, verts, prev_verts, steps: int, vert_mask, edges, rest_len, soft_props, context,
            rigid_states=None, rigid_props=None, rigid_mask=None, project: bool = True, pinned=None, pinned_verts=None,
            **project_args) -> torch.Tensor:
    """Vertex trajectory [B,steps+1,V,3] starting at ``verts``.

    ``rigid_states`` is the PRESCRIBED box trajectory [B,>=steps+1,N,13] (or a fixed [B,N,13]); the
    model reads the boxes at step t and the projection uses them at step t+1.  ``pinned`` [B,V]
    vertices follow ``pinned_verts`` [B,>=steps+1,V,3] (their prescribed trajectory) or stay fixed.
    """
    fixed = rigid_states is None or rigid_states.dim() == 3
    at = lambda t: rigid_states if fixed else rigid_states[:, min(t, rigid_states.shape[1]-1)]
    frames = [verts.float()]; prev = prev_verts.float()
    for t in range(steps):
        current = frames[-1]
        target = None if pinned_verts is None else pinned_verts[:, min(t+1, pinned_verts.shape[1]-1)]
        nxt = model(current, prev, vert_mask, edges, rest_len, soft_props, context, at(t), rigid_props, rigid_mask,
                    pinned=pinned, pinned_next=target)
        if project:
            nxt = project_soft(current, nxt, vert_mask, context, at(t+1), rigid_props, rigid_mask, edges, rest_len, pinned=pinned, **project_args)
        nxt = torch.where(torch.isfinite(nxt).all(-1, keepdim=True), nxt, current)
        frames.append(nxt); prev = current
    return torch.stack(frames, 1)

# ------------------------------------------------------------------------- io
def build_from_checkpoint(ckpt: dict) -> SoftGNN:
    """SoftGNN in eval mode from {'kind','config','state_dict',...}."""
    if ckpt.get('kind') != SoftGNN.kind: raise ValueError(f"Not a soft_gnn checkpoint: {ckpt.get('kind')}")
    config = ckpt.get('config', {})
    state = dict(ckpt['state_dict'])
    centre = bool(config.get('centre', any(key.startswith('centre_encoder.') for key in state)))   # older checkpoints: no centre node
    model = SoftGNN(centre=centre, **{key: config[key] for key in ('width', 'rounds', 'box_radius') if key in config})
    return load_weights(model, state).eval()

def load_weights(model: SoftGNN, state: dict) -> SoftGNN:
    """load_state_dict that zero-pads encoder input columns added after the checkpoint was written
    (new input features are always appended, so an older checkpoint computes exactly what it did)."""
    state = dict(state); own = model.state_dict()
    for key, value in list(state.items()):
        if key.endswith('_encoder.0.weight') and key in own and value.shape[1] < own[key].shape[1] and value.shape[0] == own[key].shape[0]:
            state[key] = torch.cat((value, value.new_zeros(value.shape[0], own[key].shape[1]-value.shape[1])), 1)
    model.load_state_dict(state)
    return model
