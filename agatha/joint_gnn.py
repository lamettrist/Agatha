"""Articulated rigid bodies (hinge / ball / prismatic / fixed joints) on top of ContactGNN.

JointGNN subclasses contact_gnn.ContactGNN: node features, contact edges, integrator, normaliser
and checkpoint layout are inherited unchanged.  It adds
  * joint edges (both directions) with explicit constraint geometry, and a virtual static world
    node for world-anchored joints,
  * no contact edge between two bodies joined by a joint (PyBullet parent/child semantics),
  * joint_coordinates / joint_violation / joint_losses, and project_joints, a non-learned
    position + velocity projection run after contact_gnn.project.
A scene without joints goes through exactly the ContactGNN computation.

Joint row (JOINT_DIM = 18 floats):
  [0] parent slot (-1 = world: anchor_parent and axis are then WORLD coordinates), [1] child slot,
  [2:6] one-hot type (hinge, ball, prismatic, fixed), [6:9] anchor in the parent body frame,
  [9:12] anchor in the child body frame, [12:15] unit axis in the parent body frame (zeros for
  ball / fixed), [15] lower, [16] upper (rad or m), [17] joint damping.

ZERO CONVENTION (the row carries no rest orientation): at joint coordinate 0 the child frame has
the SAME orientation as the parent frame (identity link/joint frame orientations in PyBullet's
createMultiBody; the world frame for parent -1).  Hence the axis has the same components in both
body frames, hinge angle = twist of q_parent^-1 q_child about the axis (PyBullet's sign), and a
fixed / prismatic joint keeps the two orientations equal.  Prismatic displacement is the anchor
gap along the (parent) axis, zero when the two anchors coincide.

Conventions as in contact_gnn: Y-up metres, quaternion xyzw, state = [pos3, quat4, linvel3,
angvel3 (world)]; all geometry in float32 with autocast disabled.
"""
import inspect
import warnings
from types import SimpleNamespace
from typing import Any
import torch
from torch import nn
import torch.nn.functional as F
from . import contact_gnn as cg
from .contact_gnn import (ContactGNN, MAX_LIN, MAX_ANG, _mlp, _no_autocast, _pad_inputs,
                          quat_mul, quat_conj, rotvec_to_quat, quat_to_rotvec)
from .dynamics import normalize_q, rotation

JOINT_DIM = 18
JOINT_TYPES = ('hinge', 'ball', 'prismatic', 'fixed')
JOINT_EDGE_DIM = 80                          # 73 legacy features + 7 gravity-torque features (see _joint_features)
LEGACY_JOINT_EDGE_DIM = 73
HINGE_FREE, SLIDE_FREE = 6.2, 19.          # (upper - lower) beyond this means "no limit"
LIMIT_SLOP = (.05, .005)                     # (rad, m) tolerance of the projection's limit clamp, see solve_joints
JOINT_CONTACT_SLOP = .002                    # m of penetration project_joints tolerates for bodies of an articulation

# ------------------------------------------------------------------- geometry
def _rot(matrix: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """R v for [...,3,3] and [...,3] (broadcast multiply, see contact_gnn._mm)."""
    return (matrix*v[..., None, :]).sum(-1)

def _rot_t(matrix: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """R^T v."""
    return (matrix*v[..., :, None]).sum(-2)

def _length(v: torch.Tensor) -> torch.Tensor:
    return (v.square().sum(-1)+1e-12).sqrt()

def _unit(v: torch.Tensor) -> torch.Tensor:
    return v/_length(v)[..., None]

def _indices(joints: torch.Tensor, joint_mask: torch.Tensor, count: int, body_mask: torch.Tensor | None = None):
    """parent, child (clamped to real slots), world flag and validity [B,J]."""
    parent = joints[..., 0].round().long(); child = joints[..., 1].round().long()
    world = parent < 0
    valid = joint_mask.bool() & (child >= 0) & (child < count) & (parent < count) & (parent != child)
    parent, child = parent.clamp(0, count-1), child.clamp(0, count-1)
    if body_mask is not None:
        valid = valid & body_mask.gather(1, child) & (world | body_mask.gather(1, parent))
    return parent, child, world & valid, valid

def jointed_pairs(joints: torch.Tensor, joint_mask: torch.Tensor, count: int) -> torch.Tensor:
    """[B,N,N] symmetric: the two bodies are joined by a joint (they do not collide)."""
    parent, child, world, valid = _indices(joints, joint_mask, count)
    on = (valid & ~world).reshape(-1)
    batch = torch.arange(joints.shape[0], device=joints.device)[:, None].expand_as(parent).reshape(-1)[on]
    p, c = parent.reshape(-1)[on], child.reshape(-1)[on]
    pairs = torch.zeros(joints.shape[0], count, count, dtype=torch.bool, device=joints.device)
    pairs[batch, p, c] = True; pairs[batch, c, p] = True
    return pairs

def _frames(state: torch.Tensor, joints: torch.Tensor, joint_mask: torch.Tensor, body_mask: torch.Tensor | None = None) -> SimpleNamespace:
    """All per-joint geometry [B,J,...] in world coordinates (differentiable in state)."""
    state, joints = state.float(), joints.float()
    parent, child, world, valid = _indices(joints, joint_mask, state.shape[1], body_mask)
    take = lambda index: state.gather(1, index[..., None].expand(-1, -1, 13))
    sp, sc = take(parent), take(child)
    identity = state.new_zeros(13); identity[6] = 1
    sp = torch.where(world[..., None], identity, sp)                 # the world: origin, identity, at rest
    hinge, ball, prismatic, fixed = (joints[..., 2:6] > .5).unbind(-1)
    axis = joints[..., 12:15]; axis = axis/(axis.norm(dim=-1, keepdim=True)+1e-9)
    qp, qc = normalize_q(torch.stack((sp[..., 3:7], sc[..., 3:7]))).unbind(0)
    rp, rc = rotation(torch.stack((qp, qc))).unbind(0)
    arm_p, arm_c = _rot(rp, joints[..., 6:9]), _rot(rc, joints[..., 9:12])
    gap = (sc[..., :3]+arm_c)-(sp[..., :3]+arm_p)                    # child anchor minus parent anchor
    axis_p, axis_c = _rot(rp, axis), _rot(rc, axis)
    relative = quat_mul(quat_conj(qp), qc)
    s, w = (relative[..., :3]*axis).sum(-1), relative[..., 3]
    angle = torch.atan2(2*s*w, w*w-s*s+1e-12)                        # twist about the axis, in (-pi, pi]
    slide = (gap*axis_p).sum(-1)
    coordinate = torch.where(hinge, angle, torch.where(prismatic, slide, torch.zeros_like(slide)))*valid
    violation = gap-(prismatic*slide)[..., None]*axis_p              # the part of the gap the joint forbids
    lower, upper = joints[..., 15], joints[..., 16]
    span = upper-lower
    limited = (hinge & (span < HINGE_FREE)) | (prismatic & (span < SLIDE_FREE))
    return SimpleNamespace(parent=parent, child=child, world=world, valid=valid, hinge=hinge, ball=ball,
                           prismatic=prismatic, fixed=fixed, axis=axis, sp=sp, sc=sc, qp=qp, qc=qc, rp=rp, rc=rc,
                           arm_p=arm_p, arm_c=arm_c, gap=gap, axis_p=axis_p, axis_c=axis_c, relative=relative,
                           coordinate=coordinate, violation=violation, lower=lower, upper=upper, limited=limited,
                           local_p=joints[..., 6:9], local_c=joints[..., 9:12], damping=joints[..., 17])

def _violations(f: SimpleNamespace) -> dict[str, torch.Tensor]:
    excess = ((f.lower-f.coordinate).clamp_min(0)+(f.coordinate-f.upper).clamp_min(0))*(f.hinge | f.prismatic)
    swing = torch.atan2(_length(torch.cross(f.axis_p, f.axis_c, dim=-1)), (f.axis_p*f.axis_c).sum(-1))
    turn = 2*torch.atan2(_length(f.relative[..., :3]), f.relative[..., 3].abs())
    zero = torch.zeros_like(swing)
    off_axis = torch.where(f.hinge, swing, torch.where(f.prismatic | f.fixed, turn, zero))
    return dict(anchor_gap=_length(f.violation)*f.valid, limit_excess=excess*f.valid, off_axis=off_axis*f.valid)

def joint_coordinates(state: torch.Tensor, joints: torch.Tensor, joint_mask: torch.Tensor) -> torch.Tensor:
    """[B,J] hinge angle (rad, in (-pi, pi]) / prismatic displacement (m); 0 for ball, fixed and unused rows."""
    with _no_autocast(state):
        return _frames(state, joints, joint_mask).coordinate

def joint_violation(state: torch.Tensor, joints: torch.Tensor, joint_mask: torch.Tensor) -> dict[str, torch.Tensor]:
    """Constraint errors [B,J] (differentiable, 0 for unused rows).

    anchor_gap: metres between the two anchors (prismatic: only the part off the axis).
    limit_excess: rad / m outside [lower, upper].
    off_axis: rad between the two bodies' hinge axes; for prismatic and fixed joints the full
    relative rotation angle (both must keep the orientations equal); 0 for ball joints.
    """
    with _no_autocast(state):
        return _violations(_frames(state, joints, joint_mask))

def joint_losses(pred_next, true_next, joints, joint_mask) -> dict[str, torch.Tensor]:
    """Scalar penalties (mean over real joints) for violations beyond those of the true state:
    anchor (0.5 mm free, 5 mm scale), limit (0.02 rad / 5 mm scale), axis (0.002 rad free, 0.02 rad scale)."""
    with _no_autocast(pred_next):
        frames = _frames(pred_next, joints, joint_mask)
        predicted = _violations(frames)
        with torch.no_grad(): allowed = _violations(_frames(true_next, joints, joint_mask))
        count = frames.valid.sum().clamp_min(1)
        def penalty(key: str, free: float, scale):
            excess = (predicted[key]-allowed[key]-free).clamp_min(0)/scale
            return (F.smooth_l1_loss(excess, torch.zeros_like(excess), reduction='none')*frames.valid).sum()/count
        limit_scale = torch.where(frames.prismatic, .005, .02)
        return dict(anchor=penalty('anchor_gap', .0005, .005), limit=penalty('limit_excess', 0., limit_scale),
                    axis=penalty('off_axis', .002, .02))

def articulations(joints: torch.Tensor, joint_mask: torch.Tensor, count: int) -> torch.Tensor:
    """[B,N] long: id (smallest member slot) of the articulation a body belongs to, -1 for bodies without a joint.

    Bodies joined (transitively) by joints share an id; world anchors do not connect articulations.
    """
    parent, child, world, valid = _indices(joints, joint_mask, count)
    label = torch.arange(count, device=joints.device).expand(joints.shape[0], count).clone()
    link = valid & ~world
    big = torch.full_like(parent, count)
    for _ in range(max(joints.shape[1], 1)):
        low = torch.where(link, torch.minimum(label.gather(1, parent), label.gather(1, child)), big)
        new = label.scatter_reduce(1, parent, low, 'amin').scatter_reduce(1, child, low, 'amin')
        if torch.equal(new, label): break
        label = new
    member = torch.zeros(joints.shape[0], count+1, dtype=torch.bool, device=joints.device)
    member.scatter_(1, torch.where(valid, child, big), True); member.scatter_(1, torch.where(link, parent, big), True)
    return torch.where(member[:, :count], label, torch.full_like(label, -1))

def _relative_error(f_pred: SimpleNamespace, f_true: SimpleNamespace):
    """(twist error about the joint axis [B,J] rad wrapped, safe flag, full relative-rotation error [B,J] rad,
    error of the child position expressed in the parent frame [B,J] m) between a predicted and a true state."""
    error = quat_mul(quat_conj(f_true.relative), f_pred.relative)            # rel_true^-1 rel_pred
    s, w = (error[..., :3]*f_pred.axis).sum(-1), error[..., 3]
    safe = (s*s+w*w) > 1e-3                                                  # atan2 has no usable gradient at s = w = 0
    s, w = torch.where(safe, s, torch.zeros_like(s)), torch.where(safe, w, torch.ones_like(w))
    twist = torch.atan2(2*s*w, w*w-s*s)
    turn = 2*torch.atan2(_length(error[..., :3]), error[..., 3].abs())
    local = lambda f: _rot_t(f.rp, f.sc[..., :3]-f.sp[..., :3])
    return twist, safe, turn, _length(local(f_pred)-local(f_true))

def joint_tracking_losses(pred_next, true_next, joints, joint_mask, weight: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    """Scalar accuracy terms on the joint coordinates (the constraint terms are joint_losses).

    coord: Huber of the hinge-angle error (twist of the predicted against the true relative rotation about the
    joint axis, wrapped, scale 0.05 rad) and of the prismatic displacement error (scale 0.005 m), mean over
    hinge / prismatic joints.  relative: Huber of the child's position in the parent frame (scale 0.01 m) plus
    Huber of the relative orientation error (scale 0.1 rad), mean over all joints (parent = world: the child's
    absolute pose).  weight [B,J] (optional, >= 0) weights the joints, e.g. joints that move in the window.
    """
    with _no_autocast(pred_next):
        f = _frames(pred_next, joints, joint_mask)
        with torch.no_grad(): t = _frames(true_next, joints, joint_mask)
        twist, safe, turn, offset = _relative_error(f, t)
        w = f.valid.float() if weight is None else weight.float()*f.valid
        huber = lambda x: F.smooth_l1_loss(x, torch.zeros_like(x), reduction='none')
        axial = f.hinge | f.prismatic
        coord = torch.where(f.hinge, huber(twist/.05)*safe, huber((f.coordinate-t.coordinate)/.005)*f.prismatic)
        relative = huber(offset/.01)+huber(turn/.1)
        return dict(coord=(coord*w*axial).sum()/(w*axial).sum().clamp_min(1e-6), relative=(relative*w).sum()/w.sum().clamp_min(1e-6))

@torch.no_grad()
def joint_motion(states: torch.Tensor, joints: torch.Tensor, joint_mask: torch.Tensor, rot_tol: float = .01, slide_tol: float = .001) -> torch.Tensor:
    """[B,J] bool: the joint moves within a window of true states [B,F,N,13]: its relative rotation changes by more
    than rot_tol rad or (prismatic) its displacement by more than slide_tol m against the first frame."""
    with _no_autocast(states):
        batch, frames = states.shape[:2]
        spread = lambda x: x[:, None].expand(batch, frames, *x.shape[1:]).reshape(batch*frames, *x.shape[1:])
        f = _frames(states.reshape(batch*frames, *states.shape[2:]), spread(joints), spread(joint_mask))
        relative = f.relative.reshape(batch, frames, -1, 4); slide = (f.coordinate*f.prismatic).reshape(batch, frames, -1)
        error = quat_mul(quat_conj(relative[:, :1]).expand_as(relative), relative)
        turn = 2*torch.atan2(_length(error[..., :3]), error[..., 3].abs())
        return ((turn.amax(1) > rot_tol) | ((slide-slide[:, :1]).abs().amax(1) > slide_tol)) & joint_mask.bool()

# ---------------------------------------------------------------------- model
class JointGNN(ContactGNN):
    kind = 'joint_gnn'

    def __init__(self, width: int = 128, rounds: int = 6, edge_radius: float = .12, joint_features: int = JOINT_EDGE_DIM,
                 shared_rounds: int = 0):
        super().__init__(width, rounds, edge_radius)
        # shared_rounds = R > 0: rounds R.. only update scenes that contain a joint, so a joint-free scene is computed by
        # the first R rounds alone (with a 4-round contact_gnn warm start and R = 4: by exactly the box model's own
        # computation, whatever the extra rounds learn).  0 = every round updates every scene (previous behaviour).
        self.shared_rounds = int(shared_rounds)
        if joint_features not in (JOINT_EDGE_DIM, LEGACY_JOINT_EDGE_DIM): raise ValueError(f'joint_features must be {JOINT_EDGE_DIM} or {LEGACY_JOINT_EDGE_DIM}')
        self.joint_features = joint_features      # 73 = checkpoints from before the gravity-torque features
        self.joint_encoder = _mlp(joint_features, width, width)
        self.joint_updates = nn.ModuleList(_mlp(3*width, width, width) for _ in range(rounds))
        self.world_node = nn.Parameter(torch.zeros(width))
        with torch.no_grad():                     # joint messages start small next to the (pre-trained) contact messages
            for update in self.joint_updates: update[-1].weight.mul_(.1)

    @property
    def config(self) -> dict[str, Any]:
        return dict(super().config, joint_features=self.joint_features, shared_rounds=self.shared_rounds)

    # ---- joint edges (float32, no autocast)
    @staticmethod
    def _joint_features(f: SimpleNamespace, to_child: bool, gravity: torch.Tensor | None = None) -> torch.Tensor:
        """[B,J,JOINT_EDGE_DIM] features of the edge whose receiver is the child (or the parent).

        The last 7 (only with gravity [B] given; absent in legacy 73-feature checkpoints) are the torque of gravity
        on the receiver about its own anchor per unit mass, (anchor -> COM) x g / 9.81 scaled by 5: world frame,
        receiver frame, and its component about the hinge axis (what makes an open lid fall shut)."""
        clamp_v = lambda s: torch.cat((s[..., 7:10].clamp(-MAX_LIN, MAX_LIN), s[..., 10:13].clamp(-MAX_ANG, MAX_ANG)), -1)
        mp, mc = clamp_v(f.sp), clamp_v(f.sc)
        # velocity of each body at the child's anchor point (on a prismatic joint the two anchors are apart)
        point_p = mp[..., :3]+torch.cross(mp[..., 3:], f.arm_p+f.gap, dim=-1)*~f.world[..., None]
        point_c = mc[..., :3]+torch.cross(mc[..., 3:], f.arm_c, dim=-1)
        other_arm_p = f.arm_p*~f.world[..., None]                            # the world has no lever arm
        if to_child:
            mine, sign, arm, other_arm, local, other = f.rc, 1., f.arm_c, other_arm_p, f.local_c, f.rp
        else:
            mine, sign, arm, other_arm, local, other = f.rp, -1., f.arm_p, f.arm_c, f.local_p, f.rc
        own = lambda x: _rot_t(mine, x)
        gap, violation = -sign*f.gap, -sign*f.violation                     # other anchor minus own anchor
        slip = -sign*(point_c-point_p)                                      # other minus own, at the anchor
        spin = -sign*(mc[..., 3:]-mp[..., 3:])
        along = (spin*f.axis_p).sum(-1, keepdim=True)
        off = spin-along*f.axis_p
        q = f.coordinate[..., None]
        hinge, prismatic = f.hinge[..., None].float(), f.prismatic[..., None].float()
        has = hinge+prismatic
        to_lower, to_upper = q-f.lower[..., None], f.upper[..., None]-q
        scale = hinge/torch.pi+prismatic*5; sharp = hinge/.05+prismatic/.01
        flags = torch.cat((hinge, f.ball[..., None].float(), prismatic, f.fixed[..., None].float(),
                           torch.full_like(q, float(to_child)), f.world[..., None].float()), -1)
        extra = ()
        if gravity is not None:
            up = arm.new_tensor([0., 1., 0.])
            torque = torch.cross(arm, up.expand_as(arm), dim=-1)*(gravity.float()[:, None, None]/9.81*5)
            extra = (torque.clamp(-5, 5), own(torque).clamp(-5, 5), ((torque*f.axis_p).sum(-1, keepdim=True)*hinge).clamp(-5, 5))
        return torch.cat((
            flags,
            arm.clamp(-1, 1), local.clamp(-1, 1), other_arm.clamp(-1, 1), own(other_arm).clamp(-1, 1),
            (gap*20).clamp(-10, 10), own(gap*20).clamp(-10, 10), torch.tanh(gap/.01),
            (violation*20).clamp(-10, 10), own(violation*20).clamp(-10, 10), torch.tanh(_length(violation)[..., None]/.01),
            f.axis_p, own(f.axis_p), f.axis, _rot_t(other, f.axis_p), torch.cross(f.axis_p, f.axis_c, dim=-1)*10,
            q*hinge/torch.pi, q.sin()*hinge, q.cos()*hinge, (q*5).clamp(-5, 5)*prismatic,
            f.limited[..., None].float(), (to_lower*scale).clamp(-2, 4)*has, (to_upper*scale).clamp(-2, 4)*has,
            torch.tanh(to_lower*sharp)*has, torch.tanh(to_upper*sharp)*has,
            (slip/2).clamp(-10, 10), own(slip/2).clamp(-10, 10), ((slip*f.axis_p).sum(-1, keepdim=True)/2).clamp(-10, 10),
            (along/10).clamp(-10, 10), (off/10).clamp(-10, 10), own(off/10).clamp(-10, 10),
            f.damping[..., None].clamp(0, 5), *extra), -1)

    def joint_graph(self, state, props, context, mask, joints, joint_mask):
        """Joint edge features [K,JOINT_EDGE_DIM] with flat receiver / sender node indices.

        Bodies use ContactGNN's numbering b*(N+2)+slot; the world node of scene b is B*(N+2)+b.
        Receivers are dynamic bodies only (as for contact edges).
        """
        props, context = _pad_inputs(props, context); state = state.float()
        batch, count = state.shape[:2]; total = count+2
        f = _frames(state, joints, joint_mask, body_mask=mask)
        movable = mask & (props[..., 5] < .5)
        offset = torch.arange(batch, device=state.device)[:, None]*total
        to_child = f.valid & movable.gather(1, f.child)
        to_parent = f.valid & ~f.world & movable.gather(1, f.parent)
        sender_child = torch.where(f.world, batch*total+offset//total, offset+f.parent)
        gravity = context[:, 3] if self.joint_features > LEGACY_JOINT_EDGE_DIM else None
        features = torch.cat((self._joint_features(f, True, gravity)[to_child], self._joint_features(f, False, gravity)[to_parent]))
        receiver = torch.cat(((offset+f.child)[to_child], (offset+f.parent)[to_parent]))
        sender = torch.cat((sender_child[to_child], (offset+f.child)[to_parent]))
        return features, receiver, sender

    # ---- network
    def predict_raw(self, state, props, context, mask, joints=None, joint_mask=None) -> torch.Tensor:
        """Normalized network output [B,N,12] (float32)."""
        batch, count = state.shape[:2]; total = count+2
        jointed = joints is not None and joint_mask is not None and joints.shape[1] > 0 and bool(joint_mask.any())
        with _no_autocast(state):
            node, edge, receiver, sender = self.graph(state, props, context, mask)
            if jointed:
                skip = torch.zeros(batch, total, total, dtype=torch.bool, device=state.device)
                skip[:, :count, :count] = jointed_pairs(joints, joint_mask, count)
                keep = ~skip.reshape(-1)[receiver*total+sender % total]
                edge, receiver, sender = edge[keep], receiver[keep], sender[keep]
                joint_edge, joint_receiver, joint_sender = self.joint_graph(state, props, context, mask, joints, joint_mask)
        h = self.node_encoder(node); e = self.edge_encoder(edge)
        shared = self.shared_rounds if 0 < self.shared_rounds < self.rounds else self.rounds
        if jointed:
            h = torch.cat((h, self.world_node.to(h.dtype).expand(batch, -1)))
            je = self.joint_encoder(joint_edge)
            if shared < self.rounds:                                     # per-scene gate of the joint-only rounds
                scene = joint_mask.bool().any(1)
                node_gate = torch.cat((scene.repeat_interleave(total), scene))[:, None]
                edge_gate = node_gate[receiver]
        for index, (edge_update, node_update) in enumerate(zip(self.edge_updates, self.node_updates)):
            if index >= shared and not jointed: break
            update = edge_update(torch.cat((e, h[receiver], h[sender]), -1))
            e = e+update if index < shared else torch.where(edge_gate, e+update, e)
            total_message = torch.zeros_like(h).index_add_(0, receiver, e.to(h.dtype))
            if jointed:
                je = je+self.joint_updates[index](torch.cat((je, h[joint_receiver], h[joint_sender]), -1))
                total_message = total_message.index_add_(0, joint_receiver, je.to(h.dtype))
            update = node_update(torch.cat((h, total_message), -1))
            h = h+update if index < shared else torch.where(node_gate, h+update, h)
        return self.decoder(h[:batch*total]).float().reshape(batch, total, 12)[:, :count]

    def forward(self, state, props, context, mask, joints=None, joint_mask=None) -> torch.Tensor:
        out = self.predict_raw(state, props, context, mask, joints, joint_mask)
        with _no_autocast(state):
            raw = out.float()*self.out_std+self.out_mean
            return self.integrate(state, raw, props, context, mask, clamp=True)

# ------------------------------------------- contact code with jointed pairs excluded
_PATCHED: dict[str, Any] = {}

def _contact_functions() -> dict[str, Any]:
    """contact_gnn's project / penetration_depths / physics_losses with one extra keyword, _exclude
    [B,N,N]: body pairs that never collide.

    contact_gnn builds its pair mask inside those functions and offers no hook, and that file must
    stay untouched; rather than keep ~250 copied lines in sync, the three functions are re-compiled
    from their own source with a one-line pair filter added.  If the source no longer has the
    expected lines the originals are used (jointed pairs then collide in the projection) and a
    warning is raised.
    """
    if _PATCHED: return _PATCHED
    edits = dict(
        project=(('stable_margin: float = .003)', 'stable_margin: float = .003, _exclude=None, _slop=None, _articulation=None)'),
                 ('pair = pair & ~torch.eye(count, dtype=torch.bool, device=device)\n',
                  'pair = pair & ~torch.eye(count, dtype=torch.bool, device=device)\n'
                  '        if _exclude is not None: pair = pair & ~_exclude\n'
                  '        if _slop is not None: slop = slop+_slop[..., None]   # per-body penetration tolerance [B,N] (m)\n'),
                 # a pair tolerates the larger of its two bodies' slops ([B,N,1] -> [B,a,b] on the broad-phase subset)
                 ('depth = (inside-slop).clamp_min(0)*pair_\n',
                  'pair_slop = slop if _slop is None else torch.maximum(take(slop), take(slop).transpose(1, 2))\n'
                  '                depth = (inside-pair_slop).clamp_min(0)*pair_\n'),
                 # a body of an articulation sleeps only while every dynamic body of that articulation is slow
                 ('        vel = torch.where(asleep[..., None], torch.zeros_like(vel), vel)\n',
                  '        if _articulation is not None:\n'
                  '            fast = dynamic & ~((vel.norm(dim=-1) < sleep_lin*heavy) & (ang.norm(dim=-1) < sleep_ang*heavy))\n'
                  '            linked = (_articulation[:, :, None] == _articulation[:, None, :]) & (_articulation[:, :, None] >= 0)\n'
                  '            asleep = asleep & ~(linked & fast[:, None, :]).any(-1)\n'
                  '        vel = torch.where(asleep[..., None], torch.zeros_like(vel), vel)\n')),
        penetration_depths=(('mask: torch.Tensor)', 'mask: torch.Tensor, _exclude=None)'),
                            ('pair = pair & ~torch.eye(count, dtype=torch.bool, device=state.device)\n',
                             'pair = pair & ~torch.eye(count, dtype=torch.bool, device=state.device)\n'
                             '            if _exclude is not None: pair = pair & ~_exclude\n')),
        physics_losses=(('mask, rest_next)', 'mask, rest_next, _exclude=None)'),
                        ('penetration_depths(pred_next, props, context, mask)', 'penetration_depths(pred_next, props, context, mask, _exclude)'),
                        ('penetration_depths(true_next, props, context, mask)', 'penetration_depths(true_next, props, context, mask, _exclude)')))
    try:
        namespace = dict(vars(cg))
        for name, replacements in edits.items():
            source = inspect.getsource(getattr(cg, name))
            for old, new in replacements:
                if source.count(old) != 1: raise RuntimeError(f'contact_gnn.{name} changed: {old.strip()!r} not found once')
                source = source.replace(old, new)
            exec(compile(source, f'<joint_gnn patch of contact_gnn.{name}>', 'exec'), namespace)
        _PATCHED.update({name: namespace[name] for name in edits}, patched=True)
    except Exception as error:                                   # pragma: no cover - only if contact_gnn changes
        warnings.warn(f'joint_gnn: cannot exclude jointed pairs from contact_gnn functions ({error}); using them unchanged')
        ignore = lambda function: (lambda *args, _exclude=None, _slop=None, _articulation=None, **kwargs: function(*args, **kwargs))
        _PATCHED.update({name: ignore(getattr(cg, name)) for name in edits}, patched=False)
    return _PATCHED

def penetration_depths(state, props, context, mask, joints=None, joint_mask=None) -> dict[str, torch.Tensor]:
    """contact_gnn.penetration_depths, ignoring overlap between two bodies joined by a joint."""
    exclude = None if joints is None else jointed_pairs(joints, joint_mask, state.shape[1])
    return _contact_functions()['penetration_depths'](state, props, context, mask, _exclude=exclude)

def physics_losses(pred_next, true_next, prev_state, props, context, mask, rest_next, joints=None, joint_mask=None) -> dict[str, torch.Tensor]:
    """contact_gnn.physics_losses with jointed pairs excluded from the penetration penalty."""
    exclude = None if joints is None else jointed_pairs(joints, joint_mask, pred_next.shape[1])
    return _contact_functions()['physics_losses'](pred_next, true_next, prev_state, props, context, mask, rest_next, _exclude=exclude)

# ----------------------------------------------------------------- projection
@torch.no_grad()
def solve_joints(state, props, context, mask, joints, joint_mask, iters: int = 6, snap: bool = True, limits: bool = True,
                 limit_slop: tuple[float, float] = LIMIT_SLOP) -> torch.Tensor:
    """Non-learned joint projection of a state [B,N,13] (no contact handling).

    Position pass (up to iters sweeps over all joints at once; a body's corrections are averaged
    over its joints): (1) rotate the two bodies so that hinge axes are parallel, fixed / prismatic
    orientations equal and hinge angles inside their limits, (2) close the anchor gaps (prismatic:
    the off-axis gap plus the limit excess).  Every correction is split between the two bodies by
    inverse mass and inverse box inertia with an exact 3x3 solve per joint; static bodies and the
    world do not move.  While closing a gap a body may turn freely on a ball joint, only about the
    axis on a hinge and not at all on a fixed / prismatic joint, so step 2 does not undo step 1.
    With snap set, the sweeps also stop as soon as one no longer halves the largest error (long
    chains: averaged sweeps are slow on smooth errors), and one root-to-leaf pass then places every
    child exactly on its parent's joint (the child alone moves there, whatever the masses).  That
    pass covers the first joint of each dynamic child of an acyclic articulation; closed loops keep
    whatever the sweeps left.
    Velocity pass: the same two steps on velocities remove the relative motion the joint forbids
    (off the hinge axis, into a limit, of the two bodies at the child's anchor point); linear
    momentum is conserved.  limits=False ignores [lower, upper] in both passes.
    limit_slop = (rad, m) widens [lower, upper] of hinge / prismatic joints in both passes: a coordinate is only
    clamped (and its outward velocity only removed) once it is further than this outside its limits.  Bullet's
    own limits are soft (they overshoot on impact and yield under load: 1.7% of the limited hinge frames of
    data/joints_v1 are > 0.05 rad outside, up to 0.3 rad), so a hard clamp contradicts the ground truth: with
    PyBullet as the model and this function applied after every step, 80% of 210 test episodes stay within
    5 mm of pure PyBullet after 1 s with (0, 0), 90% with the default and 95% with (0.1, 0.01) (96% with
    limits=False).  (0, 0) restores the hard clamp.
    Sweeps stop early once every error is below 0.02 mm / 2e-5 rad (1e-4 m/s for velocities).
    """
    with _no_autocast(state):
        props, context = _pad_inputs(props, context)
        original = state; current = state.float(); joints = joints.float()
        batch, count = current.shape[:2]; device = current.device
        parent, child, world, valid = _indices(joints, joint_mask, count, mask)
        if not bool(valid.any()): return original
        dynamic = mask & (props[..., 5] < .5)
        inverse_mass = dynamic/props[..., 3].clamp_min(1e-3)
        size2 = props[..., :3].square()
        inverse_inertia = inverse_mass[..., None]*12/(size2.sum(-1, keepdim=True)-size2).clamp_min(1e-8)   # box frame diagonal
        free_p = (valid & ~world).float()
        index3 = lambda index: index[..., None].expand(-1, -1, 3)
        pick = lambda values, index: values.gather(1, index3(index))
        mass_p, mass_c = inverse_mass.gather(1, parent)*free_p, inverse_mass.gather(1, child)*valid
        inertia_p, inertia_c = pick(inverse_inertia, parent)*free_p[..., None], pick(inverse_inertia, child)*valid[..., None]
        offset = torch.arange(batch, device=device)[:, None]*count
        flat = torch.cat((offset+parent, offset+child), 1).reshape(-1)           # [B*2J] parent ends then child ends per scene
        hinge, ball, prismatic, fixed = (joints[..., 2:6] > .5).unbind(-1)
        hinge, ball, prismatic, fixed = hinge & valid, ball & valid, prismatic & valid, fixed & valid
        lower, upper = joints[..., 15], joints[..., 16]
        if not limits: lower, upper = torch.full_like(lower, -1e9), torch.full_like(upper, 1e9)
        span = upper-lower
        limited = (hinge & (span < HINGE_FREE)) | (prismatic & (span < SLIDE_FREE))
        slack = torch.where(prismatic, float(limit_slop[1]), float(limit_slop[0]))
        lower, upper = lower-slack, upper+slack                                # both passes tolerate Bullet's soft limits
        angular = hinge | prismatic | fixed
        touching = torch.cat((mass_p > 0, mass_c > 0), 1)                       # [B,2J] parent ends then child ends
        eye = torch.eye(3, device=device)
        stuck = ((mass_p+mass_c) <= 0)[..., None, None]*eye                     # keeps the 3x3 systems regular
        ball_w, hinge_w = ball[..., None, None].float(), hinge[..., None, None].float()

        def spread(delta_p, delta_c, involved):
            """Per-body mean [B,N,3] of the per-joint corrections of the joints in `involved`."""
            total = current.new_zeros(batch*count, 3).index_add_(0, flat, torch.cat((delta_p, delta_c), 1).reshape(-1, 3))
            number = current.new_zeros(batch*count).index_add_(0, flat, (involved.repeat(1, 2) & touching).float().reshape(-1))
            return (total/number.clamp_min(1)[:, None]).reshape(batch, count, 3)

        def skew(v):
            x, y, z = v.unbind(-1); o = torch.zeros_like(x)
            return torch.stack((o, -z, y, z, o, -x, -y, x, o), -1).reshape(*v.shape[:-1], 3, 3)

        def solve(matrix, vector):
            return torch.linalg.solve(matrix+stuck, vector[..., None]).squeeze(-1)

        def frames(now, reach: bool = False):
            f = _frames(now, joints, valid)
            if reach: f.arm_p = f.arm_p+f.gap                               # parent lever to the child's anchor point
            # world-frame inverse inertia of each end, and the rotation an anchor impulse may cause
            f.full_p = (f.rp*inertia_p[..., None, :])@f.rp.transpose(-1, -2)
            f.full_c = (f.rc*inertia_c[..., None, :])@f.rc.transpose(-1, -2)
            outer = f.axis_p[..., :, None]*f.axis_p[..., None, :]
            about = lambda inertia, matrix: ((inertia > 0).all(-1)/(_rot_t(matrix, f.axis_p).square()/inertia.clamp_min(1e-12)).sum(-1).clamp_min(1e-12))[..., None, None]
            f.free_p = ball_w*f.full_p+hinge_w*about(inertia_p, f.rp)*outer     # 1 / (a^T I a) about a hinge axis
            f.free_c = ball_w*f.full_c+hinge_w*about(inertia_c, f.rc)*outer
            f.skew_p, f.skew_c = skew(f.arm_p), skew(f.arm_c)
            f.anchor = (mass_p+mass_c)[..., None, None]*eye-f.skew_p@f.free_p@f.skew_p-f.skew_c@f.free_c@f.skew_c
            return f

        def rotated(now, turn):
            return torch.cat((now[..., :3], normalize_q(quat_mul(rotvec_to_quat(turn), now[..., 3:7])), now[..., 7:]), -1)

        converged, previous = False, float('inf')
        for _ in range(iters):
            # 1. orientation: hinge axes parallel and angle within limits; fixed / prismatic orientations equal
            f = frames(current)
            swing = torch.cross(f.axis_c, f.axis_p, dim=-1)
            swing = _unit(swing)*torch.atan2(_length(swing), (f.axis_c*f.axis_p).sum(-1))[..., None]
            excess = f.coordinate-torch.minimum(torch.maximum(f.coordinate, lower), upper)
            delta = torch.where(hinge[..., None], swing-excess[..., None]*f.axis_p, quat_to_rotvec(quat_mul(f.qp, quat_conj(f.qc))))
            delta = delta*angular[..., None]                                   # rotation the child needs relative to the parent
            slide = (f.gap*f.axis_p).sum(-1)
            beyond = slide-torch.minimum(torch.maximum(slide, lower), upper)
            gap = (f.violation+(prismatic*beyond)[..., None]*f.axis_p)*valid[..., None]
            turn_error, error = float(delta.abs().max()), max(float(delta.abs().max()), float(gap.abs().max()))
            converged = error <= 2e-5
            if converged or (snap and error > .5*previous): break            # done, or stalling: leave the rest to the snap
            previous = error
            if turn_error > 2e-5:
                impulse = solve(f.full_p+f.full_c, delta)[..., None]
                turn = spread(-(f.full_p@impulse).squeeze(-1), (f.full_c@impulse).squeeze(-1), angular)
                current = rotated(current, turn)
                f = frames(current)
                slide = (f.gap*f.axis_p).sum(-1)
                beyond = slide-torch.minimum(torch.maximum(slide, lower), upper)
                gap = (f.violation+(prismatic*beyond)[..., None]*f.axis_p)*valid[..., None]
            # 2. anchors
            push = solve(f.anchor, gap)
            move = spread(push*mass_p[..., None], -push*mass_c[..., None], valid)
            torque = lambda arm: torch.cross(arm, push, dim=-1)[..., None]
            turn = spread((f.free_p@torque(f.arm_p)).squeeze(-1), -(f.free_c@torque(f.arm_c)).squeeze(-1), valid)
            current = rotated(torch.cat((current[..., :3]+move, current[..., 3:]), -1), turn)
        if not converged:
            if snap:
                slots = joints.shape[1]
                depth = torch.zeros(batch, count, dtype=torch.long, device=device)
                for _ in range(slots+1):                                       # depth of a body = joints between it and its root
                    step = (torch.where(world, torch.zeros_like(parent), depth.gather(1, parent))+1)*valid
                    depth = torch.zeros_like(depth).scatter_reduce(1, child, step, 'amax')
                level = depth.gather(1, child)
                same = (child[:, :, None] == child[:, None, :]) & valid[:, :, None] & valid[:, None, :]
                earlier = torch.ones(slots, slots, dtype=torch.bool, device=device).tril(-1)
                tree = valid & ~(same & earlier).any(-1) & (mass_c > 0) & (level <= slots)
                for rank in range(1, int((level*tree).max())+1):
                    chosen = tree & (level == rank)
                    if not bool(chosen.any()): continue
                    f = _frames(current, joints, chosen)
                    swing = torch.cross(f.axis_c, f.axis_p, dim=-1)
                    swing = _unit(swing)*torch.atan2(_length(swing), (f.axis_c*f.axis_p).sum(-1))[..., None]
                    delta = torch.where(hinge[..., None], swing, quat_to_rotvec(quat_mul(f.qp, quat_conj(f.qc))))
                    quat = normalize_q(quat_mul(rotvec_to_quat(delta*angular[..., None]), f.qc))
                    relative = quat_mul(quat_conj(f.qp), quat)                 # axes now parallel: clamp the hinge angle
                    s, w = (relative[..., :3]*f.axis).sum(-1), relative[..., 3]
                    angle = torch.atan2(2*s*w, w*w-s*s+1e-12)
                    excess = (angle-torch.minimum(torch.maximum(angle, lower), upper))*hinge
                    quat = normalize_q(quat_mul(rotvec_to_quat(-excess[..., None]*f.axis_p), quat))
                    slide = (f.gap*f.axis_p).sum(-1)
                    slide = torch.minimum(torch.maximum(slide, lower), upper)*prismatic
                    place = f.sp[..., :3]+f.arm_p+slide[..., None]*f.axis_p-_rot(rotation(quat), f.local_c)
                    rows, columns = chosen.nonzero(as_tuple=True)
                    current = current.clone()
                    current[rows, child[rows, columns], :7] = torch.cat((place, quat), -1)[rows, columns]

        f = frames(current, reach=True)
        vel, ang = current[..., 7:10], current[..., 10:13]
        edge = .002                                                            # "at the limit" tolerance (rad / m)
        at_upper, at_lower = limited & (f.coordinate >= upper-edge), limited & (f.coordinate <= lower+edge)
        outward = lambda rate: ((at_upper & (rate > 0)) | (at_lower & (rate < 0))).float()
        for _ in range(iters):
            # 1. angular: hinges keep only the spin about their axis (none of it into a limit)
            spin = pick(ang, child)-pick(ang, parent)*free_p[..., None]
            along = (spin*f.axis_p).sum(-1)
            keep = torch.where(hinge, along*(1-outward(along)), torch.zeros_like(along))
            bad = (spin-keep[..., None]*f.axis_p)*angular[..., None]
            worst = float(bad.abs().max())
            impulse = solve(f.full_p+f.full_c, bad)[..., None]
            ang = ang+spread((f.full_p@impulse).squeeze(-1), -(f.full_c@impulse).squeeze(-1), angular)
            # 2. linear: the two anchors move together (prismatic: except along the axis, unless into a limit)
            slip = (pick(vel, child)+torch.cross(pick(ang, child), f.arm_c, dim=-1)
                    -(pick(vel, parent)+torch.cross(pick(ang, parent), f.arm_p, dim=-1))*free_p[..., None])
            rate = (slip*f.axis_p).sum(-1)
            keep = torch.where(prismatic, rate*(1-outward(rate)), torch.zeros_like(rate))
            bad = (slip-keep[..., None]*f.axis_p)*valid[..., None]
            worst = max(worst, float(bad.abs().max()))
            if worst <= 1e-4: break
            impulse = solve(f.anchor, bad)
            torque = lambda arm: torch.cross(arm, impulse, dim=-1)[..., None]
            vel = vel+spread(impulse*mass_p[..., None], -impulse*mass_c[..., None], valid)
            ang = ang+spread((f.free_p@torque(f.arm_p)).squeeze(-1), -(f.free_c@torque(f.arm_c)).squeeze(-1), valid)
        output = torch.nan_to_num(torch.cat((current[..., :7], vel, ang), -1), nan=0., posinf=0., neginf=0.)
        return torch.where(dynamic[..., None], output.to(original.dtype), original)

@torch.no_grad()
def project_joints(prev, pred, props, context, mask, joints, joint_mask, iters: int = 6, outer: int = 1, snap: bool = True,
                   limits: bool = True, limit_slop: tuple[float, float] = LIMIT_SLOP, joint_slop: float = JOINT_CONTACT_SLOP,
                   articulation_sleep: bool = True, slop_scope: str = 'scene', **contact_options) -> torch.Tensor:
    """contact_gnn.project (jointed pairs do not collide) followed by solve_joints.

    outer > 1 repeats the pair, so that a link pulled back into the support by its joint is lifted
    out again; contact_options go to contact_gnn.project (iters there stays at its own default).
    joint_slop: extra penetration (m) the contact projection tolerates.  Bullet's Featherstone links rest
    0.5-3 mm inside the support / each other; lifting them out every step (slop 0) fights the joints.
    slop_scope 'scene' (default): every body of a scene that contains a joint gets it (free boxes there lean on /
    are hit by the links); 'articulation': only bodies that belong to an articulation (a contact involving one such
    body uses it).  A scene without joints is always projected exactly as by contact_gnn.project.
    articulation_sleep: a body of an articulation is only put to sleep (velocity zeroed, pose snapped back) while
    EVERY dynamic body of its articulation is below the sleep speeds; otherwise a slow link next to a moving one
    (the base of a laptop whose lid is falling, a ragdoll's torso) is frozen while its neighbours pull on it.
    joint_slop=0, articulation_sleep=False restore the previous behaviour.
    """
    count = pred.shape[1]
    exclude = jointed_pairs(joints, joint_mask, count)
    group = articulations(joints, joint_mask, count)
    jointed = group >= 0
    if slop_scope not in ('scene', 'articulation'): raise ValueError(f'slop_scope {slop_scope!r}')
    tolerant = jointed.any(1, keepdim=True).expand_as(jointed) if slop_scope == 'scene' else jointed
    slop = tolerant.float()*float(joint_slop) if joint_slop and bool(jointed.any()) else None
    group = group if articulation_sleep and bool(jointed.any()) else None
    functions = _contact_functions()
    contact = functions['project']
    extra = dict(_slop=slop, _articulation=group) if functions.get('patched') else {}
    state = pred
    for _ in range(max(outer, 1)):
        state = contact(prev, state, props, context, mask, _exclude=exclude, **extra, **contact_options)
        state = solve_joints(state, props, context, mask, joints, joint_mask, iters, snap=snap, limits=limits, limit_slop=limit_slop)
    return state

# ------------------------------------------------------------------------- io
def init_from_contact_gnn(model: JointGNN, ckpt: dict) -> dict[str, Any]:
    """Warm start: copy every ContactGNN weight (and the output normaliser) into a JointGNN.

    Message-passing rounds beyond the checkpoint's start as the identity (their final LayerNorm is
    zeroed), so that without joints the model reproduces the ContactGNN exactly.  The joint encoder,
    joint updates and world node keep their fresh initialisation.  Returns what was done.
    """
    if ckpt.get('kind') != ContactGNN.kind: raise ValueError(f"Not a contact_gnn checkpoint: {ckpt.get('kind')}")
    source = ckpt['state_dict']; config = ckpt.get('config', {})
    width = source['decoder.0.weight'].shape[1]
    rounds = len({key.split('.')[1] for key in source if key.startswith('edge_updates.')})
    if width != model.width: raise ValueError(f'width mismatch: checkpoint {width}, model {model.width}')
    if rounds > model.rounds: raise ValueError(f'checkpoint has {rounds} rounds, model only {model.rounds}')
    target = model.state_dict()
    unexpected = [key for key in source if key not in target or target[key].shape != source[key].shape]
    if unexpected: raise ValueError(f'checkpoint weights without a place in JointGNN: {unexpected}')
    missing = model.load_state_dict(source, strict=False).missing_keys
    with torch.no_grad():
        for index in range(rounds, model.rounds):
            for update in (model.edge_updates[index], model.node_updates[index]):
                update[-1].weight.zero_(); update[-1].bias.zero_()
    model.edge_radius = float(config.get('edge_radius', model.edge_radius))
    return dict(loaded=len(source), new=missing, identity_rounds=list(range(rounds, model.rounds)))

def build_from_checkpoint(ckpt: dict) -> JointGNN:
    """JointGNN in eval mode from {'kind','config','state_dict',...}.

    A 'contact_gnn' checkpoint is also accepted: it yields a warm-started JointGNN with the same
    number of rounds (exact on joint-free scenes, untrained joint edges).
    """
    config = {key: ckpt.get('config', {})[key] for key in ('width', 'rounds', 'edge_radius', 'joint_features', 'shared_rounds') if key in ckpt.get('config', {})}
    if ckpt.get('kind') == JointGNN.kind and 'joint_features' not in config:     # checkpoints from before the gravity-torque features
        config['joint_features'] = int(ckpt['state_dict']['joint_encoder.0.weight'].shape[1])
    if ckpt.get('kind') == ContactGNN.kind:
        config.pop('joint_features', None); config.pop('shared_rounds', None)
        model = JointGNN(**config); init_from_contact_gnn(model, ckpt)
    elif ckpt.get('kind') == JointGNN.kind:
        model = JointGNN(**config); model.load_state_dict(ckpt['state_dict'])
    else:
        raise ValueError(f"Not a joint_gnn checkpoint: {ckpt.get('kind')}")
    model.eval()
    return model
