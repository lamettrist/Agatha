"""Isolated DIRECT clients. No global mutable physics world, no network."""
import math
from typing import Any
import numpy as np
import pybullet as p
from .coordinates import to_bullet, from_bullet, quat_from_bullet, euler_to_bullet
from .schema import Scene, Action

DT = 1 / 240

class World:
    def __init__(self, scene: Scene, dynamics: dict | None = None):
        """dynamics (optional, default behaviour unchanged) overrides parameters the API
        cannot express: {'objects': {id: {'restitution','rolling_friction','spinning_friction'}},
        'support_friction': .5, 'support_restitution': 0.}."""
        self.scene = scene.model_copy(deep=True)
        dynamics = dynamics or {}
        overrides = dynamics.get("objects") or {}
        self.client = p.connect(p.DIRECT)
        self.bodies: dict[str, int] = {}
        self.gravity = 9.81
        self.tilt = [0., 0., 0.]
        c = self.client
        p.setPhysicsEngineParameter(fixedTimeStep=DT, numSolverIterations=50,
                                   deterministicOverlappingPairs=1, physicsClientId=c)
        p.setGravity(0, 0, -self.gravity, physicsClientId=c)
        self._box("floor", (20, .1, 20), (0, -.05, 0), 0)
        w, d = scene.support.size
        self._box("support", (w, scene.support.thickness, d), (0, scene.support.height-scene.support.thickness/2, 0), 0,
                  friction=dynamics.get("support_friction", .5),
                  restitution=dynamics.get("support_restitution", 0))
        for o in scene.objects:
            if o.collision is not None or o.shape != 'box' or o.restitution or o.soft_approx or o.linear_damping!=.04 or o.angular_damping!=.04:
                from .scene_reconstruction.bodies import create_body
                create_body(self, o, overrides.get(o.id))
            else:
                params = dict(restitution=o.restitution, **{k:v for k,v in overrides.get(o.id, {}).items() if k != 'restitution'})
                params['restitution'] = overrides.get(o.id, {}).get('restitution', o.restitution)
                self._box(o.id, o.size, o.position, 0 if o.is_static else o.mass_kg,
                          euler_to_bullet(o.rotation_euler), o.friction, **params)

        if not scene.support.collisions_enabled:
            self._support_collisions(False)

    def _support_collisions(self, enabled: bool):
        self.scene.support.collisions_enabled = enabled
        ids = ["support"] + [o.id for o in self.scene.objects if o.is_static
            and o.size[0] >= self.scene.support.size[0]*.95
            and o.size[2] >= self.scene.support.size[1]*.95
            and abs(o.position[1]+o.size[1]/2-self.scene.support.height)<.03]
        for id in ids:
            if id in self.bodies:
                p.setCollisionFilterGroupMask(self.bodies[id], -1,
                    1 if enabled else 0, -1 if enabled else 0, physicsClientId=self.client)

    def _support_alias_ids(self) -> list[str]:
        """Static objects that duplicate the support slab (e.g. the demo 'desk')."""
        s = self.scene.support
        return [o.id for o in self.scene.objects if o.is_static
                and o.size[0] >= s.size[0]*.95 and o.size[2] >= s.size[1]*.95
                and abs(o.position[1]+o.size[1]/2-s.height) < .03]

    def contact_pairs(self, max_distance: float = 1e-3) -> set[tuple[str, str]]:
        """Sorted name pairs with a contact point in the most recent substep."""
        reverse = {v: k for k, v in self.bodies.items()}
        pairs = set()
        for cp in p.getContactPoints(physicsClientId=self.client):
            if cp[8] <= max_distance and cp[1] in reverse and cp[2] in reverse:
                pairs.add(tuple(sorted((reverse[cp[1]], reverse[cp[2]]))))
        return pairs

    def contact_flags(self, pairs: set[tuple[str, str]] | None = None,
                      max_distance: float = 1e-3) -> dict[str, tuple[bool, bool, bool]]:
        """Per present object: (touching support, touching floor, touching another object).

        A static duplicate of the support counts as the support, not as an object."""
        pairs = self.contact_pairs(max_distance) if pairs is None else pairs
        support = {"support", *self._support_alias_ids()}
        flags = {o.id: [False, False, False] for o in self.scene.objects if o.id in self.bodies}
        for a, b in pairs:
            for me, other in ((a, b), (b, a)):
                if me in flags:
                    flags[me][0 if other in support else 1 if other == "floor" else 2] = True
        return {k: tuple(v) for k, v in flags.items()}

    def _box(self, name, size, pos, mass, quat=(0,0,0,1), friction=.5, restitution=0,
             rolling_friction=None, spinning_friction=None):
        half = (size[0]/2, size[2]/2, size[1]/2)
        shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half, physicsClientId=self.client)
        b = p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=shape,
                             basePosition=to_bullet(pos), baseOrientation=quat,
                             physicsClientId=self.client)
        extra = {}
        if rolling_friction is not None: extra["rollingFriction"] = rolling_friction
        if spinning_friction is not None: extra["spinningFriction"] = spinning_friction
        p.changeDynamics(b, -1, lateralFriction=friction, restitution=restitution,
                         linearDamping=.04, angularDamping=.04, physicsClientId=self.client, **extra)
        self.bodies[name] = b

    def apply(self, actions: list[Action]) -> None:
        for a in actions:
            if hasattr(a, "id") and a.id not in {o.id for o in self.scene.objects}:
                raise ValueError(f"Unknown object: {a.id}")
            if hasattr(a, "id") and a.id not in self.bodies:
                raise ValueError(f"Object already removed: {a.id}")
            if a.type == "remove":
                p.removeBody(self.bodies.pop(a.id), physicsClientId=self.client)
            elif a.type == "set_mass":
                o = next(o for o in self.scene.objects if o.id == a.id)
                o.mass_kg, o.mass_source = a.kg, "user"
                if not o.is_static:
                    p.changeDynamics(self.bodies[a.id], -1, mass=a.kg, physicsClientId=self.client)
            elif a.type == "push":
                b = self.bodies[a.id]
                mass = p.getDynamicsInfo(b, -1, physicsClientId=self.client)[0]
                if mass > 0:
                    v, av = p.getBaseVelocity(b, physicsClientId=self.client)
                    n = math.hypot(*a.direction)
                    direction = a.direction if len(a.direction)==3 else (a.direction[0],0,a.direction[1])
                    dv = to_bullet(tuple(v/n*a.impulse_Ns/mass for v in direction))
                    p.resetBaseVelocity(b, [v[i]+dv[i] for i in range(3)], av, physicsClientId=self.client)
            elif a.type == "tilt_support":
                self.tilt[0 if a.axis == "x" else 2] = math.radians(a.angle_deg)
                p.resetBasePositionAndOrientation(self.bodies["support"],
                    to_bullet((0, self.scene.support.height-self.scene.support.thickness/2, 0)),
                    euler_to_bullet(self.tilt), physicsClientId=self.client)
            elif a.type == "set_support_collisions":
                self._support_collisions(a.enabled)
            elif a.type == "set_gravity":
                self.gravity = a.g
                p.setGravity(0, 0, -a.g, physicsClientId=self.client)

    def snapshot(self, t: float) -> dict[str, Any]:
        objects = []
        for o in self.scene.objects:
            if o.id not in self.bodies:
                continue
            pos, q = p.getBasePositionAndOrientation(self.bodies[o.id], physicsClientId=self.client)
            vel, av = p.getBaseVelocity(self.bodies[o.id], physicsClientId=self.client)
            objects.append(dict(id=o.id, position=from_bullet(pos), quaternion=quat_from_bullet(q),
                                velocity=from_bullet(vel), angular_velocity=from_bullet(av)))
        return dict(t=t, objects=objects)

    def step(self, count=1):
        for _ in range(count):
            p.stepSimulation(physicsClientId=self.client)

    def close(self):
        if p.isConnected(self.client):
            p.disconnect(self.client)
    def __enter__(self): return self
    def __exit__(self, *args): self.close()


def summarize(scene: Scene, frames: list[dict], contacts: list[dict] | None = None,
              tilt: list[float] | None = None) -> dict:
    """Shared event definitions for physics and learned predictions."""
    initial = {o.id: o for o in scene.objects}
    fell, tipped = set(), set()
    # Test the COM in the tilted support's local frame, below its top by 3cm.
    rot = np.asarray(p.getMatrixFromQuaternion(p.getQuaternionFromEuler(tilt or [0,0,0]))).reshape(3,3)
    for frame in frames:
        for state in frame["objects"]:
            o = initial[state["id"]]
            if o.is_static: continue
            pos = np.asarray(state["position"])
            local = rot.T @ (pos - [0, scene.support.height-scene.support.thickness/2, 0])
            if (not scene.support.collisions_enabled or abs(local[0]) > scene.support.size[0]/2 or abs(local[2]) > scene.support.size[1]/2) and local[1] < scene.support.thickness/2-.03:
                fell.add(o.id)
            r = p.getMatrixFromQuaternion(state["quaternion"])
            if r[4] < math.cos(math.radians(35)):
                tipped.add(o.id)
    final = frames[-1]
    displacement = sum(float(np.linalg.norm(np.asarray(s["position"])-initial[s["id"]].position))
                       for s in final["objects"] if not initial[s["id"]].is_static)
    return dict(frames=frames, events=dict(contacts=contacts or [], fell=sorted(fell), tipped=sorted(tipped)),
                final_state=final, total_displacement_m=displacement,
                chaos_score=len(fell)+displacement)


def simulate(scene: Scene, actions: list[Action], duration: float = 3, fps: int = 30) -> dict:
    if fps not in (10,20,30) or not .1 <= duration <= 5:
        raise ValueError("duration must be 0.1..5 seconds; fps 10, 20, or 30")
    with World(scene) as world:
        world.apply(actions)
        frames = [world.snapshot(0)]
        contacts, seen = [], set()
        reverse = {v:k for k,v in world.bodies.items()}
        for i in range(1, round(duration*fps)+1):
            # Sample contacts at every physics substep, including brief collisions.
            for _ in range(240//fps):
                world.step()
                for cp in p.getContactPoints(physicsClientId=world.client):
                    pair = tuple(sorted((reverse[cp[1]], reverse[cp[2]])))
                    if pair not in seen:
                        seen.add(pair)
                        contacts.append(dict(ids=pair, t=round(i/fps,6)))
            frames.append(world.snapshot(i/fps))
        return dict(engine="physics", **summarize(world.scene, frames, contacts, world.tilt),
                    support_rotation_euler=world.tilt, support_collisions_enabled=world.scene.support.collisions_enabled, estimated=True,
                    disclaimer="Box approximation with estimated parameters; not a guaranteed real-world outcome.")
