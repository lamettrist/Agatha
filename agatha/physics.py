"""Isolated DIRECT clients. No global mutable physics world, no network."""
import math
from typing import Any
import numpy as np
import pybullet as p
from .coordinates import to_bullet, from_bullet, quat_from_bullet, euler_to_bullet
from .schema import Scene, Action

DT = 1 / 240

class World:
    def __init__(self, scene: Scene):
        self.scene = scene.model_copy(deep=True)
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
        self._box("support", (w, .08, d), (0, scene.support.height-.04, 0), 0)
        for o in scene.objects:
            self._box(o.id, o.size, o.position, 0 if o.is_static else o.mass_kg,
                      euler_to_bullet(o.rotation_euler), o.friction)

    def _box(self, name, size, pos, mass, quat=(0,0,0,1), friction=.5):
        half = (size[0]/2, size[2]/2, size[1]/2)
        shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half, physicsClientId=self.client)
        b = p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=shape,
                             basePosition=to_bullet(pos), baseOrientation=quat,
                             physicsClientId=self.client)
        p.changeDynamics(b, -1, lateralFriction=friction, restitution=0,
                         linearDamping=.04, angularDamping=.04, physicsClientId=self.client)
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
                    dv = to_bullet((a.direction[0]/n*a.impulse_Ns/mass, 0,
                                    a.direction[1]/n*a.impulse_Ns/mass))
                    p.resetBaseVelocity(b, [v[i]+dv[i] for i in range(3)], av, physicsClientId=self.client)
            elif a.type == "tilt_support":
                self.tilt[0 if a.axis == "x" else 2] = math.radians(a.angle_deg)
                p.resetBasePositionAndOrientation(self.bodies["support"],
                    to_bullet((0, self.scene.support.height-.04, 0)),
                    euler_to_bullet(self.tilt), physicsClientId=self.client)
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
            local = rot.T @ (pos - [0, scene.support.height-.04, 0])
            if (abs(local[0]) > scene.support.size[0]/2 or abs(local[2]) > scene.support.size[1]/2) and local[1] < .01:
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
        return dict(engine="physics", **summarize(scene, frames, contacts, world.tilt),
                    support_rotation_euler=world.tilt, estimated=True,
                    disclaimer="Box approximation with estimated parameters; not a guaranteed real-world outcome.")
