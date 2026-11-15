"""Collision construction shared by reconstruction, validation and interactive physics."""
import numpy as np
import pybullet as p
from agatha.coordinates import to_bullet, from_bullet, euler_to_bullet
from agatha.schema import Scene, CollisionPart


def create_body(world, obj, overrides=None):
    c=world.client
    parts=obj.collision.parts if obj.collision else [CollisionPart(type=obj.shape if obj.shape=='cylinder' else 'box',
        size=obj.size, radius=obj.size[0]/2, height=obj.size[1])]
    # Fixed links allow arbitrary convex hulls and compound primitive collisions.
    shapes=[]
    for part in parts:
        args=dict(physicsClientId=c)
        if part.type=='box':
            shape=p.createCollisionShape(p.GEOM_BOX,halfExtents=[part.size[0]/2,part.size[2]/2,part.size[1]/2],**args)
        elif part.type=='cylinder':
            shape=p.createCollisionShape(p.GEOM_CYLINDER,radius=part.radius,height=part.height,**args)
        else:
            shape=p.createCollisionShape(p.GEOM_MESH,vertices=[to_bullet(v) for v in part.vertices],**args)
        shapes.append(shape)
    # Geometry is rotated/translated relative to the COM using fixed links.
    n=len(parts)
    b=p.createMultiBody(baseMass=0 if obj.is_static else obj.mass_kg,baseCollisionShapeIndex=-1,
        basePosition=to_bullet(obj.position),baseOrientation=euler_to_bullet(obj.rotation_euler),
        linkMasses=[0.]*n,linkCollisionShapeIndices=shapes,linkVisualShapeIndices=[-1]*n,
        linkPositions=[to_bullet(part.position) for part in parts],
        linkOrientations=[euler_to_bullet(part.rotation_euler) for part in parts],
        linkInertialFramePositions=[[0,0,0]]*n,linkInertialFrameOrientations=[[0,0,0,1]]*n,
        linkParentIndices=[0]*n,linkJointTypes=[p.JOINT_FIXED]*n,linkJointAxis=[[0,0,0]]*n,
        physicsClientId=c)
    # Massless fixed links need an explicit inertia on the massive base.
    w,h,d=obj.size; m=obj.mass_kg
    inertia=[m*(h*h+d*d)/12,m*(w*w+h*h)/12,m*(w*w+d*d)/12]  # Bullet XYZ = API X,-Z,Y
    p.changeDynamics(b,-1,localInertiaDiagonal=inertia,physicsClientId=c)
    params=dict(lateralFriction=obj.friction,restitution=obj.restitution,
                linearDamping=obj.linear_damping,angularDamping=obj.angular_damping)
    for key,value in (overrides or {}).items():
        mapped={'restitution':'restitution','rolling_friction':'rollingFriction','spinning_friction':'spinningFriction'}.get(key)
        if mapped: params[mapped]=value
    for link in range(-1,n): p.changeDynamics(b,link,physicsClientId=c,**params)
    world.bodies[obj.id]=b


def aabb(world, name):
    b=world.bodies[name]; c=world.client
    boxes=[p.getAABB(b,i,physicsClientId=c) for i in range(-1,p.getNumJoints(b,physicsClientId=c))]
    # The empty base has a zero-volume AABB; ignore it when fixed links carry shapes.
    if len(boxes)>1: boxes=boxes[1:]
    return np.min([b[0] for b in boxes],0),np.max([b[1] for b in boxes],0)


def contact_metrics(world, tolerance=.005):
    """Actual narrow-phase closest distances, including all fixed-link shapes."""
    p.performCollisionDetection(physicsClientId=world.client)
    dynamic=[o for o in world.scene.objects if not o.is_static]
    floats=[]; penetrations=[]; distances={}
    s=world.scene.support
    alias={o.id for o in world.scene.objects if o.is_static and o.size[0]>=s.size[0]*.95 and o.size[2]>=s.size[1]*.95 and abs(o.position[1]+o.size[1]/2-s.height)<.03}
    for obj in dynamic:
        target=obj.supported_by or 'support'
        targets=[target]+list(alias) if target=='support' else [target]
        values=[]
        for name in targets:
            if name in world.bodies:
                values.extend(cp[8] for cp in p.getClosestPoints(world.bodies[obj.id],world.bodies[name],
                    distance=10,physicsClientId=world.client))
        distance=min(values) if values else None
        distances[obj.id]=distance
        if distance is None or distance>tolerance: floats.append(obj.id)
    names=list(world.bodies)
    for i,a in enumerate(names):
        for b in names[i+1:]:
            if a not in {o.id for o in dynamic} and b not in {o.id for o in dynamic}: continue
            cps=p.getClosestPoints(world.bodies[a],world.bodies[b],distance=0,physicsClientId=world.client)
            depth=min([cp[8] for cp in cps],default=0.)
            if depth < -tolerance: penetrations.append(dict(ids=[a,b],depth_m=float(-depth)))
    return dict(float_count=len(floats),floating_ids=floats,penetration_count=len(penetrations),
                penetrations=penetrations,support_distance_m=distances,tolerance_m=tolerance)


def resolve_overlaps(world, iterations=80):
    """Minimal horizontal separation of narrow-phase overlaps before damped settling."""
    dynamic=[o for o in world.scene.objects if not o.is_static]
    corrections={o.id:np.zeros(3) for o in dynamic}
    for _ in range(iterations):
        changed=False
        for i,a in enumerate(dynamic):
            for b in dynamic[i+1:]:
                cps=p.getClosestPoints(world.bodies[a.id],world.bodies[b.id],distance=-.0005,physicsClientId=world.client)
                if not cps: continue
                alo,ahi=aabb(world,a.id);blo,bhi=aabb(world,b.id)
                overlap=np.minimum(ahi,bhi)-np.maximum(alo,blo)
                axis=min((0,1),key=lambda axis:overlap[axis])  # Bullet horizontal X,Y
                if overlap[axis]<=0:continue
                direction=1 if (alo[axis]+ahi[axis])>(blo[axis]+bhi[axis]) else -1
                amount=float(overlap[axis]+.001)
                wa=b.mass_kg/(a.mass_kg+b.mass_kg);wb=1-wa
                for obj,weight,sign in ((a,wa,direction),(b,wb,-direction)):
                    pos,q=p.getBasePositionAndOrientation(world.bodies[obj.id],physicsClientId=world.client)
                    pos=list(pos);pos[axis]+=sign*amount*weight
                    p.resetBasePositionAndOrientation(world.bodies[obj.id],pos,q,physicsClientId=world.client)
                    delta=np.zeros(3);delta[axis]=sign*amount*weight;corrections[obj.id]+=np.array(from_bullet(delta))
                changed=True
        if not changed:break
    return {k:v.tolist() for k,v in corrections.items() if np.linalg.norm(v)>1e-8}


def settle(scene: Scene, seconds=2., *, extend_support=True):
    """Snap actual collision bottoms to declared support, damp, simulate, save poses.

    Never relocates overlapping objects to invented stacks. Remaining failures are reported.
    """
    from agatha.physics import World
    result=scene.model_copy(deep=True)
    with World(result) as world:
        for obj in world.scene.objects:
            if obj.is_static: continue
            target=obj.supported_by or 'support'
            if target not in world.bodies: continue
            low,_=aabb(world,obj.id); _,high=aabb(world,target)
            pos,q=p.getBasePositionAndOrientation(world.bodies[obj.id],physicsClientId=world.client)
            pos=list(pos); pos[2]+=high[2]-low[2]
            p.resetBasePositionAndOrientation(world.bodies[obj.id],pos,q,physicsClientId=world.client)
            for link in range(-1,p.getNumJoints(world.bodies[obj.id],physicsClientId=world.client)):
                p.changeDynamics(world.bodies[obj.id],link,linearDamping=.9,angularDamping=.9,physicsClientId=world.client)
        corrections=resolve_overlaps(world)
        # Cropped images can omit support edges. Keep all declared table-supported
        # collision footprints on a finite slab, recording any inferred extension.
        required=np.array(world.scene.support.size,dtype=float)
        for obj in world.scene.objects:
            if obj.is_static or (obj.supported_by or 'support')!='support':continue
            lo,hi=aabb(world,obj.id)
            required=np.maximum(required,[2*max(abs(lo[0]),abs(hi[0]))+.004,2*max(abs(lo[1]),abs(hi[1]))+.004])
        original=np.array(world.scene.support.size)
        if extend_support and np.any(required>original+.001):
            result.support.size=tuple(required);world.scene.support.size=tuple(required)
            p.removeBody(world.bodies.pop('support'),physicsClientId=world.client)
            world._box('support',(required[0],world.scene.support.thickness,required[1]),
                (0,world.scene.support.height-world.scene.support.thickness/2,0),0)
        world.step(round(seconds*240))
        for obj in result.objects:
            if obj.is_static: continue
            pos,q=p.getBasePositionAndOrientation(world.bodies[obj.id],physicsClientId=world.client)
            from agatha.coordinates import quat_from_bullet
            obj.position=tuple(from_bullet(pos))
            obj.rotation_euler=tuple(p.getEulerFromQuaternion(quat_from_bullet(q)))
        metrics=contact_metrics(world)
        metrics['horizontal_corrections_m']=corrections
        metrics['inferred_support_extension_m']=(required-original).tolist() if extend_support else [0.,0.]
    return result,metrics


def stability(scene: Scene, seconds=5):
    from agatha.physics import World
    maximum=0.; per_object={o.id:0. for o in scene.objects if not o.is_static}
    with World(scene) as world:
        for _ in range(round(seconds*240)):
            world.step()
            for state in world.snapshot(0)['objects']:
                if state['id'] not in per_object: continue
                obj=next(o for o in scene.objects if o.id==state['id'])
                drift=float(np.linalg.norm(np.array(state['position'])-obj.position))
                per_object[obj.id]=max(per_object[obj.id],drift)
                maximum=max(maximum,drift)
        metrics=contact_metrics(world)
    return dict(duration_s=seconds,max_drift_m=maximum,per_object_max_drift_m=per_object,
                passed=maximum<.01,final_contacts=metrics)
