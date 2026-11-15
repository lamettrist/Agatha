"""API coordinates: meters, kilograms, seconds, radians; right-handed Y-up.

Euler angles use fixed XYZ (quaternion convention xyzw). Support height is its top.
All actions are applied once at t=0 in list order.
"""
from typing import Annotated, Literal
import math
from pydantic import BaseModel, ConfigDict, Field, model_validator

Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]
Positive = Annotated[float, Field(gt=0, le=100)]

class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

class CollisionPart(StrictModel):
    type: Literal["box", "cylinder", "convex_hull"] = "box"
    size: tuple[Positive, Positive, Positive] | None = None
    radius: Positive | None = None
    height: Positive | None = None
    vertices: list[Vec3] = Field(default_factory=list, max_length=4096)
    position: Vec3 = (0, 0, 0)
    rotation_euler: Vec3 = (0, 0, 0)

    @model_validator(mode="after")
    def geometry(self):
        if self.type == "box" and self.size is None:
            raise ValueError("box collision requires size")
        if self.type == "cylinder" and (self.radius is None or self.height is None):
            raise ValueError("cylinder collision requires radius and height")
        if self.type == "convex_hull" and len(self.vertices) < 4:
            raise ValueError("convex hull requires at least four vertices")
        return self

class Collision(StrictModel):
    parts: list[CollisionPart] = Field(min_length=1, max_length=64)

class SceneObject(StrictModel):
    id: str = Field(min_length=1, max_length=80)
    label: str = Field(min_length=1, max_length=120)
    shape: Literal["box", "cylinder", "compound", "convex_hull"] = "box"
    collision: Collision | None = None
    shape_method: str = "legacy_box"
    visual_mesh_url: str | None = None
    scale: Vec3 = (1, 1, 1)
    supported_by: str | None = None
    soft_approx: bool = False
    restitution: float = Field(default=0, ge=0, le=1)
    physical_properties_estimated: bool = True
    linear_damping: float = Field(default=.04, ge=0, le=1)
    angular_damping: float = Field(default=.04, ge=0, le=1)
    size: tuple[Positive, Positive, Positive]
    position: Vec3
    rotation_euler: Vec3 = (0, 0, 0)
    mass_kg: float = Field(default=0.5, gt=0, le=1000)
    mass_source: Literal["estimated", "exa", "user"] = "estimated"
    friction: float = Field(default=0.5, ge=0, le=2)
    is_static: bool = False
    confidence: float = Field(default=1, ge=0, le=1)

class Support(StrictModel):
    thickness: float = Field(default=.08, gt=0, le=.5)
    collisions_enabled: bool = True
    size: tuple[Positive, Positive] = (1.2, 0.8)  # width, depth
    height: float = Field(default=0.75, ge=0, le=5)

class Scene(StrictModel):
    objects: list[SceneObject] = Field(max_length=256)
    support: Support = Field(default_factory=Support)
    @model_validator(mode="after")
    def unique_ids(self):
        ids = [o.id for o in self.objects]
        if len(ids) != len(set(ids)) or set(ids) & {"support", "floor"}:
            raise ValueError("Object IDs must be unique and cannot be support/floor")
        return self

class Push(StrictModel):
    type: Literal["push"]
    id: str
    direction: tuple[float, float] | Vec3
    impulse_Ns: float = Field(gt=0, le=20)
    @model_validator(mode="after")
    def nonzero(self):
        if math.hypot(*self.direction) < 1e-8:
            raise ValueError("push direction must be nonzero")
        return self
class Remove(StrictModel):
    type: Literal["remove"]
    id: str
class SetMass(StrictModel):
    type: Literal["set_mass"]
    id: str
    kg: float = Field(gt=0, le=1000)
class Tilt(StrictModel):
    type: Literal["tilt_support"]
    angle_deg: float = Field(ge=-60, le=60)
    axis: Literal["x", "z"]
class Gravity(StrictModel):
    type: Literal["set_gravity"]
    g: float = Field(ge=0, le=30)
class SupportCollisions(StrictModel):
    type: Literal['set_support_collisions']
    enabled: bool

Action = Annotated[Push | Remove | SetMass | Tilt | Gravity | SupportCollisions, Field(discriminator="type")]

class SimulationRequest(StrictModel):
    scene_id: str
    actions: list[Action] = Field(default_factory=list, max_length=20)
    engine: Literal["physics", "learned", "both"] = "both"
    learned_architecture: Literal['default','mlp_multistep','graph_impulse','contact_gnn'] = 'default'
    duration: float = Field(default=3, ge=0.1, le=5)
    fps: Literal[10, 20, 30] = 30
