"""Proper rotation Rx(pi/2): API (x,y,z) -> Bullet (x,-z,y).

A reflection (x,z,y) would reverse handedness and is deliberately avoided.
"""
from collections.abc import Sequence
import pybullet as p

def to_bullet(v: Sequence[float]) -> tuple[float, float, float]:
    return v[0], -v[2], v[1]
def from_bullet(v: Sequence[float]) -> tuple[float, float, float]:
    return v[0], v[2], -v[1]
def quat_to_bullet(q: Sequence[float]) -> tuple[float, float, float, float]:
    return q[0], -q[2], q[1], q[3]
def quat_from_bullet(q: Sequence[float]) -> tuple[float, float, float, float]:
    return q[0], q[2], -q[1], q[3]
def euler_to_bullet(e: Sequence[float]):
    return quat_to_bullet(p.getQuaternionFromEuler(e))
