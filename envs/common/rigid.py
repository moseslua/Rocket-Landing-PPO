"""ENU rigid-body units and vehicle parameters; no backend dynamics."""

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np


@dataclass(frozen=True)
class RigidVehicle:
    dry_mass: float = 1200.0
    inertia: tuple[float, float, float] = (4000.0, 4000.0, 600.0)
    max_thrust: float = 36000.0
    isp: float = 300.0
    g0: float = 9.80665
    gravity: float = 9.81
    lever_arm: float = 2.0
    gimbal_limit: float = np.deg2rad(6)
    gimbal_rate_limit: float = np.deg2rad(30)
    rcs_roll_torque: float = 300.0
    min_throttle: float = 0.4
    engine_time_constant: float = 0.1
    drag_area: float = 0.5

    def __post_init__(self):
        for name in (
            "dry_mass",
            "max_thrust",
            "isp",
            "g0",
            "lever_arm",
            "gimbal_rate_limit",
            "rcs_roll_torque",
            "engine_time_constant",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if min(self.inertia) <= 0 or self.gravity < 0:
            raise ValueError("inertia must be positive and gravity nonnegative")
        if not 0 < self.min_throttle <= 1 or not 0 < self.gimbal_limit < np.pi / 2:
            raise ValueError("invalid throttle or gimbal limits")
        if self.drag_area < 0:
            raise ValueError("drag area cannot be negative")


@dataclass(frozen=True)
class RigidClock:
    physics_dt: float = 0.005
    substeps: int = 10
    timeout: float = 60.0

    @property
    def policy_dt(self) -> float:
        return self.physics_dt * self.substeps


@dataclass(frozen=True)
class RigidScenario:
    position: tuple[float, float, float] = (0.0, 0.0, 1000.0)
    velocity: tuple[float, float, float] = (0.0, 0.0, -40.0)
    quaternion: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    angular_rate: tuple[float, float, float] = (0.0, 0.0, 0.0)
    mass: float = 1700.0
    wind: tuple[float, float, float] = (0.0, 0.0, 0.0)
    gimbal: tuple[float, float] = (0.0, 0.0)
    throttle: float = 1.0


class RigidState(NamedTuple):
    vector: object
    gimbal: object
    throttle: object
    time: object
    reason: object
    wind: object
    rng: object
    mass_correction: object = 0.0
    time_correction: object = 0.0


POSITION_SCALE = 2500.0
VELOCITY_SCALE = 100.0
RATE_SCALE = np.deg2rad(30)
MASS_SCALE = 2000.0
