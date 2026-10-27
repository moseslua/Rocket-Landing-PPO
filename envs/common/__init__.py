"""Shared declarative units/configuration only; dynamics stay independent."""

import math
from dataclasses import dataclass
from enum import IntEnum


class Termination(IntEnum):
    RUNNING = 0
    TOUCHDOWN = 1  # Not a success label: attitude is not modeled in this prototype.
    OUT_OF_FUEL = 2
    TIMEOUT = 3


@dataclass(frozen=True)
class Vehicle:
    dry_mass: float = 1000.0
    max_thrust: float = 30000.0
    isp: float = 300.0
    gravity: float = 9.81
    g0: float = 9.80665

    def __post_init__(self):
        for name in ("dry_mass", "max_thrust", "isp", "g0"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.gravity) or self.gravity < 0:
            raise ValueError("gravity must be finite and nonnegative")


@dataclass(frozen=True)
class Clock:
    physics_dt: float = 0.005
    substeps: int = 10
    timeout: float = 60.0

    def __post_init__(self):
        if not isinstance(self.substeps, int) or self.substeps < 1:
            raise ValueError("substeps must be a positive integer")
        for name in ("physics_dt", "timeout"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")

    @property
    def policy_dt(self) -> float:
        return self.physics_dt * self.substeps


@dataclass(frozen=True)
class Scenario:
    position: tuple[float, float, float] = (0.0, 0.0, 100.0)
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    mass: float = 1500.0
    time: float = 0.0

    def __post_init__(self):
        for name in ("position", "velocity"):
            values = getattr(self, name)
            if len(values) != 3 or not all(math.isfinite(v) for v in values):
                raise ValueError(f"{name} must contain three finite ENU components")
        if self.position[2] < 0:
            raise ValueError("initial altitude must be nonnegative")
        if not math.isfinite(self.mass) or self.mass <= 0:
            raise ValueError("mass must be finite and positive")
        if not math.isfinite(self.time) or self.time < 0:
            raise ValueError("time must be finite and nonnegative")
