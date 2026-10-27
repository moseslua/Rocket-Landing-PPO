# ruff: noqa: B008 -- immutable dataclass configs are static JAX/API defaults
"""Receding-horizon convex translational guidance + quaternion attitude control.

This is a finite-horizon baseline, not a certified lossless-convexification
powered-descent solver. It returns normalized actuator commands for Sim-T.
"""

import contextlib
from dataclasses import dataclass

import cvxpy as cp
import numpy as np


@dataclass(frozen=True)
class GuidanceConfig:
    horizon: int = 12
    dt: float = 0.2
    max_thrust: float = 36000.0
    gravity: float = 9.81
    thrust_weight: float = 0.0001
    terminal_weight: float = 10.0
    kp: float = 2.0
    kd: float = 1.0


class GuidanceController:
    def __init__(self, config=GuidanceConfig()):
        self.config = config
        if config.horizon < 1 or config.dt <= 0:
            raise ValueError("horizon and dt must be positive")
        self.position = cp.Variable((config.horizon + 1, 3))
        self.velocity = cp.Variable((config.horizon + 1, 3))
        self.thrust = cp.Variable((config.horizon, 3))
        self.initial_position = cp.Parameter(3)
        self.initial_velocity = cp.Parameter(3)
        self.initial_mass = cp.Parameter(nonneg=True)
        constraints = [
            self.position[0] == self.initial_position,
            self.velocity[0] == self.initial_velocity,
        ]
        for k in range(config.horizon):
            acceleration = self.thrust[k] / self.initial_mass + np.array(
                [0.0, 0.0, -config.gravity]
            )
            constraints += [
                self.position[k + 1]
                == self.position[k]
                + self.config.dt * self.velocity[k]
                + 0.5 * self.config.dt**2 * acceleration,
                self.velocity[k + 1] == self.velocity[k] + self.config.dt * acceleration,
                cp.norm(self.thrust[k]) <= config.max_thrust,
            ]
        objective = cp.Minimize(
            cp.sum_squares(self.position[-1]) * config.terminal_weight
            + cp.sum_squares(self.velocity[-1]) * config.terminal_weight
            + config.thrust_weight * cp.sum_squares(self.thrust)
        )
        self.problem = cp.Problem(objective, constraints)

    def desired_thrust(self, position, velocity, mass):
        self.initial_position.value = np.asarray(position, float)
        self.initial_velocity.value = np.asarray(velocity, float)
        self.initial_mass.value = float(mass)
        with contextlib.suppress(cp.error.SolverError):
            self.problem.solve(solver=cp.CLARABEL, warm_start=True)
        if self.thrust.value is not None and self.problem.status in (
            cp.OPTIMAL,
            cp.OPTIMAL_INACCURATE,
        ):
            return np.asarray(self.thrust.value[0])
        return np.array([0.0, 0.0, self.initial_mass.value * self.config.gravity])

    def attitude_action(self, quaternion, angular_rate, desired_thrust, vehicle=None):
        """Map desired ENU thrust to throttle/gimbal/RCS with quaternion tracking."""
        q = np.asarray(quaternion, float)
        q /= np.linalg.norm(q)
        desired = np.asarray(desired_thrust, float)
        norm = np.linalg.norm(desired)
        if norm < 1e-8:
            desired = np.array([0.0, 0.0, 1.0])
        else:
            desired /= norm
        uv = np.cross(q[1:], [0.0, 0.0, 1.0])
        up = np.array([0.0, 0.0, 1.0]) + 2 * (q[0] * uv + np.cross(q[1:], uv))
        error = np.cross(up, desired)
        omega = np.asarray(angular_rate, float)
        # First-order gimbal command in the local body frame.
        gimbal = np.array([error[1], -error[0]]) * self.config.kp - omega[:2] * self.config.kd
        gimbal = np.clip(gimbal, -0.104719755, 0.104719755)
        throttle = np.clip(norm / self.config.max_thrust, 0.4, 1.0)
        normalized_throttle = 2 * (throttle - 0.4) / 0.6 - 1
        roll = np.clip(-self.config.kd * omega[2], -1, 1)
        return np.array(
            [normalized_throttle, gimbal[0] / 0.104719755, gimbal[1] / 0.104719755, roll],
            dtype=np.float32,
        )

    def act(self, state, vehicle=None):
        vector = np.asarray(state.vector if hasattr(state, "vector") else state)
        desired = self.desired_thrust(vector[:3], vector[3:6], vector[13])
        return self.attitude_action(vector[6:10], vector[10:13], desired, vehicle)
