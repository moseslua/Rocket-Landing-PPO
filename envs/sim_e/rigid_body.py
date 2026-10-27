# ruff: noqa: B008 -- immutable dataclass configs are static JAX/API defaults
"""Independent NumPy/SciPy rigid-body numerical reference, not full Sim-E yet."""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation

from envs.common.rigid import RigidClock, RigidScenario, RigidState, RigidVehicle


def reset(scenario=RigidScenario()):
    vector = np.array(
        (
            *scenario.position,
            *scenario.velocity,
            *scenario.quaternion,
            *scenario.angular_rate,
            scenario.mass,
        ),
        dtype=np.float64,
    )
    vector[6:10] /= np.linalg.norm(vector[6:10])
    return RigidState(
        vector, np.array(scenario.gimbal), scenario.throttle, 0, 0, np.array(scenario.wind), 0
    )


def quaternion_rotate(q, v):
    uv = np.cross(q[1:], v)
    return v + 2 * (q[0] * uv + np.cross(q[1:], uv))


def rhs(_time, vector, thrust_body, torque, vehicle):
    q, omega, mass = vector[6:10], vector[10:13], vector[13]
    acceleration = quaternion_rotate(q, thrust_body) / mass + [0, 0, -vehicle.gravity]
    torque = np.cross([0, 0, -vehicle.lever_arm], thrust_body) + [0, 0, torque]
    omega_dot = (torque - np.cross(omega, vehicle.inertia * omega)) / vehicle.inertia
    q_dot = 0.5 * np.r_[-q[1:] @ omega, q[0] * omega + np.cross(q[1:], omega)]
    return np.r_[
        vector[3:6],
        acceleration,
        q_dot,
        omega_dot,
        -np.linalg.norm(thrust_body) / (vehicle.isp * vehicle.g0),
    ]


def step(state, action, vehicle=RigidVehicle(), clock=RigidClock(), *, method="RK45"):
    if state.reason != 0:
        return state
    throttle = vehicle.min_throttle + (1 - vehicle.min_throttle) * (action[0] + 1) / 2
    gimbal = action[1:3] * vehicle.gimbal_limit
    thrust_body = (
        np.array([np.sin(gimbal[1]), -np.sin(gimbal[0]), np.cos(gimbal[0]) * np.cos(gimbal[1])])
        * throttle
        * vehicle.max_thrust
    )
    vector = state.vector.copy()
    time = state.time
    for _ in range(clock.substeps):
        result = solve_ivp(
            rhs,
            (0, clock.physics_dt),
            vector,
            args=(thrust_body, action[3] * vehicle.rcs_roll_torque, vehicle),
            method=method,
            rtol=1e-9,
            atol=1e-11,
        )
        vector = result.y[:, -1]
        vector[6:10] /= np.linalg.norm(vector[6:10])
        time += clock.physics_dt
        if vector[2] <= 0 or vector[13] <= vehicle.dry_mass:
            break
    vector[2] = max(vector[2], 0)
    tilt = np.arccos(np.clip(quaternion_rotate(vector[6:10], [0, 0, 1])[2], -1, 1))
    bounds = np.any(np.abs(vector[:2]) > 1500) or tilt > np.deg2rad(60)
    reason = (
        1
        if vector[2] <= 0
        else 3
        if vector[13] <= vehicle.dry_mass
        else 2
        if bounds
        else 4
        if time >= clock.timeout
        else 0
    )
    return state._replace(vector=vector, gimbal=gimbal, throttle=throttle, time=time, reason=reason)


def attitude_only(quaternion, omega, duration=1.0):
    """Analytic torque-free body-z spin for rotation-invariant unit tests."""
    rotation = Rotation.from_quat(quaternion[[1, 2, 3, 0]])
    return (rotation * Rotation.from_rotvec([0, 0, omega[2] * duration])).as_quat()[[3, 0, 1, 2]]
