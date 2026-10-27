# ruff: noqa: B008 -- immutable dataclass configs are static API defaults
"""Independent higher-fidelity development model with explicit touchdown checks."""

import numpy as np
from scipy.integrate import solve_ivp

from envs.common.rigid import RigidClock, RigidScenario, RigidVehicle
from envs.sim_e.rigid_body import reset as nominal_reset

SIM_E_HIGH_FIDELITY_VERSION = "independent-development-0.1.0"


class Effects:
    def __init__(self, slosh_mass=25.0, slosh_length=1.0, seed=0):
        self.slosh_mass, self.slosh_length = slosh_mass, slosh_length
        self.rng = np.random.default_rng(seed)


def reset(scenario=RigidScenario()):
    return nominal_reset(scenario)


def _rotate(q, vector):
    uv = np.cross(q[1:], vector)
    return vector + 2 * (q[0] * uv + np.cross(q[1:], uv))


def rhs(_t, x, command, vehicle, effects, wind):
    q, omega, mass = x[6:10], x[10:13], x[13]
    slosh, slosh_rate, throttle, throttle_rate = x[14:18]
    throttle_accel = (command[0] - throttle - 2 * 0.7 * 0.08 * throttle_rate) / 0.08**2
    isp = vehicle.isp * (1 - 0.08 * (1 - throttle))
    thrust = vehicle.max_thrust * np.clip(throttle, 0, 1)
    gimbal = command[1:3]
    body_thrust = (
        np.array([np.sin(gimbal[1]), -np.sin(gimbal[0]), np.cos(gimbal[0]) * np.cos(gimbal[1])])
        * thrust
    )
    lever = vehicle.lever_arm * (1 + 0.15 * (mass - vehicle.dry_mass) / max(mass, 1))
    torque = np.cross([0, 0, -lever], body_thrust) + [0, 0, command[3] * vehicle.rcs_roll_torque]
    slosh_accel = -9.81 / effects.slosh_length * slosh - 0.2 * slosh_rate
    acceleration = (
        (_rotate(q, body_thrust) + [effects.slosh_mass * slosh_accel, 0, 0]) / mass
        + wind
        + [0, 0, -vehicle.gravity]
    )
    inertia = np.array(vehicle.inertia) * (0.8 + 0.2 * mass / max(mass, vehicle.dry_mass))
    qdot = 0.5 * np.r_[-q[1:] @ omega, q[0] * omega + np.cross(q[1:], omega)]
    omega_dot = (torque - np.cross(omega, inertia * omega)) / inertia
    return np.r_[
        x[3:6],
        acceleration,
        qdot,
        omega_dot,
        -thrust / (isp * vehicle.g0),
        slosh_rate,
        slosh_accel,
        throttle_rate,
        throttle_accel,
    ]


def touchdown_reason(vector):
    q = vector[6:10]
    up = _rotate(q, [0, 0, 1])
    tilt = np.arccos(np.clip(up[2], -1, 1))
    soft = (
        abs(vector[5]) <= 2
        and np.linalg.norm(vector[3:5]) <= 1
        and tilt <= np.deg2rad(5)
        and np.linalg.norm(vector[10:13]) <= np.deg2rad(5)
        and np.linalg.norm(vector[:2]) <= 5
    )
    return 1 if soft else 2


def integrate(state, action, vehicle=RigidVehicle(), clock=RigidClock(), effects=None):
    if state.reason != 0:
        return state
    effects = effects or Effects()
    command = np.r_[
        vehicle.min_throttle + (1 - vehicle.min_throttle) * (action[0] + 1) / 2,
        action[1:3] * vehicle.gimbal_limit,
        action[3],
    ]
    x0 = np.r_[state.vector, 0.0, 0.0, state.throttle, 0.0]

    def event(_t, x, *_args):
        return x[2]

    event.terminal, event.direction = True, -1
    wind = state.wind + 0.01 * effects.rng.normal(size=3)
    horizon = min(clock.policy_dt, clock.timeout - state.time)
    result = solve_ivp(
        rhs,
        (0, horizon),
        x0,
        args=(command, vehicle, effects, wind),
        method="DOP853",
        rtol=1e-8,
        atol=1e-10,
        events=event,
    )
    if not result.success:
        raise RuntimeError(result.message)
    x = result.y[:, -1]
    vector = x[:14]
    vector[6:10] /= np.linalg.norm(vector[6:10])
    landed = x[2] <= 0
    empty = vector[13] <= vehicle.dry_mass
    reason = (
        touchdown_reason(vector)
        if landed
        else 3
        if empty
        else 4
        if state.time + result.t[-1] >= clock.timeout
        else 0
    )
    vector[2] = max(0, vector[2])
    return state._replace(
        vector=vector, throttle=x[16], time=state.time + result.t[-1], reason=reason
    )
