# ruff: noqa: B008 -- immutable dataclass configs are static JAX/API defaults
"""JAX 6-DOF rigid-body core with 14-value ENU state.

Quaternion is scalar-first and rotates body vectors into ENU. Quaternion and
corresponding angular-velocity dynamics are frame-convention paired.
"""

from typing import NamedTuple

# Immutable configuration defaults are safe to share between JIT calls.
# Ruff B008: frozen dataclasses intentionally form part of the static JIT signature.
import jax
import jax.numpy as jnp
import numpy as np

from envs.common.rigid import RigidClock, RigidScenario, RigidState, RigidVehicle

_SIM_VERSION = "6dof-training-development-0.1.0"


_DEFAULT_VEHICLE = RigidVehicle()
_DEFAULT_CLOCK = RigidClock()
_DEFAULT_SCENARIO = RigidScenario()


class Command(NamedTuple):
    throttle: jax.Array
    gimbal: jax.Array
    rcs_roll: jax.Array


def quaternion_normalize(q):
    return q / jnp.maximum(jnp.linalg.norm(q), 1e-12)


def quaternion_rotate(q, vector):
    uv = jnp.cross(q[1:], vector)
    uuv = jnp.cross(q[1:], uv)
    return vector + 2 * (q[0] * uv + uuv)


def body_up(q):
    return quaternion_rotate(q, jnp.array([0.0, 0.0, 1.0], dtype=q.dtype))


def tilt(q):
    return jnp.arccos(jnp.clip(body_up(q)[2], -1, 1))


def quaternion_derivative(q, omega):
    scalar = -0.5 * jnp.dot(q[1:], omega)
    vector = 0.5 * (q[0] * omega + jnp.cross(q[1:], omega))
    return jnp.concatenate((jnp.atleast_1d(scalar), vector))


def command_from_action(action, vehicle=_DEFAULT_VEHICLE):
    throttle = vehicle.min_throttle + (1 - vehicle.min_throttle) * (action[0] + 1) / 2
    return Command(
        throttle, action[1:3] * vehicle.gimbal_limit, action[3] * vehicle.rcs_roll_torque
    )


def reset(scenario=_DEFAULT_SCENARIO, vehicle=_DEFAULT_VEHICLE, clock=_DEFAULT_CLOCK, *, key=None):
    if scenario.mass < vehicle.dry_mass:
        raise ValueError("initial mass must be at least dry mass")
    vector = jnp.asarray(
        (
            *scenario.position,
            *scenario.velocity,
            *scenario.quaternion,
            *scenario.angular_rate,
            scenario.mass,
        ),
        jnp.float32,
    )
    q = vector[6:10]
    vector = vector.at[6:10].set(quaternion_normalize(q))
    return RigidState(
        vector,
        jnp.asarray(scenario.gimbal),
        jnp.asarray(scenario.throttle),
        jnp.asarray(0, jnp.float32),
        jnp.asarray(0, jnp.int32),
        jnp.asarray(scenario.wind),
        jax.random.PRNGKey(0) if key is None else key,
        0,
        0,
    )


def rhs(vector, thrust_body, roll_torque, vehicle=_DEFAULT_VEHICLE):
    q, omega, mass = vector[6:10], vector[10:13], vector[13]
    thrust_inertial = quaternion_rotate(q, thrust_body)
    acceleration = thrust_inertial / mass + jnp.array([0, 0, -vehicle.gravity], vector.dtype)
    torque = jnp.cross(jnp.array([0, 0, -vehicle.lever_arm], vector.dtype), thrust_body)
    torque = torque + jnp.array([0, 0, roll_torque], vector.dtype)
    inertia = jnp.asarray(vehicle.inertia, dtype=vector.dtype)
    omega_dot = (torque - jnp.cross(omega, inertia * omega)) / inertia
    mdot = -jnp.linalg.norm(thrust_body) / (vehicle.isp * vehicle.g0)
    return jnp.concatenate(
        (
            vector[3:6],
            acceleration,
            quaternion_derivative(q, omega),
            omega_dot,
            jnp.atleast_1d(mdot),
        )
    )


def rk4(vector, dt, thrust_body, torque, vehicle=_DEFAULT_VEHICLE):
    k1 = rhs(vector, thrust_body, torque, vehicle)
    k2 = rhs(vector + k1 * dt / 2, thrust_body, torque, vehicle)
    k3 = rhs(vector + k2 * dt / 2, thrust_body, torque, vehicle)
    k4 = rhs(vector + k3 * dt, thrust_body, torque, vehicle)
    next_vector = vector + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return next_vector.at[6:10].set(quaternion_normalize(next_vector[6:10]))


def touchdown_reason(state):
    vector = state.vector
    soft = (
        (jnp.abs(vector[3]) <= 1)
        & (jnp.abs(vector[5]) <= 2)
        & (tilt(vector[6:10]) <= np.deg2rad(5))
    )
    soft = soft & (jnp.linalg.norm(vector[10:13]) <= np.deg2rad(5))
    soft = soft & (jnp.linalg.norm(vector[:2]) <= 5)
    return jnp.where(soft, 1, 2).astype(jnp.int32)


def step(state, action, vehicle=_DEFAULT_VEHICLE, clock=_DEFAULT_CLOCK):
    """One 20 Hz action, zero-order held at 200 Hz. State is frozen after termination."""
    command = command_from_action(action, vehicle)

    def advance(current):
        dt = jnp.minimum(clock.physics_dt, clock.timeout - current.time)
        gimbal_delta = jnp.clip(
            command.gimbal - current.gimbal,
            -vehicle.gimbal_rate_limit * dt,
            vehicle.gimbal_rate_limit * dt,
        )
        gimbal = current.gimbal + gimbal_delta
        throttle = current.throttle + jnp.clip(
            command.throttle - current.throttle,
            -dt / vehicle.engine_time_constant,
            dt / vehicle.engine_time_constant,
        )
        thrust_body = (
            jnp.array(
                [jnp.sin(gimbal[1]), -jnp.sin(gimbal[0]), jnp.cos(gimbal[0]) * jnp.cos(gimbal[1])]
            )
            * throttle
            * vehicle.max_thrust
        )
        vector = rk4(current.vector, dt, thrust_body, command.rcs_roll, vehicle)
        vector = vector.at[3:6].add(current.wind * 0.01 * dt)
        landing = vector[2] <= 0
        bounds = (
            (jnp.abs(vector[:2]) > 1500).any()
            | (vector[2] > 5000)
            | (tilt(vector[6:10]) > np.deg2rad(60))
        )
        empty = vector[13] <= vehicle.dry_mass
        expired = current.time + dt >= clock.timeout
        reason = jnp.where(
            landing,
            touchdown_reason(state._replace(vector=vector.at[2].set(0))),
            jnp.where(bounds, 2, jnp.where(empty, 3, jnp.where(expired, 4, 0))),
        )
        return current._replace(
            vector=vector.at[2].set(jnp.maximum(vector[2], 0)),
            gimbal=gimbal,
            throttle=throttle,
            time=current.time + dt,
            reason=reason.astype(jnp.int32),
        )

    def substep(current, _):
        next_state = jax.lax.cond(current.reason == 0, advance, lambda s: s, current)
        return next_state, None

    final, _ = jax.lax.scan(substep, state, None, length=clock.substeps)
    return final


def rollout(initial, actions, vehicle=_DEFAULT_VEHICLE, clock=_DEFAULT_CLOCK):
    def advance(state, action):
        next_state = step(state, action, vehicle, clock)
        return next_state, next_state

    return jax.lax.scan(advance, initial, actions)


def batch_rollout(initial, actions, vehicle=_DEFAULT_VEHICLE, clock=_DEFAULT_CLOCK):
    return jax.vmap(lambda state, action: rollout(state, action, vehicle, clock))(initial, actions)
