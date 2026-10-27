# ruff: noqa: B008 -- immutable dataclass configs are static JAX/API defaults
"""Ideal 3-DOF environment dynamics with actuator/throttle mapping."""

from dataclasses import dataclass

# Immutable configuration defaults are safe to share between JIT calls.
# Ruff B008: frozen dataclasses intentionally form part of the static JIT signature.
import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class Vehicle:
    dry_mass: float = 1000.0
    max_thrust: float = 30000.0
    isp: float = 300.0
    g0: float = 9.80665
    gravity: float = 9.81
    min_throttle: float = 0.4
    max_tilt: float = 0.35
    time_constant: float = 0.1


_DEFAULT_VEHICLE = Vehicle()


_DEFAULT_VEHICLE = Vehicle()


class State:
    def __init__(self, vector, time, previous_action, reason, key, wind, actual_throttle=0.0):
        self.vector, self.time, self.previous_action = vector, time, previous_action
        self.reason, self.key, self.wind = reason, key, wind
        self.actual_throttle = actual_throttle


def reset(initial, key):
    vector = jnp.asarray(initial, dtype=jnp.float32)
    return State(
        vector,
        jnp.array(0.0, dtype=jnp.float32),
        jnp.zeros(3, dtype=jnp.float32),
        jnp.array(0, dtype=jnp.int32),
        key,
        jnp.zeros(3, dtype=jnp.float32),
        0.0,
    )


def action_force(action, vehicle=_DEFAULT_VEHICLE):
    throttle = vehicle.min_throttle + (1 - vehicle.min_throttle) * (action[0] + 1) / 2
    direction = jnp.array([action[1] * vehicle.max_tilt, action[2] * vehicle.max_tilt, 1.0])
    return direction / jnp.linalg.norm(direction) * throttle * vehicle.max_thrust


def step(state, action, vehicle=_DEFAULT_VEHICLE, dt=0.005, substeps=10, timeout=60.0):
    action = jnp.clip(action, -1, 1)
    force = action_force(action, vehicle)
    thrust = force[2]
    mdot = -jnp.linalg.norm(force) / (vehicle.isp * vehicle.g0)

    def rhs(y):
        return jnp.concatenate(
            (y[3:6], force / y[6] + jnp.array([0.0, 0.0, -vehicle.gravity]), jnp.atleast_1d(mdot))
        )

    def integrate(y, _):
        k1 = rhs(y)
        k2 = rhs(y + k1 * dt / 2)
        k3 = rhs(y + k2 * dt / 2)
        k4 = rhs(y + k3 * dt)
        out = y + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        return out, None

    vector, _ = jax.lax.scan(integrate, state.vector, None, length=substeps)
    time = state.time + dt * substeps
    landed = vector[2] <= 0
    soft = (jnp.abs(vector[3]) <= 1) & (jnp.abs(vector[5]) <= 2)
    soft = soft & (jnp.linalg.norm(vector[:2]) <= 5)
    reason = jnp.where(
        landed,
        jnp.where(soft, 1, 2),
        jnp.where(time >= timeout, 4, jnp.where(vector[6] <= vehicle.dry_mass, 3, 0)),
    )
    vector = vector.at[2].set(jnp.maximum(vector[2], 0))
    return State(vector, time, action, reason.astype(jnp.int32), state.key, state.wind, thrust)
