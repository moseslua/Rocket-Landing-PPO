"""Ideal variable-mass point dynamics in ENU; forces in N, time in s.

JIT with configuration closed over or passed as static arguments. No hidden RNG,
reset-at-done, actuator model, reward, or completed Gym environment.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from envs.common import Clock, Scenario, Termination, Vehicle

_DEFAULT_SCENARIO = Scenario()
_DEFAULT_VEHICLE = Vehicle()
_DEFAULT_CLOCK = Clock()


class State(NamedTuple):
    position: jax.Array
    velocity: jax.Array
    mass: jax.Array
    time: jax.Array
    reason: jax.Array
    mass_correction: jax.Array  # Kahan residual: exact mass is approximately mass - residual.
    time_correction: jax.Array


def reset(
    scenario=_DEFAULT_SCENARIO, vehicle=_DEFAULT_VEHICLE, clock=_DEFAULT_CLOCK, *, dtype=jnp.float32
):
    if jnp.dtype(dtype) == jnp.float64 and not jax.config.x64_enabled:
        raise ValueError("float64 reset requires JAX x64 enabled")
    if jnp.dtype(dtype) not in (jnp.dtype(jnp.float32), jnp.dtype(jnp.float64)):
        raise ValueError("use float32 or float64 state")
    if scenario.mass < vehicle.dry_mass:
        raise ValueError("initial mass is below dry mass")
    reason = (
        Termination.TOUCHDOWN
        if scenario.position[2] == 0
        else Termination.OUT_OF_FUEL
        if scenario.mass == vehicle.dry_mass
        else Termination.TIMEOUT
        if scenario.time >= clock.timeout
        else Termination.RUNNING
    )
    return State(
        jnp.asarray(scenario.position, dtype),
        jnp.asarray(scenario.velocity, dtype),
        jnp.asarray(scenario.mass, dtype),
        jnp.asarray(scenario.time, dtype),
        jnp.asarray(reason, jnp.int32),
        jnp.asarray(0, dtype),
        jnp.asarray(0, dtype),
    )


def sample_reset(key):
    """Developer cases only: z 100–200 m, vz -10–0 m/s, mass 1400–1600 kg.

    Split independently for each reset component. Future per-step stochastic
    effects must use fresh step keys; no disturbances are simulated here.
    """
    height_key, speed_key, mass_key = jax.random.split(key, 3)
    height = jax.random.uniform(height_key, (), minval=100.0, maxval=200.0)
    speed = jax.random.uniform(speed_key, (), minval=-10.0, maxval=0.0)
    mass = jax.random.uniform(mass_key, (), minval=1400.0, maxval=1600.0)
    return State(
        jnp.array([0, 0, height], dtype=jnp.float32),
        jnp.array([0, 0, speed], dtype=jnp.float32),
        mass.astype(jnp.float32),
        jnp.array(0, jnp.float32),
        jnp.array(Termination.RUNNING, jnp.int32),
        jnp.array(0, jnp.float32),
        jnp.array(0, jnp.float32),
    )


def _rhs(y, force, flow, vehicle):
    return jnp.concatenate(
        (
            y[3:6],
            force / y[6] + jnp.array([0, 0, -vehicle.gravity], dtype=y.dtype),
            jnp.reshape(-flow, (1,)),
        )
    )


def _rk4(y, dt, force, flow, vehicle):
    k1 = _rhs(y, force, flow, vehicle)
    k2 = _rhs(y + dt * k1 / 2, force, flow, vehicle)
    k3 = _rhs(y + dt * k2 / 2, force, flow, vehicle)
    k4 = _rhs(y + dt * k3, force, flow, vehicle)
    return y + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def _compensated_add(value, correction, increment):
    adjusted = increment - correction
    result = value + adjusted
    return result, (result - value) - adjusted


def step(state, thrust, vehicle=_DEFAULT_VEHICLE, clock=_DEFAULT_CLOCK):
    """Hold a clipped inertial thrust vector for one policy interval.

    Input state/thrust must be finite, valid, and shaped (3,). This is a low-level
    JIT core; eager reset validates scenarios, not traced inputs at every step.
    """
    force = jnp.asarray(thrust, dtype=state.mass.dtype)
    if force.shape != (3,):
        raise ValueError("thrust must have shape (3,)")
    magnitude = jnp.linalg.norm(force)
    force = force * jnp.minimum(1.0, vehicle.max_thrust / jnp.maximum(magnitude, 1e-12))
    flow = jnp.linalg.norm(force) / (vehicle.isp * vehicle.g0)

    def advance(current):
        y = jnp.concatenate((current.position, current.velocity, current.mass[None]))
        fuel_dt = jnp.where(
            flow > 0, (current.mass - vehicle.dry_mass - current.mass_correction) / flow, jnp.inf
        )
        time_dt = clock.timeout - current.time + current.time_correction
        dt = jnp.maximum(0, jnp.minimum(clock.physics_dt, jnp.minimum(fuel_dt, time_dt)))
        candidate = _rk4(y, dt, force, flow, vehicle)

        # With constant force, upward acceleration grows as mass burns. A
        # descending-to-ascending turn can hide two ground crossings in one step.
        def turning_time(_):
            def bisect_velocity(_i, bounds):
                low, high = bounds
                middle = (low + high) / 2
                descending = _rk4(y, middle, force, flow, vehicle)[5] < 0
                return jnp.where(descending, middle, low), jnp.where(descending, high, middle)

            _, minimum_dt = jax.lax.fori_loop(0, 24, bisect_velocity, (jnp.zeros_like(dt), dt))
            return minimum_dt

        bracket_end = jax.lax.cond(
            (y[5] < 0) & (candidate[5] > 0), turning_time, lambda _: dt, None
        )
        minimum_height = _rk4(y, bracket_end, force, flow, vehicle)[2]
        contact = minimum_height <= 0

        def localize(_):
            def bisect(_i, bounds):
                low, high = bounds
                middle = (low + high) / 2
                above = _rk4(y, middle, force, flow, vehicle)[2] > 0
                return jnp.where(above, middle, low), jnp.where(above, high, middle)

            _, hit_dt = jax.lax.fori_loop(0, 24, bisect, (jnp.zeros_like(dt), bracket_end))
            return _rk4(y, hit_dt, force, flow, vehicle).at[2].set(0), hit_dt

        candidate, elapsed = jax.lax.cond(
            contact,
            localize,
            lambda _: (candidate, dt),
            operand=None,
        )
        # Resolve numerically equal event horizons within a few time ulps.
        tie_tolerance = 4 * jnp.finfo(dt.dtype).eps * clock.physics_dt
        empty = fuel_dt <= dt + tie_tolerance
        expired = time_dt <= dt + tie_tolerance
        reason = jnp.where(
            contact,
            Termination.TOUCHDOWN,
            jnp.where(
                empty,
                Termination.OUT_OF_FUEL,
                jnp.where(expired, Termination.TIMEOUT, Termination.RUNNING),
            ),
        )
        mass, mass_correction = _compensated_add(
            current.mass, current.mass_correction, -flow * elapsed
        )
        time, time_correction = _compensated_add(current.time, current.time_correction, elapsed)
        fuel_boundary = empty & (~contact | (jnp.abs(elapsed - fuel_dt) <= tie_tolerance))
        time_boundary = expired & (~contact | (jnp.abs(elapsed - time_dt) <= tie_tolerance))
        mass = jnp.where(fuel_boundary, vehicle.dry_mass, mass)
        time = jnp.where(time_boundary, clock.timeout, time)
        mass_correction = jnp.where(fuel_boundary, 0, mass_correction)
        time_correction = jnp.where(time_boundary, 0, time_correction)
        return State(
            candidate[:3],
            candidate[3:6],
            mass,
            time,
            reason.astype(jnp.int32),
            mass_correction,
            time_correction,
        )

    def substep(current, _):
        next_state = jax.lax.cond(current.reason == 0, advance, lambda s: s, current)
        return next_state, None

    final, _ = jax.lax.scan(substep, state, None, length=clock.substeps)
    return final


def rollout(initial, thrusts, vehicle=_DEFAULT_VEHICLE, clock=_DEFAULT_CLOCK):
    """Return final state and post-action states. Terminal states remain frozen."""

    def advance(state, thrust):
        next_state = step(state, thrust, vehicle, clock)
        return next_state, next_state

    return jax.lax.scan(advance, initial, thrusts)


def batch_rollout(initial, thrusts, vehicle=_DEFAULT_VEHICLE, clock=_DEFAULT_CLOCK):
    """Leading axes: initial [env, ...], thrusts [env, policy_step, 3]."""
    return jax.vmap(lambda state, actions: rollout(state, actions, vehicle, clock))(
        initial, thrusts
    )
