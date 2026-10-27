"""Independent ideal 3-DOF reference written in NumPy/SciPy.

Only declarations are shared with Sim-T. No training-simulator numerical code is
imported. No slosh, Dryden wind, or other planned high-fidelity effects yet.
"""

from typing import NamedTuple

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from envs.common import Clock, Scenario, Termination, Vehicle

_DEFAULT_SCENARIO = Scenario()
_DEFAULT_VEHICLE = Vehicle()
_DEFAULT_CLOCK = Clock()


class State(NamedTuple):
    position: np.ndarray
    velocity: np.ndarray
    mass: float
    time: float
    reason: int


def reset(scenario=_DEFAULT_SCENARIO, vehicle=_DEFAULT_VEHICLE, clock=_DEFAULT_CLOCK):
    if scenario.mass < vehicle.dry_mass:
        raise ValueError("initial mass is below dry mass")
    if scenario.position[2] == 0:
        reason = Termination.TOUCHDOWN
    elif scenario.mass == vehicle.dry_mass:
        reason = Termination.OUT_OF_FUEL
    elif scenario.time >= clock.timeout:
        reason = Termination.TIMEOUT
    else:
        reason = Termination.RUNNING
    return State(
        np.array(scenario.position, dtype=np.float64),
        np.array(scenario.velocity, dtype=np.float64),
        scenario.mass,
        scenario.time,
        reason,
    )


def sample_reset(seed):
    """Independent sampler for the same developer ranges, not a frozen test set."""
    rng = np.random.default_rng(seed)
    return reset(
        Scenario(
            position=(0, 0, rng.uniform(100, 200)),
            velocity=(0, 0, rng.uniform(-10, 0)),
            mass=rng.uniform(1400, 1600),
        )
    )


def step(
    state,
    thrust,
    vehicle=_DEFAULT_VEHICLE,
    clock=_DEFAULT_CLOCK,
    *,
    method="RK45",
    rtol=1e-9,
    atol=1e-11,
):
    """Adaptive integration over one constant-thrust interval with terminal events."""
    return _integrate(state, thrust, vehicle, clock, method, rtol, atol)[0]


def _integrate(state, thrust, vehicle, clock, method, rtol, atol):
    thrust = np.asarray(thrust, dtype=np.float64)
    if thrust.shape != (3,) or not np.isfinite(thrust).all():
        raise ValueError("thrust must contain three finite ENU components")
    if method not in ("RK45", "DOP853"):
        raise ValueError("supported reference methods are RK45 and DOP853")
    if not np.isfinite([rtol, atol]).all() or rtol <= 0 or atol <= 0:
        raise ValueError("solver tolerances must be finite and positive")
    if state.reason != Termination.RUNNING:
        return state, 0
    size = np.linalg.norm(thrust)
    if size > vehicle.max_thrust:
        thrust = thrust * vehicle.max_thrust / size
    burn_rate = float(np.linalg.norm(thrust)) / (vehicle.isp * vehicle.g0)
    fuel_horizon = (state.mass - vehicle.dry_mass) / burn_rate if burn_rate else np.inf
    time_horizon = clock.timeout - state.time
    horizon = min(clock.policy_dt, fuel_horizon, time_horizon)

    def equations(_time, y):
        derivative = np.empty(7)
        derivative[:3] = y[3:6]
        derivative[3:6] = thrust / y[6]
        derivative[5] -= vehicle.gravity
        derivative[6] = -burn_rate
        return derivative

    def ground(_time, y):
        return y[2]

    ground.terminal = True
    ground.direction = -1

    def bottom_of_descent(_time, y):
        return y[5]

    bottom_of_descent.direction = 1
    y0 = np.r_[state.position, state.velocity, state.mass]
    result = solve_ivp(
        equations,
        (0, horizon),
        y0,
        method=method,
        rtol=rtol,
        atol=atol,
        events=(ground, bottom_of_descent),
        dense_output=True,
    )
    if not result.success:
        raise RuntimeError(f"reference integration failed: {result.message}")
    y = result.y[:, -1]
    elapsed = float(result.t[-1])
    # solve_ivp sign-change events can miss an entire dip below the ground.
    # Independently locate a velocity minimum, then root the dense trajectory
    # on the descending side if that minimum is at/below the ground.
    hidden_contact = False
    if result.t_events[1].size:
        minimum_time = float(result.t_events[1][0])
        if result.sol(minimum_time)[2] <= 0:
            elapsed = brentq(lambda t: result.sol(t)[2], 0, minimum_time, xtol=1e-14)
            y = result.sol(elapsed)
            hidden_contact = True
    # A root exactly at the integration endpoint can be a few ulps above zero.
    at_ground = hidden_contact or bool(result.t_events[0].size) or (abs(y[2]) <= 1e-12 and y[5] < 0)
    if at_ground:
        y[2] = 0
        reason = Termination.TOUCHDOWN
    elif fuel_horizon <= horizon:
        y[6] = vehicle.dry_mass
        reason = Termination.OUT_OF_FUEL
    elif time_horizon <= horizon:
        reason = Termination.TIMEOUT
    else:
        reason = Termination.RUNNING
    time = clock.timeout if reason == Termination.TIMEOUT else state.time + elapsed
    return State(y[:3], y[3:6], float(y[6]), time, reason), result.nfev


def rollout(
    initial,
    thrusts,
    vehicle=_DEFAULT_VEHICLE,
    clock=_DEFAULT_CLOCK,
    *,
    method="RK45",
    rtol=1e-9,
    atol=1e-11,
    with_nfev=False,
):
    """Return final state and post-action trajectory; optionally report RHS calls."""
    state = initial
    states = []
    nfev = 0
    for thrust in thrusts:
        state, calls = _integrate(state, thrust, vehicle, clock, method, rtol, atol)
        states.append(state)
        nfev += calls
    history = State(
        *(
            np.asarray([s[i] for s in states])
            if states
            else np.empty((0, *np.shape(initial[i])), dtype=np.asarray(initial[i]).dtype)
            for i in range(len(State._fields))
        )
    )
    return (state, history, nfev) if with_nfev else (state, history)
