"""Behavioral checks for ideal 3-DOF cores, not full research gate validation."""

from dataclasses import replace
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from envs.common import Clock, Scenario, Termination, Vehicle
from envs.sim_e import point_mass as sim_e
from envs.sim_t import point_mass as sim_t
from envs.sim_t.history import delayed, init_history, push_history


def test_configuration_rejects_invalid_inputs():
    for kwargs in ({"dry_mass": 0}, {"isp": -1}, {"gravity": np.nan}):
        with pytest.raises(ValueError):
            Vehicle(**kwargs)
    with pytest.raises(ValueError):
        Clock(physics_dt=0)
    with pytest.raises(ValueError):
        Clock(substeps=1.5)
    with pytest.raises(ValueError):
        Scenario(position=(0, 0, -1))
    for backend in (sim_t, sim_e):
        with pytest.raises(ValueError):
            backend.reset(Scenario(mass=999))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_free_fall_matches_analytic_motion(dtype):
    scenario = Scenario(position=(10, -5, 100), velocity=(2, 3, 4))
    expected_position = np.array(scenario.position) + np.array(scenario.velocity) - [0, 0, 4.905]
    with jax.enable_x64(dtype == np.float64):
        initial = sim_t.reset(scenario, dtype=dtype)
        final, trajectory = jax.jit(sim_t.rollout)(initial, jnp.zeros((20, 3), dtype=dtype))
        assert trajectory.position.shape == (20, 3)
        assert final.position.dtype == dtype
        tolerance = 1e-4 if dtype == np.float32 else 1e-10
        np.testing.assert_allclose(final.position, expected_position, atol=tolerance, rtol=0)
        np.testing.assert_allclose(final.velocity, [2, 3, -5.81], atol=tolerance, rtol=0)
        assert float(final.mass) == scenario.mass
    reference, _ = sim_e.rollout(sim_e.reset(scenario), np.zeros((20, 3)))
    np.testing.assert_allclose(reference.position, expected_position, atol=1e-10, rtol=0)


def test_float64_is_not_silently_downcast():
    with jax.enable_x64(False), pytest.raises(ValueError, match="x64"):
        sim_t.reset(Scenario(), dtype=np.float64)


@pytest.mark.parametrize("method", ["RK45", "DOP853"])
def test_constant_burn_matches_variable_mass_rocket_equation(method):
    vehicle = Vehicle()
    scenario = Scenario(position=(0, 0, 1000), velocity=(0, 0, 10))
    thrust = 20000.0
    duration = 5.0
    mass_flow = thrust / (vehicle.isp * vehicle.g0)
    expected_mass = scenario.mass - mass_flow * duration
    expected_velocity = (
        scenario.velocity[2]
        + vehicle.isp * vehicle.g0 * np.log(scenario.mass / expected_mass)
        - vehicle.gravity * duration
    )
    actions = np.tile([0, 0, thrust], (100, 1))
    final, _ = sim_t.rollout(sim_t.reset(scenario), jnp.asarray(actions, dtype=jnp.float32))
    np.testing.assert_allclose(final.mass, expected_mass, atol=0.02, rtol=0)
    np.testing.assert_allclose(final.velocity[2], expected_velocity, atol=1e-3, rtol=0)
    reference, _ = sim_e.rollout(sim_e.reset(scenario), actions, method=method)
    np.testing.assert_allclose(reference.mass, expected_mass, atol=1e-8, rtol=0)
    np.testing.assert_allclose(reference.velocity[2], expected_velocity, atol=1e-8, rtol=0)


@pytest.mark.parametrize("backend", [sim_t, sim_e])
def test_touchdown_is_localized_and_absorbing(backend):
    scenario = Scenario(position=(0, 0, 0.012), velocity=(2, 0, -1))
    initial = backend.reset(scenario)
    final = backend.step(initial, np.zeros(3))
    expected_time = (-1 + np.sqrt(1 + 2 * 9.81 * 0.012)) / 9.81
    assert int(final.reason) == Termination.TOUCHDOWN
    assert float(final.position[2]) == 0
    assert float(final.time) == pytest.approx(expected_time, abs=2e-7)
    assert float(final.velocity[2]) == pytest.approx(-1 - 9.81 * expected_time, abs=2e-6)
    assert float(final.position[0]) == pytest.approx(2 * expected_time, abs=1e-6)
    frozen = backend.step(final, np.array([0, 0, 30000]))
    for before, after in zip(final, frozen, strict=True):
        np.testing.assert_array_equal(before, after)


@pytest.mark.parametrize("backend", [sim_t, sim_e])
def test_fuel_event_and_timeout_stop_inside_a_control_interval(backend):
    vehicle = Vehicle(gravity=0, isp=10, g0=10, max_thrust=1000)
    initial = backend.reset(Scenario(position=(0, 0, 100), mass=1000.02), vehicle=vehicle)
    final = backend.step(initial, np.array([0, 0, 1000]), vehicle=vehicle)
    assert int(final.reason) == Termination.OUT_OF_FUEL
    assert float(final.mass) == vehicle.dry_mass
    # Float32 reset rounds 1000.02; use the represented fuel, not the decimal literal.
    expected_time = (float(initial.mass) - vehicle.dry_mass) / 10
    assert float(final.time) == pytest.approx(expected_time, abs=2e-7)
    clock = Clock(timeout=0.0123)
    timed_out = backend.step(backend.reset(Scenario()), np.zeros(3), clock=clock)
    assert int(timed_out.reason) == Termination.TIMEOUT
    assert float(timed_out.time) == pytest.approx(clock.timeout, abs=1e-7)


@pytest.mark.parametrize("backend", [sim_t, sim_e])
def test_initial_boundary_priority_and_midstep_ties(backend):
    boundary = backend.reset(Scenario(position=(0, 0, 0), mass=1000, time=60))
    assert int(boundary.reason) == Termination.TOUCHDOWN
    vehicle = Vehicle(gravity=0)
    clock = Clock(physics_dt=0.005, substeps=10, timeout=0.025)
    falling = backend.reset(Scenario(position=(0, 0, 0.025), velocity=(0, 0, -1)))
    final = backend.step(falling, np.zeros(3), vehicle=vehicle, clock=clock)
    assert int(final.reason) == Termination.TOUCHDOWN
    fuel_vehicle = Vehicle(gravity=0, isp=10, g0=10, max_thrust=1000)
    emptying = backend.reset(Scenario(mass=1000.25), vehicle=fuel_vehicle)
    final = backend.step(emptying, np.array([0, 0, 1000]), vehicle=fuel_vehicle, clock=clock)
    assert int(final.reason) == Termination.OUT_OF_FUEL
    contact_and_fuel = backend.reset(
        Scenario(position=(0, 0, 0.025), velocity=(0, 0, -1), mass=1000.25),
        vehicle=fuel_vehicle,
        clock=clock,
    )
    final = backend.step(
        contact_and_fuel, np.array([1000, 0, 0]), vehicle=fuel_vehicle, clock=clock
    )
    assert int(final.reason) == Termination.TOUCHDOWN
    assert float(final.mass) == pytest.approx(1000, abs=1e-4)


def test_seeded_resets_and_batched_scan_are_repeatable():
    keys = jax.random.split(jax.random.PRNGKey(15), 4)
    initial = jax.jit(jax.vmap(sim_t.sample_reset))(keys)
    repeated = jax.jit(jax.vmap(sim_t.sample_reset))(keys)
    np.testing.assert_array_equal(initial.position, repeated.position)
    assert np.unique(np.asarray(initial.position[:, 2])).size == 4
    actions = jnp.broadcast_to(jnp.array([0, 0, 18000.0]), (4, 20, 3))
    batch_final, batch_history = jax.jit(sim_t.batch_rollout)(initial, actions)
    serial = jax.tree.map(lambda value: value[0], initial)
    for action in actions[0]:
        serial = sim_t.step(serial, action)
    np.testing.assert_allclose(batch_final.position[0], serial.position, atol=1e-4, rtol=0)
    np.testing.assert_array_equal(batch_final.position, batch_history.position[:, -1])
    second_final, _ = jax.jit(sim_t.batch_rollout)(initial, actions)
    np.testing.assert_array_equal(second_final.position, batch_final.position)
    a, b = sim_e.sample_reset(15), sim_e.sample_reset(15)
    np.testing.assert_array_equal(a.position, b.position)
    assert not np.array_equal(a.position, sim_e.sample_reset(16).position)


def test_rk4_convergence_and_independent_nominal_agreement():
    vehicle = Vehicle(gravity=0, isp=1, g0=1000, max_thrust=50000)
    scenario = Scenario(position=(0, 0, 1000), mass=1600)
    action = np.array([30000, 10000, 20000])
    duration = 2.0
    flow = np.linalg.norm(action) / (vehicle.isp * vehicle.g0)
    exact_velocity = action / flow * np.log(scenario.mass / (scenario.mass - flow * duration))
    errors = []
    with jax.enable_x64():
        for substeps in (1, 2, 4):
            clock = Clock(physics_dt=duration / substeps, substeps=substeps)
            result = sim_t.step(sim_t.reset(scenario, dtype=np.float64), action, vehicle, clock)
            errors.append(np.linalg.norm(np.asarray(result.velocity) - exact_velocity))
        assert errors[1] < errors[0] / 10
        assert errors[2] < errors[1] / 10
        actions = np.tile(action, (40, 1))
        initial = sim_t.reset(scenario, dtype=np.float64)
        final, _ = jax.jit(partial(sim_t.rollout, vehicle=vehicle))(initial, actions)
        for method in ("RK45", "DOP853"):
            reference, _ = sim_e.rollout(
                sim_e.reset(scenario), actions, vehicle=vehicle, method=method
            )
            np.testing.assert_allclose(final.position, reference.position, atol=1e-7, rtol=0)
            np.testing.assert_allclose(final.velocity, reference.velocity, atol=1e-8, rtol=0)


def test_delay_history_uses_physics_ticks_without_negative_wraparound():
    history = init_history(jnp.array([0.0, 1.0]))
    assert history.shape == (13, 2)
    np.testing.assert_array_equal(delayed(history, 0.06), [0, 1])
    push = jax.jit(push_history)
    for tick in range(1, 15):
        history = push(history, jnp.array([float(tick), tick + 1.0]))
    read = jax.jit(delayed)
    for seconds, expected in [(0, 14), (0.005, 13), (0.06, 2), (0.0075, 12)]:
        assert float(read(history, seconds)[0]) == expected
    for invalid in (-0.001, 0.061, np.nan):
        assert np.isnan(np.asarray(read(history, invalid))).all()


@pytest.mark.parametrize("backend", [sim_t, sim_e])
def test_first_ground_crossing_is_not_missed_when_thrust_reverses_descent(backend):
    # Both endpoints can be above ground: detect the intervening downward crossing.
    scenario = Scenario(position=(0, 0, 1e-6), velocity=(0, 0, -0.01))
    final = backend.step(backend.reset(scenario), np.array([0, 0, 30000]))
    assert int(final.reason) == Termination.TOUCHDOWN
    assert 0 < float(final.time) < 0.0002
    assert float(final.velocity[2]) < 0
    assert float(final.position[2]) == 0


@pytest.mark.parametrize("backend", [sim_t, sim_e])
def test_force_norm_is_clipped_and_terminal_events_keep_priority(backend):
    initial = backend.reset(Scenario())
    clipped = backend.step(initial, np.array([0, 0, 30000]))
    excessive = backend.step(initial, np.array([0, 0, 60000]))
    np.testing.assert_allclose(clipped.velocity, excessive.velocity, atol=1e-7, rtol=0)
    np.testing.assert_allclose(clipped.mass, excessive.mass, atol=1e-6, rtol=0)
    vehicle = Vehicle(gravity=0, isp=10, g0=10, max_thrust=1000)
    # Fuel runs out after touchdown; the earlier contact must win without extra burn.
    falling = backend.reset(
        Scenario(position=(0, 0, 0.005), velocity=(0, 0, -1), mass=1000.25), vehicle=vehicle
    )
    final = backend.step(falling, np.array([1000, 0, 0]), vehicle=vehicle)
    assert int(final.reason) == Termination.TOUCHDOWN
    assert float(final.mass) > vehicle.dry_mass


def test_long_float32_burn_preserves_fuel_and_clock():
    initial = sim_t.reset(Scenario(position=(0, 0, 10000)))
    actions = jnp.tile(jnp.array([0, 0, 20000.0]), (1201, 1))
    final, _ = jax.jit(sim_t.rollout)(initial, actions)
    assert int(final.reason) == Termination.TIMEOUT
    assert float(final.time) == 60
    expected = 1500 - 20000 * 60 / (300 * 9.80665)
    assert float(final.mass) == pytest.approx(expected, abs=0.001)


@pytest.mark.parametrize("backend", [sim_t, sim_e])
def test_empty_rollout_has_consistent_shapes(backend):
    initial = backend.reset(Scenario())
    final, history = backend.rollout(initial, np.empty((0, 3), dtype=np.float32))
    np.testing.assert_array_equal(final.position, initial.position)
    assert history.position.shape == (0, 3)
    assert history.mass.shape == (0,)


def test_reference_rejects_nonfinite_actions_and_unsupported_methods():
    initial = sim_e.reset(Scenario())
    with pytest.raises(ValueError):
        sim_e.step(initial, np.array([0, np.nan, 0]))
    with pytest.raises(ValueError):
        sim_e.step(initial, np.zeros(3), method="BDF")
    clock = replace(Clock(), timeout=1)
    assert clock.policy_dt == 0.05
