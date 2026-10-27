"""Development pipeline tests; not full experiment/gate validation."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from agents.ppo import PPOConfig, gae, init_train_state, update, update_multipliers
from agents.replay import ReplayBuffer
from agents.sac import SACConfig
from agents.sac import init as init_sac
from agents.sac import update as update_sac
from controllers.guidance import GuidanceConfig, GuidanceController
from controllers.residual import compose_residual
from envs.common.adapter import PointMassLander, RigidLander
from envs.common.rigid import RigidVehicle
from envs.common.scenarios import initialize, load
from envs.sim_e import high_fidelity
from envs.sim_e.rigid_body import attitude_only
from envs.sim_e.rigid_body import reset as ref_reset
from envs.sim_t import rigid_body, rigid_env
from envs.sim_t.rigid_env import RigidConfig


def test_quaternion_attitude_and_torque_free_rotation():
    vehicle = RigidVehicle(max_thrust=1, gravity=0, isp=1000, min_throttle=1.0)
    q = np.array([np.cos(0.2), 0, np.sin(0.2), 0])
    omega = np.array([0.0, 0.0, 1.0])
    state = rigid_body.reset()
    vector = (
        state.vector.at[6:10]
        .set(jnp.asarray(q, dtype=jnp.float32))
        .at[10:13]
        .set(jnp.asarray(omega, dtype=jnp.float32))
    )
    derivative = rigid_body.rhs(vector, jnp.zeros(3), 0, vehicle)
    np.testing.assert_allclose(derivative[3:6], 0, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(vector[6:10]), 1)
    expected = attitude_only(q, omega)
    # Zero external torque, zero-throttle force: angular momentum/energy invariant.
    action = jnp.array([-1.0, 0.0, 0.0, 0.0])
    next_state = rigid_body.step(state._replace(vector=vector), action, vehicle)
    assert np.linalg.norm(next_state.vector[6:10]) == pytest.approx(1, abs=1e-6)
    initial_inertia_momentum = np.asarray(vehicle.inertia) * omega
    final_inertia_momentum = np.asarray(vehicle.inertia) * np.asarray(next_state.vector[10:13])
    np.testing.assert_allclose(initial_inertia_momentum, final_inertia_momentum, atol=1e-5)
    initial_energy = 0.5 * np.sum(initial_inertia_momentum * omega)
    final_energy = 0.5 * np.sum(final_inertia_momentum * np.asarray(next_state.vector[10:13]))
    assert final_energy == pytest.approx(initial_energy, abs=1e-5)
    assert expected.shape == (4,)


def test_rigid_jax_batch_and_delay_reset_are_deterministic():
    keys = jax.random.split(jax.random.PRNGKey(0), 3)
    config = RigidConfig(delay=0.06, observation_noise=0)
    first = jax.jit(rigid_env.batch_reset)(keys)
    second = jax.jit(rigid_env.batch_reset)(keys)
    np.testing.assert_array_equal(first[1], second[1])
    assert first[1].shape == (3, 24)
    actions = jnp.zeros((3, 2, 4))
    states = first[0]
    final, history = jax.jit(jax.vmap(lambda s, a: rigid_env.rollout(s, a, config)))(
        states, actions
    )
    assert history.observation.shape == (3, 2, 24)
    assert np.isfinite(np.asarray(final.core.vector)).all()
    reference = ref_reset()
    high = high_fidelity.integrate(reference, np.zeros(4))
    assert np.isfinite(high.vector).all()


def test_gym_adapters_and_done_semantics():
    for environment in (PointMassLander(seed=1), RigidLander(seed=1)):
        obs, info = environment.reset(seed=5)
        assert np.isfinite(obs).all()
        assert np.isfinite(info["privileged"]).all()
        next_obs, reward, done, truncated, info = environment.step(
            np.zeros(environment.action_space.shape)
        )
        assert next_obs.shape == environment.observation_space.shape
        assert isinstance(reward, float) and isinstance(done, bool) and not truncated
        assert info["costs"].shape == (3,)
        environment.close()
    check_env(PointMassLander(), skip_render_check=True)


def test_frozen_scenarios_are_hashed_and_not_overwritten(tmp_path):
    path = tmp_path / "set.json"
    receipt = initialize(path, count=3)
    assert receipt["count"] == 3
    assert len(load(path)["scenarios"]) == 3
    with pytest.raises(FileExistsError):
        initialize(path, count=3)
    path.write_text(path.read_text().replace('"mass":1', '"mass":2'))
    with pytest.raises(ValueError, match="hash mismatch"):
        load(path)


def test_guidance_and_residual_composition():
    controller = GuidanceController(GuidanceConfig(horizon=2, dt=0.05))
    action = controller.act(
        np.r_[0.0, 0.0, 100.0, 0.0, 0.0, -10.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1700.0]
    )
    assert action.shape == (4,)
    assert np.isfinite(action).all()
    np.testing.assert_allclose(
        compose_residual([0.0, 0.0, 0.0, 0.0], [1.0, -1.0, 0.5, 2.0], 0.2), [0.2, -0.2, 0.1, 0.2]
    )
    with pytest.raises(ValueError):
        compose_residual([0.0, 0.0], [0.0])


def test_gae_and_ppo_update_change_parameters():
    rewards = jnp.array([1.0, 0.0, 1.0])
    values = jnp.array([0.5, 0.5, 0.5])
    dones = jnp.array([0.0, 0.0, 1.0])
    adv, ret = gae(rewards, values, dones, 0.99, 0.95)
    assert adv.shape == ret.shape == (3,)
    config = PPOConfig(observation_size=4, privileged_size=5, action_size=2, epochs=1)
    model, state, optimizer = init_train_state(jax.random.PRNGKey(0), config)
    batch = {
        "observation": jnp.ones((4, 4)),
        "privileged": jnp.ones((4, 5)),
        "raw_action": jnp.zeros((4, 2)),
        "logp": jnp.zeros(4),
        "advantage": jnp.array([1.0, 2.0, 3.0, 4.0]),
        "return": jnp.ones(4),
        "cost_advantage": jnp.zeros((4, 3)),
        "cost_return": jnp.zeros((4, 3)),
    }
    next_state, metrics = update(model, optimizer, state, batch, config)
    assert int(next_state.update) == 1
    assert np.isfinite(float(metrics["loss"]))
    constrained = update_multipliers(state, jnp.ones((4, 2, 3)), replace(config, constrained=True))
    assert np.all(np.asarray(constrained.multipliers) > 0)


def test_replay_and_sac_update_are_finite():
    replay = ReplayBuffer(4, 3, 2)
    for _ in range(4):
        replay.add(np.zeros(3), np.zeros(2), 0.0, np.ones(3), False)
    batch = replay.sample(2)
    config = SACConfig(observation_size=3, action_size=2)
    actor, critic, ap, cp, alpha = init_sac(jax.random.PRNGKey(0), config)
    next_values = update_sac(
        actor,
        critic,
        ap,
        cp,
        cp,
        alpha,
        jax.tree.map(jnp.asarray, batch),
        jax.random.PRNGKey(1),
        config,
    )
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(next_values))
