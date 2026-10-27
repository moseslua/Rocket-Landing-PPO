# ruff: noqa: B008 -- immutable dataclass configs are static JAX/API defaults
"""Vectorizable JAX 6-DOF training environment with separate CMDP costs."""

from dataclasses import dataclass
from typing import NamedTuple

# Immutable configuration defaults are safe to share between JIT calls.
# Ruff B008: frozen dataclasses intentionally form part of the static JIT signature.
import jax
import jax.numpy as jnp

from envs.common.rigid import RigidClock, RigidVehicle
from envs.sim_t.history import delayed, init_history, push_history
from envs.sim_t.rigid_body import (
    RigidState,
    tilt,
)
from envs.sim_t.rigid_body import (
    step as core_step,
)

OBSERVATION_SIZE = 24
ACTION_SIZE = 4
CONSTRAINT_SIZE = 3


@dataclass(frozen=True)
class RigidConfig:
    timeout: float = 60.0
    difficulty: float = 1.0
    domain_randomization: bool = True
    observation_noise: float = 0.0
    delay: float = 0.0
    penalty: float = 0.0
    bounds: float = 1500.0
    max_altitude: float = 5000.0
    budget: tuple = (0.01, 0.01, 0.01)
    shaping: bool = True


_DEFAULT_CONFIG = RigidConfig()
_DEFAULT_VEHICLE = RigidVehicle()
_DEFAULT_CLOCK = RigidClock()


class EnvState(NamedTuple):
    core: RigidState
    history: jax.Array
    previous_action: jax.Array
    key: jax.Array


class Transition(NamedTuple):
    observation: jax.Array
    reward: jax.Array
    costs: jax.Array
    done: jax.Array
    privileged: jax.Array
    state: EnvState


def sample_reset(key, config=_DEFAULT_CONFIG, vehicle=_DEFAULT_VEHICLE):
    keys = jax.random.split(key, 9)
    d = config.difficulty
    position = jnp.array(
        [
            jax.random.uniform(keys[0], (), minval=-400 * d, maxval=400 * d),
            jax.random.uniform(keys[1], (), minval=-400 * d, maxval=400 * d),
            jax.random.uniform(keys[2], (), minval=500 * d, maxval=2500 * d),
        ]
    )
    velocity = jnp.array(
        [
            jax.random.uniform(keys[3], (), minval=-25, maxval=25),
            jax.random.uniform(keys[4], (), minval=-25, maxval=25),
            jax.random.uniform(keys[5], (), minval=-100, maxval=-30),
        ]
    )
    angle = jax.random.uniform(keys[6], (), minval=0, maxval=jnp.deg2rad(15)) * d
    quaternion = jnp.array([jnp.cos(angle / 2), 0.0, jnp.sin(angle / 2), 0.0])
    mass = jax.random.uniform(
        keys[7], (), minval=vehicle.dry_mass * 1.15, maxval=vehicle.dry_mass * 1.5
    )
    omega = jax.random.uniform(keys[8], (3,), minval=-jnp.deg2rad(5), maxval=jnp.deg2rad(5)) * d
    vector = jnp.concatenate((position, velocity, quaternion, omega, jnp.array([mass])))
    core = RigidState(
        vector,
        jnp.zeros(2),
        jnp.array(1.0),
        jnp.array(0.0),
        jnp.array(0, jnp.int32),
        jnp.zeros(3),
        key,
        0,
        0,
    )
    return EnvState(core, init_history(jnp.zeros(OBSERVATION_SIZE)), jnp.zeros(ACTION_SIZE), key)


def initial_from_arrays(
    position,
    velocity,
    quaternion=(1.0, 0.0, 0.0, 0.0),
    angular_rate=(0.0, 0.0, 0.0),
    mass=1700.0,
    wind=(0.0, 0.0, 0.0),
    key=None,
):
    vector = jnp.concatenate(
        (
            jnp.asarray(position, jnp.float32),
            jnp.asarray(velocity, jnp.float32),
            jnp.asarray(quaternion, jnp.float32),
            jnp.asarray(angular_rate, jnp.float32),
            jnp.asarray([mass], jnp.float32),
        )
    )
    vector = vector.at[6:10].set(vector[6:10] / jnp.linalg.norm(vector[6:10]))
    core = RigidState(
        vector,
        jnp.zeros(2),
        jnp.array(1.0),
        jnp.array(0.0),
        jnp.array(0, jnp.int32),
        jnp.asarray(wind, jnp.float32),
        jax.random.PRNGKey(0) if key is None else key,
        0,
        0,
    )
    return core


def observation(state, baseline=None):
    v = state.core.vector
    q = v[6:10]
    up_uv = jnp.cross(q[1:], jnp.array([0.0, 0.0, 1.0]))
    up = jnp.array([0.0, 0.0, 1.0]) + 2 * (q[0] * up_uv + jnp.cross(q[1:], up_uv))
    right_uv = jnp.cross(q[1:], jnp.array([1.0, 0.0, 0.0]))
    right = jnp.array([1.0, 0.0, 0.0]) + 2 * (q[0] * right_uv + jnp.cross(q[1:], right_uv))
    return jnp.concatenate(
        (
            v[:3] / 2500,
            v[3:6] / 100,
            up,
            right,
            v[10:13] / jnp.deg2rad(30),
            v[13:14] / 2000,
            state.previous_action,
            jnp.zeros(4) if baseline is None else baseline,
        )
    ).astype(jnp.float32)


def reset(key, config=_DEFAULT_CONFIG, vehicle=_DEFAULT_VEHICLE):
    reset_key, obs_key = jax.random.split(key)
    state = sample_reset(reset_key, config, vehicle)._replace(key=obs_key)
    current = observation(state)
    history = init_history(current)
    if config.observation_noise:
        current = current + config.observation_noise * jax.random.normal(obs_key, current.shape)
    return state._replace(history=history), current, privileged(state)


def privileged(state):
    return jnp.concatenate(
        (
            state.core.vector,
            state.core.wind,
            jnp.array([36000.0, 1200.0, 300.0, 4000.0, 4000.0, 600.0, 0.1]),
        )
    )


def reward_cost(previous, state, action, config):
    v, n = previous.core.vector, state.core.vector
    reason = state.core.reason
    potential = -(
        0.01 * jnp.linalg.norm(v[:3]) + 0.05 * jnp.linalg.norm(v[3:6]) + 0.5 * tilt(v[6:10])
    )
    next_potential = -(
        0.01 * jnp.linalg.norm(n[:3]) + 0.05 * jnp.linalg.norm(n[3:6]) + 0.5 * tilt(n[6:10])
    )
    shaping = jnp.where(config.shaping, 0.995 * next_potential - potential, 0.0)
    fuel = (n[13] - v[13]) / jnp.maximum(v[13] - 1200.0, 1.0)
    smooth = -0.01 * jnp.sum((action - previous.previous_action) ** 2)
    terminal = jnp.where(reason == 1, 100.0, jnp.where(reason > 1, -100.0, 0.0))
    elevation = jnp.arctan2(n[2], jnp.maximum(jnp.linalg.norm(n[:2]), 1e-6))
    costs = jnp.array(
        [
            elevation < jnp.deg2rad(15),
            tilt(n[6:10]) > jnp.deg2rad(20),
            jnp.linalg.norm(n[10:13]) > jnp.deg2rad(15),
        ],
        dtype=jnp.float32,
    )
    return shaping + fuel + smooth + terminal - config.penalty * jnp.sum(costs), costs


def step(state, action, config=_DEFAULT_CONFIG, vehicle=_DEFAULT_VEHICLE, clock=_DEFAULT_CLOCK):
    action = jnp.clip(action, -1, 1)
    step_key, next_key = jax.random.split(state.key)
    core = core_step(state.core, action, vehicle, clock)
    next_state = EnvState(core, state.history, action, next_key)
    exact = observation(next_state)
    noisy = exact + config.observation_noise * jax.random.normal(step_key, exact.shape)
    history = push_history(state.history, noisy)
    next_state = next_state._replace(history=history)
    obs = delayed(history, config.delay, clock.physics_dt)
    reward, costs = reward_cost(state, next_state, action, config)
    return Transition(obs, reward, costs, core.reason != 0, privileged(next_state), next_state)


def rollout(state, actions, config=_DEFAULT_CONFIG):
    def advance(carry, action):
        transition = step(carry, action, config)
        return transition.state, transition

    return jax.lax.scan(advance, state, actions)


def batch_reset(keys, config=_DEFAULT_CONFIG):
    return jax.vmap(lambda key: reset(key, config))(keys)


def batch_step(states, actions, config=_DEFAULT_CONFIG):
    return jax.vmap(lambda state, action: step(state, action, config))(states, actions)
