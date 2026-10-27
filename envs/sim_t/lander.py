# ruff: noqa: B008 -- immutable dataclass configs are static JAX/API defaults
"""Trainable 3-DOF point-mass environment with CMDP signals."""

from dataclasses import dataclass
from typing import NamedTuple

# Immutable configuration defaults are safe to share between JIT calls.
# Ruff B008: frozen dataclasses intentionally form part of the static JIT signature.
import jax
import jax.numpy as jnp

from envs.sim_t.dynamics3d import State
from envs.sim_t.dynamics3d import reset as dynamics_reset
from envs.sim_t.dynamics3d import step as dynamics_step


@dataclass(frozen=True)
class Limits:
    vertical_speed: float = 2.0
    horizontal_speed: float = 1.0
    position_error: float = 5.0
    tilt: float = 0.0872664626
    angular_rate: float = 0.0872664626


@dataclass(frozen=True)
class Config:
    timeout: float = 60.0
    seed: int = 0
    domain_randomization: bool = True
    curriculum: bool = False
    difficulty: float = 1.0
    reward_weights: tuple = (0.01, 0.05, 0.5)
    penalty: float = 0.0
    constraint_budgets: tuple = (0.01, 0.01, 0.01)
    max_horizontal: float = 1500.0
    max_altitude: float = 3000.0
    limits: Limits = Limits()


_DEFAULT_CONFIG = Config()


_DEFAULT_CONFIG = Config()


class Rollout(NamedTuple):
    observation: jax.Array
    privileged: jax.Array
    reward: jax.Array
    costs: jax.Array
    done: jax.Array
    state: State


OBSERVATION_SIZE = 19
PRIVILEGED_SIZE = 21


def initial_condition(key, difficulty=1.0, randomize=True):
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    height = jax.random.uniform(k1, (), minval=50, maxval=250) * difficulty
    x = jax.random.uniform(k2, (), minval=-100, maxval=100) * difficulty
    y = jax.random.uniform(k3, (), minval=-100, maxval=100) * difficulty
    vz = jax.random.uniform(k4, (), minval=-50, maxval=-20)
    mass = jax.random.uniform(k5, (), minval=1400, maxval=1600)
    return jnp.array([x, y, height, 0.0, 0.0, vz, mass])


def observation(state, previous_action, baseline_action=None):
    v = state.vector
    normalized = jnp.concatenate((v[:3] / 2500, v[3:6] / 100, v[6:7] / 2000))
    # Identity attitude occupies six dimensions; no Euler-angle discontinuity.
    attitude = jnp.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=v.dtype)
    return jnp.concatenate(
        (
            normalized,
            attitude,
            previous_action,
            jnp.zeros(3, dtype=v.dtype) if baseline_action is None else baseline_action,
        )
    )


def privileged(state):
    return jnp.concatenate((state.vector, state.wind, jnp.ones(11, dtype=state.vector.dtype)))


def reward_cost(state, next_state, action, config=_DEFAULT_CONFIG):
    v, n = state.vector, next_state.vector
    potential = -(0.01 * jnp.linalg.norm(v[:3]) + 0.05 * jnp.linalg.norm(v[3:6]))
    next_potential = -(0.01 * jnp.linalg.norm(n[:3]) + 0.05 * jnp.linalg.norm(n[3:6]))
    shaping = 0.995 * next_potential - potential
    fuel = (n[6] - v[6]) / 500
    smooth = -0.01 * jnp.sum((action - state.previous_action) ** 2)
    terminal = jnp.where(
        next_state.reason == 1, 100.0, jnp.where(next_state.reason > 1, -100.0, 0.0)
    )
    reward = shaping + fuel + smooth + terminal
    elevation = jnp.arctan2(n[2], jnp.maximum(jnp.linalg.norm(n[:2]), 1e-6))
    costs = jnp.array([elevation < jnp.deg2rad(15), 0.0, 0.0], dtype=jnp.float32)
    reward = reward - config.penalty * jnp.sum(costs)
    return reward, costs


def reset(key, config=_DEFAULT_CONFIG):
    reset_key, _ = jax.random.split(key)
    vector = initial_condition(reset_key, config.difficulty, config.domain_randomization)
    state = dynamics_reset(vector, reset_key)
    obs = observation(state, state.previous_action)
    return obs, state, privileged(state)


def step(state, action, config=_DEFAULT_CONFIG):
    next_state = dynamics_step(state, jnp.clip(action, -1, 1), timeout=config.timeout)
    done = next_state.reason != 0
    reward, costs = reward_cost(state, next_state, action, config)
    obs = observation(next_state, next_state.previous_action)
    return Rollout(obs, privileged(next_state), reward, costs, done, next_state)


def rollout(state, actions, config=_DEFAULT_CONFIG):
    def advance(carry, action):
        transition = step(carry, action, config)
        return transition.state, transition

    return jax.lax.scan(advance, state, actions)
