# ruff: noqa: B008 -- immutable configs are intentional static API defaults
"""Residual action composition over an explicitly selected baseline backend."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from envs.sim_t import rigid_env


@dataclass(frozen=True)
class ResidualConfig:
    scale: float = 0.2
    baseline_backend: str = "jax_feedback"


def jax_feedback_baseline(state):
    """Development JAX thrust/attitude feedback, NOT CVXPY B0.

    A smooth desired vertical/lateral acceleration is converted to normalized
    throttle and bounded tilt corrections. This exists to exercise residual
    composition without hiding a pure-RL run behind a residual label.
    """
    v = state.core.vector
    thrust = 12000.0 + 400.0 * v[2] - 600.0 * v[5]
    throttle = jnp.clip(thrust / 36000.0, 0.4, 1.0)
    normalized_throttle = 2 * (throttle - 0.4) / 0.6 - 1
    pitch = jnp.clip(-v[0] / 500 - v[3] / 100, -1, 1)
    yaw = jnp.clip(-v[1] / 500 - v[4] / 100, -1, 1)
    roll = jnp.clip(-v[12] / jnp.deg2rad(15), -1, 1)
    return jnp.array([normalized_throttle, pitch, yaw, roll])


def baseline_action(state, config=ResidualConfig()):
    if config.baseline_backend != "jax_feedback":
        raise ValueError(
            "only the jax_feedback development baseline is available "
            "in vectorized residual training"
        )
    return jax_feedback_baseline(state)


def residual_action(state, correction, config=ResidualConfig()):
    if state.core.vector.ndim != 1:
        raise ValueError("residual_action expects one unbatched environment state")
    baseline = baseline_action(state, config)
    correction = jnp.asarray(correction)
    if correction.shape != baseline.shape:
        raise ValueError("residual correction must match the baseline action shape")
    applied = jnp.clip(baseline + config.scale * jnp.clip(correction, -1, 1), -1, 1)
    return applied, baseline


class ResidualRigid:
    """Expose baseline action to actor; apply baseline + bounded correction."""

    def __init__(self, environments, env_config, residual_config):
        self.environments, self.config, self.residual_config = (
            environments,
            env_config,
            residual_config,
        )
        self.core = rigid_env

    def reset(self, key):
        states, observations, privileged = rigid_env.batch_reset(
            jax.random.split(key, self.environments), self.config
        )
        baseline = jax.vmap(lambda state: baseline_action(state, self.residual_config))(states)
        return states, observations.at[:, -4:].set(baseline), privileged

    def step(self, state, action):
        baseline = jax.vmap(lambda value: baseline_action(value, self.residual_config))(state)
        applied = jnp.clip(baseline + self.residual_config.scale * jnp.clip(action, -1, 1), -1, 1)
        transition = rigid_env.batch_step(state, applied, self.config)
        return transition._replace(observation=transition.observation.at[:, -4:].set(baseline))
