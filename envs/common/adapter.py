"""Gymnasium adapters kept outside JIT numerical paths."""

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces

from envs.sim_t import lander, rigid_env


class _Base(gym.Env):
    metadata = {"render_modes": []}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._key = jax.random.PRNGKey(seed)
        self._key, key = jax.random.split(self._key)
        self._state, obs, info = self._reset(key)
        return np.asarray(obs), {"privileged": np.asarray(info)}

    def step(self, action):
        if self._state is None:
            raise RuntimeError("reset must be called before step")
        transition = self._step(jnp.asarray(action, dtype=jnp.float32))
        self._state = transition.state
        state = self._state
        reason = state.reason if hasattr(state, "reason") else state.core.reason
        return (
            np.asarray(transition.observation),
            float(transition.reward),
            bool(transition.done),
            False,
            {
                "costs": np.asarray(transition.costs),
                "termination_reason": int(reason),
                "privileged": np.asarray(transition.privileged),
            },
        )


class PointMassLander(_Base):
    observation_space = spaces.Box(-np.inf, np.inf, shape=(19,), dtype=np.float32)
    action_space = spaces.Box(-1, 1, shape=(3,), dtype=np.float32)

    def __init__(self, config=None, seed=0):
        self.config = config or lander.Config()
        self._key, self._state = jax.random.PRNGKey(seed), None

    def _reset(self, key):
        obs, state, info = lander.reset(key, self.config)
        return state, obs, info

    def _step(self, action):
        return lander.step(self._state, action, self.config)


class RigidLander(_Base):
    observation_space = spaces.Box(-np.inf, np.inf, shape=(24,), dtype=np.float32)
    action_space = spaces.Box(-1, 1, shape=(4,), dtype=np.float32)

    def __init__(self, config=None, seed=0):
        self.config = config or rigid_env.RigidConfig()
        self._key, self._state = jax.random.PRNGKey(seed), None

    def _reset(self, key):
        state, obs, info = rigid_env.reset(key, self.config)
        return state, obs, info

    def _step(self, action):
        return rigid_env.step(self._state, action, self.config)
