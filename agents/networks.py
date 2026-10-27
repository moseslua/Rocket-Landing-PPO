"""Small Flax policies/critics shared by PPO and SAC."""

import flax.linen as nn
import jax
import jax.numpy as jnp


class MLP(nn.Module):
    width: int = 64
    depth: int = 2

    @nn.compact
    def __call__(self, x):
        for _ in range(self.depth):
            x = nn.elu(nn.Dense(self.width)(x))
        return x


class ActorCritic(nn.Module):
    action_size: int = 4
    width: int = 64
    depth: int = 2

    @nn.compact
    def __call__(self, observation, privileged=None):
        x = MLP(self.width, self.depth)(observation)
        logits = nn.Dense(self.action_size, kernel_init=nn.initializers.zeros_init())(x)
        value_features = x if privileged is None else MLP(self.width, self.depth)(privileged)
        value = nn.Dense(1)(value_features)[..., 0]
        cost = nn.Dense(3)(value_features)
        return logits, value, cost


def init_actor_critic(key, observation_size, action_size, privileged_size=None, width=64, depth=2):
    model = ActorCritic(action_size, width, depth)
    privileged_size = privileged_size or observation_size
    params = model.init(key, jnp.zeros((1, observation_size)), jnp.zeros((1, privileged_size)))
    return model, params


def gaussian_log_probability(mean, log_std, action):
    variance = jnp.exp(2 * log_std)
    return -0.5 * jnp.sum(
        (action - mean) ** 2 / variance + 2 * log_std + jnp.log(2 * jnp.pi), axis=-1
    )


def gaussian_sample(key, mean, log_std):
    return mean + jax.random.normal(key, mean.shape) * jnp.exp(log_std)


def squash(key, mean, log_std):
    raw = mean + jax.random.normal(key, mean.shape) * jnp.exp(log_std)
    return jnp.tanh(raw), raw


def squash_log_probability(mean, log_std, raw):
    base = gaussian_log_probability(mean, log_std, raw)
    return base - jnp.sum(jnp.log(1 - jnp.tanh(raw) ** 2 + 1e-6), axis=-1)


class SACNetworks(nn.Module):
    action_size: int = 4
    privileged: bool = False

    @nn.compact
    def __call__(self, observation, action=None):
        x = MLP(64, 2)(observation)
        mean = nn.Dense(self.action_size)(x)
        log_std = jnp.clip(nn.Dense(self.action_size)(x), -5, 2)
        q1 = q2 = jnp.zeros(observation.shape[:-1])
        if action is not None:
            q_input = jnp.concatenate((observation, action), axis=-1)
            q1 = nn.Dense(1)(MLP(64, 2)(q_input))[..., 0]
            q2 = nn.Dense(1)(MLP(64, 2)(q_input))[..., 0]
        return mean, log_std, q1, q2
