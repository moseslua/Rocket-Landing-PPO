# ruff: noqa: B008 -- immutable configs are intentional static API defaults
"""Soft actor-critic with twin critics, persistent optimizers and replay updates."""

from dataclasses import dataclass
from typing import NamedTuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

from agents.networks import MLP, squash, squash_log_probability


@dataclass(frozen=True)
class SACConfig:
    observation_size: int = 24
    action_size: int = 4
    gamma: float = 0.995
    tau: float = 0.005
    learning_rate: float = 3e-4
    batch_size: int = 32
    target_entropy: float = -4.0


class Actor(nn.Module):
    action_size: int

    @nn.compact
    def __call__(self, observation):
        x = MLP(64, 2)(observation)
        return nn.Dense(self.action_size)(x), jnp.clip(nn.Dense(self.action_size)(x), -5, 2)


class Critic(nn.Module):
    @nn.compact
    def __call__(self, observation, action):
        x = MLP(64, 2)(jnp.concatenate((observation, action), axis=-1))
        return nn.Dense(1)(x)[..., 0], nn.Dense(1)(MLP(64, 2)(x))[..., 0]


class SACTrainState(NamedTuple):
    actor_params: dict
    critic_params: dict
    target_params: dict
    log_alpha: jax.Array
    actor_opt: object
    critic_opt: object
    alpha_opt: object
    updates: jax.Array


def init(key, config=SACConfig()):
    actor_key, critic_key = jax.random.split(key)
    actor, critic = Actor(config.action_size), Critic()
    obs, action = jnp.zeros((1, config.observation_size)), jnp.zeros((1, config.action_size))
    ap, cp = actor.init(actor_key, obs), critic.init(critic_key, obs, action)
    return actor, critic, ap, cp, jnp.array(0.0)


def init_train_state(key, config=SACConfig()):
    actor, critic, ap, cp, alpha = init(key, config)
    actor_optimizer = optax.adam(config.learning_rate)
    critic_optimizer = optax.adam(config.learning_rate)
    alpha_optimizer = optax.adam(config.learning_rate)
    state = SACTrainState(
        ap,
        cp,
        cp,
        alpha,
        actor_optimizer.init(ap),
        critic_optimizer.init(cp),
        alpha_optimizer.init(alpha),
        jnp.array(0),
    )
    return actor, critic, state, (actor_optimizer, critic_optimizer, alpha_optimizer)


def update(
    actor,
    critic,
    actor_params,
    critic_params,
    target_params,
    log_alpha,
    batch,
    key,
    config=SACConfig(),
    state=None,
    optimizers=None,
):
    if state is None:
        # Functional primitive used by tests: construct fresh optimizer state and
        # discard it, preserving the original public signature.
        _, _, state, optimizers = init_train_state(key, config)
        state = state._replace(
            actor_params=actor_params,
            critic_params=critic_params,
            target_params=target_params,
            log_alpha=log_alpha,
        )
    next_state, metrics = update_state(actor, critic, state, optimizers, batch, key, config)
    return (
        next_state.actor_params,
        next_state.critic_params,
        next_state.target_params,
        next_state.log_alpha,
    )


def update_state(actor, critic, state, optimizers, batch, key, config=SACConfig()):
    actor_optimizer, critic_optimizer, alpha_optimizer = optimizers
    key1, key2 = jax.random.split(key)
    next_mean, next_log_std = actor.apply(state.actor_params, batch["next_observation"])
    next_action, next_raw = squash(key1, next_mean, next_log_std)
    next_logp = squash_log_probability(next_mean, next_log_std, next_raw)
    q1, q2 = critic.apply(state.target_params, batch["next_observation"], next_action)
    target = batch["reward"] + config.gamma * (1 - batch["done"]) * (
        jnp.minimum(q1, q2) - jnp.exp(state.log_alpha) * next_logp
    )

    def critic_loss(params):
        q1, q2 = critic.apply(params, batch["observation"], batch["action"])
        return jnp.mean((q1 - target) ** 2 + (q2 - target) ** 2)

    (critic_loss_value, critic_grad) = jax.value_and_grad(critic_loss)(state.critic_params)
    updates, critic_opt = critic_optimizer.update(
        critic_grad, state.critic_opt, state.critic_params
    )
    critic_params = optax.apply_updates(state.critic_params, updates)

    def actor_loss(params):
        mean, log_std = actor.apply(params, batch["observation"])
        action, raw = squash(key2, mean, log_std)
        logp = squash_log_probability(mean, log_std, raw)
        q1, q2 = critic.apply(critic_params, batch["observation"], action)
        return jnp.mean(jnp.exp(state.log_alpha) * logp - jnp.minimum(q1, q2)), logp

    (actor_loss_value, sampled_logp), actor_grad = jax.value_and_grad(actor_loss, has_aux=True)(
        state.actor_params
    )
    updates, actor_opt = actor_optimizer.update(actor_grad, state.actor_opt, state.actor_params)
    actor_params = optax.apply_updates(state.actor_params, updates)

    def alpha_loss(value):
        return -jnp.mean(value * (jax.lax.stop_gradient(sampled_logp) + config.target_entropy))

    alpha_grad = jax.grad(alpha_loss)(state.log_alpha)
    updates, alpha_opt = alpha_optimizer.update(alpha_grad, state.alpha_opt, state.log_alpha)
    log_alpha = optax.apply_updates(state.log_alpha, updates)
    target_params = jax.tree.map(
        lambda old, new: (1 - config.tau) * old + config.tau * new,
        state.target_params,
        critic_params,
    )
    next_state = state._replace(
        actor_params=actor_params,
        critic_params=critic_params,
        target_params=target_params,
        log_alpha=log_alpha,
        actor_opt=actor_opt,
        critic_opt=critic_opt,
        alpha_opt=alpha_opt,
        updates=state.updates + 1,
    )
    return next_state, {
        "actor_loss": actor_loss_value,
        "critic_loss": critic_loss_value,
        "alpha": jnp.exp(log_alpha),
    }
