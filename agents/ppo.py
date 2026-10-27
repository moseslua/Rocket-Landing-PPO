# ruff: noqa: B008 -- immutable configs are intentional static API defaults
"""Clipped-ratio PPO with GAE, asymmetric critics and PID-Lagrangian constraints."""

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from agents.networks import gaussian_log_probability, gaussian_sample, init_actor_critic


@dataclass(frozen=True)
class PPOConfig:
    observation_size: int = 24
    privileged_size: int = 24
    action_size: int = 4
    width: int = 64
    depth: int = 2
    learning_rate: float = 3e-4
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip: float = 0.2
    value_coefficient: float = 0.5
    cost_value_coefficient: float = 0.1
    entropy_coefficient: float = 0.01
    epochs: int = 2
    minibatches: int = 2
    max_gradient_norm: float = 0.5
    initial_log_std: float = -0.5
    constrained: bool = False
    multiplier_cap: float = 100.0
    kp: float = 0.1
    ki: float = 0.01
    kd: float = 0.0
    budgets: tuple = (0.01, 0.01, 0.01)
    privileged_critic: bool = True
    anneal_learning_rate: bool = False
    total_steps: int = 1000


class TrainState(NamedTuple):
    params: dict
    log_std: jax.Array
    opt_state: object
    multipliers: jax.Array
    integral: jax.Array
    previous_gap: jax.Array
    update: jax.Array


class Carry(NamedTuple):
    env_state: object
    observation: jax.Array
    privileged: jax.Array
    key: jax.Array


def init_train_state(key, config=PPOConfig()):
    model, params = init_actor_critic(
        key,
        config.observation_size,
        config.action_size,
        config.privileged_size,
        config.width,
        config.depth,
    )
    schedule = (
        optax.linear_schedule(config.learning_rate, 0.0, config.total_steps)
        if config.anneal_learning_rate
        else config.learning_rate
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.max_gradient_norm), optax.adam(schedule)
    )
    train_state = TrainState(
        params,
        jnp.full((config.action_size,), config.initial_log_std),
        optimizer.init((params, jnp.zeros(config.action_size))),
        jnp.zeros(3),
        jnp.zeros(3),
        jnp.zeros(3),
        jnp.array(0),
    )
    return model, train_state, optimizer


def action_value(model, train_state, observation, privileged, key, deterministic=False):
    mean, value, costs = model.apply(train_state.params, observation, privileged)
    action = (
        jnp.tanh(mean)
        if deterministic
        else jnp.tanh(gaussian_sample(key, mean, train_state.log_std))
    )
    return action, value, costs


def gae(rewards, values, dones, gamma, lam, next_values=None):
    """Time-major GAE; `next_values` bootstraps the post-rollout state.

    Without `next_values`, the final value is reused, which is only appropriate
    for a rollout known to end at a terminal boundary. All callers collecting
    partial episodes must provide the true bootstrap value.
    """
    if next_values is None:
        next_values = values[-1]

    def backward(carry, data):
        reward, value, done, next_value = data
        delta = reward + gamma * next_value * (1 - done) - value
        carry = delta + gamma * lam * (1 - done) * carry
        return carry, carry

    successors = jnp.concatenate((values[1:], jnp.expand_dims(next_values, 0)), axis=0)
    _, advantages = jax.lax.scan(
        backward, jnp.zeros_like(values[0]), (rewards, values, dones, successors), reverse=True
    )
    return advantages, advantages + values


def update(model, optimizer, train_state, batch, config=PPOConfig()):
    def loss(params_log_std):
        params, log_std = params_log_std
        mean, values, cost_values = model.apply(params, batch["observation"], batch["privileged"])
        logp = gaussian_log_probability(mean, log_std, batch["raw_action"])
        old_logp = batch["logp"]
        ratio = jnp.exp(logp - old_logp)
        advantages = (batch["advantage"] - batch["advantage"].mean()) / (
            batch["advantage"].std() + 1e-8
        )
        policy_loss = -jnp.mean(
            jnp.minimum(
                ratio * advantages, jnp.clip(ratio, 1 - config.clip, 1 + config.clip) * advantages
            )
        )
        value_loss = jnp.mean((batch["return"] - values) ** 2)
        cost_loss = jnp.mean((batch["cost_return"] - cost_values) ** 2)
        entropy = jnp.mean(jnp.sum(log_std + 0.5 * jnp.log(2 * jnp.pi * jnp.e), axis=-1))
        total = (
            policy_loss
            + config.value_coefficient * value_loss
            + config.cost_value_coefficient * cost_loss
            - config.entropy_coefficient * entropy
        )
        if config.constrained:
            cost_advantages = batch["cost_advantage"]
            # Per-constraint weighting is applied before averaging samples/channels.
            lagrangian = jnp.mean(train_state.multipliers * cost_advantages)
            total += lagrangian
        return total, {
            "loss": total,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "cost_loss": cost_loss,
            "entropy": entropy,
            "approx_kl": jnp.mean(old_logp - logp),
        }

    (_, metrics), gradients = jax.value_and_grad(loss, has_aux=True)(
        (train_state.params, train_state.log_std)
    )
    updates, opt_state = optimizer.update(
        gradients, train_state.opt_state, (train_state.params, train_state.log_std)
    )
    params, log_std = optax.apply_updates((train_state.params, train_state.log_std), updates)
    return train_state._replace(
        params=params,
        log_std=jnp.clip(log_std, -5, 2),
        opt_state=opt_state,
        update=train_state.update + 1,
    ), metrics


def update_multipliers(train_state, episode_costs, config=PPOConfig()):
    if not config.constrained:
        return train_state
    # Mean step cost is an unbiased estimator of the expected cost rate only
    # when every rollout has the same length. Variable-length CMDP budgets need
    # explicit per-episode normalization (documented in experiment reporting).
    gap = jnp.mean(episode_costs, axis=(0, 1)) - jnp.asarray(config.budgets)
    integral = jnp.clip(
        train_state.integral + config.ki * gap, -config.multiplier_cap, config.multiplier_cap
    )
    increment = config.kp * gap + integral + config.kd * (gap - train_state.previous_gap)
    multipliers = jnp.clip(train_state.multipliers + increment, 0, config.multiplier_cap)
    return train_state._replace(multipliers=multipliers, integral=integral, previous_gap=gap)


def collect(model, train_state, env_state, key, config, env, steps):
    def advance(carry, _):
        key, action_key, reset_key = jax.random.split(carry.key, 3)
        mean, value, cost_value = model.apply(
            train_state.params, carry.observation, carry.privileged
        )
        raw = gaussian_sample(action_key, mean, train_state.log_std)
        action = jnp.tanh(raw)
        logp = gaussian_log_probability(mean, train_state.log_std, raw)
        transition = env.step(carry.env_state, action)
        # Reset only done lanes, after capturing terminal values/rewards/costs.
        reset_state, reset_obs, reset_privileged = env.reset(reset_key)
        next_state = jax.tree.map(
            lambda fresh, old: jnp.where(transition.done, fresh, old), reset_state, transition.state
        )
        obs = jnp.where(transition.done[..., None], reset_obs, transition.observation)
        priv = jnp.where(transition.done[..., None], reset_privileged, transition.privileged)
        data = {
            "observation": carry.observation,
            "privileged": carry.privileged,
            "raw_action": raw,
            "logp": logp,
            "reward": transition.reward,
            "costs": transition.costs,
            "done": transition.done,
            "value": value,
            "cost_value": cost_value,
        }
        return Carry(next_state, obs, priv, key), data

    return jax.lax.scan(advance, env_state, None, length=steps)


def flatten_time_env(x):
    return x.reshape((-1, *x.shape[2:]))


def train(env, config=PPOConfig(), seed=0, steps=1000, rollout_steps=32, callback=None):
    """Train against vectorized functional reset/step with [env, ...] lanes."""
    model, train_state, optimizer = init_train_state(jax.random.PRNGKey(seed), config)
    env_state, obs, privileged = env.reset(jax.random.PRNGKey(seed + 1))
    carry = Carry(env_state, obs, privileged, jax.random.PRNGKey(seed + 2))
    history = []
    batch_size = env_state.core.vector.shape[0] if hasattr(env_state, "core") else 1
    updates = max(1, int(np.ceil(steps / (rollout_steps * batch_size))))
    for iteration in range(updates):
        carry, data = jax.jit(lambda c, s: collect(model, s, c, c.key, config, env, rollout_steps))(
            carry, train_state
        )
        _, next_value, next_cost_value = model.apply(
            train_state.params, carry.observation, carry.privileged
        )
        advantages, returns = gae(
            data["reward"], data["value"], data["done"], config.gamma, config.gae_lambda, next_value
        )
        cost_adv, cost_returns = jax.vmap(
            lambda r, v, d: gae(r, v, d, config.gamma, config.gae_lambda), in_axes=(2, 2, None)
        )(data["costs"], data["cost_value"], data["done"])
        batch = {
            name: flatten_time_env(data[name])
            for name in ("observation", "privileged", "raw_action", "logp")
        }
        batch.update(
            {
                "advantage": flatten_time_env(advantages),
                "return": flatten_time_env(returns),
                "cost_advantage": flatten_time_env(cost_adv),
                "cost_return": flatten_time_env(cost_returns),
            }
        )
        rng = np.random.default_rng(seed + iteration)
        for _epoch in range(config.epochs):
            indices = np.array_split(
                rng.permutation(batch["observation"].shape[0]), config.minibatches
            )
            for index in indices:
                minibatch = jax.tree.map(lambda value, idx=index: value[idx], batch)
                train_state, metrics = jax.jit(lambda s, b: update(model, optimizer, s, b, config))(
                    train_state, minibatch
                )
        train_state = update_multipliers(train_state, data["costs"], config)
        record = jax.tree.map(lambda x: np.asarray(x).item(), metrics)
        record.update(
            {
                "reward": float(jnp.mean(data["reward"])),
                "cost": np.asarray(jnp.mean(data["costs"], axis=(0, 1))).tolist(),
                "multiplier": np.asarray(train_state.multipliers).tolist(),
                "steps": (iteration + 1) * rollout_steps * batch_size,
            }
        )
        history.append(record)
        if callback:
            callback(record)
    return train_state, history
