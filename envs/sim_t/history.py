"""Physics-rate history primitive, not yet wired into a noisy observation model."""

import jax.numpy as jnp

# Immutable configuration defaults are safe to share between JIT calls.
# Ruff B008: frozen dataclasses intentionally form part of the static JIT signature.


def init_history(observation, length=13):
    if length < 1:
        raise ValueError("history length must be positive")
    return jnp.broadcast_to(observation, (length, *observation.shape))


def push_history(history, observation):
    """Newest value is index zero; initial values pad pre-episode history."""
    return jnp.concatenate((observation[None], history[:-1]), axis=0)


def delayed(history, delay_seconds, physics_dt=0.005):
    """Round to nearest tick (ties up); invalid requests produce NaNs under JIT."""
    delay = jnp.asarray(delay_seconds)
    index = jnp.floor(delay / physics_dt + 0.5).astype(jnp.int32)
    valid = jnp.isfinite(delay) & (delay >= 0) & (delay <= (len(history) - 1) * physics_dt)
    value = history[jnp.clip(index, 0, len(history) - 1)]
    return jnp.where(valid, value, jnp.nan)
