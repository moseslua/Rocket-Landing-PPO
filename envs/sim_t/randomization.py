"""Pure sampled disturbance helpers. Step keys are never reused."""

import jax
import jax.numpy as jnp


def split_step(key, step):
    return jax.random.fold_in(key, step)


def ou_step(previous, key, intensity=1.0, correlation_time=1.0, dt=0.005):
    noise = jax.random.normal(key, previous.shape)
    decay = jnp.exp(-dt / correlation_time)
    diffusion = intensity * jnp.sqrt(1 - decay**2)
    return previous * decay + noise * diffusion


def clipped_uniform(key, low, high, difficulty=1.0):
    return jax.random.uniform(key, (), minval=low * difficulty, maxval=high * difficulty)
