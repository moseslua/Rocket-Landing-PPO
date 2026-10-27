"""Third-party dependency checks, not rocket simulator or training validation."""

import json
from collections.abc import Callable

import cvxpy as cp
import flax.linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
from scipy.integrate import solve_ivp


def check_jax_stack() -> str:
    """Check deterministic JAX execution and a small Flax/Optax update."""

    def loss(params: jax.Array) -> jax.Array:
        return jnp.mean((params - 3.0) ** 2)

    params = jnp.zeros(4)
    optimizer = optax.adam(0.1)
    state = optimizer.init(params)
    updates, state = optimizer.update(jax.grad(loss)(params), state, params)
    updated = optax.apply_updates(params, updates)
    if not np.all(np.isfinite(updated)) or not np.all(np.asarray(updated) > 0):
        raise RuntimeError("JAX/Optax update failed")

    model = nn.Dense(features=2)
    variables = model.init(jax.random.PRNGKey(0), jnp.ones((1, 4)))
    output = model.apply(variables, jnp.ones((1, 4)))
    if output.shape != (1, 2):
        raise RuntimeError("Flax model returned an unexpected shape")
    return f"{jax.default_backend()}: JAX, Flax, and Optax passed"


def check_guidance_stack() -> str:
    """Solve a small convex problem with the planned Clarabel solver."""
    variable = cp.Variable(2)
    problem = cp.Problem(
        cp.Minimize(cp.sum_squares(variable - np.array([1.0, -2.0]))),
        [variable >= 0],
    )
    problem.solve(solver=cp.CLARABEL)
    if problem.status != cp.OPTIMAL or not np.allclose(variable.value, [1.0, 0.0], atol=1e-6):
        raise RuntimeError(f"CVXPY/Clarabel failed: {problem.status}")
    return "CVXPY and Clarabel passed"


def check_evaluation_stack() -> str:
    """Check SciPy's adaptive RK45 integrator on free fall."""
    solution = solve_ivp(
        lambda _time, state: [state[1], -9.81],
        (0.0, 1.0),
        [100.0, 0.0],
        method="RK45",
        rtol=1e-10,
        atol=1e-12,
    )
    if not solution.success or not np.allclose(solution.y[:, -1], [95.095, -9.81], atol=1e-7):
        raise RuntimeError("SciPy adaptive integration failed")
    return "NumPy and SciPy RK45 passed"


def check_lunar_lander() -> str:
    """Check that seeded LunarLanderContinuous transitions are reproducible."""

    def rollout() -> tuple[np.ndarray, np.ndarray]:
        environment = gym.make("LunarLanderContinuous-v3")
        try:
            initial, _ = environment.reset(seed=42)
            next_state, *_ = environment.step(np.array([0.25, -0.5], dtype=np.float32))
            return initial, next_state
        finally:
            environment.close()

    first = rollout()
    second = rollout()
    if not all(np.array_equal(a, b) for a, b in zip(first, second, strict=True)):
        raise RuntimeError("Seeded LunarLander transitions are not reproducible")
    return "Gymnasium LunarLanderContinuous-v3 passed"


def run_checks() -> dict[str, str]:
    checks: dict[str, Callable[[], str]] = {
        "jax_stack": check_jax_stack,
        "guidance_stack": check_guidance_stack,
        "evaluation_stack": check_evaluation_stack,
        "lunar_lander": check_lunar_lander,
    }
    return {name: check() for name, check in checks.items()}


def main(_argv: list[str]) -> int:
    print(json.dumps(run_checks(), indent=2))
    return 0
