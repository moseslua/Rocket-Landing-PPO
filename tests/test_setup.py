import cvxpy as cp
import gymnasium as gym
import numpy as np

from rocket_landing import __version__
from rocket_landing.cli import environment_info, main
from rocket_landing.smoke import check_evaluation_stack, check_guidance_stack, check_jax_stack


def test_package_and_doctor() -> None:
    assert __version__ == "0.1.0"
    info = environment_info()
    assert info["python"].startswith("3.11.")
    assert info["jax_backend"] in {"cpu", "gpu", "tpu"}
    assert main(["doctor"]) == 0


def test_jax_flax_optax() -> None:
    assert "passed" in check_jax_stack()


def test_cvxpy_with_clarabel() -> None:
    assert "CLARABEL" in cp.installed_solvers()
    assert "passed" in check_guidance_stack()


def test_scipy_adaptive_integration() -> None:
    assert "passed" in check_evaluation_stack()


def test_lunar_lander_continuous_is_available() -> None:
    environment = gym.make("LunarLanderContinuous-v3")
    try:
        observation, info = environment.reset(seed=42)
        assert observation.shape == (8,)
        assert isinstance(info, dict)
        assert environment.action_space.shape == (2,)
    finally:
        environment.close()


def test_seeded_lunar_lander_reset() -> None:
    first = gym.make("LunarLanderContinuous-v3").reset(seed=42)[0]
    second = gym.make("LunarLanderContinuous-v3").reset(seed=42)[0]
    np.testing.assert_array_equal(first, second)
