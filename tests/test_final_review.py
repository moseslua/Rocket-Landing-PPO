"""Regression tests for final scientific-contract review."""

import numpy as np
import pytest

from envs.common.scenarios import initialize, initialize_stress_grid
from envs.sim_e.high_fidelity import touchdown_reason
from rocket_landing.evaluation import evaluate_fixture
from rocket_landing.runner import expand
from rocket_landing.training import run_training
from scripts.evaluate import restore
from tests.test_workflows import config


def test_sim_e_distinguishes_landing_from_crash():
    safe = np.array([0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1700.0])
    crash = safe.copy()
    crash[5] = -20
    assert touchdown_reason(safe) == 1
    assert touchdown_reason(crash) == 2


def test_stress_grid_is_hashed_and_unsupported_effects_are_not_nominal(tmp_path):
    receipts = initialize_stress_grid(tmp_path, levels=2, count=1)
    assert len(receipts) == 10
    with pytest.raises(NotImplementedError, match="observation_delay"):
        evaluate_fixture(tmp_path / "observation_delay/level-1.json", max_steps=1)
    result = evaluate_fixture(tmp_path / "steady_wind/level-1.json", max_steps=1)
    assert result["episodes"][0]["stress"]["factor"] == "steady_wind"


def test_residual_checkpoint_evaluation_composes_baseline(tmp_path):
    raw = config("ppo_residual")
    run_training(raw, tmp_path)
    fixture = tmp_path / "fixture.json"
    initialize(fixture, count=1)
    model, state, metadata = restore(tmp_path / "final.msgpack")
    result = evaluate_fixture(
        fixture, model, state, max_steps=1, policy="learned", checkpoint_metadata=metadata
    )
    assert result["summary"]["episodes"] == 1
    assert metadata["config"]["agent"]["residual"]


def test_main_experiment_matrix_expands_all_methods():
    raw = config()
    raw["experiment"] = {
        "name": "e1_main",
        "methods": ["b0", "ppo", "ppo_lagrangian", "ppo_residual", "ppo_residual_lagrangian"],
        "seeds": 1,
    }
    runs = expand(raw)
    assert len(runs) == 5
    assert runs[-1]["agent"]["residual"]
    assert runs[-1]["agent"]["constrained"]


def test_fixture_constraint_rates_are_per_step(tmp_path):
    fixture = tmp_path / "fixture.json"
    initialize(fixture, count=1)
    result = evaluate_fixture(fixture, max_steps=2)
    assert all(0 <= rate <= 1 for rate in result["summary"]["constraint_rates"])
