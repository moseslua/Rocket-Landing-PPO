"""End-to-end development workflow tests; deliberately tiny budgets."""

from pathlib import Path

import jax
import numpy as np
import pytest
from omegaconf import OmegaConf

from agents.ppo import PPOConfig, init_train_state
from envs.common.residual_env import ResidualConfig, ResidualRigid, residual_action
from envs.common.scenarios import initialize
from envs.sim_t.rigid_env import RigidConfig
from rocket_landing.evaluation import evaluate_fixture
from rocket_landing.figures import learning_curves
from rocket_landing.runner import expand
from rocket_landing.statistics import aggregate
from rocket_landing.training import run_training

ROOT = Path(__file__).resolve().parent.parent


def config(agent="ppo"):
    raw = OmegaConf.to_container(OmegaConf.load(ROOT / "configs/config.yaml"))
    raw["env"] = OmegaConf.to_container(OmegaConf.load(ROOT / "configs/env/rigid.yaml"))
    raw["agent"] = OmegaConf.to_container(OmegaConf.load(ROOT / f"configs/agent/{agent}.yaml"))
    raw["experiment"] = OmegaConf.to_container(OmegaConf.load(ROOT / "configs/experiment/e0.yaml"))
    raw["training"].update({"steps": 4, "environments": 1, "rollout_steps": 2})
    return raw


@pytest.mark.parametrize("agent", ["sac", "ppo_residual"])
def test_sac_and_residual_training(agent):
    run = run_training(config(agent))
    assert len(run["history"]) > 0
    assert all(np.isfinite(row["reward"]) for row in run["history"])


def test_residual_baseline_is_exposed_and_composed():
    environment = ResidualRigid(1, RigidConfig(), ResidualConfig())
    state, obs, _ = environment.reset(jax.random.PRNGKey(0))
    correction = np.zeros(4)
    single_state = jax.tree.map(lambda value: value[0], state)
    applied, baseline = residual_action(single_state, correction)
    np.testing.assert_allclose(applied, baseline)
    assert obs.shape == (1, 24)
    assert not np.allclose(obs[0, -4:], 0)


def test_rliable_aggregation_and_figure(tmp_path):
    result = aggregate({"a": [1.0, 2.0, 3.0], "b": [2.0, 3.0, 4.0]}, seeds=3, reps=10)
    assert result["a"]["iqm"] == 2
    run = run_training(config())
    run["config"]["experiment"]["name"] = "test"
    path = learning_curves([run], tmp_path / "learning.png")
    assert path.stat().st_size > 100


def test_experiment_expansion_preserves_penalty_sweep():
    raw = config()
    raw["experiment"] = {"name": "e3_constraints", "seeds": 2, "penalty_weights": [1.0, 10.0]}
    runs = expand(raw)
    assert len(runs) == 4
    assert [run["seed"] for run in runs] == [0, 1, 0, 1]
    assert [run["env"]["penalty"] for run in runs] == [1.0, 1.0, 10.0, 10.0]


def test_fixture_evaluation_uses_backend_and_checkpoint(tmp_path):
    raw = config()
    run_training(raw, tmp_path)
    fixture = tmp_path / "fixture.json"
    initialize(fixture, count=1)
    agent_config = PPOConfig(
        **{
            key: value
            for key, value in raw["agent"].items()
            if key in PPOConfig.__dataclass_fields__
        }
    )
    model, fresh, _ = init_train_state(jax.random.PRNGKey(0), agent_config)
    from rocket_landing.checkpoints import load

    state, _ = load(tmp_path / "final.msgpack", fresh)
    result = evaluate_fixture(fixture, model, state, limit=1, max_steps=2, policy="learned")
    assert result["policy"] == "learned"
    assert result["episodes"][0]["fuel_ratio"] is None
    assert not result["summary"]["fuel_baseline_complete"]
