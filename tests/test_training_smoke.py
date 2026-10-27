"""Small executable training/reproducibility checks, not learning-quality gates."""

from pathlib import Path

import jax
import numpy as np
from omegaconf import OmegaConf

from agents.ppo import PPOConfig, gae, init_train_state, train
from envs.common.scenarios import initialize
from envs.sim_t.rigid_env import RigidConfig, batch_reset
from rocket_landing.checkpoints import load, save
from rocket_landing.evaluation import evaluate_fixture
from rocket_landing.training import VectorRigid, run_training

ROOT = Path(__file__).resolve().parent.parent


def development_config():
    raw = OmegaConf.to_container(OmegaConf.load(ROOT / "configs/config.yaml"), resolve=False)
    raw["env"] = OmegaConf.to_container(OmegaConf.load(ROOT / "configs/env/rigid.yaml"))
    raw["agent"] = OmegaConf.to_container(OmegaConf.load(ROOT / "configs/agent/ppo.yaml"))
    raw["experiment"] = OmegaConf.to_container(OmegaConf.load(ROOT / "configs/experiment/e0.yaml"))
    raw["training"].update({"steps": 4, "environments": 1, "rollout_steps": 2})
    return raw


def test_privileged_shape_matches_actual_environment():
    states, observations, privileged = batch_reset(jax.random.split(jax.random.PRNGKey(0), 2))
    assert observations.shape[-1] == PPOConfig().observation_size
    assert privileged.shape[-1] == PPOConfig().privileged_size


def test_gae_uses_explicit_bootstrap_value():
    advantages, returns = gae(
        np.array([0.0, 0.0]), np.array([1.0, 1.0]), np.array([0.0, 0.0]), 1.0, 1.0, np.array(10.0)
    )
    np.testing.assert_allclose(returns, [10.0, 10.0])


def test_ppo_training_and_safe_checkpoint_round_trip(tmp_path):
    environment = VectorRigid(1, RigidConfig())
    config = PPOConfig(epochs=1, minibatches=1)
    state, history = train(environment, config, steps=4, rollout_steps=2)
    assert len(history) == 2 and history[-1]["steps"] == 4
    path = tmp_path / "final.msgpack"
    save(path, state, {"seed": 0})
    _, fresh, _ = init_train_state(jax.random.PRNGKey(0), config)
    restored, metadata = load(path, fresh)
    assert metadata == {"seed": 0}
    assert int(restored.update) == int(state.update)
    np.testing.assert_array_equal(restored.log_std, state.log_std)


def test_training_command_writes_receipt_and_frozen_evaluation(tmp_path):
    raw = development_config()
    run = run_training(raw, tmp_path)
    assert (tmp_path / "run.json").is_file()
    assert (tmp_path / "final.msgpack").is_file()
    assert run["receipt"]["resolved_config"]["training"]["steps"] == 4
    fixture = tmp_path / "scenarios.json"
    initialize(fixture, count=1)
    evaluation = evaluate_fixture(fixture, limit=1, max_steps=2)
    assert evaluation["summary"]["episodes"] == 1
    assert "success_rate" in evaluation["summary"]
