"""Training orchestration with explicit PPO, SAC, residual and classical paths."""

import copy
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from agents.ppo import PPOConfig, train
from agents.replay import ReplayBuffer
from agents.sac import SACConfig, update_state
from agents.sac import init_train_state as init_sac
from envs.common.adapter import RigidLander
from envs.common.residual_env import ResidualConfig, ResidualRigid
from envs.sim_t import rigid_env
from rocket_landing.checkpoints import save
from rocket_landing.reproducibility import atomic_json, receipt


class VectorRigid:
    def __init__(self, environments, config):
        self.environments, self.config = environments, config

    def reset(self, key):
        return rigid_env.batch_reset(jax.random.split(key, self.environments), self.config)

    def step(self, state, action):
        return rigid_env.batch_step(state, action, self.config)


def make_config(raw):
    return PPOConfig(
        **{
            key: value
            for key, value in raw["agent"].items()
            if key in PPOConfig.__dataclass_fields__
        }
    )


def make_environment(raw, environments):
    env_config = rigid_env.RigidConfig(
        **{
            key: value
            for key, value in raw["env"].items()
            if key in rigid_env.RigidConfig.__dataclass_fields__
        }
    )
    if raw["agent"].get("residual", False):
        return ResidualRigid(
            environments,
            env_config,
            ResidualConfig(
                raw["agent"].get("residual_scale", 0.2),
                raw["agent"].get("baseline_backend", "jax_feedback"),
            ),
        )
    return VectorRigid(environments, env_config)


def run_training(config, output=None):
    raw = copy.deepcopy(config)
    seed, algorithm = int(raw["seed"]), raw["agent"]["name"]
    environments, steps = int(raw["training"]["environments"]), int(raw["training"]["steps"])
    if environments < 1 or steps < 1:
        raise ValueError("training steps and environments must be positive")
    environment = make_environment(raw, environments)
    run = {
        "receipt": receipt(raw, seed, "development_training"),
        "config": raw,
        "algorithm": algorithm,
        "status": "completed_development_smoke",
        "history": [],
    }
    if raw["agent"].get("residual", False) and algorithm not in ("ppo", "ppo_lagrangian"):
        raise ValueError("residual training currently supports PPO variants only")
    if algorithm in ("ppo", "ppo_lagrangian"):
        state, history = train(
            environment, make_config(raw), seed, steps, int(raw["training"]["rollout_steps"])
        )
        run["history"] = history
        if output:
            checkpoint = Path(output) / "final.msgpack"
            save(checkpoint, state, {"algorithm": algorithm, "config": raw})
            run["checkpoint"] = str(checkpoint)
    elif algorithm == "sac":
        run.update(_run_sac(raw, environment, steps))
    elif algorithm == "b0":
        run["history"] = _run_b0(raw, steps)
    else:
        raise ValueError(f"unsupported algorithm: {algorithm}")
    if output:
        atomic_json(Path(output) / "run.json", run)
    return run


def _run_b0(raw, steps):
    from controllers.guidance import GuidanceController

    environment, controller = RigidLander(seed=raw["seed"]), GuidanceController()
    _, info = environment.reset(seed=raw["seed"])
    history = []
    for step in range(steps):
        action = controller.act(info["privileged"][:14])
        _, reward, done, _, info = environment.step(action)
        history.append(
            {"steps": step + 1, "reward": reward, "termination": info["termination_reason"]}
        )
        if done:
            _, info = environment.reset()
    return history


def _run_sac(raw, environment, steps):
    if raw["agent"].get("residual", False):
        raise ValueError(
            "residual SAC is not implemented; use an explicitly named supported variant"
        )
    config = SACConfig(
        **{
            key: value
            for key, value in raw["agent"].items()
            if key in SACConfig.__dataclass_fields__
        }
    )
    actor, critic, state, optimizers = init_sac(jax.random.PRNGKey(raw["seed"]), config)
    replay = ReplayBuffer(
        int(raw["training"].get("replay_capacity", 10000)),
        config.observation_size,
        config.action_size,
        raw["seed"],
    )
    states, observations, privileged = environment.reset(jax.random.PRNGKey(raw["seed"] + 1))
    key, history = jax.random.PRNGKey(raw["seed"] + 2), []
    for step in range(steps):
        key, sample_key = jax.random.split(key)
        means, log_stds = actor.apply(state.actor_params, observations)
        actions = jnp.tanh(means + jax.random.normal(sample_key, means.shape) * jnp.exp(log_stds))
        transitions = environment.step(states, actions)
        reset_states, reset_obs, reset_priv = environment.reset(
            jax.random.PRNGKey(raw["seed"] + step + 3)
        )
        for index in range(environment.environments):
            done = bool(transitions.done[index])
            replay.add(
                np.asarray(observations[index]),
                np.asarray(actions[index]),
                float(transitions.reward[index]),
                np.asarray(reset_obs[index] if done else transitions.observation[index]),
                done,
                np.asarray(transitions.costs[index]),
            )
        states = jax.tree.map(
            lambda fresh, old, done=transitions.done: jnp.where(done, fresh, old),
            reset_states,
            transitions.state,
        )
        observations = jnp.where(transitions.done[..., None], reset_obs, transitions.observation)
        if replay.size >= config.batch_size:
            key, update_key = jax.random.split(key)
            state, metrics = update_state(
                actor,
                critic,
                state,
                optimizers,
                jax.tree.map(jnp.asarray, replay.sample(config.batch_size)),
                update_key,
                config,
            )
        if step % max(1, steps // 10) == 0:
            history.append(
                {
                    "steps": step + 1,
                    "reward": float(jnp.mean(transitions.reward)),
                    "critic_loss": float(metrics["critic_loss"])
                    if replay.size >= config.batch_size
                    else None,
                }
            )
    return {
        "history": history,
        "replay_size": replay.size,
        "note": "SAC development loop; no research convergence claim",
    }
