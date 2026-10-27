"""Hashed-fixture evaluation with explicit policy composition and metric semantics."""

import time

import jax
import jax.numpy as jnp
import numpy as np

from agents.ppo import action_value
from controllers.guidance import GuidanceController
from envs.common.residual_env import ResidualConfig, baseline_action
from envs.common.rigid import RigidScenario
from envs.common.scenarios import load
from envs.sim_e import high_fidelity
from envs.sim_t import rigid_env


def summarize(records):
    if not records:
        raise ValueError("at least one episode is required")
    scores = np.array([r["landing_score"] for r in records])
    ratios = np.array([np.nan if r["fuel_ratio"] is None else r["fuel_ratio"] for r in records])
    return {
        "episodes": len(records),
        "success_rate": float(np.mean([r["success"] for r in records])),
        "constraint_violation_rate": float(np.mean([r["constraint_violation"] for r in records])),
        "constraint_rates": np.mean([r["constraint_rates"] for r in records], axis=0).tolist(),
        "fuel_used_kg_mean": float(np.mean([r["fuel_used_kg"] for r in records])),
        "fuel_ratio_mean": None if np.isnan(ratios).all() else float(np.nanmean(ratios)),
        "fuel_baseline_complete": not np.isnan(ratios).any(),
        "touchdown": {
            name: float(np.mean([r["touchdown"][name] for r in records]))
            for name in records[0]["touchdown"]
        },
        "landing_score_cvar_5pct": float(scores[scores <= np.quantile(scores, 0.05)].mean()),
        "smoothness_mean": float(np.mean([r["smoothness"] for r in records])),
        "inference_ms_mean": float(np.mean([r["inference_ms"] for r in records])),
        "all_episodes_recorded": True,
    }


def _vector(state):
    return np.asarray(state.vector if hasattr(state, "vector") else state.core.vector)


def _reason(state):
    return int(state.reason if hasattr(state, "reason") else state.core.reason)


def _touchdown(vector):
    q = vector[6:10]
    uv = np.cross(q[1:], [0, 0, 1])
    up = np.array([0, 0, 1]) + 2 * (q[0] * uv + np.cross(q[1:], uv))
    return {
        "vertical_speed": abs(float(vector[5])),
        "horizontal_speed": float(np.linalg.norm(vector[3:5])),
        "tilt_rad": float(np.arccos(np.clip(up[2], -1, 1))),
        "angular_rate_rad_per_s": float(np.linalg.norm(vector[10:13])),
        "position_error_m": float(np.linalg.norm(vector[:2])),
    }


def _costs(vector):
    elevation = np.arctan2(vector[2], max(np.linalg.norm(vector[:2]), 1e-6))
    tilt = _touchdown(vector)["tilt_rad"]
    return np.array(
        [
            elevation < np.deg2rad(15),
            tilt > np.deg2rad(20),
            np.linalg.norm(vector[10:13]) > np.deg2rad(15),
        ],
        dtype=float,
    )


def episode_record(
    state, steps, initial_mass, fuel_baseline=None, inference_ms=0.0, smoothness=0.0, costs=None
):
    vector, touchdown, reason = _vector(state), _touchdown(_vector(state)), _reason(state)
    success = (
        reason == 1
        and touchdown["vertical_speed"] <= 2
        and touchdown["horizontal_speed"] <= 1
        and touchdown["tilt_rad"] <= np.deg2rad(5)
        and touchdown["angular_rate_rad_per_s"] <= np.deg2rad(5)
        and touchdown["position_error_m"] <= 5
        and steps > 0
    )
    costs = _costs(vector) if costs is None else np.asarray(costs, dtype=float)
    used = initial_mass - float(vector[13])
    ratio = None if fuel_baseline is None else used / fuel_baseline
    return {
        "success": bool(success),
        "constraint_violation": bool(np.any(costs > 0)),
        "constraint_rates": (costs / max(steps, 1)).tolist(),
        "fuel_used_kg": used,
        "fuel_ratio": ratio,
        "touchdown": touchdown,
        "landing_score": float(-sum(touchdown.values())),
        "smoothness": float(smoothness),
        "inference_ms": float(inference_ms),
        "termination_reason": reason,
    }


def evaluate_fixture(
    path,
    model=None,
    train_state=None,
    backend="sim_t",
    limit=None,
    fuel_baselines=None,
    policy="zero",
    checkpoint_metadata=None,
    max_steps=1200,
):
    """Evaluate explicit policies on immutable scenarios, never training/tuning.

    Development residual checkpoints support their recorded jax_feedback
    baseline. CVXPY residual composition is not silently substituted. Stress
    effects unsupported by the selected backend are rejected explicitly.
    """
    if (model is None) != (train_state is None):
        raise ValueError("both model and train_state are required for learned evaluation")
    if model is None and policy not in ("zero", "b0"):
        raise ValueError("learned evaluation requires a loaded policy checkpoint")
    if policy == "learned" and model is None:
        raise ValueError("learned policy requires model and train_state")
    residual_config = None
    if checkpoint_metadata:
        agent = checkpoint_metadata["config"]["agent"]
        if agent.get("residual", False):
            residual_config = ResidualConfig(
                agent.get("residual_scale", 0.2), agent.get("baseline_backend", "jax_feedback")
            )
    data = load(path)
    records, controller = [], GuidanceController() if policy == "b0" else None
    for scenario in data["scenarios"][:limit]:
        stress = scenario.get("stress", {})
        if stress.get("factor") not in (None, "steady_wind"):
            raise NotImplementedError(
                f"{stress['factor']} evaluation is not implemented for this backend"
            )
        if backend == "sim_e":
            state = high_fidelity.reset(
                RigidScenario(
                    **{
                        key: scenario[key]
                        for key in (
                            "position",
                            "velocity",
                            "mass",
                            "quaternion",
                            "angular_rate",
                            "wind",
                        )
                    }
                )
            )
        elif backend == "sim_t":
            core = rigid_env.initial_from_arrays(
                scenario["position"],
                scenario["velocity"],
                scenario["quaternion"],
                scenario["angular_rate"],
                scenario["mass"],
                scenario["wind"],
                jax.random.PRNGKey(scenario["disturbance_seed"]),
            )
            state = rigid_env.EnvState(core, jnp.zeros((13, 24)), jnp.zeros(4), core.rng)
        else:
            raise ValueError("backend must be sim_t or sim_e")
        previous_action, smoothness, inference, costs = np.zeros(4), 0.0, 0.0, np.zeros(3)
        initial_mass, steps = float(_vector(state)[13]), 0
        while steps < max_steps and _reason(state) == 0:
            start = time.perf_counter()
            baseline = None
            if residual_config is not None and backend == "sim_t":
                baseline = np.asarray(baseline_action(state, residual_config))
            if model is not None:
                obs = rigid_env.observation(state)[None]
                if baseline is not None:
                    obs = obs.at[:, -4:].set(baseline)
                priv = rigid_env.privileged(state)[None]
                action = np.asarray(
                    action_value(model, train_state, obs, priv, jax.random.PRNGKey(0), True)[0][0]
                )
            elif controller is not None:
                action = controller.act(_vector(state))
            else:
                action = np.zeros(4)
            if baseline is not None:
                action = np.clip(baseline + residual_config.scale * np.clip(action, -1, 1), -1, 1)
            elif residual_config is not None:
                raise ValueError(
                    "residual learned evaluation currently supports Sim-T jax_feedback only"
                )
            inference += (time.perf_counter() - start) * 1000
            smoothness += float(np.sum((action - previous_action) ** 2))
            previous_action = action
            if backend == "sim_t":
                transition = rigid_env.step(state, jnp.asarray(action))
                costs += np.asarray(transition.costs)
                state = transition.state
            else:
                state = high_fidelity.integrate(state, action)
                costs += _costs(_vector(state))
            steps += 1
        baseline_fuel = (
            None
            if fuel_baselines is None
            else fuel_baselines.get(str(scenario["seed"]), fuel_baselines.get(scenario["seed"]))
        )
        record = episode_record(
            state, steps, initial_mass, baseline_fuel, inference / max(steps, 1), smoothness, costs
        )
        record.update({"steps": steps, "seed": scenario["seed"], "stress": stress})
        records.append(record)
    return {
        "summary": summarize(records),
        "episodes": records,
        "backend": backend,
        "fixture": str(path),
        "policy": policy,
        "checkpoint_metadata": checkpoint_metadata,
        "max_steps": max_steps,
    }


def bootstrap_difference(first, second, seed=0, samples=1000):
    first, second = np.asarray(first), np.asarray(second)
    if first.shape != second.shape or first.ndim != 1:
        raise ValueError("paired metric arrays must be one-dimensional and equal length")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(first), size=(samples, len(first)))
    differences = (second - first)[indices].mean(axis=1)
    return {
        "mean": float((second - first).mean()),
        "ci95": np.quantile(differences, [0.025, 0.975]).tolist(),
        "probability_improvement": float(np.mean(differences > 0)),
    }
