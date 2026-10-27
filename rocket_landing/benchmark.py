"""Reproducible developer numerical benchmark, NOT end-to-end PPO throughput.

Uses only developer cases and analytic solutions. Never reads frozen test sets.
"""

import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

import jax
import jax.numpy as jnp
import numpy as np

from envs.common import Clock, Scenario, Vehicle
from envs.sim_e import SIM_E_VERSION
from envs.sim_e import point_mass as sim_e
from envs.sim_t import SIM_T_VERSION
from envs.sim_t import point_mass as sim_t


@dataclass(frozen=True)
class BenchmarkConfig:
    environments: int = 64
    steps: int = 100
    repeats: int = 3
    seed: int = 0

    def __post_init__(self):
        for name in ("environments", "steps", "repeats"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.seed, int) or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an unsigned 32-bit integer")
        if self.steps * Clock().policy_dt >= Clock().timeout:
            raise ValueError("benchmark must end before timeout")


def _git(root, *args):
    try:
        return subprocess.run(
            ["git", "-C", str(root), *args], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def _metadata():
    root = Path(__file__).resolve().parent.parent
    lock = root / "uv.lock"
    status = _git(root, "status", "--porcelain", "--untracked-files=normal")
    sources = sorted(
        path
        for directory in ("envs", "rocket_landing", "tests")
        for path in (root / directory).rglob("*.py")
    )
    sources.extend(root / name for name in ("pyproject.toml", "uv.lock"))
    hashes = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sources
        if path.is_file()
    }
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return {
        "utc": datetime.now(UTC).isoformat(),
        "git": {
            "commit": _git(root, "rev-parse", "HEAD"),
            "dirty": None if status is None else bool(status),
            "status": status,
        },
        "lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest() if lock.is_file() else None,
        "source_sha256": digest,
        "source_files": hashes,
        "simulator_versions": {"sim_t": SIM_T_VERSION, "sim_e": SIM_E_VERSION},
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(d) for d in jax.devices()],
        "jax_x64_enabled": bool(jax.config.x64_enabled),
        "dependencies": {
            name: version(name) for name in ("jax", "jaxlib", "numpy", "scipy", "rocket-landing-rl")
        },
    }


def _analytic_burn(scenario, force, duration, vehicle):
    force = np.asarray(force, dtype=np.float64)
    flow = np.linalg.norm(force) / (vehicle.isp * vehicle.g0)
    final_mass = scenario.mass - flow * duration
    log_ratio = np.log(scenario.mass / final_mass)
    displacement_factor = (flow * duration - final_mass * log_ratio) / flow**2
    gravity = np.array([0.0, 0.0, -vehicle.gravity])
    position = (
        np.asarray(scenario.position)
        + np.asarray(scenario.velocity) * duration
        + force * displacement_factor
        + gravity * duration**2 / 2
    )
    velocity = np.asarray(scenario.velocity) + force / flow * log_ratio + gravity * duration
    return position, velocity, final_mass


def run_benchmark(config):
    vehicle, clock = Vehicle(), Clock()
    keys = jax.random.split(jax.random.PRNGKey(config.seed), config.environments)
    initial = jax.vmap(sim_t.sample_reset)(keys)
    # Keep cases far from the ground. This benchmark measures active integration,
    # not cheap repeated absorbing steps; terminal localization is tested elsewhere.
    initial = initial._replace(position=initial.position.at[:, 2].add(10000))
    thrust = np.array([1000.0, -500.0, 18000.0], dtype=np.float32)
    actions = jnp.broadcast_to(jnp.asarray(thrust), (config.environments, config.steps, 3))
    simulate = jax.jit(sim_t.batch_rollout)
    start = time.perf_counter()
    warm = jax.block_until_ready(simulate(initial, actions))
    warmup = time.perf_counter() - start
    times = []
    final = warm[0]
    for _ in range(config.repeats):
        start = time.perf_counter()
        final, _history = jax.block_until_ready(simulate(initial, actions))
        times.append(time.perf_counter() - start)
    if np.any(np.asarray(final.reason) != 0):
        raise RuntimeError("benchmark encountered terminal cases; not a valid active-step timing")
    if not all(np.isfinite(np.asarray(field)).all() for field in final):
        raise RuntimeError("benchmark produced nonfinite state")

    duration = config.steps * clock.policy_dt
    scenario = Scenario(position=(0, 0, 10000), velocity=(2, -1, -5), mass=1500)
    controls = np.tile(thrust.astype(np.float64), (config.steps, 1))
    exact_position, exact_velocity, exact_mass = _analytic_burn(scenario, thrust, duration, vehicle)
    comparison = []
    for method in ("RK45", "DOP853"):
        for rtol in (1e-6, 1e-9):
            samples = []
            for _ in range(config.repeats):
                start = time.perf_counter()
                reference, _, nfev = sim_e.rollout(
                    sim_e.reset(scenario),
                    controls,
                    method=method,
                    rtol=rtol,
                    atol=rtol * 0.01,
                    with_nfev=True,
                )
                samples.append(time.perf_counter() - start)
            comparison.append(
                {
                    "method": method,
                    "rtol": rtol,
                    "atol": rtol * 0.01,
                    "seconds": samples,
                    "median_seconds": statistics.median(samples),
                    "nfev": nfev,
                    "position_error_m": float(np.max(np.abs(reference.position - exact_position))),
                    "velocity_error_m_per_s": float(
                        np.max(np.abs(reference.velocity - exact_velocity))
                    ),
                    "mass_error_kg": abs(reference.mass - exact_mass),
                }
            )
    matched_initial = sim_t.reset(scenario)
    matched_final, _ = jax.block_until_ready(jax.jit(sim_t.rollout)(matched_initial, controls))
    median = statistics.median(times)
    return {
        "kind": "developer-numerical-benchmark",
        "limitations": [
            "Ideal 3-DOF dynamics only; not PPO or full environment throughput",
            "Developer cases only; no frozen evaluation sets used",
            "Reference method comparison is not a final Sim-E solver selection",
        ],
        "config": {
            **asdict(config),
            "vehicle": asdict(vehicle),
            "clock": asdict(clock),
            "dtype": "float32",
            "thrust_enu_n": thrust.tolist(),
            "reference_scenario": asdict(scenario),
            "training_reset": {
                "z_m": [10100, 10200],
                "vz_m_per_s": [-10, 0],
                "mass_kg": [1400, 1600],
                "other_components": 0,
            },
        },
        "metadata": _metadata(),
        "sim_t": {
            "warmup_compile_seconds": warmup,
            "steady_seconds": times,
            "median_seconds": median,
            "policy_transitions_per_second": config.environments * config.steps / median,
            "physics_substeps_per_second": config.environments
            * config.steps
            * clock.substeps
            / median,
            "position_error_m": float(
                np.max(np.abs(np.asarray(matched_final.position) - exact_position))
            ),
            "velocity_error_m_per_s": float(
                np.max(np.abs(np.asarray(matched_final.velocity) - exact_velocity))
            ),
            "mass_error_kg": abs(float(matched_final.mass) - exact_mass),
        },
        "reference_comparison": comparison,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environments", type=int, default=64)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    config = BenchmarkConfig(args.environments, args.steps, args.repeats, args.seed)
    output = args.output or Path("results") / f"prototype-{uuid4().hex}.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite benchmark receipt: {output}")
    report = run_benchmark(config)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "receipt": str(output),
                "sim_t": report["sim_t"],
                "reference_comparison": report["reference_comparison"],
            },
            indent=2,
        )
    )
    return 0
