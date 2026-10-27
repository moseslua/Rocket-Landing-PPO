"""Immutable hashed nominal and one-factor stress scenario fixtures."""

import hashlib
import json
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 2
STRESS_FACTORS = {
    "steady_wind": (0.0, 20.0),
    "thrust_shortfall": (0.0, 0.15),
    "observation_delay": (0.0, 0.15),
    "sensor_noise": (1.0, 5.0),
    "mid_descent_thrust_drop": (0.0, 1.0),
}


def scenario(seed, difficulty=1.0, stress=None):
    rng = np.random.default_rng(seed)
    data = {
        "seed": int(seed),
        "position": [
            float(rng.uniform(-400, 400) * difficulty),
            float(rng.uniform(-400, 400) * difficulty),
            float(rng.uniform(500, 2500) * difficulty),
        ],
        "velocity": [
            float(rng.uniform(-25, 25)),
            float(rng.uniform(-25, 25)),
            float(rng.uniform(-100, -30)),
        ],
        "mass": float(rng.uniform(1400, 1600)),
        "quaternion": [1.0, 0.0, 0.0, 0.0],
        "angular_rate": [0.0, 0.0, 0.0],
        "wind": [float(rng.uniform(-10, 10)), float(rng.uniform(-10, 10)), 0.0],
        "disturbance_seed": int(rng.integers(0, 2**32 - 1)),
        "stress": stress or {"factor": None, "level": 0.0},
    }
    factor = data["stress"].get("factor")
    if factor == "steady_wind":
        data["wind"] = [data["stress"]["level"], 0.0, 0.0]
    return data


def initialize(
    path, count=2000, start_seed=0, difficulty=1.0, *, allow_overwrite=False, stress=None
):
    path = Path(path)
    if path.exists() and not allow_overwrite:
        raise FileExistsError(f"refusing to overwrite frozen fixture: {path}")
    data = {
        "schema_version": SCHEMA_VERSION,
        "count": count,
        "difficulty": difficulty,
        "scenarios": [scenario(start_seed + i, difficulty, stress) for i in range(count)],
    }
    payload = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(payload).hexdigest()
    receipt = {
        "sha256": digest,
        "schema_version": SCHEMA_VERSION,
        "count": count,
        "difficulty": difficulty,
        "stress": stress,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    Path(str(path) + ".sha256.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def initialize_stress_grid(root, levels=6, count=500, start_seed=100000):
    root = Path(root)
    receipts = []
    for factor, (low, high) in STRESS_FACTORS.items():
        for level_index, level in enumerate(np.linspace(low, high, levels)):
            name = f"{factor}/level-{level_index}.json"
            receipt = initialize(
                root / name,
                count,
                start_seed + level_index * count,
                stress={"factor": factor, "level": float(level)},
            )
            receipts.append({"fixture": name, **receipt})
    return receipts


def load(path):
    path, receipt_path = Path(path), Path(str(path) + ".sha256.json")
    if not path.is_file() or not receipt_path.is_file():
        raise FileNotFoundError("scenario data and immutable hash receipt are required")
    raw = path.read_bytes()
    receipt = json.loads(receipt_path.read_text())
    if hashlib.sha256(raw).hexdigest() != receipt["sha256"]:
        raise ValueError(f"scenario hash mismatch: {path}; refusing evaluation")
    data = json.loads(raw)
    if data.get("schema_version") != SCHEMA_VERSION or len(data["scenarios"]) != receipt["count"]:
        raise ValueError("scenario schema/count mismatch")
    return data
