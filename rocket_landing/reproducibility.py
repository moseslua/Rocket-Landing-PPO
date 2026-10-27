"""Run provenance and source identity; never fabricate missing Git metadata."""

import hashlib
import json
import platform
import subprocess
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import jax
import numpy as np
from omegaconf import OmegaConf


def git(root, *args):
    try:
        return subprocess.run(
            ["git", "-C", str(root), *args], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def hash_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_metadata(root):
    root = Path(root)
    files = sorted(
        p
        for folder in ("envs", "agents", "controllers", "rocket_landing")
        for p in (root / folder).rglob("*.py")
    )
    files += [root / "pyproject.toml", root / "uv.lock"]
    hashes = {str(p.relative_to(root)): hash_file(p) for p in files if p.is_file()}
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(), hashes


def receipt(config, seed, run_kind, root=None):
    root = Path(root or Path(__file__).resolve().parent.parent)
    status = git(root, "status", "--porcelain", "--untracked-files=normal")
    digest, sources = source_metadata(root)
    return {
        "run_kind": run_kind,
        "seed": int(seed),
        "created_utc": datetime.now(UTC).isoformat(),
        "resolved_config": OmegaConf.to_container(OmegaConf.create(config), resolve=True),
        "git": {
            "commit": git(root, "rev-parse", "HEAD"),
            "dirty": None if status is None else bool(status),
            "status": status,
        },
        "source_sha256": digest,
        "source_files": sources,
        "lock_sha256": hash_file(root / "uv.lock") if (root / "uv.lock").is_file() else None,
        "simulator_versions": {
            "sim_t": "6dof-training-development-0.1.0",
            "sim_e": "independent-development-0.1.0",
        },
        "dependencies": {
            name: version(name)
            for name in ("jax", "jaxlib", "flax", "optax", "numpy", "scipy", "cvxpy")
        },
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
    }


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, default=_json_default, allow_nan=False) + "\n")
    temporary.replace(path)


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value)}")
