"""Hydra training entry point: uv run python scripts/train.py experiment=e0."""

from pathlib import Path

import hydra
from omegaconf import OmegaConf

from rocket_landing.training import run_training


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config):
    raw = OmegaConf.to_container(config, resolve=True)
    output = Path(raw["training"].get("output", "outputs/training"))
    output.mkdir(parents=True, exist_ok=True)
    run = run_training(raw, output)
    print(
        {
            "status": run["status"],
            "algorithm": run["algorithm"],
            "output": str(output),
            "updates": len(run["history"]),
        }
    )


if __name__ == "__main__":
    main()
