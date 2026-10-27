"""Explicit experiment matrix, budget confirmation, and resumable run artifacts."""

from pathlib import Path

import hydra
from omegaconf import OmegaConf

from rocket_landing.reproducibility import atomic_json
from rocket_landing.training import run_training

METHOD_AGENTS = {
    "b0": "b0",
    "ppo": "ppo",
    "ppo_lagrangian": "ppo_lagrangian",
    "ppo_residual": "ppo_residual",
    "ppo_residual_lagrangian": "ppo_residual_lagrangian",
}


def expand(config):
    experiment = config["experiment"]
    methods = experiment.get("methods", [config["agent"]["name"]])
    runs = []
    for method in methods:
        if method not in METHOD_AGENTS:
            raise ValueError(f"unknown method {method}")
        if config["agent"]["name"] != method:
            # Matrix expansion uses each method's canonical config, never only
            # the algorithm family selected on the command line.
            agent = OmegaConf.to_container(
                OmegaConf.load(
                    Path(__file__).resolve().parent.parent
                    / "configs/agent"
                    / f"{METHOD_AGENTS[method]}.yaml"
                )
            )
        else:
            agent = config["agent"]
        penalties = (
            experiment.get("penalty_weights", [0.1, 1.0, 10.0, 100.0])
            if experiment["name"].startswith("e3")
            and not agent.get("constrained", False)
            and method != "b0"
            else [agent.get("penalty", config["env"].get("penalty", 0.0))]
        )
        for penalty in penalties:
            for seed in range(experiment["seeds"]):
                run = {
                    **config,
                    "agent": agent,
                    "seed": seed,
                    "env": {**config["env"], "penalty": penalty},
                }
                runs.append(run)
    return runs


def execute(config, output):
    output = Path(output)
    runs = []
    for index, run_config in enumerate(expand(config)):
        destination = output / f"run-{index:04d}-seed-{run_config['seed']}"
        run = run_training(run_config, destination)
        runs.append(
            {
                "index": index,
                "seed": run_config["seed"],
                "method": run_config["agent"]["name"],
                "history": run["history"],
                "status": run["status"],
                "path": str(destination),
            }
        )
    atomic_json(
        output / "experiment.json",
        {"experiment": config["experiment"], "runs": runs, "research_gate_passed": False},
    )
    return runs


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config):
    raw = OmegaConf.to_container(config, resolve=True)
    if raw["training"]["steps"] > 100000 and not raw.get("confirm_large_run", False):
        raise ValueError("large research run requires confirm_large_run=true and a compute review")
    runs = execute(raw, raw["training"]["output"])
    print({"runs": len(runs), "output": raw["training"]["output"], "research_gate_passed": False})


if __name__ == "__main__":
    main()
