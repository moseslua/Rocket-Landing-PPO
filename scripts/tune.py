"""Optuna tuning entry point; uses development/validation sets only."""

import hydra
import optuna
from omegaconf import OmegaConf

from rocket_landing.training import run_training


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config):
    raw = OmegaConf.to_container(config, resolve=True)
    if raw["tuning"].get("use_evaluation_sim", False):
        raise ValueError("Sim-E and frozen test sets are forbidden for tuning")
    study = optuna.create_study(direction="maximize", study_name=raw["experiment"]["name"])

    def objective(trial):
        candidate = OmegaConf.create(raw)
        candidate.agent.learning_rate = trial.suggest_float("learning_rate", 1e-5, 3e-3, log=True)
        candidate.training.steps = raw["tuning"]["steps_per_trial"]
        candidate.training.environments = raw["tuning"]["environments"]
        result = run_training(OmegaConf.to_container(candidate, resolve=True))
        return result["history"][-1]["reward"] if result["history"] else float("-inf")

    study.optimize(objective, n_trials=raw["tuning"]["trials"])
    print(
        {
            "best_value": study.best_value,
            "best_params": study.best_params,
            "scope": "development/validation only; not Sim-E tuning",
        }
    )


if __name__ == "__main__":
    main()
