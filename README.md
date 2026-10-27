# Rocket Landing PPO

Executable development baseline for the constrained residual-RL study in [`PLAN.md`](PLAN.md). Includes JAX/SciPy numerical prototypes, 6-DOF development environments, guidance, PPO/SAC learning primitives, experiment commands, immutable fixtures and reporting.

> **Status:** **56 tests and the main CPU development workflows pass.** This is not a completed research study, externally validated high-fidelity simulator, or passed G1–G5 gate. Read [the development pipeline guide](docs/development-pipeline.md) for implemented behavior and explicit limitations.

## Setup and verification

Requires Python 3.11, uv and Make.

```bash
make setup       # Locked .venv and development tools
make check       # Ruff, all tests, third-party smoke checks
make hooks       # Optional local pre-commit hooks
make physics     # Numerical dynamics and environment tests
make benchmark   # Ideal-core numerical benchmark receipt
```

Local setup uses CPU JAX. A CUDA machine needs a separately verified driver-specific installation. Docker is configured but was not tested because its daemon was unavailable. Optional `cvxpygen` requires a working native toolchain; its `ecos` build failed with this Mac's SDK.

Weights & Biases defaults to offline in `.env.example`; `.env` is ignored. Experiment artifacts and checkpoints stay out of Git.

## Development workflows

```bash
# PPO / constrained PPO / explicit development residual / SAC
make train ARGS="training.steps=32 training.environments=2 training.output=outputs/dev"
make train ARGS="agent=ppo_lagrangian training.steps=32"
make train ARGS="agent=ppo_residual training.steps=32"
make train ARGS="agent=sac training.steps=64"

# Tiny Optuna trial on development data only
make tune ARGS="tuning.trials=1 tuning.steps_per_trial=32"

# Immutable development fixture, evaluation and figure
uv run python scripts/freeze_test_set.py results/dev-fixture.json --count 2
uv run python scripts/evaluate.py results/dev-fixture.json --backend sim_e --limit 1
uv run python scripts/evaluate.py results/dev-fixture.json --checkpoint outputs/dev/final.msgpack
uv run python scripts/make_figures.py outputs/dev/run.json --output results/dev.png
```

Experiment configs E0–E5 and a large unfrozen candidate profile are in `configs/`. `scripts/run_experiment.py` expands methods and penalty sweeps; large runs require explicit budget confirmation. These commands execute development runs, not research gates.

## Main implementation map

- `envs/sim_t/`: JAX point-mass and 6-DOF dynamics/environments.
- `envs/sim_e/`: independent NumPy/SciPy reference and higher-fidelity development model.
- `agents/`: PPO, PID-Lagrangian, SAC and replay.
- `controllers/`: finite-horizon CVXPY guidance and residual composition.
- `rocket_landing/`: training/evaluation orchestration, checkpoints, provenance, statistics and figures.
- `tests/`: analytic physics, determinism, environments, algorithms, checkpoints and workflow tests.
- `docs/development-pipeline.md`: behavior, commands and remaining research/engineering work.

Do not tune on Sim-E or frozen test sets. Do not mix simulator versions or alter frozen configurations after G4.
