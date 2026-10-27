# Rocket Landing RL

`PLAN.md` is the research contract. The repository has an executable development baseline, not completed research simulators, validated experiments, or passed gates. Read `docs/development-pipeline.md` and `specs/full-pipeline.md` before expanding functionality.

## Commands

- `make setup`: locked Python 3.11 environment.
- `make check`: Ruff, all tests, numerical/LunarLander smoke checks.
- `make physics`: numerical/environment tests.
- `make train ARGS="training.steps=32"`: tiny development PPO run.
- `make tune ARGS="tuning.trials=1"`: development-only tuning.
- `uv run python scripts/evaluate.py FIXTURE --checkpoint CHECKPOINT`: immutable-fixture evaluation.

## Research constraints

- Do not call dependency or tiny CPU smoke checks training success, simulator validation, or a passed research gate.
- Sim-T is JAX; Sim-E is independent NumPy/SciPy. Never tune on Sim-E or frozen test sets.
- Residual development training uses explicitly named `jax_feedback`, not CVXPY B0. Do not conflate it with B3/B4 final results.
- Record configs, seeds, Git/source/lock hashes and simulator versions. Keep all results, including negative outcomes.
- Preserve frozen parameters after G4. Keep artifacts and credentials out of Git; retain final checkpoints separately.
- Current unsupported stress effects and incomplete E5/high-fidelity features must fail explicitly or be documented, never silently treated as nominal.
