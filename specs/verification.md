# Verified development baseline

Date: 2026-10-27

## Observed verification

- `make check`: Ruff check/format, **56 passed**, third-party JAX/Flax/Optax, CVXPY/Clarabel, SciPy and LunarLander smoke checks.
- Real Hydra PPO run: small CPU rollout/update run with msgpack checkpoint and resolved run receipt.
- Real SAC run: replay insertion and twin-critic/actor/temperature updates.
- Real residual PPO run: explicit `jax_feedback` baseline composition.
- Optuna: one tiny development-only trial completed.
- Evaluation: immutable nominal Sim-T and independent Sim-E fixture paths; learned checkpoint evaluation; zero development success is not a research result.
- Figures: generated from saved run JSON.
- Residual learned checkpoint: trained and evaluated with explicit baseline composition.
- Wheel: built and all four top-level packages imported from an external install target.
- Lock/dependency checks: `uv lock --check --offline`, `uv pip check` passed.
- rliable: imports, IQM, bootstrap reproducibility and paired probability-of-improvement tested.

## Not claimed

No G1–G5 gate, convergence, 95% success, 500M-step training, GPU throughput, external physics validation, tuned B0, final frozen test set, paper-quality figure, or supported hypothesis.

## Remaining implementation/research work

See `docs/development-pipeline.md`: full-fidelity effects and persistent states, vectorized CVXPY residual baseline, observation history/GRU/normalization/curriculum, complete E5 expansion and unsupported stress backends, LunarLander PPO training, large seeded runs and research-gate validation. Docker/CUDA are not verified.

## Local artifacts

Ignored under `outputs/` and `results/`. No credentials, checkpoints or generated test sets are committed.
