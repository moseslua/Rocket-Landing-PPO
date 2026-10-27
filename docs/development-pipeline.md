# Development research pipeline

`PLAN.md` is the research contract. This repository is an executable **development baseline**, not a completed 20-week study or a validated flight simulator. No G1–G5 gate is claimed.

## Implemented and exercised

- Ideal 3-DOF point-mass JAX RK4 and independent NumPy/SciPy reference with analytic tests, event handling, batching and precision checks.
- JAX 6-DOF rigid-body development dynamics, observations, CMDP costs, Gymnasium adapters and an independent SciPy development model with engine/slosh/moving-CoM effects.
- CVXPY/Clarabel finite-horizon guidance and attitude feedback. This is not a certified lossless-convexification solver or a tuned B0 result.
- PPO clipped updates, GAE bootstrap, asymmetric critic, per-constraint PID multipliers, SAC twin-Q replay updates and explicit residual composition.
- Residual development training uses `jax_feedback`, not CVXPY B0. Configs and run receipts name that backend explicitly. B0-plus-residual experiments require a vectorized B0 implementation.
- Hydra training/tuning/experiment commands, msgpack checkpoints, hashed scenario fixtures, evaluation metrics, deterministic rliable reports and figures.
- E0–E5 config files and method/penalty expansion; nominal and one-factor stress fixture generation. Unsupported stress effects are rejected rather than treated as nominal.

## Commands

```bash
make check
make train ARGS="training.steps=32 training.environments=2 training.output=outputs/dev"
make train ARGS="agent=ppo_residual_lagrangian training.steps=32"
make train ARGS="agent=sac training.steps=64"
make tune ARGS="tuning.trials=1 tuning.steps_per_trial=32"
uv run python scripts/freeze_test_set.py results/dev-fixture.json --count 2
uv run python scripts/evaluate.py results/dev-fixture.json --backend sim_e --limit 1
uv run python scripts/evaluate.py results/dev-fixture.json --checkpoint outputs/dev/final.msgpack
uv run python scripts/make_figures.py outputs/dev/run.json --output results/dev.png
```

`scripts/run_experiment.py` expands the method matrix from the selected experiment config. Large runs require `confirm_large_run=true`; that flag is a budget acknowledgment, not evidence of compute readiness. `configs/research.yaml` is an unfrozen candidate, not Gate G4.

## Evaluation and artifacts

Every training run records resolved config, seed, Git state, source/lock hashes, simulator versions, dependencies and hardware. Checkpoints are msgpack, not pickle. `results/`, `outputs/` and checkpoints are ignored by Git; retain final research checkpoints separately.

Frozen fixtures require a sibling `.sha256.json`. Evaluation refuses modified data. Fuel ratios are null unless a valid per-scenario convex-optimal baseline is supplied. Constraint rates are violation fractions per episode; episode violation rate is any violation. Success requires a landing termination and all touchdown criteria, not just small terminal errors.

The rliable wrapper isolates NumPy global RNG during reports and restores it afterwards. It includes a narrow legacy decorator compatibility adapter for current pandas/arch loading; dependency versions are locked and import/bootstrap behavior is tested.

## Important limits

- Full-fidelity effects are development models, not externally validated or complete: Dryden wind, detailed sensor bias/variable delay, actuator backlash/quantization, and persistent slosh/actuator states need further work.
- Sim-T wind is a simplified translational acceleration, not relative-flow aerodynamics. No feasibility-certified domain sampler or curriculum scheduler yet.
- PPO does not yet implement observation stacks/GRU, running normalization, full research tuning or 4,096-env throughput validation. SAC has development replay/update paths, not research convergence evidence.
- Stress fixture generation exists, but thrust shortfall, delay, sensor noise and mid-descent drop evaluation need explicit backend implementations. Only steady-wind stress evaluation is supported currently.
- E5 needs complete no-history/GRU, privileged-critic, curriculum and residual-scale experiment expansion. Some config fields document intended research variations but are not all executable yet.
- No LunarLander PPO training, 95% success, 500M-step training, large seed matrix, GPU benchmark, final test-set freeze, statistical hypothesis decision, or paper result has been produced.

These limitations are tracked openly rather than represented as passed research work.
