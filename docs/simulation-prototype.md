# Simulation prototype

This is an ideal 3-DOF numerical foundation, not a complete landing environment. `PLAN.md` remains the research contract. No research gate has passed; there is no PPO training yet.

## Implemented

- `envs/common`: immutable `Scenario`, `Vehicle`, `Clock`, and `Termination` declarations.
- `envs/sim_t/point_mass.py`: JAX variable-mass dynamics, RK4, physics substeps, functional reset/step, scan rollout, and vmap batching.
- `envs/sim_e/point_mass.py`: independent NumPy/SciPy equations and event handling; RK45 and DOP853 reference solvers. This is **not** the planned high-fidelity Sim-E.
- `envs/sim_t/history.py`: standalone physics-rate history primitive; not yet part of an observation pipeline.

The numerical implementations share data declarations only. Neither imports the other's equations, integrator, or event code.

## Units and API

Use ENU position in meters, velocity in m/s, total mass in kg, elapsed time in seconds, and inertial thrust in newtons. State contains position (3), velocity (3), mass, elapsed time, and a termination code. The JAX state also retains two compensated-summation residuals for mass and time; preserve those fields when passing state between calls.

```python
from functools import partial

import jax
import jax.numpy as jnp

from envs.common import Clock, Scenario, Vehicle
from envs.sim_t import point_mass

vehicle, clock = Vehicle(), Clock()
state = point_mass.reset(Scenario(position=(0, 0, 1000)))
advance = jax.jit(partial(point_mass.step, vehicle=vehicle, clock=clock))
next_state = advance(state, jnp.array([0, 0, 20000], dtype=jnp.float32))

forces = jnp.tile(jnp.array([0, 0, 20000], dtype=jnp.float32), (100, 1))
final, history = jax.jit(point_mass.rollout)(state, forces)
# history contains post-action states: position shape (100, 3).
```

`batch_rollout` accepts a batched State `[environment, ...]` and forces `[environment, policy_step, 3]`. It returns batched final states and history. `sample_reset(key)` supplies repeatable developer cases, not the research initial-condition envelope or a feasibility-filtered curriculum. Parameters/configs must be closed over or supplied as static JIT arguments. The low-level JAX kernel expects finite valid state and forces; it does not add runtime host checks inside JIT.

The SciPy API accepts the same declarative scenario and physical forces:

```python
import numpy as np
from envs.sim_e import point_mass as reference

state = reference.reset(Scenario(position=(0, 0, 1000)))
next_state = reference.step(state, np.array([0, 0, 20000]), method="RK45")
# method="DOP853" is also supported; tolerances are explicit keyword arguments.
```

Thrust norm is clipped to maximum thrust. Zero thrust is allowed for analytic tests. **This is not the final policy action mapping**: minimum throttle, gimbal/attitude, and RCS are absent. Illustrative vehicle defaults are not frozen research parameters.

## Clock, precision, and terminal events

The default physics step is 5 ms (200 Hz). Ten substeps make one 50 ms control interval (20 Hz); thrust is held constant within the interval. Tests may use coarser clocks for convergence checks.

JAX uses explicit float32 states by default. Tests exercise that production path and opt-in float64 via `jax.enable_x64()`. Requesting float64 while x64 is disabled raises an error instead of silently downcasting. SciPy uses float64. Mass/time use compensated summation in JAX to avoid accumulated fuel and clock drift. Position/velocity still have float32 roundoff.

The first event stops integration: ground contact, dry mass, or timeout. Contact wins ties, then fuel. Terminal states are absorbing; repeated actions do not advance time or burn fuel. A TOUCHDOWN code is **not success**: this prototype has no attitude or full touchdown scoring.

JAX refines ground contact within a physics interval with RK4 bisection; SciPy uses root events and dense-output bracketing. Both also check a descending-to-ascending turn, so a dip below ground is not missed merely because the interval endpoints are above ground. Sim-T and the reference implement these checks independently. These ideal-model checks do not establish contact handling for future stiff/discontinuous effects.

The delay primitive pads startup with the reset observation and retains 13 samples. With observations written at every 5 ms physics tick, it covers 0–60 ms. Requested delays round to the nearest tick (ties upwards); invalid delays produce NaNs under JIT rather than wrapping into the wrong history index. It is not yet wired to `step` and does not add sensor noise or bias.

## Reproduce checks and measurements

```bash
make physics
make check
make benchmark ARGS="--environments 64 --steps 100 --repeats 3 --seed 0"
```

Benchmark receipts go to ignored `results/prototype-<unique-id>.json`. An explicit `--output` is supported but existing files are never overwritten. Each receipt includes resolved numerical configuration, reset distributions, seed, Git commit and dirty status, relevant source hashes, lockfile hash, simulator versions, dependency versions, backend, platform, and timings. Source archives or installed wheels may lack Git/lock provenance; those fields remain null, not invented.

JAX timing synchronizes results, reports warmup/compilation separately, and measures active integration above ground rather than frozen terminal steps. The reference comparison uses a separate explicit constant-burn scenario with an analytic solution and reports function evaluations and error at two tolerances. Neither command accesses frozen test sets or tunes policies.

A first macOS arm64 CPU run (64 environments × 100 control intervals, 3 repeats) measured approximately 210k ideal-core policy transitions/s, with about 0.23 s warmup/compilation. Float32 error after 5 s was about 1 mm position, 5e-6 m/s velocity, and 5e-5 kg mass. Both SciPy methods were near roundoff on this easy case; their timings do **not** justify a permanent integrator choice. Raw receipt: `results/prototype-bb4ef313189d405294a1fe6fe46c46ee.json` (local ignored artifact). This is not 6-DOF or PPO throughput, and cannot validate the study's compute budget.

## Not implemented

- 6-DOF attitude, inertia, torque-free rotation, and quaternion tests.
- Engine/actuator dynamics, aero, wind, slosh, noisy/delayed integrated observations.
- Final action/observation/reward/cost contract, Gymnasium adapter, rendering, bounds checks, and automatic episode resets.
- Guidance, PPO or other agents, feasibility filter, curriculum, frozen sets, and research experiments.

Next: define the trainable 3-DOF environment contract, then implement observations/actions/rewards/costs and a Gymnasium adapter before attempting E0/G1 learning. Higher-fidelity Sim-E and frozen test sets must not be used for tuning.
