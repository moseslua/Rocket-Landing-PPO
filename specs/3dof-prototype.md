# 3-DOF numerical prototype

## Scope
Implement ideal point-mass translational dynamics, not the complete landing environment. State is ENU position (m), ENU velocity (m/s), total mass (kg), elapsed time (s), terminal code. Force is an inertial ENU thrust vector (N), held constant for each policy interval; its norm is clipped to max thrust. Zero thrust is allowed for analytic tests. This is NOT the final four-action throttle/gimbal/RCS contract or a fuel-optimal controller.

Defaults: g = 9.81 m/s², g0 = 9.80665 m/s², Isp = 300 s, dry mass = 1000 kg, maximum thrust = 30000 N. These illustrative developer parameters are not frozen research choices. Policy interval 0.05 s; RK4 physics step 0.005 s. No attitude, aero, wind, actuator lag, observation noise, rewards, costs, curriculum, auto-reset, or PPO yet.

## Contract
- `Vehicle`, `Clock`, `Scenario`, `Termination` are shared immutable data; no shared RHS/integrator/event code.
- `envs.sim_t.point_mass`: pure JAX `reset`, seeded `sample_reset`, `step`, `rollout`, and `batch_rollout`. Explicit float32 by default. Float64 caller requires JAX x64 enabled; never silently downcast. `State` is a NamedTuple pytree. Static dataclass configuration is compatible with caller-owned JIT. Step uses scan over physics substeps; rollout uses scan over control intervals; batch uses vmap. No internal reset at done.
- `envs.sim_e.point_mass`: independent NumPy float64 `reset`, `sample_reset`, `step`, and `rollout`; SciPy solve_ivp with RK45 or DOP853. Same units and developer scenario inputs. This is an independent ideal reference, not completed high-fidelity Sim-E.
- Equation: rdot=v, vdot=F/m+(0,0,-g), mdot=-norm(F)/(Isp*g0).
- First event wins: touchdown, dry-mass boundary, timeout. Touchdown wins ties, then fuel. Initial terminal boundaries are classified at reset. No underground touchdown scoring. JAX brackets/refines the zero crossing inside its current physics interval using RK4; SciPy uses a root event. Fuel/time horizons are limited before integration. Terminal states are absorbing (no extra burn or elapsed time).
- Only position/horizontal and vertical velocity touchdown thresholds can be assessed; no full success metric without attitude. No landing-success code implied by TOUCHDOWN.
- JAX history utilities are a separate physics-sampled FIFO, initialized with the reset observation. 13 samples cover 0–60 ms at 5 ms. Sampling rounds to nearest physics tick, ties upwards. No negative wraparound: invalid requested delays yield NaNs under JIT; eager callers should validate config. Integrated observations are future work.
- Seeded reset uses independent backend-native samplers on documented developer ranges. Same seed/backend/version reproducibility is tested; cross-backend bit identity is not promised. No ongoing disturbance RNG yet.

## Acceptance checks
1. Analytic free fall and variable-mass rocket-equation burn pass in production float32 and opt-in float64/reference precision.
2. RK4 converges on a coarser developer case; independent SciPy methods converge against analytic solutions.
3. Touchdown, fuel cutoff, timeout, ties, and absorbing states behave consistently; terminal time is within numerical tolerance.
4. Seeded reset is repeatable, multiple seeds vary, and batched scan matches serial stepping under JIT.
5. Delay history handles 0/5/60 ms, rounding, startup, invalid values, and consecutive writes.
6. Benchmark warms JIT, synchronizes results, separates compilation time, compares reference accuracy/function evaluations, and writes config/seed/git dirty state/sim versions/dependency versions/hardware/lock hash with results. No learning-throughput claim.
7. Existing checks still pass and wheel contains public `envs` modules.
