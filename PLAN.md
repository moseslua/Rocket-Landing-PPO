# RL Rocket Landing — Research Project Plan

Oct 26, 2026 · @Moses

## Overview

This 20-week simulation study tests whether a **constrained residual RL policy**, learned online on top of convex powered-descent guidance, lands a rocket more robustly than classical guidance alone, pure RL, or RL with penalty-based constraints.

**Research question:** Does combining residual learning (RL corrects a classical controller) with constrained RL (safety limits handled as a CMDP, not as reward penalties) improve robustness under model mismatch, measured by sim-to-sim transfer, without giving up fuel efficiency?

The design is a 2×2 factorial (pure vs. residual × penalty vs. constrained) plus a classical baseline. This isolates the contribution of each idea instead of reporting one combined method. Everything happens in simulation; a separate higher-fidelity evaluation simulator stands in for the real world.

The main outputs are an open-source landing environment with two fidelity levels, a benchmark of five methods with rigorous statistics, and a paper or technical report.

## Hypotheses and success criteria

Four hypotheses are fixed before the main experiments run, so results can't be reinterpreted after the fact.

- **H1 (constraints):** Lagrangian-constrained RL has a lower constraint-violation rate than penalty-based RL at equal or higher success rate, across the whole penalty-weight sweep.
- **H2 (sample efficiency):** Residual RL reaches 95% success on the training simulator in at least 3× fewer environment steps than pure RL.
- **H3 (robustness):** Under sim-to-sim transfer, the constrained residual policy loses less success rate than both classical guidance and pure RL.
- **H4 (efficiency, secondary):** Under nominal conditions, RL policies use at most 10% more propellant than the convex-optimal solution for the same initial condition.

The touchdown criteria and targets below are starting values. Freeze them at Gate G4, before the main experiments.

| Quantity | Target | Notes |
| --- | --- | --- |
| Touchdown vertical speed | ≤ 2.0 m/s | Part of the success definition |
| Touchdown horizontal speed | ≤ 1.0 m/s | Part of the success definition |
| Touchdown tilt | ≤ 5° | Part of the success definition |
| Touchdown angular rate | ≤ 5°/s | Part of the success definition |
| Landing position error | ≤ 5 m from pad center | Part of the success definition |
| Success rate, held-out training-sim test set | ≥ 99% | Best method |
| Success rate, evaluation-sim test set | ≥ 95% | Best method |
| Constraint-violation rate | ≤ 1% of episodes | Glide slope, tilt, angular rate |
| Fuel vs. convex optimum, nominal | ≤ 110% | H4 |

## Scope, assumptions and non-goals

The project covers only the terminal powered-descent phase: from engine ignition at roughly 0.5 to 2.5 km altitude down to touchdown on a flat pad.

**In scope**

- A single gimballed engine with throttle, plus roll control from small reaction thrusters.
- 3-DOF point-mass dynamics (Phase 1 only) and 6-DOF rigid-body dynamics with variable mass (all main results).
- A generic, illustrative vehicle with parameters chosen for a well-posed problem, not modeled on any specific real rocket.
- Wind, gusts, actuator dynamics, sensor noise and latency.

**Assumptions**

- The policy receives a noisy estimate of the full state. State estimation and navigation sensors are not modeled beyond noise, bias and delay.
- The pad location is fixed and known.
- The vehicle can hover at minimum throttle (thrust-to-weight above 1 at minimum throttle near touchdown). The harder "hoverslam" regime, where it cannot, is a stretch goal.

**Non-goals**

- Ascent, boostback, atmospheric entry and grid-fin control.
- Landing-leg contact dynamics beyond a simple touchdown check.
- Hardware, hardware-in-the-loop testing and flight software.
- Vision-based navigation.

## Problem formulation

The task is a continuous-control MDP in a pad-centered, flat-Earth inertial frame (East-North-Up). The policy acts at 20 Hz and physics runs at 200 Hz with RK4 integration.

**State (14 values):** position r (3), velocity v (3), attitude quaternion q (4), body angular rate ω (3) and mass m (1).

**Observation:** the state with noise, bias and delay added, then normalized. Attitude is given as the first two columns of the rotation matrix (a continuous 6D representation), never as Euler angles. The observation also includes the previous action and, for residual variants, the classical controller's current action.

**Action (4 values):** the policy outputs values in \[−1, 1\] through a tanh, which are mapped to physical commands.

| Action | Physical range | Notes |
| --- | --- | --- |
| Throttle | \[τ\_min, 1\] of max thrust, τ\_min = 0.4 | Minimum-throttle floor makes the feasible thrust set nonconvex |
| Gimbal pitch | ±6° | Rate-limited |
| Gimbal yaw | ±6° | Rate-limited |
| Roll torque | ±1 (normalized) | Reaction thrusters |

**Dynamics:** 6-DOF rigid body with variable mass. T\_b is engine thrust in the body frame, d the gimbal-to-CoM lever arm, and J the inertia tensor.

```latex
\begin{aligned}
\dot{r} &= v \\
\dot{v} &= g + \tfrac{1}{m} R(q)\, T_b + a_{\text{aero}} + a_{\text{wind}} \\
\dot{q} &= \tfrac{1}{2}\, q \otimes \begin{bmatrix} 0 \\ \omega \end{bmatrix} \\
J \dot{\omega} &= d \times T_b + \tau_{\text{rcs}} - \omega \times J\omega \\
\dot{m} &= -\frac{\lVert T_b \rVert}{I_{sp}\, g_0}
\end{aligned}
```

**Episode termination:**

1. **Landed:** altitude reaches zero with all touchdown criteria met (success).
2. **Crashed:** altitude reaches zero with any criterion violated.
3. **Out of bounds:** the vehicle leaves a bounding box around the pad, or tilt exceeds 60°.
4. **Out of fuel:** propellant mass reaches zero.
5. **Timeout:** 60 s (1,200 policy steps). This prevents the policy from hovering indefinitely.

## Reward design and constraints

The objective is "land, using little fuel." Safety limits are separate constraints with their own budgets, which makes this a constrained MDP (CMDP).

```latex
\max_{\pi}\; \mathbb{E}_{\pi}\Big[\sum_t \gamma^t r_t\Big] \quad \text{s.t.} \quad \mathbb{E}_{\pi}\Big[\sum_t \gamma^t c_{i,t}\Big] \le d_i, \quad i = 1,\dots,k
```

**Reward terms.** The weights below are starting points, to be tuned once in Phase 2 and then frozen for every method.

| Term | Form | Initial weight |
| --- | --- | --- |
| Shaping | γΦ(s′) − Φ(s), with Φ(s) = −(w\_r‖r‖ + w\_v‖v‖ + w\_θ·tilt) | w\_r = 0.01, w\_v = 0.05, w\_θ = 0.5 |
| Fuel | −Δm per step, normalized by initial propellant | 1.0 |
| Smoothness | −‖a\_t − a\_{t−1}‖² | 0.01 |
| Landing bonus | +100, plus up to +50 scaled by touchdown quality | Terminal |
| Crash or out of bounds | −100 | Terminal |
| Timeout | −50 | Terminal; discourages hovering |

Potential-based shaping is used because it provably leaves the optimal policy unchanged, so it can speed up learning without redefining the task.

**Constraint costs.** Each cost is 1 in any step where the limit is violated and 0 otherwise. Its budget d\_i is set as an allowed fraction of steps.

| Constraint | Limit |
| --- | --- |
| Glide-slope cone | Elevation angle seen from the pad center stays ≥ 15° (inside an upward-opening cone) |
| Tilt | ≤ 20° from vertical during descent |
| Angular rate | ≤ 15°/s |

**Penalty variant (for comparison).** The same costs are folded into the reward as −λ\_i·c\_i with fixed λ\_i. Sweeping λ\_i over {0.1, 1, 10, 100} tests H1 fairly against the best-tuned penalty, not an arbitrary one.

**Reward-hacking checks.** Watch for hovering until timeout, landing just outside the success window, and oscillating to exploit the shaping term. Rendering 20 random episodes every evaluation cycle catches most of these early.

## Simulators

The project uses two simulators. Policies train only in the fast **training sim (Sim-T)** and are tested in a separate, slower, higher-fidelity **evaluation sim (Sim-E)**. The performance drop between them is the stand-in for the reality gap.

Sim-T is written in JAX and vectorized to run at least 4,096 environments in parallel on one GPU. Sim-E is an independent NumPy/SciPy implementation with an adaptive-step integrator. Using a separate code path means a bug in Sim-T is unlikely to be repeated in Sim-E and hidden from the transfer test.

| Effect | Sim-T (training) | Sim-E (evaluation) |
| --- | --- | --- |
| Integrator | RK4, fixed 5 ms step | Adaptive (RK45), tight tolerance |
| Engine response | First-order lag | Second-order response plus dead time |
| Specific impulse | Constant | Varies with throttle |
| Center of mass | Fixed | Moves as propellant burns |
| Propellant slosh | None | Pendulum slosh model |
| Wind | Steady wind plus Ornstein-Uhlenbeck gusts | Dryden turbulence model |
| Aerodynamics | Constant drag coefficient | Coefficients vary with angle of attack |
| Sensors | Gaussian noise, fixed delay | Noise, slowly drifting bias, variable delay |
| Actuators | Gimbal rate limit | Rate limit, quantization, small backlash |

**Physics validation (unit tests, run on every commit):**

- Free fall matches the analytic solution.
- Vertical burn at constant thrust matches the rocket equation for velocity change and mass.
- Torque-free rotation conserves angular momentum and rotational energy.
- Quaternion norm stays at 1 within tolerance.
- With all extra effects switched off, Sim-T and Sim-E produce matching trajectories within a small tolerance.
- The same seed gives bit-identical episodes in Sim-T.

Before the full simulators are ready, the pipeline is smoke-tested on Gymnasium's LunarLanderContinuous and on the 3-DOF Sim-T.

## Domain randomization and curriculum

Each training episode samples vehicle parameters, disturbances and initial conditions from the ranges below. A curriculum scales these ranges up from easy to full as the policy improves.

**Randomized parameters (Sim-T).** These ranges are starting values, to be refined by the ablation in E5.

| Parameter | Range around nominal |
| --- | --- |
| Dry mass | ±10% |
| Initial propellant | ±15% |
| Max thrust | ±5% |
| Specific impulse | ±3% |
| Inertia tensor | ±15% per axis |
| Gimbal misalignment | 0 to 0.5° |
| Lateral CoM offset | 0 to 5 cm |
| Engine lag time constant | 50 to 200 ms |
| Steady wind | 0 to 10 m/s, any direction |
| Gust intensity | 0 to 3 m/s RMS |
| Sensor noise | 0.5× to 2× nominal σ |
| Observation delay | 0 to 60 ms |

**Initial conditions (full envelope)**

| Variable | Range |
| --- | --- |
| Altitude | 500 to 2,500 m |
| Horizontal offset from pad | ±400 m |
| Vertical velocity | −100 to −30 m/s |
| Horizontal velocity | ±25 m/s |
| Tilt | 0 to 15° |
| Angular rate | ±5°/s per axis |

**Feasibility filter.** Every sampled initial condition is checked with the convex guidance solver. If no fuel-feasible landing exists, the sample is rejected and redrawn. This keeps the success-rate denominator meaningful, so "failures" are never cases no controller could solve.

**Curriculum.** A difficulty factor κ ∈ \[0.1, 1\] scales both the initial-condition envelope and the randomization ranges. Training starts at κ = 0.1, just above the pad with small disturbances. κ rises by 0.1 whenever the rolling success rate over the last 10,000 episodes exceeds 90%. The final 30% of training stays at κ = 1. All RL variants use the same curriculum, and E5 tests its value by removing it.

## Baselines

The comparison covers five methods, the four corners of the 2×2 design plus classical guidance, and one reference that can't be beaten.

| ID | Method | Controller | Constraint handling | Role |
| --- | --- | --- | --- | --- |
| B0 | Classical guidance | Convex guidance + attitude controller | Hard constraints in the optimizer | Classical baseline |
| B1 | Pure RL, penalty | PPO from scratch | Fixed penalties (swept) | RL baseline |
| B2 | Pure RL, constrained | PPO-Lagrangian from scratch | CMDP | Isolates the constraint method |
| B3 | Residual RL, penalty | B0 + PPO correction | Fixed penalties (swept) | Isolates the residual idea |
| B4 | Residual RL, constrained | B0 + PPO-Lagrangian correction | CMDP | Proposed method |
| Ref | Open-loop convex optimum | Solved once with the true model | Exact | Fuel lower bound, nominal cases only |

**Classical baseline (B0).** The outer loop is lossless-convexification powered-descent guidance (in the style of Açıkmeşe and Ploen), formulated in CVXPY and solved with Clarabel, or compiled with cvxpygen for speed. It is re-solved at 5 Hz in receding-horizon (MPC) fashion and outputs a desired thrust vector. The inner loop is a quaternion-error LQR attitude controller that tracks that thrust direction through the gimbal. B0 gets the same tuning budget as the RL methods: 30 trials of its weights and gains on the Sim-T validation set. A weak classical baseline would make every RL result look better than it is.

**Residual formulation (B3, B4).** The applied command is the classical action plus a bounded correction:

```latex
a_t = a^{\text{B0}}_t + \alpha \cdot \pi_\theta(o_t), \qquad \alpha = 0.2 \times \text{actuator range}
```

The policy's final layer is initialized near zero, so training starts from B0's behavior. The residual scale α is ablated in E5.

**Algorithm check.** B1 and B4 are also trained with SAC, so the conclusions aren't specific to PPO.

## RL methods and training setup

PPO is the main algorithm, because it is stable and scales well to thousands of parallel online environments. The constrained variant adds Lagrange multipliers updated with a PID controller (PID-Lagrangian, Stooke et al.), which damps the oscillation that plain gradient-ascent multipliers often show.

**Starting PPO hyperparameters (6-DOF)**

| Setting | Value |
| --- | --- |
| Parallel environments | 4,096 |
| Rollout length | 32 steps per environment |
| Minibatches × epochs | 8 × 5 |
| Learning rate | 3e-4, linearly annealed |
| Discount γ | 0.995 (episodes run up to 1,200 steps) |
| GAE λ | 0.95 |
| Clip range | 0.2 |
| Networks | Separate actor and critic, 2 × 256 units, ELU |
| Initial policy log-std | −0.5, state-independent |
| Normalization | Running observation and value normalization |
| Training budget | 500M environment steps per run |

**Constrained variant.** Each constraint has its own cost critic and a multiplier λ\_i ≥ 0, capped at 100. The multipliers are updated once per rollout from the gap between the measured cost and its budget d\_i.

**Asymmetric actor-critic.** The critic sees privileged simulator information the actor never gets: true vehicle parameters, true wind and noise-free state. This helps the value estimate under heavy randomization, and the deployed actor stays realistic. E5 ablates it.

**Memory.** The actor sees a stack of the last 4 observations, so it can infer delays and disturbances. A GRU actor is tried as an alternative in E5.

**Tuning protocol.** Each method gets the same budget: 30 Optuna trials of 100M steps, scored on a Sim-T validation set separate from all test sets. The best configuration per method is then frozen at Gate G4. Sim-E is never used for tuning.

## Experimental design

Six experiments map directly to the hypotheses. E1 to E4 answer the research question, E5 explains why the proposed method works, and E0 de-risks the pipeline.

| ID | Question | Methods | Seeds | Tests |
| --- | --- | --- | --- | --- |
| E0 | Does the pipeline work end to end? | PPO on LunarLander and 3-DOF Sim-T | 3 | Gate G1 |
| E1 | Which method lands best, in Sim-T and in Sim-E? | B0 to B4 | 10 | H3, H4 |
| E2 | How fast does each method learn? | B1 to B4 learning curves | 10 (from E1) | H2 |
| E3 | Lagrangian vs. best-tuned penalty | B1, B3 at 4 penalty weights; B2, B4 | 5 per weight | H1 |
| E4 | Where does each method break? | B0 to B4 on stress sweeps | 10 (E1 policies) | H3 |
| E5 | Which parts of B4 matter? | B4 with one component removed | 5 per variant | Mechanism |

**E4 stress sweeps.** One factor is pushed past its training range at a time, with everything else nominal. Each sweep has 6 levels and 500 episodes per level.

- Steady wind: 0 to 20 m/s (training maximum is 10).
- Thrust shortfall: 0 to 15% (training range is ±5%).
- Observation delay: 0 to 150 ms (training maximum is 60).
- Sensor noise: 1× to 5× nominal.
- Mid-descent thrust drop: a sudden 10% loss at a random time.

**E5 ablations of B4:** no shaping term, no curriculum, no domain randomization, no privileged critic, no observation history (and a GRU instead), and residual scale α ∈ {0.05, 0.1, 0.2, 0.5}.

E1 policies are reused in E2 and E4, so the main training cost is E1, E3 and E5.

## Evaluation metrics and statistics

The primary metrics are success rate and constraint-violation rate on the Sim-E test set. They are named here, before any results exist, so they can't be swapped for a more flattering metric later.

**Frozen test sets.** Generated once at Gate G4, saved to disk with a hash, and never used for tuning:

- Sim-T held-out set: 2,000 episodes (initial condition + parameter sample + disturbance seed).
- Sim-E set: 2,000 episodes drawn from the same distributions.
- Stress grids for E4: 6 levels × 500 episodes per factor.

**Metrics**

| Metric | Definition | Primary? |
| --- | --- | --- |
| Success rate | Share of episodes meeting all touchdown criteria | Yes |
| Constraint-violation rate | Share of episodes with any constraint violated, also reported per constraint | Yes |
| Fuel ratio | Propellant used ÷ convex-optimal propellant for the same initial condition | No |
| Touchdown errors | Distributions of vertical speed, horizontal speed, tilt and position error | No |
| Tail performance | CVaR at 5% of a continuous landing score | No |
| Smoothness | Total variation of the commands per episode | No |
| Inference cost | Wall-clock time per control step on one CPU core | No |

**Statistics.** Each configuration is trained with 10 seeds (5 for E3 and E5). Results are aggregated with the `rliable` library: interquartile mean (IQM) with 95% stratified-bootstrap confidence intervals, performance profiles, and probability of improvement between method pairs. A hypothesis counts as supported only if its confidence interval excludes zero difference. Every comparison run is reported, including those that didn't favor the proposed method.

**Failure analysis.** Failed episodes from E1 and E4 are clustered by initial condition and failure mode, and representative episodes are rendered as videos. This shows whether methods fail on the same hard cases or on different ones.

## Infrastructure, repository and reproducibility

The rule is that any number or figure in the paper can be regenerated from a config, a seed and a git commit, with one command.

**Stack.** Python 3.11; JAX with Flax and Optax for Sim-T and the agents; NumPy and SciPy for Sim-E; CVXPY with Clarabel (and cvxpygen) for guidance; Hydra for configs; Optuna for tuning; Weights & Biases for logging; pytest and pre-commit for tests and style; a Docker image pinning every dependency.

**Repository layout**

```
rocket-landing-rl/
├── envs/
│   ├── sim_t/          # JAX training sim: dynamics, randomization, rewards, costs
│   ├── sim_e/          # Independent high-fidelity evaluation sim
│   └── common/         # Shared interfaces, frames, vehicle parameters
├── controllers/        # B0: convex guidance + attitude LQR
├── agents/             # PPO, PPO-Lagrangian, SAC, residual wrapper
├── configs/            # Hydra: env/, agent/, experiment/ (one file per E0–E5)
├── scripts/
│   ├── train.py
│   ├── tune.py
│   ├── evaluate.py     # Runs any policy on a frozen test set
│   └── make_figures.py
├── test_sets/          # Frozen test sets + hashes
├── tests/              # Physics, determinism, Sim-T vs Sim-E agreement
├── results/            # Evaluation outputs (not checkpoints)
├── Dockerfile
└── Makefile            # make test, make e1, make fig-e1, ...
```

**Reproducibility rules**

1. Every run logs its full resolved config, seed, git commit and simulator version.
2. Simulator changes bump a version tag. Results from different simulator versions are never mixed in one figure.
3. Test sets are generated once and checked against their stored hashes before each evaluation.
4. Checkpoints are kept for every final run, not only the best seed.
5. Each figure in the paper has a matching `make` target.

## Compute budget

The plan needs roughly 250 GPU-hours and 500 CPU-hours, including a 2× margin for debugging and reruns. This assumes about 250,000 environment steps per second end to end (simulation plus PPO updates) on one A100-class GPU. Measure the real throughput in week 5 and rescale this table then.

| Item | Runs | Environment steps | GPU-hours |
| --- | --- | --- | --- |
| Tuning (B1 to B4) | 4 methods × 30 trials | 12B | \~15 |
| E1 main runs | 4 RL methods × 10 seeds | 20B | \~22 |
| E3 penalty sweep | 2 methods × 4 weights × 5 seeds | 20B | \~22 |
| E5 ablations | \~10 variants × 5 seeds | 25B | \~28 |
| SAC check | 2 methods × 5 seeds | Lower, but SAC runs slower | \~20 |
| **Total with 2× margin** |  |  | **\~250** |

CPU time goes mostly to Sim-E evaluation and to B0, which solves a convex program 5 times per simulated second. Those episodes are independent, so they spread across all available cores.

If compute is tight, cut E5 to 3 seeds and SAC to 3 seeds first. Keep 10 seeds for E1, because the main claims rest on it.

## Timeline and milestones

The project runs 20 weeks in seven overlapping phases. Week 11 (Gate G4) is the critical point: after it, nothing that affects the results changes.

&#91;embedded content: project timeline · 7 phases, 5 gates, 20 weeks\]

Phases overlap where the work is independent; Sim-E is built while the 6-DOF training sim is being tuned. A phase can't close until its gate passes.

**Gate criteria**

1. **G1 (end of week 3):** PPO reaches ≥ 95% success on nominal 3-DOF Sim-T, and all physics tests pass.
2. **G2 (end of week 7):** B0 reaches ≥ 95% success on nominal 6-DOF Sim-T. Throughput is measured and the compute budget rescaled.
3. **G3 (end of week 9):** Sim-T and Sim-E agree on nominal trajectories, and Sim-E's tests pass.
4. **G4 (end of week 11):** Hyperparameters, reward weights, success criteria, simulator versions and test sets are frozen, and the hypotheses are written down.
5. **G5 (end of week 20):** Every figure regenerates from a `make` target, and the code, test sets and checkpoints are released.

## Risks and mitigations

The two risks most likely to sink the project are a simulator bug discovered late and a classical baseline that is too weak. Both are addressed by gates before the main experiments.

| Risk | Likelihood | Impact | Mitigation |
| --- | --- | --- | --- |
| Physics bug found after main runs | Medium | High | Unit tests on every commit; Sim-T vs. Sim-E agreement at Gate G3; simulator version frozen at G4 |
| Classical baseline under-tuned | Medium | High | Same tuning budget as RL; established guidance formulation; report tuning effort |
| PPO fails to learn 6-DOF landing | Medium | High | Curriculum; residual variants start from B0; fall back to 3-DOF as the main result |
| Lagrange multipliers oscillate | Medium | Medium | PID-Lagrangian; multiplier cap; tune cost budgets on validation set |
| Sim-E shares Sim-T's mistakes | Low | High | Separate code path, different integrator, written from the equations rather than ported |
| Infeasible initial conditions inflate failures | Medium | Medium | Convex feasibility filter on every sample |
| Compute overrun | Medium | Medium | Measure throughput in week 5; cut E5 and SAC seeds first |
| Hypotheses not supported | Medium | Medium | Report as a careful negative result; the environment and benchmark still stand as contributions |
| Scope creep (grid fins, entry, hoverslam) | High | Medium | Explicit non-goals; extensions only after the write-up draft exists |

## Deliverables and open decisions

**Deliverables**

1. An open-source landing environment: Sim-T (JAX, 3-DOF and 6-DOF) and Sim-E, with physics tests and documentation.
2. Reference implementations of B0 to B4, with tuned configs and trained checkpoints for every seed.
3. The frozen test sets and an evaluation script that scores any policy on them.
4. A paper or technical report with every figure reproducible by a `make` target.

**Open decisions** (resolve in week 1)

- [ ] Confirm the research question, or swap in another (for example, RL vs. convex guidance on fuel use and compute).
- [ ] Vehicle regime: hover-capable at minimum throttle (planned) or the harder hoverslam regime.
- [ ] Framework: JAX (planned, for simulator speed) or PyTorch (if the team knows it better).
- [ ] Compute: confirm access to roughly 250 GPU-hours.
- [ ] Target venue and deadline, which may compress or extend the timeline.

**Background reading** (cited from memory; check details before relying on them)

- Açıkmeşe and Ploen, convex programming for Mars powered-descent guidance (lossless convexification), 2007.
- Gaudet, Linares and Furfaro, deep reinforcement learning for six-degree-of-freedom planetary landing, 2020.
- Stooke, Achiam and Abbeel, PID Lagrangian methods for constrained RL, 2020.
- Agarwal et al., deep RL at the edge of the statistical precipice (the `rliable` paper), 2021.
- Silver et al. and Johannink et al., residual policy learning, 2018.
