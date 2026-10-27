# 3-DOF prototype impact

## Target
New `envs/common`, `envs/sim_t`, `envs/sim_e` modules and tests. Add a benchmark command and include `envs` in the installable wheel and Docker context.

## Dependents
No existing Python code imports `envs`. Existing smoke tests remain third-party checks. Packaging, Makefile, README, AGENTS, and environment state need updates to expose the new numerical cores without claiming a complete environment.

## Affected work
PLAN.md Phase 1 physics scaffolding only; no G1 success claim and no experimental tuning.

## Test coverage
Existing six setup tests do not test rocket physics. Add analytic free fall and variable-mass burn, dtype, event precedence/absorption, independent simulator agreement, RK4 convergence, seeded reset, batching/scan, and delay-buffer tests. Verify built wheel imports outside the repository.

## Risk: Medium
New numerical APIs have no existing callers, but units, terminal events, precision, and provenance errors could compromise later research.

## Recommended action
Specify contracts, write failing behavioral tests, implement independent cores, then benchmark only developer scenarios (never frozen test sets).
