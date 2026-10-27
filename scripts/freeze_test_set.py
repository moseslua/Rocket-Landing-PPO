"""Explicit one-time nominal/stress fixture initialization, never tuning."""

import argparse
from pathlib import Path

from envs.common.scenarios import STRESS_FACTORS, initialize, initialize_stress_grid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--count", type=int, default=2000)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--difficulty", type=float, default=1.0)
    parser.add_argument("--stress-factor", choices=STRESS_FACTORS)
    parser.add_argument("--stress-level", type=float)
    parser.add_argument("--grid", action="store_true")
    args = parser.parse_args()
    if args.grid:
        print(initialize_stress_grid(args.output, count=args.count, start_seed=args.start_seed))
    else:
        stress = None
        if args.stress_factor:
            if args.stress_level is None:
                parser.error("--stress-level is required with --stress-factor")
            stress = {"factor": args.stress_factor, "level": args.stress_level}
        print(initialize(args.output, args.count, args.start_seed, args.difficulty, stress=stress))


if __name__ == "__main__":
    main()
