"""Generate figures from saved training/evaluation JSON artifacts."""

import argparse
import json
from pathlib import Path

from rocket_landing.figures import learning_curves, method_comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kind", choices=("learning", "comparison"), default="learning")
    args = parser.parse_args()
    runs = [json.loads(path.read_text()) for path in args.inputs]
    if args.kind == "learning":
        output = learning_curves(runs, args.output)
    else:
        output = method_comparison(
            {path.stem: [run] for path, run in zip(args.inputs, runs, strict=True)}, args.output
        )
    print(output)


if __name__ == "__main__":
    main()
