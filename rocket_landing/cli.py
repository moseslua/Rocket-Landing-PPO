"""Command-line diagnostics for the development environment."""

import argparse
import json
import platform
from collections.abc import Sequence

import jax

from rocket_landing import __version__


def environment_info() -> dict[str, object]:
    """Return non-secret runtime information for setup diagnostics."""
    return {
        "package_version": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "jax": jax.__version__,
        "jax_backend": jax.default_backend(),
        "jax_devices": [device.platform for device in jax.devices()],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rocket-landing")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("doctor", help="Show Python and JAX runtime information")
    subparsers.add_parser("smoke", help="Check installed numerical and Gymnasium dependencies")
    subparsers.add_parser(
        "benchmark", help="Benchmark the ideal 3-DOF numerical prototype", add_help=False
    )
    args, remaining = parser.parse_known_args(argv)
    if args.command == "benchmark":
        from rocket_landing.benchmark import main as benchmark_main

        return benchmark_main(remaining)
    if remaining:
        parser.error(f"unrecognized arguments: {' '.join(remaining)}")

    if args.command == "doctor":
        print(json.dumps(environment_info(), indent=2))
    else:
        from rocket_landing.smoke import main as smoke_main

        return smoke_main([])
    return 0
