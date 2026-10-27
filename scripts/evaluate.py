"""Evaluate a named policy on an immutable fixture; never tune here."""

import argparse
from pathlib import Path

import jax
from flax.serialization import msgpack_restore

from agents.ppo import PPOConfig, init_train_state
from rocket_landing.checkpoints import load
from rocket_landing.evaluation import evaluate_fixture
from rocket_landing.reproducibility import atomic_json


def restore(path):
    payload = msgpack_restore(path.read_bytes())
    if payload["format"] != "rocket-landing-msgpack-v1":
        raise ValueError("unsupported checkpoint format")
    metadata = payload["metadata"]
    raw = metadata["config"]
    config = PPOConfig(
        **{
            key: value
            for key, value in raw["agent"].items()
            if key in PPOConfig.__dataclass_fields__
        }
    )
    model, fresh, _ = init_train_state(jax.random.PRNGKey(0), config)
    state, metadata = load(path, fresh)
    return model, state, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--policy", choices=("zero", "b0", "learned"), default="zero")
    parser.add_argument("--backend", choices=("sim_t", "sim_e"), default="sim_t")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    model = train_state = metadata = None
    if args.policy == "learned" or args.checkpoint:
        if not args.checkpoint:
            parser.error("--checkpoint is required for learned evaluation")
        model, train_state, metadata = restore(args.checkpoint)
        policy = "learned"
    else:
        policy = args.policy
    result = evaluate_fixture(
        args.fixture,
        model,
        train_state,
        args.backend,
        args.limit,
        policy=policy,
        checkpoint_metadata=metadata,
    )
    if args.output:
        atomic_json(args.output, result)
    print(result["summary"])


if __name__ == "__main__":
    main()
