"""Safe msgpack checkpoints: parameters, normalization, and algorithm state."""

from pathlib import Path

import flax
import jax
from flax.serialization import msgpack_restore, msgpack_serialize


def save(path, train_state, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "rocket-landing-msgpack-v1",
        "metadata": metadata,
        "parameters": msgpack_serialize(train_state.params),
        "log_std": msgpack_serialize(train_state.log_std),
        "multipliers": msgpack_serialize(train_state.multipliers),
        "update": int(train_state.update),
    }
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(msgpack_serialize(payload))
    temporary.replace(path)


def load(path, train_state):
    payload = msgpack_restore(Path(path).read_bytes())
    if payload["format"] != "rocket-landing-msgpack-v1":
        raise ValueError("unsupported checkpoint format")
    restored = flax.serialization.from_state_dict(
        train_state.params, msgpack_restore(payload["parameters"])
    )
    return train_state._replace(
        params=restored,
        log_std=jax.numpy.asarray(msgpack_restore(payload["log_std"])),
        multipliers=jax.numpy.asarray(msgpack_restore(payload["multipliers"])),
        update=jax.numpy.asarray(payload["update"]),
    ), payload["metadata"]
