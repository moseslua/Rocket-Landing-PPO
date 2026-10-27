"""rliable metrics with reproducible stratified bootstrap reporting.

rliable 1.2 / arch 7.2 currently samples through NumPy's global RNG. This
wrapper temporarily isolates and restores that state for deterministic calls.
It also adapts the legacy pandas decorator arch imports during module loading.
"""

import functools
import inspect
from contextlib import contextmanager

import numpy as np
import pandas.util._decorators as pandas_decorators


def enable_rliable_import():
    decorator = pandas_decorators.deprecate_kwarg
    try:
        parameters = list(inspect.signature(decorator).parameters.values())
        required = [p for p in parameters if p.default is inspect.Parameter.empty]
    except (TypeError, ValueError):
        return
    if len(required) < 3 or getattr(decorator, "_rocket_legacy_compatible", False):
        return

    @functools.wraps(decorator)
    def compatible(old_arg_name, new_arg_name=None, mapping=None, **kwargs):
        return decorator(old_arg_name, new_arg_name, mapping, **kwargs)

    compatible._rocket_legacy_compatible = True
    pandas_decorators.deprecate_kwarg = compatible


enable_rliable_import()

from rliable import library as rly  # noqa: E402
from rliable import metrics  # noqa: E402


@contextmanager
def isolated_seed(seed):
    previous = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(previous)


def aggregate(metric_by_method, seeds=10, reps=5000, random_state=0):
    """IQM and percentile stratified-bootstrap 95% CIs, with restored RNG state."""
    if reps < 1 or seeds < 1:
        raise ValueError("seeds and bootstrap repetitions must be positive")
    arrays = {name: np.asarray(values, dtype=float) for name, values in metric_by_method.items()}
    if not arrays or any(values.ndim != 1 or values.size != seeds for values in arrays.values()):
        raise ValueError("each method must contain exactly `seeds` one-dimensional scores")
    if any(not np.isfinite(values).all() for values in arrays.values()):
        raise ValueError("metric values must be finite")
    with isolated_seed(random_state):
        point, interval = rly.get_interval_estimates(
            {name: values[:, None] for name, values in arrays.items()},
            lambda x: np.array([metrics.aggregate_iqm(x)]),
            reps=reps,
        )
    return {
        name: {"iqm": float(point[name][0]), "ci95": interval[name][:, 0].tolist()}
        for name in arrays
    }


def probability_of_improvement(first, second, reps=1000, random_state=0):
    """P(second > first), paired per task, with independent-seed bootstrap CI."""
    first, second = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    if first.ndim != 1 or first.shape != second.shape:
        raise ValueError("paired scores must have the same one-dimensional shape")
    with isolated_seed(random_state):
        point, interval = rly.get_interval_estimates(
            {"comparison": [second[:, None], first[:, None]]},
            metrics.probability_of_improvement,
            reps=reps,
        )
    return {
        "probability": float(point["comparison"]),
        "ci95": np.asarray(interval["comparison"]).tolist(),
    }
