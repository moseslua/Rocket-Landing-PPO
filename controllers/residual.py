"""Residual policy action composition; independent of learning algorithm."""

import numpy as np


def compose_residual(baseline_action, correction, scale=0.2):
    baseline_action = np.asarray(baseline_action, dtype=float)
    correction = np.asarray(correction, dtype=float)
    if baseline_action.shape != correction.shape:
        raise ValueError("baseline and residual action shapes must match")
    if not 0 <= scale <= 1:
        raise ValueError("residual scale must be in [0, 1]")
    return np.clip(baseline_action + scale * np.clip(correction, -1, 1), -1, 1)
