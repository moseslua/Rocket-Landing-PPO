"""Independent analytic and numerical 3-DOF references for environment tests."""

import numpy as np
from scipy.integrate import solve_ivp


def integrate_free(position, velocity, mass, duration, force, gravity=9.81, isp=300.0, g0=9.80665):
    force = np.asarray(force, float)
    flow = np.linalg.norm(force) / (isp * g0)

    def f(_t, y):
        return np.r_[y[3:6], force / y[6] + [0, 0, -gravity], -flow]

    y0 = np.r_[position, velocity, mass]
    result = solve_ivp(f, (0, duration), y0, method="DOP853", rtol=1e-11, atol=1e-12)
    return result.y[:, -1]
