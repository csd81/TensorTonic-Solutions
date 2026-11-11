import numpy as np


def exp_power_backward(
    x: float, exponent: float, upstream_gradient: float
) -> tuple:
    """
    Returns a tuple of exp output, exp gradient, power output, and power gradient.
    """
    x_val = np.float64(x)
    n_val = np.float64(exponent)
    u_val = np.float64(upstream_gradient)

    # 1. Exponential forward and backward
    y_exp = np.exp(x_val)
    g_exp = u_val * y_exp

    # 2. Constant power forward and backward
    if n_val == 0.0:
        y_power = np.float64(1.0)
        g_power = np.float64(0.0)
    else:
        y_power = np.power(x_val, n_val)
        g_power = u_val * n_val * np.power(x_val, n_val - 1.0)

    return float(y_exp), float(g_exp), float(y_power), float(g_power)
