import numpy as np

def finite_difference_derivative(coefficients: list, x: float, h: float) -> tuple[float, float, float]:
    """
    Returns the value at x, the value at x plus h, and the estimated slope.
    """
    coeffs = np.asarray(coefficients, dtype=np.float64)
    x_val = np.float64(x)
    h_val = np.float64(h)

    def eval_poly(val: np.float64) -> np.float64:
        # Horner's method iterating from highest degree to constant term
        result = np.float64(0.0)
        for c in reversed(coeffs):
            result = result * val + c
        return result

    fx = eval_poly(x_val)
    fx_h = eval_poly(x_val + h_val)
    slope = (fx_h - fx) / h_val

    return (float(fx), float(fx_h), float(slope))
