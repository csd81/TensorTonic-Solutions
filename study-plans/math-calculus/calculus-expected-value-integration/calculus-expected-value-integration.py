import numpy as np

def expected_value_x_squared(mu: float, sigma: float, n: int) -> dict:
    """
    Returns trapezoidal, Simpson, and analytical second moments.
    """
    # 1. Establish the 5-sigma integration boundaries
    a = mu - 5.0 * sigma
    b = mu + 5.0 * sigma
    
    # 2. Build exactly n + 1 evenly spaced float64 grid points
    x = np.linspace(a, b, n + 1, dtype=np.float64)
    h = (b - a) / n
    
    # 3. Define the integrand: x^2 * Gaussian_PDF(x)
    norm_const = 1.0 / (sigma * np.sqrt(2.0 * np.pi))
    pdf = norm_const * np.exp(-((x - mu) ** 2) / (2.0 * (sigma ** 2)))
    f_vals = (x ** 2) * pdf
    
    # 4. Implement the Compound Trapezoidal Rule
    # Formula: (h / 2) * [f(x_0) + 2*f(x_1) + ... + 2*f(x_{n-1}) + f(x_n)]
    trap_weights = np.ones(n + 1, dtype=np.float64)
    trap_weights[1:-1] = 2.0
    trapezoidal_est = (h / 2.0) * np.sum(trap_weights * f_vals)
    
    # 5. Implement the Compound Simpson's Rule
    # Pattern: (h / 3) * [f(x_0) + 4*f(x_1) + 2*f(x_2) + 4*f(x_3) + ... + 4*f(x_{n-1}) + f(x_n)]
    simp_weights = np.ones(n + 1, dtype=np.float64)
    simp_weights[1:-1:2] = 4.0  # Odd indices get coefficient 4
    simp_weights[2:-1:2] = 2.0  # Even internal indices get coefficient 2
    simpsons_est = (h / 3.0) * np.sum(simp_weights * f_vals)
    
    # 6. Compute analytical baseline: E[X^2] = Var(X) + (E[X])^2 = sigma^2 + mu^2
    analytical_val = (mu ** 2) + (sigma ** 2)
    
    return {
        "trapezoidal": float(trapezoidal_est),
        "simpsons": float(simpsons_est),
        "analytical": float(analytical_val)
    }
