import numpy as np

def kl_divergence_analysis(P: list, Q: list, mu1: float, sigma1: float, mu2: float, sigma2: float, n: int) -> dict:
    """
    Returns discrete, numerical Gaussian, and analytical Gaussian KL values.
    """
    # --- 1. Discrete KL Divergence ---
    first = np.asarray(P, dtype=np.float64)
    second = np.asarray(Q, dtype=np.float64)
    terms = np.where(first > 0.0, first * np.log(first / second), 0.0)
    discrete = float(np.sum(terms))
    
    # --- 2. Numerical Gaussian KL Divergence (Simpson's Rule) ---
    width = max(sigma1, sigma2)
    left = min(mu1, mu2) - 5.0 * width
    right = max(mu1, mu2) + 5.0 * width
    
    x = np.linspace(left, right, n + 1, dtype=np.float64)
    h = (right - left) / n
    
    # Calculate density p(x)
    density = np.exp(-0.5 * ((x - mu1) / sigma1) ** 2) / (sigma1 * np.sqrt(2.0 * np.pi))
    
    # Numerically stable log-ratio log(p(x)/q(x)) via algebraic expansion
    log_ratio = np.log(sigma2 / sigma1) - 0.5 * ((x - mu1) / sigma1) ** 2 + 0.5 * ((x - mu2) / sigma2) ** 2
    
    # Compound Simpson's Rule weights vector assembly
    coefficients = np.ones(n + 1, dtype=np.float64)
    coefficients[1:-1:2] = 4.0
    coefficients[2:-1:2] = 2.0
    
    numerical = float(h / 3.0 * np.sum(coefficients * density * log_ratio))
    
    # --- 3. Analytical Gaussian KL Divergence ---
    analytical = float(np.log(sigma2 / sigma1) + (sigma1 ** 2 + (mu1 - mu2) ** 2) / (2.0 * sigma2 ** 2) - 0.5)
    
    return {
        "discrete_kl": discrete, 
        "numerical_kl": numerical, 
        "analytical_kl": analytical
    }
