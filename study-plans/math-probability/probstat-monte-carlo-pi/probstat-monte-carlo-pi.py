import numpy as np

def monte_carlo_pi(n_points: int, seed: int) -> dict:
    """
    Returns the pi estimate and its absolute error.
    """
    rng = np.random.default_rng(seed)
    
    points = rng.random((n_points, 2))
    
    squared_sums = points[:, 0]**2 + points[:, 1]**2
    m = np.sum(squared_sums <= 1.0)
    
    estimate = 4 * m / n_points
    error = abs(estimate - np.pi)
    
    
    return {
        "error": round(float(error), 4),
        "estimate": round(float(estimate), 4)
    }