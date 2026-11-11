import numpy as np

def dropout(
    x: list,
    p: float = 0.5,
    rng: np.random.Generator = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (output, dropout_pattern) as NumPy arrays matching the shape of x.
    """
    x_arr = np.asarray(x, dtype=float)
    
    # Generate uniform random samples in [0.0, 1.0)
    rand_vals = rng.random(x_arr.shape) if rng is not None else np.random.random(x_arr.shape)
    
    # Retain elements with probability (1 - p) and scale by 1 / (1 - p)
    keep_mask = rand_vals >= p
    dropout_pattern = keep_mask.astype(float) / (1.0 - p)
    
    output = x_arr * dropout_pattern
    return output, dropout_pattern
