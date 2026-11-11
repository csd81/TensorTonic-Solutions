import numpy as np

def generate_random_array(shape: list, kind: str, seed: int) -> np.ndarray:
    """
    Returns a seeded 2D float64 random array with the requested shape.
    """
    rng = np.random.default_rng(seed)
    if kind=="uniform":
        return rng.random(size=shape, dtype=np.float64)
    if kind=="normal":
        return rng.standard_normal(size=shape, dtype=np.float64)
    else:
        raise ValueError("Kind must be either 'uniform' or 'normal'")  