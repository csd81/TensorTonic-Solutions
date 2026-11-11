import numpy as np

def apply_causal_mask(scores: list, mask_value: float = -1e9) -> np.ndarray:
    """
    Returns a causally masked NumPy array matching the shape of scores.
    """
    masked = np.array(scores, dtype=float, copy=True)
    T = masked.shape[-1]
    future_mask = np.triu(np.ones((T, T), dtype=bool), k=1)
    masked[..., future_mask] = mask_value
    return masked
