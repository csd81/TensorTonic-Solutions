import numpy as np

def streaming_minmax(D: int, batches: list, eps: float = 1e-8) -> dict:
    """
    Returns a dictionary with normalized_batches, min, and max.
    """
    running_min = np.full(D, np.inf, dtype=np.float64)
    running_max = np.full(D, -np.inf, dtype=np.float64)
    normalized_batches = []

    for batch in batches:
        batch_arr = np.asarray(batch, dtype=np.float64)

        # Update running per-feature extrema before normalizing
        running_min = np.minimum(running_min, np.min(batch_arr, axis=0))
        running_max = np.maximum(running_max, np.max(batch_arr, axis=0))

        # Compute denominator with epsilon guard against division by zero
        denom = np.maximum(running_max - running_min, eps)
        norm_batch = (batch_arr - running_min) / denom

        normalized_batches.append(norm_batch)

    return {
        "normalized_batches": normalized_batches,
        "min": running_min,
        "max": running_max
    }
