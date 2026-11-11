import numpy as np

def gradient_descent_step(values: list, gradients: list, learning_rate: float) -> tuple[list, float]:
    """
    Returns a fresh list of updated values and the predicted objective change.
    """
    if not values:
        return ([], 0.0)

    vals = np.asarray(values, dtype=np.float64)
    grads = np.asarray(gradients, dtype=np.float64)
    lr = np.float64(learning_rate)

    updated_vals = vals - lr * grads
    predicted_change = np.sum(grads * (updated_vals - vals))

    return ([float(v) for v in updated_vals], float(predicted_change))
