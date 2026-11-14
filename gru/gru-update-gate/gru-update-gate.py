import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
  return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def update_gate(
    h_prev: np.ndarray, x_t: np.ndarray, W_z: np.ndarray, b_z: np.ndarray
) -> np.ndarray:
  """
    Returns the float64 update-gate values.
    """
  # Concatenate h_prev and x_t along the last axis: (H+D,) or (N, H+D)
  combined = np.concatenate([h_prev, x_t], axis=-1)

  # Linear transform: (H+D,) @ (H+D, H) + (H,) -> (H,) or (N, H+D) @ (H+D, H) + (H,) -> (N, H)
  linear_out = combined @ W_z.T + b_z

  # Apply sigmoid and ensure float64 dtype
  return sigmoid(linear_out).astype(np.float64)
