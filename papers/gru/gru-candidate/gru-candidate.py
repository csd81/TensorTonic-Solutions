import numpy as np


def candidate_hidden(
    r_t: np.ndarray,
    h_prev: np.ndarray,
    x_t: np.ndarray,
    W_h: np.ndarray,
    b_h: np.ndarray,
) -> np.ndarray:
  """
    Returns the float64 candidate hidden state.
    """
  # 1. Elementwise multiply reset gate with previous hidden state: (H,) or (N, H)
  gated_h = r_t * h_prev

  # 2. Concatenate gated_h before x_t along the last axis: (H+D,) or (N, H+D)
  combined = np.concatenate([gated_h, x_t], axis=-1)

  # 3. Linear transform followed by tanh activation
  linear_out = combined @ W_h.T + b_h
  return np.tanh(linear_out).astype(np.float64)
