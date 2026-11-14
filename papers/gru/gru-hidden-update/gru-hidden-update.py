import numpy as np


def hidden_state_update(
    z_t: np.ndarray, h_prev: np.ndarray, h_tilde: np.ndarray
) -> np.ndarray:
  """
    Returns the float64 updated hidden state.
    """
  # Linear interpolation between old hidden state and new candidate state
  h_t = z_t * h_prev + (1.0 - z_t) * h_tilde
  return h_t.astype(np.float64)
