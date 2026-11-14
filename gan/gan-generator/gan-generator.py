import numpy as np


def generator(z: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
  """
    Returns generated samples as a float64 array with shape (B, D).
    """
  z_arr = np.asarray(z, dtype=np.float64)
  W_arr = np.asarray(W, dtype=np.float64)
  b_arr = np.asarray(b, dtype=np.float64)

  logits = z_arr @ W_arr + b_arr
  return np.tanh(logits).astype(np.float64)
