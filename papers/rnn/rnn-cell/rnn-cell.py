import numpy as np


def rnn_cell(
    x_t: np.ndarray,
    h_prev: np.ndarray,
    W_xh: np.ndarray,
    W_hh: np.ndarray,
    b_h: np.ndarray,
) -> np.ndarray:
  """Returns the float64 next hidden state."""
  x_t = np.asarray(x_t, dtype=np.float64)
  h_prev = np.asarray(h_prev, dtype=np.float64)
  W_xh = np.asarray(W_xh, dtype=np.float64)
  W_hh = np.asarray(W_hh, dtype=np.float64)
  b_h = np.asarray(b_h, dtype=np.float64)

  pre_activation = x_t @ W_xh.T + h_prev @ W_hh.T + b_h
  return np.tanh(pre_activation).astype(np.float64)
