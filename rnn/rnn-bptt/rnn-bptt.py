import numpy as np


def bptt_single_step(
    dh_next: np.ndarray,
    h_t: np.ndarray,
    h_prev: np.ndarray,
    x_t: np.ndarray,
    W_hh: np.ndarray,
    W_xh: np.ndarray,
) -> dict:
  """Returns the five float64 local RNN gradients."""
  dh_next = np.asarray(dh_next, dtype=np.float64)
  h_t = np.asarray(h_t, dtype=np.float64)
  h_prev = np.asarray(h_prev, dtype=np.float64)
  x_t = np.asarray(x_t, dtype=np.float64)
  W_hh = np.asarray(W_hh, dtype=np.float64)
  W_xh = np.asarray(W_xh, dtype=np.float64)

  # Derivative through tanh: da_t = dh_next * (1 - h_t^2)
  da_t = dh_next * (1.0 - h_t**2)

  # Gradients with respect to inputs and previous hidden state
  dh_prev = da_t @ W_hh
  dx = da_t @ W_xh

  # Gradients with respect to parameters (summed across batch)
  dW_hh = da_t.T @ h_prev
  dW_xh = da_t.T @ x_t
  db_h = np.sum(da_t, axis=0)

  return {
      "dh_prev": dh_prev,
      "dx": dx,
      "dW_hh": dW_hh,
      "dW_xh": dW_xh,
      "db_h": db_h,
  }
