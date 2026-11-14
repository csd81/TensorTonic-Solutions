import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
  return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def gru_cell(
    x_t: np.ndarray,
    h_prev: np.ndarray,
    W_r: np.ndarray,
    W_z: np.ndarray,
    W_h: np.ndarray,
    b_r: np.ndarray,
    b_z: np.ndarray,
    b_h: np.ndarray,
) -> np.ndarray:
  """
    Returns the float64 next hidden state.
    """
  # Concatenate [h_prev, x_t] once for both gates (note: h_prev comes first!)
  hx = np.concatenate([h_prev, x_t], axis=-1)

  # 1. Reset gate
  r_t = sigmoid(hx @ W_r.T + b_r)

  # 2. Update gate
  z_t = sigmoid(hx @ W_z.T + b_z)

  # 3. Candidate hidden state
  gated_hx = np.concatenate([r_t * h_prev, x_t], axis=-1)
  h_tilde = np.tanh(gated_hx @ W_h.T + b_h)

  # 4. Next hidden state
  h_t = z_t * h_prev + (1.0 - z_t) * h_tilde

  return h_t.astype(np.float64)
