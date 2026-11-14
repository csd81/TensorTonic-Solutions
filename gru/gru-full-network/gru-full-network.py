import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
  return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


class GRU:

  def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
    self.hidden_dim = hidden_dim
    scale = np.sqrt(2.0 / (input_dim + hidden_dim))
    self.W_r = np.random.randn(hidden_dim, hidden_dim + input_dim) * scale
    self.W_z = np.random.randn(hidden_dim, hidden_dim + input_dim) * scale
    self.W_h = np.random.randn(hidden_dim, hidden_dim + input_dim) * scale
    self.b_r = np.zeros(hidden_dim)
    self.b_z = np.zeros(hidden_dim)
    self.b_h = np.zeros(hidden_dim)
    self.W_y = np.random.randn(output_dim, hidden_dim) * np.sqrt(
        2.0 / (hidden_dim + output_dim)
    )
    self.b_y = np.zeros(output_dim)

  def forward(self, X: np.ndarray) -> dict:
    """
        Returns outputs and final_hidden_state as float64 arrays.
        """
    N, T, _ = X.shape

    # 1. Initialize h_0 to zeros with shape (N, H)
    h_t = np.zeros((N, self.hidden_dim), dtype=np.float64)

    outputs_list = []

    # 2. Unroll across time steps t = 0, ..., T - 1
    for t in range(T):
      x_t = X[:, t, :]  # Shape: (N, D)

      # Concatenate [h_{t-1}, x_t] along feature axis -> (N, H + D)
      hx = np.concatenate([h_t, x_t], axis=-1)

      # Reset and update gates -> (N, H)
      r_t = sigmoid(hx @ self.W_r.T + self.b_r)
      z_t = sigmoid(hx @ self.W_z.T + self.b_z)

      # Candidate hidden state -> (N, H)
      gated_hx = np.concatenate([r_t * h_t, x_t], axis=-1)
      h_tilde = np.tanh(gated_hx @ self.W_h.T + self.b_h)

      # Update hidden state -> (N, H)
      h_t = z_t * h_t + (1.0 - z_t) * h_tilde

      # 3. Project hidden state to output space -> (N, O)
      y_t = h_t @ self.W_y.T + self.b_y
      outputs_list.append(y_t)

    # 4. Stack along time axis (axis=1) to form (N, T, O)
    outputs = np.stack(outputs_list, axis=1).astype(np.float64)

    return {
        "outputs": outputs,
        "final_hidden_state": h_t.astype(np.float64),
    }
