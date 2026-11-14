import numpy as np


def bert_pooler(
    hidden_states: np.ndarray, W_pool: np.ndarray, b_pool: np.ndarray
) -> np.ndarray:
  """
    Returns the float64 pooled states with shape (B, D).
    """
  # Extract the hidden state at sequence position 0 ([CLS] token) -> shape (B, D)
  cls_hidden_state = hidden_states[:, 0, :]

  # Apply affine projection (H @ W + b) and elementwise tanh
  pooled = np.tanh(cls_hidden_state @ W_pool + b_pool)

  return pooled.astype(np.float64)
