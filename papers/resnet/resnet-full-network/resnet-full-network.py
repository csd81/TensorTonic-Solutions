import numpy as np


def resnet_forward(x, conv1, W1_b1, W2_b1, W1_b2, W2_b2, Ws_b2, fc):
  """Returns the network logits as a nested list rounded to 4 decimals."""
  # Convert every numeric input to a NumPy float64 array
  x = np.array(x, dtype=np.float64)
  conv1 = np.array(conv1, dtype=np.float64)
  W1_b1 = np.array(W1_b1, dtype=np.float64)
  W2_b1 = np.array(W2_b1, dtype=np.float64)
  W1_b2 = np.array(W1_b2, dtype=np.float64)
  W2_b2 = np.array(W2_b2, dtype=np.float64)
  Ws_b2 = np.array(Ws_b2, dtype=np.float64)
  fc = np.array(fc, dtype=np.float64)

  # 1. Apply initial projection and ReLU
  h = np.maximum(0, x @ conv1)

  # 2. Identity block (skip connection = h)
  f1 = np.maximum(0, h @ W1_b1) @ W2_b1
  out1 = np.maximum(0, f1 + h)

  # 3. Projection block (skip connection = out1 @ Ws_b2)
  f2 = np.maximum(0, out1 @ W1_b2) @ W2_b2
  skip2 = out1 @ Ws_b2
  out2 = np.maximum(0, f2 + skip2)

  # 4. Final linear classifier without softmax
  logits = out2 @ fc

  # Return rounded nested list
  return np.round(logits, 4).tolist()
