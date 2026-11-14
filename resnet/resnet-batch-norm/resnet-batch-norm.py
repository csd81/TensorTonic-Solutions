import numpy as np


def batch_norm_block(x, W1, W2, gamma1, beta1, gamma2, beta2, mode):
  """Returns the normalized residual-block result and selected mode in a dictionary."""

  def batch_norm(z, gamma, beta):
    mean = np.mean(z, axis=0, keepdims=True)
    var = np.var(z, axis=0, keepdims=True)
    z_hat = (z - mean) / np.sqrt(var + 1e-5)
    return gamma * z_hat + beta

  def relu(z):
    return np.maximum(0, z)

  x = np.array(x, dtype=np.float64)
  W1 = np.array(W1, dtype=np.float64)
  W2 = np.array(W2, dtype=np.float64)
  gamma1 = np.array(gamma1, dtype=np.float64)
  beta1 = np.array(beta1, dtype=np.float64)
  gamma2 = np.array(gamma2, dtype=np.float64)
  beta2 = np.array(beta2, dtype=np.float64)

  if mode == "post":
    # Post-activation: Linear -> BN -> ReLU -> Linear -> BN -> Add Skip -> ReLU
    out1 = np.dot(x, W1)
    bn1 = batch_norm(out1, gamma1, beta1)
    r1 = relu(bn1)

    out2 = np.dot(r1, W2)
    bn2 = batch_norm(out2, gamma2, beta2)

    added = bn2 + x
    out = relu(added)
  elif mode == "pre":
    # Pre-activation: BN -> ReLU -> Linear -> BN -> ReLU -> Linear -> Add Skip
    bn1 = batch_norm(x, gamma1, beta1)
    r1 = relu(bn1)
    out1 = np.dot(r1, W1)

    bn2 = batch_norm(out1, gamma2, beta2)
    r2 = relu(bn2)
    out2 = np.dot(r2, W2)

    out = out2 + x
  else:
    raise ValueError(f"Unknown mode: {mode}")

  out_rounded = np.round(out, 4).tolist()

  return {"output": out_rounded, "mode": mode}
