import numpy as np


def bert_fine_tuning_step(
    hidden_states: np.ndarray,
    labels: np.ndarray,
    classifier_W: np.ndarray,
    classifier_b: np.ndarray,
    learning_rate: float,
) -> dict:
  """Returns updated classifier parameters and the pre-update loss."""
  # 1. Extract position-zero hidden states for every sample: shape (B, D)
  H_0 = hidden_states[:, 0, :]

  # 2. Compute logits Z = H_0 @ W + b: shape (B, C)
  Z = H_0 @ classifier_W + classifier_b

  # 3. Compute numerically stable softmax probabilities P: shape (B, C)
  exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
  P = exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

  # 4. Compute mean cross-entropy loss before the update
  B = hidden_states.shape[0]
  true_probs = P[np.arange(B), labels]
  loss = float(-np.mean(np.log(true_probs)))

  # 5. Compute softmax-cross-entropy gradients
  dZ = P.copy()
  dZ[np.arange(B), labels] -= 1.0
  dZ /= B

  dW = H_0.T @ dZ
  db = np.sum(dZ, axis=0)

  # 6. Apply gradient-descent update
  new_classifier_W = classifier_W - learning_rate * dW
  new_classifier_b = classifier_b - learning_rate * db

  return {
      "new_classifier_W": new_classifier_W,
      "new_classifier_b": new_classifier_b,
      "loss": loss,
  }
