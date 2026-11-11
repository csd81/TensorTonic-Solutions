import numpy as np

def info_nce_loss(Z1: list, Z2: list, temperature: float = 0.1) -> float:
    """
    Returns the loss as a float.
    """
    z1 = np.asarray(Z1, dtype=float)
    z2 = np.asarray(Z2, dtype=float)

    # Pairwise temperature-scaled similarity logits: shape (N, N)
    logits = (z1 @ z2.T) / temperature

    # Row-wise max subtraction for numerical stability
    row_max = np.max(logits, axis=1, keepdims=True)
    shifted = logits - row_max

    # Log-softmax over each row: -S_ii + log(sum_j exp(S_ij))
    log_sum_exp = row_max.squeeze(axis=1) + np.log(np.sum(np.exp(shifted), axis=1))
    pos_logits = np.diag(logits)

    loss = np.mean(log_sum_exp - pos_logits)
    return float(loss)
