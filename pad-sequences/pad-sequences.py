import numpy as np

def pad_sequences(seqs: list, pad_value: int = 0, max_len: int | None = None) -> np.ndarray:
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    if not seqs:
        return np.empty((0, 0), dtype=int)

    target_len = max_len if max_len is not None else max((len(seq) for seq in seqs), default=0)
    out = np.full((len(seqs), target_len), pad_value, dtype=int)

    for i, seq in enumerate(seqs):
        if target_len > 0 and seq:
            truncated = seq[:target_len]
            out[i, :len(truncated)] = truncated

    return out
