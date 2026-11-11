import numpy as np


def bigram_probabilities(tokens: list) -> dict:
    """
    Returns a dictionary with vocab, counts, and probabilities.
    """
    # 1. Build sorted vocabulary and index lookup
    vocab = sorted(set(tokens))
    vocab_size = len(vocab)
    token_to_idx = {token: idx for idx, token in enumerate(vocab)}

    # 2. Populate V x V integer bigram count matrix
    counts = np.zeros((vocab_size, vocab_size), dtype=np.int64)
    for i in range(len(tokens) - 1):
        row = token_to_idx[tokens[i]]
        col = token_to_idx[tokens[i + 1]]
        counts[row, col] += 1

    # 3. Apply Add-1 (Laplace) smoothing per row
    row_sums = counts.sum(axis=1, keepdims=True)
    probabilities = (counts + 1.0) / (row_sums + vocab_size)

    return {
        "vocab": vocab,
        "counts": counts,
        "probabilities": probabilities,
    }
