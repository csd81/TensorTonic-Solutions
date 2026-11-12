import numpy as np

def temperature_data_mixture(token_counts: list, alpha: int | float, training_budget: int) -> dict:
    """
    Returns a dict: probabilities, expected_tokens, expected_epochs (float64 arrays).
    """
    counts = np.asarray(token_counts, dtype=np.float64)
    n_sources = counts.shape[0]

    probabilities = np.zeros(n_sources, dtype=np.float64)
    expected_tokens = np.zeros(n_sources, dtype=np.float64)
    expected_epochs = np.zeros(n_sources, dtype=np.float64)

    pos = counts > 0
    if not np.any(pos):
        return {
            "probabilities": probabilities,
            "expected_tokens": expected_tokens,
            "expected_epochs": expected_epochs,
        }

    pos_counts = counts[pos]
    log_counts = np.log(pos_counts)
    max_log = np.max(log_counts)

    # Log-sum-exp style stabilization so huge counts/exponents never overflow
    scaled_logs = float(alpha) * (log_counts - max_log)
    weights = np.exp(scaled_logs)
    pos_probs = weights / np.sum(weights)

    probabilities[pos] = pos_probs
    expected_tokens[pos] = float(training_budget) * pos_probs
    expected_epochs[pos] = expected_tokens[pos] / pos_counts

    return {
        "probabilities": probabilities,
        "expected_tokens": expected_tokens,
        "expected_epochs": expected_epochs,
    }
