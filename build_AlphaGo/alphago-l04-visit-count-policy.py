import numpy as np

def visit_count_policy(
    visits: np.ndarray,
    legal_mask: np.ndarray,
    temperature: float,
    uniform_draw: float | None,
) -> tuple[np.ndarray, int]:
    """
    Returns the float64 policy array and selected action index.
    """
    counts = np.asarray(visits, dtype=np.int64)
    legal = np.asarray(legal_mask, dtype=bool)
    legal_indices = np.flatnonzero(legal)
    policy = np.zeros(counts.size, dtype=np.float64)

    if temperature == 0.0:
        # At temperature zero, use a one-hot policy at the lowest-index legal action with largest count
        action = int(legal_indices[np.argmax(counts[legal])])
        policy[action] = 1.0
        return policy, action

    positive = legal & (counts > 0)
    if not np.any(positive):
        # If every legal visit count is zero, use a uniform legal policy
        policy[legal] = 1.0 / legal_indices.size
    else:
        # Compute in log-space to handle tiny temperature values robustly (prevents overflow)
        log_weights = np.log(counts[positive].astype(np.float64)) / temperature
        log_weights = log_weights - log_weights.max()  # Max subtraction trick
        weights = np.exp(log_weights)
        policy[positive] = weights / weights.sum()

    # Select the first action whose cumulative probability is strictly greater than uniform_draw
    cumulative = 0.0
    action = int(legal_indices[-1])  # Fallback to the last legal action
    for index in legal_indices:
        cumulative += float(policy[index])
        if uniform_draw < cumulative:
            action = int(index)
            break

    return policy, action
