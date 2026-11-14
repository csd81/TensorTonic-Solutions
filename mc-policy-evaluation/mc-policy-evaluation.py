import numpy as np

def mc_policy_evaluation(episodes: list, gamma: float, n_states: int) -> np.ndarray:
    """
    Returns the state values as a NumPy array.
    """
    returns_sum = np.zeros(n_states, dtype=np.float64)
    returns_count = np.zeros(n_states, dtype=np.int64)

    for episode in episodes:
        G = 0.0
        first_visit_returns = {}
        
        # Traverse the episode backward
        for state, reward in reversed(episode):
            G = reward + gamma * G
            # Overwriting ensures we keep the earliest occurrence (t closest to 0)
            first_visit_returns[state] = G
            
        # Accumulate first-visit returns for the episode
        for state, g_val in first_visit_returns.items():
            returns_sum[state] += g_val
            returns_count[state] += 1

    # Compute averages safely (unvisited states remain 0.0)
    V = np.zeros(n_states, dtype=np.float64)
    visited = returns_count > 0
    V[visited] = returns_sum[visited] / returns_count[visited]

    return np.round(V, 4)
