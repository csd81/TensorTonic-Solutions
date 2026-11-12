import numpy as np

def select_puct_action(visits: np.ndarray, total_values: np.ndarray, priors: np.ndarray, legal_mask: np.ndarray, cpuct: float) -> tuple[int, np.ndarray]:
    """
    Returns: a tuple of the Python action index and the float64 score array.
    """
    # Ensure float64 precision for computation and output
    visits_f = visits.astype(np.float64)
    total_values_f = total_values.astype(np.float64)
    priors_f = priors.astype(np.float64)
    
    # Compute Q_a: avoid division by zero using np.where
    # Q_a = 0 where visits == 0, else total_values / visits
    q = np.where(visits_f == 0.0, 0.0, total_values_f / visits_f)
    
    # Compute T = sum(N_b)
    T = np.sum(visits_f)
    
    # Compute the selection score S_a
    # Note: Using np.sqrt(T) / (1 + visits_f)
    scores = q + cpuct * priors_f * (np.sqrt(T) / (1.0 + visits_f))
    
    # Set illegal actions to negative infinity (-inf)
    scores = np.where(legal_mask, scores, -np.inf)
    
    # Find the best action: np.argmax returns the first occurrence (lowest index) in case of a tie
    best_action = int(np.argmax(scores))
    
    return best_action, scores
