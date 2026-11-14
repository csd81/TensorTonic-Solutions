import numpy as np

def q_learning_update(Q: list, s: int, a: int, r: float, s_next: int, alpha: float, gamma: float) -> np.ndarray:
    """
    Returns a NumPy array with the same shape as Q.
    """
    # Create a float copy so the original input isn't modified and decimals aren't truncated
    q_updated = np.array(Q, dtype=float)
    
    # Calculate the TD target using the max Q-value of the next state
    target = r + gamma * np.max(q_updated[s_next])
    
    # Update only the selected (s, a) entry
    q_updated[s, a] += alpha * (target - q_updated[s, a])
    
    return q_updated
