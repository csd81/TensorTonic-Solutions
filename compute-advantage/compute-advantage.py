import numpy as np

def compute_advantage(states: list, rewards: list, V: list, gamma: float) -> np.ndarray:
    """
    Returns the advantages as a NumPy array.
    """
    T = len(rewards)
    returns = np.zeros(T, dtype=np.float64)
    
    # Compute discounted returns in one backward pass
    G = 0.0
    for t in reversed(range(T)):
        G = rewards[t] + gamma * G
        returns[t] = G
        
    # Subtract the state value baseline for each visited state
    V_arr = np.asarray(V, dtype=np.float64)
    advantages = returns - V_arr[states]
    
    return np.round(advantages, 4)
