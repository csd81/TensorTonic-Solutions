import numpy as np

def initialize_mcts_edges(policy_logits: np.ndarray, legal_mask: np.ndarray) -> dict:
    """
    Returns: N as an int64 array; W, Q, and P as float64 arrays in a dictionary.
    """
    n_actions = len(policy_logits)
    
    # Initialize MCTS statistics with zeros using specified dtypes
    N = np.zeros(n_actions, dtype=np.int64)
    W = np.zeros(n_actions, dtype=np.float64)
    Q = np.zeros(n_actions, dtype=np.float64)
    P = np.zeros(n_actions, dtype=np.float64)
    
    # Extract logits for legal moves to find the maximum safely
    legal_logits = policy_logits[legal_mask]
    m = np.max(legal_logits)
    
    # Compute the stable softmax probabilities only for the legal actions
    exp_logits = np.exp(legal_logits - m)
    priors = exp_logits / np.sum(exp_logits)
    
    # Map the computed priors back onto the legal slots of the P array
    P[legal_mask] = priors
    
    return {"N": N, "W": W, "Q": Q, "P": P}
