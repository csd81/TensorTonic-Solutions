import numpy as np

def tsne_affinity(X: list, perplexity: float = 5.0, tol: float = 1e-5, max_binary_search: int = 50) -> list:
    """
    Returns the complete symmetric t-SNE affinity matrix as nested lists.
    """
    X = np.array(X, dtype=np.float64)
    n = len(X)
    
    # Step 1: Compute all pairwise squared Euclidean distances
    sum_X = np.sum(np.square(X), axis=1)
    D = sum_X.reshape(-1, 1) + sum_X.reshape(1, -1) - 2 * np.dot(X, X.T)
    D = np.maximum(D, 0.0) # Clip negative values caused by floating-point precision issues
    
    target_entropy = np.log(perplexity)
    P_cond = np.zeros((n, n), dtype=np.float64)
    
    # Step 2: Binary-search beta for each row
    for i in range(n):
        # Exclude the diagonal element for row i
        d_i = np.delete(D[i], i)
        
        beta = 1.0
        beta_min = None
        beta_max = None
        
        for step in range(max_binary_search):
            # Compute conditional probabilities safely using a max-shift log-sum-exp trick
            shifted_d = -beta * d_i
            max_shifted = np.max(shifted_d)
            exp_d = np.exp(shifted_d - max_shifted)
            sum_exp = np.sum(exp_d)
            p_i_row_cond = exp_d / sum_exp
            
            # Compute natural log entropy (clip to prevent log(0))
            p_clean = np.maximum(p_i_row_cond, 1e-12)
            entropy = -np.sum(p_clean * np.log(p_clean))
            
            if abs(entropy - target_entropy) <= tol:
                break
                
            # Follow bounds and update rules exactly
            if entropy > target_entropy:
                beta_min = beta
                if beta_max is None:
                    beta *= 2
                else:
                    beta = (beta_min + beta_max) / 2.0
            else:
                beta_max = beta
                if beta_min is None:
                    beta /= 2
                else:
                    beta = (beta_min + beta_max) / 2.0
                    
        # Recompute final clean probabilities for row i
        shifted_d = -beta * d_i
        max_shifted = np.max(shifted_d)
        exp_d = np.exp(shifted_d - max_shifted)
        p_i_row_cond = exp_d / np.sum(exp_d)
        
        # Re-insert the 0.0 diagonal element back into position i
        P_cond[i] = np.insert(p_i_row_cond, i, 0.0)
        
    # Step 3: Symmetrize and normalize across the dataset joint probability
    P = (P_cond + P_cond.T) / (2.0 * n)
    
    # Round to four decimal places and cast back to native nested Python lists
    return np.round(P, 4).tolist()
