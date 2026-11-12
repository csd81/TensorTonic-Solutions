import numpy as np

def compute_perplexity(token_ids: list, logits: list) -> dict:
    """
    Returns a dictionary with token_probabilities, token_nlls, avg_nll, 
    perplexity, and sequence_log_prob rounded to 4 decimal places.
    """
    epsilon = 1e-12
    Z = np.array(logits, dtype=np.float64)
    t = np.array(token_ids)
    N = len(token_ids)
    
    token_probabilities = []
    token_nlls = []
    sequence_log_prob = 0.0
    
    # Each row i in logits evaluates the next token prediction at index i + 1
    for i in range(N - 1):
        target_token = t[i + 1]
        row_logits = Z[i]
        
        # --- Numerical Stability Trick ---
        # Subtract the maximum value in the row to prevent np.exp overflow
        stable_logits = row_logits - np.max(row_logits)
        exp_logits = np.exp(stable_logits)
        p_i = exp_logits[target_token] / np.sum(exp_logits)
        
        # Negative log-likelihood and sequence log probability
        l_i = -np.log(p_i + epsilon)
        sequence_log_prob += np.log(p_i + epsilon)
        
        token_probabilities.append(p_i)
        token_nlls.append(l_i)
        
    avg_nll = np.mean(token_nlls)
    perplexity = np.exp(avg_nll)
    
    return {
        "token_probabilities": [round(float(p), 4) for p in token_probabilities],
        "token_nlls": [round(float(l), 4) for l in token_nlls],
        "avg_nll": round(float(avg_nll), 4),
        "perplexity": round(float(perplexity), 4),
        "sequence_log_prob": round(float(sequence_log_prob), 4)
    }
