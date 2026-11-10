import numpy as np

def scaled_dot_product_attention(Q: list, K: list, V: list) -> np.ndarray:
    """
    Returns the scaled-attention output as a float64 array.
    """
    # Convert lists to float64 NumPy arrays
    Q_arr = np.array(Q, dtype=np.float64)
    K_arr = np.array(K, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    
    # Get the key dimension size
    d_k = Q_arr.shape[1]
    
    # 1. Compute attention scores: S = (Q @ K.T) / sqrt(d_k)
    scores = np.dot(Q_arr, K_arr.T) / np.sqrt(d_k)
    
    # 2. Compute numerically stable softmax row-wise
    # Subtract row-wise maximums
    max_scores = np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(scores - max_scores)
    
    # Normalize by the sum of exponentials along each row
    sum_exp_scores = np.sum(exp_scores, axis=1, keepdims=True)
    P = exp_scores / sum_exp_scores
    
    # 3. Compute the final output matrix: O = P @ V
    O = np.dot(P, V_arr)
    
    return O
