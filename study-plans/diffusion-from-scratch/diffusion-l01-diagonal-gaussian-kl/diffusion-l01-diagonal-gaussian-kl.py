import numpy as np

def diagonal_gaussian_kl(
    mean_p: np.ndarray, 
    log_var_p: np.ndarray, 
    mean_q: np.ndarray, 
    log_var_q: np.ndarray
) -> np.ndarray:
    """
    Returns KL(p || q) summed over the last axis, as a float64 array with the leading shape of the inputs.
    """
    # Ensure float64 precision without modifying inputs
    mean_p = np.asarray(mean_p, dtype=np.float64)
    log_var_p = np.asarray(log_var_p, dtype=np.float64)
    mean_q = np.asarray(mean_q, dtype=np.float64)
    log_var_q = np.asarray(log_var_q, dtype=np.float64)

    # Compute variance of q and variance ratio p/q
    var_q = np.exp(log_var_q)
    var_ratio = np.exp(log_var_p - log_var_q)
    
    # Compute squared difference of means
    mean_diff_sq = np.square(mean_p - mean_q)
    
    # Element-wise KL divergence per coordinate
    kl_per_dim = 0.5 * (log_var_q - log_var_p + var_ratio + (mean_diff_sq / var_q) - 1.0)
    
    # Sum over the last axis (D) to preserve leading dimensions
    return np.sum(kl_per_dim, axis=-1, dtype=np.float64)
