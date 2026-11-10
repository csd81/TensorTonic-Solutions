import numpy as np

def softmax_stability_analysis(z: list) -> dict:
    """
    Returns direct and stable softmax results with an issue flag.
    """
    logits = np.asarray(z, dtype=np.float64)
    
    # 1. Compute naive softmax directly
    with np.errstate(all='ignore'):
        exponentials = np.exp(logits)
        naive = exponentials / np.sum(exponentials)
        
    # 2. Compute numerically stable softmax
    with np.errstate(all='ignore'):
        shifted = np.exp(logits - np.max(logits))
        stable = shifted / np.sum(shifted)
        
    # 3. Detect nonfinite values in the naive result
    naive_has_issues = bool(np.any(~np.isfinite(naive)))
    
    return {
        "naive": naive.tolist(),
        "stable": stable.tolist(),
        "naive_has_issues": naive_has_issues
    }
