import numpy as np

def detect_mode_collapse(generated_samples: np.ndarray, threshold: float = 0.1) -> dict:
    """
    Returns diversity_score and is_collapsed in a dictionary.
    """
    samples = np.asarray(generated_samples, dtype=np.float64)
    
    # Population standard deviation (ddof=0) for each feature across the batch (axis=0)
    feature_stds = np.std(samples, axis=0, ddof=0)
    
    # Average standard deviation across all features
    diversity_score = float(np.mean(feature_stds))
    
    # Strict comparison against the threshold
    is_collapsed = bool(diversity_score < threshold)
    
    return {
        "diversity_score": diversity_score,
        "is_collapsed": is_collapsed
    }
