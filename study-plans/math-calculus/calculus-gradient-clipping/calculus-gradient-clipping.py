import numpy as np

def gradient_clipping(gradients: list, max_norm: float) -> dict:
    """
    Returns clipped gradients, norms, and clipping checks.
    """
    # 1. Calculate the squared norm of each gradient sub-array to find the global L2 norm
    squared_sum = 0.0
    flat_orig = []
    for g in gradients:
        arr = np.asarray(g, dtype=np.float64)
        squared_sum += np.sum(arr ** 2)
        flat_orig.extend(arr.flatten())
        
    orig_norm = float(np.sqrt(squared_sum))
    flat_orig = np.array(flat_orig, dtype=np.float64)
    
    # 2. Determine scaling factor and clipping status
    was_clipped = orig_norm > max_norm
    scale = max_norm / orig_norm if was_clipped else 1.0
    
    # 3. Apply the shared scale factor and rebuild the nested structure
    clipped_gradients = []
    flat_clipped = []
    for g in gradients:
        arr = np.asarray(g, dtype=np.float64)
        clipped_arr = arr * scale
        clipped_gradients.append(clipped_arr.tolist())
        flat_clipped.extend(clipped_arr.flatten())
        
    flat_clipped = np.array(flat_clipped, dtype=np.float64)
    
    # 4. Calculate the clipped global norm
    clipped_norm = float(orig_norm * scale)
    
    # 5. Check direction preservation using the 1e-6 absolute tolerance
    if orig_norm == 0.0:
        direction_preserved = True
    else:
        # Cosine similarity between original and clipped vectors
        dot_prod = np.dot(flat_orig, flat_clipped)
        norm_orig = np.linalg.norm(flat_orig)
        norm_clipped = np.linalg.norm(flat_clipped)
        
        if norm_orig == 0.0 or norm_clipped == 0.0:
            direction_preserved = True
        else:
            cosine_sim = dot_prod / (norm_orig * norm_clipped)
            direction_preserved = bool(np.abs(1.0 - cosine_sim) <= 1e-6)
            
    return {
        "clipped_global_norm": clipped_norm,
        "clipped_gradients": clipped_gradients,
        "direction_preserved": direction_preserved,
        "original_global_norm": orig_norm,
        "was_clipped": bool(was_clipped)
    }
