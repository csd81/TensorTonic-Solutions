import numpy as np

def classification_head(encoder_output: np.ndarray, W_head: np.ndarray) -> np.ndarray:
    """
    Returns float64 class logits with shape (B, C).
    """
    eps = 1e-6
    
    # 1. Select sequence position zero ([CLS] token), shape: (B, D)
    h = encoder_output[:, 0, :]
    
    # 2. Compute mean and population variance across the embedding dimension D
    mean = np.mean(h, axis=-1, keepdims=True)
    var = np.var(h, axis=-1, keepdims=True)  # ddof=0 by default (population variance)
    
    # 3. Apply LayerNorm
    h_norm = (h - mean) / np.sqrt(var + eps)
    
    # 4. Project normalized token to class logits, shape: (B, C)
    logits = h_norm @ W_head
    
    return logits.astype(np.float64)
