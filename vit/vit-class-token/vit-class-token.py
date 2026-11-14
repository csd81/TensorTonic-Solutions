import numpy as np

def prepend_class_token(patches: np.ndarray, cls_token: np.ndarray) -> np.ndarray:
    """
    Returns the float64 sequence with the class token at position zero.
    """
    batch_size, _, embed_dim = patches.shape
    cls_tokens = np.broadcast_to(cls_token, (batch_size, 1, embed_dim))
    return np.concatenate((cls_tokens, patches), axis=1).astype(np.float64, copy=False)
