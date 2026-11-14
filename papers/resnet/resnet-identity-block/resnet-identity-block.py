import numpy as np

def identity_block(x, W1, W2):
    """
    Returns the identity residual-block output as a nested list.
    """
    x = np.asarray(x, dtype=np.float64)
    W1 = np.asarray(W1, dtype=np.float64)
    W2 = np.asarray(W2, dtype=np.float64)
    
    # First linear transformation + ReLU
    hidden = np.maximum(0.0, x @ W1.T)
    
    # Second linear transformation + residual skip connection + ReLU
    out = np.maximum(0.0, hidden @ W2.T + x)
    
    return np.round(out, 4).tolist()
