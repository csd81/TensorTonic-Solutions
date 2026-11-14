import numpy as np

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Returns the position-wise feed-forward output.
    """
    # Convert inputs to float64 to satisfy precision requirements
    x = np.asarray(x, dtype=np.float64)
    W1 = np.asarray(W1, dtype=np.float64)
    b1 = np.asarray(b1, dtype=np.float64)
    W2 = np.asarray(W2, dtype=np.float64)
    b2 = np.asarray(b2, dtype=np.float64)
    
    # First linear transformation + ReLU activation
    hidden = np.maximum(0.0, x @ W1 + b1)
    
    # Second linear transformation
    output = hidden @ W2 + b2
    
    return output
