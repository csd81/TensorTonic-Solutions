import numpy as np

def squeeze_excitation(x: list, W1: list, b1: list, W2: list, b2: list) -> dict:
    """
    Implements a Squeeze-and-Excitation Block for a channel-first feature map.
    Returns a dictionary containing squeezed, z1, scale, and output arrays as lists.
    """
    # Convert inputs to float64 NumPy arrays
    x = np.array(x, dtype=np.float64)
    W1 = np.array(W1, dtype=np.float64)
    b1 = np.array(b1, dtype=np.float64)
    W2 = np.array(W2, dtype=np.float64)
    b2 = np.array(b2, dtype=np.float64)
    
    # 1. Squeeze: Global Average Pooling across spatial dimensions (H, W)
    s = np.mean(x, axis=(1, 2))
    
    # 2. Excitation Step 1: Fully connected layer + ReLU
    z1 = np.dot(W1, s) + b1
    h = np.maximum(0, z1)
    
    # 3. Excitation Step 2: Fully connected layer + Sigmoid
    z2 = np.dot(W2, h) + b2
    scale = 1 / (1 + np.exp(-z2))
    
    # 4. Scale: Multiply original feature maps channel-wise by the scale factors
    output = x * scale[:, np.newaxis, np.newaxis]
    
    # Return rounded values to 4 decimal places inside standard Python lists
    return {
        "squeezed": np.round(s, 4).tolist(),
        "z1": np.round(h, 4).tolist(),
        "scale": np.round(scale, 4).tolist(),
        "output": np.round(output, 4).tolist()
    }
