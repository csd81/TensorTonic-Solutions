import numpy as np

def layer_normalization(x: list, gamma: list, beta: list, eps: float = 1e-05, mode: str = 'forward', d_output: list | None = None) -> dict:
    """
    Returns forward results and optional backward gradients as a dictionary of lists.
    """
    # Convert inputs to float64 for high numerical stability
    X = np.array(x, dtype=np.float64)
    G = np.array(gamma, dtype=np.float64)
    B = np.array(beta, dtype=np.float64)
    N, D = X.shape

    # --- Forward Pass ---
    # Compute population statistics along each row independently
    mean = np.mean(X, axis=1, keepdims=True)
    var = np.var(X, axis=1, keepdims=True)
    
    # Normalize
    std_inv = 1.0 / np.sqrt(var + eps)
    X_hat = (X - mean) * std_inv
    
    # Scale and shift
    out = G * X_hat + B

    # Custom rounding helper to match Python primitive rounding behavior
    def r4(arr):
        if arr.ndim == 1:
            return [round(float(v), 4) for v in arr]
        return [[round(float(v), 4) for v in row] for row in arr]

    # Prepare forward results using the platform-compliant rounding rule
    result = {
        'output': r4(out),
        'mean': r4(mean.flatten()),
        'var': r4(var.flatten()),
        'x_hat': r4(X_hat)
    }

    # --- Backward Pass ---
    if mode == 'backward' and d_output is not None:
        dY = np.array(d_output, dtype=np.float64)
        
        # Gradients w.r.t parameters
        dgamma = np.sum(dY * X_hat, axis=0)
        dbeta = np.sum(dY, axis=0)
        
        # Gradient w.r.t input x
        dX_hat = dY * G
        dvar = np.sum(dX_hat * (X - mean) * -0.5 * (std_inv ** 3), axis=1, keepdims=True)
        dmean = np.sum(dX_hat * -std_inv, axis=1, keepdims=True) + dvar * np.mean(-2.0 * (X - mean), axis=1, keepdims=True)
        
        # Simplified standard expression for dX 
        dX = dX_hat * std_inv + dvar * 2.0 * (X - mean) / D + dmean / D
        
        # Add backward metrics using the same rounding function
        result['dx'] = r4(dX)
        result['dgamma'] = r4(dgamma)
        result['dbeta'] = r4(dbeta)

    return result
