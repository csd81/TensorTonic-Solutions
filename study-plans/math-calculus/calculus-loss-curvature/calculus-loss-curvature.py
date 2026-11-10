import numpy as np

def loss_curvature_analysis(y_hat: list, y: list, delta: float) -> dict:
    """
    Returns first and second derivatives for three losses.
    """
    # Convert lists to NumPy arrays for efficient vectorized computations
    yh = np.array(y_hat, dtype=np.float64)
    y_arr = np.array(y, dtype=np.float64)
    
    # --- 1. Mean Squared Error (MSE) ---
    mse_dl = 2.0 * (yh - y_arr)
    mse_d2l = np.full_like(yh, 2.0)
    
    # --- 2. Binary Cross-Entropy (CE) ---
    ce_dl = (yh - y_arr) / (yh * (1.0 - yh))
    ce_d2l = (y_arr / (yh ** 2)) + ((1.0 - y_arr) / ((1.0 - yh) ** 2))
    
    # --- 3. Huber Loss ---
    r = yh - y_arr
    abs_r = np.abs(r)
    
    # Condition: |r| <= delta (includes boundary as per requirements)
    in_quadratic = abs_r <= delta
    
    # Vectorized assignment based on the current branch
    huber_dl = np.where(in_quadratic, r, delta * np.sign(r))
    huber_d2l = np.where(in_quadratic, 1.0, 0.0)
    
    # Construct the final nested dictionary converting arrays to standard Python lists
    return {
        "mse": {
            "dL": mse_dl.tolist(),
            "d2L": mse_d2l.tolist()
        },
        "ce": {
            "dL": ce_dl.tolist(),
            "d2L": ce_d2l.tolist()
        },
        "huber": {
            "dL": huber_dl.tolist(),
            "d2L": huber_d2l.tolist()
        }
    }
