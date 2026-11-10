def backprop_graph(w1: float, w2: float, x: float) -> dict:
    """
    Returns the forward intermediates and leaf gradients.
    """
    # --- Forward Pass ---
    x_sq = float(x ** 2)
    prod1 = float(w1 * x)
    prod2 = float(w2 * x_sq)
    sum_val = float(prod1 + prod2)
    output = float(max(0.0, sum_val))
    
    # --- Backward Pass ---
    # Gradient of output with respect to sum_val (ReLU derivative)
    # Use zero when input is less than or equal to zero
    d_sum_val = 1.0 if sum_val > 0.0 else 0.0
    
    # Gradients of products from the sum gate
    d_prod1 = d_sum_val
    d_prod2 = d_sum_val
    
    # Leaf gradients for weights
    d_w1 = float(d_prod1 * x)
    d_w2 = float(d_prod2 * x_sq)
    
    # Leaf gradient for x (accumulating both paths)
    d_x1 = d_prod1 * w1
    d_x_sq = d_prod2 * w2
    d_x2 = d_x_sq * (2.0 * x)
    d_x = float(d_x1 + d_x2)
    
    return {
        "forward": {
            "x_sq": x_sq,
            "prod1": prod1,
            "prod2": prod2,
            "sum_val": sum_val,
            "output": output
        },
        "gradients": {
            "d_w1": d_w1,
            "d_w2": d_w2,
            "d_x": d_x
        }
    }
