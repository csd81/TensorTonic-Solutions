def gradient_descent_nonconvex(w0: float, lr: float, n_iters: int) -> dict:
    """
    Returns critical points, classifications, and the full trajectory.
    """
    # 1. Critical points from L'(w) = 4w^3 - 6w = 2w(2w^2 - 3) = 0
    # Roots are -sqrt(1.5), 0, +sqrt(1.5)
    root_val = (1.5) ** 0.5
    critical_points = [-root_val, 0.0, root_val]
    
    # 2. Classify critical points using L''(w) = 12w^2 - 6
    classifications = []
    for w in critical_points:
        d2L = 12 * (w ** 2) - 6
        if d2L > 0:
            classifications.append("local_min")
        elif d2L < 0:
            classifications.append("local_max")
        else:
            classifications.append("saddle")  # Fallback rule if d2L == 0
            
    # 3. Simulate gradient descent trajectory
    trajectory = [float(w0)]
    w_current = w0
    for _ in range(n_iters):
        # L'(w) = 4w^3 - 6w
        grad = 4 * (w_current ** 3) - 6 * w_current
        w_current = w_current - lr * grad
        trajectory.append(float(w_current))
        
    return {
        "classifications": classifications,
        "critical_points": critical_points,
        "trajectory": trajectory
    }
