def vanilla_gradient_descent(x0: float, y0: float, lr: float, n_iters: int) -> dict:
    """
    Returns the trajectory, final point, and final function value.
    """
    current_x = float(x0)
    current_y = float(y0)
    
    # Trajectory starts with the initial point
    trajectory = [[current_x, current_y]]
    
    for _ in range(n_iters):
        # Compute gradient components at the current position
        grad_x = 2.0 * current_x
        grad_y = 6.0 * current_y
        
        # Simultaneous update
        current_x = current_x - lr * grad_x
        current_y = current_y - lr * grad_y
        
        # Append updated point to history
        trajectory.append([current_x, current_y])
        
    final_point = [current_x, current_y]
    final_value = (current_x ** 2) + 3.0 * (current_y ** 2)
    
    return {
        "trajectory": trajectory,
        "final_point": final_point,
        "final_value": final_value
    }
