def lr_stability_analysis(curvature: float, w0: float, n_iters: int) -> dict:
    """
    Returns the stability bound, learning rates, and three trajectories.
    """
    # 1. Compute the upper convergence bound
    convergence_bound = 2.0 / curvature
    
    # 2. Define the three test learning rates
    alpha_convergent = 0.5 / curvature
    alpha_oscillatory = 1.9 / curvature
    alpha_divergent = 2.1 / curvature
    
    alpha_values = [alpha_convergent, alpha_oscillatory, alpha_divergent]
    
    # Helper function to generate full trajectories
    def generate_trajectory(alpha: float) -> list:
        trajectory = [float(w0)]
        current_w = float(w0)
        multiplier = 1.0 - alpha * curvature
        
        for _ in range(n_iters):
            current_w = current_w * multiplier
            trajectory.append(current_w)
            
        return trajectory

    # 3. Simulate all three trajectories
    convergent_trajectory = generate_trajectory(alpha_convergent)
    oscillatory_trajectory = generate_trajectory(alpha_oscillatory)
    divergent_trajectory = generate_trajectory(alpha_divergent)
    
    # 4. Construct and return the dictionary
    return {
        "alpha_values": alpha_values,
        "convergence_bound": convergence_bound,
        "convergent_trajectory": convergent_trajectory,
        "divergent_trajectory": divergent_trajectory,
        "oscillatory_trajectory": oscillatory_trajectory
    }
