def lr_schedule_analysis(alpha_0: float, k: float) -> dict:
    """
    Returns the schedule limit and two convergence results.
    """
    # If initial learning rate is 0, everything is zero
    if alpha_0 == 0.0:
        return {
            "limit": 0.0,
            "sum_diverges": False,
            "sum_sq_converges": True
        }
    
    # If decay constant k is 0, the learning rate remains constant at alpha_0
    if k == 0.0:
        return {
            "limit": float(alpha_0),
            "sum_diverges": True,
            "sum_sq_converges": False
        }
    
    # Standard inverse-time decay behavior (k > 0 and alpha_0 > 0)
    return {
        "limit": 0.0,
        "sum_diverges": True,
        "sum_sq_converges": True
    }
