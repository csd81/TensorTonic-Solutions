def gradient_descent_quadratic(a: float, b: float, c: float, x0: float, lr: float, steps: int) -> float:
    """
    Returns the final scalar x after the requested iterations.
    """
    x = x0
    for _ in range(steps):
        # Calculate the gradient (derivative) of the quadratic function
        gradient = 2 * a * x + b
        
        # Update x by moving in the opposite direction of the gradient
        x -= lr * gradient
        
    return float(x)
