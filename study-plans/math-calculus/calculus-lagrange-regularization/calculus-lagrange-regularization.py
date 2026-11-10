import numpy as np

def regularization_equivalence(A: list, b: list, lambda_val: float) -> dict:
    """
    Returns the unconstrained, penalized, and equivalent constrained results.
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    I = np.eye(A.shape[0])
    
    # 1. Compute unregularized solution (Aw = b)
    w_unregularized = np.linalg.solve(A, b)
    
    # 2. Compute penalized solution ((A + 2*lambda*I)w = b)
    w_regularized = np.linalg.solve(A + 2 * lambda_val * I, b)
    
    # 3. Set constraint bound as the squared norm of the penalized solution
    constraint_bound = float(np.sum(w_regularized ** 2))
    
    # 4. Use bisection to recover the nonnegative multiplier nu
    # Objective function: ||w(nu)||^2 - constraint_bound = 0
    def f(nu):
        w_nu = np.linalg.solve(A + 2 * nu * I, b)
        return np.sum(w_nu ** 2) - constraint_bound
    
    # Search space for bisection (||w(nu)||^2 decreases monotonically as nu increases)
    low = 0.0
    high = max(10.0, lambda_val * 10.0)
    
    for _ in range(100):
        mid = (low + high) / 2
        if f(mid) > 0:
            low = mid
        else:
            high = mid
            
    lagrange_multiplier = (low + high) / 2
    
    # Compute constrained solution using the recovered multiplier
    w_constrained = np.linalg.solve(A + 2 * lagrange_multiplier * I, b)
    
    # Check if the solutions match within an absolute tolerance of 10^-6
    solutions_match = bool(np.allclose(w_regularized, w_constrained, atol=1e-6))
    
    return {
        "constraint_bound": constraint_bound,
        "lagrange_multiplier": lagrange_multiplier,
        "solutions_match": solutions_match,
        "w_constrained": w_constrained.tolist(),
        "w_regularized": w_regularized.tolist(),
        "w_unregularized": w_unregularized.tolist()
    }
