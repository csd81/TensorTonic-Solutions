import numpy as np

def chain_rule_3layer(w1: float, w2: float, w3: float, x: float) -> dict:
    """
    Returns chain-rule factors and analytical and numerical gradients.
    """
    # 1. Helper function for the logistic sigmoid
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))
    
    # 2. Forward pass function to compute y given w1
    def forward(w1_val):
        z1 = w1_val * x
        a1 = sigmoid(z1)
        z2 = w2 * a1
        a2 = sigmoid(z2)
        z3 = w3 * a2
        y = sigmoid(z3)
        return z1, a1, z2, a2, z3, y

    # 3. Compute layers and activations using the actual w1
    z1, a1, z2, a2, z3, y = forward(w1)
    
    # 4. Compute the derivative values σ'(z) at each layer
    dsig_z3 = sigmoid(z3) * (1.0 - sigmoid(z3))
    dsig_z2 = sigmoid(z2) * (1.0 - sigmoid(z2))
    dsig_z1 = sigmoid(z1) * (1.0 - sigmoid(z1))
    
    # 5. Arrange the six local factors exactly in the requested order
    factors = [
        float(dsig_z3),
        float(w3),
        float(dsig_z2),
        float(w2),
        float(dsig_z1),
        float(x)
    ]
    
    # 6. Calculate the analytical gradient by multiplying the factors
    analytical_gradient = float(np.prod(factors))
    
    # 7. Calculate the numerical gradient using central difference with h = 0.00001
    h = 0.00001
    _, _, _, _, _, y_plus  = forward(w1 + h)
    _, _, _, _, _, y_minus = forward(w1 - h)
    numerical_gradient = float((y_plus - y_minus) / (2.0 * h))
    
    return {
        "factors": factors,
        "analytical_gradient": analytical_gradient,
        "numerical_gradient": numerical_gradient
    }
