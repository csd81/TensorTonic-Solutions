import numpy as np

def softmax_cross_entropy_gradient(logits: list, labels: list) -> dict:
    """
    Returns probabilities, loss, two gradients, and their comparison.
    """
    logits = np.array(logits, dtype=float)
    labels = np.array(labels, dtype=float)
    
    # 1. Numerically stable softmax probabilities
    shifted_logits = logits - np.max(logits)
    exp_logits = np.exp(shifted_logits)
    softmax_probs = exp_logits / np.sum(exp_logits)
    
    # 2. Cross-entropy loss
    loss = -np.sum(labels * np.log(softmax_probs))
    
    # 3. Logit gradient via Chain Rule through the Softmax Jacobian
    # dL/dp = -y / p
    dL_dp = -labels / softmax_probs
    
    # Constructing the Softmax Jacobian matrix J where J[i, j] = dp_i / dz_j
    n = len(logits)
    jacobian = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                jacobian[i, j] = softmax_probs[i] * (1.0 - softmax_probs[j])
            else:
                jacobian[i, j] = -softmax_probs[i] * softmax_probs[j]
                
    # dL/dz = dL/dp @ J
    grad_chain_rule = np.dot(dL_dp, jacobian)
    
    # 4. Direct gradient verification (p - y)
    grad_direct = softmax_probs - labels
    
    # 5. Check if they match within absolute tolerance of 10^-7
    match = bool(np.allclose(grad_chain_rule, grad_direct, atol=1e-7))
    
    return {
        "grad_chain_rule": grad_chain_rule.tolist(),
        "grad_direct": grad_direct.tolist(),
        "loss": float(loss),
        "match": match,
        "softmax": softmax_probs.tolist()
    }
