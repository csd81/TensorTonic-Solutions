import torch

def policy_value_training_step(
    features: torch.Tensor,
    policy_weight: torch.Tensor,
    policy_bias: torch.Tensor,
    value_weight: torch.Tensor,
    value_bias: torch.Tensor,
    target_policies: torch.Tensor,
    target_values: torch.Tensor,
    learning_rate: float,
    l2_coefficient: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns detached float32 tensors: loss, policy weight/bias, value weight/bias.
    """
    # Clone parameters and enable gradient tracking
    w_p = policy_weight.clone().detach().requires_grad_(True)
    b_p = policy_bias.clone().detach().requires_grad_(True)
    w_v = value_weight.clone().detach().requires_grad_(True)
    b_v = value_bias.clone().detach().requires_grad_(True)
    
    # 1. Forward through both heads
    # policy logits: shape (B, A)
    policy_logits = torch.matmul(features, w_p.t()) + b_p
    # softmax across actions (dim=-1)
    p = torch.softmax(policy_logits, dim=-1)
    
    # value logits: shape (B,)
    value_logits = torch.matmul(features, w_v) + b_v
    v = torch.tanh(value_logits)
    
    # 2. Compute the losses
    # Cross-entropy loss for policies: stable version using log_softmax is preferred, 
    # but since target_policies can contain zeros, we do -1/B * sum(target * log(p + epsilon)) 
    # or use log_softmax directly: sum(target * log_softmax)
    log_p = torch.log_softmax(policy_logits, dim=-1)
    policy_loss = -torch.mean(torch.sum(target_policies * log_p, dim=-1))
    
    # MSE loss for values
    value_loss = torch.mean((v - target_values) ** 2)
    
    # L2 Regularization (squared entries sum for all weights and biases)
    l2_reg = torch.sum(w_p ** 2) + torch.sum(b_p ** 2) + torch.sum(w_v ** 2) + torch.sum(b_v ** 2)
    
    # Total loss
    loss = policy_loss + value_loss + l2_coefficient * l2_reg
    
    # 3. Differentiate objective
    grads = torch.autograd.grad(loss, [w_p, b_p, w_v, b_v])
    
    # 4. Update parameters using standard SGD formula: theta' = theta - eta * grad
    w_p_new = w_p - learning_rate * grads[0]
    b_p_new = b_p - learning_rate * grads[1]
    w_v_new = w_v - learning_rate * grads[2]
    b_v_new = b_v - learning_rate * grads[3]
    
    # Return detached tensors ensuring original shapes and devices are preserved
    return (
        loss.detach(),
        w_p_new.detach(),
        b_p_new.detach(),
        w_v_new.detach(),
        b_v_new.detach()
    )
