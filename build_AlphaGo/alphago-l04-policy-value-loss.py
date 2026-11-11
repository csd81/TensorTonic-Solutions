import torch
import torch.nn.functional as F

def alpha_go_policy_value_loss(
    policy_logits: torch.Tensor,
    predicted_values: torch.Tensor,
    target_policies: torch.Tensor,
    target_values: torch.Tensor,
    parameters: list,
    l2_coefficient: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns scalar float32 tensors: total, policy, value, and regularization loss.
    """
    # 1. Policy Loss: Cross-entropy using log_softmax for numerical stability
    log_p = F.log_softmax(policy_logits, dim=-1)
    policy_loss = -torch.mean(torch.sum(target_policies * log_p, dim=-1))
    
    # 2. Value Loss: Mean squared error between predicted and target values
    value_loss = F.mse_loss(predicted_values, target_values, reduction='mean')
    
    # 3. Regularization Loss: L2 norm squared over all parameters
    reg_loss = torch.tensor(0.0, dtype=torch.float32, device=policy_logits.device)
    for param in parameters:
        reg_loss += torch.sum(param ** 2)
    reg_loss = l2_coefficient * reg_loss
    
    # 4. Total Loss
    total_loss = policy_loss + value_loss + reg_loss
    
    return total_loss, policy_loss, value_loss, reg_loss
