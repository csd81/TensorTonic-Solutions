import torch

def compute_loss(pred: torch.Tensor, target: torch.Tensor, method: str, delta: float = 1.0) -> float:
    """
    Returns the mean loss as a Python float using PyTorch tensor operations.
    """
    if method == "mse":
        loss = torch.mean((pred - target) ** 2)
        return float(loss.item())
        
    elif method == "cross_entropy":
        # Manually shift the logits for perfect float32 numerical stability
        shifted = pred - pred.max(dim=1, keepdim=True).values
        
        # Compute log-sum-exp on the stable, shifted values
        log_sum_exp = torch.log(torch.sum(torch.exp(shifted), dim=1))
        
        # Gather the logit corresponding to the target class
        row_indices = torch.arange(pred.shape[0])
        target_logits = shifted[row_indices, target]
        
        loss = torch.mean(log_sum_exp - target_logits)
        return float(loss.item())

        
    elif method == "huber":
        abs_diff = torch.abs(pred - target)
        
        # Apply the conditional Huber loss formula element-wise
        huber_elements = torch.where(
            abs_diff <= delta,
            0.5 * (abs_diff ** 2),
            delta * (abs_diff - 0.5 * delta)
        )
        
        loss = torch.mean(huber_elements)
        return float(loss.item())
        
    else:
        raise ValueError(f"Unknown loss method: {method}")
