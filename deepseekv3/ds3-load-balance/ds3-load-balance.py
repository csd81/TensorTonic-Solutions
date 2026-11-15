import torch

def update_expert_bias(expert_bias: torch.Tensor, expert_load: torch.Tensor, update_speed: float) -> torch.Tensor:
    """
    Returns the updated routing-bias tensor.
    """
    mean_load = expert_load.mean()
    # Subtract update_speed if overloaded (> mean), add if underloaded (< mean), 0 if equal
    adjustment = torch.sign(mean_load - expert_load) * update_speed
    return (expert_bias + adjustment).to(torch.float64)
