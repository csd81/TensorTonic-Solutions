import torch
import torch.nn.functional as F

def direct_preference_optimization_loss(
    policy_chosen: torch.Tensor, policy_rejected: torch.Tensor,
    reference_chosen: torch.Tensor, reference_rejected: torch.Tensor,
    beta: int | float, label_smoothing: int | float = 0.0,
) -> dict:
    """
    Returns a dict: loss, per_example_loss, chosen_rewards, rejected_rewards, preference_accuracy (tensors).
    """
    dtype = policy_chosen.dtype
    device = policy_chosen.device
    beta_t = torch.tensor(beta, dtype=dtype, device=device)
    eps_t = torch.tensor(label_smoothing, dtype=dtype, device=device)

    chosen_rewards = beta_t * (policy_chosen - reference_chosen)
    rejected_rewards = beta_t * (policy_rejected - reference_rejected)
    logits = chosen_rewards - rejected_rewards

    per_example_loss = (
        -(1.0 - eps_t) * F.logsigmoid(logits)
        - eps_t * F.logsigmoid(-logits)
    )
    loss = per_example_loss.mean()
    preference_accuracy = (chosen_rewards > rejected_rewards).to(dtype).mean()

    return {
        "loss": loss,
        "per_example_loss": per_example_loss,
        "chosen_rewards": chosen_rewards,
        "rejected_rewards": rejected_rewards,
        "preference_accuracy": preference_accuracy,
    }
