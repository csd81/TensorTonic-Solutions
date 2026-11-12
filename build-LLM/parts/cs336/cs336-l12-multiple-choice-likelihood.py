import torch
import torch.nn.functional as F

def multiple_choice_likelihood(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    continuation_mask: torch.Tensor,
) -> dict:
    """
    Returns a dict: choice_scores (floating tensor), predicted_indices (int64 tensor).
    """
    # Compute log-probabilities over the vocabulary axis in the logits' native dtype
    log_probs = F.log_softmax(logits, dim=-1)

    # Gather target token log-probabilities -> shape: (batch, choices, tokens)
    target_log_probs = log_probs.gather(dim=-1, index=token_ids.unsqueeze(-1)).squeeze(-1)

    # Zero out non-continuation positions and sum over the token dimension
    mask = continuation_mask.to(dtype=torch.bool)
    masked_log_probs = torch.where(mask, target_log_probs, torch.zeros_like(target_log_probs))
    choice_scores = masked_log_probs.sum(dim=-1)

    # Select the highest-scoring choice per batch item (ties break toward the lowest index)
    predicted_indices = torch.argmax(choice_scores, dim=-1).to(dtype=torch.int64)

    return {
        "choice_scores": choice_scores,
        "predicted_indices": predicted_indices,
    }
