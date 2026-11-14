import torch
import torch.nn.functional as F

def tiny_mask_predictor(ids: torch.Tensor, embedding: torch.Tensor, readout: torch.Tensor) -> torch.Tensor:
    """
    Returns logits of shape (rows, length, vocabulary) for token IDs of shape (rows, length).
    """
    hidden = embedding[ids]
    context = hidden.mean(dim=1, keepdim=True)
    return (hidden + context) @ readout

def llada_classifier_free_guidance(
    x: torch.Tensor, 
    prompt_length: int, 
    mask_id: int, 
    w: float, 
    embedding: torch.Tensor, 
    readout: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns the prompt-masked IDs, the guided logits, and the guided log-probabilities.
    """
    # 1. Create the unconditional copy without modifying the original input x
    x_uncond = x.clone()
    x_uncond[:, :prompt_length] = mask_id

    # 2. Stack conditional (x) and unconditional (x_uncond) along the batch dimension
    # Shape: (2 * B, L)
    combined_ids = torch.cat([x, x_uncond], dim=0)

    # 3. Run the mask predictor once on the stacked batch
    # Shape: (2 * B, L, V)
    combined_logits = tiny_mask_predictor(combined_ids, embedding, readout)

    # 4. Split back into conditional and unconditional logits of shape (B, L, V)
    z_c, z_u = combined_logits.chunk(2, dim=0)

    # 5. Compute guided logits and normalized log-probabilities
    guided_logits = z_u + (1.0 + w) * (z_c - z_u)
    guided_log_probs = F.log_softmax(guided_logits, dim=-1)

    return x_uncond, guided_logits, guided_log_probs
