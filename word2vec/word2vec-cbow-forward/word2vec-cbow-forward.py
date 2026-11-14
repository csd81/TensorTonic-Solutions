import torch
import torch.nn.functional as F


def cbow_forward(
    context_ids: torch.Tensor,
    target_id: int,
    W_in: torch.Tensor,
    W_out: torch.Tensor,
) -> torch.Tensor:
    """
    Returns the scalar float64 CBOW cross-entropy loss.
    """
    # Gather context embeddings (M, D) and average across the M context words -> (D,)
    h = W_in[context_ids].mean(dim=0)

    # Compute logits for every word in the vocabulary -> (V,)
    logits = torch.matmul(W_out, h)

    # Compute negative log-softmax at target_id for numerical stability
    loss = -F.log_softmax(logits, dim=0)[target_id]

    return loss.to(dtype=torch.float64)
