import torch
import torch.nn.functional as F

def siglip_pairwise_loss(
    image_embeddings: torch.Tensor, text_embeddings: torch.Tensor,
    logit_scale: int | float, bias: int | float,
) -> dict:
    """
    Returns a dict: loss (scalar tensor), logits and pair_labels (floating tensors).
    """
    dtype = image_embeddings.dtype
    device = image_embeddings.device
    batch_size = image_embeddings.shape[0]

    eps = torch.tensor(1e-12, dtype=dtype, device=device)
    scale = torch.tensor(logit_scale, dtype=dtype, device=device)
    bias_t = torch.tensor(bias, dtype=dtype, device=device)

    img_norm = torch.linalg.vector_norm(image_embeddings, dim=1, keepdim=True).clamp(min=eps)
    txt_norm = torch.linalg.vector_norm(text_embeddings, dim=1, keepdim=True).clamp(min=eps)

    v_hat = image_embeddings / img_norm
    t_hat = text_embeddings / txt_norm

    logits = scale * (v_hat @ t_hat.transpose(0, 1)) + bias_t

    pair_labels = 2.0 * torch.eye(batch_size, dtype=dtype, device=device) - 1.0
    loss = -F.logsigmoid(pair_labels * logits).mean()

    return {
        "loss": loss.to(dtype),
        "logits": logits,
        "pair_labels": pair_labels,
    }
