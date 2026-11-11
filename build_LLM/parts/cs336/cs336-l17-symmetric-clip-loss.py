import torch
import torch.nn.functional as F

def symmetric_clip_loss(
    image_embeddings: torch.Tensor, text_embeddings: torch.Tensor, temperature: int | float,
) -> dict:
    """
    Returns a dict: loss and two directional accuracies (scalars), similarities (tensor).
    """
    dtype = image_embeddings.dtype
    device = image_embeddings.device
    batch_size = image_embeddings.shape[0]

    eps = torch.tensor(1e-12, dtype=dtype, device=device)
    temp = torch.tensor(temperature, dtype=dtype, device=device)

    img_norm = torch.linalg.vector_norm(image_embeddings, dim=1, keepdim=True).clamp(min=eps)
    txt_norm = torch.linalg.vector_norm(text_embeddings, dim=1, keepdim=True).clamp(min=eps)

    v_hat = image_embeddings / img_norm
    t_hat = text_embeddings / txt_norm

    similarities = (v_hat @ t_hat.transpose(0, 1)) / temp

    targets = torch.arange(batch_size, dtype=torch.int64, device=device)
    loss_i2t = F.cross_entropy(similarities, targets)
    loss_t2i = F.cross_entropy(similarities.transpose(0, 1), targets)
    loss = 0.5 * (loss_i2t + loss_t2i)

    i2t_preds = similarities.argmax(dim=1)
    t2i_preds = similarities.argmax(dim=0)

    image_to_text_accuracy = (i2t_preds == targets).to(dtype).mean()
    text_to_image_accuracy = (t2i_preds == targets).to(dtype).mean()

    return {
        "loss": loss.to(dtype),
        "similarities": similarities,
        "image_to_text_accuracy": image_to_text_accuracy,
        "text_to_image_accuracy": text_to_image_accuracy,
    }
