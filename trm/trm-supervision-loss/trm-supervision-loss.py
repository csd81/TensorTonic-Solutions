import torch
import torch.nn.functional as F

def trm_supervision_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    halt_logits: torch.Tensor,
    loss_type: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    valid = labels != -100
    safe_labels = torch.where(valid, labels, torch.zeros_like(labels))

    if loss_type == "stablemax":
        mapped = torch.where(
            logits >= 0,
            logits + 1,
            1 / (1 - torch.clamp(logits, max=0)),
        )
        log_probs = torch.log(mapped / mapped.sum(dim=-1, keepdim=True))
    elif loss_type == "softmax":
        log_probs = F.log_softmax(logits, dim=-1)
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")

    picked = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    token_loss = torch.where(valid, -picked, torch.zeros_like(picked))
    answer_loss = (token_loss.sum(dim=-1) / valid.sum(dim=-1)).mean()

    with torch.no_grad():
        targets = ((logits.argmax(dim=-1) == labels) | ~valid).all(dim=-1)

    halt_loss = F.binary_cross_entropy_with_logits(
        halt_logits, targets.to(halt_logits.dtype)
    )
    total = answer_loss + 0.5 * halt_loss
    return total, answer_loss, halt_loss, targets
