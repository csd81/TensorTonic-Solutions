import torch

def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    return logits / temperature

def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    indices = torch.argsort(logits, descending=True, stable=True)[:k]
    filtered = torch.full_like(logits, float("-inf"))
    filtered[indices] = logits[indices]
    return filtered

def sample_from_logits(logits: torch.Tensor, random_value: float) -> int:
    cumulative = torch.softmax(logits, dim=-1).cumsum(dim=-1)
    return int(
        torch.searchsorted(
            cumulative,
            torch.tensor(random_value, dtype=logits.dtype, device=logits.device),
        )
        .clamp(max=logits.numel() - 1)
        .item()
    )
