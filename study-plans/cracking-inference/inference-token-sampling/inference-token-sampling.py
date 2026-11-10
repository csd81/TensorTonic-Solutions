import torch

def sample_next_token(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
    uniform_draws: torch.Tensor,
) -> torch.Tensor:
    batch, vocab = logits.shape
    if temperature == 0.0:
        return torch.argmax(logits, dim=-1)
    
    scaled = logits / temperature
    if top_k > 0 and top_k < vocab:
        topk_idx = torch.argsort(scaled, dim=-1, descending=True, stable=True)[..., :top_k]
        keep_mask = torch.zeros_like(scaled, dtype=torch.bool)
        keep_mask.scatter_(-1, topk_idx, True)
        scaled = scaled.masked_fill(~keep_mask, float("-inf"))
        
    probs = torch.softmax(scaled, dim=-1)
    if top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1, stable=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        keep_sorted = (cumulative - sorted_probs) < top_p
        keep_sorted[..., 0] = True
        keep_mask = torch.zeros_like(probs, dtype=torch.bool)
        keep_mask.scatter_(-1, sorted_idx, keep_sorted)
        probs = probs.masked_fill(~keep_mask, 0.0)
        
    probs = probs / probs.sum(dim=-1, keepdim=True)
    cumulative_natural = torch.cumsum(probs, dim=-1)
    draws = uniform_draws.unsqueeze(-1)
    sampled = torch.searchsorted(cumulative_natural, draws, right=True).squeeze(-1)
    return torch.clamp(sampled, max=vocab - 1)
