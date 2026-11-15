import torch
import torch.nn.functional as F

def contrastive_loss(
    query: torch.Tensor,
    positive: torch.Tensor,
    hard_negatives: torch.Tensor,
    query_ids: torch.Tensor,
    positive_ids: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """
    Returns the contrastive loss with hard, in-batch and same-tower negatives as a 0-dimensional tensor.
    """
    # 1. L2-normalize all embeddings along the feature dimension
    q = F.normalize(query, p=2, dim=-1)                # (B, d)
    p_pos = F.normalize(positive, p=2, dim=-1)         # (B, d)
    p_neg = F.normalize(hard_negatives, p=2, dim=-1)   # (B, H, d)

    # 2. Compute scaled cosine similarity logits
    # Positive pair logits: (B, 1)
    pos_logits = (q * p_pos).sum(dim=-1, keepdim=True) / temperature

    # Hard negative logits: (B, H) (works naturally even when H == 0)
    hn_logits = torch.einsum("bd,bhd->bh", q, p_neg) / temperature

    # In-batch target logits: (B, B)
    ib_logits = (q @ p_pos.T) / temperature

    # Same-tower logits: query-query and positive-positive, each (B, B)
    qq_logits = (q @ q.T) / temperature
    pp_logits = (p_pos @ p_pos.T) / temperature

    # 3. Build duplicate mask: True where (i, j) is a duplicate (same query_id OR same positive_id)
    same_q = query_ids.unsqueeze(1) == query_ids.unsqueeze(0)
    same_p = positive_ids.unsqueeze(1) == positive_ids.unsqueeze(0)
    dup_mask = same_q | same_p  # (B, B), diagonal is guaranteed True

    # Mask out duplicates with -inf so exp(-inf) == 0 in logsumexp
    neg_inf = torch.tensor(float("-inf"), dtype=q.dtype, device=q.device)
    ib_logits = ib_logits.masked_fill(dup_mask, neg_inf)
    qq_logits = qq_logits.masked_fill(dup_mask, neg_inf)
    pp_logits = pp_logits.masked_fill(dup_mask, neg_inf)

    # 4. Concatenate all candidate logits per query: (B, 1 + H + 3B)
    all_logits = torch.cat(
        [pos_logits, hn_logits, ib_logits, qq_logits, pp_logits], dim=-1
    )

    # 5. Compute -log(Z^P / (Z^P + Z^HN + Z^IB + Z^STN)) = logsumexp(all_logits) - pos_logits
    per_sample_loss = torch.logsumexp(all_logits, dim=-1) - pos_logits.squeeze(-1)

    return per_sample_loss.mean()
