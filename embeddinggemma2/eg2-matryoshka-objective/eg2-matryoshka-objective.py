import torch

def contrastive_loss(query: torch.Tensor, positive: torch.Tensor, hard_negatives: torch.Tensor, query_ids: torch.Tensor, positive_ids: torch.Tensor, temperature: float) -> torch.Tensor:
    q = query / torch.linalg.vector_norm(query, dim=-1, keepdim=True)
    p = positive / torch.linalg.vector_norm(positive, dim=-1, keepdim=True)
    n = hard_negatives / torch.linalg.vector_norm(hard_negatives, dim=-1, keepdim=True)
    positive_logits = (q * p).sum(dim=-1) / temperature
    hard_logits = (q.unsqueeze(1) * n).sum(dim=-1) / temperature
    keep = (query_ids.unsqueeze(1) != query_ids.unsqueeze(0)) & (positive_ids.unsqueeze(1) != positive_ids.unsqueeze(0))
    in_batch = (q @ p.T / temperature).masked_fill(~keep, float("-inf"))
    query_tower = (q @ q.T / temperature).masked_fill(~keep, float("-inf"))
    positive_tower = (p @ p.T / temperature).masked_fill(~keep, float("-inf"))
    logits = torch.cat([positive_logits.unsqueeze(1), hard_logits, in_batch, query_tower, positive_tower], dim=1)
    return (torch.logsumexp(logits, dim=1) - positive_logits).mean()

def spread_out_loss(query: torch.Tensor, positive: torch.Tensor) -> torch.Tensor:
    batch = query.shape[0]
    pairs = batch * (batch - 1)
    off_diagonal = ~torch.eye(batch, dtype=torch.bool, device=query.device)
    query_gram = query @ query.T
    positive_gram = positive @ positive.T
    query_term = query_gram[off_diagonal].square().sum() / pairs
    positive_term = positive_gram[off_diagonal].square().sum() / pairs
    return query_term + positive_term

def truncate_teacher(teacher: torch.Tensor, width: int) -> torch.Tensor:
    head = teacher[..., :width]
    return head / torch.linalg.vector_norm(head, dim=-1, keepdim=True)

def distillation_loss(student_query: torch.Tensor, student_positive: torch.Tensor, student_negatives: torch.Tensor, teacher_query: torch.Tensor, teacher_positive: torch.Tensor, teacher_negatives: torch.Tensor) -> torch.Tensor:
    width = student_query.shape[-1]
    query_term = (student_query - truncate_teacher(teacher_query, width)).square().sum(dim=-1).mean()
    positive_term = (student_positive - truncate_teacher(teacher_positive, width)).square().sum(dim=-1).mean()
    if student_negatives.shape[1] == 0:
        negative_term = student_query.new_zeros(())
    else:
        negative_distances = (student_negatives - truncate_teacher(teacher_negatives, width)).square().sum(dim=-1)
        negative_term = negative_distances.mean(dim=1).mean()
    return query_term + positive_term + negative_term

def matryoshka_objective(
    student_query: torch.Tensor, 
    student_positive: torch.Tensor, 
    student_negatives: torch.Tensor, 
    teacher_query: torch.Tensor, 
    teacher_positive: torch.Tensor, 
    teacher_negatives: torch.Tensor, 
    query_ids: torch.Tensor, 
    positive_ids: torch.Tensor, 
    temperature: float, 
    dims: list[int], 
    weights: list[float]
) -> tuple[torch.Tensor, torch.Tensor]:
    w_C, w_S, w_D = weights
    per_width_losses = []

    for m in dims:
        # Slice student prefixes and re-normalize to unit length
        sq_m = student_query[..., :m]
        sq_m = sq_m / torch.linalg.vector_norm(sq_m, dim=-1, keepdim=True)

        sp_m = student_positive[..., :m]
        sp_m = sp_m / torch.linalg.vector_norm(sp_m, dim=-1, keepdim=True)

        if student_negatives.numel() > 0 and student_negatives.shape[1] > 0:
            sn_m = student_negatives[..., :m]
            sn_m = sn_m / torch.linalg.vector_norm(sn_m, dim=-1, keepdim=True)
        else:
            sn_m = student_negatives[..., :m]

        # Compute component losses at width m
        loss_c = contrastive_loss(sq_m, sp_m, sn_m, query_ids, positive_ids, temperature) if w_C > 0 else student_query.new_zeros(())
        loss_s = spread_out_loss(sq_m, sp_m) if w_S > 0 else student_query.new_zeros(())
        loss_d = distillation_loss(sq_m, sp_m, sn_m, teacher_query, teacher_positive, teacher_negatives) if w_D > 0 else student_query.new_zeros(())

        l_m = w_C * loss_c + w_S * loss_s + w_D * loss_d
        per_width_losses.append(l_m)

    losses_tensor = torch.stack(per_width_losses)
    total_loss = losses_tensor.sum()

    return total_loss, losses_tensor
