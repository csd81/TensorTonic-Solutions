import torch


def distillation_loss(
    student_query: torch.Tensor,
    student_positive: torch.Tensor,
    student_negatives: torch.Tensor,
    teacher_query: torch.Tensor,
    teacher_positive: torch.Tensor,
    teacher_negatives: torch.Tensor,
) -> torch.Tensor:
    """Returns the truncated embedding distillation loss as a 0-dimensional tensor."""

    def truncate_and_normalize(teacher_tensor: torch.Tensor, d: int) -> torch.Tensor:
        # Keep the first d coordinates
        truncated = teacher_tensor[..., :d]
        # Rescale to unit length (L2 norm)
        norm = torch.norm(truncated, p=2, dim=-1, keepdim=True)
        return truncated / norm

    # Infer student width d from student_query
    d = student_query.shape[-1]

    # Process each pair (query, positive, negatives)
    q_target = truncate_and_normalize(teacher_query, d)
    p_target = truncate_and_normalize(teacher_positive, d)
    n_target = truncate_and_normalize(teacher_negatives, d)

    # Calculate squared Euclidean distances (L2 norm squared)
    loss_q = torch.mean(torch.sum((student_query - q_target) ** 2, dim=-1))
    loss_p = torch.mean(torch.sum((student_positive - p_target) ** 2, dim=-1))

    if student_negatives.numel() > 0 and student_negatives.shape[1] > 0:
        # Average squared distance over the H negatives of each example, then over the batch
        loss_n = torch.mean(torch.sum((student_negatives - n_target) ** 2, dim=-1))
    else:
        loss_n = student_query.new_tensor(0.0)

    # Total unweighted sum of the three terms
    total_loss = loss_q + loss_p + loss_n
    return total_loss
