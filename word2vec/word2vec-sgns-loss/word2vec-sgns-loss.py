import torch
import torch.nn.functional as F


def sgns_loss(
    center_vec: torch.Tensor, pos_vec: torch.Tensor, neg_vecs: torch.Tensor
) -> torch.Tensor:
    """
    Returns the scalar float64 SGNS loss.
    """
    # Positive pair dot product: u_o^T v_c (scalar)
    pos_dot = torch.dot(center_vec, pos_vec)

    # Negative pairs dot products: U_neg @ v_c -> shape (K,)
    neg_dots = torch.matmul(neg_vecs, center_vec)

    # Numerically stable log-sigmoid loss
    pos_loss = -F.logsigmoid(pos_dot)
    neg_loss = -F.logsigmoid(-neg_dots).sum()

    return (pos_loss + neg_loss).to(dtype=torch.float64)
