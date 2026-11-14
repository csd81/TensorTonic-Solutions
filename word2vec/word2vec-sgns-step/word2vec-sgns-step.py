import torch


def sgns_sgd_step(
    W_in: torch.Tensor,
    W_out: torch.Tensor,
    center_id: int,
    pos_id: int,
    neg_ids: torch.Tensor,
    lr: float,
) -> dict:
    """
    Returns updated W_in and W_out float64 tensors in a dictionary.
    """
    # Clone pre-update embeddings so gradients use original values
    v_c = W_in[center_id].clone()  # (D,)
    u_o = W_out[pos_id].clone()  # (D,)
    u_negs = W_out[neg_ids].clone()  # (K, D)

    # Forward scores (dot products)
    s_o = torch.dot(u_o, v_c)  # scalar
    s_negs = torch.matmul(u_negs, v_c)  # (K,)

    # Gradients with respect to scores
    g_o = torch.sigmoid(s_o) - 1.0  # scalar
    g_negs = torch.sigmoid(s_negs)  # (K,)

    # Parameter gradients
    grad_v_c = g_o * u_o + torch.matmul(g_negs, u_negs)  # (D,)
    grad_u_o = g_o * v_c  # (D,)
    grad_u_negs = g_negs.unsqueeze(1) * v_c.unsqueeze(0)  # (K, D)

    # Apply SGD updates to copies of the embedding matrices
    W_in_new = W_in.clone()
    W_out_new = W_out.clone()

    W_in_new[center_id] -= lr * grad_v_c
    W_out_new[pos_id] -= lr * grad_u_o

    # index_add_ safely accumulates gradients when neg_ids has duplicate indices
    W_out_new.index_add_(0, neg_ids, -lr * grad_u_negs)

    return {"W_in": W_in_new, "W_out": W_out_new}
