import torch

def assistant_only_sft_mask(
    input_ids: torch.Tensor, role_ids: torch.Tensor,
    attention_mask: torch.Tensor, assistant_role: int = 1,
) -> dict:
    """
    Returns a dict: labels (input token dtype), loss_mask (Boolean tensor).
    """
    batch_size, seq_len = input_ids.shape
    device = input_ids.device

    labels = torch.full(
        (batch_size, seq_len),
        fill_value=-100,
        dtype=input_ids.dtype,
        device=device,
    )
    loss_mask = torch.zeros(
        (batch_size, seq_len),
        dtype=torch.bool,
        device=device,
    )

    if seq_len > 1:
        valid_transitions = (
            attention_mask[:, :-1]
            & attention_mask[:, 1:]
            & (role_ids[:, 1:] == assistant_role)
        )
        loss_mask[:, :-1] = valid_transitions
        labels[:, :-1] = torch.where(
            valid_transitions,
            input_ids[:, 1:],
            labels[:, :-1],
        )

    return {
        "labels": labels,
        "loss_mask": loss_mask,
    }
