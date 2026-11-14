import torch

def llada_low_confidence_remasking(
    prompt: torch.Tensor, 
    gen_length: int, 
    steps: int, 
    step_logits: torch.Tensor, 
    mask_id: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns the final sequence, the step at which each response position was revealed, and the transfer schedule.
    """
    B, P = prompt.shape
    device = prompt.device
    
    # 1. Compute the transfer schedule: c_k = floor(L / N) + 1[k < (L mod N)]
    base_count = gen_length // steps
    remainder = gen_length % steps
    schedule = torch.full((steps,), base_count, dtype=torch.long, device=device)
    if remainder > 0:
        schedule[:remainder] += 1
        
    # 2. Initialize the working sequence x of shape (B, P + L) and reveal steps tracker
    masked_resp = torch.full((B, gen_length), mask_id, dtype=prompt.dtype, device=device)
    x = torch.cat([prompt.clone(), masked_resp], dim=1)
    reveal_steps = torch.full((B, gen_length), -1, dtype=torch.long, device=device)
    
    # 3. Run reverse diffusion steps
    for k in range(steps):
        c_k = int(schedule[k].item())
        if c_k == 0:
            continue
            
        logits_k = step_logits[k]  # Shape: (B, L, V)
        
        # Greedy token predictions and their softmax confidences
        probs = torch.softmax(logits_k, dim=-1)
        confidences, predictions = torch.max(probs, dim=-1)  # Both shape: (B, L)
        
        # Mask out positions that have already been revealed
        is_masked = (x[:, P:] == mask_id)
        confidences = torch.where(is_masked, confidences, torch.tensor(-float('inf'), dtype=confidences.dtype, device=device))
        
        # Select top c_k most confident masked positions per batch row
        _, topk_indices = torch.topk(confidences, k=c_k, dim=-1)  # Shape: (B, c_k)
        
        # Gather the predictions for the chosen positions and update x and reveal_steps
        chosen_tokens = torch.gather(predictions, dim=1, index=topk_indices)
        x[:, P:].scatter_(dim=1, index=topk_indices, src=chosen_tokens)
        reveal_steps.scatter_(dim=1, index=topk_indices, value=k)
        
    return x, reveal_steps, schedule
