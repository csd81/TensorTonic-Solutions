import torch

def transfer_schedule(masked_count: int, steps: int) -> torch.Tensor:
    schedule = torch.full((steps,), masked_count // steps, dtype=torch.long)
    schedule[: masked_count % steps] += 1
    return schedule

def low_confidence_step(x: torch.Tensor, response_logits: torch.Tensor, window: torch.Tensor, count: int, mask_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    prompt_length = x.shape[1] - response_logits.shape[1]
    response = x[:, prompt_length:]
    probabilities = torch.softmax(response_logits, dim=-1)
    confidence, prediction = probabilities.max(dim=-1)
    candidates = (response == mask_id) & window
    confidence = torch.where(candidates, confidence, torch.full_like(confidence, -torch.inf))
    chosen = torch.topk(confidence, k=count, dim=-1).indices
    revealed = torch.zeros_like(candidates).scatter(1, chosen, True)
    response = torch.where(revealed, prediction, response)
    return torch.cat([x[:, :prompt_length], response], dim=1), revealed

def llada_semi_autoregressive_remasking(
    prompt: torch.Tensor, 
    gen_length: int, 
    block_length: int, 
    steps: int, 
    step_logits: torch.Tensor, 
    mask_id: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns the final sequence, the global step at which each response position was revealed, and the per-block transfer schedules.
    """
    B, _ = prompt.shape
    device = prompt.device

    num_blocks = gen_length // block_length
    steps_per_block = steps // num_blocks

    # Initialize working sequence x of shape (B, P + L) without modifying input tensors
    masked_response = torch.full((B, gen_length), mask_id, dtype=prompt.dtype, device=device)
    x = torch.cat([prompt.clone(), masked_response], dim=1)

    # Track global reveal steps and per-block schedules
    reveal_steps = torch.full((B, gen_length), -1, dtype=torch.long, device=device)
    schedules = torch.empty((num_blocks, steps_per_block), dtype=torch.long, device=device)

    for j in range(num_blocks):
        # Compute transfer schedule for block j
        block_schedule = transfer_schedule(block_length, steps_per_block).to(device)
        schedules[j] = block_schedule

        # Boolean window masking only the response positions inside block j
        window = torch.zeros((B, gen_length), dtype=torch.bool, device=device)
        start_pos = j * block_length
        end_pos = (j + 1) * block_length
        window[:, start_pos:end_pos] = True

        for i in range(steps_per_block):
            g = j * steps_per_block + i
            count = int(block_schedule[i].item())
            
            # Run one low-confidence remasking step on the active block window
            x, revealed = low_confidence_step(
                x=x,
                response_logits=step_logits[g],
                window=window,
                count=count,
                mask_id=mask_id
            )
            
            # Record global step g for newly revealed tokens
            reveal_steps = torch.where(revealed, g, reveal_steps)

    return x, reveal_steps, schedules
