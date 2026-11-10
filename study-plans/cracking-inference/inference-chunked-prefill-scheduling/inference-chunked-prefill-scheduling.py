import torch

def schedule_chunked_prefill(
    arrival_steps: torch.Tensor,
    prompt_tokens: torch.Tensor,
    decode_tokens: torch.Tensor,
    max_tokens_per_iteration: int,
    prefill_chunk_size: int,
) -> torch.Tensor:
    """
    Simulates a shared token budget across prompt prefill and output decoding.
    
    Returns an int64 (E, 4) tensor: iteration, request index, phase (0 prefill, 1 decode), token count.
    """
    num_requests = arrival_steps.size(0)
    
    # Track remaining tokens and metadata using standard lists for quick iteration
    rem_prompt = prompt_tokens.clone().tolist()
    rem_decode = decode_tokens.clone().tolist()
    arrival = arrival_steps.tolist()
    
    # Store the iteration index when each request finishes its prefill phase
    prefill_completed_iteration = [None] * num_requests
    
    events = []
    iteration = 0
    
    while True:
        # Stop processing when every request has finished generating its decode tokens
        if all(rem == 0 for rem in rem_decode):
            break
            
        budget = max_tokens_per_iteration
        
        # 1. DECODE PHASE (Priority)
        # Served in ascending original request index order
        for i in range(num_requests):
            if budget <= 0:
                break
            
            # Eligible if arrived, prefill completed strictly BEFORE this iteration, and has remaining decodes
            if (iteration >= arrival[i] and 
                prefill_completed_iteration[i] is not None and 
                prefill_completed_iteration[i] < iteration and 
                rem_decode[i] > 0):
                
                rem_decode[i] -= 1
                budget -= 1
                events.append([iteration, i, 1, 1])
                
        # 2. PREFILL PHASE
        # Served by arrival iteration, breaking ties by original request index
        prefill_candidates = [
            i for i in range(num_requests) 
            if iteration >= arrival[i] and rem_prompt[i] > 0
        ]
        
        # Sort candidates: primary key = arrival step, secondary key = original index
        prefill_candidates.sort(key=lambda idx: (arrival[idx], idx))
        
        for i in prefill_candidates:
            if budget <= 0:
                break
                
            # Process a chunk bounded by remaining budget, max chunk size, and remaining prompt tokens
            chunk = min(budget, prefill_chunk_size, rem_prompt[i])
            if chunk > 0:
                rem_prompt[i] -= chunk
                budget -= chunk
                events.append([iteration, i, 0, chunk])
                
                # If prefill finishes, record the iteration (cannot decode until next iteration)
                if rem_prompt[i] == 0:
                    prefill_completed_iteration[i] = iteration
                    
        iteration += 1
        
    if not events:
        return torch.empty((0, 4), dtype=torch.int64)
        
    return torch.tensor(events, dtype=torch.int64)
