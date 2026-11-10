import torch

def simulate_continuous_batching(
    arrival_steps: torch.Tensor,
    required_tokens: torch.Tensor,
    max_active_sequences: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (active_schedule, completion_step): a bool (S, N) tensor and an integer (N,) tensor.
    """
    N = arrival_steps.shape[0]
    
    # Track the state of each request
    num_tokens_generated = torch.zeros(N, dtype=torch.int64)
    completion_steps = torch.zeros(N, dtype=torch.int64)
    
    # Status: 0 = waiting, 1 = active, 2 = completed
    status = torch.zeros(N, dtype=torch.int8)
    
    active_requests = []
    active_history = []
    
    step = 0
    while True:
        # 1. Release slots from requests that finished in the previous step
        active_requests = [r for r in active_requests if status[r] == 1]
        
        # 2. Admit newly arrived waiting requests if slots are available
        if len(active_requests) < max_active_sequences:
            # Find requests that are waiting and have already arrived
            waiting_and_arrived = [
                i for i in range(N) 
                if status[i] == 0 and arrival_steps[i] <= step
            ]
            
            if waiting_and_arrived:
                # Sort by arrival step first, then by original request order (index)
                waiting_and_arrived.sort(key=lambda idx: (arrival_steps[idx].item(), idx))
                
                # Admit up to capacity
                available_slots = max_active_sequences - len(active_requests)
                to_admit = waiting_and_arrived[:available_slots]
                
                for r in to_admit:
                    status[r] = 1
                    active_requests.append(r)
                    
        # 3. Check termination: Stop if all requests are completed
        if len(active_requests) == 0 and (status == 2).all():
            break
            
        # 4. Record active schedule for the current step
        row = torch.zeros(N, dtype=torch.bool)
        if active_requests:
            row[active_requests] = True
        active_history.append(row)
        
        # 5. Process token generation for all active requests
        for r in active_requests:
            num_tokens_generated[r] += 1
            if num_tokens_generated[r] == required_tokens[r]:
                status[r] = 2
                completion_steps[r] = step
                
        step += 1
        
    # Stack history rows into a single (S, N) tensor
    if active_history:
        active_schedule = torch.stack(active_history)
    else:
        active_schedule = torch.empty((0, N), dtype=torch.bool)
        
    return active_schedule, completion_steps
