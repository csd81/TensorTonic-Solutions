import torch

def schedule_dynamic_batches(
    arrival_times: torch.Tensor,
    max_batch_size: int,
    max_queue_delay: float,
) -> tuple:
    """
    Groups arriving requests into batches based on max_batch_size or max_queue_delay.
    
    Returns a tuple containing:
        - batch_assignment: torch.Tensor (int64) of batch IDs for each request.
        - batch_dispatch_time: torch.Tensor (float32) of dispatch times for each request.
    """
    N = arrival_times.size(0)
    
    # Pair each arrival time with its original input index and sort
    indexed_arrivals = [(float(arrival_times[i]), i) for i in range(N)]
    indexed_arrivals.sort(key=lambda x: (x[0], x[1]))
    
    batch_assignment = [0] * N
    batch_dispatch_time = [0.0] * N
    
    curr_batch_idx = 0
    curr_batch = []
    curr_deadline = None
    
    for arr_time, orig_idx in indexed_arrivals:
        if not curr_batch:
            # Start a new batch
            curr_batch.append((arr_time, orig_idx))
            curr_deadline = arr_time + max_queue_delay
            
            if len(curr_batch) == max_batch_size:
                for _, o_idx in curr_batch:
                    batch_assignment[o_idx] = curr_batch_idx
                    batch_dispatch_time[o_idx] = arr_time
                curr_batch = []
                curr_deadline = None
                curr_batch_idx += 1
        else:
            if arr_time >= curr_deadline:
                # The current batch expires at its deadline
                for _, o_idx in curr_batch:
                    batch_assignment[o_idx] = curr_batch_idx
                    batch_dispatch_time[o_idx] = curr_deadline
                curr_batch_idx += 1
                
                # Start the next batch with the current request
                curr_batch = [(arr_time, orig_idx)]
                curr_deadline = arr_time + max_queue_delay
                
                if len(curr_batch) == max_batch_size:
                    for _, o_idx in curr_batch:
                        batch_assignment[o_idx] = curr_batch_idx
                        batch_dispatch_time[o_idx] = arr_time
                    curr_batch = []
                    curr_deadline = None
                    curr_batch_idx += 1
            else:
                # Add to the current forming batch
                curr_batch.append((arr_time, orig_idx))
                
                if len(curr_batch) == max_batch_size:
                    for _, o_idx in curr_batch:
                        batch_assignment[o_idx] = curr_batch_idx
                        batch_dispatch_time[o_idx] = arr_time
                    curr_batch = []
                    curr_deadline = None
                    curr_batch_idx += 1
                    
    # Dispatch any remaining underfilled batch at its deadline
    if curr_batch:
        for _, o_idx in curr_batch:
            batch_assignment[o_idx] = curr_batch_idx
            batch_dispatch_time[o_idx] = curr_deadline
            
    return (
        torch.tensor(batch_assignment, dtype=torch.long), 
        torch.tensor(batch_dispatch_time, dtype=torch.float32)
    )
