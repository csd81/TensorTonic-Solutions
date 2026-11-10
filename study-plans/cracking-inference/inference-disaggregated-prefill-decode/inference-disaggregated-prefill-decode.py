import torch

def simulate_disaggregated_serving(
    arrival_times: torch.Tensor,
    prompt_lengths: torch.Tensor,
    output_lengths: torch.Tensor,
    prefill_replica_rates: torch.Tensor,
    decode_replica_rates: torch.Tensor,
    kv_transfer_bandwidth: float,
    fixed_transfer_latency: float,
) -> tuple:
    """
    Returns (assigned_replicas, ttft, completion_time): integer (N, 2) assignments 
    and two floating (N,) absolute timestamp tensors, in seconds.
    """
    N = arrival_times.shape[0]
    
    # Track availability times for each replica (initialized to 0.0)
    prefill_avail = torch.zeros(prefill_replica_rates.shape[0], dtype=torch.float64)
    decode_avail = torch.zeros(decode_replica_rates.shape[0], dtype=torch.float64)
    
    # Output structures to be filled in original request order
    assigned_replicas = torch.zeros((N, 2), dtype=torch.long)
    ttft = torch.zeros(N, dtype=torch.float64)
    completion_time = torch.zeros(N, dtype=torch.float64)
    
    # --- PHASE 1: PREFILL SERVICE ---
    # Sorted by arrival time, breaking ties by original request index
    prefill_order = sorted(range(N), key=lambda idx: (float(arrival_times[idx]), idx))
    
    kv_ready_times = torch.zeros(N, dtype=torch.float64)
    
    for idx in prefill_order:
        arr_time = float(arrival_times[idx])
        p_len = int(prompt_lengths[idx])
        
        # Select prefill replica with minimum availability time (tie-break by lowest index)
        pref_rep_idx = int(torch.argmin(prefill_avail))
        assigned_replicas[idx, 0] = pref_rep_idx
        
        # Calculate timing
        start_prefill = max(arr_time, float(prefill_avail[pref_rep_idx]))
        duration_prefill = p_len / float(prefill_replica_rates[pref_rep_idx])
        end_prefill = start_prefill + duration_prefill
        
        # Update replica availability
        prefill_avail[pref_rep_idx] = end_prefill
        
        # Calculate KV-ready time
        kv_ready_times[idx] = end_prefill + (p_len / kv_transfer_bandwidth) + fixed_transfer_latency

    # --- PHASE 2: DECODE SERVICE ---
    # Sorted by KV-ready time, breaking ties by original request index
    decode_order = sorted(range(N), key=lambda idx: (float(kv_ready_times[idx]), idx))
    
    for idx in decode_order:
        kv_ready = float(kv_ready_times[idx])
        out_len = int(output_lengths[idx])
        
        # Select decode replica with minimum availability time (tie-break by lowest index)
        dec_rep_idx = int(torch.argmin(decode_avail))
        assigned_replicas[idx, 1] = dec_rep_idx
        
        # Calculate timing
        start_decode = max(kv_ready, float(decode_avail[dec_rep_idx]))
        dec_rate = float(decode_replica_rates[dec_rep_idx])
        
        # TTFT is the absolute timestamp when the first token is produced
        ttft[idx] = start_decode + (1.0 / dec_rate)
        
        # Completion time after serving all requested output tokens
        duration_decode = out_len / dec_rate
        end_decode = start_decode + duration_decode
        
        # Update replica availability
        decode_avail[dec_rep_idx] = end_decode
        completion_time[idx] = end_decode

    return assigned_replicas, ttft, completion_time
