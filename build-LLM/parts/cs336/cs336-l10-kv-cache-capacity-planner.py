def kv_cache_capacity_planner(
    model_parameters: int, bytes_per_parameter: int, num_layers: int,
    sequence_length: int, num_attention_heads: int, num_kv_heads: int,
    head_dim: int, bytes_per_element: int, memory_per_gpu: int,
    reserved_memory: int, memory_bandwidth: int,
) -> dict:
    # 1. Compute model memory consumption
    model_bytes = model_parameters * bytes_per_parameter
    
    # 2. Compute per-request KV cache memory consumption
    # Factor 2 accounts for separate key and value caches
    kv_bytes_per_request = 2 * num_layers * sequence_length * num_kv_heads * head_dim * bytes_per_element
    
    # 3. Calculate remaining available memory and determine maximum batch size
    available_memory = memory_per_gpu - reserved_memory - model_bytes
    if available_memory < 0:
        max_batch_size = 0
    else:
        max_batch_size = available_memory // kv_bytes_per_request
        
    # 4. Calculate step execution time (seconds per token)
    total_bytes_read = model_bytes + (max_batch_size * kv_bytes_per_request)
    seconds_per_token = total_bytes_read / memory_bandwidth
    
    # 5. Calculate token generation throughput (tokens per second)
    if max_batch_size == 0:
        tokens_per_second = 0.0
    else:
        tokens_per_second = max_batch_size / seconds_per_token

    return {
        "model_bytes": model_bytes,
        "kv_bytes_per_request": kv_bytes_per_request,
        "max_batch_size": max_batch_size,
        "seconds_per_token": seconds_per_token,
        "tokens_per_second": tokens_per_second
    }
