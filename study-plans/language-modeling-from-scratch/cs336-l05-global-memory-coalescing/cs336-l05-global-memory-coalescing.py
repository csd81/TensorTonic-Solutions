import numpy as np

def coalescing_analysis(byte_addresses: np.ndarray, access_width_bytes: int = 4, cache_line_bytes: int = 128) -> dict:
    """
    Returns a dict: line_ids (int64 array), transaction_count (int), useful_byte_fraction (float).
    """
    # 1. Track all unique bytes requested across all lanes to compute U
    unique_bytes = set()
    
    # 2. Collect all cache line IDs touched by any lane
    touched_lines = set()
    
    for addr in byte_addresses:
        # Every lane requests access_width_bytes starting from addr
        # Add requested byte offsets to the global unique bytes set
        unique_bytes.update(range(addr, addr + access_width_bytes))
        
        # Calculate start and end cache line IDs for this request
        start_line = addr // cache_line_bytes
        end_line = (addr + access_width_bytes - 1) // cache_line_bytes
        
        # Add all line IDs from start_line up to and including end_line
        touched_lines.update(range(start_line, end_line + 1))
        
    # 3. Create the sorted NumPy array of unique line IDs
    line_ids = np.array(sorted(touched_lines), dtype=np.int64)
    
    # 4. Count total transactions (C)
    transaction_count = len(line_ids)
    
    # 5. Compute the useful byte fraction: U / (C * L)
    U = len(unique_bytes)
    total_fetched_bytes = transaction_count * cache_line_bytes
    useful_byte_fraction = float(U / total_fetched_bytes) if total_fetched_bytes > 0 else 0.0
    
    return {
        "line_ids": line_ids,
        "transaction_count": transaction_count,
        "useful_byte_fraction": useful_byte_fraction
    }
