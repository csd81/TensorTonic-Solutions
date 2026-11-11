import math

def gpu_occupancy(
    threads_per_block: int, registers_per_thread: int, shared_mem_per_block: int,
    max_threads_per_sm: int, max_warps_per_sm: int, max_blocks_per_sm: int,
    max_registers_per_sm: int, max_shared_mem_per_sm: int, warp_size: int = 32,
) -> dict:
    """
    Returns a dict: blocks_per_sm (int), resident_warps (int), occupancy (float).
    """
    # 1. Calculate allocated warps per block (w)
    w = (threads_per_block + warp_size - 1) // warp_size
    
    # 2. Calculate effective threads and register usage per block
    effective_threads_per_block = w * warp_size
    registers_per_block = registers_per_thread * effective_threads_per_block
    
    # 3. Determine the maximum blocks allowed by each individual resource constraint
    limits = []
    
    # Thread limit
    limits.append(max_threads_per_sm // threads_per_block)
    
    # Warp limit
    limits.append(max_warps_per_sm // w)
    
    # Block absolute limit
    limits.append(max_blocks_per_sm)
    
    # Register limit (only if registers are used)
    if registers_per_thread > 0:
        limits.append(max_registers_per_sm // registers_per_block)
        
    # Shared memory limit (only if shared memory is used)
    if shared_mem_per_block > 0:
        limits.append(max_shared_mem_per_sm // shared_mem_per_block)
        
    # 4. The actual block count is the tightest (minimum) of all limits
    blocks_per_sm = max(0, min(limits))
    
    # 5. Derive final metrics
    if blocks_per_sm == 0:
        return {
            "blocks_per_sm": 0,
            "resident_warps": 0,
            "occupancy": 0.0
        }
        
    resident_warps = blocks_per_sm * w
    occupancy = resident_warps / max_warps_per_sm
    
    return {
        "blocks_per_sm": int(blocks_per_sm),
        "resident_warps": int(resident_warps),
        "occupancy": float(occupancy)
    }
