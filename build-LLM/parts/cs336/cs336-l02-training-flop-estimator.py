def flop_estimator(matmuls: list[list[int]], attention_flops: int = 0) -> dict:
    """
    Returns integer forward_flops, backward_flops, and total_flops in a dictionary.
    """
    # Sum up the FLOPs for all given matrix multiplications (2 * B * D * K)
    matmul_flops = sum(2 * B * D * K for B, D, K in matmuls)
    
    # Calculate forward, backward, and total FLOPs using exact integer arithmetic
    forward_flops = matmul_flops + attention_flops
    backward_flops = 2 * forward_flops
    total_flops = forward_flops + backward_flops
    
    return {
        "forward_flops": forward_flops,
        "backward_flops": backward_flops,
        "total_flops": total_flops
    }
