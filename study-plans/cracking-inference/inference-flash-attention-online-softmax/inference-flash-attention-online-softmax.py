import torch

def flash_attention_online_softmax(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_block_size: int,
    key_block_size: int,
    causal: bool = False,
) -> torch.Tensor:
    """
    Computes scaled dot-product attention using query-key tiles and online softmax
    without storing the full score matrix.
    
    Returns an attention tensor shaped (batch, query length, value width).
    """
    B, Sq, dk = query.shape
    _, Sk, dv = value.shape
    
    output = torch.zeros((B, Sq, dv), dtype=query.dtype, device=query.device)
    
    for b in range(B):
        # Tile along the query length (rows)
        for i in range(0, Sq, query_block_size):
            i_end = min(i + query_block_size, Sq)
            Q_block = query[b, i:i_end, :]  # Shape: (Br, dk)
            Br = Q_block.shape[0]
            
            # Initialize running online softmax statistics
            m = torch.full((Br, 1), float('-inf'), dtype=query.dtype, device=query.device)
            l = torch.zeros((Br, 1), dtype=query.dtype, device=query.device)
            a = torch.zeros((Br, dv), dtype=query.dtype, device=query.device)
            
            # Tile along the key/value length (columns)
            for j in range(0, Sk, key_block_size):
                j_end = min(j + key_block_size, Sk)
                K_block = key[b, j:j_end, :]    # Shape: (Bc, dk)
                V_block = value[b, j:j_end, :]  # Shape: (Bc, dv)
                
                # Compute raw scale dot-product scores for the current block
                S_block = (Q_block @ K_block.T) / (dk ** 0.5)  # Shape: (Br, Bc)
                
                # Apply causal mask if required using absolute sequence coordinates
                if causal:
                    q_idx = torch.arange(i, i_end, device=query.device).view(-1, 1)
                    k_idx = torch.arange(j, j_end, device=query.device).view(1, -1)
                    mask = q_idx < k_idx
                    S_block = S_block.masked_fill(mask, float('-inf'))
                
                # Find maximum score per row for the current block
                m_block = torch.max(S_block, dim=-1, keepdim=True).values
                m_new = torch.max(m, m_block)
                
                # Safeguard against inf - inf / NaN edge cases for fully-masked rows
                m_new_safe = torch.where(m_new == float('-inf'), 0.0, m_new)
                exp_m = torch.where(m == float('-inf'), 0.0, torch.exp(m - m_new_safe))
                exp_s = torch.where(S_block == float('-inf'), 0.0, torch.exp(S_block - m_new_safe))
                
                # Update running denominators and partial value accumulations
                l_new = exp_m * l + torch.sum(exp_s, dim=-1, keepdim=True)
                a_new = exp_m * a + exp_s @ V_block
                
                m = m_new
                l = l_new
                a = a_new
                
            # Normalize to find final local query block output
            out_block = torch.where(l == 0, 0.0, a / l)
            output[b, i:i_end, :] = out_block
            
    return output
