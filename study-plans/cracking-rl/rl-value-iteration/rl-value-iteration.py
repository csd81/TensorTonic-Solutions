def value_iteration(P: list, R: list, gamma: float, tol: float = 1e-6, max_iters: int = 1000) -> tuple:
    """
    Returns (values, policy): rounded float values and integer actions, one per state.
    """
    num_states = len(P)
    num_actions = len(P[0])
    
    # Initialize state values to zero
    V = [0.0] * num_states
    
    # Value Iteration Loop
    for _ in range(max_iters):
        V_new = [0.0] * num_states
        max_diff = 0.0
        
        for s in range(num_states):
            max_q = float('-inf')
            for a in range(num_actions):
                q_sa = 0.0
                for s_next in range(num_states):
                    q_sa += P[s][a][s_next] * (R[s][a][s_next] + gamma * V[s_next])
                if q_sa > max_q:
                    max_q = q_sa
            
            V_new[s] = max_q
            diff = abs(V_new[s] - V[s])
            if diff > max_diff:
                max_diff = diff
                
        V = V_new
        if max_diff < tol:
            break
            
    # Policy Extraction Loop
    policy = []
    for s in range(num_states):
        q_values = []
        for a in range(num_actions):
            q_sa = 0.0
            for s_next in range(num_states):
                q_sa += P[s][a][s_next] * (R[s][a][s_next] + gamma * V[s_next])
            q_values.append(q_sa)
            
        best_q = max(q_values)
        
        # Select the lowest action index on ties (within 1e-12 precision)
        best_action = 0
        for a in range(num_actions):
            if best_q - q_values[a] <= 1e-12:
                best_action = a
                break
        policy.append(best_action)
        
    # Round state-value floats to four decimals
    V_rounded = [round(val, 4) for val in V]
    
    return (V_rounded, policy)
