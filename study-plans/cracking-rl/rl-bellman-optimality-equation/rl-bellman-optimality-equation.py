def bellman_optimality_backup(P: list, R: list, gamma: float, V: list) -> list:
    """
    Applies one synchronous Bellman optimality backup to the supplied state values.
    Returns a list of updated state values rounded to four decimal places.
    """
    num_states = len(P)
    num_actions = len(P[0])
    V_new = []
    
    for s in range(num_states):
        max_q_value = float('-inf')
        
        for a in range(num_actions):
            q_value = 0.0
            for s_prime in range(num_states):
                prob = P[s][a][s_prime]
                reward = R[s][a][s_prime]
                q_value += prob * (reward + gamma * V[s_prime])
                
            if q_value > max_q_value:
                max_q_value = q_value
                
        V_new.append(round(max_q_value, 4))
        
    return V_new
