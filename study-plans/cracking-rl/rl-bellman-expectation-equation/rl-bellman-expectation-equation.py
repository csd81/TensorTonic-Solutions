def bellman_expectation_backup(P: list, R: list, policy: list, gamma: float, V: list) -> list:
    """
    Returns one updated state value per state, rounded to four decimals.
    """
    num_states = len(V)
    num_actions = len(policy[0])
    V_new = []
    
    for s in range(num_states):
        state_value = 0.0
        for a in range(num_actions):
            action_prob = policy[s][a]
            
            # Skip computation if the action has 0 probability
            if action_prob == 0:
                continue
                
            action_value = 0.0
            for s_next in range(num_states):
                prob = P[s][a][s_next]
                reward = R[s][a][s_next]
                
                # Compute transition value: P(s'|s,a) * [R(s,a,s') + gamma * V(s')]
                action_value += prob * (reward + gamma * V[s_next])
                
            # Weight the action's expected value by the policy probability
            state_value += action_prob * action_value
            
        # Round the result to 4 decimal places as requested
        V_new.append(round(state_value, 4))
        
    return V_new
