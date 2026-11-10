def mc_policy_evaluation(episodes: list, gamma: float, num_states: int) -> list:
    """
    Returns first-visit state values rounded to four decimals; unvisited states are 0.0.
    """
    # Track returns accumulated and total first-visit counts for each state
    state_returns = [0.0] * num_states
    state_counts = [0] * num_states
    
    for episode in episodes:
        if not episode:
            continue
            
        T = len(episode)
        # Calculate returns (G) backward to optimize the process: G_t = r_t + gamma * G_{t+1}
        # Alternatively, compute rewards forward for each step.
        # Let's map out first-visit time steps in this episode
        visited_states = set()
        first_visits = {} # state -> time step t
        
        for t, (state, _) in enumerate(episode):
            if state not in visited_states:
                visited_states.add(state)
                first_visits[state] = t
                
        # Compute the return G_t for each first-visit step
        for state, t_first in first_visits.items():
            g_return = 0.0
            discount = 1.0
            
            # Sum up discounted rewards from t_first onwards
            for k in range(t_first, T):
                reward = episode[k][1]
                g_return += discount * reward
                discount *= gamma
                
            state_returns[state] += g_return
            state_counts[state] += 1

    # Calculate final average value per state
    v_estimates = [0.0] * num_states
    for s in range(num_states):
        if state_counts[s] > 0:
            v_estimates[s] = round(state_returns[s] / state_counts[s], 4)
        else:
            v_estimates[s] = 0.0
            
    return v_estimates
