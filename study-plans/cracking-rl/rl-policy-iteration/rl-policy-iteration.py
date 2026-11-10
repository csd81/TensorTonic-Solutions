def policy_iteration(
    P: list, R: list, gamma: float, eval_tol: float = 1e-8, max_iters: int = 200
) -> tuple:
    """
    Finds the optimal policy and state-value function using Policy Iteration.
    
    Args:
        P: Nest list of shape (S, A, S) representing transition probabilities.
        R: Nested list of shape (S, A, S) representing transition rewards.
        gamma: Discount factor (0 <= gamma < 1).
        eval_tol: Convergence tolerance for synchronous policy evaluation.
        max_iters: Maximum number of policy improvement steps.
        
    Returns:
        A tuple (V, policy) containing:
            - V: List of state values rounded to 4 decimal places.
            - policy: List of optimal integer actions per state.
    """
    num_states = len(P)
    num_actions = len(P[0])
    
    # Initialize: every state chooses action 0 and has a value of 0.0
    V = [0.0] * num_states
    policy = [0] * num_states
    
    for _ in range(max_iters):
        # 1. Synchronous Policy Evaluation
        while True:
            delta = 0.0
            V_new = [0.0] * num_states
            for s in range(num_states):
                a = policy[s]
                expected_value = 0.0
                for s_next in range(num_states):
                    prob = P[s][a][s_next]
                    reward = R[s][a][s_next]
                    expected_value += prob * (reward + gamma * V[s_next])
                V_new[s] = expected_value
                delta = max(delta, abs(V_new[s] - V[s]))
            V = V_new
            if delta < eval_tol:
                break
        
        # 2. Policy Improvement
        policy_stable = True
        new_policy = list(policy)
        
        for s in range(num_states):
            # Compute Q(s, a) for all actions
            q_values = [0.0] * num_actions
            for a in range(num_actions):
                q_sa = 0.0
                for s_next in range(num_states):
                    prob = P[s][a][s_next]
                    reward = R[s][a][s_next]
                    q_sa += prob * (reward + gamma * V[s_next])
                q_values[a] = q_sa
            
            # Select the optimal action handling the 1e-12 tolerance tie-breaking
            best_a = 0
            for a in range(1, num_actions):
                # An action strictly improves if it outperforms the current best by > 1e-12.
                # Otherwise, it's a tie, and we naturally keep the lower index 'best_a'.
                if q_values[a] > q_values[best_a] + 1e-12:
                    best_a = a
            
            if best_a != policy[s]:
                policy_stable = False
            new_policy[s] = best_a
            
        if policy_stable:
            break
            
        policy = new_policy
        
    # Round the values to 4 decimal places before returning
    V_rounded = [round(v, 4) for v in V]
    return (V_rounded, policy)
