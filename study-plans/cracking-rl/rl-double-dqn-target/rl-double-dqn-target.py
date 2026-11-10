def double_dqn_targets(Q_online_next: list, Q_target_next: list, rewards: list, dones: list, gamma: float) -> list:
    """
    Returns one Double DQN target per sample, rounded to four decimals.
    """
    targets = []
    
    for i in range(len(rewards)):
        if dones[i] == 1:
            # For terminal states, target is just the step reward
            target_val = rewards[i]
        else:
            # Action selection using the online network (lowest index on ties)
            q_online = Q_online_next[i]
            best_a = 0
            max_q = q_online[0]
            for a in range(1, len(q_online)):
                if q_online[a] > max_q:
                    max_q = q_online[a]
                    best_a = a
            
            # Action evaluation using the target network
            target_val = rewards[i] + gamma * Q_target_next[i][best_a]
            
        targets.append(round(float(target_val), 4))
        
    return targets
