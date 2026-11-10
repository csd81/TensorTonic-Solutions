def replay_buffer_sample(capacity: int, transitions: list, sample_indices: list) -> tuple:
    """
    Returns (states, actions, rewards, next_states, dones), each a list in sample order.
    """
    # Initialize the buffer with None up to its capacity
    buffer = [None] * capacity
    
    # Store transitions in a circular buffer manner
    for i, t in enumerate(transitions):
        buffer[i % capacity] = t
        
    # Prepare the output lists
    states = []
    actions = []
    rewards = []
    next_states = []
    dones = []
    
    # Retrieve transitions based on the requested physical slot indices
    for idx in sample_indices:
        t = buffer[idx]
        states.append(t[0])
        actions.append(t[1])
        rewards.append(float(round(t[2], 4)))
        next_states.append(t[3])
        dones.append(t[4])
        
    return (states, actions, rewards, next_states, dones)
