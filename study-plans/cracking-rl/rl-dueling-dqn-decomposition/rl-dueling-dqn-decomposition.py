def dueling_q_values(V: list, A_advantages: list) -> list:
    """
    Returns the Q-value matrix as a nested list, rounded to four decimals.
    """
    Q_values = []
    
    # Iterate through each sample in the batch
    for i in range(len(V)):
        v_i = V[i]
        a_row = A_advantages[i]
        
        # Calculate the mean advantage for the current sample's action row
        mean_a = sum(a_row) / len(a_row)
        
        # Apply the dueling DQN formula: Q = V + (A - mean(A))
        # and round each resulting Q-value to 4 decimal places
        q_row = [round(v_i + (a - mean_a), 4) for a in a_row]
        Q_values.append(q_row)
        
    return Q_values
