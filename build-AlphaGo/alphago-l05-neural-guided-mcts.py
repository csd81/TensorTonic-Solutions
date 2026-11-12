import numpy as np

def neural_guided_mcts(
    root_state: str,
    transitions: dict,
    legal_actions: dict,
    players: dict,
    policy_logits: dict,
    value_predictions: dict,
    terminal_values: dict,
    simulations: int,
    cpuct: float,
    temperature: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Runs a deterministic neural-guided MCTS on a given two-player game tree.
    Returns a tuple of (visit_counts, mean_values, policy) for the root's legal actions.
    """
    # Helper to parse players consistently as integers
    def get_player(s):
        p = players[s]
        if isinstance(p, str):
            return int(p.strip(','))
        return int(p)

    # Initialize tree statistics: visit counts (N) and total accumulated values (W)
    # Keys will be (state, action) tuples
    N = {}
    W = {}

    # Track which non-terminal states have been expanded
    expanded = {root_state}

    # Precompute softmax priors for any expanded state to avoid repeated computation
    priors = {}
    def get_priors(s):
        if s not in priors:
            logits = np.array(policy_logits[s], dtype=np.float64)
            e_x = np.exp(logits - np.max(logits))
            priors[s] = e_x / e_x.sum()
        return priors[s]

    root_actions = legal_actions[root_state]

    for _ in range(simulations):
        state = root_state
        path = []  # To store the sequence of (state, action) taken

        # 1. Selection phase
        while state not in terminal_values and state in expanded:
            actions = legal_actions[state]
            state_priors = get_priors(state)

            # Compute total visits for the current state's children
            sum_n = sum(N.get((state, a), 0) for a in actions)
            sqrt_sum_n = np.sqrt(sum_n)

            best_score = -np.inf
            best_action = None

            for idx, a in enumerate(actions):
                n_sa = N.get((state, a), 0)
                p_sa = state_priors[idx]

                # Compute mean value Q(s, a) from the current state's player perspective
                q_sa = W.get((state, a), 0.0) / n_sa if n_sa > 0 else 0.0

                # PUCT formula
                score = q_sa + cpuct * p_sa * sqrt_sum_n / (1 + n_sa)

                # Tie-breaking: choose the lowest action on exact equality
                if score > best_score:
                    best_score = score
                    best_action = a

            path.append((state, best_action))
            
            # Move to next state (handling string/int conversion safely)
            if str(best_action) in transitions[state]:
                state = transitions[state][str(best_action)]
            else:
                state = transitions[state][best_action]

        # 2. Evaluation phase
        if state in terminal_values:
            leaf_value = float(str(terminal_values[state]).strip(','))
            leaf_player = get_player(state)
        else:
            leaf_value = float(str(value_predictions[state]).strip(','))
            leaf_player = get_player(state)
            expanded.add(state)  # Expand the nonterminal leaf node for future simulations

        # 3. Backup phase
        for s, a in path:
            N[(s, a)] = N.get((s, a), 0) + 1
            # Align the leaf value with the parent player's perspective
            parent_player = get_player(s)
            signed_value = leaf_value if parent_player == leaf_player else -leaf_value
            W[(s, a)] = W.get((s, a), 0.0) + signed_value

    # Prepare outputs for the root's legal actions
    root_N = np.array([N.get((root_state, a), 0) for a in root_actions], dtype=np.int64)
    root_Q = np.array([W.get((root_state, a), 0.0) / N[(root_state, a)] if N.get((root_state, a), 0) > 0 else 0.0 for a in root_actions], dtype=np.float64)

    # 4. Policy derivation based on temperature
    if temperature == 0.0:
        policy = np.zeros(len(root_actions), dtype=np.float64)
        max_idx = np.argmax(root_N)  # np.argmax automatically breaks ties with the lowest index
        policy[max_idx] = 1.0
    else:
        power_N = root_N ** (1.0 / temperature)
        policy = power_N / np.sum(power_N)

    return root_N, root_Q, policy
