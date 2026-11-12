def value_iteration_step(
    values: list, transitions: list, rewards: list, gamma: float
) -> list[float]:
  """Returns one updated floating-point value for every state."""
  new_values = []

  for s in range(len(values)):
    best_action_value = max(
        rewards[s][a]
        + gamma
        * sum(
            transitions[s][a][s_next] * values[s_next]
            for s_next in range(len(values))
        )
        for a in range(len(rewards[s]))
    )
    new_values.append(float(best_action_value))

  return new_values
