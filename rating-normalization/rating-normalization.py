def rating_normalization(matrix: list) -> list:
  """
    Returns the mean-centered user-item matrix.
    """
  normalized_matrix = []

  for row in matrix:
    # Find all nonzero ratings for the current user
    nonzero_ratings = [r for r in row if r != 0]

    if not nonzero_ratings:
      # If a user has no ratings, all entries remain 0.0
      normalized_matrix.append([0.0 for _ in row])
    else:
      mean_rating = sum(nonzero_ratings) / len(nonzero_ratings)
      # Subtract mean for nonzero entries, keep 0 as 0.0
      normalized_row = [
          float(r - mean_rating) if r != 0 else 0.0 for r in row
      ]
      normalized_matrix.append(normalized_row)

  return normalized_matrix
