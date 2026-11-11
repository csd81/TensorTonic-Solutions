def k_means_assignment(points: list, centroids: list) -> list:
    """
    Returns the nearest-centroid index for every point.
    """
    assignments = []
    for p in points:
        best_idx = 0
        best_dist = float("inf")
        for j, c in enumerate(centroids):
            dist = sum((pd - cd) ** 2 for pd, cd in zip(p, c))
            if dist < best_dist:
                best_dist = dist
                best_idx = j
        assignments.append(best_idx)
    return assignments
