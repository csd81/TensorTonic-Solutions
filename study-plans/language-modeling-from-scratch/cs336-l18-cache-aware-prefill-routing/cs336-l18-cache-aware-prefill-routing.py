def cache_aware_prefill_routing(
    request_token_counts, cached_prefix_lengths, worker_ids, worker_capacities
):
  """
    Returns: dictionary containing assignments and worker loads
    """
  num_workers = len(worker_ids)
  assigned_tokens = [0] * num_workers
  worker_uncached_loads = [0] * num_workers
  assignments = []

  for req_idx, req_len in enumerate(request_token_counts):
    prefixes = cached_prefix_lengths[req_idx]
    best_worker_idx = -1
    best_key = None

    for w_idx in range(num_workers):
      projected_assigned = assigned_tokens[w_idx] + req_len

      # Feasibility depends on full request tokens, independent of cache hits
      if projected_assigned <= worker_capacities[w_idx]:
        prefix_len = prefixes[w_idx]
        uncached_tokens = req_len - prefix_len
        projected_uncached_load = worker_uncached_loads[w_idx] + uncached_tokens
        remaining_capacity = worker_capacities[w_idx] - projected_assigned

        # Tuple ordered by priority (smallest wins)
        key = (
            -prefix_len,
            projected_uncached_load,
            -remaining_capacity,
            worker_ids[w_idx],
        )

        if best_key is None or key < best_key:
          best_key = key
          best_worker_idx = w_idx

    if best_worker_idx == -1:
      assignments.append(-1)
    else:
      assignments.append(worker_ids[best_worker_idx])
      assigned_tokens[best_worker_idx] += req_len
      worker_uncached_loads[best_worker_idx] += (
          req_len - prefixes[best_worker_idx]
      )

  return {
      "assignments": assignments,
      "worker_uncached_loads": worker_uncached_loads,
  }
