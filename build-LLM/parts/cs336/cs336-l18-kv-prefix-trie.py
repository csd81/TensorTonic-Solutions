def kv_prefix_trie(cached_sequences: list, requests: list) -> dict:
    """
    Returns a dict with matches: records of matched_length, cache_node, and uncached_suffix.
    """
    # Map node_id -> {token: child_node_id}
    children = {0: {}}
    next_node_id = 1

    # Insert all cached sequences into the trie in input order
    for seq in cached_sequences:
        curr_node = 0
        for token in seq:
            node_edges = children[curr_node]
            if token not in node_edges:
                node_edges[token] = next_node_id
                children[next_node_id] = {}
                next_node_id += 1
            curr_node = node_edges[token]

    # Query each request against the constructed trie
    matches = []
    for req in requests:
        curr_node = 0
        matched_length = 0
        for token in req:
            node_edges = children[curr_node]
            if token in node_edges:
                curr_node = node_edges[token]
                matched_length += 1
            else:
                break

        matches.append(
            {
                "matched_length": matched_length,
                "cache_node": curr_node,
                "uncached_suffix": list(req[matched_length:]),
            }
        )

    return {"matches": matches}
