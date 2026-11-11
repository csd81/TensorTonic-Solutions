def trace_reachable_graph(nodes: list, output_id: str) -> tuple:
    """
    Returns a tuple of reachable ID and parent-to-child edge lists.
    """
    nodes_by_id = {node["id"]: node for node in nodes}

    # Backward traversal from output_id to collect all reachable node IDs
    reachable = set()
    stack = [output_id]
    while stack:
        current_id = stack.pop()
        if current_id not in reachable:
            reachable.add(current_id)
            for parent_id in nodes_by_id[current_id]["parents"]:
                if parent_id not in reachable:
                    stack.append(parent_id)

    # Filter reachable IDs and build edges in original node-list and parent-list order
    ordered_ids = []
    edges = []
    for node in nodes:
        node_id = node["id"]
        if node_id in reachable:
            ordered_ids.append(node_id)
            for parent_id in node["parents"]:
                edges.append([parent_id, node_id])

    return (ordered_ids, edges)
