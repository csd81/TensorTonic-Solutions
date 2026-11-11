import heapq


def topological_sort(nodes: list, output_id: str) -> list:
    """
    Returns a list of reachable node IDs in deterministic topological order.
    """
    # 1. Index nodes and record original positions
    node_map = {}
    position = {}
    for idx, node in enumerate(nodes):
        nid = node["id"]
        node_map[nid] = node["parents"]
        position[nid] = idx

    # 2. Backward traversal from output_id to find all reachable ancestors
    reachable = set()
    stack = [output_id]
    while stack:
        curr = stack.pop()
        if curr not in reachable:
            reachable.add(curr)
            for parent in node_map[curr]:
                if parent not in reachable:
                    stack.append(parent)

    # 3. Build forward adjacency list and in-degrees for reachable subgraph
    children = {nid: [] for nid in reachable}
    in_degree = {nid: 0 for nid in reachable}

    for nid in reachable:
        for parent in node_map[nid]:
            children[parent].append(nid)
            in_degree[nid] += 1

    # 4. Kahn's algorithm with min-heap keyed by original input position
    ready_heap = [
        (position[nid], nid) for nid in reachable if in_degree[nid] == 0
    ]
    heapq.heapify(ready_heap)

    order = []
    while ready_heap:
        _, curr = heapq.heappop(ready_heap)
        order.append(curr)

        for child in children[curr]:
            in_degree[child] -= 1
            if in_degree[child] == 0:
                heapq.heappush(ready_heap, (position[child], child))

    return order
