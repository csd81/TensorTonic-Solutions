def autograd(operations: list, input_values: list) -> dict:
    """
    Returns the final scalar and gradients of the initial inputs.
    """
    # Initialize the list of all node values with the given input values
    nodes = list(input_values)
    num_inputs = len(input_values)
    
    # Forward Pass: Compute each operation and append the resulting scalar value
    for op in operations:
        op_type = op[0]
        if op_type == "add":
            val = nodes[op[1]] + nodes[op[2]]
        elif op_type == "mul":
            val = nodes[op[1]] * nodes[op[2]]
        elif op_type == "neg":
            val = -nodes[op[1]]
        nodes.append(val)
        
    # Total number of nodes in the graph
    total_nodes = len(nodes)
    
    # Initialize gradients for all nodes to 0.0
    gradients = [0.0] * total_nodes
    # Seed the final output node's gradient with 1.0
    gradients[-1] = 1.0
    
    # Backward Pass: Traverse operations in reverse order
    # Each operation 'i' corresponds to the node at index (num_inputs + i)
    for i in reversed(range(len(operations))):
        op = operations[i]
        node_idx = num_inputs + i
        grad_out = gradients[node_idx]
        
        op_type = op[0]
        if op_type == "add":
            idx1, idx2 = op[1], op[2]
            gradients[idx1] += grad_out * 1.0
            gradients[idx2] += grad_out * 1.0
        elif op_type == "mul":
            idx1, idx2 = op[1], op[2]
            gradients[idx1] += grad_out * nodes[idx2]
            gradients[idx2] += grad_out * nodes[idx1]
        elif op_type == "neg":
            idx1 = op[1]
            gradients[idx1] += grad_out * -1.0
            
    # Round final results to 4 decimal places
    final_output = round(nodes[-1], 4)
    input_gradients = [round(g, 4) for g in gradients[:num_inputs]]
    
    return {
        "gradients": input_gradients,
        "output": final_output
    }
