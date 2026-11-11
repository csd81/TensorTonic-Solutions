import numpy as np

def forward_pass(x: list, weights: list, biases: list) -> dict:
    """
    Returns all activations and pre-activations rounded to 4 decimal places.
    """
    # Convert inputs to float64 arrays
    a = np.array(x, dtype=np.float64)
    
    activations = [a.tolist()]
    pre_activations = []
    
    num_layers = len(weights)
    
    for l in range(num_layers):
        W = np.array(weights[l], dtype=np.float64)
        b = np.array(biases[l], dtype=np.float64)
        
        # z = W @ a + b
        z = W @ a + b
        
        # Store pre-activation rounded to 4 decimal places
        pre_activations.append(np.round(z, 4).tolist())
        
        # Apply ReLU for hidden layers only (all layers except the last one)
        if l < num_layers - 1:
            a = np.maximum(0, z)
        else:
            a = z  # Final layer uses pre-activation directly
            
        # Store activation rounded to 4 decimal places
        activations.append(np.round(a, 4).tolist())
        
    return {
        "activations": activations,
        "pre_activations": pre_activations
    }
