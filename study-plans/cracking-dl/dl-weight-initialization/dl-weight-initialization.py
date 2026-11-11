import math

def weight_init_params(layer_dims: list, method: str) -> list:
    """
    Returns initialization metadata for each weight matrix.
    """
    results = []
    
    # Process every consecutive pair of layers
    for i in range(len(layer_dims) - 1):
        fan_in = layer_dims[i]
        fan_out = layer_dims[i+1]
        
        # Calculate scale depending on the selected initialization formula
        if method == "xavier_normal":
            scale = math.sqrt(2.0 / (fan_in + fan_out))
        elif method == "xavier_uniform":
            scale = math.sqrt(6.0 / (fan_in + fan_out))
        elif method == "kaiming_normal":
            scale = math.sqrt(2.0 / fan_in)
        elif method == "kaiming_uniform":
            scale = math.sqrt(6.0 / fan_in)
        elif method == "random_normal":
            scale = 1.0
        else:
            scale = 1.0
            
        results.append({
            "fan_in": fan_in,
            "fan_out": fan_out,
            "scale": round(scale, 4),
            "shape": [fan_out, fan_in]
        })
        
    return results
