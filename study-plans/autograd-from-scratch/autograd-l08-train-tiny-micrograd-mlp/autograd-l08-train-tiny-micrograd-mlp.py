import torch

def train_tiny_micrograd_mlp(
    inputs: torch.Tensor, 
    targets: torch.Tensor, 
    weights: list[torch.Tensor], 
    biases: list[torch.Tensor], 
    learning_rate: float, 
    steps: int
) -> tuple:
    """
    Returns final predictions, final loss, trained weights, trained biases, and pre-update loss history.
    """
    # Clone weights and biases to keep the originals unchanged
    W = [w.clone() for w in weights]
    B_list = [b.clone() for b in biases]
    loss_history = []
    
    B_size = inputs.shape[0]
    
    for step in range(steps):
        # Forward pass: track activations for the backward pass
        H_layers = [inputs]
        for w, b in zip(W, B_list):
            H_layers.append(torch.tanh(H_layers[-1] @ w.t() + b))
            
        p = H_layers[-1].squeeze(-1)
        loss = torch.sum((p - targets) ** 2) if B_size > 0 else torch.tensor(0.0, dtype=inputs.dtype, device=inputs.device)
        loss_history.append(loss)
        
        if B_size == 0:
            continue
            
        # Backward pass: compute gradients manually
        # dL/dp = 2 * (p - y)
        dH = (2 * (p - targets)).unsqueeze(-1)
        
        for l in reversed(range(len(W))):
            H_prev = H_layers[l]
            H_curr = H_layers[l+1]
            w = W[l]
            
            # Local gradient through tanh activation: dL/dZ = dL/dH * (1 - tanh^2(Z))
            dZ = dH * (1.0 - H_curr ** 2)
            
            # Gradients with respect to weights and biases
            dW = dZ.t() @ H_prev
            dB = dZ.sum(dim=0)
            
            # Backpropagate to the previous layer's hidden state
            dH = dZ @ w
            
            # Gradient descent update step
            W[l] = w - learning_rate * dW
            B_list[l] = B_list[l] - learning_rate * dB
            
    # Final forward pass to compute final outputs and loss
    H = inputs
    for w, b in zip(W, B_list):
        H = torch.tanh(H @ w.t() + b)
    p = H.squeeze(-1)
    loss = torch.sum((p - targets) ** 2) if B_size > 0 else torch.tensor(0.0, dtype=inputs.dtype, device=inputs.device)
    
    return p, loss, W, B_list, loss_history
