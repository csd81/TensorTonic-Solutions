import torch

def gradient_accumulation(w_init: torch.Tensor, micro_batches: list, lr: float, accum_steps: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (final_weights, last_mean_gradient) as float32 tensors.
    """
    # Clone the initial weights and enable gradients
    w = w_init.clone().detach().float().requires_grad_(True)
    
    last_mean_grad = None
    num_batches = len(micro_batches)
    
    # Process micro-batches in groups of size accum_steps
    for i in range(0, num_batches, accum_steps):
        group = micro_batches[i : i + accum_steps]
        
        total_loss = 0.0
        for x, t in group:
            # Predict using dot product: w^T * x
            pred = torch.dot(w, x.float())
            # Squared-error loss
            loss = (pred - t) ** 2
            total_loss += loss
            
        # Compute the mean loss for the accumulation group
        mean_loss = total_loss / accum_steps
        
        # Reset gradients before the backward pass for this group
        if w.grad is not None:
            w.grad.zero_()
            
        # Backward pass computes the mean gradient
        mean_loss.backward()
        
        # Track the mean gradient used in this update
        last_mean_grad = w.grad.clone().detach()
        
        # Apply the gradient descent step
        with torch.no_grad():
            w -= lr * w.grad
            
    return w.detach(), last_mean_grad
