import torch

def gradient_accumulation_step(
    param: torch.Tensor, 
    microbatch_inputs: list[torch.Tensor],
    microbatch_targets: list[torch.Tensor], 
    lr: float,
) -> dict:
    """
    Performs one SGD update using gradients accumulated across microbatches of a linear model.
    """
    # 1. Calculate the total number of examples N across all microbatches
    total_n = sum(X_m.size(0) for X_m in microbatch_inputs)
    
    # 2. Clone the parameter to avoid mutating the original tensor or its gradient
    w = param.detach().clone().requires_grad_(True)
    
    # 3. Perform a backward pass per microbatch to accumulate full-batch gradients
    for X_m, y_m in zip(microbatch_inputs, microbatch_targets):
        # Compute the predicted targets: shape (N_m,)
        predictions = X_m @ w
        
        # Microbatch contribution to the full-batch mean squared error
        loss_m = torch.sum((predictions - y_m) ** 2) / total_n
        
        # Accumulate the gradients into w.grad
        loss_m.backward()
    
    # 4. Extract the accumulated full gradient
    full_grad = w.grad.detach()
    
    # 5. Perform a single manual SGD step
    new_param = w.detach() - lr * full_grad
    
    return {
        "new_param": new_param,
        "full_grad": full_grad
    }
