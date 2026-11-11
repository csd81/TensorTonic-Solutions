import torch
import torch.nn as nn

def manual_train_step(model: nn.Module, X: torch.Tensor, y: torch.Tensor, criterion: nn.Module, lr: float) -> float:
    """
    Returns the pre-update batch loss as a Python float.
    """
    # 1. Forward pass and loss calculation
    predictions = model(X)
    loss = criterion(predictions, y)
    
    # 2. Backward pass to compute gradients
    loss.backward()
    
    # 3. Manual parameter update and gradient clearing
    with torch.no_grad():
        for param in model.parameters():
            if param.requires_grad and param.grad is not None:
                param -= lr * param.grad
                param.grad.zero_()
                
    # 4. Return pre-update loss as a standard Python float
    return loss.item()
